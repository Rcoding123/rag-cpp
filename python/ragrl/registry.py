"""Run registry: build-avx512/rl/runs/<name>-<utcstamp>-<sha7>/ (root: $RAGRL_ROOT).

Each run directory holds:
    config.json, config.sha256    the resolved config (write-once)
    provenance.json               git sha, dirty-diff / status sha1, ragrl code sha256, venv freeze sha256,
                                  module sha256, versions, argv, cpu (checked again at seal / confirm)
    baselines/                    copies of the baseline of_train nets (sha256 in baselines.json)
    bin/                          the _ragof module (and of_train.exe) the run imports / runs
    metrics.jsonl                 typed records (train, val, select, timing, ...)
    seeds/seedK/                  policy.json + artifacts, val.json, worker.log
    select.json                   validation-only selection (val_select), the declared headline
    live.jsonl                    the live-viewer stream (tools/of_train_live.py)
    sealed.json, sealed_episodes.npz   write-once scoring
    redflags.json, report.md, STATUS
Global: runs/index.jsonl; the usage ledger is build-avx512/rl/ledger_usage.jsonl (ledger.py).
"""
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import threading
import time
from importlib import metadata
from pathlib import Path

from . import native

REPO = native.REPO


def rl_root() -> Path:
    return Path(os.environ.get("RAGRL_ROOT", REPO / "build-avx512" / "rl"))


def runs_root() -> Path:
    return rl_root() / "runs"


def _git(*args) -> str:
    try:
        return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True, timeout=60).stdout
    except (OSError, subprocess.SubprocessError):
        return ""


def freeze_sha256() -> tuple[str, list[str]]:
    """sha256 of the venv's installed distributions (name==version, sorted): the 'pip freeze' hash."""
    pins = sorted({f"{(d.metadata['Name'] or '').lower()}=={d.version}" for d in metadata.distributions()})
    return hashlib.sha256("\n".join(pins).encode()).hexdigest(), pins


PKG_DIR = Path(__file__).resolve().parent


def code_sha256(pkg: Path = PKG_DIR) -> str:
    """sha256 over the ragrl package's own *.py files (sorted relative paths + contents): the
    pipeline code that trains, evaluates and scores. Checked again at seal / confirm."""
    h = hashlib.sha256()
    for f in sorted(pkg.rglob("*.py"), key=lambda p: p.relative_to(pkg).as_posix()):
        h.update(f.relative_to(pkg).as_posix().encode() + b"\0")
        h.update(f.read_bytes() + b"\0")
    return h.hexdigest()


def provenance(bin_dir: Path, source_dir: Path | None = None) -> dict:
    sha = _git("rev-parse", "HEAD").strip()
    diff = _git("diff", "HEAD")
    status = _git("status", "--porcelain", "--untracked-files=all")
    fz, pins = freeze_sha256()
    mod = native.find_module(bin_dir)
    versions = {}
    for name in ["torch", "numpy", "gymnasium", "stable-baselines3", "sb3-contrib", "lightgbm", "optuna"]:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return {
        "git_sha": sha,
        "dirty": bool(status.strip()),
        "dirty_diff_sha1": hashlib.sha1(diff.encode()).hexdigest() if diff else "",
        "status_sha1": hashlib.sha1(status.encode()).hexdigest() if status else "",  # includes untracked files
        "ragrl_sha256": code_sha256(),
        "ragrl_dir": str(PKG_DIR),
        "freeze_sha256": fz,
        "module": mod.name,
        "module_sha256": native.sha256_file(mod),
        "module_build_info": dict(native.load().BUILD_INFO) if hasattr(native.load(), "BUILD_INFO") else {},
        "source_module_dir": str(source_dir if source_dir is not None else _module_source(bin_dir)),
        "versions": versions,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "cpu": platform.processor(),
        "cpu_count": os.cpu_count(),
        "argv": sys.argv,
    }


def _module_source(bin_dir: Path) -> str:
    f = Path(bin_dir).parent / "module_source.txt"
    return f.read_text(encoding="utf-8").strip() if f.exists() else str(native.ragof_dir())


def new_run(cfg: dict, cfg_sha: str, root: Path | None = None) -> Path:
    """Creates the run directory, copies the module (and of_train.exe if present) into bin/;
    module_source.txt records where they came from (the snapshot)."""
    root = root or runs_root()
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    sha7 = (_git("rev-parse", "--short=7", "HEAD").strip() or "nogit")
    run = root / f"{cfg['name']}-{stamp}-{sha7}"
    k = 1
    while run.exists():
        k += 1
        run = root / f"{cfg['name']}-{stamp}-{sha7}-{k}"
    (run / "bin").mkdir(parents=True)
    (run / "seeds").mkdir()
    src = native.ragof_dir()
    shutil.copy2(native.find_module(src), run / "bin")
    for exe in ["of_train.exe", "of_train"]:
        if (src / exe).exists():
            shutil.copy2(src / exe, run / "bin")
    (run / "module_source.txt").write_text(str(src.resolve()) + "\n", encoding="utf-8")
    return run


def write_json(path: Path, obj, once: bool = False):
    path = Path(path)
    if once and path.exists():
        raise FileExistsError(f"{path} exists (write-once)")
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, sort_keys=True, allow_nan=False, default=_default) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def read_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _default(x):
    import numpy as np
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, Path):
        return str(x)
    raise TypeError(f"not JSON: {type(x)}")


def finite(x):
    """NaN/inf -> None, recursively (JSON has no NaN)."""
    import math
    if isinstance(x, float):
        return x if math.isfinite(x) else None
    if isinstance(x, dict):
        return {k: finite(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [finite(v) for v in x]
    return x


class Metrics:
    """Appends typed JSON records to metrics.jsonl (thread-safe, flushed)."""

    def __init__(self, path: Path, **tags):
        self.path = Path(path)
        self.tags = tags
        self._lock = threading.Lock()

    def write(self, rtype: str, **fields):
        rec = {"type": rtype, **self.tags, **fields}
        line = json.dumps(finite(rec), sort_keys=True, default=_default, allow_nan=False)
        with self._lock, open(self.path, "a", encoding="utf-8") as f:
            f.write(line + "\n")


def read_metrics(run: Path, rtype: str | None = None) -> list[dict]:
    out = []
    files = [run / "metrics.jsonl"] + sorted((run / "seeds").glob("seed*/metrics.jsonl"))
    for f in files:
        if f.exists():
            for line in f.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    r = json.loads(line)
                    if rtype is None or r.get("type") == rtype:
                        out.append(r)
    return out


def append_index(run: Path, entry: dict):
    idx = run.parent / "index.jsonl"
    with open(idx, "a", encoding="utf-8") as f:
        f.write(json.dumps({"run": run.name, **entry}, sort_keys=True, default=_default) + "\n")
