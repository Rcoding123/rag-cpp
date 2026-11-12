"""Loads the `_ragof` module (the C++ simulator binding) from $RAGOF_DIR.

A run copies the module into RUN/bin and its workers import it from there
(RAGOF_DIR=RUN/bin), so a rebuild of build-avx512/Release can never swap or
lock the module under a running job, and the run's provenance names the exact
binary (sha256)."""
import hashlib
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
_R = None


def ragof_dir() -> Path:
    return Path(os.environ.get("RAGOF_DIR", REPO / "build-avx512" / "Release"))


def load():
    """The `_ragof` module (imported once per process)."""
    global _R
    if _R is None:
        d = str(ragof_dir())
        if d not in sys.path:
            sys.path.insert(0, d)
        import _ragof
        _R = _ragof
    return _R


def module_file() -> Path:
    return Path(load().__file__)


def find_module(directory: Path) -> Path:
    found = sorted(Path(directory).glob("_ragof*.pyd")) + sorted(Path(directory).glob("_ragof*.so"))
    if not found:
        raise FileNotFoundError(f"no _ragof module in {directory} (build with -DRAG_PYTHON=ON, or set RAGOF_DIR)")
    return found[0]


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def block(name: str) -> dict:
    """The ledger's entry for a block (R.BLOCKS, the C++ table; never hard-coded here)."""
    for b in load().BLOCKS:
        if b["name"] == name:
            return dict(b)
    raise KeyError(name)
