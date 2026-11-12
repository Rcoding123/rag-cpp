"""The only module that opens sealed blocks (a pytest greps for open_sealed).

    as_trainer()   first line of every training, selection, HPO and live
                   process: forbid_sealed() makes opening a sealed block in
                   this process impossible from then on (C++ guard).
    as_scorer()    `ragrl seal` / `ragrl confirm` only, in a fresh process:
                   open_sealed() makes the process a scorer (it can no longer
                   build train/validation environments), and the opening is
                   appended to the usage ledger.

The usage ledger is ONE file per machine, build-avx512/rl/ledger_usage.jsonl
under the repository, whatever $RAGRL_ROOT is (a different run root must not
give a fresh audit log). $RAGRL_LEDGER_FILE redirects it for tests only: every
entry and sealed.json record the file used, and a sealed test opening logged
anywhere but the canonical file is a hard red flag (R10).

Recorded ES (OF16, ragrl/real.py): the C++ block "real" is one gate for every recorded tape;
open_real() splits it by the day manifest's roles and logs each opening as block real_dev,
real_test or real_confirm with the manifest sha256:
    dev       loads freely (logged for the audit trail, never refused here); the process
              becomes a C++ scorer, so a dev diagnostic runs in a process that never trains
    test      a fresh process only (C++ role "none", nothing real opened before), with the run
              name and its config sha256; refused if ANY earlier real_test opening has the same
              config sha256 (whatever the run directory's name or the manifest: a renamed or
              copied run, or an edited manifest, is still the same experiment)
    confirm   orchestrator only, a claim id never used before, days no earlier confirm used

Pre-viewed ranges: 900M+[0, 100000) of the test block was viewed by OF2-OF14
(of_train --test, of2_baselines, OF11/OF13/OF14 scorings, on every market).
`range_reuse` reports overlaps with it and with earlier openings in the ledger.
"""
import json
import os
import time
from pathlib import Path

from . import native

# Block-relative ranges viewed before the pipeline's ledger existed (NEXT_SESSION.md section 4).
PREVIEWED = {"test": [(0, 100_000, "OF2-OF14: of_train --test / of2_baselines / OF11-OF14 scorings (all markets)")]}


def canonical_usage_file() -> Path:
    return native.REPO / "build-avx512" / "rl" / "ledger_usage.jsonl"


def usage_file() -> Path:
    o = os.environ.get("RAGRL_LEDGER_FILE")
    return Path(o) if o else canonical_usage_file()


def is_canonical() -> bool:
    return usage_file().resolve() == canonical_usage_file().resolve()


def as_trainer():
    native.load().forbid_sealed()


def _now():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def log_usage(entry: dict):
    f = usage_file()
    f.parent.mkdir(parents=True, exist_ok=True)
    with open(f, "a", encoding="utf-8") as out:
        out.write(json.dumps({**entry, "ledger_file": str(f)}, sort_keys=True) + "\n")


def read_usage():
    """Complete lines only (a line still being appended has no trailing newline yet)."""
    f = usage_file()
    if not f.exists():
        return []
    text = f.read_text(encoding="utf-8")
    lines = text.split("\n")
    if not text.endswith("\n"):
        lines = lines[:-1]
    return [json.loads(l) for l in lines if l.strip()]


def _overlap(a0, n0, a1, n1) -> bool:
    return a0 < a1 + n1 and a1 < a0 + n0


def range_reuse(block: str, first: int, count: int, market: str, run: str, experiment: str, family: str, family_size: int) -> dict:
    """Earlier views of block[first, +count) on `market`.

    Openings by the same experiment (identical config sha256 and ragrl code: a regeneration
    plays the identical policies, so it is not a new look) don't count. Up to family_size - 1
    other experiments of the declared family may share the range (their multiplicity is
    declared: t_req uses M x family_size). Everything else is reuse: pre-viewed ranges,
    openings outside the family, and family members beyond family_size."""
    pre = [{"range": [a, n], "what": w} for a, n, w in PREVIEWED.get(block, []) if _overlap(first, count, a, n)]
    family_exps, outside = set(), []
    for u in read_usage():
        if u.get("dry_run") or u.get("block") != block or u.get("run") == run:
            continue
        if not _overlap(first, count, int(u.get("first", 0)), int(u.get("count", 0))):
            continue
        if u.get("market") not in (None, market):  # entries without a market count against every market
            continue
        if u.get("experiment") and u.get("experiment") == experiment:
            continue
        if family and u.get("family") == family and u.get("experiment"):
            family_exps.add(u["experiment"])
        else:
            outside.append({"run": u.get("run"), "first": u.get("first"), "count": u.get("count"), "family": u.get("family")})
    over = max(0, len(family_exps) - (family_size - 1))
    return {"previewed": pre, "outside_family": outside, "family_experiments": len(family_exps), "family_over": over,
            "reused": bool(pre or outside or over)}


def next_confirm_first() -> int:
    """The first 950M+ index no earlier confirmation used (confirmations never share episodes)."""
    ends = [int(u.get("first", 0)) + int(u.get("count", 0)) for u in read_usage() if u.get("block") == "confirm" and not u.get("dry_run")]
    return max(ends, default=0)


def as_scorer(block: str, reason: str, run: str, first: int, count: int, claim: str = "", **meta):
    """Opens a sealed block for scoring and logs it. confirm needs a claim id never used before and a fresh range."""
    R = native.load()
    if block == "confirm":
        if not claim:
            raise PermissionError("confirm needs an orchestrator claim id")
        used = read_usage()
        if any(u.get("block") == "confirm" and u.get("claim") == claim for u in used):
            raise PermissionError(f"claim id {claim!r} was already used (950M+ is once per claim)")
        if any(u.get("block") == "confirm" and not u.get("dry_run") and _overlap(first, count, int(u.get("first", 0)), int(u.get("count", 0)))
               for u in used):
            raise PermissionError(f"confirm [{first}, +{count}) overlaps an earlier confirmation (next free index {next_confirm_first()})")
    R.open_sealed(block, reason)
    log_usage({"time": _now(), "run": run, "block": block, "first": first, "count": count, "reason": reason, "pid": os.getpid(),
               "claim": claim, "dry_run": False, **meta})


def log_dry_run(run: str, first: int, count: int, **meta):
    """A smoke-test 'seal' on the viz block (look-only, never sealed): logged, nothing opened."""
    log_usage({"time": _now(), "run": run, "block": "viz", "first": first, "count": count, "reason": "dry run (smoke)",
               "pid": os.getpid(), "claim": "", "dry_run": True, **meta})


REAL_KINDS = ("dev", "seen", "test", "confirm")
_real_opened = []  # this process's open_real() kinds, in order
_real_tapes = {}   # resolved tape path -> sha256 opened in this process (real.py loads only these)


def real_tape_allowed(path) -> str | None:
    """The sha256 open_real() registered for this tape path in this process, or None."""
    return _real_tapes.get(str(Path(path).resolve()))


def real_test_openings_of(config_sha256: str) -> list:
    """Every earlier real_test opening of this experiment (same config sha256, any run name, any manifest)."""
    return [u for u in real_openings("test") if config_sha256 and u.get("config_sha256") == config_sha256]


def real_openings(kind: str, run: str | None = None) -> list:
    """Ledger entries of real_<kind> openings (optionally for one run)."""
    return [u for u in read_usage() if u.get("block") == f"real_{kind}" and not u.get("dry_run") and (run is None or u.get("run") == run)]


def used_real_confirm_days() -> set:
    return {d for u in real_openings("confirm") for d in u.get("days", [])}


def open_real(kind: str, reason: str, *, manifest_sha256: str, days, run: str = "", config_sha256: str = "", claim: str = "",
              orchestrator: bool = False, tapes: dict | None = None, paths: dict | None = None, **meta):
    """Opens the C++ "real" block for recorded-ES tapes of one manifest role and logs it (block real_<kind>).
    Everything is checked before the block is opened, so a refusal leaves the process and the ledger as they were.
    paths: {day: csv path}, registered (with tapes' sha256) as the only tapes real.py may load in this process."""
    if kind not in REAL_KINDS:
        raise ValueError(f"real tapes: kind must be one of {REAL_KINDS}, not {kind!r}")
    if not manifest_sha256:
        raise ValueError("open_real needs the manifest sha256")
    days = [str(d) for d in days]
    R = native.load()
    if kind in ("test", "confirm"):
        role = R.process_role()
        if R.sealed_forbidden() or role != "none" or _real_opened or R.is_open("real"):
            raise PermissionError(f"real {kind} tapes open only in a fresh scorer process (this one: role {role!r}, "
                                  f"forbid_sealed {R.sealed_forbidden()}, real tapes opened before {_real_opened or R.is_open('real')})")
    if kind == "test":
        if not run or not config_sha256:
            raise PermissionError("a real_test opening needs the run name and its config sha256")
        before = real_test_openings_of(config_sha256)
        if before:
            raise PermissionError(f"config {config_sha256[:12]} already opened the real test days ({before[0].get('time')}, run "
                                  f"{before[0].get('run')!r}, manifest {str(before[0].get('manifest_sha256'))[:12]}): once per experiment")
    if kind == "confirm":
        if not orchestrator:
            raise PermissionError("real confirm days are for the orchestrator only")
        if not claim:
            raise PermissionError("real confirm needs an orchestrator claim id")
        if any(u.get("claim") == claim for u in real_openings("confirm")):
            raise PermissionError(f"claim id {claim!r} was already used (real confirm days are once per claim)")
        reused = sorted(set(days) & used_real_confirm_days())
        if reused:
            raise PermissionError(f"real confirm days {reused} were used by an earlier confirmation")
    if not R.is_open("real"):
        R.open_sealed("real", reason)
    _real_opened.append(kind)
    for d, path in (paths or {}).items():
        _real_tapes[str(Path(path).resolve())] = (tapes or {}).get(d, "")
    log_usage({"time": _now(), "run": run, "block": f"real_{kind}", "days": days, "manifest_sha256": manifest_sha256,
               "config_sha256": config_sha256, "tapes": tapes or {}, "reason": reason, "pid": os.getpid(), "claim": claim,
               "dry_run": False, **meta})
