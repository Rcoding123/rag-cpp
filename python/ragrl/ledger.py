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
