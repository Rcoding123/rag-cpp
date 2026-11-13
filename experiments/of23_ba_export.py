#!/usr/bin/env python3
"""OF23 -> Build Alpha: the test-month trade lists of OF23's agents in Build Alpha's external trade-list format.

  python experiments/of23_ba_export.py [--out build-avx512/of23/buildalpha]

Reads OF23's finished work files (build-avx512/of23: cache and worker outputs; run of23_mine.py --run first) and
writes one CSV per agent, outside the repository (the trades come from real data):
  yyyymmdd,hhmmss,yyyymmdd,hhmmss,result     entry and exit in CT wall-clock time, result in dollars after OF23's
                                             base costs (1 tick = $12.50)
Entry = the open of bar m + 1; exit = the start of the minute the exit happened in (Build Alpha's list has no
seconds within a minute that matter for its tests), or 15:00 for an end-of-day exit. Plan section 8 (Build Alpha
is descriptive only, after OF23's test months are scored and committed).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "experiments"))
import of23_mine as M  # noqa: E402

TICK_USD = 12.50


def hhmmss(minute: int) -> str:
    t = 8 * 60 + 30 + int(minute)
    return f"{t // 60:02d}{t % 60:02d}00"


def agent_trades(C, name):
    """(rows, exits) over the test months, as OF23's report builds them."""
    J = M.WORK / "jobs"
    rows, exits = [], []
    for f in range(len(M.TEST_MONTHS)):
        if name == "gbm_primary":
            r = json.loads((J / f"gbm_pooled_{f}_1.json").read_text(encoding="utf-8"))
            rr, ee = r["rows"], r["exits"]
        else:
            arm = name.split("_", 1)[1]
            off = 1 if arm == "free" else 0
            per = [json.loads((J / f"rl_{arm}_{f}_{s}.json").read_text(encoding="utf-8")) for s in M.RL["seeds"]]
            te = per[0]["te"]
            A = np.array([p["acts"] for p in per])
            nA = len(M.MENU_E) + off
            vote = np.array([np.bincount(A[:, k], minlength=nA).argmax() for k in range(A.shape[1])]) if len(te) else np.zeros(0, int)
            rr, ee = M.rl_walk(C, te, vote, off)
        rows += list(rr)
        exits += list(ee)
    return np.asarray(rows, np.int64), np.asarray(exits, np.int64)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(M.WORK / "buildalpha"))
    a = ap.parse_args(argv)
    out = Path(a.out)
    if REPO in out.resolve().parents and "build" not in out.resolve().relative_to(REPO).parts[0]:
        raise SystemExit("real-data trade lists never go into the repository")
    out.mkdir(parents=True, exist_ok=True)
    C, _ = M.load_cache()
    for name in ("rl_trade", "rl_free", "gbm_primary"):
        rows, exits = agent_trades(C, name)
        order = np.lexsort((C["m"][rows], C["day"][rows]))
        lines = []
        for i in order:
            r, e = rows[i], exits[i]
            day = str(C["days"][C["day"][r]]).replace("-", "")
            x = int(C["xmin"][r, e])
            usd = float(C["net"][r, e]) * TICK_USD
            lines.append(f"{day},{hhmmss(C['m'][r] + 1)},{day},{hhmmss(x)},{usd:.2f}")
        (out / f"{name}.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")
        net = np.array([float(C["net"][rows[i], exits[i]]) for i in order])
        print(f"{name}: {len(lines)} trades, net {net.mean():+.2f} ticks/trade (${net.sum() * TICK_USD:,.0f} in all) -> {out / (name + '.csv')}")
    return 0


if __name__ == "__main__":
    from ragrl.cli import hard_exit
    hard_exit(main())
