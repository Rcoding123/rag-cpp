#!/usr/bin/env python3
"""Phase 5 power analysis: minimum detectable NET edge (ticks/trade) for a day-block-tested short-horizon signal.

Uses NO signal and no features, only the mid-price path on the RTH grid, so it cannot leak into later tests.
Method (Monte Carlo): for horizon h and k trades/day (non-overlapping slots of width h, random slot, RANDOM SIDE so the true
edge is exactly 0), draw N_test days with replacement from the pool, average signed mid-change per trade, repeat B times.
The SD of that mean is the standard error SE(h, k, N). A one-sided test "CI lower bound > 0" at level alpha with power
1-beta needs a true net edge of  MDE = (z_{1-alpha} + z_{1-beta}) * SE.  GROSS edge needed = MDE + round-trip cost.
Real signal trades are conditioned on extreme features (more volatile), so a x1.3 SD inflation is shown as sensitivity.

usage: power_analysis.py [out_json]
"""
import json
import sys

import numpy as np
from statistics import NormalDist
norm = NormalDist()

import rth_grid

HORIZONS_S = (1, 5, 30, 60)
KS = (10, 50, 200)
ALPHA = 0.025            # one-sided
POWER = 0.80
BONF_M = 8               # planned number of pre-registered primary tests (Bonferroni sensitivity)
SPLITS = {"walk_forward_test_days": 42, "holdout_days": 15}
COSTS = {"market_round_trip": 1.28, "limit_round_trip_commission_only": 0.24}  # OF22/OF23 ticks, see PLAN
B = 1500
rng = np.random.default_rng(20261118)


def day_slots(df, hs):
    """Mid change over h seconds from slot starts spaced h seconds apart; NaN if either end invalid / instrument changed."""
    n = len(df)
    steps = int(hs * 10)
    mid, ok, inst = df.mid.values, df.valid.values, df.instrument_id.values
    starts = np.arange(0, n - steps, steps)
    e = starts + steps
    good = ok[starts] & ok[e] & (inst[starts] == inst[e])
    r = np.where(good, mid[e] - mid[starts], np.nan)
    return r[good]


def main():
    days = rth_grid.trade_days()
    pool = {h: [] for h in HORIZONS_S}
    ndays_valid = 0
    for d in days:
        df = rth_grid.load_day(d)
        if df.valid.mean() < 0.5:
            continue
        ndays_valid += 1
        for h in HORIZONS_S:
            pool[h].append(day_slots(df, h))
    res = {"days_in_pool": ndays_valid, "horizons": {}, "costs": COSTS, "alpha": ALPHA, "power": POWER, "bonf_m": BONF_M}
    z = norm.inv_cdf(1 - ALPHA) + norm.inv_cdf(POWER)
    zb = norm.inv_cdf(1 - ALPHA / BONF_M) + norm.inv_cdf(POWER)
    for h in HORIZONS_S:
        slots = pool[h]
        allr = np.concatenate(slots)
        hres = {"sd_ticks": float(allr.std()), "mean_abs_ticks": float(np.abs(allr).mean()),
                "frac_zero": float((allr == 0).mean()), "slots_per_day_median": int(np.median([len(s) for s in slots])),
                "by_k": {}}
        for k in KS:
            kk = {}
            for tag, nd in SPLITS.items():
                means = np.empty(B)
                for b in range(B):
                    pick = rng.integers(0, len(slots), nd)
                    tot = 0.0
                    cnt = 0
                    for i in pick:
                        s = slots[i]
                        m = min(k, len(s))
                        x = s[rng.choice(len(s), m, replace=False)] * rng.choice((-1.0, 1.0), m)
                        tot += x.sum()
                        cnt += m
                    means[b] = tot / cnt
                se = float(means.std())
                kk[tag] = {"n_trades": int(min(k, np.median([len(s) for s in slots])) * nd), "se": se, "mde_net": z * se,
                           "mde_net_bonf": zb * se, "mde_net_sd_x1.3": z * se * 1.3}
            hres["by_k"][str(k)] = kk
        res["horizons"][str(h)] = hres
    out = sys.argv[1] if len(sys.argv) > 1 else "/srv/quant/results/phase5_power.json"
    json.dump(res, open(out, "w"), indent=1)
    # table
    print(f"pool days={ndays_valid}  z={z:.3f} (bonf m={BONF_M}: {zb:.3f})")
    for h in HORIZONS_S:
        hr = res["horizons"][str(h)]
        print(f"h={h}s  sd={hr['sd_ticks']:.3f}t  P(no move)={hr['frac_zero']:.2f}  slots/day={hr['slots_per_day_median']}")
        for k, kk in hr["by_k"].items():
            for tag, v in kk.items():
                print(f"   k={k:>3} {tag:<22} trades={v['n_trades']:>6} SE={v['se']:.4f} MDE_net={v['mde_net']:.4f} "
                      f"bonf={v['mde_net_bonf']:.4f} x1.3={v['mde_net_sd_x1.3']:.4f}")


if __name__ == "__main__":
    main()
