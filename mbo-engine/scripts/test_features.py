"""Hand-computed tests for scripts/features.py: kernel masking/NaN rules, windows, VWAP, densification, trading mask."""
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, __file__.rsplit("/", 1)[0])
import features as F

fails = 0
def check(name, got, exp, tol=1e-12):
    global fails
    ok = bool(np.all(np.isclose(np.asarray(got, float), np.asarray(exp, float), rtol=tol, atol=tol, equal_nan=True)))
    fails += not ok
    print(("ok   " if ok else "FAIL ") + name, "" if ok else f"got {got} expected {exp}")

n = 1400
z = np.zeros(n)
buy = z.copy(); sell = z.copy(); pv = z.copy()
buy[0] = 2; pv[0] = 101 * 2            # bucket 0: buy 2 @ 101
sell[10] = 1; pv[10] = 100 * 1         # bucket 10: sell 1 @ 100
bsz = np.tile([5, 0, 0, 0, 0], (n, 1)).astype(float); asz = np.tile([4, 0, 0, 0, 0], (n, 1)).astype(float)
bid = z + 100; ask = z + 101
raw, x, vm = F.kernel(buy, sell, pv, bid, ask, bsz, asz, z, z, np.ones(n, bool), np.ones(13), 0.0)
check("delta5s window holds buy+sell", raw[10, 0], 1)
check("delta5s at i=49 (window 0..49)", raw[49, 0], 1)
check("delta5s at i=50 drops bucket 0 (window 1..50)", raw[50, 0], -1)
check("delta5s at i=60 drops bucket 10", raw[60, 0], 0)
check("delta120s at i=1199 holds both", raw[1199, 2], 1)
check("delta120s at i=1200 drops bucket 0", raw[1200, 2], -1)
check("delta30s at i=299 / 300", (raw[299, 1], raw[300, 1]), (1, -1))
check("vwap distance bucket 0: mid 100.5 - vwap 101", raw[0, 11], -0.5)
check("vwap distance bucket 10: vwap (202+100)/3", raw[10, 11], 100.5 - 302 / 3)
check("imbalance L1 / L1-5", (raw[0, 6], raw[0, 7]), (1 / 9, 1 / 9))
check("returns zero on flat mid", raw[1300, 8:11], [0, 0, 0])
check("ofi zero without events", raw[:, 3].sum(), 0)
check("x5 is clipped raw5; x6 = 2*raw6", (x[0, 5], x[0, 6]), (min(4, raw[0, 5]), 2 / 9))
# scaled value after one update: rms = sqrt(1 + (r^2-1)*alpha)
r0 = raw[0, 0]; rm = np.sqrt(1 + (r0 * r0 - 1) * F.NORM_ALPHA)
check("scaling x = raw/rms with updated rms", x[0, 0], r0 / rm)

# NaN rules: invalid quote in buckets 100..109 and a move so returns are non-trivial
n = 1400
bid = np.full(n, 100.0); ask = np.full(n, 101.0); bid[500:] += 2; ask[500:] += 2        # mid +2 ticks from bucket 500
qok = np.ones(n, bool); qok[100:110] = False
rms = np.ones(13)
raw, x, _ = F.kernel(np.zeros(n), np.zeros(n), np.zeros(n), bid, ask, np.tile(bsz[:1], (n, 1)), np.tile(asz[:1], (n, 1)), np.zeros(n), np.zeros(n), qok, rms, 0.0)
check("invalid bucket: imbalance and vwap NaN", (np.isnan(raw[105, 6]), np.isnan(raw[105, 7]), np.isnan(raw[105, 11])), (True, True, True))
check("flow features still finite on invalid bucket", np.isfinite(raw[105, [0, 1, 2, 3, 4, 5, 12]]).all(), True)
check("5s return NaN at i where i-50 was invalid (i=150..159)", np.isnan(raw[150:160, 8]).all() and np.isfinite(raw[160, 8]), True)
check("5s return NaN while current invalid", np.isnan(raw[105, 8]), True)
check("return +2 ticks across the move (i=500..549 vs 5s ago)", raw[549, 8], 2)
check("x NaN where raw NaN", np.isnan(x[105, 6]) and np.isnan(x[150, 8]), True)
def expected_rms(vals):
    r = 1.0
    for v in vals:
        if not np.isnan(v):
            r = np.sqrt(r * r + (v * v - r * r) * F.NORM_ALPHA)
    return r
check("rms[8] equals a plain-python recompute that skips NaN raw values", rms[8], expected_rms(raw[:, 8]))
check("rms[11] (vwap distance, NaN on the invalid stretch) likewise", rms[11], expected_rms(raw[:, 11]))
check("rms[0] (flow feature, never NaN) likewise", rms[0], expected_rms(raw[:, 0]))
# EWMA state is not updated by NaN raw: compare with run where invalid rows are dropped from a copy -> same rms for feature 6 isn't comparable; instead check rms[6]
r2 = np.ones(13); F.kernel(np.zeros(n), np.zeros(n), np.zeros(n), bid, ask, np.tile(bsz[:1], (n, 1)), np.tile(asz[:1], (n, 1)), np.zeros(n), np.zeros(n), np.ones(n, bool), r2, 0.0)
check("rms for a feature with NaNs skips those updates (differs from the all-valid run)", bool(rms[11] != r2[11]) or bool(rms[6] != r2[6]), True)

# ---- densification -------------------------------------------------------------------------------------------
B = F.BUCKET_NS
def bar(bk, bidpx, askpx, flags=0, ofi=0.0, absorb=0.0, sz=3):
    d = dict(ts_end_ns=(bk + 1) * B, instrument_id=7, flags=flags, bid_px=bidpx, ask_px=askpx, last_trade_px=0.0, ofi=ofi, absorb=absorb, bk=bk)
    for i in range(1, 6):
        d[f"bid_sz{i}"] = sz; d[f"ask_sz{i}"] = sz + 1
    return d
rows = pd.DataFrame([bar(12, 100.0, 100.25, ofi=3.0, absorb=1.5), bar(15, 100.25, 100.5, flags=F.BAR_SNAPSHOT, ofi=-2.0), bar(18, 100.25, 100.5, flags=F.BAR_CROSSED)])
status = {7: (np.array([0, 16 * B], dtype=np.int64), np.array([True, False]))}   # trading until bucket 16 start, then not
prior = pd.Series(bar(5, 99.75, 100.0, absorb=9.0))
d = F.dense_unit(rows, np.array([], np.int64), np.array([]), np.array([]), np.array([], bool), 7, 10, 20, status, prior)
check("prior row carried into buckets before the first new row (10,11)", d["bid"][:2], [399, 399])
check("quote forward-filled after bucket 12", d["bid"][2:5], [400, 400, 400])
check("quote updates at 15", d["bid"][5], 401)
check("OFI only at event buckets (12:+3, 15:-2), not forward-filled", d["ofi"], [0, 0, 3, 0, 0, -2, 0, 0, 0, 0])
check("absorb forward-filled (state), prior's 9 then 1.5 from 12", d["absorb"][:5], [9, 9, 1.5, 1.5, 1.5])
check("snapshot bucket invalid", d["qok"][5], False)
check("crossed flag state forward-filled from 18", d["qok"][8:10], [False, False])
check("not trading from bucket 16 on (status)", d["trading"][6:], [False] * 4)
check("trading before", d["trading"][:6], [True] * 6)
check("valid mid-buckets: 10,11 (prior) ok, 12..14 ok", d["qok"][:5], [True] * 5)
# no prior and window starts before the first row => no quote until the first row
d2 = F.dense_unit(rows, np.array([], np.int64), np.array([]), np.array([]), np.array([], bool), 7, 10, 20, status, None)
check("without prior: buckets before the first row have no quote", d2["qok"][:2], [False, False])
# trades aggregation
ts = np.array([12 * B + 5, 12 * B + 9, 13 * B + 1], dtype=np.int64)
d3 = F.dense_unit(rows, ts, np.array([400.0, 401.0, 400.0]), np.array([2.0, 3.0, 1.0]), np.array([True, False, True]), 7, 10, 20, status, prior)
check("buy/sell/pv per bucket", (d3["buy"][2], d3["sell"][2], d3["pv"][2], d3["buy"][3], d3["pv"][3]), (2, 3, 400 * 2 + 401 * 3, 1, 400))
print("FAILED" if fails else "features: all tests passed")
sys.exit(1 if fails else 0)
