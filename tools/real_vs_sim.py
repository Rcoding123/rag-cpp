#!/usr/bin/env python3
"""real_vs_sim.py -- OF12: the same order-flow statistics, from the same code, on real ES
(Sierra Chart data) and on the ES simulator (of_calibrate export ticks).

usage (Python 3.12, standard library only; -I keeps the run isolated):
  python -I tools/real_vs_sim.py --bars ES_continuous_1min.csv --scid-dir C:/SierraChart/Data
      --depth ESZ26_FUT_CME.2026-11-03.depth [more .depth files]
      --sim discovery=build-avx512/tune/of12/discovery_*.csv --sim default_warm=... --sim default=...
      [--jobs 16] [--out experiments/of12_real_vs_sim.txt]
The first --sim market is the one the verdict column judges. experiments/of12_regen.sh runs it all.

The real data is only read (opened read-only; a Sierra file still being written is read as a
snapshot of its current size). The output holds aggregate statistics only.

Three data shapes, each built from both sides and then measured by ONE function:
  1-min bars    real: the 1-min CSV (unadjusted prices; returns never cross a contract change);
                sim: its trades aggregated per minute of the session.
  1-s bundles   real: Sierra .scid records (one per second with trades: trade count, volume,
                bid/ask volume by aggressor, OHLC of trade prices), front month by RTH volume per
                day; sim: its trades aggregated per whole second, bid volume = sell-aggressor lots.
  book          real: the L2 .depth event stream replayed into a book and sampled every 100 ms
                (the state after every event up to that instant); sim: its 100 ms Q lines (the
                simulator's own step). Levels are price levels 1..5 ticks from the best (0 if empty).
Standard errors: delete-one-block jackknife; real blocks = trading days (book: 5-min blocks of
the one RTH window, 30-min blocks overnight), sim blocks = sessions.
Time: Sierra stores UTC; US Central is computed here (second Sunday of March 02:00 to the first
Sunday of November 02:00 local is CDT, UTC-5; otherwise CST, UTC-6). RTH = 08:30-15:00 CT.
"""
import argparse
import glob
import math
import multiprocessing as mp
import os
import struct
import sys
from collections import Counter
from datetime import datetime, timezone
from operator import mul

UTC = timezone.utc
SESSION = 23400                        # seconds of RTH, = the simulator's session_seconds
SC_EPOCH_US = 25569 * 86400 * 1000000  # Sierra time zero (1899-12-30) to the Unix epoch, microseconds
GRID_US = 100000                       # book sampling step (the simulator's dt)
NAN = float("nan")

# ---------------------------------------------------------------- time ---
_dst = {}


def _nth_sunday(y, m, n):
    first = (6 - datetime(y, m, 1).weekday()) % 7 + 1
    return first + 7 * (n - 1)


def ct_offset_h(unix_s):
    """US Central offset from UTC in hours at a Unix time (CDT -5, CST -6)."""
    y = datetime.fromtimestamp(unix_s, UTC).year
    if y not in _dst:
        _dst[y] = (datetime(y, 3, _nth_sunday(y, 3, 2), 8, tzinfo=UTC).timestamp(),   # 02:00 CST
                   datetime(y, 11, _nth_sunday(y, 11, 1), 7, tzinfo=UTC).timestamp())  # 02:00 CDT
    a, b = _dst[y]
    return -5 if a <= unix_s < b else -6


def ct_local(unix_s):
    return unix_s + 3600 * ct_offset_h(unix_s)


def rth_open_unix(ct_day):
    """Unix time of 08:30 CT on a CT calendar day (days since 1970-01-01)."""
    local = ct_day * 86400 + 30600
    return local - 3600 * ct_offset_h(local + 6 * 3600)


def day_str(ct_day):
    return datetime.fromtimestamp(ct_day * 86400, UTC).strftime("%Y-%m-%d")


# -------------------------------------------------------- accumulators ---
# An accumulator is a flat dict of sums (str keys; histograms use (name, value) keys), so
# blocks add and a leave-one-out total is the sum minus one block.
def addv(a, k, v):
    a[k] = a.get(k, 0.0) + v


def pair_lists(a, k, xs, ys):
    if not xs:
        return
    addv(a, k + ".n", len(xs))
    addv(a, k + ".sx", sum(xs))
    addv(a, k + ".sy", sum(ys))
    addv(a, k + ".sxx", sum(map(mul, xs, xs)))
    addv(a, k + ".syy", sum(map(mul, ys, ys)))
    addv(a, k + ".sxy", sum(map(mul, xs, ys)))


def mom_list(a, k, xs):
    if not xs:
        return
    sq = [x * x for x in xs]
    addv(a, k + ".n", len(xs))
    addv(a, k + ".s1", sum(xs))
    addv(a, k + ".s2", sum(sq))
    addv(a, k + ".s3", sum(map(mul, sq, xs)))
    addv(a, k + ".s4", sum(map(mul, sq, sq)))


def hist(a, name, values):
    for v, c in Counter(values).items():
        addv(a, (name, v), c)


def merge(accs):
    t = {}
    for a in accs:
        for k, v in a.items():
            t[k] = t.get(k, 0.0) + v
    return t


# ----------------------------------------------------------- statistics ---
def f_ratio(k1, k2, scale=1.0):
    return lambda a: a.get(k1, 0.0) / a[k2] * scale if a.get(k2, 0.0) > 0 else NAN


def f_corr(k, square=False):
    def f(a):
        n = a.get(k + ".n", 0.0)
        if n < 3:
            return NAN
        sx, sy = a[k + ".sx"], a[k + ".sy"]
        vx, vy = n * a[k + ".sxx"] - sx * sx, n * a[k + ".syy"] - sy * sy
        if vx <= 0 or vy <= 0:
            return NAN
        c = (n * a[k + ".sxy"] - sx * sy) / math.sqrt(vx * vy)
        return c * c if square else c
    return f


def f_sd(k):
    def f(a):
        n = a.get(k + ".n", 0.0)
        if n < 2:
            return NAN
        m = a[k + ".s1"] / n
        return math.sqrt(max(0.0, a[k + ".s2"] / n - m * m))
    return f


def f_kurt(k):
    def f(a):
        n = a.get(k + ".n", 0.0)
        if n < 4:
            return NAN
        m, e2, e3, e4 = a[k + ".s1"] / n, a[k + ".s2"] / n, a[k + ".s3"] / n, a[k + ".s4"] / n
        m2 = e2 - m * m
        m4 = e4 - 4 * m * e3 + 6 * m * m * e2 - 3 * m ** 4
        return m4 / (m2 * m2) - 3 if m2 > 0 else NAN
    return f


def f_quant(name, q):
    def f(a):
        items = sorted((k[1], c) for k, c in a.items() if type(k) is tuple and k[0] == name and c > 0.5)
        tot = sum(c for _, c in items)
        if tot <= 0:
            return NAN
        cum = 0.0
        for v, c in items:
            cum += c
            if cum >= q * tot - 1e-9:
                return float(v)
        return float(items[-1][0])
    return f


def jackknife(blocks, funcs):
    """[(estimate, SE)] per function; SE = delete-one-block jackknife (nan with < 2 blocks)."""
    tot = merge(blocks)
    est = [f(tot) for f in funcs]
    B = len(blocks)
    if B < 2:
        return [(e, NAN) for e in est]
    loo = []
    for b in blocks:
        d = dict(tot)
        for k, v in b.items():
            d[k] -= v
        loo.append([f(d) for f in funcs])
    out = []
    for j, e in enumerate(est):
        vals = [row[j] for row in loo if not math.isnan(row[j])]
        if len(vals) < 2 or math.isnan(e):
            out.append((e, NAN))
            continue
        m = sum(vals) / len(vals)
        out.append((e, math.sqrt((len(vals) - 1) / len(vals) * sum((v - m) ** 2 for v in vals))))
    return out


# ---------------------------------------------------- shape: 1-min bars ---
# bars: a list over consecutive minutes; each entry None (no bar) or
# (open, high, low, close [ticks], volume, trades, bid_volume, ask_volume, contract).
def bar_stats(bars, shape):
    a = {}
    pres = [b for b in bars if b]
    addv(a, "b.n", len(pres))
    addv(a, "b.trades", sum(b[5] for b in pres))
    addv(a, "b.vol", sum(b[4] for b in pres))
    n = len(bars)
    r = [None] * n
    for m in range(1, n):
        b0, b1 = bars[m - 1], bars[m]
        if b0 and b1 and b0[8] == b1[8]:
            r[m] = float(b1[3] - b0[3])
    rs = [x for x in r if x is not None]
    mom_list(a, "r", rs)
    a["_rmax"] = max((abs(x) for x in rs), default=0.0)  # per-block diagnostic only (sums are meaningless)

    def lagged(xs, ys, lag, key, fx=lambda v: v, fy=lambda v: v):
        px, py = [], []
        for i in range(n - lag):
            x, y = xs[i], ys[i + lag]
            if x is not None and y is not None:
                px.append(fx(x))
                py.append(fy(y))
        pair_lists(a, key, px, py)

    lagged(r, r, 1, "rac1")
    for lag in (1, 5, 30):
        lagged(r, r, lag, "aac%d" % lag, abs, abs)
    delta = [float(b[7] - b[6]) if b else None for b in bars]
    lagged(delta, r, 0, "imp")
    lagged(delta, r, 1, "dp1")
    f5 = [None] * n  # close(m + 5) - close(m), all six bars present, one contract
    for m in range(n - 5):
        seg = bars[m:m + 6]
        if all(seg) and all(b[8] == seg[0][8] for b in seg):
            f5[m] = float(seg[5][3] - seg[0][3])
    lagged(delta, f5, 0, "dp5")
    if shape:
        for m, b in enumerate(bars):
            if b:
                addv(a, "hh%02d" % (m // 30), b[4])
                addv(a, "hh.tot", b[4])
    return a


BAR_ROWS = [
    ("trades / min", f_ratio("b.trades", "b.n"), "%.0f", 1),
    ("volume / min (lots)", f_ratio("b.vol", "b.n"), "%.0f", 1),
    ("lots / trade", f_ratio("b.vol", "b.trades"), "%.2f", 0.1),
    ("1-min return sd (ticks)", f_sd("r"), "%.2f", 0.2),
    ("1-min excess kurtosis", f_kurt("r"), "%.2f", 1),
    ("ac1 of 1-min returns", f_corr("rac1"), "%+.3f", 0.05),
    ("ac1 of |1-min returns|", f_corr("aac1"), "%+.3f", 0.05),
    ("ac5 of |1-min returns|", f_corr("aac5"), "%+.3f", 0.05),
    ("ac30 of |1-min returns|", f_corr("aac30"), "%+.3f", 0.05),
    ("R^2(1-min change ~ 1-min delta)", f_corr("imp", True), "%.3f", 0.05),
    ("corr(1-min delta, next 1 min)", f_corr("dp1"), "%+.3f", 0.03),
    ("corr(1-min delta, next 5 min)", f_corr("dp5"), "%+.3f", 0.03),
]
SHAPE_ROWS = [("volume share %02d:%02d-%02d:%02d CT (%%)" % ((8 * 60 + 30 + 30 * k) // 60, (8 * 60 + 30 + 30 * k) % 60,
                                                          (8 * 60 + 60 + 30 * k) // 60, (8 * 60 + 60 + 30 * k) % 60),
               f_ratio("hh%02d" % k, "hh.tot", 100), "%.2f", 0.5) for k in range(13)]


# ---------------------------------------------------- shape: 1-s bundles ---
# Per second of the session: trades n, volume, bid volume (sell aggressor), ask volume, and
# the last trade price of the second (ticks; None without trades).
def bundle_stats(n, vol, bv, av, close):
    a = {}
    L = len(n)
    addv(a, "s.n", L)
    addv(a, "s.trades", sum(n))
    addv(a, "s.vol", sum(vol))
    addv(a, "s.zero", sum(1 for x in n if x == 0))
    hist(a, "tps", n)
    carry, last, chg = [None] * L, None, 0
    for i, c in enumerate(close):
        if c is not None:
            if last is not None and c != last:
                chg += 1
            last = c
        carry[i] = last
    addv(a, "s.pchg", chg)
    d = [av[i] - bv[i] for i in range(L)]
    for w, key in ((10, "imp10"), (30, "dp30")):
        W = L // w
        dw = [sum(d[w * k:w * k + w]) for k in range(W)]
        pe = [carry[w * k + w - 1] for k in range(W)]
        xs, ys = [], []
        for k in range(1, W):
            if key == "imp10" and pe[k] is not None and pe[k - 1] is not None:
                xs.append(float(dw[k]))
                ys.append(float(pe[k] - pe[k - 1]))
            if key == "dp30" and k + 1 < W and pe[k] is not None and pe[k + 1] is not None:
                xs.append(float(dw[k]))
                ys.append(float(pe[k + 1] - pe[k]))
        pair_lists(a, key, xs, ys)
    s = [float((x > 0) - (x < 0)) for x in d]
    for lag in (1, 10, 100):
        pair_lists(a, "sac%d" % lag, s[:-lag], s[lag:])
    return a


SEC_ROWS = [
    ("trades / s, mean", f_ratio("s.trades", "s.n"), "%.1f", 1),
    ("trades / s, median", f_quant("tps", 0.5), "%.0f", 1),
    ("trades / s, 90th pct", f_quant("tps", 0.9), "%.0f", 1),
    ("trades / s, 99th pct", f_quant("tps", 0.99), "%.0f", 1),
    ("lots / s, mean", f_ratio("s.vol", "s.n"), "%.1f", 1),
    ("share of seconds with no trade (%)", f_ratio("s.zero", "s.n", 100), "%.2f", 1),
    ("last-price changes / min (1-s closes)", f_ratio("s.pchg", "s.n", 60), "%.1f", 1),
    ("R^2(10 s price change ~ 10 s delta)", f_corr("imp10", True), "%.3f", 0.05),
    ("corr(30 s delta, next 30 s change)", f_corr("dp30"), "%+.3f", 0.03),
    ("1-s delta sign autocorr, lag 1 s", f_corr("sac1"), "%+.3f", 0.05),
    ("1-s delta sign autocorr, lag 10 s", f_corr("sac10"), "%+.3f", 0.05),
    ("1-s delta sign autocorr, lag 100 s", f_corr("sac100"), "%+.3f", 0.05),
]


# ----------------------------------------------------------- shape: book ---
# samples: book states every 100 ms, index i at second t0 + i / 10; each None (no valid book)
# or (bid, ask [ticks], bid sizes x5, ask sizes x5, bid orders x5 or None, ask orders x5 or None).
# keys[i]: (label, block) or None (excluded). bundles: {second: (trades, volume, bid vol, ask vol)}.
def book_stats(samples, keys, t0, bundles):
    accs = {}
    N = len(samples)
    y = [None] * N  # mid at the next mid change after i minus mid now (ticks), as of10_discovery's event corr
    nxt, lab = None, None
    for i in range(N - 1, -1, -1):
        s, k = samples[i], keys[i]
        if s is None or k is None:
            nxt, lab = None, None
            continue
        if k[0] != lab:
            nxt, lab = None, k[0]
        mid = s[0] + s[1]
        if nxt is not None:
            y[i] = (nxt - mid) / 2
        if i > 0:
            p = samples[i - 1]
            if p is not None and keys[i - 1] is not None and p[0] + p[1] != mid:
                nxt = mid
    prev = None
    for i in range(N):
        s, k = samples[i], keys[i]
        if s is None or k is None:
            prev = None
            continue
        a = accs.get(k)
        if a is None:
            a = accs[k] = {}
        bid, ask, b5, a5, ob5, oa5 = s
        addv(a, "k.n", 1)
        if ask - bid == 1:
            addv(a, "k.sp1", 1)
        addv(a, "k.l1b", b5[0])
        addv(a, "k.l1a", a5[0])
        addv(a, ("l1", b5[0]), 1)
        addv(a, ("l1", a5[0]), 1)
        for j in range(1, 5):
            addv(a, "k.l%d" % (j + 1), b5[j] + a5[j])
        if ob5 is not None:
            addv(a, "k.on", 1)
            addv(a, "k.o1lots", b5[0] + a5[0])
            for j in range(5):
                addv(a, "k.o%d" % (j + 1), ob5[j] + oa5[j])
        s1 = b5[0] + a5[0]
        i1 = (b5[0] - a5[0]) / s1 if s1 > 0 else 0.0
        sb, sa = sum(b5), sum(a5)
        i5 = (sb - sa) / (sb + sa) if sb + sa > 0 else 0.0
        yi = y[i]
        if yi is not None:
            for key, x in (("ev1", i1), ("ev5", i5)):
                addv(a, key + ".n", 1)
                addv(a, key + ".sx", x)
                addv(a, key + ".sy", yi)
                addv(a, key + ".sxx", x * x)
                addv(a, key + ".syy", yi * yi)
                addv(a, key + ".sxy", x * yi)
            if i1 != 0 and yi != 0:
                addv(a, "k.hitn", 1)
                if (i1 > 0) == (yi > 0):
                    addv(a, "k.hit", 1)
        if prev is not None and prev[1][0] == k[0]:
            p = prev[0]
            addv(a, "k.pairs", 1)
            if p[0] + p[1] != bid + ask:
                addv(a, "k.midchg", 1)
                if p[1] - p[0] == 1 and ask - bid == 1:
                    addv(a, "k.midchg1", 1)  # a whole-book tick, not a spread opening or closing
            dep = 0
            pb, pb5 = p[0], p[2]  # bid side: same price -> net decrease; lower -> the levels above are gone
            if bid == pb:
                dep += max(0, pb5[0] - b5[0])
            elif bid < pb:
                m = pb - bid
                dep += sum(pb5[:min(m, 5)])
                if m < 5:
                    dep += max(0, pb5[m] - b5[0])
            pa, pa5 = p[1], p[3]
            if ask == pa:
                dep += max(0, pa5[0] - a5[0])
            elif ask > pa:
                m = ask - pa
                dep += sum(pa5[:min(m, 5)])
                if m < 5:
                    dep += max(0, pa5[m] - a5[0])
            addv(a, "k.dep", dep)
        prev = (s, k)
    # 1-s bundles against the book: traded volume in the window, and markouts of the second's
    # delta (ask - bid volume) on the mid at the end of the second.
    def mid_at(e):
        if 0 <= e < N and samples[e] is not None and keys[e] is not None:
            return samples[e][0] + samples[e][1], keys[e]
        return None, None
    for X in range(t0, t0 + N // 10):
        e = 10 * (X + 1 - t0)
        m0, k = mid_at(e)
        if m0 is None:
            continue
        a = accs[k]
        bd = bundles.get(X)
        addv(a, "k.tsec", 1)
        if bd is None:
            continue
        addv(a, "k.tvol", bd[1])
        d = bd[3] - bd[2]
        if d == 0:
            continue
        sg = 1 if d > 0 else -1
        for h in (-1, 1, 5, 30):
            m1, k1 = mid_at(e + 10 * h)
            if m1 is None or k1[0] != k[0]:
                continue
            yy = (m1 - m0) / 2 if h > 0 else (m0 - m1) / 2
            tag = "c" if h < 0 else str(h)
            addv(a, "mk%s.sum" % tag, sg * yy)
            addv(a, "mk%s.cnt" % tag, 1)
            for key, xv, yv in (("mkc%s" % tag, float(d), yy),):
                addv(a, key + ".n", 1)
                addv(a, key + ".sx", xv)
                addv(a, key + ".sy", yv)
                addv(a, key + ".sxx", xv * xv)
                addv(a, key + ".syy", yv * yv)
                addv(a, key + ".sxy", xv * yv)
    return accs


def f_dep_per_side(a):
    return a["k.dep"] / (a["k.pairs"] * 0.1) / 2 if a.get("k.pairs", 0) > 0 else NAN


def f_traded_share(a):
    dep = a.get("k.dep", 0.0) / (a["k.pairs"] * 0.1) if a.get("k.pairs", 0) > 0 else 0.0
    tv = a.get("k.tvol", 0.0) / a["k.tsec"] if a.get("k.tsec", 0) > 0 else NAN
    return tv / dep if dep > 0 else NAN


BOOK_ROWS = [
    ("spread = 1 tick (%)", f_ratio("k.sp1", "k.n", 100), "%.2f", 1),
    ("L1 size, bid mean (lots)", f_ratio("k.l1b", "k.n"), "%.1f", 1),
    ("L1 size, ask mean (lots)", f_ratio("k.l1a", "k.n"), "%.1f", 1),
    ("L1 size, 10th pct", f_quant("l1", 0.1), "%.0f", 1),
    ("L1 size, median", f_quant("l1", 0.5), "%.0f", 1),
    ("L1 size, 90th pct", f_quant("l1", 0.9), "%.0f", 1),
    ("L2 size, mean per side", f_ratio("k.l2", "k.n", 0.5), "%.1f", 1),
    ("L3 size, mean per side", f_ratio("k.l3", "k.n", 0.5), "%.1f", 1),
    ("L4 size, mean per side", f_ratio("k.l4", "k.n", 0.5), "%.1f", 1),
    ("L5 size, mean per side", f_ratio("k.l5", "k.n", 0.5), "%.1f", 1),
    ("mid changes / min (100 ms samples)", f_ratio("k.midchg", "k.pairs", 600), "%.1f", 1),
    ("  of which 1-tick spread before and after", f_ratio("k.midchg1", "k.pairs", 600), "%.1f", 1),
    ("event corr(I1, next mid change)", f_corr("ev1"), "%+.3f", 0.03),
    ("event corr(I5, next mid change)", f_corr("ev5"), "%+.3f", 0.03),
    ("next mid change goes the way I1 points (%)", f_ratio("k.hit", "k.hitn", 100), "%.1f", 1),
    ("L1 depletion, lots/s per side (net, 100 ms)", f_dep_per_side, "%.1f", 1),
    ("traded lots / s", f_ratio("k.tvol", "k.tsec"), "%.1f", 1),
    ("traded lots / L1 depletion", f_traded_share, "%.2f", 0.05),
    ("1-s delta: same-second mid move (signed, ticks)", f_ratio("mkc.sum", "mkc.cnt"), "%+.3f", 0.05),
    ("1-s delta markout +1 s (signed, ticks)", f_ratio("mk1.sum", "mk1.cnt"), "%+.3f", 0.05),
    ("1-s delta markout +5 s (signed, ticks)", f_ratio("mk5.sum", "mk5.cnt"), "%+.3f", 0.05),
    ("1-s delta markout +30 s (signed, ticks)", f_ratio("mk30.sum", "mk30.cnt"), "%+.3f", 0.05),
    ("corr(1-s delta, same-second mid move)", f_corr("mkcc"), "%+.3f", 0.03),
    ("corr(1-s delta, next 1 s mid move)", f_corr("mkc1"), "%+.3f", 0.03),
    ("corr(1-s delta, next 5 s mid move)", f_corr("mkc5"), "%+.3f", 0.03),
    ("corr(1-s delta, next 30 s mid move)", f_corr("mkc30"), "%+.3f", 0.03),
]
BOOK_REAL_ONLY = [
    ("orders at L1, per side", f_ratio("k.o1", "k.on", 0.5), "%.1f", 1),
    ("lots per order at L1", f_ratio("k.o1lots", "k.o1"), "%.2f", 0.1),
    ("orders at L2, per side", f_ratio("k.o2", "k.on", 0.5), "%.1f", 1),
    ("orders at L5, per side", f_ratio("k.o5", "k.on", 0.5), "%.1f", 1),
    ("share of samples with order counts (%)", f_ratio("k.on", "k.n", 100), "%.1f", 1),
]


# ------------------------------------------------------------- real I/O ---
def read_snapshot(path, header, rec):
    """The whole file as bytes (a snapshot: the size is fixed at open), whole records only."""
    with open(path, "rb") as f:
        size = os.fstat(f.fileno()).st_size
        data = f.read(size)
    hsize = struct.unpack_from("<I", data, 4)[0]
    if hsize != header:
        raise ValueError("%s: header size %d, expected %d" % (path, hsize, header))
    n = (len(data) - hsize) // rec
    return data, hsize, n


def first_at_or_after(data, hsize, rec, n, t):
    lo, hi = 0, n
    while lo < hi:
        m = (lo + hi) // 2
        if struct.unpack_from("<q", data, hsize + rec * m)[0] < t:
            lo = m + 1
        else:
            hi = m
    return lo


def scid_range(data, hsize, n, t0_us, t1_us):
    """scid records with Sierra time in [t0_us, t1_us)."""
    i = first_at_or_after(data, hsize, 40, n, t0_us)
    j = first_at_or_after(data, hsize, 40, n, t1_us)
    return struct.iter_unpack("<q4f4I", data[hsize + 40 * i:hsize + 40 * j])


def ticks(p100):
    return int(p100 + 0.5) // 25  # Sierra price x 100 -> 0.25-point ticks


def task_scid_days(args):
    """One .scid file: per RTH day, (volume, contract, 1-s bundle statistics)."""
    path, contract, days = args
    data, hsize, n = read_snapshot(path, 56, 40)
    out = {}
    for d in days:
        o = rth_open_unix(d)
        o_us = o * 1000000 + SC_EPOCH_US
        N = [0] * SESSION
        V, BV, AV = [0] * SESSION, [0] * SESSION, [0] * SESSION
        C = [None] * SESSION
        tot = 0
        for t, _o, _h, _l, c, nt, vol, bv, av in scid_range(data, hsize, n, o_us, o_us + SESSION * 1000000):
            s = (t - o_us) // 1000000
            N[s] += nt
            V[s] += vol
            BV[s] += bv
            AV[s] += av
            C[s] = ticks(c)
            tot += vol
        if tot > 0:
            out[d] = (tot, contract, bundle_stats(N, V, BV, AV, C))
    return out


def task_depth(args):
    """One .depth file: the book every 100 ms, the matching 1-s bundles from the .scid, statistics
    per (label, block): RTH 5-min blocks, ETH 30-min blocks, the 16:00-17:00 CT halt excluded."""
    path, scid_path, until_us = args
    data, hsize, n = read_snapshot(path, 64, 24)
    if until_us is not None:  # a fixed end for a file Sierra is still writing
        n = first_at_or_after(data, hsize, 24, n, until_us)
    bids, asks = {}, {}
    samples = []
    first_t = struct.unpack_from("<q", data, hsize)[0]
    g = (first_t // 1000000 + 1) * 1000000  # the first whole second after the first event
    t0 = (g - SC_EPOCH_US) // 1000000
    crossed = 0

    def snap():
        nonlocal crossed
        if not bids or not asks:
            return None
        bb, ba = max(bids), min(asks)
        if ba <= bb:
            crossed += 1
            return None
        bl = [bids.get(bb - j) for j in range(5)]
        al = [asks.get(ba + j) for j in range(5)]
        b5 = tuple(x[0] if x else 0 for x in bl)
        a5 = tuple(x[0] if x else 0 for x in al)
        ob5 = tuple(x[1] if x else 0 for x in bl)
        oa5 = tuple(x[1] if x else 0 for x in al)
        if ob5[0] == 0 and oa5[0] == 0:
            ob5 = oa5 = None
        return (bb, ba, b5, a5, ob5, oa5)

    for t, cmd, _fl, nord, price, qty, _r in struct.iter_unpack("<qBBHfII", data[hsize:hsize + 24 * n]):
        if t > g:
            s = snap()
            while g < t:
                samples.append(s)
                g += GRID_US
        if cmd == 1:
            bids.clear()
            asks.clear()
        elif cmd == 2 or cmd == 4:
            bids[ticks(price)] = (qty, nord)
        elif cmd == 3 or cmd == 5:
            asks[ticks(price)] = (qty, nord)
        elif cmd == 6:
            bids.pop(ticks(price), None)
        elif cmd == 7:
            asks.pop(ticks(price), None)
    keys = []
    for i in range(len(samples)):
        u = t0 + i // 10
        tod = ct_local(u) % 86400
        if 30600 <= tod < 54000:
            keys.append(("RTH", u // 300))
        elif 57600 <= tod < 61200:
            keys.append(None)
        else:
            keys.append(("ETH", u // 1800))
    sd, sh, sn = read_snapshot(scid_path, 56, 40)
    bundles = {}
    for t, _o, _h, _l, _c, nt, vol, bv, av in scid_range(sd, sh, sn, t0 * 1000000 + SC_EPOCH_US,
                                                          (t0 + len(samples) // 10 + 2) * 1000000 + SC_EPOCH_US):
        u = (t - SC_EPOCH_US) // 1000000
        b = bundles.get(u, (0, 0, 0, 0))
        bundles[u] = (b[0] + nt, b[1] + vol, b[2] + bv, b[3] + av)
    accs = book_stats(samples, keys, t0, bundles)
    cover = Counter(k[0] for s, k in zip(samples, keys) if s is not None and k is not None)
    rth_span = [t0 + i // 10 for i, k in enumerate(keys) if k is not None and k[0] == "RTH"]
    return {"accs": accs, "cover": dict(cover), "crossed": crossed, "events": n, "t0": t0, "t1": t0 + len(samples) // 10,
            "rth": (min(rth_span), max(rth_span) + 1) if rth_span else None, "path": os.path.basename(path)}


def depth_rth_seconds(path, until_us=None):
    """Seconds of RTH (from 08:30 CT) a .depth file covers: the simulator's matched book window."""
    data, hsize, n = read_snapshot(path, 64, 24)
    if until_us is not None:
        n = first_at_or_after(data, hsize, 24, n, until_us)
    if n == 0:
        return 0, None
    t_first = struct.unpack_from("<q", data, hsize)[0]
    t_last = struct.unpack_from("<q", data, hsize + 24 * (n - 1))[0]
    a, b = (t_first - SC_EPOCH_US) // 1000000, (t_last - SC_EPOCH_US) // 1000000
    day = ct_local(a) // 86400
    o = rth_open_unix(day)
    lo, hi = max(a, o), min(b, o + SESSION)
    return max(0, hi - lo), (lo, hi)


def read_bars(path):
    """RTH bars per CT day (full sessions only), ETH bars per trade date, and data checks."""
    rth, eth = {}, {}
    ct_mismatch = rows = 0
    with open(path, "r", newline="") as f:
        head = f.readline().strip().split(",")
        col = {c: i for i, c in enumerate(head)}
        for line in f:
            p = line.rstrip("\r\n").split(",")
            if len(p) < len(head):
                continue
            rows += 1
            u = int(datetime.strptime(p[col["time_utc"]], "%Y-%m-%d %H:%M:%S").replace(tzinfo=UTC).timestamp())
            loc = ct_local(u)
            if datetime.fromtimestamp(loc, UTC).strftime("%Y-%m-%d %H:%M:%S") != p[col["time_ct"]]:
                ct_mismatch += 1
            bar = (round(float(p[col["open"]]) * 4), round(float(p[col["high"]]) * 4), round(float(p[col["low"]]) * 4),
                   round(float(p[col["close"]]) * 4), int(p[col["volume"]]), int(p[col["trades"]]),
                   int(p[col["bid_volume"]]), int(p[col["ask_volume"]]), p[col["contract"]])
            day, tod = divmod(loc, 86400)
            if 30600 <= tod < 54000:
                rth.setdefault(day, {})[(tod - 30600) // 60] = bar
            else:
                eth.setdefault(p[col["trade_date"]], {})[u // 60] = (bar, day, tod)
    full = {d: [m[i] for i in range(390)] for d, m in rth.items() if len(m) == 390}
    eth_lists = {}
    for td, m in eth.items():
        ks = sorted(m)
        eth_lists[td] = [m[k][0] if k in m else None for k in range(ks[0], ks[-1] + 1)]
    return full, eth_lists, {"rows": rows, "ct_mismatch": ct_mismatch, "rth_days": len(rth), "full_days": len(full)}


# -------------------------------------------------------------- sim I/O ---
def task_sim(args):
    """One simulator session (of_calibrate export from the open): statistics of all three shapes."""
    market, path, window_s = args
    N, V, BV, AV = [0] * SESSION, [0] * SESSION, [0] * SESSION, [0] * SESSION
    O, H, Lo, C = [None] * SESSION, [None] * SESSION, [None] * SESSION, [None] * SESSION
    samples = [None] * (SESSION * 10 + 1)
    bundles = {}
    with open(path, "r") as f:
        for line in f:
            c0 = line[0]
            if c0 == "T":
                _, t, p, sz, side = line.split(",")
                ms = round(float(t) * 1000)
                s = (ms - 1) // 1000  # a step's trades happen in (t - 0.1, t]
                if s >= SESSION:
                    continue
                pt, q = round(float(p) * 4), int(sz)
                N[s] += 1
                V[s] += q
                if side[0] == "B":
                    AV[s] += q
                else:
                    BV[s] += q
                if O[s] is None:
                    O[s] = H[s] = Lo[s] = pt
                H[s] = max(H[s], pt)
                Lo[s] = min(Lo[s], pt)
                C[s] = pt
            elif c0 == "Q":
                f_ = line.split(",")
                k = round((float(f_[1]) - 0.0005) * 10)
                if 0 <= k < len(samples):
                    b5 = tuple(int(x) for x in f_[4:9])
                    a5 = tuple(int(x) for x in f_[9:14])
                    samples[k] = (round(float(f_[2]) * 4), round(float(f_[3]) * 4), b5, a5, None, None)
    bars = []
    for m in range(SESSION // 60):
        sl = range(60 * m, 60 * m + 60)
        tr = [s for s in sl if N[s] > 0]
        if not tr:
            bars.append(None)
            continue
        bars.append((O[tr[0]], max(H[s] for s in tr), min(Lo[s] for s in tr), C[tr[-1]], sum(V[s] for s in sl),
                     sum(N[s] for s in sl), sum(BV[s] for s in sl), sum(AV[s] for s in sl), "SIM"))
    for s in range(SESSION):
        if N[s]:
            bundles[s] = (N[s], V[s], BV[s], AV[s])
    win = [("W", 0) if 0 < i <= window_s * 10 else None for i in range(len(samples))]
    full = [("F", 0) if i > 0 else None for i in range(len(samples))]
    return {"market": market, "path": os.path.basename(path), "bars": bar_stats(bars, True),
            "sec": bundle_stats(N, V, BV, AV, C), "bookw": book_stats(samples, win, 0, bundles).get(("W", 0), {}),
            "bookf": book_stats(samples, full, 0, bundles).get(("F", 0), {}), "trades": sum(N)}


# ------------------------------------------------------------- report ---
def fmt(v, se, f):
    if v is None or math.isnan(v):
        return "-"
    s = f % v
    if se is not None and not math.isnan(se):
        s += " +- " + (f.replace("+", "") % se)
    return s


def verdict(r, se_r, s, se_s, floor=0.0):
    if any(x is None or math.isnan(x) for x in (r, s)):
        return "-"
    d = s - r
    sec = math.sqrt((0 if math.isnan(se_r) else se_r) ** 2 + (0 if math.isnan(se_s) else se_s) ** 2)
    if abs(d) <= 2 * sec:
        return "within 2 SE"
    if abs(d) <= 0.2 * abs(r):
        return "within 20%"
    if abs(r) >= floor and r != 0 and s / r > 0:  # a ratio only when real is clearly away from zero
        return "off x%.2f" % (s / r)
    return ("off by %+.3f" if abs(d) < 10 else "off by %+.0f") % d


def table(lines, title, rows, cols, judge, ref=0):
    """cols: [(header, blocks or None)]; judge: index of the judged column (vs column ref)."""
    res = [jackknife(b, [r[1] for r in rows]) if b else None for _, b in cols]
    w0 = max(len(r[0]) for r in rows) + 2
    widths = [max(len(h), 20) + 2 for h, _ in cols]
    lines.append("")
    lines.append(title)
    hdr = "statistic".ljust(w0) + "".join(h.rjust(w) for (h, _), w in zip(cols, widths))
    if judge is not None:
        hdr += "  verdict (%s vs %s)" % (cols[judge][0], cols[ref][0])
    lines.append(hdr)
    lines.append("-" * len(hdr))
    out = {}
    for i, r in enumerate(rows):
        cells = []
        for j, ((h, _), w) in enumerate(zip(cols, widths)):
            cells.append((fmt(*res[j][i], r[2]) if res[j] else "-").rjust(w))
        line = r[0].ljust(w0) + "".join(cells)
        if judge is not None and res[judge] and res[ref]:
            v = verdict(res[ref][i][0], res[ref][i][1], res[judge][i][0], res[judge][i][1], r[3])
            line += "  " + v
            out[r[0]] = (res[ref][i], res[judge][i], r[3], r[2], v)
        lines.append(line)
    return out


# Which simulator parameter most plausibly drives each statistic (read off simulator.hpp for
# v3 = discovery_market(), not fitted): row -> (group, driver). The ranked list shows the worst
# row of each group, so one cause (e.g. the deep queue) is not listed five times.
_DEPTH = ("book depth", "stationary depth at level i = maker_rate * exp(-i / maker_decay) * maker_size_mean / cancel_rate: "
          "v3 12 * 3 / 0.15 = 240 lots at L1, falling with distance (maker_decay 3); real ES: ~17 lots at L1 rising to ~35 at "
          "L5, ~1.2 lots per resting order")
_LEAD = ("flow leads the mid", "the mid lags fair value, then catches up in the direction of the flow: value_impact 0.05/lot "
         "and informed meta-orders (informed_meta_rate 3.0, chasing value - mid) move value first, and the mid follows only as "
         "the stale queue depletes (reprice_cancel 12, reprice_band 0.5); real ES moves within the second and slightly reverts")
_SIGN = ("1-s flow memory", "order splitting: meta-orders (informed_meta_rate 3.0 x gap, uninformed_meta_rate 0.27, "
         "Pareto(meta_min 60, meta_alpha 1.5) lots at meta_child_rate 2/s) and momentum takers (momentum_rate 0.8) against "
         "noise_rate 0.6/side: one-sided seconds persist; real 1-s net flow is nearly memoryless")
_ACT = ("trade count", "arrival rates x activity: noise_rate 0.6/side, meta children 2/s per live meta-order, momentum; "
        "Poisson within each 100 ms step (taker_size_mean 4 makes the volume right with half the trades); CAVEAT: Sierra's trade count looks like one trade per resting order filled (real 1.31 lots/trade ~ the ~1.2 lots per resting order in the book), the sim counts one per aggressive order, so trade counts and sizes may differ by definition: compare volume (lots/s), which doesn't depend on it")
_SIZE = ("trade size", "taker_size_mean 4, taker_size_logsd 0.9 (every taker kind and meta child draws from it); "
         "real ES prints average ~1.3 lots; CAVEAT: Sierra's trade count looks like one trade per resting order filled (real 1.31 lots/trade ~ the ~1.2 lots per resting order in the book), the sim counts one per aggressive order, so trade counts and sizes may differ by definition: compare volume (lots/s), which doesn't depend on it")
_VOL = ("1-min volatility", "value_vol 0.07 per sqrt(s) x sqrt(activity) plus news jumps (jump_rate 1/1200, jump_sd 8) "
        "and value_impact: v3's fair value diffuses about half as fast as real ES 2026")
_TAIL = ("1-min tails", "jumps (jump_rate 1/1200, jump_sd 8) on a book whose stale side has emptied: a sweep can print "
         "far past value in one step (see the per-session line under Table 1)")
_CLUS = ("volatility clustering", "activity: log-OU with activity_tau 600 s, activity_vol 0.045, activity_news_excite 0.8; "
         "real |r| memory lasts hours (ac30 ~0.20), v3's dies within minutes")
_MID = ("mid-change rate", "the mid moves only when fair value forces a stale level out (reprice_band 0.5, reprice_cancel "
        "12) and the spread is refilled within a step (spread_fill_prob 0.6): no 100 ms flicker; real ES ticks far more "
        "often at 100 ms, mostly back and forth within the second")
_SPREAD = ("spread", "spread_fill_prob 0.6 per step and deep queues: v3's spread is never 2 ticks in 100 ms samples")
_IMB = ("imbalance", "stale-side depletion (reprice_cancel 12, reprice_take 0.15) is the only thing that makes L1 imbalance "
        "predict; deeper levels (L2-L5) carry no information in v3")
_DEPL = ("L1 depletion", "cancel_rate 0.15/lot/s on a 240-lot queue plus reprice_cancel on stale levels: v3 depletes the "
         "best mostly by cancellation; in real ES most of the L1 churn is trading")
_SHAPE = ("intraday shape", "open_boost 1.5, close_boost 1.0, boost_minutes 30 (e-folding), and v3's meta-orders and "
          "icebergs start cold at the open")
_IMP = ("price-on-flow R^2", "value_impact 0.05 per lot of surprise, reprice_take 0.15 (prints at the stale best as the "
        "price ticks) and the informed share of flow")
_SAME = ("same-second impact", "the mid moves within the second of the flow far less in v3: impact goes to fair value "
         "(value_impact) and reaches the mid only through stale-side depletion")
DRIVERS = {
    "L1 size, bid mean (lots)": _DEPTH, "L1 size, ask mean (lots)": _DEPTH, "L1 size, 10th pct": _DEPTH,
    "L1 size, median": _DEPTH, "L1 size, 90th pct": _DEPTH, "L2 size, mean per side": _DEPTH,
    "L3 size, mean per side": _DEPTH, "L4 size, mean per side": _DEPTH, "L5 size, mean per side": _DEPTH,
    "corr(1-s delta, next 1 s mid move)": _LEAD, "corr(1-s delta, next 5 s mid move)": _LEAD,
    "corr(1-s delta, next 30 s mid move)": _LEAD, "1-s delta markout +1 s (signed, ticks)": _LEAD,
    "1-s delta markout +5 s (signed, ticks)": _LEAD, "1-s delta markout +30 s (signed, ticks)": _LEAD,
    "corr(30 s delta, next 30 s change)": _LEAD, "corr(1-min delta, next 1 min)": _LEAD, "corr(1-min delta, next 5 min)": _LEAD,
    "1-s delta sign autocorr, lag 1 s": _SIGN, "1-s delta sign autocorr, lag 10 s": _SIGN, "1-s delta sign autocorr, lag 100 s": _SIGN,
    "trades / min": _ACT, "trades / s, mean": _ACT, "trades / s, median": _ACT, "trades / s, 90th pct": _ACT,
    "trades / s, 99th pct": _ACT, "share of seconds with no trade (%)": _ACT,
    "lots / trade": _SIZE, "volume / min (lots)": _SIZE, "lots / s, mean": _SIZE, "traded lots / s": _SIZE,
    "1-min return sd (ticks)": _VOL, "1-min excess kurtosis": _TAIL, "ac1 of 1-min returns": _TAIL,
    "ac1 of |1-min returns|": _CLUS, "ac5 of |1-min returns|": _CLUS, "ac30 of |1-min returns|": _CLUS,
    "mid changes / min (100 ms samples)": _MID, "  of which 1-tick spread before and after": _MID,
    "last-price changes / min (1-s closes)": _MID, "spread = 1 tick (%)": _SPREAD,
    "event corr(I1, next mid change)": _IMB, "event corr(I5, next mid change)": _IMB,
    "next mid change goes the way I1 points (%)": _IMB,
    "L1 depletion, lots/s per side (net, 100 ms)": _DEPL, "traded lots / L1 depletion": _DEPL,
    "R^2(1-min change ~ 1-min delta)": _IMP, "R^2(10 s price change ~ 10 s delta)": _IMP,
    "1-s delta: same-second mid move (signed, ticks)": _SAME, "corr(1-s delta, same-second mid move)": _SAME,
}
for _r in SHAPE_ROWS:
    DRIVERS[_r[0]] = _SHAPE


def ranked(lines, judged, n=10):
    best = {}
    for name, (ref, sim, floor, f, v) in judged.items():
        r, s = ref[0], sim[0]
        if math.isnan(r) or math.isnan(s) or v in ("within 2 SE", "within 20%"):
            continue
        group, text = DRIVERS.get(name, (name, "(no single parameter)"))
        sev = abs(s - r) / max(abs(r), floor)
        cur = best.get(group)
        if cur is None:
            best[group] = [sev, name, r, s, f, v, text, []]
        elif sev > cur[0]:
            best[group] = [sev, name, r, s, f, v, text, cur[7] + [cur[1]]]
        else:
            cur[7].append(name)
    items = sorted(best.values(), key=lambda x: (-x[0], x[1]))
    lines.append("")
    lines.append("Biggest real-vs-v3 mismatches: the worst row of each group, ranked by |v3 - real| / max(|real|, floor) (floors in")
    lines.append("the tool: 1 for counts and sizes, 0.03-0.05 for correlations); the driver is the simulator parameter that most")
    lines.append("plausibly causes it, read off simulator.hpp (discovery_market()), not fitted. Real = RTH (book: the Nov 3 morning).")
    for k, (sev, name, r, s, f, v, text, others) in enumerate(items[:n], 1):
        lines.append("%2d. %s: real %s, v3 %s (%s)" % (k, name.strip(), f % r, f % s, v))
        if others:
            lines.append("    also off in this group: " + "; ".join(o.strip() for o in sorted(others)))
        lines.append("    driver: " + text)


def tails_note(lines, label, blocks, names, unit):
    """Per-block 1-min excess kurtosis and largest |1-min return| (outliers that dominate pooled moments)."""
    ks = sorted((f_kurt("r")(b), i) for i, b in enumerate(blocks))
    mx = sorted((b.get("_rmax", 0.0), i) for i, b in enumerate(blocks))
    lines.append("  %s: per-%s 1-min excess kurtosis median %.1f (min %.1f, max %.1f: %s); largest |1-min return| per %s:"
                 " median %.0f ticks, max %.0f (%s)" % (label, unit, ks[len(ks) // 2][0], ks[0][0], ks[-1][0], names[ks[-1][1]],
                                                        unit, mx[len(mx) // 2][0], mx[-1][0], names[mx[-1][1]]))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--bars", required=True)
    ap.add_argument("--scid-dir", required=True)
    ap.add_argument("--depth", nargs="+", required=True)
    ap.add_argument("--sim", action="append", required=True, help="market=glob of of_calibrate export CSVs")
    ap.add_argument("--scid-glob", default="ES?26_FUT_CME.scid", help="contracts in --scid-dir (default: ESH26..ESZ26)")
    ap.add_argument("--jobs", type=int, default=16)
    ap.add_argument("--out", default="-")
    ap.add_argument("--depth-until", default="", help="UTC 'YYYY-MM-DD HH:MM:SS': ignore depth events from then on")
    ap.add_argument("--note", default="", help="free text for the header (e.g. the seeds the sessions used)")
    a = ap.parse_args()
    sims = []
    for s in a.sim:
        name, pat = s.split("=", 1)
        files = sorted(glob.glob(pat))
        if not files:
            sys.exit("no files for --sim %s" % s)
        sims.append((name, files))
    until_us = None
    if a.depth_until:
        until_us = int(datetime.strptime(a.depth_until, "%Y-%m-%d %H:%M:%S").replace(tzinfo=UTC).timestamp()) * 1000000 + SC_EPOCH_US
    rth_cover = [depth_rth_seconds(p, until_us)[0] for p in a.depth]
    if sum(1 for x in rth_cover if x > 0) > 1:
        sys.exit("more than one .depth file covers RTH: the matched sim window assumes one RTH morning")
    window_s = max(rth_cover)
    scids = sorted(glob.glob(os.path.join(a.scid_dir, a.scid_glob)))
    full, eth, chk = read_bars(a.bars)
    days = sorted(full)

    jobs = [("scid", (p, os.path.basename(p).split("_")[0], days)) for p in scids]
    for p in a.depth:
        sym = os.path.basename(p).split(".")[0]
        jobs.append(("depth", (p, os.path.join(a.scid_dir, sym + ".scid"), until_us)))
    for name, files in sims:
        for p in files:
            jobs.append(("sim", (name, p, window_s)))
    fn = {"scid": task_scid_days, "depth": task_depth, "sim": task_sim}
    with mp.Pool(min(a.jobs, len(jobs))) as pool:
        res = pool.map(_run, [(k, fn[k].__name__, x) for k, x in jobs], chunksize=1)

    scid_days, depth_res, sim_res = {}, [], {name: [] for name, _ in sims}
    for (k, _), r in zip(jobs, res):
        if k == "scid":
            for d, v in r.items():
                scid_days.setdefault(d, []).append(v)
        elif k == "depth":
            depth_res.append(r)
        else:
            sim_res[r["market"]].append(r)
    front = {}
    for d in sorted(scid_days):
        best = max(scid_days[d], key=lambda v: v[0])
        front[d] = best
    sec_blocks = [front[d][2] for d in sorted(front)]
    csv_vs_front = sum(1 for d in front if full[d][0][8] != front[d][1])
    periods = {}
    for d in sorted(front):
        periods.setdefault(front[d][1], []).append(d)

    bar_blocks = [bar_stats(full[d], True) for d in days]
    eth_blocks = [bar_stats(eth[td], False) for td in sorted(eth)]
    names = [n for n, _ in sims]
    L = []
    L.append("OF12: real ES (Sierra Chart) vs the ES simulator -- the same statistics from the same code (tools/real_vs_sim.py)")
    L.append("Generated by experiments/of12_regen.sh. Aggregate statistics only (the market data stays outside the repo).")
    L.append("")
    L.append("Data")
    L.append("  1-min bars: %s: %d rows; RTH 08:30-15:00 CT on %d full days %s .. %s (%d CT days with RTH bars; days without all"
             % (os.path.basename(a.bars), chk["rows"], len(days), day_str(days[0]), day_str(days[-1]), chk["rth_days"]))
    L.append("    390 bars, i.e. half days and today's partial session, excluded); CT computed here from UTC: %d of %d rows disagree"
             " with the file's time_ct column" % (chk["ct_mismatch"], chk["rows"]))
    L.append("    ETH column: every non-RTH bar (17:00 CT open to 08:30, 15:00-16:00), per trade date (%d), returns between"
             " consecutive minutes only" % len(eth))
    L.append("  1-s bundles: Sierra .scid records (one per second with trades), front month by RTH volume per day: " +
             ", ".join("%s %s..%s (%d d)" % (c, day_str(v[0]), day_str(v[-1]), len(v)) for c, v in sorted(periods.items(), key=lambda x: x[1][0])))
    L.append("    %d days; on %d of them the 1-min file's contract is not the front month by volume" % (len(front), csv_vs_front))
    for r in sorted(depth_res, key=lambda r: r["t0"]):
        cov = ", ".join("%s %.1f min" % (k, v / 600) for k, v in sorted(r["cover"].items()))
        span = ""
        if r["rth"]:
            lo, hi = r["rth"]
            span = "; RTH %s-%s CT" % (datetime.fromtimestamp(ct_local(lo), UTC).strftime("%H:%M:%S"),
                                       datetime.fromtimestamp(ct_local(hi), UTC).strftime("%H:%M:%S"))
        L.append("  book: %s: %d events, %s .. %s UTC, sampled every 100 ms: %s%s; %d crossed/locked samples dropped"
                 % (r["path"], r["events"], datetime.fromtimestamp(r["t0"], UTC).strftime("%m-%d %H:%M"),
                    datetime.fromtimestamp(r["t1"], UTC).strftime("%m-%d %H:%M"), cov, span, r["crossed"]))
    for name, files in sims:
        rs = sim_res[name]
        L.append("  sim %s: %d sessions x 6.5 h from the open (%s .. %s), %.1f trades/s overall"
                 % (name, len(rs), rs[0]["path"], rs[-1]["path"], sum(x["trades"] for x in rs) / (len(rs) * SESSION)))
    if a.depth_until:
        L.append("  book events from %s UTC on are ignored (Sierra is still writing the last .depth file)" % a.depth_until)
    if a.note:
        L.append("  " + a.note)
    L.append("  The simulator's book table uses its first %d s after the open (the real RTH book window's length) as the"
             " matched window; its full-session book is in the last table." % window_s)
    L.append("  SE: delete-one-block jackknife. Real: blocks = days (bars, 1-s), 5-min blocks of the ONE RTH book window"
             " (within-morning noise only:")
    L.append("    no day-to-day variation; treat the real RTH book column as one observation), 30-min ETH blocks."
             " Sim: blocks = sessions.")
    L.append("  Verdict (v3 vs real): within 2 SE (combined) / within 20% / off xR (ratio v3/real) / off by D (signs differ, or")
    L.append("    |real| is below the row's floor: 0.03-0.05 for correlations, so a ratio of two near-zero numbers isn't quoted).")

    def sim_cols(key):
        return [(n, [r[key] for r in sim_res[n]]) for n in names]

    judged = {}
    judged.update(table(L, "Table 1. 1-minute bars, RTH (real: the year; sim: whole sessions). Prices in ticks of 0.25.",
                        BAR_ROWS + SHAPE_ROWS, [("real RTH", bar_blocks)] + sim_cols("bars") + [("real ETH", eth_blocks)], 1))
    L.append("Tails per block (a pooled moment above can hinge on one block):")
    tails_note(L, "real RTH", bar_blocks, [day_str(d) for d in days], "day")
    for n in names:
        tails_note(L, n, [r["bars"] for r in sim_res[n]], [r["path"] for r in sim_res[n]], "session")
    judged.update(table(L, "Table 2. 1-second bundles, RTH (real: front month; one bundle = all trades within a whole second).",
                        SEC_ROWS, [("real RTH", sec_blocks)] + sim_cols("sec"), 1))

    def book_blocks(label):
        return [a_ for r in depth_res for k, a_ in sorted(r["accs"].items()) if k[0] == label]
    rth_b, eth_b = book_blocks("RTH"), book_blocks("ETH")
    cols = [("real RTH book", rth_b)] + [(n + " 0-%.0fm" % (window_s / 60), [r["bookw"] for r in sim_res[n]]) for n in names] + \
           [("real ETH book", eth_b)]
    judged.update(table(L, "Table 3. Book (100 ms samples), the first %.0f min after the 08:30 CT open: real = Nov 3 only; "
                        "sim = the same minutes of each session. ETH = overnight/pre-open/post-close depth." % (window_s / 60),
                        BOOK_ROWS, cols, 1))
    table(L, "Table 3b. Real-only book detail (the simulator has no order counts).", BOOK_REAL_ONLY,
          [("real RTH book", rth_b), ("real ETH book", eth_b)], None)
    table(L, "Table 3c. Simulator book over whole 6.5 h sessions (context; verdict vs the real RTH morning window).",
          BOOK_ROWS, [("real RTH book", rth_b)] + sim_cols("bookf"), 1)
    ranked(L, judged)
    L.append("")
    L.append("Not compared (one side lacks the detail): per-trade sign autocorrelation and markouts after single prints (the real")
    L.append("  data is 1-s bundles: since Nov 3 Sierra stamps whole seconds; .depth carries no trades); quote-level (sub-100 ms)")
    L.append("  book dynamics (the sim moves in 100 ms steps); order counts per level (sim: none); real 1-s and book statistics")
    L.append("  outside the windows above. Markouts here are of a whole second's delta against the depth mid at the end of the")
    L.append("  second, on both sides.")
    text = "\n".join(L) + "\n"
    if a.out == "-":
        sys.stdout.write(text)
    else:
        with open(a.out, "w", newline="\n") as f:
            f.write(text)


def _run(job):
    kind, name, x = job
    return globals()[name](x)


if __name__ == "__main__":
    main()
