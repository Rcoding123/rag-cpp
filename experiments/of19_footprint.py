#!/usr/bin/env python3
"""OF19: footprint order-flow patterns on true ES tick data (docs/ai/OF19_FOOTPRINT_PLAN.md).

  python experiments/of19_footprint.py --out experiments/of19_footprint.txt [--trades build-avx512/real/trades]

Reads tools/scid_to_trades.py's per-day trades (2026-09-08 .. 2026-11-11, front month, one row per trade with
the aggressor side and the record's bid / ask), each npz checked against its sidecar sha256. The first 10 days
only feed trailing thresholds; the other 37 are scored. Patterns, fills, exits, statistics and the decision
rule are the plan's; implementation details the plan leaves open are marked "detail:" below.
"""
import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "python"))
from ragrl import real  # noqa: E402

START, END = "2026-09-08", "2026-11-11"
WARM = 10
FEE = 0.24
STRESS = 0.5
LAT = 0.1
SWEEP_GAP = 1.5e-6  # seconds: records 1 us apart belong to one exchange message
BAR = 60.0
NBARS = 390
RTH = NBARS * BAR
BUCKET = 1800.0
BLOCKS, NBOOT, SEED = (2, 5, 10), 10_000, 0
K_CTRL = 10
CAP = {"P1": 900.0, "P2": 300.0, "P3": 900.0, "P4": 900.0, "P5": 900.0}

PRIMARY = {"P1": dict(r=3, k=3), "P2": dict(W=30, tgt=4), "P3": dict(N=1, e=5), "P4": dict(d=4), "P5": dict(M=2, trig="stack")}
RULES = ([("P1", dict(r=r, k=k)) for r in (3, 4) for k in (3, 4)]
         + [("P2", dict(W=W, tgt=t)) for W in (15, 30, 60) for t in (4, 8)]
         + [("P3", dict(N=N, e=e)) for N in (1, 20) for e in (0, 5)]
         + [("P4", dict(d=d)) for d in (4, 8)]
         + [("P5", dict(M=M, trig=g)) for M in (1, 2) for g in ("stack", "delta")])
M_PRI, M_SEC = 5, len(RULES)
TREQ_PRI = 2 + math.sqrt(2 * math.log(M_PRI))
TREQ_SEC = 2 + math.sqrt(2 * math.log(M_SEC))


def key(p, kw):
    return p + " " + " ".join(f"{k}={v}" for k, v in kw.items())


# ---------------------------------------------------------------- data
def load_days(trades_dir: Path):
    m = real.load_manifest()
    if END >= m["confirm_rule"]["from"]:
        raise RuntimeError("the span reaches the confirm block")
    out = []
    for sp in sorted(trades_dir.glob("*.json")):
        d = sp.stem
        if not (START <= d <= END):
            continue
        side = json.loads(sp.read_text(encoding="utf-8"))
        npz = trades_dir / f"{d}.npz"
        if hashlib.sha256(npz.read_bytes()).hexdigest() != side["npz_sha256"]:
            raise RuntimeError(f"{npz}: sha256 does not match its sidecar")
        if side["volume"] + side["dropped_volume"] != side["bars_volume"]:
            raise RuntimeError(f"{d}: trade volume does not match the OF18 bars")
        out.append((d, npz))
    return out


def load_trades(npz):
    z = np.load(npz)
    t, px, bid, ask = z["t"], z["price"].astype(np.int64), z["bid"].astype(np.int64), z["ask"].astype(np.int64)
    size, side = z["size"].astype(np.int64), z["side"].astype(np.int64)
    bad = (bid >= ask) | (bid <= 0) | (ask <= 0)
    # detail: an invalid quote takes the last valid one (plan section 2)
    if bad.any():
        good = np.flatnonzero(~bad)
        j = np.searchsorted(good, np.arange(len(t)), side="right") - 1
        j = np.maximum(j, 0)
        bid, ask = np.where(bad, bid[good[j]], bid), np.where(bad, ask[good[j]], ask)
    qc = {"buy_at_ask": int(((side > 0) & (px == ask)).sum()), "buys": int((side > 0).sum()),
          "sell_at_bid": int(((side < 0) & (px == bid)).sum()), "sells": int((side < 0).sum()), "bad_quotes": int(bad.sum())}
    return {"t": t, "px": px, "bid": bid, "ask": ask, "size": size, "side": side}, qc


class Events:
    """Records sharing a timestamp form one event; sparse tables answer 'first event >= s with px_min <= x / px_max >= x'."""

    def __init__(self, T):
        t = T["t"]
        # Sierra stamps the records of one exchange message 1 us apart (OF19 review): a chain of records each within
        # SWEEP_GAP of the previous one is one event (a sweep), as the plan intends by "records sharing one timestamp"
        brk = np.r_[True, np.diff(t) > SWEEP_GAP]
        start = np.flatnonzero(brk)
        self.t = t[start]
        self.n = len(start)
        red = lambda f, a: f.reduceat(a, start)  # noqa: E731
        self.px_min, self.px_max = red(np.minimum, T["px"]), red(np.maximum, T["px"])
        self.bid_min, self.ask_max = red(np.minimum, T["bid"]), red(np.maximum, T["ask"])
        self.long_fill = np.maximum(self.ask_max, self.px_max)    # buy at the worst of the asks and trades
        self.short_fill = np.minimum(self.bid_min, self.px_min)   # sell at the worst of the bids and trades
        self.mn = self._table(self.px_min, np.minimum)
        self.mx = self._table(self.px_max, np.maximum)

    def _table(self, a, f):
        tab = [a]
        k = 1
        while 2 * k <= self.n:
            prev = tab[-1]
            tab.append(f(prev[:-k], prev[k:]))
            k *= 2
        return tab

    def first(self, starts, x, below: bool):
        """First event index >= start whose px_min <= x (below) or px_max >= x; self.n if none."""
        starts = np.asarray(starts, np.int64).copy()
        x = np.asarray(x)
        tab = self.mn if below else self.mx
        pos = starts
        for lvl in range(len(tab) - 1, -1, -1):
            span = 1 << lvl
            arr = tab[lvl]
            ok = pos + span <= self.n
            idx = np.minimum(pos, len(arr) - 1)
            v = arr[idx]
            miss = (v > x) if below else (v < x)
            pos = np.where(ok & miss, pos + span, pos)
        inside = pos < self.n
        hit = np.zeros(len(pos), bool)
        hit[inside] = (self.px_min[pos[inside]] <= x[inside]) if below else (self.px_max[pos[inside]] >= x[inside])
        return np.where(hit, pos, self.n)

    def at_or_after(self, times):
        return np.searchsorted(self.t, times, side="left")


# ---------------------------------------------------------------- the trade simulator (plan section 3, fills and exits)
def _exits(E, e0, side, fill, stop, tgt, cap, touch):
    s = e0 + 1
    lng = side > 0
    # stop: long when a trade prints at or below the stop; short at or above
    st = np.where(lng, E.first(s, stop, True), E.first(s, stop, False))
    # target: a trade one tick through it (or on touch)
    thr = np.where(lng, tgt + (0 if touch else 1), tgt - (0 if touch else 1))
    has_t = np.isfinite(tgt)
    thr_safe = np.where(has_t, thr, np.where(lng, 1e18, -1e18))
    tg = np.where(lng, E.first(s, thr_safe, False), E.first(s, thr_safe, True))
    tg = np.where(has_t, tg, E.n)
    cp = E.at_or_after(E.t[e0] + cap)
    last = E.n - 1
    first = np.minimum(np.minimum(st, tg), cp)
    exit_e = np.minimum(first, last)
    # detail: a stop and a target in the same event count as the stop (conservative)
    px_stop = np.where(lng, E.short_fill[np.minimum(st, last)], E.long_fill[np.minimum(st, last)])
    px_cap = np.where(lng, E.short_fill[exit_e], E.long_fill[exit_e])
    out = np.where(st <= np.minimum(tg, cp), px_stop, np.where(tg < cp, tgt, px_cap))
    out = np.where((st >= E.n) & (tg >= E.n) & (cp >= E.n), np.where(lng, E.short_fill[last], E.long_fill[last]), out)
    gross = np.where(lng, out - fill, fill - out)
    chosen = np.where(st <= np.minimum(tg, cp), np.minimum(st, last), np.where(tg < cp, tg, exit_e))
    return gross, chosen


# ---------------------------------------------------------------- footprint bars
def footprint(T):
    b = np.minimum((T["t"] // BAR).astype(np.int64), NBARS - 1)
    lo = int(T["px"].min()) - 3
    p = T["px"] - lo
    P = int(p.max()) + 4
    A = np.zeros((NBARS, P), np.int64)
    Bv = np.zeros((NBARS, P), np.int64)
    np.add.at(A, (b[T["side"] > 0], p[T["side"] > 0]), T["size"][T["side"] > 0])
    np.add.at(Bv, (b[T["side"] < 0], p[T["side"] < 0]), T["size"][T["side"] < 0])
    has = np.zeros(NBARS, bool)
    has[np.unique(b)] = True
    high = np.full(NBARS, -1, np.int64)
    low = np.full(NBARS, 1 << 40, np.int64)
    np.maximum.at(high, b, T["px"])
    np.minimum.at(low, b, T["px"])
    close = np.zeros(NBARS, np.int64)
    last = np.r_[np.flatnonzero(b[1:] != b[:-1]), len(b) - 1]
    close[b[last]] = T["px"][last]
    return {"A": A, "B": Bv, "lo": lo, "high": high, "low": low, "close": close, "has": has,
            "delta": A.sum(1) - Bv.sum(1), "end": BAR * (np.arange(NBARS) + 1)}


def stacks(F, r, k):
    """Per bar: (buy stack prices, sell stack prices) as lists of runs (absolute ticks)."""
    A, Bv, lo = F["A"], F["B"], F["lo"]
    buy = np.zeros_like(A, bool)
    sell = np.zeros_like(A, bool)
    buy[:, 1:] = (A[:, 1:] >= r * Bv[:, :-1]) & (A[:, 1:] >= 10) & (Bv[:, :-1] > 0)
    sell[:, :-1] = (Bv[:, :-1] >= r * A[:, 1:]) & (Bv[:, :-1] >= 10) & (A[:, 1:] > 0)
    out = []
    for bi in range(NBARS):
        res = []
        for mask in (buy[bi], sell[bi]):
            runs, i, n = [], 0, len(mask)
            idx = np.flatnonzero(mask)
            if len(idx):
                br = np.flatnonzero(np.diff(idx) != 1)
                starts = np.r_[idx[0], idx[br + 1]]
                ends = np.r_[idx[br], idx[-1]]
                runs = [(s + lo, e + lo) for s, e in zip(starts, ends) if e - s + 1 >= k]
            res.append(runs)
        out.append(res)
    return out


# ---------------------------------------------------------------- signals: arrays (sig_t, side, stop, tgt)
def sig_P1(F, st):
    rows = []
    for b in range(NBARS):
        if not F["has"][b]:
            continue
        bs, ss = st[b]
        if bs and not ss:
            hi_run = max(bs, key=lambda x: x[1])  # detail: the highest buy stack
            rows.append((F["end"][b], 1, hi_run[0] - 2, np.nan))
        elif ss and not bs:
            lo_run = min(ss, key=lambda x: x[0])  # detail: the lowest sell stack
            rows.append((F["end"][b], -1, lo_run[1] + 2, np.nan))
    return rows


def sig_P3(F, E, N, e, shell=False):
    A, Bv, lo = F["A"], F["B"], F["lo"]
    rows = []
    for b in range(N, NBARS):
        if not F["has"][b] or not F["has"][b - N:b].all():
            continue
        for side in (1, -1):
            if side > 0:
                ext = F["low"][b]
                if not ext < F["low"][b - N:b].min():
                    continue
                i = ext - lo
                cond = Bv[b, i] <= e and Bv[b, i] < Bv[b, i + 1] < Bv[b, i + 2]
                trig, stop = F["high"][b] + 1, ext - 1
            else:
                ext = F["high"][b]
                if not ext > F["high"][b - N:b].max():
                    continue
                i = ext - lo
                cond = A[b, i] <= e and A[b, i] < A[b, i - 1] < A[b, i - 2]
                trig, stop = F["low"][b] - 1, ext + 1
            if not (cond or shell):
                continue
            s0 = E.at_or_after(F["end"][b])
            if s0 >= E.n:
                continue
            ti = E.first([s0], [trig], side < 0)[0]
            ci = E.first([s0], [stop], side > 0)[0]
            if ti >= E.n or ti >= ci or E.t[ti] > F["end"][b] + 5 * BAR:
                continue
            rows.append((E.t[ti], side, stop, np.nan))
    return rows


def sig_P4(F, d, shell=False):
    A, Bv, lo = F["A"], F["B"], F["lo"]
    rows = []
    for b in range(NBARS):
        if not F["has"][b]:
            continue
        H, L, C = F["high"][b], F["low"][b], F["close"][b]
        uh = A[b, H - lo] > 0 and Bv[b, H - lo] > 0
        ul = A[b, L - lo] > 0 and Bv[b, L - lo] > 0
        far_h, far_l = C <= H - d, C >= L + d
        if shell:
            uh, ul = not uh, not ul  # finished extremes, same distance
        go_h, go_l = uh and far_h, ul and far_l
        if go_h and go_l:
            continue
        if go_h:
            rows.append((F["end"][b], 1, np.nan, float(H)))   # stop set from the fill: -8 (stop_dist)
        elif go_l:
            rows.append((F["end"][b], -1, np.nan, float(L)))
    return rows


def sig_P5(F, st, M, trig, dq, shell=False):
    rows = []
    for b in range(20, NBARS - 1):
        if not F["has"][b] or not F["has"][b - 20:b].all():
            continue
        for side in (1, -1):
            if side > 0:
                L = F["low"][b - 20:b].min()
                if not F["low"][b] < L:
                    continue
                if trig == "stack":
                    ok = any(r0 <= F["low"][b] + 2 for r0, _ in st[b][1])
                else:
                    ok = F["delta"][b] <= dq[0][b]
                stop = F["low"][b] - 1
                back = [j for j in range(b + 1, min(b + M, NBARS - 1) + 1) if F["has"][j] and F["close"][j] > L]
            else:
                Hh = F["high"][b - 20:b].max()
                if not F["high"][b] > Hh:
                    continue
                if trig == "stack":
                    ok = any(r1 >= F["high"][b] - 2 for _, r1 in st[b][0])
                else:
                    ok = F["delta"][b] >= dq[1][b]
                stop = F["high"][b] + 1
                back = [j for j in range(b + 1, min(b + M, NBARS - 1) + 1) if F["has"][j] and F["close"][j] < Hh]
            if (ok or shell) and back:
                rows.append((F["end"][back[0]], side, stop, np.nan))
    return rows


def absorption_stats(T, Ws=(15, 30, 60)):
    """Per second t = 1..23400 and side: S_t (aggressor volume at the touch price over (t - W, t]), the touch price,
    and the window extreme of trade prices before and after (for the conditions). Returns dict per side."""
    t = T["t"]
    sec = np.minimum(np.floor(t).astype(np.int64) + 1, int(RTH))  # trade in (s-1, s] -> second s
    S = int(RTH)
    lo = int(T["px"].min()) - 3
    P = int(T["px"].max()) - lo + 4
    secs = np.arange(1, S + 1)
    last_tr = np.searchsorted(t, secs, side="right") - 1
    valid = last_tr >= 0
    lt = np.maximum(last_tr, 0)
    out = {}
    pmin = np.full(S + 1, 1 << 40, np.int64)
    pmax = np.full(S + 1, -1, np.int64)
    np.minimum.at(pmin, sec, T["px"])
    np.maximum.at(pmax, sec, T["px"])
    for side in (1, -1):
        m = T["side"] == (-1 if side > 0 else 1)  # long: sellers into the bid; short: buyers into the ask
        vol = np.zeros((S + 1, P), np.int64)
        np.add.at(vol, (sec[m], T["px"][m] - lo), T["size"][m])
        cum = np.cumsum(vol, axis=0)
        touch = np.where(side > 0, T["bid"][lt], T["ask"][lt])
        pi = np.clip(touch - lo, 0, P - 1)
        here = cum[secs, pi]
        out[side] = {"touch": touch, "valid": valid,
                     "S": {W: np.where(valid, here - cum[np.maximum(secs - W, 0), pi], 0) for W in Ws}}
    out["pmin"], out["pmax"] = pmin, pmax
    return out


def window_ext(arr, W, f, fill):
    """f over arr[s - W + 1 .. s] for s = 1..S (arr indexed 0..S)."""
    pad = np.r_[np.full(W - 1, fill, arr.dtype), arr[1:]]
    return f(np.lib.stride_tricks.sliding_window_view(pad, W), axis=1)


def fwd_ext(arr, n, f, fill):
    """f over arr[s + 1 .. s + n] for s = 1..S."""
    pad = np.r_[arr[2:], np.full(n, fill, arr.dtype)]
    return f(np.lib.stride_tricks.sliding_window_view(pad, n), axis=1)


def sig_P2(T, E, AB, W, thr, shell=False):
    """thr: per side an array over seconds 1..S of the trailing q95 (by 30-min bucket)."""
    S = int(RTH)
    rows = []
    for side in (1, -1):
        a = AB[side]
        touch = a["touch"]
        if side > 0:
            wmin = window_ext(AB["pmin"], W, np.min, 1 << 40)
            fmin = fwd_ext(AB["pmin"], 10, np.min, 1 << 40)
            held, conf = wmin >= touch, fmin >= touch
        else:
            wmax = window_ext(AB["pmax"], W, np.max, -1)
            fmax = fwd_ext(AB["pmax"], 10, np.max, -1)
            held, conf = wmax <= touch, fmax <= touch
        Sw = a["S"][W]
        big = (Sw >= np.maximum(thr[side], 1)) if not shell else np.ones(len(Sw), bool)  # the shell drops the S condition
        cand = a["valid"] & big & held
        nxt = 0
        for i in np.flatnonzero(cand & conf):
            t = i + 1
            if t < nxt or t + 10 > S:
                continue
            nxt = t + W
            P = int(touch[i])
            sig = float(t + 10)
            s0 = E.at_or_after(sig)
            if s0 >= E.n:
                continue
            if side > 0:
                ti = E.first([s0], [P + 1], False)[0]
                ci = E.first([s0], [P - 2], True)[0]
                stop = P - 2
            else:
                ti = E.first([s0], [P - 1], True)[0]
                ci = E.first([s0], [P + 2], False)[0]
                stop = P + 2
            if ti >= E.n or ti >= ci or E.t[ti] > sig + 60:
                continue
            rows.append((E.t[ti], side, stop, np.nan))
    return rows


# ---------------------------------------------------------------- per-day driver
def bucket_of(sec_t):
    return np.minimum((np.asarray(sec_t) // BUCKET).astype(np.int64), int(RTH // BUCKET))


def run_rows(E, rows, p, kw, touch=False):
    if not rows:
        return np.zeros(0), np.zeros(0, bool), np.zeros(0), np.zeros(0, np.int64), np.zeros(0), np.zeros(0)
    a = np.array(rows, float)
    sig_t, side, stop, tgt = a[:, 0], a[:, 1].astype(np.int64), a[:, 2], a[:, 3]
    e0 = E.at_or_after(sig_t + LAT)
    keep = e0 < E.n
    e0c = np.minimum(e0, E.n - 1)
    fill = np.where(side > 0, E.long_fill[e0c], E.short_fill[e0c]).astype(float)
    if p == "P4":
        stop = np.where(side > 0, fill - 8, fill + 8)
    if p == "P2":
        tgt = np.where(side > 0, fill + kw["tgt"], fill - kw["tgt"])
    keep &= np.where(side > 0, fill > stop, fill < stop)
    gross, xe = _exits(E, e0c, side, fill, stop, tgt, CAP[p], touch)
    return gross, keep, sig_t, side, np.abs(fill - stop), E.t[xe] - E.t[e0c]


def controls(E, rng, sig_t, side, stop_dist, tgt_dist, cap):
    """K random seconds per signal trade, same 30-min bucket and side; stop / target as distances from the own fill."""
    if len(sig_t) == 0:
        return np.zeros(0)
    b = bucket_of(sig_t)
    lo = b * BUCKET
    hi = np.minimum(lo + BUCKET, RTH - 1)
    rt = lo[:, None] + rng.random((len(sig_t), K_CTRL)) * (hi - lo)[:, None]
    rs = np.repeat(side[:, None], K_CTRL, 1).ravel()
    rt = rt.ravel()
    sd = np.repeat(stop_dist[:, None], K_CTRL, 1).ravel()
    td = np.repeat(tgt_dist[:, None], K_CTRL, 1).ravel()
    e0 = E.at_or_after(rt + LAT)
    keep = e0 < E.n
    e0c = np.minimum(e0, E.n - 1)
    fill = np.where(rs > 0, E.long_fill[e0c], E.short_fill[e0c]).astype(float)
    stop = np.where(rs > 0, fill - sd, fill + sd)
    tgt = np.where(np.isfinite(td), np.where(rs > 0, fill + td, fill - td), np.nan)
    g, _ = _exits(E, e0c, rs, fill, stop, tgt, cap, False)
    g = np.where(keep, g, np.nan)
    return np.nanmean(g.reshape(len(sig_t), K_CTRL), axis=1)


def day_pass1(T, AB, F):
    """Trailing-threshold inputs: per W and side the per-second S, and bar delta, with buckets."""
    out = {}
    for W in (15, 30, 60):
        out[W] = {s: AB[s]["S"][W][AB[s]["valid"]] for s in (1, -1)}
        out[W]["bucket"] = {s: bucket_of(np.arange(1, int(RTH) + 1)[AB[s]["valid"]] - 1) for s in (1, -1)}
    out["delta"] = F["delta"][F["has"]]
    out["delta_bucket"] = bucket_of(F["end"][F["has"]] - 1)
    return out


def thresholds(hist):
    nb = int(RTH // BUCKET) + 1
    thr = {}
    for W in (15, 30, 60):
        thr[W] = {}
        for s in (1, -1):
            vals = np.concatenate([h[W][s] for h in hist])
            bks = np.concatenate([h[W]["bucket"][s] for h in hist])
            q = np.array([np.quantile(vals[bks == k], 0.95) if (bks == k).any() else np.inf for k in range(nb)])
            thr[W][s] = q[bucket_of(np.arange(int(RTH)))]
    dv = np.concatenate([h["delta"] for h in hist])
    db = np.concatenate([h["delta_bucket"] for h in hist])
    q10 = np.array([np.quantile(dv[db == k], 0.10) if (db == k).any() else -np.inf for k in range(nb)])
    q90 = np.array([np.quantile(dv[db == k], 0.90) if (db == k).any() else np.inf for k in range(nb)])
    bb = bucket_of(BAR * np.arange(NBARS))
    thr["delta"] = (q10[bb], q90[bb])
    return thr


def revisits(F, E, d):
    """Unfinished vs finished highs (and lows): the share revisited within 15 and 30 min, close at least d away."""
    A, Bv, lo = F["A"], F["B"], F["lo"]
    res = {}
    for kind in ("unfinished", "finished"):
        for horizon in (900.0, 1800.0):
            hits = n = 0
            for b in range(NBARS):
                if not F["has"][b]:
                    continue
                H, L, C = F["high"][b], F["low"][b], F["close"][b]
                for ext, far, below in ((H, C <= H - d, False), (L, C >= L + d, True)):
                    if not far:
                        continue
                    unf = A[b, ext - lo] > 0 and Bv[b, ext - lo] > 0
                    if (kind == "unfinished") != unf:
                        continue
                    s0 = E.at_or_after(F["end"][b])
                    if s0 >= E.n:
                        continue
                    j = E.first([s0], [ext], below)[0]
                    n += 1
                    hits += j < E.n and E.t[j] <= F["end"][b] + horizon
            res[(kind, horizon)] = (hits, n)
    return res


# ---------------------------------------------------------------- statistics
def block_idx(n, b, rng):
    k = math.ceil(n / b)
    st = rng.integers(0, n - b + 1, size=(NBOOT, k))
    return (st[:, :, None] + np.arange(b)[None, None, :]).reshape(NBOOT, k * b)[:, :n]


def ratio_se(num, den, idx):
    ses = []
    for I in idx.values():
        dn = den[I].sum(1)
        r = np.where(dn > 0, num[I].sum(1) / np.maximum(dn, 1), np.nan)
        ses.append(float(np.nanstd(r, ddof=1)))
    return max(ses)


def diff_se(n1, d1, n2, d2, idx):
    """Block SE of (sum n1 / sum d1) - (sum n2 / sum d2) over resampled days (the largest over block lengths)."""
    ses = []
    for I in idx.values():
        a, b = d1[I].sum(1), d2[I].sum(1)
        r = np.where((a > 0) & (b > 0), n1[I].sum(1) / np.maximum(a, 1) - n2[I].sum(1) / np.maximum(b, 1), np.nan)
        ses.append(float(np.nanstd(r, ddof=1)))
    return max(ses)


def n_(x, d=2, sign=True):
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "-"
    return f"{x:+.{d}f}" if sign else f"{x:.{d}f}"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--trades", default=str(REPO / "build-avx512" / "real" / "trades"))
    a = ap.parse_args(argv)
    days = load_days(Path(a.trades))
    names = [d for d, _ in days]
    if len(days) != 47:
        raise RuntimeError(f"expected 47 tick days, found {len(days)}")
    hist, qc_all = [], []
    ndays = len(days)
    nsc = ndays - WARM
    acc = {key(p, kw): {k: np.zeros(nsc) for k in ("net", "gross", "n", "win", "skip", "long_n", "long_net", "ctrl_diff", "ctrl_n",
                                                    "shell_gross", "shell_n", "touch_net", "short_hold_n", "short_hold_gross")}
           for p, kw in RULES}
    tod = {key(p, kw): np.zeros(int(RTH // BUCKET) + 1) for p, kw in RULES}
    rev = {}
    for di, (d, npz) in enumerate(days):
        T, qc = load_trades(npz)
        qc_all.append(qc)
        AB = absorption_stats(T)
        F = footprint(T)
        if di >= WARM:
            si = di - WARM
            thr = thresholds(hist[-WARM:])
            E = Events(T)
            rng = np.random.default_rng([SEED, di])
            st3 = {(r, k): stacks(F, r, k) for r in (3, 4) for k in (3, 4)}
            for p, kw in RULES:
                kk = key(p, kw)
                shell_rows = None
                if p == "P1":
                    rows = sig_P1(F, st3[(kw["r"], kw["k"])])
                elif p == "P2":
                    rows = sig_P2(T, E, AB, kw["W"], thr[kw["W"]])
                    shell_rows = sig_P2(T, E, AB, kw["W"], thr[kw["W"]], shell=True)
                elif p == "P3":
                    rows = sig_P3(F, E, kw["N"], kw["e"])
                    shell_rows = sig_P3(F, E, kw["N"], kw["e"], shell=True)
                elif p == "P4":
                    rows = sig_P4(F, kw["d"])
                    shell_rows = sig_P4(F, kw["d"], shell=True)
                else:
                    rows = sig_P5(F, st3[(3, 3)], kw["M"], kw["trig"], thr["delta"])
                    shell_rows = sig_P5(F, st3[(3, 3)], kw["M"], kw["trig"], thr["delta"], shell=True)
                g, keep, sig_t, side, sdist, hold = run_rows(E, rows, p, kw)
                A_ = acc[kk]
                A_["skip"][si] = int((~keep).sum())
                g, sig_t, side, sdist, hold = g[keep], sig_t[keep], side[keep], sdist[keep], hold[keep]
                A_["short_hold_n"][si] = (hold < 60).sum()
                A_["short_hold_gross"][si] = g[hold < 60].sum()
                A_["n"][si] = len(g)
                A_["gross"][si] = g.sum()
                A_["net"][si] = (g - FEE).sum()
                A_["win"][si] = (g - FEE > 0).sum()
                A_["long_n"][si] = (side > 0).sum()
                A_["long_net"][si] = (g[side > 0] - FEE).sum()
                np.add.at(tod[kk], bucket_of(sig_t), 1)
                if p in ("P2", "P4"):
                    gt, kt, *_ = run_rows(E, rows, p, kw, touch=True)
                    A_["touch_net"][si] = (gt[kt] - FEE).sum()
                tdist = np.full(len(g), np.nan)
                if p == "P2":
                    tdist[:] = kw["tgt"]
                elif p == "P4" and len(rows):
                    ar = np.array(rows, float)[keep]
                    e0 = np.minimum(E.at_or_after(ar[:, 0] + LAT), E.n - 1)
                    fl = np.where(ar[:, 1] > 0, E.long_fill[e0], E.short_fill[e0])
                    tdist = np.abs(ar[:, 3] - fl)
                cg = controls(E, rng, sig_t, side, sdist, tdist, CAP[p])
                ok = np.isfinite(cg)
                A_["ctrl_diff"][si] = (g[ok] - cg[ok]).sum()
                A_["ctrl_n"][si] = ok.sum()
                if shell_rows is not None:
                    gs, ks, *_ = run_rows(E, shell_rows, p, kw)
                    A_["shell_gross"][si] = gs[ks].sum()
                    A_["shell_n"][si] = ks.sum()
            for dd in (4, 8):
                for kx, v in revisits(F, E, dd).items():
                    h0, n0 = rev.get((dd, *kx), (0, 0))
                    rev[(dd, *kx)] = (h0 + v[0], n0 + v[1])
            print(f"{d}: scored ({si + 1}/{nsc})", flush=True)
        hist.append(day_pass1(T, AB, F))
    # ---- statistics
    idx = {b: block_idx(nsc, b, np.random.default_rng(SEED)) for b in BLOCKS}
    half = nsc // 2
    R = {}
    for p, kw in RULES:
        kk = key(p, kw)
        A_ = acc[kk]
        n = A_["n"].sum()
        r = {"p": p, "kw": kw, "trades": int(n), "skipped": int(A_["skip"].sum())}
        if n > 0:
            net = A_["net"].sum() / n
            se = ratio_se(A_["net"], A_["n"], idx)
            n1, n2 = A_["n"][:half].sum(), A_["n"][half:].sum()
            r.update(net=net, gross=A_["gross"].sum() / n, se=se, t=net / se if se > 0 else float("nan"), win=A_["win"].sum() / n,
                     h1=A_["net"][:half].sum() / n1 if n1 else float("nan"), h2=A_["net"][half:].sum() / n2 if n2 else float("nan"),
                     long_share=A_["long_n"].sum() / n,
                     long_net=A_["long_net"].sum() / A_["long_n"].sum() if A_["long_n"].sum() else float("nan"),
                     short_net=(A_["net"].sum() - A_["long_net"].sum()) / (n - A_["long_n"].sum()) if n > A_["long_n"].sum() else float("nan"))
            r["stress"] = r["net"] - STRESS
            cn = A_["ctrl_n"].sum()
            r["ctrl"] = (A_["ctrl_diff"].sum() / cn, ratio_se(A_["ctrl_diff"], A_["ctrl_n"], idx)) if cn else (float("nan"),) * 2
            sn = A_["shell_n"].sum()
            r["shell"] = ((r["gross"] - A_["shell_gross"].sum() / sn, diff_se(A_["gross"], A_["n"], A_["shell_gross"], A_["shell_n"], idx),
                           int(sn)) if sn else (float("nan"), float("nan"), 0))
            sh = A_["short_hold_n"].sum()
            r["short_hold"] = (A_["short_hold_gross"].sum() / sh if sh else float("nan"), int(sh))
            r["touch"] = A_["touch_net"].sum() / n if p in ("P2", "P4") else float("nan")
        else:
            r.update(net=float("nan"), gross=float("nan"), se=float("nan"), t=float("nan"), win=float("nan"), h1=float("nan"),
                     h2=float("nan"), stress=float("nan"), ctrl=(float("nan"),) * 2, shell=(float("nan"), float("nan"), 0),
                     touch=float("nan"), short_hold=(float("nan"), 0),
                     long_share=float("nan"), long_net=float("nan"), short_net=float("nan"))
        R[kk] = r

    def edge(r, t_bar):
        return r["trades"] > 0 and r["net"] > 0 and r["t"] >= t_bar and r["h1"] > 0 and r["h2"] > 0 and r["stress"] > 0

    L = []
    w = L.append
    q = {k: sum(x[k] for x in qc_all) for k in qc_all[0]}
    share = (q["buy_at_ask"] + q["sell_at_bid"]) / max(1, q["buys"] + q["sells"])
    w("OF19: footprint order-flow patterns on true ES tick data (docs/ai/OF19_FOOTPRINT_PLAN.md)")
    w("Generated by experiments/of19_footprint.py from tools/scid_to_trades.py (Sierra tick records, front month by RTH volume).")
    w(f"days {ndays} ({names[0]} .. {names[-1]}); warm-up {WARM} (thresholds only); scored {nsc} ({names[WARM]} .. {names[-1]}; "
      f"halves split before {names[WARM + half]})")
    zs = sum(json.loads((Path(a.trades) / f"{d}.json").read_text(encoding="utf-8"))["inferred_side_records"] for d in names)
    w(f"quote check: buy-aggressor trades at the record's ask {q['buy_at_ask'] / max(1, q['buys']):.4f}, sell-aggressor at the bid "
      f"{q['sell_at_bid'] / max(1, q['sells']):.4f} (both: {share:.4f}; the plan stops below 0.90); invalid quotes {q['bad_quotes']}; "
      f"zero-side records (side from the quote) {zs}")
    if share < 0.90:
        w("QUOTE CHECK FAILED: the study stops here (plan section 2).")
        Path(a.out).write_text("\n".join(L) + "\n", encoding="utf-8")
        print("\n".join(L))
        return
    w(f"fills at the recorded touch (sweep-aware: records 1 us apart are one event), fees {FEE} ticks per round trip, latency {LAT * 1000:.0f} ms; stress net - {STRESS}; "
      f"SE the largest moving-block bootstrap SE over blocks of {', '.join(map(str, BLOCKS))} scored days ({NBOOT} resamples, seed {SEED})")
    w(f"primary M = {M_PRI}, t_req {TREQ_PRI:.2f}; secondary M = {M_SEC}, t_req {TREQ_SEC:.2f}")
    w("")
    w(f"  {'rule':<22}{'trades':>8}{'skip':>6}{'win':>6}{'gross':>8}{'net':>8}{'se':>6}{'t':>7}{'half1':>8}{'half2':>8}{'long/short net':>17}"
      f"{'vs ctrl':>9}{'(se)':>6}{'vs shell':>9}{'(se)':>6}  edge")
    for p, kw in RULES:
        kk = key(p, kw)
        r = R[kk]
        pri = kw == PRIMARY[p]
        e_ = edge(r, TREQ_PRI) if pri else False
        c_ = edge(r, TREQ_SEC)
        tag = ("PRIMARY EDGE" if e_ else "") + (" candidate" if c_ else "") or "-"
        w(f"  {('*' if pri else ' ') + kk:<22}{r['trades']:>8}{r['skipped']:>6}{n_(r['win'], 2, False):>6}{n_(r['gross']):>8}{n_(r['net']):>8}"
          f"{n_(r['se'], 2, False):>6}{n_(r['t'], 1):>7}{n_(r['h1']):>8}{n_(r['h2']):>8}{n_(r['long_net']) + ' / ' + n_(r['short_net']):>17}"
          f"{n_(r['ctrl'][0]):>9}{n_(r['ctrl'][1], 2, False):>6}{n_(r['shell'][0]):>9}{n_(r['shell'][1], 2, False):>6}  {tag}")
    w("  (* = primary setting; 'vs ctrl' = gross minus the random control's gross per trade; 'vs shell' = gross minus the same rule")
    w("   without its footprint condition, per trade; both descriptive. The random control takes the signal's side on the same day")
    w("   and bucket, so for wide-stop rules (P3 N=20, P5) it mostly measures that day's trend: 'vs ctrl' there is not evidence that")
    w("   the rule adds anything; 'vs shell' is the comparison that isolates the footprint condition)")
    w("")
    w("touch fills (targets filled on touch, an optimistic bound): " + "; ".join(
        f"{key(p, kw)} net {n_(R[key(p, kw)]['touch'])}" for p, kw in RULES if p in ("P2", "P4")))
    w("power (net per trade a rule would need): " + "; ".join(
        f"{key(p, kw)} {(TREQ_PRI if kw == PRIMARY[p] else TREQ_SEC) * R[key(p, kw)]['se']:.2f}" for p, kw in RULES
        if math.isfinite(R[key(p, kw)]["se"])))
    w("revisit rates (F4 benchmark; highs and lows whose bar closed at least d away; no SE): " + "; ".join(
        f"d={dd} {kind} {int(hz / 60)} min {h / cnt:.3f} ({cnt})" for (dd, kind, hz), (h, cnt) in sorted(rev.items()) if cnt))
    w("trades by 30-min bucket from 08:30 (primary rules): " + "; ".join(
        f"{p} " + "/".join(str(int(x)) for x in tod[key(p, PRIMARY[p])][:13]) for p in PRIMARY))
    flags = [kk for kk, r in R.items() if (math.isfinite(r["t"]) and r["t"] > 6) or (r["trades"] >= 50 and r["win"] > 0.75)
             or (r["short_hold"][1] > 0 and r["short_hold"][0] > 3)]
    w("trades held under a minute (gross per trade, count): " + "; ".join(
        f"{kk} {n_(r['short_hold'][0])} ({r['short_hold'][1]})" for kk, r in R.items() if r["short_hold"][1]))
    w(f"red flags (t > 6, a win rate > 0.75 on 50+ trades, or gross > 3 ticks on trades held under a minute): {len(flags)} "
      f"{flags if flags else ''}")
    w("")
    pe = [key(p, kw) for p, kw in RULES if kw == PRIMARY[p] and edge(R[key(p, kw)], TREQ_PRI)]
    sc = [key(p, kw) for p, kw in RULES if edge(R[key(p, kw)], TREQ_SEC)]
    if pe or sc:
        w(f"DECISION (plan section 5): primary edges {pe}; secondary candidates {sc}. Not a claim: OF19b, one confirmation on the next "
          "20 front-month RTH days from 2026-11-12.")
    else:
        w("DECISION (plan section 5): no primary edge and no secondary candidate. No footprint pattern at these settings nets more")
        w("than its power bound above (X = t_req x SE per rule); smaller edges are not excluded.")
    txt = "\n".join(L) + "\n"
    Path(a.out).write_text(txt, encoding="utf-8")
    print(txt)


if __name__ == "__main__":
    main()
