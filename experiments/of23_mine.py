#!/usr/bin/env python3
"""OF23: mining turning-point entries with wide exits on real ES bars (docs/ai/OF23_SWING_MINING_PLAN.md).

  python experiments/of23_mine.py --check                 candidate counts per family and month (no labels, no P&L)
  python experiments/of23_mine.py --run [--jobs 6]        prepare + every worker + report -> experiments/of23_results.txt
  python experiments/of23_mine.py --prepare | --report    the two halves of --run (workers are spawned by --run)

Reads the OF18 1-min bar cache (2025-12-15 .. 2026-11-11, sha256-checked against the sidecars) and, for the order
of target and stop inside a bar, the front contract's .scid records of each day (1-s bundles through 2026-09-05,
tick records from 2026-09-08). Nothing on or after 2026-11-12 is read. Work files (labels, features, worker
outputs) go to build-avx512/of23/, never into the repository. Every constant is the plan's.
"""
import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from datetime import date
from pathlib import Path
from statistics import NormalDist

import numpy as np

REPO = Path(__file__).resolve().parents[1]
for p in (REPO / "python", REPO / "experiments", REPO / "tools"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
import ragrl  # noqa: E402,F401  (the WMI workaround, before lightgbm or torch)
from ragrl import real  # noqa: E402

WORK = REPO / "build-avx512" / "of23"
BARS = REPO / "build-avx512" / "real" / "bars"
SCID_DIR = Path("C:/SierraChart/Data")
OUT = REPO / "experiments" / "of23_results.txt"
START, END = "2025-12-15", "2026-11-11"
TICK_FROM = "2026-09-08"
MAX_GAP_MIN = 5
N_MIN = 390
FIRST_M, LAST_M = 20, 359          # decided at the close of bar m (08:50 .. 14:30), entered at open(m + 1)
GAP_WIN = 30

TS = (8, 12, 16, 24, 32, 48)
SS = (4, 6, 8, 12, 16)
HS = (30, 60, 120, None)           # None = end of day
TRS = (False, True)
EXITS = [(T, S, H, tr) for T in TS for S in SS for H in HS for tr in TRS]
NE = len(EXITS)
MENU = [(8, 4, 30), (12, 6, 60), (16, 8, 60), (24, 8, 120), (32, 12, None), (48, 16, None)]
MENU_E = [EXITS.index((T, S, H, tr)) for (T, S, H) in MENU for tr in TRS]

C_MKT, C_LIM = 0.64, 0.12          # per side, ticks
C_MKT_X, C_LIM_X, STOP_X = 0.75, 0.20, 1.0   # stress, plus one more tick on stop fills
K_TARGET, K_STOP, K_TIME = 0, 1, 2

QS = (0.10, 0.25, 0.50, 1.00)
FLOOR = 0.5
TEST_MONTHS = [f"2026-{m:02d}" for m in range(3, 12)]
BLOCKS, B_BOOT, SEED = (5, 10, 21), 10_000, 0
N_RAND = 200
FAMILIES = ("C1", "C2", "C3", "C4", "C5", "C6")
PIPES = ["pooled"] + list(FAMILIES)
M_SEC = 15
T_REQ = 2.0 + math.sqrt(2.0 * math.log(M_SEC))
RESOLUTION_MAX = 0.25
LGB = dict(objective="l2", num_leaves=15, min_data_in_leaf=100, learning_rate=0.03, feature_fraction=0.7,
           bagging_fraction=0.7, bagging_freq=1, lambda_l2=10.0, seed=0, deterministic=True, force_row_wise=True,
           num_threads=4, verbose=-1)
ROUNDS, CLIP = 200, 60.0
RL = dict(iters=300, hidden=64, gamma=1.0, lam=0.95, clip=0.2, epochs=4, minibatch=512, lr=3e-4, ent=0.01,
          seeds=(0, 1, 2, 3, 4), reward_scale=10.0)
LEVELS = ("pdh", "pdl", "pdc", "ppoc", "pvah", "pval")
FEATS = ["c1_1", "c1_5", "c1_15", "c2", "c3", "c4", "c5", "c6", "c3_count", "c3_name", "dir",
         "d_ext_entry", "d_ext_far", "d_open", "d_vwap", "vwap_z", "atr", "range_atr", "close_loc", "relvol",
         "delta1", "delta5", "delta15", "delta30", "ret5", "ret15", "ret60", "ret_open",
         "slope5", "slope15", "slope60", "d_level", "fvg5_n", "fvg15_n", "d_fvg", "minute", "dow"]
FAM_COLS = {"C1": ("c1_1", "c1_5", "c1_15"), "C2": ("c2",), "C3": ("c3",), "C4": ("c4",), "C5": ("c5",), "C6": ("c6",)}


def sha256_lf(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


# ================================================================ data
def load_days():
    """[(day, sidecar, bars)] of the used days (OF18's rule), the dropped days and the manifest sha256."""
    m = real.load_manifest()
    cutoff = date.fromisoformat(m["confirm_rule"]["from"])
    if date.fromisoformat(END) >= cutoff:
        raise RuntimeError("the span reaches the confirm block")
    days, dropped = [], []
    for side_p in sorted(BARS.glob("*.json")):
        d = side_p.stem
        if not (START <= d <= END):
            continue
        if date.fromisoformat(d) >= cutoff:
            raise RuntimeError(f"{d}: the confirm block is never read")
        side = json.loads(side_p.read_text(encoding="utf-8"))
        npz = BARS / f"{d}.npz"
        if hashlib.sha256(npz.read_bytes()).hexdigest() != side["npz_sha256"]:
            raise RuntimeError(f"{npz}: sha256 does not match its sidecar")
        if side["gap_minutes"] > MAX_GAP_MIN:
            dropped.append(d)
            continue
        z = np.load(npz)
        days.append((d, side, {k: z[k].astype(np.float64) for k in z.files}))
    return days, dropped, real.manifest_sha256()


def check_day(day: str):
    if day >= real.load_manifest()["confirm_rule"]["from"]:
        raise RuntimeError(f"{day}: the confirm block is never read")


def records_of(day: str, contract: str):
    """The day's RTH records of one contract and the open in Sierra microseconds."""
    from scid_to_bars import Scid
    check_day(day)
    o_unix, c_unix = real.rth_span(day)
    t0, t1 = real.sierra_us(o_unix), real.sierra_us(c_unix)
    s = Scid(SCID_DIR / f"{contract}_FUT_CME.scid")
    try:
        r = s.records(t0, t1)
    finally:
        s.close()
    return r, t0


def path_from_records(r, t0: int, style: str, mode: str) -> dict:
    """Steps (o, h, l, minute) in time order: mode 'sec' = one step per second, 'tick' = one per record."""
    tk = lambda x: np.rint(x.astype(np.float64) / 100.0 / 0.25)  # noqa: E731
    c = tk(r["c"])
    if style == "tick":
        o = h = l = c
    else:
        o, h, l = tk(r["o"]), tk(r["h"]), tk(r["l"])
    us = r["t"].astype(np.int64) - t0
    if mode == "tick":
        return {"o": o.copy(), "h": h.copy(), "l": l.copy(), "minute": (us // 60_000_000).astype(np.int64)}
    sec = us // 1_000_000
    if len(sec) == 0:
        return {k: np.zeros(0) for k in ("o", "h", "l")} | {"minute": np.zeros(0, np.int64)}
    st = np.flatnonzero(np.r_[True, sec[1:] != sec[:-1]])
    return {"o": o[st], "h": np.maximum.reduceat(h, st), "l": np.minimum.reduceat(l, st),
            "minute": (sec[st] // 60).astype(np.int64)}


def path_from_bars(b: dict) -> dict:
    live = b["volume"] > 0
    i = np.flatnonzero(live)
    return {"o": b["open"][i], "h": b["high"][i], "l": b["low"][i], "minute": i.astype(np.int64)}


# ================================================================ candidates and features
def tf_bars(b: dict, tf: int) -> dict:
    """Completed tf-minute bars from the n 1-min bars available (bar k covers minutes k*tf .. k*tf+tf-1)."""
    n = len(b["open"]) // tf
    if n == 0:
        z = np.zeros(0)
        return {"open": z, "high": z, "low": z, "close": z}
    sl = slice(0, n * tf)
    r = lambda a: a[sl].reshape(n, tf)  # noqa: E731
    return {"open": r(b["open"])[:, 0], "high": r(b["high"]).max(1), "low": r(b["low"]).min(1), "close": r(b["close"])[:, -1]}


def atr_of(b: dict) -> np.ndarray:
    h, l, c = b["high"], b["low"], b["close"]
    tr = h - l
    if len(c) > 1:
        pc = c[:-1]
        tr[1:] = np.maximum(h[1:], pc) - np.minimum(l[1:], pc)
    a = np.full(len(c), np.nan)
    cs = np.r_[0.0, np.cumsum(tr)]
    for m in range(19, len(c)):
        a[m] = (cs[m + 1] - cs[m - 19]) / 20.0
    return a


def levels_for(prev, day: str, side: dict, dropped) -> dict | None:
    """The prior session's levels: only from the same contract and only when no session between was dropped."""
    if prev is None or prev[1]["contract"] != side["contract"] or any(prev[0] < d < day for d in dropped):
        return None
    return prior_levels(prev[2])


def prior_levels(prev: dict) -> dict:
    """pdh, pdl, pdc and the prior day's POC / value area (bar volume spread evenly over each bar's range)."""
    from of20_levels import value_area
    px, sz = [], []
    for lo, hi, v in zip(prev["low"], prev["high"], prev["volume"]):
        if v <= 0:
            continue
        t = np.arange(int(lo), int(hi) + 1)
        px.append(t)
        sz.append(np.full(len(t), v / len(t)))
    poc, vah, val = value_area(np.concatenate(px), np.concatenate(sz))
    return {"pdh": float(prev["high"].max()), "pdl": float(prev["low"].min()), "pdc": float(prev["close"][-1]),
            "ppoc": float(poc), "pvah": float(vah), "pval": float(val)}


def swings(tb: dict, tf: int, n1: int) -> dict:
    """{1-min bar m: [(dir, ...)]}: 5-bar swings of tf bars confirmed at the close of tf bar k+2."""
    out = {}
    lo, hi = tb["low"], tb["high"]
    for k in range(2, len(lo) - 2):
        m = (k + 3) * tf - 1
        if m >= n1:
            continue
        nb = [j for j in range(k - 2, k + 3) if j != k]
        if all(lo[k] < lo[j] for j in nb):
            out.setdefault(m, []).append(1)
        if all(hi[k] > hi[j] for j in nb):
            out.setdefault(m, []).append(-1)
    return out


def fvgs(b: dict, tb: dict, tf: int) -> list:
    """Fair value gaps of tf bars: [(dir, bottom, top, open_from_m, closed_at_m or None, retest m or None, retest ok)]."""
    n1 = len(b["open"])
    out = []
    H, L, C = tb["high"], tb["low"], tb["close"]
    for i in range(2, len(H)):
        for d in (1, -1):
            if d == 1 and L[i] > H[i - 2]:
                bot, top = H[i - 2], L[i]
            elif d == -1 and H[i] < L[i - 2]:
                bot, top = H[i], L[i - 2]
            else:
                continue
            m0 = (i + 1) * tf - 1          # the gap exists from the close of bar i
            closed = None
            for k in range(i + 1, len(C)):  # invalidated by a close of its own timeframe beyond the far edge
                if (d == 1 and C[k] < bot) or (d == -1 and C[k] > top):
                    closed = (k + 1) * tf - 1
                    break
            rm, ok = None, False
            for m in range(m0 + 1, n1):
                if closed is not None and m > closed:
                    break
                if b["low"][m] <= top and b["high"][m] >= bot:
                    rm = m
                    ok = not ((d == 1 and b["close"][m] < bot) or (d == -1 and b["close"][m] > top))
                    break
            out.append((d, bot, top, m0, closed, rm, ok))
    return out


def ema_slope(tb: dict, m: int, tf: int, atr: float) -> float:
    k = (m + 1) // tf          # completed tf bars by the close of m
    if k < 2 or not atr > 0:
        return float("nan")
    c = tb["close"][:k]
    a, e, prev = 2.0 / 9.0, c[0], c[0]
    for x in c[1:]:
        prev, e = e, e + a * (x - e)
    return float((e - prev) / atr)


def candidates(b: dict, prev_lv, dow: int) -> list:
    """Candidates (merged per bar and direction) with their features, from the n bars given (n <= 390)."""
    n = len(b["open"])
    o, h, l, c, v = b["open"], b["high"], b["low"], b["close"], b["volume"]
    atr = atr_of(b)
    tp = (h + l + c) / 3.0
    cv, cvp, cvp2 = np.cumsum(v), np.cumsum(v * tp), np.cumsum(v * tp * tp)
    tb = {tf: tf_bars(b, tf) for tf in (5, 15, 60)}
    sw = {1: swings(b, 1, n), 5: swings(tb[5], 5, n), 15: swings(tb[15], 15, n)}
    gaps = {tf: fvgs(b, tb[tf], tf) for tf in (5, 15)}
    retest = {}
    for tf in (5, 15):
        for (d, bot, top, m0, closed, rm, ok) in gaps[tf]:
            if rm is not None and ok:
                retest.setdefault(rm, set()).add(d)
    rows = {}
    for m in range(FIRST_M, min(LAST_M, n - 1) + 1):
        if not atr[m] > 0 or np.any(v[max(0, m - GAP_WIN + 1):m + 1] <= 0):
            continue
        fl = {}  # (dir) -> {flag: value}

        def add(d, k, val=1):
            fl.setdefault(d, {})
            fl[d][k] = fl[d].get(k, 0) + val
        for tf, key in ((1, "c1_1"), (5, "c1_5"), (15, "c1_15")):
            for d in sw[tf].get(m, ()):
                add(d, key)
        if m >= 30:
            ph, pl = h[:m].max(), l[:m].min()
            if h[m] > ph and c[m] < ph:
                add(-1, "c2")
            if l[m] < pl and c[m] > pl:
                add(1, "c2")
        lv_fired = {}
        if prev_lv is not None and m >= 1:
            for li, name in enumerate(LEVELS):
                L = prev_lv[name]
                if c[m - 1] < L and h[m] >= L + 1 and c[m] < L:
                    lv_fired.setdefault(-1, []).append(li)
                elif c[m - 1] > L and l[m] <= L - 1 and c[m] > L:
                    lv_fired.setdefault(1, []).append(li)
            for d, lis in lv_fired.items():
                add(d, "c3")
        vw = cvp[m] / cv[m] if cv[m] > 0 else float("nan")
        sd = math.sqrt(max(cvp2[m] / cv[m] - vw * vw, 0.0)) if cv[m] > 0 else float("nan")
        if m >= 60 and sd > 0:
            if h[m] >= vw + 2 * sd and c[m] < vw + 2 * sd:
                add(-1, "c4")
            if l[m] <= vw - 2 * sd and c[m] > vw - 2 * sd:
                add(1, "c4")
        for d in retest.get(m, ()):
            add(d, "c5")
        rng = h[m] - l[m]
        if m >= 30 and rng >= 2 * atr[m] and v[m] >= np.quantile(v[:m], 0.9):
            if h[m] > h[m - 29:m].max() and c[m] <= l[m] + rng / 3.0:
                add(-1, "c6")
            if l[m] < l[m - 29:m].min() and c[m] >= h[m] - rng / 3.0:
                add(1, "c6")
        if len(fl) != 1:
            continue  # none, or both directions on one bar: dropped
        d, f = next(iter(fl.items()))
        A = atr[m]
        sh, sl_ = h[:m + 1].max(), l[:m + 1].min()
        lis = lv_fired.get(d, [])
        if lis:
            near = min(lis, key=lambda li: (abs(c[m] - prev_lv[LEVELS[li]]), li))
        r = {k: float(f.get(k, 0)) for k in FEATS[:8]}
        r["c3_count"] = float(len(lis))
        r["c3_name"] = float(near) if lis else -1.0
        r["dir"] = float(d)
        r["d_ext_entry"] = ((sh - c[m]) if d == -1 else (c[m] - sl_)) / A
        r["d_ext_far"] = ((c[m] - sl_) if d == -1 else (sh - c[m])) / A
        r["d_open"] = (c[m] - o[0]) / A
        r["d_vwap"] = (c[m] - vw) / A
        r["vwap_z"] = d * (c[m] - vw) / sd if sd > 0 else float("nan")
        r["atr"] = A
        r["range_atr"] = rng / A
        r["close_loc"] = (c[m] - l[m]) / rng if rng > 0 else float("nan")
        pv = v[max(0, m - 30):m]
        r["relvol"] = v[m] / pv.mean() if len(pv) and pv.mean() > 0 else float("nan")
        for k in (1, 5, 15, 30):
            s0 = max(0, m - k + 1)
            vv = v[s0:m + 1].sum()
            r[f"delta{k}"] = d * (b["buy"][s0:m + 1].sum() - b["sell"][s0:m + 1].sum()) / vv if vv > 0 else float("nan")
        for k in (5, 15, 60):
            ref = c[m - k] if m - k >= 0 else o[0]
            r[f"ret{k}"] = d * (c[m] - ref) / A
        r["ret_open"] = d * (c[m] - o[0]) / A
        for tf in (5, 15, 60):
            r[f"slope{tf}"] = d * ema_slope(tb[tf], m, tf, A)
        r["d_level"] = min(abs(c[m] - prev_lv[k]) for k in LEVELS) / A if prev_lv is not None else float("nan")
        dmin = float("nan")
        for tf in (5, 15):
            cnt = 0
            for (gd, bot, top, m0, closed, rm, ok) in gaps[tf]:
                if m0 <= m and (closed is None or closed > m):
                    cnt += 1
                    dist = 0.0 if bot <= c[m] <= top else min(abs(c[m] - bot), abs(c[m] - top)) / A
                    dmin = dist if not (dmin <= dist) else dmin
            r[f"fvg{tf}_n"] = float(cnt)
        r["d_fvg"] = dmin
        r["minute"] = float(m)
        r["dow"] = float(dow)
        r["m"] = m
        rows[m] = r
    return [rows[k] for k in sorted(rows)]


# ================================================================ labels (exits, fills, costs)
def exit_minutes(b: dict, j: int) -> tuple:
    """(gap exit minute or 390) for a trade entered at open(j): the first bar after a gap bar that trades."""
    v = b["volume"]
    g = np.flatnonzero(v[j:] <= 0)
    if len(g) == 0:
        return N_MIN
    g0 = j + int(g[0])
    after = np.flatnonzero(v[g0 + 1:] > 0)
    return g0 + 1 + int(after[0]) if len(after) else N_MIN


def labels(path: dict, b: dict, m: int, d: int, exits=None) -> dict:
    """Per exit: move (ticks before costs, stop slippage included), kind, exit minute (flat from), both (a step
    reached target and stop: stop first)."""
    exits = EXITS if exits is None else exits
    j = m + 1
    P0 = b["open"][j]
    i0 = int(np.searchsorted(path["minute"], j))
    o, hh, ll, mn = path["o"][i0:], path["h"][i0:], path["l"][i0:], path["minute"][i0:]
    if d > 0:
        fav, adv, oadv = hh - P0, P0 - ll, P0 - o
    else:
        fav, adv, oadv = P0 - ll, hh - P0, o - P0
    gx = exit_minutes(b, j)
    cf = np.maximum.accumulate(fav) if len(fav) else fav
    ca = np.maximum.accumulate(adv) if len(adv) else adv
    closes = d * (b["close"] - P0)
    trail = {}
    for S in {e[1] for e in exits if e[3]}:
        bfc = np.full(N_MIN + 1, -np.inf)
        if j + 1 <= N_MIN:
            bfc[j + 1:N_MIN + 1] = np.maximum.accumulate(closes[j:N_MIN])
        thr_min = np.where(bfc >= S, S - bfc, float(S))
        thr = thr_min[np.minimum(mn, N_MIN)]
        hit = np.flatnonzero(adv >= thr)
        trail[S] = (int(hit[0]) if len(hit) else len(adv), thr)
    out = {"move": np.zeros(len(exits)), "kind": np.zeros(len(exits), np.int8), "xmin": np.zeros(len(exits), np.int16),
           "both": np.zeros(len(exits), bool)}
    for k, (T, S, H, tr) in enumerate(exits):
        tm = min(gx, N_MIN if H is None else j + H)
        tm = min(tm, N_MIN)
        while tm < N_MIN and b["volume"][tm] <= 0:
            tm += 1
        n_el = int(np.searchsorted(mn, tm))
        iT = int(np.searchsorted(cf, T + 1)) if len(cf) else 0
        if tr:
            iS, thr = trail[S]
        else:
            iS, thr = (int(np.searchsorted(ca, S)) if len(ca) else 0), None
        if iS < n_el and iS <= iT:
            th = S if thr is None else thr[iS]
            out["move"][k], out["kind"][k], out["xmin"][k] = -max(th, oadv[iS]) - 1.0, K_STOP, mn[iS]
            out["both"][k] = iS == iT
        elif iT < n_el:
            out["move"][k], out["kind"][k], out["xmin"][k] = float(T), K_TARGET, mn[iT]
        else:
            px = b["close"][N_MIN - 1] if tm >= N_MIN else b["open"][tm]
            out["move"][k], out["kind"][k], out["xmin"][k] = d * (px - P0), K_TIME, tm
    return out


def net_of(move, kind, stress=False):
    tgt = kind == K_TARGET
    if stress:
        return move - np.where(kind == K_STOP, STOP_X, 0.0) - C_MKT_X - np.where(tgt, C_LIM_X, C_MKT_X)
    return move - C_MKT - np.where(tgt, C_LIM, C_MKT)


# ================================================================ prepare
def prepare(log=print):
    WORK.mkdir(parents=True, exist_ok=True)
    days, dropped, msha = load_days()
    rows, lab, lab_bar, tickchk, rand = [], [], [], {}, {}
    test_days = {d for d, _, _ in days if d[:7] in TEST_MONTHS}
    prev = None
    t0 = time.time()
    for di, (day, side, b) in enumerate(days):
        dow = date.fromisoformat(day).weekday()
        cs = [c for c in candidates(b, levels_for(prev, day, side, dropped), dow) if b["volume"][c["m"] + 1] > 0]
        r, ts = records_of(day, side["contract"])
        style = side["style"]
        P = path_from_records(r, ts, style, "tick" if style == "tick" else "sec")
        Pb = path_from_bars(b)
        Ps = path_from_records(r, ts, style, "sec") if style == "tick" else None
        for cnd in cs:
            cnd["day"] = di
            L = labels(P, b, cnd["m"], int(cnd["dir"]))
            lab.append(L)
            lab_bar.append(labels(Pb, b, cnd["m"], int(cnd["dir"])))
            if Ps is not None:
                Ls = labels(Ps, b, cnd["m"], int(cnd["dir"]))
                tickchk.setdefault("sec", []).append(Ls["move"])
                tickchk.setdefault("kind_sec", []).append(Ls["kind"])
                tickchk.setdefault("row", []).append(len(rows))
            rows.append(cnd)
        if day in test_days:  # labels at every bar and direction: the random-time control
            mv = np.zeros((LAST_M - FIRST_M + 1, 2, NE), np.float32)
            kd = np.zeros(mv.shape, np.int8)
            xm = np.zeros(mv.shape, np.int16)
            for mi, m in enumerate(range(FIRST_M, LAST_M + 1)):
                for si, d in enumerate((1, -1)):
                    L = labels(P, b, m, d)
                    mv[mi, si], kd[mi, si], xm[mi, si] = L["move"], L["kind"], L["xmin"]
            ok = np.array([b["volume"][m + 1] > 0 for m in range(FIRST_M, LAST_M + 1)])
            kd[~ok] = -1  # no entry into a gap bar
            rand[day] = (mv, kd, xm)
        prev = (day, side, b)
        if di % 20 == 0:
            log(f"prepare: {day} ({di + 1}/{len(days)}), {len(rows)} candidates, {time.time() - t0:.0f} s")
    X = np.array([[c[f] for f in FEATS] for c in rows], np.float64)
    meta = {"days": [d for d, _, _ in days], "contracts": [s["contract"] for _, s, _ in days],
            "styles": [s["style"] for _, s, _ in days], "dropped": dropped, "manifest_sha256": msha,
            "runner_sha256": sha256_lf(Path(__file__)), "n": len(rows)}
    st = lambda L, k: np.stack([x[k] for x in L]) if L else np.zeros((0, NE))  # noqa: E731
    np.savez(WORK / "cache.npz", X=X, day=np.array([c["day"] for c in rows]), m=np.array([c["m"] for c in rows]),
             move=st(lab, "move").astype(np.float32), kind=st(lab, "kind"), xmin=st(lab, "xmin"), both=st(lab, "both"),
             move_bar=st(lab_bar, "move").astype(np.float32), kind_bar=st(lab_bar, "kind"), both_bar=st(lab_bar, "both"),
             chk_row=np.array(tickchk.get("row", []), np.int64),
             chk_move_sec=np.array(tickchk.get("sec", np.zeros((0, NE))), np.float32),
             chk_kind_sec=np.array(tickchk.get("kind_sec", np.zeros((0, NE))), np.int8))
    np.savez(WORK / "rand.npz", **{f"{d}__{k}": a for d, (mv, kd, xm) in rand.items() for k, a in (("move", mv), ("kind", kd), ("xmin", xm))})
    (WORK / "meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    log(f"prepare: {len(rows)} candidates on {len(days)} days in {time.time() - t0:.0f} s")


def load_cache():
    z = np.load(WORK / "cache.npz")
    C = {k: z[k] for k in z.files}
    meta = json.loads((WORK / "meta.json").read_text(encoding="utf-8"))
    C["days"] = np.array(meta["days"])
    C["month"] = np.array([meta["days"][i][:7] for i in C["day"]])
    C["net"] = net_of(C["move"], C["kind"]).astype(np.float32)
    return C, meta


# ================================================================ the in-order walk
def walk(day, m, take, xmin, net):
    """One trade at a time, in time order per day. take (n, K) bool, xmin/net (n, K): per column the
    non-overlapping total net, trades and taken mask."""
    n, K = take.shape
    tot, cnt = np.zeros(K), np.zeros(K)
    taken = np.zeros((n, K), bool)
    flat = np.zeros(K)
    last = None
    for i in np.lexsort((m, day)):
        if day[i] != last:
            flat[:] = -1
            last = day[i]
        t = take[i] & (m[i] >= flat)
        if t.any():
            tot += np.where(t, net[i], 0.0)
            cnt += t
            flat = np.where(t, xmin[i], flat)
            taken[i] = t
    return tot, cnt, taken


def family_mask(C, pipe):
    if pipe == "pooled":
        return np.ones(len(C["day"]), bool)
    cols = [FEATS.index(k) for k in FAM_COLS[pipe]]
    return C["X"][:, cols].sum(1) > 0


def inner_blocks(months):
    return np.array(["2026-01" if mo == "2025-12" else mo for mo in months])


def choose(tot, cnt, ndays, keys):
    """(index) of the largest net per day among those with >= FLOOR trades per day, else the most trades;
    ties by the keys' order (keys sorted already by the tie rule)."""
    per, rate = np.nan_to_num(tot / ndays), np.nan_to_num(cnt / ndays)
    ok = rate >= FLOOR
    if ok.any():
        best = per[ok].max()
        cands = [k for k in keys if ok[k] and per[k] == best]
    else:
        best = rate.max()
        cands = [k for k in keys if rate[k] == best]
    return cands[0]


def tie_order(with_q: bool):
    """Combination indices sorted by the tie rule: smaller q, smaller T, S, shorter H, trailing off."""
    hk = lambda H: 10_000 if H is None else H  # noqa: E731
    combos = [(qi, ei) for qi in (range(len(QS)) if with_q else [len(QS) - 1]) for ei in range(NE)]
    combos.sort(key=lambda x: (QS[x[0]], EXITS[x[1]][0], EXITS[x[1]][1], hk(EXITS[x[1]][2]), EXITS[x[1]][3]))
    return combos


def fit_predict(Xtr, ytr, Xte_list):
    import lightgbm as lgb
    if len(ytr) < 40:
        return [np.full(len(x), float(np.mean(np.clip(ytr, -CLIP, CLIP))) if len(ytr) else 0.0) for x in Xte_list]
    p = dict(LGB)
    if len(ytr) < 2000:
        p["min_data_in_leaf"] = max(1, len(ytr) // 20)
    bst = lgb.train(p, lgb.Dataset(Xtr, np.clip(ytr, -CLIP, CLIP), free_raw_data=True), num_boost_round=ROUNDS)
    return [bst.predict(x, num_threads=LGB["num_threads"]) for x in Xte_list]


def gbm_fold(C, pipe, train, test, ndays_tr, with_model=True):
    """Steps 1-4 of plan section 6 for one fold (ndays_tr = every used training day, whether or not the pipeline
    has a candidate on it). Returns the choice and the test trades."""
    tr, te = np.flatnonzero(train), np.flatnonzero(test)
    combos = tie_order(with_model)
    if len(tr) == 0:
        qi, e = combos[0]
        return {"pipe": pipe, "model": with_model, "e": int(e), "q": QS[qi], "train_net_day": 0.0, "train_trades_day": 0.0,
                "train_net_trade": float("nan"), "rows": [], "exits": []}
    if with_model:
        blk = inner_blocks(C["month"][tr])
        oof = np.zeros((len(tr), NE))
        tpred = np.zeros((len(te), NE))
        ub = sorted(set(blk))
        for e in range(NE):
            for bk in ub:
                fit = blk != bk
                po, pt = fit_predict(C["X"][tr[fit]], C["net"][tr[fit], e], [C["X"][tr[~fit]], C["X"][te]])
                oof[~fit, e] = po
                tpred[:, e] += pt / len(ub)
        cuts = np.array([[np.quantile(oof[:, e], 1 - q) if q < 1 else -np.inf for e in range(NE)] for q in QS])
        take = np.stack([oof[:, e] > cuts[qi, e] for qi, e in combos], 1)
    else:
        oof = tpred = cuts = None
        take = np.ones((len(tr), len(combos)), bool)
    E = [e for _, e in combos]
    tot, cnt, _ = walk(C["day"][tr], C["m"][tr], take, C["xmin"][tr][:, E], C["net"][tr][:, E])
    k = choose(tot, cnt, ndays_tr, list(range(len(combos))))
    qi, e = combos[k]
    if with_model:
        tk = tpred[:, e] > cuts[qi, e]
    else:
        tk = np.ones(len(te), bool)
    _, _, taken = walk(C["day"][te], C["m"][te], tk[:, None], C["xmin"][te][:, [e]], C["net"][te][:, [e]])
    rows = te[taken[:, 0]]
    return {"pipe": pipe, "model": with_model, "e": int(e), "q": QS[qi], "train_net_day": float(tot[k] / ndays_tr),
            "train_trades_day": float(cnt[k] / ndays_tr), "train_net_trade": float(tot[k] / cnt[k]) if cnt[k] else float("nan"),
            "rows": rows.tolist(), "exits": [int(e)] * len(rows)}


def train_days(C, mo):
    """Every used day before the test month (from the cache's day list, candidates or not)."""
    return int(sum(1 for d in C["days"] if d[:7] < mo))


def folds(C):
    for f, mo in enumerate(TEST_MONTHS):
        yield f, mo, C["month"] < mo, C["month"] == mo


# ================================================================ RL arms (plan section 6b)
def rl_train_test(C, arm, train, test, seed):
    import torch
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    tr, te = np.flatnonzero(train), np.flatnonzero(test)
    X = C["X"]
    mu, sd = np.nanmean(X[tr], 0), np.nanstd(X[tr], 0)
    sd[~(sd > 0)] = 1.0
    Z = np.nan_to_num((X - mu) / sd).astype(np.float32)
    nA = len(MENU_E) + (1 if arm == "free" else 0)
    off = 1 if arm == "free" else 0
    F = Z.shape[1]
    pol = torch.nn.Sequential(torch.nn.Linear(F, RL["hidden"]), torch.nn.Tanh(), torch.nn.Linear(RL["hidden"], RL["hidden"]),
                              torch.nn.Tanh(), torch.nn.Linear(RL["hidden"], nA))
    val = torch.nn.Sequential(torch.nn.Linear(F, RL["hidden"]), torch.nn.Tanh(), torch.nn.Linear(RL["hidden"], RL["hidden"]),
                              torch.nn.Tanh(), torch.nn.Linear(RL["hidden"], 1))
    opt = torch.optim.Adam(list(pol.parameters()) + list(val.parameters()), lr=RL["lr"])
    # episodes: each training day's candidates in time order; next index after taking exit a
    order = tr[np.lexsort((C["m"][tr], C["day"][tr]))]
    dstart = {}
    for pos, i in enumerate(order):
        dstart.setdefault(C["day"][i], [pos, pos])[1] = pos + 1
    dm = C["m"][order]
    nxt = np.zeros((len(order), nA), np.int64)
    rew = np.zeros((len(order), nA), np.float32)
    for dy, (a0, a1) in dstart.items():
        for pos in range(a0, a1):
            i = order[pos]
            if off:
                nxt[pos, 0], rew[pos, 0] = pos + 1, 0.0
            for a, e in enumerate(MENU_E):
                x = C["xmin"][i, e]
                nxt[pos, a + off] = a0 + int(np.searchsorted(dm[a0:a1], x, side="left"))
                rew[pos, a + off] = C["net"][i, e] / RL["reward_scale"]
    ends = np.zeros(len(order), np.int64)
    for dy, (a0, a1) in dstart.items():
        ends[a0:a1] = a1
    starts = np.array([v[0] for v in dstart.values()])
    Zo = torch.from_numpy(Z[order])
    ne = len(starts)
    for it in range(RL["iters"]):
        ptr = starts.copy()
        live = np.ones(ne, bool)
        steps = []  # per step: (episode ids, positions, actions, logp, value, reward)
        while live.any():
            ep = np.flatnonzero(live)
            pos = ptr[ep]
            obs = Zo[pos]
            with torch.no_grad():
                dist = torch.distributions.Categorical(logits=pol(obs))
                a = dist.sample()
                lp = dist.log_prob(a).numpy()
                v = val(obs)[:, 0].numpy()
            a = a.numpy()
            steps.append((ep, pos, a, lp, v, rew[pos, a]))
            ptr[ep] = nxt[pos, a]
            live[ep] = ptr[ep] < ends[starts[ep]]
        gae, nv = np.zeros(ne), np.zeros(ne)
        adv_steps = []
        for ep, pos, a, lp, v, r in reversed(steps):  # GAE per episode, vectorized over episodes
            delta = r + RL["gamma"] * nv[ep] - v
            gae[ep] = delta + RL["gamma"] * RL["lam"] * gae[ep]
            nv[ep] = v
            adv_steps.append((pos, a, lp, gae[ep].copy(), gae[ep] + v))
        P = np.concatenate([x[0] for x in adv_steps])
        A = torch.from_numpy(np.concatenate([x[1] for x in adv_steps]).astype(np.int64))
        LP = torch.from_numpy(np.concatenate([x[2] for x in adv_steps]).astype(np.float32))
        ADV = torch.from_numpy(np.concatenate([x[3] for x in adv_steps]).astype(np.float32))
        RET = torch.from_numpy(np.concatenate([x[4] for x in adv_steps]).astype(np.float32))
        ADV = (ADV - ADV.mean()) / (ADV.std() + 1e-8)
        n = len(P)
        for _ in range(RL["epochs"]):
            perm = rng.permutation(n)
            for s0 in range(0, n, RL["minibatch"]):
                mb = perm[s0:s0 + RL["minibatch"]]
                obs = Zo[P[mb]]
                dist = torch.distributions.Categorical(logits=pol(obs))
                ratio = torch.exp(dist.log_prob(A[mb]) - LP[mb])
                s1, s2 = ratio * ADV[mb], torch.clamp(ratio, 1 - RL["clip"], 1 + RL["clip"]) * ADV[mb]
                loss = -torch.min(s1, s2).mean() + 0.5 * ((val(obs)[:, 0] - RET[mb]) ** 2).mean() - RL["ent"] * dist.entropy().mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
    with torch.no_grad():
        greedy = pol(torch.from_numpy(Z[te])).argmax(1).numpy()
    return te.tolist(), greedy.tolist()


def rl_walk(C, te, acts, off):
    """The in-order test walk of a greedy policy (actions per test candidate)."""
    te = np.asarray(te)
    acts = np.asarray(acts)
    take = acts >= off
    e = np.array([MENU_E[a - off] if a >= off else MENU_E[0] for a in acts])
    xm = C["xmin"][te, e][:, None]
    nt = C["net"][te, e][:, None]
    _, _, taken = walk(C["day"][te], C["m"][te], take[:, None], xm, nt)
    return te[taken[:, 0]].tolist(), e[taken[:, 0]].tolist()


# ================================================================ workers
def worker(kind, args):
    C, meta = load_cache()
    out = WORK / "jobs"
    out.mkdir(exist_ok=True)
    if kind == "gbm":
        pipe, f, model = args[0], int(args[1]), args[2] == "1"
        mo = TEST_MONTHS[f]
        fam = family_mask(C, pipe)
        res = gbm_fold(C, pipe, (C["month"] < mo) & fam, (C["month"] == mo) & fam, train_days(C, mo), model)
        name = f"gbm_{pipe}_{f}_{int(model)}"
    elif kind == "rl":
        arm, f, seed = args[0], int(args[1]), int(args[2])
        mo = TEST_MONTHS[f]
        te, acts = rl_train_test(C, arm, C["month"] < mo, C["month"] == mo, seed)
        res = {"te": te, "acts": acts}
        name = f"rl_{arm}_{f}_{seed}"
    elif kind == "naive":
        pipe = args[0]
        fam = family_mask(C, pipe)
        idx = np.flatnonzero(fam)
        blk = inner_blocks(C["month"][idx])
        oof = np.zeros((len(idx), NE))
        for e in range(NE):
            for bk in sorted(set(blk)):
                fit = blk != bk
                oof[~fit, e] = fit_predict(C["X"][idx[fit]], C["net"][idx[fit], e], [C["X"][idx[~fit]]])[0]
        series = trial_series(C, idx, oof)
        np.save(out / f"naive_{pipe}.npy", series)
        res = {"pipe": pipe}
        name = f"naive_{pipe}"
    else:
        raise SystemExit(f"unknown worker {kind}")
    (out / f"{name}.json").write_text(json.dumps(res), encoding="utf-8")


def trial_series(C, idx, oof):
    """Daily net (non-overlapping walk) per (q, e) trial on all days: (len(QS) * NE, n_days)."""
    nd = len(C["days"])
    rows = []
    for q in QS:
        cuts = np.array([np.quantile(oof[:, e], 1 - q) if q < 1 else -np.inf for e in range(NE)])
        take = oof > cuts[None, :]
        _, cnt, taken = walk(C["day"][idx], C["m"][idx], take, C["xmin"][idx], C["net"][idx])
        s = np.zeros((NE, nd + 1))
        for e in range(NE):
            np.add.at(s[e], C["day"][idx][taken[:, e]], C["net"][idx][taken[:, e], e])
        s[:, nd] = cnt
        rows.append(s)
    return np.concatenate(rows, 0)  # the last column: trades


def jobs_list():
    J = []
    for f in range(len(TEST_MONTHS)):
        for p in PIPES:
            J.append(("gbm", [p, str(f), "1"]))
        for arm in ("trade", "free"):
            for s in RL["seeds"]:
                J.append(("rl", [arm, str(f), str(s)]))
    for p in PIPES:
        J.append(("naive", [p]))
    return J


def run_jobs(n_par: int, log=print):
    py = sys.executable
    J = [j for j in jobs_list() if not (WORK / "jobs" / f"{job_name(j)}.json").exists()]
    log(f"jobs: {len(J)} to run, {n_par} at a time")
    running, t0 = [], time.time()
    env = dict(os.environ, PYTHONPATH=str(REPO / "python"))
    while J or running:
        while J and len(running) < n_par:
            k, a = J.pop(0)
            lp = WORK / "jobs" / f"{job_name((k, a))}.log"
            lp.parent.mkdir(parents=True, exist_ok=True)
            fh = open(lp, "w", encoding="utf-8")
            running.append((subprocess.Popen([py, str(Path(__file__)), "--worker", k, *a], stdout=fh, stderr=subprocess.STDOUT,
                                             env=env, cwd=str(REPO)), fh, (k, a)))
        time.sleep(1.0)
        for item in list(running):
            p, fh, j = item
            if p.poll() is not None:
                fh.close()
                running.remove(item)
                if p.returncode != 0 or not (WORK / "jobs" / f"{job_name(j)}.json").exists():
                    for q_, fh_, _ in running:
                        q_.kill()
                        fh_.close()
                    raise RuntimeError(f"worker {j} failed ({p.returncode}); see {WORK / 'jobs' / (job_name(j) + '.log')}")
                log(f"done {job_name(j)} ({time.time() - t0:.0f} s, {len(J)} queued, {len(running)} running)")


def job_name(j):
    k, a = j
    if k == "gbm":
        return f"gbm_{a[0]}_{a[1]}_{a[2]}"
    if k == "rl":
        return f"rl_{a[0]}_{a[1]}_{a[2]}"
    return f"naive_{a[0]}"


# ================================================================ statistics and report
def block_idx(n):
    return {b: of18_block(n, min(b, n), np.random.default_rng(SEED)) for b in BLOCKS}


def of18_block(n, b, rng):
    k = math.ceil(n / b)
    starts = rng.integers(0, n - b + 1, size=(B_BOOT, k))
    return (starts[:, :, None] + np.arange(b)[None, None, :]).reshape(B_BOOT, k * b)[:, :n]


def ratio_se(num, den, idx):
    ses = []
    for _, I in idx.items():
        dn = den[I].sum(1)
        r = np.where(dn > 0, num[I].sum(1) / np.maximum(dn, 1), np.nan)
        ses.append(float(np.nanstd(r, ddof=1)))
    return max(ses)


def summarize(C, rows, exits, test_days, idx):
    """Pooled statistics of a trade list (candidate rows, exit per trade) over the test days."""
    rows, exits = np.asarray(rows, np.int64), np.asarray(exits, np.int64)
    dpos = {d: k for k, d in enumerate(test_days)}
    nd = len(test_days)
    num, den, nst = np.zeros(nd), np.zeros(nd), np.zeros(nd)
    if len(rows):
        net = C["net"][rows, exits].astype(np.float64)
        stn = net_of(C["move"][rows, exits].astype(np.float64), C["kind"][rows, exits], stress=True)
        dp = np.array([dpos[d] for d in C["day"][rows]])
        np.add.at(num, dp, net)
        np.add.at(den, dp, 1)
        np.add.at(nst, dp, stn)
    n = den.sum()
    if n == 0:
        nan = float("nan")
        return {"trades": 0, "per_day": 0.0, "net": nan, "se": nan, "t": nan, "net_day": 0.0, "h1": nan, "h2": nan,
                "stress": nan, "no_best5": nan, "median": nan, "no_top1pct": nan, "win": nan, "avg_win": nan, "avg_loss": nan,
                "targets": nan, "stops": nan, "times": nan, "hold": nan, "long": nan, "short": nan, "by_family": {},
                "num": num, "den": den}
    mean = num.sum() / n
    se = ratio_se(num, den, idx)
    months = np.array([C["days"][d][:7] for d in test_days])
    h1 = months <= "2026-07"
    best5 = np.argsort(-num)[:5]
    keep = np.ones(nd, bool)
    keep[best5] = False
    net_all = C["net"][rows, exits].astype(np.float64)
    kinds = C["kind"][rows, exits]
    srt = np.sort(net_all)
    top = max(1, int(round(0.01 * len(srt))))
    return {"trades": int(n), "per_day": float(n / nd), "net": float(mean), "se": se, "t": float(mean / se) if se > 0 else float("nan"),
            "net_day": float(num.sum() / nd),
            "h1": float(num[h1].sum() / den[h1].sum()) if den[h1].sum() else float("nan"),
            "h2": float(num[~h1].sum() / den[~h1].sum()) if den[~h1].sum() else float("nan"),
            "stress": float(nst.sum() / n), "no_best5": float(num[keep].sum() / den[keep].sum()) if den[keep].sum() else float("nan"),
            "median": float(np.median(net_all)), "no_top1pct": float(srt[:-top].mean()) if len(srt) > top else float("nan"),
            "win": float((net_all > 0).mean()), "avg_win": float(net_all[net_all > 0].mean()) if (net_all > 0).any() else float("nan"),
            "avg_loss": float(net_all[net_all <= 0].mean()) if (net_all <= 0).any() else float("nan"),
            "targets": float((kinds == K_TARGET).mean()), "stops": float((kinds == K_STOP).mean()), "times": float((kinds == K_TIME).mean()),
            "hold": float((C["xmin"][rows, exits].astype(np.float64) - C["m"][rows] - 1).mean()),
            "long": side_mean(net_all, C["X"][rows, FEATS.index("dir")] > 0),
            "short": side_mean(net_all, C["X"][rows, FEATS.index("dir")] < 0),
            "by_family": {fam: (float(net_all[fm].mean()), int(fm.sum())) for fam, cols in FAM_COLS.items()
                          for fm in [C["X"][rows][:, [FEATS.index(k) for k in cols]].sum(1) > 0] if fm.any()},
            "num": num, "den": den}


def side_mean(x, mask):
    return float(x[mask].mean()) if mask.any() else float("nan")


def random_time(C, R, fold_trades, test_days, rep_seed):
    """One replicate of the random-time control: per day the same number of trades, sequential, entry bars from the
    empirical time-of-day distribution of that fold's entries, random direction with its long share, its exits."""
    rng = np.random.default_rng(rep_seed)
    tot = cnt = wins = 0.0
    for f, (rows, exits) in fold_trades.items():
        if not len(rows):
            continue
        rows, exits = np.asarray(rows), np.asarray(exits)
        ms = C["m"][rows]
        long_share = float((C["X"][rows, FEATS.index("dir")] > 0).mean())
        for d in np.unique(C["day"][rows]):
            mv, kd, xm = R[str(C["days"][d])]
            sel = C["day"][rows] == d
            k_n = int(sel.sum())
            ex_pool = exits[sel] if len(set(exits.tolist())) > 1 else exits[:1]
            flat = -1
            for _ in range(k_n):
                pool = ms[ms >= flat]
                if not len(pool):
                    break
                m = int(rng.choice(pool))
                si = 0 if rng.random() < long_share else 1
                e = int(rng.choice(ex_pool))
                mi = m - FIRST_M
                x = float(net_of(np.float64(mv[mi, si, e]), np.int8(kd[mi, si, e])))
                tot += x
                wins += x > 0
                cnt += 1
                flat = int(xm[mi, si, e])
    return (tot / cnt, wins / cnt) if cnt else (float("nan"), float("nan"))


def random_candidate(C, fold_trades, pools, rep_seed):
    """pools: day -> that day's candidate rows of the pipeline (any day of the test months)."""
    rng = np.random.default_rng(10_000 + rep_seed)
    tot = cnt = 0.0
    for f, (rows, exits) in fold_trades.items():
        if not len(rows):
            continue
        rows, exits = np.asarray(rows), np.asarray(exits)
        for d in np.unique(C["day"][rows]):
            k_n = int((C["day"][rows] == d).sum())
            pool_all = pools[int(d)]
            flat = -1
            for _ in range(k_n):
                pool = pool_all[C["m"][pool_all] >= flat]
                if not len(pool):
                    break
                i = int(rng.choice(pool))
                e = int(rng.choice(exits))
                tot += float(C["net"][i, e])
                cnt += 1
                flat = int(C["xmin"][i, e])
    return tot / cnt if cnt else float("nan")


def deflated_sharpe(series, best):
    """Bailey and Lopez de Prado's deflated Sharpe ratio of trial `best` among the rows of `series` (daily)."""
    nd = series.shape[1]
    sd = series.std(1, ddof=1)
    sr = np.where(sd > 0, series.mean(1) / np.where(sd > 0, sd, 1), 0.0)
    rng = np.random.default_rng(SEED)
    ok = np.flatnonzero(sd > 0)
    if len(ok) < 3:
        return float("nan"), float("nan"), float("nan")
    pairs = rng.integers(0, len(ok), size=(2000, 2))
    rho = [np.corrcoef(series[ok[a]], series[ok[b]])[0, 1] for a, b in pairs if a != b]
    rbar = float(np.nanmean(rho))
    N = len(series)
    neff = 1 + (1 - rbar) * (N - 1)
    g = 0.5772156649
    Z = NormalDist().inv_cdf
    sr0 = math.sqrt(float(np.var(sr[ok], ddof=1))) * ((1 - g) * Z(1 - 1 / neff) + g * Z(1 - 1 / (neff * math.e)))
    x = series[best]
    s = x.std(ddof=1)
    srb = x.mean() / s if s > 0 else 0.0
    sk = float(((x - x.mean()) ** 3).mean() / s ** 3) if s > 0 else 0.0
    ku = float(((x - x.mean()) ** 4).mean() / s ** 4) if s > 0 else 3.0
    den = math.sqrt(max(1 - sk * srb + (ku - 1) / 4 * srb * srb, 1e-12))
    return float(NormalDist().cdf((srb - sr0) * math.sqrt(nd - 1) / den)), neff, rbar


def fmt(x, d=2, sign=True):
    if x is None or not np.isfinite(x):
        return "n/a"
    return f"{x:+.{d}f}" if sign else f"{x:.{d}f}"


def exit_name(e):
    T, S, H, tr = EXITS[e]
    return f"T{T}/S{S}/{'EOD' if H is None else str(H) + 'm'}{'/trail' if tr else ''}"


def report(log=print):
    C, meta = load_cache()
    Rz = np.load(WORK / "rand.npz")
    R = {}
    for k in Rz.files:
        dn, f = k.split("__")
        R.setdefault(dn, {})[f] = Rz[k]
    R = {d: (v["move"], v["kind"], v["xmin"]) for d, v in R.items()}
    J = WORK / "jobs"
    test_days = [i for i, d in enumerate(C["days"]) if d[:7] in TEST_MONTHS]
    idx = block_idx(len(test_days))
    L = []
    P = L.append
    P("OF23: turning-point entries with wide exits, mined walk-forward on real ES 1-min bars")
    P(f"plan docs/ai/OF23_SWING_MINING_PLAN.md; runner sha256 {meta['runner_sha256']}; manifest sha256 {meta['manifest_sha256']}")
    import lightgbm
    import torch
    P(f"lightgbm {lightgbm.__version__}, torch {torch.__version__}, numpy {np.__version__}")
    P(f"days {len(C['days'])} used ({meta['days'][0]} .. {meta['days'][-1]}), {len(meta['dropped'])} dropped; "
      f"candidates {len(C['day'])}; test days {len(test_days)} ({TEST_MONTHS[0]} .. {TEST_MONTHS[-1]})")
    P("")
    # ---------------- per pipeline
    results = {}
    for pipe in PIPES:
        for model in (True, False):
            ft, fold_rows = {}, []
            for f in range(len(TEST_MONTHS)):
                if model:
                    r = json.loads((J / f"gbm_{pipe}_{f}_1.json").read_text(encoding="utf-8"))
                else:
                    mo = TEST_MONTHS[f]
                    fam = family_mask(C, pipe)
                    r = gbm_fold(C, pipe, (C["month"] < mo) & fam, (C["month"] == mo) & fam, train_days(C, mo), with_model=False)
                ft[f] = (r["rows"], r["exits"])
                fold_rows.append(r)
            rows = sum((v[0] for v in ft.values()), [])
            exits = sum((v[1] for v in ft.values()), [])
            s = summarize(C, rows, exits, test_days, idx)
            results[(pipe, model)] = (s, ft, fold_rows)
    for arm in ("trade", "free"):
        off = 1 if arm == "free" else 0
        ft, seeds = {}, {s_: {} for s_ in RL["seeds"]}
        for f in range(len(TEST_MONTHS)):
            per = [json.loads((J / f"rl_{arm}_{f}_{s_}.json").read_text(encoding="utf-8")) for s_ in RL["seeds"]]
            te = per[0]["te"]
            A = np.array([p["acts"] for p in per])
            nA = len(MENU_E) + off
            vote = np.array([np.bincount(A[:, k], minlength=nA).argmax() for k in range(A.shape[1])]) if len(te) else np.zeros(0, int)
            ft[f] = rl_walk(C, te, vote, off)
            for s_, p in zip(RL["seeds"], per):
                seeds[s_][f] = rl_walk(C, te, p["acts"], off)
        rows = sum((v[0] for v in ft.values()), [])
        exits = sum((v[1] for v in ft.values()), [])
        s = summarize(C, rows, exits, test_days, idx)
        seed_s = {s_: summarize(C, sum((v[0] for v in d_.values()), []), sum((v[1] for v in d_.values()), []), test_days, idx)
                  for s_, d_ in seeds.items()}
        results[(f"rl_{arm}", None)] = (s, ft, seed_s)
    # ---------------- controls for every row
    ctrl = {}
    by_day = {int(d): np.flatnonzero(C["day"] == d) for d in test_days}
    for key, (s, ft, _) in results.items():
        if s["trades"] == 0:
            ctrl[key] = (float("nan"),) * 5
            continue
        rtw = np.array([random_time(C, R, ft, test_days, k) for k in range(1, N_RAND + 1)])
        rt = rtw[:, 0]
        mask = family_mask(C, key[0]) if key[0] in PIPES else np.ones(len(C["day"]), bool)
        pools = {d: r_[mask[r_]] for d, r_ in by_day.items()}
        rc = np.array([random_candidate(C, ft, pools, k) for k in range(1, N_RAND + 1)])
        ctrl[key] = (float(np.nanpercentile(rt, 95)), float(np.nanmean(rt)), float(np.nanmean(rc)), float(np.nanpercentile(rc, 95)),
                     float(np.nanmean(rtw[:, 1])))
    # ---------------- resolution check (tick days): 1-s minus tick, every candidate x every exit chosen anywhere
    chosen = sorted({e for (s, ft, _) in results.values() for v in ft.values() for e in v[1]} | set(MENU_E))
    cr = C["chk_row"]
    res_line = "resolution check: no tick days"
    res_ok = True
    if len(cr) and chosen:
        n_sec = net_of(C["chk_move_sec"][:, chosen].astype(np.float64), C["chk_kind_sec"][:, chosen])
        n_tick = C["net"][cr][:, chosen].astype(np.float64)
        diff = (n_sec - n_tick).mean(1)
        dd = C["day"][cr]
        ud = sorted(set(dd))
        num = np.array([diff[dd == d].sum() for d in ud])
        den = np.array([(dd == d).sum() for d in ud], float)
        ix = block_idx(len(ud))
        mean = num.sum() / den.sum()
        se = ratio_se(num, den, ix)
        upper = mean + 1.96 * se
        res_ok = upper <= RESOLUTION_MAX
        res_line = (f"resolution check ({len(ud)} tick days, {len(cr)} candidates x {len(chosen)} exits): 1-s minus tick {fmt(mean)} ticks/trade, "
                    f"SE {fmt(se, 2, False)}, upper 95% {fmt(upper)} (needs <= +{RESOLUTION_MAX}): {'PASS' if res_ok else 'FAIL'}")
    # ---------------- decision
    def edge(key, s, t_req):
        c = ctrl[key]
        conds = [s["trades"] > 0 and s["t"] >= t_req, s.get("h1", -1) > 0 and s.get("h2", -1) > 0, s.get("stress", -1) > 0,
                 s["trades"] > 0 and s["net"] > c[0], res_ok, s["per_day"] >= FLOOR, s.get("no_best5", -1) > 0]
        return all(conds), conds
    prim_s = results[("pooled", True)][0]
    prim_edge, prim_c = edge(("pooled", True), prim_s, 2.0)
    # ---------------- behaviour first
    P("== What the agents did (test months, one trade at a time) ==")
    for key, label in [(("rl_trade", None), "RL-trade (must trade, picks the exit)"), (("rl_free", None), "RL-free (may skip)"),
                       (("pooled", True), "GBM miner (pooled, primary)")]:
        s, ft, extra = results[key]
        if s["trades"] == 0:
            P(f"{label}: no trades")
            continue
        ex = {}
        for v in ft.values():
            for e in v[1]:
                ex[e] = ex.get(e, 0) + 1
        top = sorted(ex.items(), key=lambda x: -x[1])[:4]
        P(f"{label}: {s['trades']} trades, {fmt(s['per_day'], 2, False)}/day; win {fmt(100 * s['win'], 0, False)}%, avg win "
          f"{fmt(s['avg_win'])}, avg loss {fmt(s['avg_loss'])}; exits target {fmt(100 * s['targets'], 0, False)}% / stop "
          f"{fmt(100 * s['stops'], 0, False)}% / time {fmt(100 * s['times'], 0, False)}%; net {fmt(s['net'])} ticks/trade "
          f"({fmt(s['net_day'])}/day)")
        P("    most used exits: " + ", ".join(f"{exit_name(e)} {k}" for e, k in top))
        P(f"    holding {fmt(s['hold'], 0, False)} min on average; long {fmt(s['long'])}, short {fmt(s['short'])} ticks/trade; by family: "
          + ", ".join(f"{k} {fmt(v[0])} ({v[1]})" for k, v in s["by_family"].items()))
        if key[0].startswith("rl_"):
            P("    per seed net/trade: " + ", ".join(f"s{k} {fmt(v['net'])} ({v['trades']})" for k, v in extra.items()))
        mo_net = {}
        for f, (rows, exits) in ft.items():
            if rows:
                mo_net[TEST_MONTHS[f]] = float(C["net"][np.asarray(rows), np.asarray(exits)].mean())
        P("    by month: " + ", ".join(f"{k[5:]} {fmt(v, 1)}" for k, v in mo_net.items()))
    P("")
    P("== Primary: the pooled mining procedure (M = 1, t >= 2) ==")
    P("fold  month    exit                  q     train net/day  train trades/day  test net/day  test trades/day  test net/trade")
    above = 0
    for f, r in enumerate(results[("pooled", True)][2]):
        rows = np.asarray(r["rows"], np.int64)
        nd_f = sum(1 for d in test_days if C["days"][d][:7] == TEST_MONTHS[f])
        tn = float(C["net"][rows, r["e"]].mean()) if len(rows) else float("nan")
        above += bool(len(rows)) and tn > r["train_net_trade"]
        tnd = float(C["net"][rows, r["e"]].sum()) / nd_f if nd_f else float("nan")
        trd = len(rows) / nd_f if nd_f else float("nan")
        P(f"{f + 1:4d}  {TEST_MONTHS[f]}  {exit_name(r['e']):20s}  {r['q']:.2f}  {fmt(r['train_net_day']):>13s}  "
          f"{fmt(r['train_trades_day'], 2, False):>16s}  {fmt(tnd):>12s}  {fmt(trd, 2, False):>15s}  {fmt(tn):>14s}")
    c = ctrl[("pooled", True)]
    s = prim_s
    tr_oof = np.mean([r["train_net_trade"] for r in results[("pooled", True)][2] if np.isfinite(r["train_net_trade"])])
    P(f"pooled: {s['trades']} trades ({fmt(s['per_day'], 2, False)}/day), net {fmt(s['net'])} ticks/trade, SE {fmt(s['se'], 2, False)}, "
      f"t {fmt(s['t'])}; training out-of-fold net/trade (mean of folds) {fmt(tr_oof)}")
    P(f"  halves {fmt(s['h1'])} / {fmt(s['h2'])}; stress {fmt(s['stress'])}; without the 5 best days {fmt(s['no_best5'])}; "
      f"median {fmt(s['median'])}; without the top 1% {fmt(s['no_top1pct'])}")
    P(f"  random-time control: mean {fmt(c[1])}, 95th percentile {fmt(c[0])}; random-candidate control: mean {fmt(c[2])}, "
      f"95th {fmt(c[3])}")
    P("  " + res_line)
    names = ["t >= 2", "both halves > 0", "stress > 0", "above random-time p95", "resolution check", ">= 0.5 trades/day", "> 0 without best 5 days"]
    P("  conditions: " + "; ".join(f"{n} {'yes' if ok else 'no'}" for n, ok in zip(names, prim_c)))
    X_b = max(s["net"], 0) + 2 * s["se"] if s["trades"] else float("nan")
    P(f"  PRIMARY: {'EDGE (a lead for OF23b, not a claim)' if prim_edge else 'NO EDGE'}; bound X = {fmt(X_b)} ticks/trade; "
      f"detectable at t = 2: {fmt(2 * s['se'], 2, False)} ticks/trade")
    flags = []
    if s["trades"] and s["t"] > 6:
        flags.append("t > 6")
    if above > len(TEST_MONTHS) / 2:
        flags.append(f"test net above training out-of-fold net in {above} of {len(TEST_MONTHS)} folds")
    mon = {}
    for d, x in zip(test_days, s["num"]):
        mon[C["days"][d][:7]] = mon.get(C["days"][d][:7], 0.0) + x
    if s["trades"] and s["num"].sum() > 0 and max(mon.values()) > 0.5 * s["num"].sum():
        flags.append(f"one month carries more than half of the total ({max(mon, key=mon.get)})")
    if s["trades"] and s["win"] > c[4] + 0.15:
        flags.append(f"win rate {100 * s['win']:.0f}% vs random-time {100 * c[4]:.0f}%")
    if not res_ok:
        flags.append("resolution bias above 0.25 ticks")
    P("  red flags: " + ("; ".join(flags) if flags else "none"))
    P("")
    P(f"== Secondaries (M = {M_SEC}, t >= {T_REQ:.2f}) ==")
    P("row               trades  /day   net/trade  SE     t      halves          stress  no-best5  rand-p95  edge")
    sec_keys = [(p, True) for p in FAMILIES] + [(p, False) for p in FAMILIES] + [("pooled", False), ("rl_trade", None), ("rl_free", None)]
    for key in sec_keys:
        s = results[key][0]
        nm = (key[0] + ("" if key[1] is None else (" model" if key[1] else " no-model")))
        if s["trades"] == 0:
            P(f"{nm:17s} {0:6d}")
            continue
        e_, _ = edge(key, s, T_REQ)
        P(f"{nm:17s} {s['trades']:6d}  {s['per_day']:4.2f}  {fmt(s['net']):>9s}  {fmt(s['se'], 2, False):5s}  {fmt(s['t']):>5s}  "
          f"{fmt(s['h1']):>6s}/{fmt(s['h2']):<6s}  {fmt(s['stress']):>6s}  {fmt(s['no_best5']):>8s}  {fmt(ctrl[key][0]):>8s}  {'EDGE' if e_ else 'no'}")
    P("")
    # ---------------- bars vs records
    rows = np.asarray(sum((v[0] for v in results[("pooled", True)][1].values()), []), np.int64)
    exits = np.asarray(sum((v[1] for v in results[("pooled", True)][1].values()), []), np.int64)
    if len(rows):
        nb = net_of(C["move_bar"][rows, exits].astype(np.float64), C["kind_bar"][rows, exits]).mean()
        P(f"bars alone (stop first when a bar reaches both): primary net {fmt(nb)} vs {fmt(results[('pooled', True)][0]['net'])} from the records")
    short_e = [e for e in chosen if EXITS[e][2] is not None and EXITS[e][2] <= 60 and EXITS[e][0] <= 24]
    wide_e = [e for e in chosen if e not in short_e]
    allr0 = np.flatnonzero(np.isin(C["month"], TEST_MONTHS))
    for nm_, es in (("short exits", short_e), ("wide exits", wide_e)):
        if es and len(allr0):
            P(f"two-touch share, {nm_}: bars {fmt(100 * C['both_bar'][allr0][:, es].mean(), 1, False)}%, "
              f"records {fmt(100 * C['both'][allr0][:, es].mean(), 1, False)}%")
    P("two-touch share (a bar / a step reached both target and stop at the chosen exits), by month:")
    allr = np.flatnonzero(np.isin(C["month"], TEST_MONTHS))
    for mo in TEST_MONTHS:
        sel = allr[C["month"][allr] == mo]
        if len(sel) and len(chosen):
            P(f"  {mo}: bars {fmt(100 * C['both_bar'][sel][:, chosen].mean(), 1, False)}%, records {fmt(100 * C['both'][sel][:, chosen].mean(), 1, False)}%")
    P("")
    # ---------------- naive mining
    P(f"== Naive mining (descriptive): the best combination chosen on all {len(C['days'])} days at once ==")
    ser, labels_ = [], []
    for p in PIPES:
        sp = np.load(J / f"naive_{p}.npy")
        ser.append(sp)
        labels_ += [(p, "model", q, e) for q in QS for e in range(NE)]
        idx_p = np.flatnonzero(family_mask(C, p))
        _, cnt2, taken = walk(C["day"][idx_p], C["m"][idx_p], np.ones((len(idx_p), NE), bool), C["xmin"][idx_p], C["net"][idx_p])
        s2 = np.zeros((NE, len(C["days"]) + 1))
        for e in range(NE):
            np.add.at(s2[e], C["day"][idx_p][taken[:, e]], C["net"][idx_p][taken[:, e], e])
        s2[:, -1] = cnt2
        ser.append(s2)
        labels_ += [(p, "no-model", 1.0, e) for e in range(NE)]
    SC = np.concatenate(ser, 0)
    S, trials_cnt = SC[:, :-1], SC[:, -1]
    per_day = S.mean(1)
    ok = trials_cnt / S.shape[1] >= FLOOR
    pool = np.flatnonzero(ok) if ok.any() else np.flatnonzero(trials_cnt == trials_cnt.max())
    best = int(pool[np.argmax(per_day[pool])])  # argmax takes the first: the tie order
    dsr, neff, rbar = deflated_sharpe(S, best)
    x = S[best]
    t_naive = x.mean() / (x.std(ddof=1) / math.sqrt(len(x))) if x.std() > 0 else float("nan")
    p, kind_, q, e = labels_[best]
    P(f"best of {len(S)} trials: {p} {kind_} q {q:.2f} {exit_name(e)}: net {fmt(x.mean())} ticks/day over {len(x)} days, naive t {fmt(t_naive)}")
    P(f"  deflated Sharpe ratio {fmt(dsr, 3, False)} (N {len(S)}, effective {fmt(neff, 0, False)}, mean pairwise correlation {fmt(rbar, 2, False)})")
    wf = results[(p, kind_ == "model")][0]
    P(f"  the same pipeline walk-forward: net {fmt(wf['net'])} ticks/trade on the test months (t {fmt(wf['t'])})")
    P("")
    P("Behaviour is reported first at the user's request; the decision rule is the plan's (section 7). A pass is a lead "
      "for OF23b on days from 2026-11-12, never a claim.")
    OUT.write_text("\n".join(L) + "\n", encoding="utf-8")
    log("\n".join(L))


# ================================================================ check
def check(log=print):
    days, dropped, _ = load_days()
    cnt = {}
    prev = None
    for day, side, b in days:
        lv = levels_for(prev, day, side, dropped)
        for c in candidates(b, lv, date.fromisoformat(day).weekday()):
            if b["volume"][c["m"] + 1] <= 0:
                continue
            for fam, cols in FAM_COLS.items():
                if any(c[k] for k in cols):
                    cnt.setdefault(day[:7], {}).setdefault(fam, 0)
                    cnt[day[:7]][fam] += 1
            cnt.setdefault(day[:7], {}).setdefault("all", 0)
            cnt[day[:7]]["all"] += 1
        prev = (day, side, b)
    log(f"{len(days)} days, {len(dropped)} dropped")
    for mo in sorted(cnt):
        log(f"{mo}: " + ", ".join(f"{k} {v}" for k, v in sorted(cnt[mo].items())))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--check", action="store_true")
    g.add_argument("--prepare", action="store_true")
    g.add_argument("--report", action="store_true")
    g.add_argument("--run", action="store_true")
    g.add_argument("--worker", nargs="+")
    ap.add_argument("--jobs", type=int, default=6)
    a = ap.parse_args(argv)
    log = lambda s: print(s, flush=True)  # noqa: E731
    if a.check:
        check(log)
    elif a.prepare:
        prepare(log)
    elif a.report:
        report(log)
    elif a.run:
        if not (WORK / "cache.npz").exists():
            prepare(log)
        run_jobs(a.jobs, log)
        report(log)
    else:
        worker(a.worker[0], a.worker[1:])
    return 0


if __name__ == "__main__":
    from ragrl.cli import hard_exit
    hard_exit(main())
