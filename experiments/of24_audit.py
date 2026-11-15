#!/usr/bin/env python3
"""OF24 audit: is the sim agent's replay profit free of look-ahead, and is it skill or luck?

  python experiments/of24_audit.py [--dir <Sierra Data>/RagAgent] [--out build-avx512/of24/audit.txt] [--reps 2000]

Reads only the study's own files (profit ledgers, decision logs, weight files) and the OF23 cache
(build-avx512/of23, days through 2026-11-11; the forward block is never read). docs/ai/OF24_AUDIT.md.

A. Look-ahead
  A1 every decision the study logged is a candidate of OF23's causal pipeline at the same minute and direction
     (that pipeline has its own truncation test; the C++ kernel only ever receives the bars so far)
  A2 the first decision of a run reproduces from the saved weights and the cached features
  A3 every entry fill lies inside the bar after the decision bar (or one tick outside: the quote)
  A4 how much of the replay is on days the starting policy was trained on (in-sample); a ledger named
     *holdout* is from weights trained only through of24_weights_holdout.json's "until" day
B. Luck or skill
  B1 the same trades re-priced with OF23's stricter fills (targets need a trade-through)
  B2 random candidates on the same days with the same number of trades and the same exits
  B3 each weight snapshot trading the cached days one position at a time, sampled as the study samples
     (what did the online learning add?)
  B4 which inputs the policy uses: shuffle a group of inputs, see how far the choices and the net move
C. Unseen days (needs experiments/of24_export.py --until DAY --out build-avx512/of24/of24_weights_holdout.txt)
  C1 the holdout policy on its training days and on the later days it never saw, against a random exit
  C2 the same after it learned online in a replay of those days (weights_replay_c*_holdout.txt)
  C3 fixed against volatility-scaled exits on the unseen days (labels from the trade records, cached)
D. Forward
  the live ledger alone decides: the pre-set rule in verdict()
"""
import argparse
import csv
import json
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "experiments"))
import of23_mine as M  # noqa: E402

DEFAULT_DIR = Path("C:/SierraChart/SierraChartInstance_2/Data/RagAgent")
START_W = REPO / "build-avx512" / "of24" / "of24_weights.txt"
BOOK = 6
SEED = 24
K_WALK = 32   # sampled walks per snapshot (the same random numbers for every snapshot)
HOLDOUT = REPO / "build-avx512" / "of24" / "of24_weights_holdout.json"
# the forward rule, fixed before the first live session (2026-11-16)
MIN_DAYS, MIN_TRADES = 20, 200
GROUPS = {
    "setup flags": ["c1_1", "c1_5", "c1_15", "c2", "c3", "c4", "c5", "c6", "c3_count", "c3_name", "dir"],
    "location": ["d_ext_entry", "d_ext_far", "d_open", "d_vwap", "vwap_z", "d_level", "fvg5_n", "fvg15_n", "d_fvg"],
    "volatility and volume": ["atr", "range_atr", "close_loc", "relvol"],
    "order flow (buy - sell volume)": ["delta1", "delta5", "delta15", "delta30"],
    "momentum": ["ret5", "ret15", "ret60", "ret_open", "slope5", "slope15", "slope60"],
    "time": ["minute", "dow"],
}


# ---------------------------------------------------------------- the study's files
def load_weights(path):
    """The OF24 weight file (of24_kernel.hpp Policy::load) as arrays."""
    L = Path(path).read_text(encoding="ascii").split("\n")
    head = L[1].split()
    W = {"in": int(head[1]), "hidden": int(head[3]), "actions": int(head[5]), "updates": int(head[7]), "path": Path(path)}
    i = 2
    while i + 1 < len(L) and L[i]:
        name, n = L[i].split()
        W[name] = np.array(L[i + 1].split(), np.float64)
        assert W[name].size == int(n), (path, name)
        i += 2
    return W


def policy(W, X, book=None):
    """(probabilities n x 13, value n) of the policy on raw OF23 features X (n x 37), as the kernel computes them."""
    F, H, A = W["in"], W["hidden"], W["actions"]
    Z = np.nan_to_num((X - W["mu"]) / W["sd"], nan=0.0, posinf=0.0, neginf=0.0)
    Z = np.hstack([Z, np.zeros((len(Z), BOOK)) if book is None else book])
    out = []
    for p, k in (("pol", A), ("val", 1)):
        h = np.tanh(Z @ W[f"{p}_W1"].reshape(H, F).T + W[f"{p}_b1"])
        h = np.tanh(h @ W[f"{p}_W2"].reshape(H, H).T + W[f"{p}_b2"])
        out.append(h @ W[f"{p}_W3"].reshape(k, H).T + W[f"{p}_b3"])
    y = out[0] - out[0].max(1, keepdims=True)
    e = np.exp(y)
    return e / e.sum(1, keepdims=True), out[1][:, 0]


def exit_index(rule):
    """'T48/S16/EOD/trail' -> index in of23_mine.EXITS, or -1 if it is not one of OF23's exits (a scaled exit)."""
    p = rule.split("/")
    e = (int(p[0][1:]), int(p[1][1:]), None if p[2] == "EOD" else int(p[2]), len(p) > 3)
    return M.EXITS.index(e) if e in M.EXITS else -1


def read_ledger(path):
    """Closed trades of one profit ledger: date, decision minute m, direction, exit index, sim net ticks, entry price."""
    rows = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            hh, mm = r["entry_ct"].split(":")
            m = int(hh) * 60 + int(mm) - (8 * 60 + 30) - 1  # the ledger has the entry bar (m + 1)
            rows.append((r["date"], m, 1 if r["side"] == "long" else -1, exit_index(r["exit_rule"]), float(r["net_ticks"]),
                         float(r["entry_px"]), r["exit"]))
    return rows


def read_log(path):
    """(header dict, decisions [(date, m, dir, action, p, value, flags, book[6])]) of one decision log."""
    head, dec = {}, []
    for line in open(path, encoding="ascii", errors="replace"):
        p = line.rstrip("\n").split(",")
        if p[0] == "H":
            head = dict(x.split("=", 1) for x in p[1:] if "=" in x)
        elif p[0] == "D" and len(p) >= 18:
            try:
                d = (date(1899, 12, 30) + timedelta(days=int(p[1]))).isoformat()
                dec.append((d, int(p[2]), int(p[3]), int(p[4]), float(p[5]), float(p[6]), p[7], [float(x) for x in p[8:14]]))
            except ValueError:
                continue
    return head, dec


# ---------------------------------------------------------------- statistics
def block_ratio(num, den, days, reps, rng):
    """sum(num) / sum(den) with a day-block bootstrap 95% interval. num, den per row; days = day key per row."""
    u, inv = np.unique(days, return_inverse=True)
    N, D = np.bincount(inv, num, len(u)), np.bincount(inv, den, len(u))
    if D.sum() == 0:
        return float("nan"), float("nan"), float("nan")
    idx = rng.integers(0, len(u), (reps, len(u)))
    b = N[idx].sum(1) / np.maximum(D[idx].sum(1), 1e-12)
    return N.sum() / D.sum(), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def fmt_ci(t):
    return f"{t[0]:+.2f} [{t[1]:+.2f}, {t[2]:+.2f}]"


def index_cache(C):
    """(day string, m, dir) -> cache row."""
    d = C["X"][:, M.FEATS.index("dir")].astype(int)
    return {(str(C["days"][C["day"][i]]), int(C["m"][i]), int(d[i])): i for i in range(len(d))}


def random_control(C, rows, exits, n_rand, seed):
    """Mean OF23 net per trade of random candidates: per day as many trades as the agent took, one at a time in
    time order, exits drawn from the agent's own exits (of23_mine.random_candidate, per day)."""
    by_day = {}
    for i in range(len(C["day"])):
        by_day.setdefault(int(C["day"][i]), []).append(i)
    pools = {d: np.array(v) for d, v in by_day.items()}
    return np.array([M.random_candidate(C, {0: (rows, exits)}, pools, seed + k) for k in range(n_rand)])


def menu_labels(C):
    """(net, exit minute) of the 12 menu exits per cached candidate."""
    return C["net"][:, M.MENU_E].astype(np.float64), C["xmin"][:, M.MENU_E]


def sampled_walk(day, m, net12, xmin12, P, U):
    """K one-position-at-a-time walks of a policy, actions sampled from P (n x 13) with the uniforms U (n x K), the
    exits' nets and exit minutes in net12, xmin12 (n x 12). Per candidate: mean net taken and mean share taken."""
    acts = np.minimum((P.cumsum(1)[:, None, :] < U[:, :, None]).sum(2), P.shape[1] - 1)
    e = np.maximum(acts - 1, 0)
    r = np.arange(len(P))[:, None]
    nt = net12[r, e]
    _, _, taken = M.walk(day, m, acts >= 1, xmin12[r, e], nt)
    return (taken * nt).mean(1), taken.mean(1)


def uniform_exits(n):
    P = np.full((n, 13), 1.0 / 12)
    P[:, 0] = 0.0
    return P


def exit_scale(atr, typical):
    """of24_kernel.hpp exit_scale: atr / typical atr within [0.5, 2]; 1 if either is missing."""
    if not (np.isfinite(atr) and np.isfinite(typical) and atr > 0 and typical > 0):
        return 1.0
    return float(min(2.0, max(0.5, atr / typical)))


def scaled_menu(k):
    """of24_kernel.hpp scaled_exit over the menu: (target, stop, hold, trailing) x 12."""
    r = lambda x: int(np.floor(x + 0.5))  # noqa: E731  (std::lround for positive values)
    return [(max(4, r(T * k)), max(2, r(S * k)), H, tr) for (T, S, H) in M.MENU for tr in M.TRS]


def dynamic_labels(C, meta, rows, typical, path, log=print):
    """OF23 labels (net, exit minute, factor) of the volatility-scaled menu for the cache rows `rows`, from the trade
    records of those days (never the forward block: of23_mine.records_of refuses it). Cached in `path`."""
    key = f"{len(rows)}_{int(rows.sum())}_{typical:.9f}"
    if Path(path).exists():
        z = np.load(path)
        if str(z["key"]) == key:
            return z["net"], z["xmin"], z["scale"]
    by_day = {d: (side, b) for d, side, b in M.load_days()[0]}
    atr, dirs = C["X"][:, M.FEATS.index("atr")], C["X"][:, M.FEATS.index("dir")].astype(int)
    net, xmin, scale = np.zeros((len(rows), 12)), np.zeros((len(rows), 12), np.int64), np.ones(len(rows))
    for di in np.unique(C["day"][rows]):
        day = meta["days"][int(di)]
        side, b = by_day[day]
        r, ts = M.records_of(day, side["contract"])
        P = M.path_from_records(r, ts, side["style"], "tick" if side["style"] == "tick" else "sec")
        for i in np.flatnonzero(C["day"][rows] == di):
            j = rows[i]
            scale[i] = exit_scale(atr[j], typical)
            L = M.labels(P, b, int(C["m"][j]), int(dirs[j]), exits=scaled_menu(scale[i]))
            net[i], xmin[i] = M.net_of(L["move"], L["kind"]), L["xmin"]
        log(f"   (labels for scaled exits: {day})")
    np.savez(path, key=key, net=net, xmin=xmin, scale=scale)
    return net, xmin, scale


# ---------------------------------------------------------------- the checks
def check_candidates(C, key, logs, out):
    out("A1. Every logged decision is a candidate of the causal OF23 pipeline (same day, minute, direction)")
    flag_cols = [M.FEATS.index(k) for k in ("c1_1", "c1_5", "c1_15", "c2", "c3", "c4", "c5", "c6")]
    days = set(map(str, C["days"]))
    worst, total = 1.0, 0
    for name, (head, dec) in logs.items():
        dec = [d for d in dec if d[0] in days]
        if not dec:
            continue
        hit = [key.get((d[0], d[1], d[2]), -1) for d in dec]
        n_hit = sum(h >= 0 for h in hit)
        same = sum(h >= 0 and "".join(str(int(C["X"][h, c] > 0)) for c in flag_cols) == d[6] for h, d in zip(hit, dec))
        share = n_hit / len(dec)
        worst, total = min(worst, share), total + len(dec)
        out(f"   {name}: {len(dec)} decisions on cached days, {n_hit} ({100 * share:.1f}%) are cached candidates, "
            f"{same} with the same setup flags")
    if not total:
        out("   no decision on a cached day was found: NOT CHECKED")
    return worst if total else float("nan")


def check_first_decision(C, key, logs, snaps, out):
    out("A2. The first decision of a run reproduces from the saved weights and the cached features")
    by_updates = {}
    for n_, W_ in snaps.items():
        by_updates.setdefault(W_["updates"], []).append((W_["path"].stat().st_mtime, n_, W_))
    first_day = str(C["days"][0])
    worst, n = 0.0, 0
    for name, (head, dec) in logs.items():
        if not dec or int(head.get("updates", -1)) not in by_updates:
            continue
        started = int(name.split("-")[-1].split(".")[0])  # the log's name carries its start time
        from_start = head.get("weights") == "start"  # the starting file itself; else this mode's learned file
        older = [c for c in by_updates[int(head["updates"])] if c[0] <= started and (c[1] == "start") == from_start]
        if not older:
            continue  # every such snapshot file was written after this run began: not what the run loaded
        _, sn, W = max(older, key=lambda c: c[0])
        d = dec[0]
        i = key.get((d[0], d[1], d[2]), -1)
        if i < 0 or d[0] == first_day:  # the cache has no prior day for its first day, so its level features differ
            out(f"   {name} from {sn}: not checkable (its first decision is "
                f"{'on the first cached day' if i >= 0 else 'not a cached candidate'})")
            continue
        p, v = policy(W, C["X"][i:i + 1], np.array([d[7]]))
        err = max(abs(p[0, d[3]] - d[4]), abs(v[0] - d[5]))
        worst, n = max(worst, err), n + 1
        out(f"   {name} from {sn}: logged p {d[4]:.4f} value {d[5]:.4f}, recomputed {p[0, d[3]]:.4f} {v[0]:.4f}")
    if not n:
        out("   no run starts from a saved snapshot on a cached candidate")
    return worst if n else float("nan")


def check_entries(trades, bars_dir, out):
    out("A3. Entry fills lie inside the bar after the decision bar, or one tick outside it (a market order fills at the quote)")
    inside = near = n = other = 0
    slip = []
    cache = {}
    for d, m, direction, _, _, px, _ in trades:
        f = Path(bars_dir) / f"{d}.npz"
        if d not in cache:
            cache[d] = np.load(f) if d <= M.END and f.exists() else None  # never a bar file past OF23's span
        b = cache[d]
        if b is None or not (0 <= m + 1 < len(b["open"])):
            continue
        t = px / 0.25
        if abs(t - b["open"][m + 1]) > 200:  # a different contract month on the chart (continuous-contract replay)
            other += 1
            continue
        n += 1
        inside += b["low"][m + 1] <= t <= b["high"][m + 1]
        near += b["low"][m + 1] - 1 <= t <= b["high"][m + 1] + 1
        slip.append(direction * (t - b["open"][m + 1]))
    if n:
        out(f"   {n} entries checked, {inside} ({100 * inside / n:.1f}%) inside the entry bar's range, {near} ({100 * near / n:.1f}%) "
            f"within one tick of it; fill minus the bar's open, against the trade: mean {np.mean(slip):+.2f} ticks; "
            f"{other} entries on another contract month")
    else:
        out("   no entry on a day with bars was found: NOT CHECKED")
    return near / n if n else float("nan")


def reprice(C, key, trades):
    """Rows, OF23 exit indices and sim nets of the trades that are cached candidates with one of OF23's exits."""
    rows, exits, sim, day = [], [], [], []
    for d, m, direction, e, net, _, _ in trades:
        i = key.get((d, m, direction), -1)
        if i >= 0 and e >= 0:
            rows.append(i); exits.append(e); sim.append(net); day.append(d)
    return np.array(rows, int), np.array(exits, int), np.array(sim), np.array(day)


def snapshot_values(C, snaps, reps, rng, out, split=None):
    """B3: each snapshot trading the cached days one position at a time, sampled, with the same random numbers."""
    out(f"B3. Each weight snapshot trading the cached days one position at a time ({K_WALK} sampled walks, OF23 fills)")
    days = C["day"]
    per_day = 1.0 / np.bincount(days)[days]  # sums to one per day: ratios against it are per session
    U = rng.random((len(days), K_WALK))
    net12, x12 = menu_labels(C)
    num, den = sampled_walk(days, C["m"], net12, x12, uniform_exits(len(days)), U)
    out(f"   a random exit at every candidate it can take: {fmt_ci(block_ratio(num, den, days, reps, rng))} ticks per trade")
    res, probs = {}, {}
    for name, W in snaps.items():
        probs[name], _ = policy(W, C["X"])
        res[name] = sampled_walk(days, C["m"], net12, x12, probs[name], U)
        num, den = res[name]
        out(f"   {name} ({W['updates']} updates): {fmt_ci(block_ratio(num, den, days, reps, rng))} per trade, "
            f"{den.sum() / len(np.unique(days)):.1f} trades a session")
    names = list(snaps)
    for a, b in zip(names, names[1:]):
        d = res[b][0] - res[a][0]
        tv = 0.5 * np.abs(probs[b] - probs[a]).sum(1).mean()
        line = (f"   {a} -> {b}: {100 * tv:.2f}% of the choices moved; net per session changed by "
                f"{fmt_ci(block_ratio(d, per_day, days, reps, rng))} ticks")
        if split and b in split:
            seen = np.isin(C["days"][days], list(split[b]))
            if seen.any() and (~seen).any():
                line += (f"; on the days that pass replayed {fmt_ci(block_ratio(d[seen], per_day[seen], days[seen], reps, rng))}, "
                         f"on the other days {fmt_ci(block_ratio(d[~seen], per_day[~seen], days[~seen], reps, rng))}")
        out(line)
    return res, U


def input_use(C, W, U, rng, out, repeats=3):
    out("B4. What the policy uses: one group of inputs shuffled across candidates (choices moved; net per session, OF23 fills)")
    n_days = len(np.unique(C["day"]))
    p0, _ = policy(W, C["X"])
    net12, x12 = menu_labels(C)
    base = sampled_walk(C["day"], C["m"], net12, x12, p0, U)[0].sum() / n_days
    res = {}
    for g, cols in GROUPS.items():
        ci = [M.FEATS.index(c) for c in cols]
        tv, dev = [], []
        for _ in range(repeats):
            X = C["X"].copy()
            X[:, ci] = X[rng.permutation(len(X))][:, ci]
            p, _ = policy(W, X)
            tv.append(0.5 * np.abs(p - p0).sum(1).mean())
            dev.append(sampled_walk(C["day"], C["m"], net12, x12, p, U)[0].sum() / n_days - base)
        res[g] = (float(np.mean(tv)), float(np.mean(dev)))
        out(f"   {g}: choices move {100 * res[g][0]:.1f}%, net per session {res[g][1]:+.1f} ticks (from {base:+.1f})")
    w1 = np.abs(W["pol_W1"].reshape(W["hidden"], W["in"]))
    col = np.sqrt((w1 ** 2).sum(0))
    out(f"   live book inputs (depth x3, MBO x2, spread): weight norms {', '.join(f'{x:.5f}' for x in col[-BOOK:])} "
        f"against a median {np.median(col[:-BOOK]):.2f} for the bar inputs")
    return res, float(col[-BOOK:].max() / np.median(col[:-BOOK]))


def holdout_cut():
    """The last training day of the holdout policy (of24_export.py --until), or None."""
    return json.loads(HOLDOUT.read_text(encoding="ascii"))["until"] if HOLDOUT.exists() else None


def is_unseen(days, cut):
    """Days after the cut: the complement of of24_export.py's training rule (day <= cut)."""
    return np.asarray(days) > cut


def unseen_days(C, meta, snaps, D, reps, rng, out, dynamic=True):
    """C: the holdout policy (trained through the cut only) on the later days. Returns (cut, CI on unseen days)."""
    wpath = HOLDOUT.with_suffix(".txt")
    if not (HOLDOUT.exists() and wpath.exists()):
        out("   no holdout policy (experiments/of24_export.py --until DAY --out build-avx512/of24/of24_weights_holdout.txt)")
        return None, None
    cut = holdout_cut()
    W = load_weights(wpath)
    days, m = C["day"], C["m"]
    un = is_unseen(C["days"][days], cut)
    net12, x12 = menu_labels(C)
    U = rng.random((len(days), K_WALK))
    P, _ = policy(W, C["X"])

    def per_trade(mask, probs, net=None, xm=None):
        num, den = sampled_walk(days[mask], m[mask], net12[mask] if net is None else net, x12[mask] if xm is None else xm,
                                probs[mask], U[mask])
        return num, den, block_ratio(num, den, days[mask], reps, rng)
    n_un = len(np.unique(days[un]))
    out(f"C1. The holdout policy: trained through {cut} only; {n_un} later days are unseen by it ({K_WALK} sampled walks, OF23 fills)")
    out(f"   on its training days: {fmt_ci(per_trade(~un, P)[2])} ticks per trade")
    num, den, ci = per_trade(un, P)
    out(f"   on the unseen days:   {fmt_ci(ci)} ticks per trade, {den.sum() / n_un:.1f} trades a session")
    out(f"   a random exit at every candidate, unseen days: {fmt_ci(per_trade(un, uniform_exits(len(days)))[2])}")
    if "start" in snaps:
        out(f"   the live starting policy on the same days (it was trained on them): {fmt_ci(per_trade(un, policy(snaps['start'], C['X'])[0])[2])}")
    after = sorted(D.glob("weights_replay_c*_holdout.txt"))
    if after:
        Wa = load_weights(after[0])
        Pa, _ = policy(Wa, C["X"])
        na, _, cia = per_trade(un, Pa)
        per_day = 1.0 / np.bincount(days)[days]
        out(f"C2. After learning online in a replay of those days ({Wa['updates']} updates): {fmt_ci(cia)} per trade; "
            f"{100 * 0.5 * np.abs(Pa - P).sum(1)[un].mean():.2f}% of the choices moved, net per session changed by "
            f"{fmt_ci(block_ratio(na - num, per_day[un], days[un], reps, rng))} ticks (these days are no longer unseen by it)")
    if dynamic:
        rows = np.flatnonzero(un)
        dn, dx, sc = dynamic_labels(C, meta, rows, float(W["mu"][M.FEATS.index("atr")]), HOLDOUT.with_name("dyn_labels.npz"))
        nd, dd, cid = per_trade(un, P, dn, dx)
        per_day = 1.0 / np.bincount(days)[days]
        out(f"C3. Volatility-scaled exits on the unseen days (factor: median {np.median(sc):.2f}, 10-90% {np.percentile(sc, 10):.2f} "
            f"to {np.percentile(sc, 90):.2f}): {fmt_ci(cid)} per trade against {fmt_ci(ci)} with the fixed menu; net per session "
            f"changed by {fmt_ci(block_ratio(nd - num, per_day[un], days[un], reps, rng))} ticks")
    return cut, ci


def lookahead_verdict(a1, a2, a3):
    """'none found' only if the candidate and the entry checks ran and passed; a missing first-decision check is said."""
    if a1 != a1 or a3 != a3:
        return "NOT CHECKED (no replay decisions or entries on cached days were found)"
    if a1 < 0.98 or a3 < 0.98 or (a2 == a2 and a2 >= 5e-3):
        return "CHECK the A lines"
    return "none found" + ("" if a2 == a2 else ", first-decision check not available")


def verdict(n_days, n_trades, lo, haircut):
    """The forward rule: enough live sessions and trades, and the lower 95% day-block bound of net ticks per trade,
    after the measured sim-fill haircut, above zero."""
    if n_days < MIN_DAYS or n_trades < MIN_TRADES:
        return f"too early ({n_days}/{MIN_DAYS} sessions, {n_trades}/{MIN_TRADES} trades)"
    return "skill is supported" if lo - haircut > 0 else "luck is not excluded"


# ---------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default=str(DEFAULT_DIR))
    ap.add_argument("--out", default=str(REPO / "build-avx512" / "of24" / "audit.txt"))
    ap.add_argument("--reps", type=int, default=2000)
    ap.add_argument("--rand", type=int, default=200)
    ap.add_argument("--no-dynamic", action="store_true", help="skip C3 (it reads the trade records of the unseen days once)")
    a = ap.parse_args(argv)
    D = Path(a.dir)
    lines = []

    def out(s=""):
        print(s, flush=True)
        lines.append(s)
    rng = np.random.default_rng(SEED)
    C, meta = M.load_cache()
    key = index_cache(C)
    cached = set(meta["days"])
    snaps = {}
    if START_W.exists():
        snaps["start"] = load_weights(START_W)
    for f in sorted(D.glob("weights_replay_c*_after_*.txt"), key=lambda f: load_weights(f)["updates"]):
        snaps[f.stem.split("_after_")[1]] = load_weights(f)
    hold_w = {}
    if HOLDOUT.with_suffix(".txt").exists():
        hold_w["holdout"] = load_weights(HOLDOUT.with_suffix(".txt"))
    logs = {f.name: read_log(f) for f in sorted(D.glob("replay-c*.csv")) if f.stat().st_size > 2000}
    ledgers = {f.stem.replace("profit_trades_", ""): read_ledger(f) for f in sorted(D.glob("profit_trades_replay*.csv"))}
    seen_led = {k: v for k, v in ledgers.items() if "holdout" not in k}
    out(f"OF24 audit: {len(ledgers)} replay ledgers, {len(logs)} decision logs, {len(snaps)} weight snapshots, "
        f"{len(cached)} cached days ({meta['days'][0]} to {meta['days'][-1]})")
    out()
    out("A. LOOK-AHEAD")
    a1 = check_candidates(C, key, logs, out)
    a2 = check_first_decision(C, key, logs, {**snaps, **hold_w}, out)
    a3 = check_entries([t for L in ledgers.values() for t in L], M.BARS, out)
    all_tr = [t for L in seen_led.values() for t in L]
    in_s = sum(t[0] in cached for t in all_tr) / max(len(all_tr), 1)
    out(f"A4. {100 * in_s:.0f}% of the replay trades (holdout replays aside) are on days the starting policy was trained on: "
        "that replay profit is in-sample. The input scaling (mean, sd) is also from those days.")
    out()
    out("B. LUCK OR SKILL ON THE DAYS IT WAS TRAINED ON")
    out("B1/B2. The same trades with OF23's fills, and random candidates with the same exits (ticks per trade, 95% by day)")
    hair, pct, hold_line = [], [], None
    cut0 = holdout_cut()
    for name, tr in ledgers.items():
        note = ""
        if "holdout" in name:
            if cut0 is None:
                out(f"   {name}: no holdout cut on file (of24_weights_holdout.json): skipped")
                continue
            kept = [t for t in tr if is_unseen([t[0]], cut0)[0]]
            note, tr = f" ({len(tr) - len(kept)} trades on or before {cut0} left out)" if len(kept) < len(tr) else "", kept
        rows, exits, sim, day = reprice(C, key, tr)
        if len(rows) < 30 or len(rows) < 0.95 * len(tr):
            out(f"   {name}: {len(tr)} trades, {len(rows)} on OF23's exits and candidates: not re-priced "
                "(too few, or scaled exits: a partial sample would be a biased one)")
            continue
        of23 = C["net"][rows, exits].astype(np.float64)
        ones = np.ones(len(rows))
        ctl = random_control(C, rows, exits, a.rand, SEED)
        pc = float((ctl < of23.mean()).mean())
        ci_of23 = block_ratio(of23, ones, day, a.reps, rng)
        hair.append((len(rows), sim.mean() - of23.mean()))
        line = (f"{name}: {len(rows)} of {len(tr)} trades re-priced; sim fills {fmt_ci(block_ratio(sim, ones, day, a.reps, rng))}, "
                f"OF23 fills {fmt_ci(ci_of23)}; random candidates {ctl.mean():+.2f} "
                f"(5-95%: {np.percentile(ctl, 5):+.2f} to {np.percentile(ctl, 95):+.2f}); the agent beats {100 * pc:.0f}% of them")
        if "holdout" in name:
            hold_line = (line + note, ci_of23, pc)
        else:
            pct.append((len(rows), pc))
            out("   " + line)
    haircut = sum(n * h for n, h in hair) / max(sum(n for n, _ in hair), 1) if hair else float("nan")
    beat = sum(n * p for n, p in pct) / max(sum(n for n, _ in pct), 1) if pct else float("nan")
    out(f"   sim fills flatter the agent by {haircut:+.2f} ticks per trade (the haircut used on live results)")
    split = {k.replace("replay_", ""): {t[0] for t in tr} for k, tr in seen_led.items() if k.replace("replay_", "") in snaps}
    U = None
    if snaps:
        _, U = snapshot_values(C, snaps, a.reps, rng, out, split)
        use, book_share = input_use(C, snaps[list(snaps)[-1]], U, rng, out)
    out()
    out("C. UNSEEN DAYS")
    cut, ci_un = unseen_days(C, meta, snaps, D, a.reps, rng, out, dynamic=not a.no_dynamic)
    if hold_line:
        out(f"   the holdout replay in Sierra (learning online as it goes), {hold_line[0]}")
    out()
    out("D. FORWARD (the live ledger; days no weight has seen)")
    live = D / "profit_trades_live.csv"
    tr = read_ledger(live) if live.exists() else []
    if tr:
        net, day = np.array([t[4] for t in tr]), np.array([t[0] for t in tr])
        ci = block_ratio(net, np.ones(len(net)), day, a.reps, rng)
        out(f"   {len(set(day))} sessions, {len(tr)} trades, sim net {fmt_ci(ci)} per trade, after the haircut {ci[0] - haircut:+.2f}")
        v = verdict(len(set(day)), len(tr), ci[1], haircut)
    else:
        out("   no live trades yet")
        v = verdict(0, 0, float("nan"), haircut)
    out(f"   rule: at least {MIN_DAYS} sessions and {MIN_TRADES} trades, and the lower 95% bound minus the haircut above zero")
    out()
    out("VERDICT")
    out(f"   look-ahead in the decisions: {lookahead_verdict(a1, a2, a3)}"
        f" (candidates matched {100 * a1:.1f}%, first-decision error {a2:.4f}, entries within a tick of the next bar {100 * a3:.1f}%)")
    out(f"   replay profit on training days: in-sample ({100 * in_s:.0f}% of trades); {haircut:+.2f} ticks per trade of it is the sim's fills; "
        f"it beats {100 * beat:.0f}% of random-candidate draws, as a policy that has seen those days would")
    if ci_un is not None:
        out(f"   unseen days: the same recipe trained without them makes {fmt_ci(ci_un)} ticks per trade there: "
            f"{'skill is supported' if ci_un[1] > 0 else 'no skill shown (the interval includes zero)' if ci_un[2] > 0 else 'it loses' if ci_un[2] <= 0 else 'not measurable'}")
    if snaps:
        of = use["order flow (buy - sell volume)"]
        out(f"   order flow: shuffling the buy/sell-volume inputs moves {100 * of[0]:.1f}% of the choices; the live book inputs carry "
            f"{100 * book_share:.3f}% of a typical input's weight")
    out(f"   forward: {v}")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    from ragrl.cli import hard_exit
    hard_exit(main())
