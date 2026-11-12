#!/usr/bin/env python3
"""OF17: a signal census of the agents' 13 market features on real ES (docs/ai/OF17_SIGNAL_CENSUS_PLAN.md).

  PYTHONPATH=python python experiments/of17_signal_census.py --out experiments/of17_signal_census.txt

Through the kit: kit.ps1 run of17_census 'bash -c "PYTHONPATH=python .venv/Scripts/python.exe -X faulthandler -u
experiments/of17_signal_census.py --out experiments/of17_signal_census.txt"'.

Data: the manifest's `seen` days (the 8 front-month days OF16 phase 2 scored, A81), opened through
real.open_tapes("seen") (logged real_seen). Samples: OF16's decisions exactly (VecReplayEnv, warm-up 1050 s,
600 s windows, 1 s decisions, flat policy); decision i of window k fills in bucket f = 10500 + 6000 k + 10 i.
Outcomes at h in HORIZONS seconds from the tape's end-of-bucket quotes (ticks): move = mid(f+10h) - mid(f);
long net = bid(f+10h) - ask(f) - FEE; short net = bid(f) - ask(f+10h) - FEE.
Predictors, fitted on one month (Sep or Nov) and scored on the other (plan section 4):
  S  each feature alone: fitting-month 20th / 80th percentiles and the direction of (top - bottom quintile mean
     move); trade the extreme quintiles in that direction.
  G  LightGBM on the 13 features (fixed parameters), trade the sign of the prediction when |prediction| >= the
     fitting month's 80th / 95th / 99th percentile of |prediction|.
Statistics: out-of-fold mean net per trade, day-clustered bootstrap SE (score_real.day_bootstrap, B 10000, seed 0),
t = mean / SE, per-month means; M = 128, t_req = 2 + sqrt(2 ln 128). Candidate: net > 0, t >= t_req, positive in
both scored months (then OF17b on fresh confirm days; never a claim here).
"""
import argparse
import math
import os
import sys
import time
from pathlib import Path

for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(k, "1")
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "python"))
import ragrl  # noqa: E402,F401  (WMI workaround before lightgbm)
import numpy as np  # noqa: E402

from ragrl import native, real, score_real, stats  # noqa: E402

FEATS = ["delta5", "delta30", "delta120", "ofi5", "ofi30", "relvol30", "imb_l1", "imb_l5", "ret5", "ret30", "ret120",
         "vwap_dist", "absorb"]  # observation 0..12 (13..15 are position, open P&L and the clock)
HORIZONS = [1, 5, 15, 30, 60, 120, 300, 600]  # seconds
QS = [80, 95, 99]  # LightGBM |prediction| percentiles
FEE = 2 * 1.50 / 12.5  # ticks per round trip ($1.50 per side, $12.50 per tick)
WIN, WARM, STEPS = 6000, 10500, 600
M = len(FEATS) * len(HORIZONS) + len(HORIZONS) * len(QS)
LGBM = dict(objective="regression", learning_rate=0.05, num_leaves=31, min_data_in_leaf=500, feature_fraction=1.0, bagging_fraction=1.0,
            bagging_freq=0, seed=0, num_threads=1, deterministic=True, verbose=-1)
LGBM_ROUNDS = 200
ALIGN_TOL = 1e-4


def collect():
    t0 = time.time()
    m = real.load_manifest()
    tapes = real.open_tapes("seen", reason="OF17 signal census (experiments/of17_signal_census.py)")
    R = native.load()
    rp = score_real.replay_params(1050.0)
    X, F, D, quotes, excluded = [], [], [], {}, []
    align = 0.0
    for d, (tape, side) in tapes.items():
        n = score_real.windows_on_tape(tape, rp)
        played, ex = real.clean_windows(n, side, rp.window_seconds, rp.warmup_seconds)
        excluded += [{"day": d, **e} for e in ex]
        env = R.VecReplayEnv(tape, rp, played)
        o = env.reset()
        seq = []
        for _ in range(STEPS):
            seq.append(np.asarray(o, dtype=np.float64))
            o, _, done = env.step(np.ones(len(played), dtype=np.int64))
        assert done[0]
        obs = np.stack(seq)  # [600, W, 16]
        b = real.load_buckets(side["csv"])
        if len(b["bid"]) != tape.size:
            raise RuntimeError(f"{d}: {len(b['bid'])} Python buckets, TickTape has {tape.size}")
        bid, ask = b["bid"].astype(np.float64), b["ask"].astype(np.float64)
        b1, a1 = b["b1"].astype(np.float64), b["a1"].astype(np.float64)
        imb = np.where(b1 + a1 > 0, (b1 - a1) / np.maximum(1, b1 + a1), 0.0)
        for wi, k in enumerate(played):
            f = k * WIN + WARM + 10 * np.arange(STEPS)
            align = max(align, float(np.abs(imb[f - 1] - obs[:, wi, 6] / 2).max()))
            X.append(obs[:, wi, :len(FEATS)])
            F.append(f)
            D.append(np.full(STEPS, d))
        quotes[d] = (bid, ask)
    out = {"X": np.concatenate(X), "f": np.concatenate(F), "day": np.concatenate(D), "quotes": quotes, "excluded": excluded,
           "align": align, "subset": {d: real.subset_of(d, m) for d in tapes}, "manifest_sha256": real.manifest_sha256(),
           "seconds": time.time() - t0}
    if align > ALIGN_TOL:
        raise RuntimeError(f"alignment check failed: imb_l1 differs from the tape's L1 imbalance at f - 1 by {align:.3g}")
    return out


def outcomes(data, h):
    """(valid, move, long_net, short_net) per sample at horizon h seconds."""
    n = 10 * h
    N = len(data["f"])
    valid = np.zeros(N, bool)
    move, lnet, snet = (np.full(N, np.nan) for _ in range(3))
    for d, (bid, ask) in data["quotes"].items():
        i = np.flatnonzero(data["day"] == d)
        f = data["f"][i]
        ok = f + n < len(bid)
        i, f = i[ok], f[ok]
        valid[i] = True
        mid0, mid1 = 0.5 * (bid[f] + ask[f]), 0.5 * (bid[f + n] + ask[f + n])
        move[i] = mid1 - mid0
        lnet[i] = bid[f + n] - ask[f] - FEE
        snet[i] = bid[f] - ask[f + n] - FEE
    return valid, move, lnet, snet


def score_trades(side, move, lnet, snet, day, month):
    """Per-trade results for side in {-1, 0, +1}: dict of arrays (trades only)."""
    t = side != 0
    s = side[t]
    return {"net": np.where(s > 0, lnet[t], snet[t]), "gross": s * move[t], "day": day[t], "month": month[t]}


def summarize(tr, months):
    n = len(tr["net"])
    if n < 2:
        return {"trades": n, "net": float("nan"), "gross": float("nan"), "se": float("nan"), "t": float("nan"),
                "ci": [float("nan")] * 2, "by_month": {mo: float("nan") for mo in months}}
    db = score_real.day_bootstrap(tr["net"], tr["day"])
    return {"trades": n, "net": float(tr["net"].mean()), "gross": float(tr["gross"].mean()), "se": db["se"], "t": db["t"],
            "ci": db["ci95"], "by_month": {mo: (float(tr["net"][tr["month"] == mo].mean()) if (tr["month"] == mo).any() else float("nan"))
                                           for mo in months}}


def run(data):
    import lightgbm as lgb
    X, day = data["X"], data["day"]
    month = np.array([data["subset"][d] for d in day])
    months = sorted(set(month.tolist()))
    assert len(months) == 2, months
    S, G, R2 = {}, {}, {}
    for h in HORIZONS:
        valid, move, lnet, snet = outcomes(data, h)
        sideS = {j: np.zeros(len(day), np.int8) for j in range(len(FEATS))}
        sideG = {q: np.zeros(len(day), np.int8) for q in QS}
        R2[h] = {}
        for test_m in months:
            fit = valid & (month != test_m)
            sc = valid & (month == test_m)
            for j in range(len(FEATS)):
                x = X[:, j]
                lo, hi = np.percentile(x[fit], [20, 80])
                top, bot = fit & (x > hi), fit & (x < lo)
                if not top.any() or not bot.any():
                    continue
                dsign = int(np.sign(move[top].mean() - move[bot].mean()))
                sideS[j][sc & (x > hi)] = dsign
                sideS[j][sc & (x < lo)] = -dsign
            model = lgb.train(LGBM, lgb.Dataset(X[fit], move[fit]), num_boost_round=LGBM_ROUNDS)
            p_fit, p_sc = model.predict(X[fit]), model.predict(X[sc])
            yy = move[sc]
            R2[h][test_m] = float(1 - ((yy - p_sc) ** 2).sum() / ((yy - yy.mean()) ** 2).sum())
            idx = np.flatnonzero(sc)
            for q in QS:
                thr = np.percentile(np.abs(p_fit), q)
                pick = np.abs(p_sc) >= thr
                sideG[q][idx[pick]] = np.sign(p_sc[pick]).astype(np.int8)
        for j in range(len(FEATS)):
            S[(FEATS[j], h)] = summarize(score_trades(sideS[j], move, lnet, snet, day, month), months)
        for q in QS:
            G[(q, h)] = summarize(score_trades(sideG[q], move, lnet, snet, day, month), months)
        # the round-trip cost actually paid by the trades of each rule, for the descriptive table
        allv = valid
        cost = float(np.nanmean(np.concatenate([(move[allv] - lnet[allv]), (-move[allv] - snet[allv])])))
        R2[h]["cost"] = cost
    return S, G, R2, months


def n_(x, d=2, sign=True):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "-"
    return f"{x:+.{d}f}" if sign else f"{x:.{d}f}"


def report(data, S, G, R2, months):
    t_req = stats.t_required(M)
    L = []
    w = L.append
    w("OF17 signal census: the agents' 13 market features on real ES (docs/ai/OF17_SIGNAL_CENSUS_PLAN.md)")
    w("Generated by experiments/of17_signal_census.py. Data: the manifest's seen days (A81), opened as real_seen.")
    days = sorted(set(data["day"].tolist()))
    w(f"manifest sha256 {data['manifest_sha256'][:16]}  days {', '.join(days)}")
    w(f"samples {len(data['day'])} (decisions of OF16's windows), excluded windows {len(data['excluded'])}; alignment: max |imb_l1/2 - "
      f"tape L1 imbalance at f-1| = {data['align']:.2g} (tolerance {ALIGN_TOL:g})")
    w(f"costs: ${1.5:g}/side = {FEE:.2f} ticks per round trip plus the recorded spreads; M = {M}, t_req = {t_req:.2f}; "
      "day-clustered bootstrap SE (8 days)")
    w("")
    w("Round-trip cost paid (ticks: spread at entry and exit + fees, mean over all samples) and LightGBM out-of-fold R^2 by horizon")
    w(f"  {'h (s)':>6}{'cost':>8}" + "".join(f"{'R2 ' + mo:>12}" for mo in months))
    for h in HORIZONS:
        w(f"  {h:>6}{R2[h]['cost']:>8.2f}" + "".join(f"{R2[h][mo]:>12.4f}" for mo in months))
    w("")
    w("G: LightGBM on the 13 features, trade the sign when |prediction| >= the fitting month's percentile (out of fold, per trade, ticks)")
    w(f"  {'h (s)':>6}{'pct':>5}{'trades':>9}{'gross':>8}{'net':>8}{'se':>7}{'t':>7}{'day CI95':>20}" + "".join(f"{mo:>8}" for mo in months) + "  cand")
    for h in HORIZONS:
        for q in QS:
            r = G[(q, h)]
            cand = r["net"] > 0 and r["t"] >= t_req and all(r["by_month"][mo] > 0 for mo in months)
            w(f"  {h:>6}{q:>5}{r['trades']:>9}{n_(r['gross']):>8}{n_(r['net']):>8}{n_(r['se'], 2, False):>7}{n_(r['t'], 1):>7}"
              f"{'[' + n_(r['ci'][0]) + ', ' + n_(r['ci'][1]) + ']':>20}" + "".join(f"{n_(r['by_month'][mo]):>8}" for mo in months)
              + ("  YES" if cand else "  -"))
    w("")
    w("S: each feature alone, extreme quintiles in the fitting month's direction: GROSS per trade (ticks, out of fold)")
    w(f"  {'feature':<10}" + "".join(f"{str(h) + ' s':>8}" for h in HORIZONS))
    for f in FEATS:
        w(f"  {f:<10}" + "".join(f"{n_(S[(f, h)]['gross']):>8}" for h in HORIZONS))
    w("S: NET per trade (ticks)")
    w(f"  {'feature':<10}" + "".join(f"{str(h) + ' s':>8}" for h in HORIZONS))
    for f in FEATS:
        w(f"  {f:<10}" + "".join(f"{n_(S[(f, h)]['net']):>8}" for h in HORIZONS))
    w("S: day-clustered t of the net")
    w(f"  {'feature':<10}" + "".join(f"{str(h) + ' s':>8}" for h in HORIZONS))
    for f in FEATS:
        w(f"  {f:<10}" + "".join(f"{n_(S[(f, h)]['t'], 1):>8}" for h in HORIZONS))
    w("")
    w("Best gross per trade by horizon (any rule) against the round-trip cost; power: the median over the 16 rules of t_req x day-SE,")
    w("the net per trade a rule would need to clear t_req at that horizon (decisions 1 s apart overlap at long horizons)")
    allr = [(("S", f), h, S[(f, h)]) for f in FEATS for h in HORIZONS] + [(("G", q), h, G[(q, h)]) for q in QS for h in HORIZONS]
    for h in HORIZONS:
        best = max((x for x in allr if x[1] == h and not math.isnan(x[2]["gross"])), key=lambda x: x[2]["gross"])
        need = float(np.median([t_req * x[2]["se"] for x in allr if x[1] == h and not math.isnan(x[2]["se"])]))
        w(f"  {h:>4} s: {best[0][0]} {best[0][1]}: gross {best[2]['gross']:+.2f}, net {best[2]['net']:+.2f} (cost {R2[h]['cost']:.2f}); "
          f"gross / cost {best[2]['gross'] / R2[h]['cost']:.2f}; detectable net >= {need:.2f}")
    cands = [(k, h, r) for k, h, r in allr if r["net"] > 0 and r["t"] >= t_req and all(r["by_month"][mo] > 0 for mo in months)]
    flags = [(k, h, r) for k, h, r in allr if (r["t"] > 10) or (h <= 5 and r["gross"] > 2)]
    w("")
    w(f"red flags (t > 10, or gross > 2 ticks at h <= 5 s): {len(flags)}")
    for k, h, r in flags:
        w(f"  {k[0]} {k[1]} h {h}: gross {r['gross']:+.2f}, net {r['net']:+.2f}, t {n_(r['t'], 1)}")
    best_t = max(allr, key=lambda x: -1e9 if math.isnan(x[2]["t"]) else x[2]["t"])
    w(f"highest t of the net: {best_t[0][0]} {best_t[0][1]} at {best_t[1]} s: net {best_t[2]['net']:+.2f}, t {best_t[2]['t']:.2f}")
    w("")
    if cands:
        w("DECISION (plan section 5): candidates: " + "; ".join(f"{k[0]} {k[1]} at {h} s (net {r['net']:+.2f}, t {r['t']:.1f})" for k, h, r in cands)
          + ". Not a claim: OF17b, a one-time confirmation on the next 8 confirm days.")
    else:
        w(f"DECISION (plan section 5): no candidate among {M} tests. No rule on the agents' 13 features, alone or combined, whose")
        w("direction holds from one month to the other, beats a market-order round trip at 1 s to 10 min. At 1-60 s this is tight (the best")
        w("gross is a fraction of the cost, with small SEs); at 120-600 s it reads 'none detected': overlapping outcomes leave only a few")
        w("hundred independent moves per month, and the long-horizon features do predict within a month with a sign that flips between")
        w("September and November (OF17 review). Not sampled: the first 1050 s after 08:30 (the feature warm-up) and the last ~150 s.")
    return "\n".join(L) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    data = collect()
    S, G, R2, months = run(data)
    txt = report(data, S, G, R2, months)
    Path(a.out).write_text(txt, encoding="utf-8")
    print(txt)
    print(f"(collection {data['seconds']:.0f} s)")


if __name__ == "__main__":
    main()
