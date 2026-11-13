#!/usr/bin/env python3
"""OF20: absorption and exhaustion at key levels, against a plain touch of the same level (docs/ai/OF20_LEVELS_PLAN.md).

  python experiments/of20_atlevels.py --out experiments/of20_atlevels.txt

OF19's data, signals, fills, exits and statistics (experiments/of19_footprint.py, imported unchanged in behaviour;
the signal functions only also report each setup's formation time and price), with the levels of
experiments/of20_levels.py judged at the setup's formation time. Setups: absorption = OF19 P2 W 15 target 4;
exhaustion = OF19 P3 N 1 e 5; shells = the same without the footprint condition.
"""
import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "experiments"))
spec = importlib.util.spec_from_file_location("of19", REPO / "experiments" / "of19_footprint.py")
of19 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(of19)
import of20_levels as LV  # noqa: E402

SETUPS = {"abs": ("P2", dict(W=15, tgt=4)), "exh": ("P3", dict(N=1, e=5))}
TOLS = (1, 2)
LEVEL_TYPES = {"pdh": "prior high/low", "pdl": "prior high/low", "pdc": "prior close", "ppoc": "prior POC/VA", "pvah": "prior POC/VA",
               "pval": "prior POC/VA", "onh": "overnight", "onl": "overnight", "ibh": "initial balance", "ibl": "initial balance",
               "vwap": "VWAP bands"}
T_PRI = 2.0
M_SEC = 8
T_SEC = 2 + math.sqrt(2 * math.log(M_SEC))


def classify(ft, price, lv, rv):
    """Distance (ticks) to the nearest static level and to the nearest VWAP-set level counted at formation time ft,
    and the nearest level's type."""
    stat = [(abs(price - lv[k]), k) for k in LV.STATIC if lv.get(k) is not None and (k not in ("ibh", "ibl") or ft >= LV.IB_END)]
    vw = [(abs(price - v), "vwap") for v in rv.at(ft)]
    ds = min(stat)[0] if stat else 1 << 30
    dv = min(vw)[0] if vw else 1 << 30
    near = min(stat + vw) if stat + vw else (1 << 30, None)
    return ds, dv, LEVEL_TYPES.get(near[1])


def collect(T, E, F, AB, thr, lv, rv, shell):
    """Per setup: dict of arrays (gross, keep, side, sig_t, ds, dv, ltype) for its signals (or its shell's)."""
    out = {}
    for name, (p, kw) in SETUPS.items():
        meta = []
        if p == "P2":
            rows = of19.sig_P2(T, E, AB, kw["W"], thr[kw["W"]], shell=shell, meta=meta)
        else:
            rows = of19.sig_P3(F, E, kw["N"], kw["e"], shell=shell, meta=meta)
        g, keep, sig_t, side, _sd, _hold = of19.run_rows(E, rows, p, kw)
        cl = [classify(ft, px, lv, rv) for ft, px in meta]
        ds = np.array([c[0] for c in cl], float) if cl else np.zeros(0)
        dv = np.array([c[1] for c in cl], float) if cl else np.zeros(0)
        lt = np.array([c[2] or "" for c in cl], object) if cl else np.zeros(0, object)
        ft = np.array([m[0] for m in meta], float) if meta else np.zeros(0)
        assert len(meta) == len(g)
        out[name] = {"g": g[keep], "side": side[keep], "sig_t": sig_t[keep], "ds": ds[keep], "dv": dv[keep], "lt": lt[keep],
                     "ft": ft[keep], "skip": int((~keep).sum())}
    return out


def mask_for(d, levelset, tol):
    if levelset == "static":
        return d["ds"] <= tol
    if levelset == "vwap":
        return d["dv"] <= tol
    if levelset == "any":
        return np.minimum(d["ds"], d["dv"]) <= tol
    if levelset == "away":
        return np.minimum(d["ds"], d["dv"]) > 2
    raise ValueError(levelset)


TESTS = [("primary", ("abs", "exh"), "any", 2)] + [(f"{s} {ls} tol {t}", (s,), ls, t) for s in SETUPS for ls in ("static", "vwap") for t in TOLS]
EXTRA = [(f"{s} away", (s,), "away", 2) for s in SETUPS] + [(f"{s} any tol 2", (s,), "any", 2) for s in SETUPS]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--trades", default=str(REPO / "build-avx512" / "real" / "trades"))
    a = ap.parse_args(argv)
    days = of19.load_days(Path(a.trades))
    names = [d for d, _ in days]
    nsc = len(days) - of19.WARM
    keys = [t[0] for t in TESTS + EXTRA]
    # per test, per setup, per day: setup and shell sums
    acc = {k: {s: {f: np.zeros(nsc) for f in ("n", "net", "gross", "win", "sn", "sgross", "long_n", "long_net")} for s in SETUPS} for k in keys}
    types = {}
    tod = np.zeros(13)
    hist, prevT = [], None
    level_log = []
    for di, (d, npz) in enumerate(days):
        T, _qc = of19.load_trades(npz)
        AB = of19.absorption_stats(T)
        F = of19.footprint(T)
        if di >= of19.WARM:
            si = di - of19.WARM
            contract = json.loads((Path(a.trades) / f"{d}.json").read_text(encoding="utf-8"))["contract"]
            if json.loads((Path(a.trades) / f"{names[di - 1]}.json").read_text(encoding="utf-8"))["contract"] != contract:
                raise RuntimeError(f"{d}: the prior trades day is another contract (plan section 2)")
            lv = LV.static_levels(d, contract, T, prevT)
            level_log.append((d, lv))
            rv = LV.RunningVWAP(T)
            E = of19.Events(T)
            thr = of19.thresholds(hist[-of19.WARM:])
            sig = collect(T, E, F, AB, thr, lv, rv, shell=False)
            sh = collect(T, E, F, AB, thr, lv, rv, shell=True)
            for k, setups, ls, tol in TESTS + EXTRA:
                for s in setups:
                    m = mask_for(sig[s], ls, tol)
                    g = sig[s]["g"][m]
                    A_ = acc[k][s]
                    A_["n"][si], A_["gross"][si], A_["net"][si] = len(g), g.sum(), (g - of19.FEE).sum()
                    A_["win"][si] = (g - of19.FEE > 0).sum()
                    lg = sig[s]["side"][m] > 0
                    A_["long_n"][si], A_["long_net"][si] = lg.sum(), (g[lg] - of19.FEE).sum()
                    ms = mask_for(sh[s], ls, tol)
                    A_["sn"][si], A_["sgross"][si] = ms.sum(), sh[s]["g"][ms].sum()
                    if k == "primary":
                        for lt_, gg in zip(sig[s]["lt"][m], g):
                            c = types.setdefault((s, lt_), [0, 0.0])
                            c[0] += 1
                            c[1] += gg - of19.FEE
                        np.add.at(tod, np.minimum((sig[s]["ft"][m] // of19.BUCKET).astype(int), 12), 1)
            print(f"{d}: scored ({si + 1}/{nsc})", flush=True)
        hist.append(of19.day_pass1(T, AB, F))
        prevT = T
    # ---- statistics
    idx = {b: of19.block_idx(nsc, b, np.random.default_rng(of19.SEED)) for b in of19.BLOCKS}
    half = nsc // 2

    def pooled(k, f):
        return sum(acc[k][s][f] for s in SETUPS)

    def diff_weighted(k, setups):
        """sum_s w_s (setup_s - shell_s) per trade (w_s = setup s's share of the test's trades), and its joint-day block SE."""
        tot = sum(acc[k][s]["n"].sum() for s in setups)
        w = {s: acc[k][s]["n"].sum() / tot for s in setups}

        def val(I=None):
            v = 0.0
            for s in setups:
                A_ = acc[k][s]
                if I is None:
                    n, gs, sn, sg = (A_[f].sum() for f in ("n", "gross", "sn", "sgross"))
                else:
                    n, gs, sn, sg = (A_[f][I].sum(1) for f in ("n", "gross", "sn", "sgross"))
                v = v + w[s] * (np.where(n > 0, gs / np.maximum(n, 1), np.nan) - np.where(sn > 0, sg / np.maximum(sn, 1), np.nan))
            return v
        point = float(val())
        ses = [float(np.nanstd(val(I), ddof=1)) for I in idx.values()]
        return point, max(ses)

    R = {}
    for k, setups, ls, tol in TESTS + EXTRA:
        n = pooled(k, "n")
        N = n.sum()
        r = {"setups": setups, "levels": ls, "tol": tol, "trades": int(N), "shell_trades": int(pooled(k, "sn").sum())}
        if N > 1:
            net = pooled(k, "net").sum() / N
            se = of19.ratio_se(pooled(k, "net"), n, idx)
            n1, n2 = n[:half].sum(), n[half:].sum()
            r.update(net=net, gross=pooled(k, "gross").sum() / N, se=se, t=net / se if se > 0 else float("nan"),
                     win=pooled(k, "win").sum() / N, h1=pooled(k, "net")[:half].sum() / n1 if n1 else float("nan"),
                     h2=pooled(k, "net")[half:].sum() / n2 if n2 else float("nan"), stress=net - of19.STRESS,
                     shell_net=(pooled(k, "sgross").sum() / pooled(k, "sn").sum() - of19.FEE) if pooled(k, "sn").sum() else float("nan"))
            r["diff"], r["diff_se"] = diff_weighted(k, [s for s in setups if acc[k][s]["n"].sum() > 0])
            ln = pooled(k, "long_n").sum()
            r["long_net"] = pooled(k, "long_net").sum() / ln if ln else float("nan")
            r["short_net"] = (pooled(k, "net").sum() - pooled(k, "long_net").sum()) / (N - ln) if N > ln else float("nan")
        else:
            r.update({f: float("nan") for f in ("net", "gross", "se", "t", "win", "h1", "h2", "stress", "shell_net", "diff", "diff_se",
                                                 "long_net", "short_net")})
        R[k] = r

    def edge(r, bar):
        return (r["trades"] > 1 and r["net"] > 0 and r["t"] >= bar and r["h1"] > 0 and r["h2"] > 0 and r["stress"] > 0
                and r["diff"] >= 2 * r["diff_se"])

    n_ = of19.n_
    L = []
    w = L.append
    w("OF20: absorption and exhaustion at key levels, against a plain touch of the same level (docs/ai/OF20_LEVELS_PLAN.md)")
    w("Generated by experiments/of20_atlevels.py (OF19's data, signals, fills and statistics; levels experiments/of20_levels.py,")
    w("judged at each setup's formation time).")
    w(f"scored days {nsc} ({names[of19.WARM]} .. {names[-1]}; halves split before {names[of19.WARM + half]}); setups: absorption = OF19 P2 W 15 "
      f"target 4, exhaustion = OF19 P3 N 1 e 5; primary bar t >= {T_PRI:g}, secondary M = {M_SEC}, t_req {T_SEC:.2f}; an edge also needs "
      "setup - shell >= 2 x its block SE")
    w("")
    w(f"  {'test':<24}{'trades':>7}{'win':>6}{'gross':>8}{'net':>8}{'se':>6}{'t':>7}{'half1':>8}{'half2':>8}{'long/short':>15}"
      f"{'shell n':>9}{'shell net':>10}{'set-shell':>10}{'(se)':>6}  edge")
    for k, setups, ls, tol in TESTS + EXTRA:
        r = R[k]
        bar = T_PRI if k == "primary" else (T_SEC if (k, setups, ls, tol) in TESTS else None)
        tag = ("EDGE" if bar is not None and edge(r, bar) else "-") if bar is not None else "(descriptive)"
        w(f"  {k:<24}{r['trades']:>7}{n_(r['win'], 2, False):>6}{n_(r['gross']):>8}{n_(r['net']):>8}{n_(r['se'], 2, False):>6}"
          f"{n_(r['t'], 1):>7}{n_(r['h1']):>8}{n_(r['h2']):>8}{n_(r['long_net']) + '/' + n_(r['short_net']):>15}{r['shell_trades']:>9}"
          f"{n_(r['shell_net']):>10}{n_(r['diff']):>10}{n_(r['diff_se'], 2, False):>6}  {tag}")
    w("  (set-shell = setup minus shell gross per trade, weighted by the setup mix for the primary; shell = the same setup without its")
    w("   footprint condition at the same levels and tolerance, same exits)")
    w("")
    for s in SETUPS:
        at, aw = R[f"{s} any tol 2"], R[f"{s} away"]
        if math.isfinite(at["diff"]) and math.isfinite(aw["diff"]):
            se = math.hypot(at["diff_se"], aw["diff_se"])
            w(f"interaction ({s}): (setup - shell) at levels {n_(at['diff'])} minus away from all levels {n_(aw['diff'])} = "
              f"{n_(at['diff'] - aw['diff'])} (SE ~{se:.2f}, from the two block SEs; descriptive)")
    w("primary by nearest level type (trades, net per trade): " + "; ".join(
        f"{s} {lt}: {c[0]}, {n_(c[1] / c[0])}" for (s, lt), c in sorted(types.items()) if c[0]))
    w("primary trades by formation time, 30-min buckets from 08:30: " + "/".join(str(int(x)) for x in tod))
    w("power (net per trade a test would need): " + "; ".join(
        f"{k} {(T_PRI if k == 'primary' else T_SEC) * R[k]['se']:.2f}" for k, *_ in TESTS if math.isfinite(R[k]["se"])))
    flags = [k for k, r in R.items() if (math.isfinite(r["t"]) and r["t"] > 6) or (r["trades"] >= 30 and r["win"] > 0.75)]
    w(f"red flags (t > 6, or a win rate > 0.75 on 30+ trades): {len(flags)} {flags if flags else ''}")
    d0, lv0 = level_log[0]
    w(f"levels on the first scored day ({d0}): " + ", ".join(f"{k} {v}" for k, v in lv0.items()))
    w("")
    p = R["primary"]
    passed = [k for k, setups, ls, tol in TESTS if edge(R[k], T_PRI if k == "primary" else T_SEC)]
    shell_only = [k for k, setups, ls, tol in TESTS if R[k]["trades"] > 1 and R[k]["net"] > 0
                  and R[k]["t"] >= (T_PRI if k == "primary" else T_SEC) and not edge(R[k], T_PRI if k == "primary" else T_SEC)]
    if passed:
        w(f"DECISION (plan section 5): edge criteria met by {passed}. Not a claim: OF20b, one confirmation on the next 40 front-month RTH days.")
    else:
        w(f"DECISION (plan section 5): no edge. The primary (both setups within 2 ticks of any level) nets {n_(p['net'])} per trade "
          f"(t {n_(p['t'], 2)}), setup minus shell {n_(p['diff'])} (SE {n_(p['diff_se'], 2, False)}); at these levels the setups net no more")
        w(f"than {T_PRI * p['se']:.2f} ticks per trade (the primary's bound); smaller edges are not excluded."
          + (f" Net and t passed but not the shell condition for {shell_only}: the touch, not the footprint condition, carries it." if shell_only else ""))
    txt = "\n".join(L) + "\n"
    Path(a.out).write_text(txt, encoding="utf-8")
    print(txt)


if __name__ == "__main__":
    main()
