#!/usr/bin/env python3
"""OF18: month-scale momentum / reversal regimes in ES at 5-30 min (docs/ai/OF18_REGIME_PLAN.md).

  python experiments/of18_regimes.py --out experiments/of18_regimes.txt [--bars build-avx512/real/bars]

Reads the 1-min front-month RTH bars of tools/scid_to_bars.py (2025-12-15 .. 2026-11-11; each npz checked
against its sidecar's sha256; nothing on or after the confirm block's first day). Walk-forward by day:
for a feature, a horizon h and a lookback W, day D's quintile thresholds and direction come from the W
used days before D only. Statistics, the decision rule and every constant are the plan's (section 5).
"""
import argparse
import hashlib
import json
import math
import sys
from datetime import date
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "python"))
from ragrl import real  # noqa: E402  (manifest: the confirm cutoff and the OF16 days)

START, END = "2025-12-15", "2026-11-11"
TICK_FROM = "2026-09-08"  # records carry bid and ask from here (the bias check)
FEATS = ["ret_5", "ret_15", "ret_30", "delta_5", "delta_15", "delta_30", "vwap_dist"]
H_TEST, H_DESC = [5, 10, 15], [30]
WS = [5, 20, 60]
PRIMARY = ("vwap_dist", 10, 20)
COST, STRESS = 1.28, 1.50
MAX_GAP_MIN = 5
FIRST_M, STEP_M, MINUTES = 30, 5, 390
WARM_DAYS = 60
BLOCKS, B, SEED = (5, 10, 21), 10_000, 0
M_SEC = len(FEATS) * len(H_TEST) * len(WS)
T_REQ = 2.0 + math.sqrt(2.0 * math.log(M_SEC))


# ---------------------------------------------------------------- data
def load_days(bars_dir: Path):
    m = real.load_manifest()
    cutoff = date.fromisoformat(m["confirm_rule"]["from"])
    if date.fromisoformat(END) >= cutoff:
        raise RuntimeError("the span reaches the confirm block")
    days, dropped = [], []
    for side_p in sorted(bars_dir.glob("*.json")):
        d = side_p.stem
        if not (START <= d <= END):
            continue
        side = json.loads(side_p.read_text(encoding="utf-8"))
        npz = bars_dir / f"{d}.npz"
        if hashlib.sha256(npz.read_bytes()).hexdigest() != side["npz_sha256"]:
            raise RuntimeError(f"{npz}: sha256 does not match its sidecar")
        if side["gap_minutes"] > MAX_GAP_MIN:
            dropped.append((d, side["gap_minutes"]))
            continue
        z = np.load(npz)
        days.append((d, side, {k: z[k].astype(np.float64) for k in z.files}))
    of16 = set(real.listed_days("seen", m)) | set(real.listed_days("dev", m))
    return days, dropped, of16


def day_samples(bars: dict, hs: list) -> dict:
    """Decisions of one day: features [n, 7], and per h the move, validity, and the mid bias terms."""
    o, h_, l_, c, v = bars["open"], bars["high"], bars["low"], bars["close"], bars["volume"]
    buy, sell = bars["buy"], bars["sell"]
    gap = v == 0
    close_at = np.r_[o[0], c]  # close_at[m] = close(m); close(0) = open of bar 1
    typ = (h_ + l_ + c) / 3.0
    cum_pv, cum_v = np.cumsum(v * typ), np.cumsum(v)
    flow = np.r_[0.0, np.cumsum(buy - sell)]  # flow[m] = sum over bars 1..m
    mid_open = (bars["open_bid"] + bars["open_ask"]) / 2.0  # NaN before TICK_FROM
    ms = np.arange(FIRST_M, MINUTES + 1, STEP_M)
    X = np.zeros((len(ms), len(FEATS)))
    for i, m in enumerate(ms):
        X[i] = [close_at[m] - close_at[m - 5], close_at[m] - close_at[m - 15], close_at[m] - close_at[m - 30],
                flow[m] - flow[m - 5], flow[m] - flow[m - 15], flow[m] - flow[m - 30],
                close_at[m] - (cum_pv[m - 1] / cum_v[m - 1] if cum_v[m - 1] > 0 else close_at[m])]
    out = {"m": ms, "X": X}
    for h in hs:
        ok = np.zeros(len(ms), bool)
        move = np.full(len(ms), np.nan)
        tp_minus_mid = np.full(len(ms), np.nan)  # (exit trade - exit mid) - (entry trade - entry mid)
        for i, m in enumerate(ms):
            if m + h > MINUTES - 1:  # exit bar m + 1 + h must be <= 390: index m + h <= 389
                continue
            if gap[m - 30:m].any() or gap[m:m + h + 1].any():
                continue
            ok[i] = True
            move[i] = o[m + h] - o[m]
            tp_minus_mid[i] = (o[m + h] - mid_open[m + h]) - (o[m] - mid_open[m])
        out[h] = {"ok": ok, "move": move, "bias": tp_minus_mid}
    return out


# ---------------------------------------------------------------- walk-forward rules
def rule_days(S: list, j: int, h: int, W: int, scored: list):
    """Per scored day: net sum, gross sum, trades, bias sum, bias trades, and the persistence pair (estimate, own)."""
    res = {k: np.zeros(len(scored)) for k in ("net", "gross", "n", "bias", "nb", "bias_mom", "nb_mom", "bias_rev", "nb_rev")}
    est, own = np.full(len(scored), np.nan), np.full(len(scored), np.nan)
    sides_by_m = []
    for si, i in enumerate(scored):
        tr = [S[k] for k in range(i - W, i)]
        x = np.concatenate([s["X"][s[h]["ok"], j] for s in tr])
        y = np.concatenate([s[h]["move"][s[h]["ok"]] for s in tr])
        lo, hi = np.percentile(x, [20, 80])
        top, bot = x > hi, x < lo
        if not top.any() or not bot.any():
            continue
        e = y[top].mean() - y[bot].mean()
        dsign = int(np.sign(e))
        s = S[i]
        ok = s[h]["ok"]
        xs, ys = s["X"][:, j], s[h]["move"]
        t_, b_ = ok & (xs > hi), ok & (xs < lo)
        if t_.any() and b_.any():
            est[si], own[si] = e, ys[t_].mean() - ys[b_].mean()
        if dsign == 0:
            continue
        side = np.where(t_, dsign, np.where(b_, -dsign, 0))
        tr_ = side != 0
        g = side[tr_] * ys[tr_]
        res["gross"][si] = g.sum()
        res["net"][si] = (g - COST).sum()
        res["n"][si] = tr_.sum()
        bb = side[tr_] * s[h]["bias"][tr_]
        fin = np.isfinite(bb)
        res["bias"][si] = bb[fin].sum()
        res["nb"][si] = fin.sum()
        mom = side[tr_] == np.sign(xs[tr_])  # the trade goes with the feature's sign (momentum) or against it (reversal)
        res["bias_mom"][si], res["nb_mom"][si] = bb[fin & mom].sum(), (fin & mom).sum()
        res["bias_rev"][si], res["nb_rev"][si] = bb[fin & ~mom].sum(), (fin & ~mom).sum()
        sides_by_m.append((s["m"][tr_], side[tr_]))
    return res, est, own, sides_by_m


def block_indices(n: int, b: int, rng) -> np.ndarray:
    k = math.ceil(n / b)
    starts = rng.integers(0, n - b + 1, size=(B, k))
    return (starts[:, :, None] + np.arange(b)[None, None, :]).reshape(B, k * b)[:, :n]


def ratio_se(num: np.ndarray, den: np.ndarray, idx: dict) -> float:
    ses = []
    for b, I in idx.items():
        dn = den[I].sum(1)
        r = np.where(dn > 0, num[I].sum(1) / np.maximum(dn, 1), np.nan)
        ses.append(float(np.nanstd(r, ddof=1)))
    return max(ses)


def corr_se(e: np.ndarray, o: np.ndarray, idx: dict) -> tuple:
    ok = np.isfinite(e) & np.isfinite(o)
    if ok.sum() < 10:
        return float("nan"), float("nan")
    c = float(np.corrcoef(e[ok], o[ok])[0, 1])
    ses = []
    for b, I in idx.items():
        E, O = e[I], o[I]
        F = np.isfinite(E) & np.isfinite(O)
        vals = []
        for r in range(0, B, 10):  # every 10th resample: the correlation is per resample (a loop), 1000 per block length
            f = F[r]
            if f.sum() > 9:
                vals.append(np.corrcoef(E[r][f], O[r][f])[0, 1])
        ses.append(float(np.nanstd(vals, ddof=1)))
    return c, max(ses)


def summarize(res: dict, idx: dict, half: int) -> dict:
    n = res["n"].sum()
    if n == 0:
        return {"trades": 0, "net": float("nan"), "gross": float("nan"), "se": float("nan"), "t": float("nan"), "h1": float("nan"),
                "h2": float("nan"), "stress": float("nan"), "bias": float("nan"), "bias_trades": 0, "after_bias": float("nan")}
    net = res["net"].sum() / n
    se = ratio_se(res["net"], res["n"], idx)
    n1, n2 = res["n"][:half].sum(), res["n"][half:].sum()
    nb = res["nb"].sum()
    bias = res["bias"].sum() / nb if nb else float("nan")
    return {"trades": int(n), "net": float(net), "gross": float(res["gross"].sum() / n), "se": se, "t": net / se if se > 0 else float("nan"),
            "h1": float(res["net"][:half].sum() / n1) if n1 else float("nan"), "h2": float(res["net"][half:].sum() / n2) if n2 else float("nan"),
            "stress": float(net - (STRESS - COST)), "bias": bias, "bias_trades": int(nb),
            "bias_mom": (res["bias_mom"].sum(), res["nb_mom"].sum()), "bias_rev": (res["bias_rev"].sum(), res["nb_rev"].sum()),
            "after_bias": float(net - bias) if nb else float("nan")}


def edge(r: dict, t_bar: float) -> bool:
    return (r["trades"] > 0 and r["net"] > 0 and r["t"] >= t_bar and r["h1"] > 0 and r["h2"] > 0 and r["stress"] > 0
            and (not math.isfinite(r["bias"]) or r["after_bias"] > 0))


def run_all(days: list, keys: list):
    S = [day_samples(b, H_TEST + H_DESC) for _, _, b in days]
    scored = list(range(WARM_DAYS, len(days)))
    rng_idx = {b: block_indices(len(scored), b, np.random.default_rng(SEED)) for b in BLOCKS}
    half = len(scored) // 2
    out = {}
    for f, h, W in keys:
        res, est, own, sides = rule_days(S, FEATS.index(f), h, W, scored)
        r = summarize(res, rng_idx, half)
        r["persist"] = corr_se(est, own, rng_idx)
        r["persist_days"] = int((np.isfinite(est) & np.isfinite(own)).sum())
        r["sides"] = sides
        out[(f, h, W)] = r
    return out, scored, half


# ---------------------------------------------------------------- report
def n_(x, d=2, sign=True):
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "-"
    return f"{x:+.{d}f}" if sign else f"{x:.{d}f}"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--bars", default=str(REPO / "build-avx512" / "real" / "bars"))
    a = ap.parse_args(argv)
    days, dropped, of16 = load_days(Path(a.bars))
    keys = [(f, h, W) for h in H_TEST + H_DESC for f in FEATS for W in WS]
    R, scored, half = run_all(days, keys)
    names = [d for d, _, _ in days]
    L = []
    w = L.append
    w("OF18: month-scale momentum / reversal regimes in ES, 5-30 min (docs/ai/OF18_REGIME_PLAN.md)")
    w("Generated by experiments/of18_regimes.py from tools/scid_to_bars.py bars (Sierra .scid, front month by RTH volume).")
    w(f"used days {len(days)} ({names[0]} .. {names[-1]}); dropped (more than {MAX_GAP_MIN} gap minutes): "
      + ", ".join(f"{d} ({g})" for d, g in dropped))
    w(f"scored days {len(scored)} ({names[scored[0]]} .. {names[scored[-1]]}; halves split before {names[scored[half]]}); "
      f"warm-up {WARM_DAYS} used days; bias-check days (bid/ask in the records, from {TICK_FROM}): "
      f"{sum(names[i] >= TICK_FROM for i in scored)}")
    w(f"cost {COST} ticks per round trip (stress {STRESS}); decisions every {STEP_M} min from minute {FIRST_M}; entry and exit at bar opens")
    w(f"SE: the largest moving-block bootstrap SE over blocks of {', '.join(map(str, BLOCKS))} scored days (B {B}, seed {SEED}), "
      f"statistic total net / total trades; secondary M = {M_SEC}, t_req = {T_REQ:.2f}; primary t >= 2")
    w("")
    p = R[PRIMARY]
    w(f"PRIMARY {PRIMARY[0]}, h {PRIMARY[1]} min, W {PRIMARY[2]} days: trades {p['trades']}, gross {n_(p['gross'])}, net {n_(p['net'])} "
      f"(SE {n_(p['se'], 2, False)}, t {n_(p['t'], 2)}), halves {n_(p['h1'])} / {n_(p['h2'])}, stress {n_(p['stress'])}, "
      f"bias {n_(p['bias'], 3)} on {p['bias_trades']} trades, net after bias {n_(p['after_bias'])}; edge: {'YES' if edge(p, 2.0) else 'no'}")
    w(f"  detectable net (2 x SE): {2 * p['se']:.2f} ticks per trade")
    w("")
    for h in H_TEST + H_DESC:
        tested = h in H_TEST
        w(f"== h = {h} min ({'tested' if tested else 'descriptive only'}) ==")
        w(f"  {'feature':<10}{'W':>4}{'trades':>8}{'gross':>8}{'net':>8}{'se':>7}{'t':>7}{'half1':>8}{'half2':>8}{'bias':>8}"
          f"{'persist r':>11}{'(se)':>7}  cand")
        for f in FEATS:
            for W in WS:
                r = R[(f, h, W)]
                c, cse = r["persist"]
                cand = tested and edge(r, T_REQ)
                w(f"  {f:<10}{W:>4}{r['trades']:>8}{n_(r['gross']):>8}{n_(r['net']):>8}{n_(r['se'], 2, False):>7}{n_(r['t'], 1):>7}"
                  f"{n_(r['h1']):>8}{n_(r['h2']):>8}{n_(r['bias'], 2):>8}{n_(c, 3):>11}{n_(cse, 3, False):>7}  {'YES' if cand else '-'}")
        pdays = [R[(f, h, W)]["persist_days"] for f in FEATS for W in WS]
        w(f"  persist r is over the {min(pdays)}-{max(pdays)} scored days on which day D reaches both trailing tails (see the note below)")
        ses = [R[(f, h, W)]["se"] for f in FEATS for W in WS if math.isfinite(R[(f, h, W)]["se"])]
        w(f"  power: median t_req x SE = {T_REQ * float(np.median(ses)):.2f} ticks per trade")
        w("")
    # the bias by trade type, pooled over the tested rules
    bm = [R[k]["bias_mom"] for k in R if k[1] in H_TEST]
    br = [R[k]["bias_rev"] for k in R if k[1] in H_TEST]
    sm, nm = sum(x[0] for x in bm), sum(x[1] for x in bm)
    sr, nr = sum(x[0] for x in br), sum(x[1] for x in br)
    w(f"trade-price bias by trade type (tested rules, bias-check days): momentum {n_(sm / nm if nm else float('nan'), 3)} on {int(nm)} trades, "
      f"reversal {n_(sr / nr if nr else float('nan'), 3)} on {int(nr)} trades; the primary: momentum "
      f"{n_(p['bias_mom'][0] / p['bias_mom'][1] if p['bias_mom'][1] else float('nan'), 3)}, reversal "
      f"{n_(p['bias_rev'][0] / p['bias_rev'][1] if p['bias_rev'][1] else float('nan'), 3)}")
    w("note on persist r (OF18 review): 'own' (day D's top-minus-bottom move under the trailing thresholds) exists only on days that")
    w("reach both trailing tails, which must cross the VWAP, so it is negative by construction there (a no-regime random walk with the")
    w("real days' volatility gives nearly the same values) and scales with the day's volatility; with volatility clustering that makes")
    w("the negative vwap_dist W20 values. Scaled by volatility they are within a day-shuffle null (SD ~0.15): no evidence of persistence")
    w("or anti-persistence either way, and the printed SEs understate that null.")
    # time of day of the primary's trades
    mm = np.concatenate([x[0] for x in p["sides"]]) if p["sides"] else np.zeros(0)
    w("primary trades by time of day (decision minute after 08:30 -> CT hour): " + ", ".join(
        f"{8 + (30 + lo) // 60:02d}:{(30 + lo) % 60:02d}-{8 + (30 + lo + 60) // 60:02d}:{(30 + lo + 60) % 60:02d} {int(((mm >= lo) & (mm < lo + 60)).sum())}"
        for lo in range(0, 390, 60)))
    # sensitivity without the 13 OF16 days
    sec = sorted(((f, h, W) for f in FEATS for h in H_TEST for W in WS), key=lambda k: -(R[k]["t"] if math.isfinite(R[k]["t"]) else -1e9))
    sens_keys = [PRIMARY] + [k for k in sec if k != PRIMARY][:5]
    days2 = [x for x in days if x[0] not in of16]
    R2, sc2, _ = run_all(days2, sens_keys)
    w(f"sensitivity without the {len([d for d in names if d in of16])} OF16 days ({len(sc2)} scored days): " + "; ".join(
        f"{k[0]} h{k[1]} W{k[2]} net {n_(R2[k]['net'])} t {n_(R2[k]['t'], 2)} (all days: {n_(R[k]['net'])}, t {n_(R[k]['t'], 2)})" for k in sens_keys))
    # red flags and the decision
    flags = [(k, R[k]) for k in R if (math.isfinite(R[k]["t"]) and R[k]["t"] > 8) or (k[1] == 5 and R[k]["gross"] > 3)]
    w("")
    w(f"red flags (t > 8, or gross > 3 ticks at h = 5): {len(flags)}" + "".join(f"\n  {k}: net {n_(r['net'])}, t {n_(r['t'], 1)}" for k, r in flags))
    cands = [k for k in R if k[1] in H_TEST and edge(R[k], T_REQ)]
    prim = edge(p, 2.0)
    w("")
    if prim or cands:
        w("DECISION (plan section 5): " + ("primary edge; " if prim else "") + (f"secondary candidates {cands}; " if cands else "")
          + "not a claim: OF18b, one confirmation on the next 40 front-month RTH days from 2026-11-12, at the recorded touch.")
    else:
        xh = {h: T_REQ * float(np.median([R[(f, h, W)]["se"] for f in FEATS for W in WS])) for h in H_TEST}
        w("DECISION (plan section 5): no primary edge and no secondary candidate. No walk-forward regime rule on these 7 trend")
        w("features nets more than " + ", ".join(f"{xh[h]:.2f} ticks per trade at {h} min" for h in H_TEST)
          + f" (the primary: more than {2 * p['se']:.2f} at 10 min); smaller edges are not excluded.")
    txt = "\n".join(L) + "\n"
    Path(a.out).write_text(txt, encoding="utf-8")
    print(txt)


if __name__ == "__main__":
    main()
