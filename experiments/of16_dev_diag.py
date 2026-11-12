#!/usr/bin/env python3
"""of16_dev_diag.py -- OF16 phase 1 item 5 (docs/ai/OF16_REAL_ES_PLAN.md section 3): dev-day
diagnostics on recorded ES. DEV DAYS ONLY (the manifest's dev role: 2026-09-14..17, ESU26 back
month after the roll; 2026-11-03, ESZ26): diagnostics, not a score. No OF15 agent's P&L is
computed; agents only appear in (a)'s action-sensitivity check.

usage (one command; experiments/of16_regen.sh runs it from a module snapshot through the kit):
  PYTHONPATH=python RAGOF_DIR=<dir with _ragof> .venv/Scripts/python.exe experiments/of16_dev_diag.py
      [--out experiments/of16_dev_diagnostics.txt] [--sens-windows 40]

Process role: the dev tapes are opened through ragrl.real (C++ "real" block, logged as real_dev), so
this process is a scorer; the simulator side uses the visualization block only (viz: look, never
select), episodes 960,016,000,000 + [0, 200) (fresh, recorded in docs/ai/NEXT_SESSION.md), with the
OF15 v4 env settings (experiments/configs/of15_ppo_v4.json: discovery_v4, warm-up 1050 s, decide
1 s, 600 s episodes, latency 1 step, $1.50 per side).

(a) covariate shift: the 16 observations at every decision of every played dev window (flat policy,
    VecReplayEnv, warm-up 1050 s as score-real) against the 1-99% range of the same observation on the
    200 sim episodes (flat, VecScalpEnv); median and IQR both sides; features with > 5% outside, and for
    each sealed OF15 run the share of its ensemble's flat-state decisions that change when a flagged
    feature is set to its sim median (a subsample of windows; observations of the flat run, so
    position 0: entry decisions only).
(b) baselines on the dev windows: flat (must be exactly 0), hold (long from the first decision, closed at
    the window end), delta30 and the L1-imbalance rule (ragrl.score_real), net / gross / orders per window;
    and the market statistics OF12/OF13 measured on real ES (spread, L1 size, 1-min return sd).
(c) I1 markout: at each decision with a nonzero L1 imbalance (the imbalance rule's threshold: it enters on
    any nonzero sign), the mid move 100 ms, 1 s and 5 s after the fill bucket (latency 1 bucket), in the
    imbalance's direction, against the half-spread at the fill and the round-trip cost (spread + $3 of
    fees); by quintile of |imbalance|. Real: the tape's buckets (ragrl.real.load_buckets, checked
    against the observation's feature 6). Sim: the same episodes at decide 0.1 s, flat (mid paths), and a
    probe playing the imbalance rule at 1 s decisions in 100 ms steps for the half-spread actually paid.
SEs: window-clustered (each window's sum about the overall mean), since decisions in a window correlate.
"""
import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(k, "1")
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "python"))
import ragrl  # noqa: E402,F401  (WMI workaround before torch / lightgbm)
import numpy as np  # noqa: E402

from ragrl import config, native, real, score_real, stats  # noqa: E402
from ragrl.policies import Ensemble, FrozenPolicy, load_policy  # noqa: E402

VIZ_SEED0, VIZ_COUNT = 960_016_000_000, 200
OBS = ["delta5", "delta30", "delta120", "ofi5", "ofi30", "relvol30", "imb_l1", "imb_l5", "ret5", "ret30", "ret120", "vwap_dist",
       "absorb", "position", "open_pnl", "time_left"]
POLICY_STATE = {13, 14, 15}  # set by the policy / the clock, not the market
SHIFT_BAR = 0.05
HORIZONS = {"100 ms": 1, "1 s": 10, "5 s": 50}
TICK_VALUE = 12.5
WIN, WARM, STEPS = 6000, 10500, 600  # buckets per window, warm-up buckets, decisions per window


class Hold(FrozenPolicy):
    type = "diag_hold"
    score_kind = "vote"

    def step(self, obs):
        a = np.full(len(obs), 2, dtype=np.int64)
        return a, score_real._onehot(a)


def cluster_mean(x, groups):
    """Mean and window-clustered SE."""
    x, g = np.asarray(x, dtype=np.float64), np.asarray(groups)
    n = len(x)
    if n < 2:
        return float("nan"), float("nan")
    m = x.mean()
    _, inv = np.unique(g, return_inverse=True)
    s = np.bincount(inv, weights=x - m)
    k = len(s)
    if k < 2:  # one window: no clustered SE
        return float(m), float("nan")
    se = math.sqrt((s * s).sum() * k / max(1, k - 1)) / n
    return float(m), se


def fmt(m, se=None, nd=3):
    if m is None or (isinstance(m, float) and math.isnan(m)):
        return "n/a"
    return f"{m:+.{nd}f}" + (f" +- {se:.{nd}f}" if se is not None and not math.isnan(se) else "")


# ---------------------------------------------------------------- real side
def real_side(m, warm):
    t0 = time.time()
    tapes = real.open_tapes("dev", reason="OF16 phase 1 item 5: dev-day diagnostics (experiments/of16_dev_diag.py)")
    R = native.load()
    rp = score_real.replay_params(warm)
    days = {}
    for d, (tape, side) in tapes.items():
        n = score_real.windows_on_tape(tape, rp)
        played, excluded = real.clean_windows(n, side, rp.window_seconds, rp.warmup_seconds)
        rec = {"tape": tape, "side": side, "windows": n, "played": played, "excluded": excluded}
        if played:
            env = R.VecReplayEnv(tape, rp, played)
            o = env.reset()
            seq = []
            for _ in range(STEPS):
                seq.append(o)
                o, _, done = env.step(np.ones(len(played), dtype=np.int64))
            assert done[0]
            rec["obs"] = np.stack(seq)  # [600, W, 16]
            rec["baselines"] = {k: score_real.play(p, tape, rp, played) for k, p in
                                {"flat": score_real.Flat(), "hold": Hold(), "delta30": score_real.Delta30(),
                                 "imbalance_l1": score_real.ImbalanceL1()}.items()}
            b = real.load_buckets(side["csv"])
            if len(b["bid"]) != tape.size:
                raise RuntimeError(f"{d}: {len(b['bid'])} Python buckets, TickTape has {tape.size}")
            rec["buckets"] = b
        days[d] = rec
    return days, rp, time.time() - t0


def real_markouts(days):
    rows = {"I": [], "win": [], "day": [], "half": [], "realized": [], **{h: [] for h in HORIZONS}}
    mstats = {"spread1": [], "l1": [], "ret1m": [], "day": []}
    check = 0.0
    wid = 0
    for d, r in days.items():
        if "buckets" not in r:
            continue
        b = r["buckets"]
        bid, ask = b["bid"].astype(np.float64), b["ask"].astype(np.float64)
        mid = 0.5 * (bid + ask)
        b1, a1 = b["b1"].astype(np.float64), b["a1"].astype(np.float64)
        imb = np.where(b1 + a1 > 0, (b1 - a1) / np.maximum(1, b1 + a1), 0.0)
        for wi, k in enumerate(r["played"]):
            pos = k * WIN + WARM + 10 * np.arange(STEPS)  # fill buckets; decisions see bucket pos - 1
            I = imb[pos - 1]
            check = max(check, float(np.abs(I - r["obs"][:, wi, 6] / 2).max()))
            ok = pos + 50 < len(mid)
            pos, I = pos[ok], I[ok]
            nz = I != 0
            sg = np.sign(I[nz])
            f = pos[nz]
            rows["I"].append(I[nz])
            rows["win"].append(np.full(len(f), wid))
            rows["day"].append(np.full(len(f), d))
            wid += 1
            half = 0.5 * (ask[f] - bid[f])
            rows["half"].append(half)
            rows["realized"].append(sg * (mid[f + 10] - mid[f]) - half - 0.5 * (ask[f + 10] - bid[f + 10]))
            for h, n in HORIZONS.items():
                rows[h].append(sg * (mid[f + n] - mid[f]))
            span = slice(k * WIN + WARM, k * WIN + WARM + WIN)
            mstats["spread1"].append(float(((ask[span] - bid[span]) == 1).mean()))
            mstats["l1"].append(np.concatenate([b1[span], a1[span]]))
            marks = mid[k * WIN + WARM - 1 + 600 * np.arange(11)]
            mstats["ret1m"].append(np.diff(marks))
            mstats["day"].append(d)
    return {k: np.concatenate(v) for k, v in rows.items()}, mstats, check


# ---------------------------------------------------------------- sim side (viz block)
def sim_side(env_cfg):
    t0 = time.time()
    R = native.load()
    first = VIZ_SEED0 - int(native.block("viz")["base"])
    src = lambda: R.EpisodeSource.block("viz", first, VIZ_COUNT)  # noqa: E731
    p1 = config.make_params(env_cfg)
    e1 = R.VecScalpEnv(p1, src(), n_envs=VIZ_COUNT, threads=1)
    o = e1.reset()
    seq = []
    for _ in range(STEPS):
        seq.append(o)
        o, _, done = e1.step(np.ones(VIZ_COUNT, dtype=np.int64))
    assert done[0]
    obs1 = np.stack(seq)
    fine = {**env_cfg, "decide_seconds": 0.1}
    pf = config.make_params(fine)
    # flat at 100 ms: the same episodes' mid path and observations every bucket
    e2 = R.VecScalpEnv(pf, src(), n_envs=VIZ_COUNT, threads=1, privileged=True)
    o = e2.reset()
    obs10, mids = [], []
    for j in range(WIN):
        if j % 10 == 0:
            obs10.append(o)
        o, _, _ = e2.step(np.ones(VIZ_COUNT, dtype=np.int64))
        mids.append(e2.final_priv[:, 1] if j == WIN - 1 else e2.priv[:, 1])
    obs10, mids = np.stack(obs10), np.stack(mids)  # [600, N, 16], [6000, N]
    same = float(np.abs(obs10[:, :, :13] - obs1[:, :, :13]).max())
    # probe: the imbalance rule at 1 s decisions, stepped in 100 ms buckets (fills and their mid)
    e3 = R.VecScalpEnv(pf, src(), n_envs=VIZ_COUNT, threads=1, privileged=True)
    o = e3.reset()
    half, realized, wins, rwins = [], [], [], []
    entry = np.full(VIZ_COUNT, np.nan)
    sgn = np.zeros(VIZ_COUNT)
    for j in range(WIN):
        pos = o[:, 13]
        if j % 10 == 0:
            x = o[:, 6]
            enter = np.where(x > 0, 2, np.where(x < 0, 0, 1))
            a = np.where(pos == 0, enter, 1).astype(np.int64)
        else:
            a = score_real.keep_action(o)
        o, _, _ = e3.step(a)
        pr = e3.final_priv if j == WIN - 1 else e3.priv
        if j % 10 == 0:
            new = (pos == 0) & (a != 1)
            fill = pr[:, 4]
            closing = (pos != 0) & ~np.isnan(entry)
            if closing.any():
                realized.append(sgn[closing] * (fill[closing] - entry[closing]))
                rwins.append(np.nonzero(closing)[0])
                entry[closing] = np.nan
            if new.any():
                s = np.where(a[new] == 2, 1.0, -1.0)
                half.append(s * (fill[new] - pr[new, 1]))
                wins.append(np.nonzero(new)[0])
                entry[new], sgn[new] = fill[new], s
    probe = {"half": np.concatenate(half), "half_win": np.concatenate(wins), "realized": np.concatenate(realized),
             "realized_win": np.concatenate(rwins)}
    # markouts on the flat path
    rows = {"I": [], "win": [], **{h: [] for h in HORIZONS}}
    for s in range(STEPS):
        j = 10 * s
        if j + 50 >= WIN:
            break
        I = obs10[s, :, 6] / 2
        nz = I != 0
        sg = np.sign(I[nz])
        rows["I"].append(I[nz])
        rows["win"].append(np.nonzero(nz)[0])
        for h, n in HORIZONS.items():
            rows[h].append(sg * (mids[j + n, nz] - mids[j, nz]))
    return {"obs": obs1, "markout": {k: np.concatenate(v) for k, v in rows.items()}, "probe": probe, "same_episodes_max_diff": same,
            "first": first}, time.time() - t0


# ---------------------------------------------------------------- sensitivity
def sealed_runs():
    out = []
    for r in sorted((REPO / "build-avx512" / "rl" / "runs").glob("of15_*")):
        if (r / "sealed.json").exists():
            out.append(r)
    return out


def sensitivity(real_obs_sub, flagged, sim_median):
    """Per sealed run: its ensemble's action shares on the flat-state real observations, and the share of
    decisions whose action changes when each flagged feature is set to its sim median."""
    out = {}
    for run in sealed_runs():
        t = time.time()
        cfg = json.loads((run / "config.json").read_text(encoding="utf-8"))
        ens = Ensemble([load_policy(run / "seeds" / f"seed{s}") for s in cfg["train"]["seeds"]])

        def acts(obs):
            ens.reset(obs.shape[1])
            return np.stack([ens.step(np.ascontiguousarray(obs[t]))[0] for t in range(obs.shape[0])])

        base = acts(real_obs_sub)
        rec = {"kind": cfg["agent"]["kind"], "market": cfg["env"]["market"], "nonflat": float((base != 1).mean()), "changed": {}}
        for k in flagged:
            o = real_obs_sub.copy()
            o[:, :, k] = sim_median[k]
            rec["changed"][k] = float((acts(o) != base).mean())
        rec["seconds"] = round(time.time() - t, 1)
        out[run.name] = rec
    return out


# ---------------------------------------------------------------- report
def of13_line(label):
    f = REPO / "experiments" / "of13_final_real_vs_v4.txt"
    for line in f.read_text(encoding="utf-8").splitlines():
        if line.startswith(label):
            return line.strip()
    return f"({label}: not found in {f.name})"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", default=str(REPO / "experiments" / "of16_dev_diagnostics.txt"))
    ap.add_argument("--sens-windows", type=int, default=40)
    a = ap.parse_args(argv)
    T0 = time.time()
    try:
        import torch
        torch.set_num_threads(1)
    except ImportError:
        pass
    m = real.load_manifest()
    env_cfg = config.load(REPO / "experiments" / "configs" / "of15_ppo_v4.json")["env"]
    warm = score_real.check_episodes(env_cfg, m)
    days, rp, t_real = real_side(m, warm)
    sim, t_sim = sim_side(env_cfg)
    mk, mstats, align = real_markouts(days)
    played = [(d, r) for d, r in days.items() if "obs" in r]
    real_obs = np.concatenate([r["obs"].reshape(-1, 16) for _, r in played])
    sim_obs = sim["obs"].reshape(-1, 16)
    p1, p99 = np.percentile(sim_obs, 1, axis=0), np.percentile(sim_obs, 99, axis=0)
    sim_med = np.median(sim_obs, axis=0)
    share = ((real_obs < p1) | (real_obs > p99)).mean(axis=0)
    flagged = [k for k in range(16) if share[k] > SHIFT_BAR and k not in POLICY_STATE]
    # sensitivity subsample: evenly spaced played windows over all days
    allw = [(d, wi) for d, r in played for wi in range(len(r["played"]))]
    pick = [allw[i] for i in np.linspace(0, len(allw) - 1, min(a.sens_windows, len(allw))).round().astype(int)]
    sub = np.stack([days[d]["obs"][:, wi, :] for d, wi in pick], axis=1)
    t = time.time()
    sens = sensitivity(sub, flagged, sim_med) if flagged else {}
    t_sens = time.time() - t

    L = []
    w = L.append
    w("OF16 phase 1 item 5 -- DEV DAYS: DIAGNOSTICS, NOT A SCORE")
    w("Generated by experiments/of16_regen.sh (experiments/of16_dev_diag.py). docs/ai/OF16_REAL_ES_PLAN.md section 3. Aggregates only.")
    w("No OF15 agent P&L is computed here (phase 2); agents appear only in (a)'s action-sensitivity check.")
    w("")
    w("Data")
    w(f"  manifest {real.MANIFEST.relative_to(REPO).as_posix()} sha256 {real.manifest_sha256()[:16]}; module {native.module_file().name} "
      f"sha256 {native.sha256_file(native.module_file())[:16]}")
    for d, r in days.items():
        s = r["side"]
        c = s["counts"]
        w(f"  {d} {s['contract']} ({real.subset_of(d, m)}): {s['span']['start_ct']}-{s['span']['end_ct']} CT, {c['trades']} trades, "
          f"{c['quotes']} quotes, book gaps > 1 s: {s['gaps_over_1s']['count']} (max {s['gaps_over_1s']['max_s']:.1f} s), "
          f"trades outside the bid/ask {s['trades_outside_quote']['share']:.2%} (book fresh {s['trades_outside_quote']['share_book_fresh']:.2%}); "
          f"windows {r['windows']}, played {len(r['played'])}, excluded {len(r['excluded'])}"
          + (f" {[e['book_gap_s'] for e in r['excluded']]}" if r["excluded"] else "") + f"; csv sha256 {s['csv_sha256'][:12]}")
    w("  CAVEATS: 2026-09-14..17 are ESU26 AFTER the roll to ESZ26 (the expiring back month): a thinner, less active market than any")
    w("  test day, which is why they are dev days. 2026-11-03 (ESZ26, front month) has ONE played window: its book recording stops")
    w("  09:00:34-09:10:44 CT (610 s), and with the 1050 s warm-up only window 0 (08:30-09:00 incl. warm-up) avoids it.")
    w(f"  replay: {dict(decide=rp.decide_seconds, window=rp.window_seconds, warmup=rp.warmup_seconds, latency_buckets=rp.latency_buckets, fee=rp.fee_dollars)}"
      f" (= score-real; the OF15 runs' env)")
    w(f"  simulator: {env_cfg['market']}, warm-up {env_cfg['warmup_seconds']} s, decide {env_cfg['decide_seconds']} s, "
      f"{env_cfg['episode_seconds']} s episodes; visualization block (look only) episodes {VIZ_SEED0:,} + [0, {VIZ_COUNT}) "
      f"(block-relative first {sim['first']:,}); the decide-0.1 s replay of the same episodes matches the 1 s one on features 0-12 "
      f"(max |diff| {sim['same_episodes_max_diff']:.3g})")
    w(f"  alignment check: L1 imbalance from the Python buckets vs the observation's feature 6 / 2 at every decision: max |diff| {align:.3g}")
    w("")
    w(f"(a) Covariate shift: share of real dev observations ({len(real_obs):,} = {len(allw)} windows x 600 decisions, flat policy) outside the")
    w(f"    sim's 1-99% range ({len(sim_obs):,} = {VIZ_COUNT} episodes x 600). Flag: > {SHIFT_BAR:.0%}. 13-15 are policy / clock state (flat: equal).")
    w(f"{'k':>3} {'feature':10} {'outside':>8} {'sim p1':>8} {'sim p99':>8} | {'real median':>11} {'real IQR':>17} | {'sim median':>10} {'sim IQR':>17}  flag")
    for k in range(16):
        rq, sq = np.percentile(real_obs[:, k], [25, 50, 75]), np.percentile(sim_obs[:, k], [25, 50, 75])
        w(f"{k:>3} {OBS[k]:10} {share[k]:8.1%} {p1[k]:8.3f} {p99[k]:8.3f} | {rq[1]:11.3f} [{rq[0]:7.3f},{rq[2]:7.3f}] | "
          f"{sq[1]:10.3f} [{sq[0]:7.3f},{sq[2]:7.3f}]  {'OVER 5%' if k in flagged else ''}")
    by_subset = {}
    for d, r in played:
        by_subset.setdefault(real.subset_of(d, m), []).append(r["obs"].reshape(-1, 16))
    w("    outside share by dev group: " + "; ".join(
        f"{g}: " + ", ".join(f"{OBS[k]} {((np.concatenate(v)[:, k] < p1[k]) | (np.concatenate(v)[:, k] > p99[k])).mean():.1%}" for k in flagged)
        for g, v in by_subset.items()) if flagged else "    no feature over 5%")
    if flagged:
        w("")
        w(f"    Action sensitivity (flat-state real observations of {len(pick)} evenly spaced played windows, {len(pick) * 600:,} decisions;")
        w("    each sealed OF15 run's 12-seed ensemble, policies loaded from RUN/seeds; no P&L): 'non-flat' = share of decisions that")
        w("    enter (action != flat); then the share whose action changes when the feature is set to its sim median.")
        w(f"    {'run':48} {'kind':15} {'non-flat':>8} " + " ".join(f"{OBS[k]:>9}" for k in flagged))
        for name, rec in sens.items():
            w(f"    {name:48} {rec['kind']:15} {rec['nonflat']:8.1%} " + " ".join(f"{rec['changed'][k]:9.1%}" for k in flagged))
        dep = {k: [n.split('-')[0] for n, r in sens.items() if r["changed"][k] > 0.05] for k in flagged}
        for k in flagged:
            w(f"    {OBS[k]}: ensembles whose actions change on > 5% of decisions: {', '.join(dep[k]) or 'none'}")
        w("    (these are extrapolating on real data in that feature; they are still scored in phase 2)")
    w("")
    w("(b) Baselines on the dev windows (ticks per 600 s window, latency 1 bucket, $1.50 per side; gross = net + fees).")
    w(f"    {'policy':13} {'group':6} {'windows':>7} {'net':>18} {'gross':>18} {'orders':>7}")
    groups = {"sep": [d for d, _ in played if real.subset_of(d, m) == "sep"], "nov": [d for d, _ in played if real.subset_of(d, m) == "nov"]}
    groups["all"] = [d for d, _ in played]
    base_rows = {}
    for pol in ("flat", "hold", "delta30", "imbalance_l1"):
        for g, ds in groups.items():
            if not ds:
                continue
            res = {f: np.concatenate([days[d]["baselines"][pol][f] for d in ds]) for f in ("net_pnl_ticks", "orders", "fees_ticks")}
            s, gs = stats.summary(res), stats.summary(stats.gross(res))
            base_rows[(pol, g)] = s
            w(f"    {pol:13} {g:6} {s['n']:7d} {fmt(s['mean'], s['se'], 2):>18} {fmt(gs['mean'], gs['se'], 2):>18} {s['orders']:7.1f}")
    flat_all = np.concatenate([days[d]["baselines"]["flat"]["net_pnl_ticks"] for d in groups["all"]])
    w(f"    flat is exactly 0 on every window: {bool((flat_all == 0).all())}")
    w("    OF13 v4 (sim, warm-up 1050, episodes 960,013,000,000+i; experiments/of13_final_baselines_w1050.txt), for scale:")
    for line in (REPO / "experiments" / "of13_final_baselines_w1050.txt").read_text(encoding="utf-8").splitlines()[2:4] + \
            [x for x in (REPO / "experiments" / "of13_final_baselines_w1050.txt").read_text(encoding="utf-8").splitlines() if x.startswith("delta 30 s > 1.5")][:1]:
        w("      " + line)
    w("    Market statistics on the played windows (100 ms buckets of the windows, warm-up excluded) vs OF12/OF13 real ES:")
    sp = np.array(mstats["spread1"])
    l1 = np.concatenate(mstats["l1"])
    r1 = np.concatenate(mstats["ret1m"])
    w(f"      spread = 1 tick: {sp.mean():.2%} of buckets (per window min {sp.min():.1%}); L1 size median {np.median(l1):.0f} lots "
      f"(10th {np.percentile(l1, 10):.0f}, 90th {np.percentile(l1, 90):.0f}); 1-min mid change sd {r1.std(ddof=1):.2f} ticks "
      f"({len(r1)} minutes)")
    for g, ds in groups.items():
        if g == "all" or not ds:
            continue
        idx = [i for i, d in enumerate(mstats["day"]) if d in ds]
        sg = sp[idx]
        lg = np.concatenate([mstats["l1"][i] for i in idx])
        rg = np.concatenate([mstats["ret1m"][i] for i in idx])
        w(f"      {g}: spread = 1 tick {sg.mean():.2%}, L1 median {np.median(lg):.0f}, 1-min sd {rg.std(ddof=1):.2f} ({len(idx)} windows)")
    w("      OF13 (experiments/of13_final_real_vs_v4.txt; columns: real, discovery_v4, discovery, real ETH):")
    for lab in ("spread = 1 tick (%)", "L1 size, median", "1-min return sd (ticks)"):
        w("        " + of13_line(lab))
    hold = base_rows.get(("hold", "all"))
    if hold:
        w(f"      hold's per-window sd {hold['sd']:.1f} ticks vs sqrt(10) x the 1-min sd above = {math.sqrt(10) * r1.std(ddof=1):.1f} "
          f"(a 10-min mid change; hold pays one spread + 0.24 ticks of fees)")
    w("")
    rt_fee = 2 * 1.5 / TICK_VALUE
    w("(c) I1 markout: the mid move after the fill bucket, in the direction of the L1 imbalance at the decision (ticks; window-clustered SE).")
    w(f"    The rule enters on any nonzero imbalance (threshold 0). Round-trip cost = spread at entry and exit + ${2 * 1.5:.0f} fees "
      f"({rt_fee:.2f} ticks).")
    w(f"    {'side':5} {'subset':14} {'decisions':>9} " + " ".join(f"{h:>16}" for h in HORIZONS) + f" {'half-spread':>12} {'round trip cost':>15}"
      f" {'1 s rule gross':>16}")

    def mrow(side, label, mo, sel, half, half_w, realized, realized_w):
        cols = " ".join(f"{fmt(*cluster_mean(mo[h][sel], mo['win'][sel])):>16}" for h in HORIZONS)
        hm = float(np.mean(half)) if len(half) else float("nan")
        w(f"    {side:5} {label:14} {int(sel.sum()):9d} {cols} {hm:12.3f} {2 * hm + rt_fee:15.3f} "
          f"{fmt(*cluster_mean(realized, realized_w)):>16}")

    allsel = np.ones(len(mk["I"]), dtype=bool)
    mrow("real", "|I| > 0", mk, allsel, mk["half"], mk["win"], mk["realized"], mk["win"])
    sm = sim["markout"]
    pr = sim["probe"]
    mrow("sim", "|I| > 0", sm, np.ones(len(sm["I"]), dtype=bool), pr["half"], pr["half_win"], pr["realized"], pr["realized_win"])
    for side, mo in (("real", mk), ("sim", sm)):
        a_ = np.abs(mo["I"])
        edges = np.percentile(a_, [20, 40, 60, 80])
        q = np.digitize(a_, edges)
        for i in range(5):
            sel = q == i
            lo = 0.0 if i == 0 else edges[i - 1]
            hi = edges[i] if i < 4 else 1.0
            hs = mo["half"][sel] if "half" in mo else np.array([])
            rz = mo["realized"][sel] if "realized" in mo else np.array([])
            cols = " ".join(f"{fmt(*cluster_mean(mo[h][sel], mo['win'][sel])):>16}" for h in HORIZONS)
            w(f"    {side:5} {f'Q{i + 1} |I| {lo:.2f}-{hi:.2f}':14} {int(sel.sum()):9d} {cols} "
              + (f"{hs.mean():12.3f} {2 * hs.mean() + rt_fee:15.3f} {fmt(*cluster_mean(rz, mo['win'][sel])):>16}" if len(hs) else ""))
    for g in ("sep", "nov"):
        ds = set(groups[g])
        if not ds:
            continue
        sel = np.isin(mk["day"], list(ds))
        mrow("real", f"{g} days", mk, sel, mk["half"][sel], mk["win"][sel], mk["realized"][sel], mk["win"][sel])
    w("    sim half-spread / 1 s rule gross: the probe's own entries (the rule played at 1 s decisions in 100 ms steps; its exit")
    w("    fill 1 s later); real: every decision with |I| > 0 (entry at the fill bucket's touch, exit at the touch 10 buckets later).")
    w("    OF15 (v4): +0.1..+0.3 ticks gross at 1 s for entries with the L1 imbalance; OF13 real (11-03) 1-s delta markouts:")
    for lab in ("1-s delta markout +1 s", "1-s delta markout +5 s"):
        w("        " + of13_line(lab))
    w("")
    w(f"Wall time: {time.time() - T0:.0f} s (real side {t_real:.0f} s, sim side {t_sim:.0f} s, sensitivity {t_sens:.0f} s), single-threaded.")
    text = "\n".join(L) + "\n"
    Path(a.out).write_text(text, encoding="utf-8", newline="\n")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
