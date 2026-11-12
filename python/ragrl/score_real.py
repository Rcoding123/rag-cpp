"""`ragrl score-real RUN --days test|dev`: a sealed OF15 run on recorded ES (OF16 phase 2 scoring).

The run's headline ensemble, its seeds (and their per-window mean) and three baselines play
every window of every day of the role on VecReplayEnv (ReplayParams: decide 1 s, window
600 s, latency 1 bucket = 100 ms, $1.50 per side by default, and the warm-up the run TRAINED
with: cfg env.warmup_seconds, 1050 s for every OF15 run; FeatureBuilder's running scales are
far from settled after 150 s), in one process that opens the tapes once (real.open_tapes ->
ledger.open_real; test needs a fresh process). The run's warm-up, decision step and episode
length must equal the manifest's `episodes` block, or nothing is scored (as scalp_run.hpp's
check_net refuses a network trained with another market or warm-up).
Before the tapes are opened every policy and baseline is stepped once on a dummy observation
batch (a load or shape error must not burn the run's only test opening), and each block's
results are written to disk as soon as it finishes (status PARTIAL until the last one).

Baselines (Python: run_builtin only plays simulator sources, so these act on the observation):
    flat           never trades
    delta30        of_train's 30 s delta rule (BuiltinPolicy::delta30: feature 1 > 1.5 long,
                   < -1.5 short, else keep), exactly as tools/of_zoo_trace.py copies it
    imbalance_l1   when flat, enter in the sign of the L1 imbalance (feature 6; 0 -> stay flat);
                   when in a position, go flat: a 1 s hold, then flat (the I1 markout as a policy)
Statistics (OF15's stats.py): per-window summary and gross, paired against flat and delta30 with
the window-level t and percentile bootstrap, plus a day-clustered bootstrap (resample days with
replacement, B = 10000, seed 0) and the Sep / Nov subsets. Windows whose span (warm-up + window)
overlaps a book gap > 5 s (a recording outage) are not played and are listed. --stress also plays latency 3
buckets and $3 per side in the same opening.
Red flags: R1 (A77, redflags.py); RG a learned policy's real gross above its simulator gross
(sealed.json) by 2 SE (check the replay for leaks); RI the imbalance rule profiting (net > 0 by
2 SE: check the bucket alignment); R10r more than one real test opening of the run's config
sha256 (any run name, any manifest), or one logged outside the canonical ledger.
Output: RUN/real/<days>-<stamp>/ (results.json, windows_<block>.npz, STATUS), never sealed.json.
--days dev is marked "dev: not a score".
"""
import math
import time
from pathlib import Path

import numpy as np

from . import ledger, native, real, redflags, registry, stats
from .policies import FrozenPolicy

REPLAY = {"decide_seconds": 1.0, "window_seconds": 600.0, "contracts": 1}  # warm-up: the run's own (check_episodes)
STRESS = {"latency_buckets": 3, "fee_dollars": 3.0}
HEADLINES = ("ensemble", "mean_of_seeds")
M_PLAN = 32  # 16 sealed OF15 runs x 2 pre-registered headlines (plan section 3)
BOOT_B, BOOT_SEED = 10_000, 0
REFS = ("flat", "delta30")
DEV_NOTE = "dev: not a score"


# ---------------------------------------------------------------- baselines on the observation
def keep_action(obs):
    return (np.sign(obs[:, 13]) + 1).astype(np.int64)


def _onehot(a):
    s = np.zeros((len(a), 3))
    s[np.arange(len(a)), a] = 1.0
    return s


class Flat(FrozenPolicy):
    type = "real_flat"
    score_kind = "vote"

    def step(self, obs):
        a = np.ones(len(obs), dtype=np.int64)
        return a, _onehot(a)


class Delta30(FrozenPolicy):
    type = "real_delta30"
    score_kind = "vote"

    def step(self, obs):
        x = obs[:, 1]
        a = np.where(x > 1.5, 2, np.where(x < -1.5, 0, keep_action(obs))).astype(np.int64)
        return a, _onehot(a)


class ImbalanceL1(FrozenPolicy):
    type = "real_imbalance_l1"
    score_kind = "vote"

    def step(self, obs):
        flat = obs[:, 13] == 0
        x = obs[:, 6]
        enter = np.where(x > 0, 2, np.where(x < 0, 0, 1))
        a = np.where(flat, enter, 1).astype(np.int64)
        return a, _onehot(a)


def baselines() -> dict:
    return {"flat": Flat(), "delta30": Delta30(), "imbalance_l1": ImbalanceL1()}


# ---------------------------------------------------------------- replay
def replay_params(warmup_seconds: float, latency_buckets: int = 1, fee_dollars: float = 1.5, **over):
    p = {**REPLAY, "warmup_seconds": float(warmup_seconds), "latency_buckets": int(latency_buckets), "fee_dollars": float(fee_dollars),
         **over}
    return native.load().ReplayParams(**p)


def check_episodes(env_cfg: dict, manifest: dict) -> float:
    """The run's warm-up (seconds), refused unless the run's warm-up, decision step and episode length are the
    manifest's episode cut (the features an agent sees depend on how long FeatureBuilder ran first)."""
    ep = manifest["episodes"]
    want = {"warmup_seconds": float(ep["warmup_seconds"]), "decide_seconds": float(ep["decide_seconds"]),
            "episode_seconds": float(ep["window_seconds"])}
    got = {k: float(env_cfg[k]) for k in want}
    if got != want:
        raise RuntimeError(f"the run's env {got} is not the manifest's episode cut {want}: its agents would see features "
                           "built differently from training (refused, as check_net refuses a mismatched network)")
    return got["warmup_seconds"]


def check_policies(policies: dict, n: int = 3):
    """Steps every policy once on a dummy [n, 16] observation batch: load, shape and dtype errors surface here,
    before any tape is opened. Memory policies are reset again by play()."""
    obs = np.zeros((n, 16), dtype=np.float64)
    obs[:, 15] = 1.0  # time left
    for name, pol in policies.items():
        pol.reset(n)
        a, _ = pol.step(obs)
        a = np.asarray(a)
        if a.shape != (n,) or not np.isin(a, (0, 1, 2)).all():
            raise RuntimeError(f"policy {name}: step() gave actions {a!r} for {n} observations (want {n} of 0, 1, 2)")
        pol.reset(n)


def windows_on_tape(tape, params) -> int:
    """ReplayScalpEnv::windows(): consecutive full windows after the warm-up."""
    b = tape.bucket_seconds
    need = max(1, round(params.window_seconds / b))
    warm = max(1, round(params.warmup_seconds / b))
    return (tape.size - warm) // need if tape.size > warm + need else 0


def play(policy, tape, params, windows, threads: int = 1) -> dict:
    """Per-window results of `policy` on windows of one tape (one lockstep batch)."""
    R = native.load()
    env = R.VecReplayEnv(tape, params, [int(k) for k in windows], threads=max(1, threads))
    obs = env.reset()
    policy.reset(env.n_envs)
    while True:
        a, _ = policy.step(obs)
        obs, _, done = env.step(np.ascontiguousarray(a, dtype=np.int64))
        if done[0]:
            break
    st = env.episode_stats
    return {k: np.asarray(st[k], dtype=np.float64) for k in ("net_pnl_ticks", "orders", "fees_ticks")}


def play_days(policies: dict, tapes: dict, params, threads: int = 1) -> tuple:
    """({name: per-window results over every day, in day order}, [(day, window)], excluded)."""
    res = {k: {f: [] for f in ("net_pnl_ticks", "orders", "fees_ticks")} for k in policies}
    labels, excluded = [], []
    for day, (tape, side) in tapes.items():
        played, ex = real.clean_windows(windows_on_tape(tape, params), side, params.window_seconds, params.warmup_seconds)
        excluded += [{"day": day, **e} for e in ex]
        if not played:
            continue
        labels += [(day, k) for k in played]
        for name, pol in policies.items():
            r = play(pol, tape, params, played, threads)
            for f in r:
                res[name][f].append(r[f])
    out = {k: {f: (np.concatenate(v) if v else np.zeros(0)) for f, v in d.items()} for k, d in res.items()}
    return out, labels, excluded


# ---------------------------------------------------------------- statistics
def day_bootstrap(x, days, b: int = BOOT_B, seed: int = BOOT_SEED) -> dict:
    """Day-clustered bootstrap of the per-window mean: resample days with replacement, pool their windows."""
    x = np.asarray(x, dtype=np.float64)
    days = np.asarray(days)
    uniq = sorted(set(days.tolist()))
    if len(uniq) < 2:
        return {"days": len(uniq), "mean": float(x.mean()) if len(x) else float("nan"), "se": float("nan"), "t": float("nan"),
                "ci95": [float("nan"), float("nan")]}
    s = np.array([x[days == d].sum() for d in uniq])
    n = np.array([(days == d).sum() for d in uniq], dtype=np.float64)
    idx = np.random.default_rng(seed).integers(0, len(uniq), size=(b, len(uniq)))
    boots = s[idx].sum(axis=1) / n[idx].sum(axis=1)
    se = float(boots.std(ddof=1))
    mean = float(x.mean())
    return {"days": len(uniq), "mean": mean, "se": se, "t": mean / se if se > 0 else float("nan"),
            "ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]}


def _take(r: dict, mask) -> dict:
    return {f: np.asarray(v)[mask] for f, v in r.items()}


def block_stats(res: dict, labels: list, subsets: dict, learned: list) -> dict:
    days = np.array([d for d, _ in labels])
    n = len(labels)
    idx = stats.bootstrap_indices(n, BOOT_B, BOOT_SEED) if n > 1 else None
    summ = {k: stats.summary(v) for k, v in res.items()} if n > 1 else {}
    gross = {k: stats.gross(v) for k, v in res.items()}
    out = {"windows": n, "days": sorted(set(days.tolist())), "summary": summ,
           "gross": {k: {f: s[f] for f in ("mean", "se", "t")} for k, s in ({k: stats.summary(v) for k, v in gross.items()} if n > 1 else {}).items()},
           "day_bootstrap": {k: day_bootstrap(v["net_pnl_ticks"], days) for k, v in res.items()} if n else {},
           "paired": {}, "paired_day_bootstrap": {}, "subsets": {}}
    for ref in REFS:
        if ref in res and n > 1:
            out["paired"][ref] = {k: stats.paired(res[k], res[ref], idx) for k in res if k != ref}
            out["paired_day_bootstrap"][ref] = {k: day_bootstrap(res[k]["net_pnl_ticks"] - res[ref]["net_pnl_ticks"], days)
                                                for k in res if k != ref}
    for name, sdays in subsets.items():
        mask = np.isin(days, list(sdays))
        m = int(mask.sum())
        sub = {"windows": m, "days": sorted(set(days[mask].tolist()))}
        if m > 1:
            sr = {k: _take(v, mask) for k, v in res.items()}
            sub["summary"] = {k: stats.summary(v) for k, v in sr.items()}
            sub["paired"] = {"flat": {k: stats.paired(sr[k], sr["flat"]) for k in [*learned, "delta30", "imbalance_l1"] if k in sr}}
            sub["day_bootstrap"] = {k: day_bootstrap(sr[k]["net_pnl_ticks"], days[mask]) for k in [*learned] if k in sr}
        out["subsets"][name] = sub
    return out


def decision(block: dict, t_req: float) -> dict:
    """Plan section 4 per headline: net > 0, t >= t_req, day-clustered 95% CI above 0, positive in Sep and Nov."""
    out = {}
    for h in HEADLINES:
        s = block["summary"].get(h)
        if not s:
            continue
        db = block["day_bootstrap"].get(h, {})
        subs = {k: v.get("summary", {}).get(h, {}).get("mean") for k, v in block["subsets"].items()}
        ok = (s["mean"] > 0 and redflags._gt(s["t"], t_req - 1e-12) and redflags._gt(db.get("ci95", [float("nan")])[0], 0.0)
              and all(v is not None and v > 0 for v in subs.values()))
        out[h] = {"mean": s["mean"], "t": s["t"], "t_required": t_req, "day_ci95": db.get("ci95"), "subset_means": subs,
                  "edge_criteria_met": bool(ok)}
    return out


def red_flags(block: dict, sim_gross: dict, learned: list) -> tuple:
    hard, warn = [], []
    S = block["summary"]
    for k, s in S.items():
        if redflags._gt(s.get("sharpe_annual"), redflags.SHARPE_BAR) and not redflags.vacuous(s):
            lb = redflags.sharpe_lower(s)
            if redflags._gt(lb, redflags.SHARPE_BAR):
                hard.append({"id": "R1", "policy": k, "detail": f"annualized Sharpe {s['sharpe_annual']:.2f}, lower 2-SE bound {lb:.2f} > "
                                                                f"{redflags.SHARPE_BAR:g} (t {redflags._n(s.get('t'), 1)})"})
            else:
                warn.append({"id": "R1v", "policy": k, "detail": f"annualized Sharpe {s['sharpe_annual']:.2f} > {redflags.SHARPE_BAR:g} is noise: "
                                                                 f"lower 2-SE bound {redflags._n(lb)}"})
    G = block["gross"]
    for k in learned:
        if k in G and k in sim_gross:
            d = G[k]["mean"] - sim_gross[k]["mean"]
            se = math.sqrt(G[k]["se"] ** 2 + sim_gross[k]["se"] ** 2)
            if se > 0 and d > 2 * se:
                hard.append({"id": "RG", "policy": k, "detail": f"real gross {G[k]['mean']:+.2f} beats its simulator gross {sim_gross[k]['mean']:+.2f} "
                                                                f"by {d:+.2f} (> 2 SE = {2 * se:.2f}): check the replay for leaks"})
    s = S.get("imbalance_l1")
    if s and s["mean"] > 0 and redflags._gt(s["t"], 2.0):
        hard.append({"id": "RI", "policy": "imbalance_l1", "detail": f"the L1-imbalance rule nets {s['mean']:+.2f} (t {s['t']:.1f}): "
                                                                     "check the bucket alignment first"})
    g = G.get("imbalance_l1")
    if g and g["mean"] > 0 and redflags._gt(g["t"], 2.0):
        warn.append({"id": "RIg", "policy": "imbalance_l1", "detail": f"the L1-imbalance rule's gross is {g['mean']:+.2f} (t {g['t']:.1f})"})
    return hard, warn


# ---------------------------------------------------------------- the command
def score(run_dir, days: str, latency_buckets: int = 1, fee_dollars: float = 1.5, stress: bool = False, threads: int = 1,
          manifest=None, root=None, allow_env_change: bool = False) -> dict:
    from . import runner
    run_dir = Path(run_dir)
    if days not in ("test", "dev"):
        raise ValueError("--days is test or dev (confirm days are the orchestrator's)")
    if not (run_dir / "sealed.json").exists():
        raise FileNotFoundError(f"{run_dir}: not a sealed run (score-real scores sealed OF15 runs)")
    runner.use_run_module(run_dir)
    problems = runner._check_provenance(run_dir, allow_env_change=True)
    hard_env = [p for p in problems if "ragrl code" not in p]  # the scoring code itself is new (OF16): recorded, not refused
    if hard_env and not allow_env_change:
        raise RuntimeError("; ".join(hard_env) + " (--allow-env-change to score anyway; recorded)")
    cfg = registry.read_json(run_dir / "config.json")
    sealed = registry.read_json(run_dir / "sealed.json")
    root = root or real.env_root()
    mpath = real.manifest_path(manifest)
    m = real.load_manifest(mpath)
    warmup = check_episodes(cfg["env"], m)
    runner.setup_torch(cfg["train"]["torch_threads"])
    # everything that can fail is loaded and exercised before the tapes are opened (a retry of a test opening is refused)
    pols = runner.learned_policies(run_dir, cfg)
    seeds = [k for k in pols if k.startswith("seed")]
    base = baselines()
    check_policies({**pols, **base})
    blocks_p = {"base": replay_params(warmup, latency_buckets, fee_dollars)}
    if stress:
        blocks_p["stress"] = replay_params(warmup, **STRESS)
    config_sha = (run_dir / "config.sha256").read_text(encoding="utf-8").strip()
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    out_dir = run_dir / "real" / f"{days}-{stamp}"
    out_dir.mkdir(parents=True, exist_ok=False)
    (out_dir / "STATUS").write_text("OPENING\n", encoding="utf-8")
    try:
        tapes = real.open_tapes(days, manifest=mpath, root=root, run=run_dir.name, config_sha256=config_sha,
                                reason=f"ragrl score-real {run_dir.name} --days {days}", experiment=runner.experiment_id(run_dir))
    except Exception as e:
        (out_dir / "STATUS").write_text(f"NOT_OPENED: {type(e).__name__}: {e}\n", encoding="utf-8")
        raise
    subsets = {}
    for d in tapes:
        subsets.setdefault(real.subset_of(d, m), []).append(d)
    learned = [*seeds, "ensemble", "mean_of_seeds"]
    t_req = stats.t_required(M_PLAN)
    out = {"run": run_dir.name, "days_role": days, "note": DEV_NOTE if days == "dev" else "real test scoring (OF16 phase 2)",
           "manifest": str(mpath), "manifest_sha256": real.manifest_sha256(mpath), "config_sha256": config_sha,
           "tapes": {d: {"csv_sha256": s["csv_sha256"], "contract": s["contract"], "span": s["span"], "tape_stats": t.stats}
                     for d, (t, s) in tapes.items()},
           "subsets": subsets, "headlines": list(HEADLINES), "multiplicity": M_PLAN, "t_required": t_req,
           "env_problems": problems, "ledger_file": str(ledger.usage_file()), "ledger_canonical": ledger.is_canonical(),
           "bootstrap": {"B": BOOT_B, "seed": BOOT_SEED, "day_clustered": True}, "blocks": {}}
    hard, warn = [], []
    for bname, params in blocks_p.items():
        res, labels, excluded = play_days({**pols, **base}, tapes, params, threads)
        if seeds and labels:
            res["mean_of_seeds"] = stats.mean_of_seeds([res[k] for k in seeds])
        b = block_stats(res, labels, subsets, learned)
        b["params"] = {"decide_seconds": params.decide_seconds, "window_seconds": params.window_seconds, "warmup_seconds": params.warmup_seconds,
                       "latency_buckets": params.latency_buckets, "contracts": params.contracts, "fee_dollars": params.fee_dollars}
        b["excluded_windows"] = excluded
        b["decision"] = decision(b, t_req) if days == "test" else {}
        h, w = red_flags(b, sealed.get("gross") or {}, learned)
        hard += [dict(x, block=bname) for x in h]
        warn += [dict(x, block=bname) for x in w]
        out["blocks"][bname] = b
        npz = {f"{bname}__day": np.array([d for d, _ in labels]), f"{bname}__window": np.array([k for _, k in labels], dtype=np.int64)}
        for k, v in res.items():
            for f in ("net_pnl_ticks", "orders", "fees_ticks"):
                npz[f"{bname}__{k}__{f}"] = v[f]
        # this block survives a crash in the next one
        np.savez(out_dir / f"windows_{bname}.npz", **npz)
        registry.write_json(out_dir / "results.json", registry.finite({**out, "status": "PARTIAL", "hard": hard, "warnings": warn}))
        (out_dir / "STATUS").write_text(f"PARTIAL ({', '.join(out['blocks'])} of {', '.join(blocks_p)})\n", encoding="utf-8")
    if days == "test":
        opened = ledger.real_test_openings_of(config_sha)
        if len(opened) > 1:
            hard.append({"id": "R10r", "policy": "-", "detail": f"the real test days were opened {len(opened)} times for this "
                                                                f"config (runs {sorted({str(u.get('run')) for u in opened})})"})
        if not ledger.is_canonical():
            hard.append({"id": "R10r", "policy": "-", "detail": "the real test opening was logged outside the canonical usage ledger"})
    if days == "dev":
        status = "DEV_NOT_A_SCORE"
    elif hard:
        status = "RED_FLAGS"
    else:
        status = "EDGE_CRITERIA_MET" if any(v["edge_criteria_met"] for v in out["blocks"]["base"]["decision"].values()) else "NO_EDGE"
    out.update({"status": status, "hard": hard, "warnings": warn})
    registry.write_json(out_dir / "results.json", registry.finite(out))
    (out_dir / "STATUS").write_text(status + "\n", encoding="utf-8")
    base_s = out["blocks"]["base"]["summary"]
    print(f"{run_dir.name} real {days}: {status}  " + "  ".join(f"{h} {base_s[h]['mean']:+.2f} (t {base_s[h]['t']:.2f})" for h in HEADLINES if h in base_s)
          + f"  -> {out_dir}" + (f"  [{DEV_NOTE}]" if days == "dev" else ""), flush=True)
    return out
