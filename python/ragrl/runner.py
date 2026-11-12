"""The experiment protocol: train (K worker processes) -> select (validation
only) -> seal (a fresh scorer process, once) -> red flags -> report.

    run      parent: forbid_sealed(); copies the module into RUN/bin; prepare;
             workers; diag; select; then `ragrl seal RUN` as a subprocess.
    select   val_select = validation[first, +count) (disjoint from val_ckpt):
             every seed champion, the ensemble and the baselines, paired; the
             validation-best oracle; evaluation determinism (bitwise); M.
    seal     fresh process: refuses if sealed.json exists, checks the module
             and venv against provenance, opens the test block (ledger.py),
             paired evaluation of all policies, sealed.json (write-once).
    confirm  950M+ (orchestrator only, once per claim id); the pipeline never runs it.
"""
import copy
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from . import config, ledger, native, redflags, registry, report, stats
from .agents import ZOO
from .envs import indexed_source
from .evaluate import Builtin, baseline_policies, paired_eval, run_policy, stage_baselines
from .policies import Ensemble, load_policy
from .worker import setup_torch

PYTHON_DIR = Path(__file__).resolve().parents[1]


def use_run_module(run: Path):
    """Imports _ragof from RUN/bin (must be called before the first native.load() in this process)."""
    if native._R is not None and Path(native._R.__file__).parent.resolve() != (run / "bin").resolve():
        raise RuntimeError("_ragof was already imported from elsewhere in this process")
    os.environ["RAGOF_DIR"] = str(run / "bin")


def child_env(run: Path, torch_threads: int = 1) -> dict:
    env = os.environ.copy()
    env["RAGOF_DIR"] = str(run / "bin")
    env["PYTHONPATH"] = str(PYTHON_DIR) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        env[k] = str(torch_threads)
    env["PYTHONHASHSEED"] = "0"
    return env


def spawn_workers(run: Path, cfg: dict, seeds, out_root: Path | None = None, on_poll=None) -> bool:
    """Trains `seeds` in parallel worker processes (at most train.workers at a time). on_poll() -> False aborts."""
    t = cfg["train"]
    env = child_env(run, t["torch_threads"])
    pending, running = list(seeds), {}
    try:
        while pending or running:
            while pending and len(running) < t["workers"]:
                k = pending.pop(0)
                sd = (out_root or run / "seeds") / f"seed{k}"
                sd.mkdir(parents=True, exist_ok=True)
                log = open(sd / "worker.log", "w", encoding="utf-8")
                cmd = [sys.executable, "-m", "ragrl", "train", str(run), "--seed", str(k)]
                if out_root is not None:
                    cmd += ["--out-dir", str(sd)]
                running[k] = (subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env, cwd=native.REPO), log, sd)
            for k in list(running):
                proc, log, sd = running[k]
                rc = proc.poll()
                if rc is None:
                    continue
                log.close()
                del running[k]
                if rc != 0:
                    tail = (sd / "worker.log").read_text(encoding="utf-8", errors="replace")[-3000:]
                    raise RuntimeError(f"worker seed {k} failed ({rc}):\n{tail}")
            if on_poll is not None and on_poll() is False:
                return False
            time.sleep(0.3)
        return True
    finally:
        for proc, log, _ in running.values():
            proc.kill()
            proc.wait()
            log.close()


def run(cfg_path, no_seal: bool = False, root: Path | None = None, reuse_ok: bool = False) -> Path:
    cfg = config.load(cfg_path)
    sha = config.sha256(cfg)
    run_dir = registry.new_run(cfg, sha, root)
    use_run_module(run_dir)
    ledger.as_trainer()  # this process never opens a sealed block
    setup_torch(cfg["train"]["torch_threads"])  # evaluation with the workers' torch threads
    config.dump(cfg, run_dir / "config.json")
    (run_dir / "config.sha256").write_text(sha + "\n", encoding="utf-8")
    registry.write_json(run_dir / "provenance.json", registry.provenance(run_dir / "bin"))
    stage_baselines(cfg, run_dir)
    metrics = registry.Metrics(run_dir / "metrics.jsonl")
    print(f"run {run_dir}", flush=True)
    t0 = time.perf_counter()
    spec = ZOO[cfg["agent"]["kind"]]
    if spec.prepare:
        spec.prepare(cfg, run_dir)
    spawn_workers(run_dir, cfg, cfg["train"]["seeds"])
    metrics.write("timing", phase="train", seconds=time.perf_counter() - t0)
    t1 = time.perf_counter()
    diag(run_dir, cfg)
    select(run_dir)
    metrics.write("timing", phase="select", seconds=time.perf_counter() - t1)
    if cfg["train"]["recheck"]:
        t2 = time.perf_counter()
        rc = recheck(run_dir)
        metrics.write("timing", phase="recheck", seconds=time.perf_counter() - t2)
        print(f"recheck seed {rc['seed']}: same actions {rc['same_actions']}, same val {rc['same_val']}", flush=True)
    registry.append_index(run_dir, {"config_sha256": sha, "agent": cfg["agent"]["kind"], "stage": "selected"})
    if not no_seal:
        cmd = [sys.executable, "-m", "ragrl", "seal", str(run_dir)] + (["--reuse-ok"] if reuse_ok else [])
        p = subprocess.run(cmd, env=child_env(run_dir), cwd=native.REPO)
        if p.returncode != 0:
            raise RuntimeError(f"seal failed ({p.returncode})")
    return run_dir


def diag(run_dir: Path, cfg: dict, episodes: int = 20):
    """R7 probe: |corr(obs_k, hidden gap)| per observation column on train-stream episodes (privileged, train block)."""
    R = native.load()
    params = config.make_params(cfg["env"])
    n = min(episodes, 20)
    env = R.VecScalpEnv(params, R.EpisodeSource.train(int(cfg["train"]["seeds"][0])), n_envs=n, threads=min(8, n), privileged=True)
    obs = env.reset()
    xs, gs = [], []
    flat = np.ones(n, dtype=np.int64)
    while True:
        xs.append(obs)
        gs.append(env.priv[:, 2])
        obs, _, done = env.step(flat)
        if done[0]:
            break
    X = np.concatenate(xs)
    g = np.concatenate(gs)
    corr = []
    for k in range(X.shape[1]):
        sx = X[:, k].std()
        corr.append(float(np.corrcoef(X[:, k], g)[0, 1]) if sx > 0 and g.std() > 0 else 0.0)
    out = {"episodes": n, "rows": int(len(g)), "corr_obs_gap": corr, "max_abs": float(np.max(np.abs(corr)))}
    registry.write_json(run_dir / "diag.json", out)
    return out


def seed_dirs(run_dir: Path, cfg: dict):
    return {int(s): run_dir / "seeds" / f"seed{s}" for s in cfg["train"]["seeds"]}


def learned_policies(run_dir: Path, cfg: dict) -> dict:
    pols = {f"seed{s}": load_policy(d) for s, d in seed_dirs(run_dir, cfg).items()}
    members = list(pols.values())
    if len(members) > 1:
        # the ensemble needs its own member instances (memory agents keep per-episode state)
        pols["ensemble"] = Ensemble([load_policy(d) for d in seed_dirs(run_dir, cfg).values()])
    else:
        pols["ensemble"] = load_policy(next(iter(seed_dirs(run_dir, cfg).values())))
    return pols


def multiplicity(run_dir: Path, cfg: dict) -> int:
    vals = [registry.read_json(d / "val.json") for d in seed_dirs(run_dir, cfg).values()]
    return int(max(v["offers"] for v in vals) * max(v["choices_per_seed"] for v in vals) * cfg["select"]["multiplicity_extra"])


def _summaries(res: dict) -> dict:
    return {k: stats.summary(v) for k, v in res.items()}


def select(run_dir: Path) -> dict:
    """Validation-only selection record (val_select); never touches sealed blocks."""
    cfg = registry.read_json(run_dir / "config.json")
    params = config.make_params(cfg["env"])
    threads = cfg["select"]["select_threads"]
    vs = cfg["val_select"]
    src = indexed_source("validation", vs["first"], vs["count"])
    pols = learned_policies(run_dir, cfg)
    base = baseline_policies(cfg, run_dir)
    res = paired_eval(params, src, {**pols, **base}, threads)
    seeds = [k for k in pols if k.startswith("seed")]
    res["mean_of_seeds"] = stats.mean_of_seeds([res[k] for k in seeds])
    summ = _summaries(res)
    oracles = [k for k in base if k.startswith("oracle_")]
    best_oracle = max(oracles, key=lambda k: summ[k]["mean"]) if oracles else None
    paired_vs = {}
    for ref in ["flat", "delta30"]:
        if ref in res:
            paired_vs[ref] = {k: stats.paired(res[k], res[ref]) for k in [*seeds, "ensemble", "mean_of_seeds"]}
    # evaluation determinism: the first seed's champion twice on val_ckpt, and against its worker's hash
    s0 = int(cfg["train"]["seeds"][0])
    d0 = seed_dirs(run_dir, cfg)[s0]
    vc = cfg["select"]["val_ckpt"]
    h = [run_policy(params, indexed_source("validation", vc["first"], vc["count"]), load_policy(d0), threads, actions_hash=True)
         for _ in range(2)]
    worker = registry.read_json(d0 / "val.json")
    det = {"seed": s0, "same_twice": bool(h[0]["actions_sha256"] == h[1]["actions_sha256"]
                                          and np.array_equal(h[0]["net_pnl_ticks"], h[1]["net_pnl_ticks"])),
           "same_as_worker": bool(h[0]["actions_sha256"] == worker["actions_sha256"]
                                  and float(np.mean(h[0]["net_pnl_ticks"])) == worker["val_ckpt_mean"]),
           "actions_sha256": h[0]["actions_sha256"]}
    out = {"val_select": vs, "headline": cfg["score"]["headline"], "summary": summ, "paired": paired_vs, "best_oracle": best_oracle,
           "multiplicity": multiplicity(run_dir, cfg), "determinism": det,
           "val_ckpt": {f"seed{s}": registry.read_json(d / "val.json")["val_ckpt_mean"] for s, d in seed_dirs(run_dir, cfg).items()}}
    registry.write_json(run_dir / "select.json", registry.finite(out))
    np.savez(run_dir / "val_select_episodes.npz", **{f"{k}__{f}": v[f] for k, v in res.items() for f in ("net_pnl_ticks", "orders", "fees_ticks")})
    registry.Metrics(run_dir / "metrics.jsonl").write("select", headline=cfg["score"]["headline"],
                                                       headline_mean=summ[cfg["score"]["headline"]]["mean"], best_oracle=best_oracle,
                                                       multiplicity=out["multiplicity"])
    return out


def _check_provenance(run_dir: Path, allow_env_change: bool):
    """The module, the venv and the ragrl code (the package this process imported) must be the training ones."""
    prov = registry.read_json(run_dir / "provenance.json")
    mod = native.find_module(run_dir / "bin")
    problems = []
    if native.sha256_file(mod) != prov["module_sha256"]:
        problems.append("the module in RUN/bin changed since training")
    if registry.freeze_sha256()[0] != prov["freeze_sha256"]:
        problems.append("the venv's installed packages changed since training")
    if registry.code_sha256() != prov.get("ragrl_sha256"):
        problems.append(f"the ragrl code ({registry.PKG_DIR}) differs from the code that trained and selected")
    if problems and not allow_env_change:
        raise RuntimeError("; ".join(problems) + " (--allow-env-change to score anyway; recorded)")
    return problems


def experiment_id(run_dir: Path) -> str:
    """config sha256 + ragrl code sha256: a regeneration of the same experiment plays the same policies."""
    prov = registry.read_json(run_dir / "provenance.json")
    return (run_dir / "config.sha256").read_text(encoding="utf-8").strip() + ":" + str(prov.get("ragrl_sha256"))


def seal(run_dir: Path, allow_env_change: bool = False, reuse_ok: bool = False) -> dict:
    run_dir = Path(run_dir)
    if (run_dir / "sealed.json").exists():
        raise FileExistsError(f"{run_dir}: already sealed (sealed.json is write-once)")
    if not (run_dir / "select.json").exists():
        raise FileNotFoundError("select.json missing: selection (validation) must finish before scoring")
    use_run_module(run_dir)
    problems = _check_provenance(run_dir, allow_env_change)
    cfg = registry.read_json(run_dir / "config.json")
    setup_torch(cfg["train"]["torch_threads"])
    sel = registry.read_json(run_dir / "select.json")
    sc = cfg["score"]
    block = sc["block"]
    # everything that can fail is loaded before the block is opened (a retry would be a second opening: R10)
    params = config.make_params(cfg["env"])
    pols = learned_policies(run_dir, cfg)
    base = baseline_policies(cfg, run_dir)
    meta = {"market": cfg["env"]["market"], "family": sc["family"], "experiment": experiment_id(run_dir)}
    reuse = None
    if block == "test":
        reuse = ledger.range_reuse(block, sc["first"], sc["count"], cfg["env"]["market"], run_dir.name, meta["experiment"],
                                   sc["family"], sc["family_size"])
        if reuse["reused"] and not reuse_ok:
            raise PermissionError(f"test [{sc['first']}, +{sc['count']}) on {cfg['env']['market']} was viewed before: {reuse} "
                                  "(pre-register a fresh range in score.first, or --reuse-ok to score it as a comparable-only, "
                                  "multiplicity-inflated result)")
        ledger.as_scorer("test", f"ragrl seal {run_dir.name}", run_dir.name, sc["first"], sc["count"], **meta)
    else:
        ledger.log_dry_run(run_dir.name, sc["first"], sc["count"], **meta)
    src = indexed_source(block, sc["first"], sc["count"])
    res = paired_eval(params, src, {**pols, **base}, sc["threads"])
    seeds = [k for k in pols if k.startswith("seed")]
    res["mean_of_seeds"] = stats.mean_of_seeds([res[k] for k in seeds])
    summ = _summaries(res)
    idx = stats.bootstrap_indices(sc["count"], sc["bootstrap"], sc["bootstrap_seed"])
    refs = [r for r in ["flat", "delta30", sel.get("best_oracle"), "neuralq_ref_ensemble"] if r and r in res]
    learned = [*seeds, "ensemble", "mean_of_seeds"]
    paired_d = {ref: {k: stats.paired(res[k], res[ref], idx) for k in learned} for ref in refs}
    # gross (before fees) against EVERY oracle threshold: a leak shows up as more gross edge than the
    # fair-value oracle even when fees keep the net negative (red flag R2g)
    gross = {k: stats.gross(v) for k, v in res.items()}
    oracles = [k for k in base if k.startswith("oracle_")]
    paired_gross = {o: {k: stats.paired(gross[k], gross[o], idx) for k in learned} for o in oracles}
    gross_summary = {k: {f: s[f] for f in ("mean", "se", "t")} for k, s in _summaries(gross).items()}
    out = {"block": block, "dry_run": block != "test", "first": sc["first"], "count": sc["count"], "source": src.describe(),
           "market": cfg["env"]["market"], "family": sc["family"], "family_size": sc["family_size"],
           "headline": sc["headline"], "summary": summ, "paired": paired_d, "learned": learned, "baselines": list(base),
           "gross": gross_summary, "paired_gross": paired_gross,
           "bootstrap": {"B": sc["bootstrap"], "seed": sc["bootstrap_seed"]}, "env_problems": problems,
           "range_reuse": reuse, "reuse_ok": bool(reuse_ok), "ledger_file": str(ledger.usage_file()),
           "ledger_canonical": ledger.is_canonical(), "module_sha256": native.sha256_file(native.find_module(run_dir / "bin"))}
    registry.write_json(run_dir / "sealed.json", registry.finite(out), once=True)
    np.savez(run_dir / "sealed_episodes.npz", **{f"{k}__{f}": v[f] for k, v in res.items() for f in ("net_pnl_ticks", "orders", "fees_ticks")})
    flags = redflags.evaluate(run_dir)
    report.write(run_dir)
    registry.append_index(run_dir, {"stage": "sealed", "block": block, "status": flags["status"],
                                    "headline_mean": summ[sc["headline"]]["mean"]})
    print(f"{run_dir.name}: {flags['status']}  headline {sc['headline']} {summ[sc['headline']]['mean']:+.2f} ticks/episode "
          f"({'DRY RUN on viz' if block != 'test' else 'sealed test'})", flush=True)
    return out


def confirm(run_dir: Path, claim: str, orchestrator: bool, count: int = 300) -> dict:
    """950M+ fresh confirmation of a run's headline (orchestrator only; once per claim id, on episodes
    no earlier confirmation used; only for a sealed, non-dry run whose STATUS is a claim)."""
    if not orchestrator:
        raise PermissionError("confirm is for the orchestrator only (--i-am-the-orchestrator); the pipeline never runs it")
    if not claim:
        raise PermissionError("confirm needs --claim ID")
    run_dir = Path(run_dir)
    out_file = run_dir / f"confirm_{claim}.json"
    if out_file.exists():
        raise FileExistsError(f"{out_file} exists")
    if not (run_dir / "sealed.json").exists():
        raise FileNotFoundError("confirm follows a sealed scoring")
    sealed = registry.read_json(run_dir / "sealed.json")
    if sealed.get("dry_run") or sealed.get("block") != "test":
        raise PermissionError("confirm: this run was a dry run (viz), not a sealed test scoring")
    status = (run_dir / "STATUS").read_text(encoding="utf-8").strip() if (run_dir / "STATUS").exists() else ""
    if not status.startswith("CLAIM_PENDING"):
        raise PermissionError(f"confirm: STATUS is {status or 'missing'!r}, not CLAIM_PENDING_950M_CONFIRMATION")
    use_run_module(run_dir)
    _check_provenance(run_dir, False)
    cfg = registry.read_json(run_dir / "config.json")
    sel = registry.read_json(run_dir / "select.json")
    params = config.make_params(cfg["env"])
    pols = learned_policies(run_dir, cfg)
    first = ledger.next_confirm_first()
    ledger.as_scorer("confirm", f"ragrl confirm {run_dir.name}", run_dir.name, first, count, claim,
                     market=cfg["env"]["market"], experiment=experiment_id(run_dir))
    src = indexed_source("confirm", first, count, orchestrator=claim)
    head = cfg["score"]["headline"]
    cand = {"ensemble": pols["ensemble"], "flat": Builtin("flat"), "delta30": Builtin("delta30")}
    if sel.get("best_oracle"):
        cand[sel["best_oracle"]] = Builtin.oracle(float(sel["best_oracle"].split("_")[1]))
    if head == "mean_of_seeds":
        cand.update({k: v for k, v in pols.items() if k.startswith("seed")})
    res = paired_eval(params, src, cand, 24)
    if head == "mean_of_seeds":
        res["mean_of_seeds"] = stats.mean_of_seeds([res[k] for k in pols if k.startswith("seed")])
    out = {"claim": claim, "first": first, "count": count, "summary": _summaries(res),
           "paired_vs_flat": stats.paired(res[head], res["flat"], stats.bootstrap_indices(count, 10_000, 0))}
    registry.write_json(out_file, registry.finite(out), once=True)
    return out


def recheck(run_dir: Path, seed: int | None = None) -> dict:
    """R8: retrains one seed from scratch in a fresh worker and compares the champion bit for bit."""
    run_dir = Path(run_dir)
    cfg = registry.read_json(run_dir / "config.json")
    seed = int(seed or cfg["train"]["seeds"][0])
    out_root = run_dir / "recheck"
    spawn_workers(run_dir, dict(cfg, train=dict(cfg["train"], workers=1)), [seed], out_root=out_root)
    a = registry.read_json(run_dir / "seeds" / f"seed{seed}" / "val.json")
    b = registry.read_json(out_root / f"seed{seed}" / "val.json")
    pa = registry.read_json(run_dir / "seeds" / f"seed{seed}" / "policy.json")["sha256"]
    pb = registry.read_json(out_root / f"seed{seed}" / "policy.json")["sha256"]
    out = {"seed": seed, "same_actions": a["actions_sha256"] == b["actions_sha256"], "same_val": a["val_ckpt_mean"] == b["val_ckpt_mean"],
           "same_artifacts": pa == pb, "a": a["actions_sha256"], "b": b["actions_sha256"]}
    registry.write_json(run_dir / "recheck.json", out)
    if (run_dir / "sealed.json").exists():
        redflags.evaluate(run_dir)
        report.write(run_dir)
    return out


def trial_config(cfg: dict, params: dict, name: str) -> dict:
    t = copy.deepcopy(cfg)
    t["name"] = name
    t["agent"]["params"].update(params)
    t["hpo"] = None
    t["live"]["enabled"] = False
    return config.resolve(t)


def dump_json_line(path: Path, rec: dict):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(registry.finite(rec), sort_keys=True, default=registry._default) + "\n")
