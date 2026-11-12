"""Optuna search on validation only, with a fixed budget (n_trials x trial_budget_steps x trial seeds).

TPESampler(seed=sampler_seed, multivariate=True), trials run one after another
(determinism); each trial trains its trial_seeds in parallel worker processes.
Intermediate values (for MedianPruner) are the mean val_ckpt champion score over
the trial seeds at each checkpoint, reported once every seed has reached it, so
pruning decisions don't depend on timing. The objective is the mean of the
trial seeds' champions on hpo.select, a validation range disjoint from val_ckpt
and val_select (which compare the retrained best config with the baselines
afterwards, so they are not inflated by the search). Never seals. Writes hpo.jsonl, hpo.db
(sqlite) and best_config.json: the best parameters, retrained on fresh run
seeds (retrain_first_seed, ...), with the trial count added to the multiplicity.
"""
import copy
import json
import warnings
from pathlib import Path

import numpy as np

from . import config, ledger, registry
from .envs import indexed_source
from .evaluate import run_policy
from .policies import load_policy
from .worker import setup_torch
from .runner import dump_json_line, spawn_workers, trial_config, use_run_module


def _suggest(trial, name, dist):
    kind = dist[0]
    if kind == "log":
        return trial.suggest_float(name, float(dist[1]), float(dist[2]), log=True)
    if kind == "float":
        return trial.suggest_float(name, float(dist[1]), float(dist[2]))
    if kind == "int":
        return trial.suggest_int(name, int(dist[1]), int(dist[2]))
    choices = [json.dumps(c) for c in dist[1:]]
    return json.loads(trial.suggest_categorical(name, choices))


def _val_records(path: Path):
    """The val records of a worker's metrics.jsonl; a last line still being appended (no newline yet) is skipped."""
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").split("\n")[:-1]  # everything before the last newline
    return [json.loads(l) for l in lines if l.strip() and '"type": "val"' in l]


def hpo(cfg_path, root: Path | None = None) -> Path:
    import optuna
    warnings.filterwarnings("ignore", category=optuna.exceptions.ExperimentalWarning)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    cfg = config.load(cfg_path)
    h = cfg["hpo"]
    if h is None:
        raise ValueError("the config has no hpo section")
    base = copy.deepcopy(cfg)
    base["name"] = cfg["name"] + "_hpo"
    run_dir = registry.new_run(base, config.sha256(cfg), root)
    use_run_module(run_dir)
    ledger.as_trainer()
    setup_torch(cfg["train"]["torch_threads"])
    config.dump(cfg, run_dir / "config.json")
    registry.write_json(run_dir / "provenance.json", registry.provenance(run_dir / "bin"))
    params = config.make_params(cfg["env"])
    vs = h["select"]
    study = optuna.create_study(direction="maximize", study_name=cfg["name"], storage=f"sqlite:///{(run_dir / 'hpo.db').as_posix()}",
                                sampler=optuna.samplers.TPESampler(seed=h["sampler_seed"], multivariate=True),
                                pruner=optuna.pruners.MedianPruner(n_startup_trials=h["pruner_startup_trials"]))
    log = run_dir / "hpo.jsonl"
    for i in range(h["n_trials"]):
        trial = study.ask()
        sampled = {k: _suggest(trial, k, d) for k, d in h["space"].items()}
        tcfg = trial_config(cfg, sampled, f"{cfg['name']}_t{i:03d}")
        tcfg["train"]["seeds"] = list(h["trial_seeds"])
        tcfg["train"]["budget_steps"] = h["trial_budget_steps"]
        tcfg["train"]["workers"] = min(cfg["train"]["workers"], len(h["trial_seeds"]))
        tcfg = config.resolve(tcfg)
        tdir = run_dir / "trials" / f"t{i:03d}"
        tdir.mkdir(parents=True)
        config.dump(tcfg, tdir / "config.json")
        (tdir / "bin").mkdir()
        for f in (run_dir / "bin").iterdir():
            (tdir / "bin" / f.name).write_bytes(f.read_bytes())
        reported = [0]

        def poll():
            recs = [_val_records(tdir / "seeds" / f"seed{s}" / "metrics.jsonl") for s in tcfg["train"]["seeds"]]
            k = min(len(r) for r in recs)
            while reported[0] < k:
                c = reported[0]
                trial.report(float(np.mean([r[c]["champion"] for r in recs])), c)
                reported[0] += 1
                if trial.should_prune():
                    return False
            return True

        done = spawn_workers(tdir, tcfg, tcfg["train"]["seeds"], on_poll=poll)
        rec = {"trial": i, "params": sampled}
        if not done:
            study.tell(trial, state=optuna.trial.TrialState.PRUNED)
            rec.update(state="pruned", checkpoints=reported[0])
        else:
            src = indexed_source("validation", vs["first"], vs["count"])
            means = [float(np.mean(run_policy(params, src, load_policy(tdir / "seeds" / f"seed{s}"), cfg["select"]["select_threads"])
                                   ["net_pnl_ticks"])) for s in tcfg["train"]["seeds"]]
            value = float(np.mean(means))
            study.tell(trial, value)
            rec.update(state="complete", value=value, seed_values=means)
        dump_json_line(log, rec)
        print(f"trial {i}: {rec['state']} {rec.get('value', '')} {sampled}", flush=True)
    best = study.best_trial
    out = copy.deepcopy(cfg)
    out["name"] = cfg["name"] + "_best"
    out["agent"]["params"].update({k: json.loads(json.dumps(v)) for k, v in best.params.items()})
    for k, d in h["space"].items():
        if d[0] == "cat":
            out["agent"]["params"][k] = json.loads(best.params[k])
    out["hpo"] = None
    first = h["retrain_first_seed"]
    out["train"]["seeds"] = list(range(first, first + len(cfg["train"]["seeds"])))
    out["select"]["multiplicity_extra"] = cfg["select"]["multiplicity_extra"] * h["n_trials"]
    for k in ("episodes", "env_steps", "budget_rounded"):
        out["train"][k] = None
    out = config.resolve(out)
    config.dump(out, run_dir / "best_config.json")
    registry.write_json(run_dir / "hpo_summary.json", {"best_trial": best.number, "best_value": best.value, "best_params": best.params,
                                                       "n_trials": h["n_trials"], "budget_steps_per_seed": h["trial_budget_steps"]})
    print(f"best trial {best.number}: {best.value} {best.params}; retrain with: python -m ragrl run {run_dir / 'best_config.json'}")
    return run_dir
