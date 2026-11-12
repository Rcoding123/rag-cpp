"""Experiment configs (schema ragrl/1): JSON, strict (unknown keys are errors),
merged over defaults, resolved (agent defaults from the zoo, budget rounded to
whole training rounds) and written back into the run directory.

    {"schema": "ragrl/1", "name": "of15_ppo_v4",
     "env":   {"market": "discovery_v4", "warmup_seconds": 1050, ...},
     "agent": {"kind": "ppo", "params": {...}},          # merged over ZOO defaults
     "train": {"seeds": [1, ..., 12], "budget_steps": 1200000, "n_envs": null, ...},
     "select": {"val_ckpt": {"first": 0, "count": 200}, "every_steps": 60000, ...},
     "val_select": {"first": 1000000, "count": 400},
     "score": {"block": "test", "first": F, "count": 300, "headline": "ensemble", "family": ..., "family_size": K, ...},
     "hpo": null | {..., "select": {"first": 2000000, "count": 400}}, "live": {...}}

score.first has no default: a sealed range is pre-registered per experiment family
(900M+[0, 100000) was viewed by OF2-OF14). family / family_size declare how many
runs share it (all of them count in the multiplicity bar). hpo.select (the HPO
objective) is disjoint from val_ckpt and val_select (which compare agents afterwards).

Seeds are run seeds (train block, 1..700M-1); every other range is block-relative
(the C++ side checks it against the seed ledger).
"""
import copy
import hashlib
import json
import math
import os
from pathlib import Path

from . import SCHEMA

DEFAULTS = {
    "schema": SCHEMA,
    "name": None,
    "description": "",
    "env": {"market": "default", "warmup_seconds": 150.0, "latency_steps": 1, "fee_dollars": 1.5, "contracts": 1,
            "decide_seconds": 1.0, "episode_seconds": 600.0, "sim_overrides": {}},
    "agent": {"kind": None, "params": {}},
    "train": {"seeds": [1], "budget_steps": 1_200_000, "n_envs": None, "env_threads": 1, "torch_threads": 1, "eval_threads": 1,
              "workers": 1, "reward_scale": 1.0, "recheck": False,
              "episodes": None, "env_steps": None, "budget_rounded": None},  # computed by resolve()
    "select": {"val_ckpt": {"first": 0, "count": 200}, "every_steps": 60_000, "multiplicity_extra": 1, "select_threads": 24},
    "val_select": {"first": 1_000_000, "count": 400},
    "score": {"block": "test", "first": None, "count": 300, "headline": "ensemble", "family": None, "family_size": 1,
              "baselines": ["flat", "delta30", {"oracle": [0.5, 0.55, 0.75, 1.0, 2.0]}],
              "bootstrap": 10_000, "bootstrap_seed": 0, "threads": 24},
    "hpo": None,
    "live": {"enabled": False, "seed": None, "every_episodes": 50, "probe": 5},
}
HPO_DEFAULTS = {"n_trials": 20, "trial_budget_steps": 300_000, "trial_seeds": [1, 2], "sampler_seed": 1, "space": {},
                "pruner_startup_trials": 5, "retrain_first_seed": 101, "select": {"first": 2_000_000, "count": 400}}
SMOKE_MAX_BUDGET = 100_000   # 'viz' dry-run scoring only under this training budget ...
SMOKE_MAX_SCORE = 100        # ... and on at most this many episodes
OF_TRAIN_VAL = (0, 40)       # of_train's champion selection range: validation 700M+[0, 40), 20 evaluations
FREE_DICTS = {("env", "sim_overrides"), ("agent", "params"), ("hpo", "space")}


class ConfigError(ValueError):
    pass


def _merge(base, over, path=()):
    if over is None:
        return copy.deepcopy(base)
    if not isinstance(base, dict) or path in FREE_DICTS:
        return copy.deepcopy(over)
    if not isinstance(over, dict):
        raise ConfigError(f"{'.'.join(path) or 'config'}: expected an object")
    out = copy.deepcopy(base)
    for k, v in over.items():
        if k not in base:
            raise ConfigError(f"unknown key {'.'.join(path + (k,))}")
        if k == "hpo":
            out[k] = None if v is None else _merge(HPO_DEFAULTS, v, path + (k,))
        elif isinstance(base[k], dict) and (path + (k,)) not in FREE_DICTS:
            out[k] = _merge(base[k], v, path + (k,))
        else:
            out[k] = copy.deepcopy(v)
    return out


def load(path) -> dict:
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    return resolve(raw)


def resolve(raw: dict) -> dict:
    """Defaults + zoo defaults + validation. Idempotent: resolve(resolve(x)) == resolve(x)."""
    from .agents import ZOO
    if raw.get("schema") != SCHEMA:
        raise ConfigError(f"schema must be {SCHEMA!r}")
    cfg = _merge(DEFAULTS, raw)
    if not cfg["name"] or not isinstance(cfg["name"], str) or any(c in cfg["name"] for c in '/\\:*?"<>| '):
        raise ConfigError("name: a short identifier (no spaces or path characters)")
    kind = cfg["agent"]["kind"]
    if kind not in ZOO:
        raise ConfigError(f"agent.kind {kind!r}: one of {sorted(ZOO)}")
    spec = ZOO[kind]
    params = dict(spec.defaults)
    for k, v in cfg["agent"]["params"].items():
        if k not in params:
            raise ConfigError(f"agent.params.{k}: not a parameter of {kind} ({sorted(params)})")
        params[k] = v
    cfg["agent"]["params"] = params
    t = cfg["train"]
    if t["n_envs"] is None:
        t["n_envs"] = spec.n_envs
    seeds = t["seeds"]
    if not seeds or len(set(seeds)) != len(seeds) or not all(isinstance(s, int) and 1 <= s < 700_000_000 for s in seeds):
        raise ConfigError("train.seeds: distinct run seeds in [1, 700000000)")
    for k in ["n_envs", "env_threads", "torch_threads", "eval_threads", "workers", "budget_steps"]:
        if not isinstance(t[k], int) or t[k] < 1:
            raise ConfigError(f"train.{k}: a positive integer")
    budget = cpu_budget()
    if t["workers"] * (t["env_threads"] + t["torch_threads"]) > budget:
        raise ConfigError(f"train: workers x (env_threads + torch_threads) = {t['workers'] * (t['env_threads'] + t['torch_threads'])} "
                          f"> {budget} hardware threads")
    steps = steps_per_episode(cfg["env"])
    rounds = math.ceil(t["budget_steps"] / (steps * t["n_envs"]))
    t["episodes"] = rounds * t["n_envs"]          # whole rounds: the train stream's episodes 0..episodes-1
    t["env_steps"] = t["episodes"] * steps
    t["budget_rounded"] = t["env_steps"] != t["budget_steps"]
    ranges = {"select.val_ckpt": cfg["select"]["val_ckpt"], "val_select": cfg["val_select"]}
    if cfg["hpo"] is not None:
        ranges["hpo.select"] = cfg["hpo"]["select"]
    for name, blk in ranges.items():
        if not (isinstance(blk["first"], int) and isinstance(blk["count"], int) and blk["first"] >= 0 and blk["count"] >= 1):
            raise ConfigError(f"{name}: integers first >= 0, count >= 1")
    names = list(ranges)
    for i, x in enumerate(names):
        for y in names[i + 1:]:
            a, b = ranges[x], ranges[y]
            if a["first"] < b["first"] + b["count"] and b["first"] < a["first"] + a["count"]:
                raise ConfigError(f"validation ranges {x} and {y} must not overlap")
    if kind == "neuralq_ref" and cfg["select"]["val_ckpt"]["first"] < OF_TRAIN_VAL[1]:
        raise ConfigError(f"neuralq_ref: select.val_ckpt must start at >= {OF_TRAIN_VAL[1]} (of_train picked its champions on "
                          f"validation [0, {OF_TRAIN_VAL[1]}))")
    if cfg["select"]["every_steps"] < 1:
        raise ConfigError("select.every_steps >= 1")
    s = cfg["score"]
    if s["block"] not in ("test", "viz"):
        raise ConfigError("score.block: 'test' (sealed) or 'viz' (dry run for smoke tests; never a result)")
    if s["block"] == "viz" and (t["budget_steps"] > SMOKE_MAX_BUDGET or not isinstance(s["count"], int) or s["count"] > SMOKE_MAX_SCORE):
        raise ConfigError(f"score.block 'viz' is for smoke runs only (train.budget_steps <= {SMOKE_MAX_BUDGET}, "
                          f"score.count <= {SMOKE_MAX_SCORE}); the visualization block is never a result")
    if s["first"] is None:
        raise ConfigError("score.first: pre-register the sealed range explicitly (test 900M+[0, 100000) was viewed by OF2-OF14)")
    if not (isinstance(s["first"], int) and isinstance(s["count"], int) and s["first"] >= 0 and s["count"] >= 2):
        raise ConfigError("score: integers first >= 0, count >= 2")
    if s["family"] is not None and not (isinstance(s["family"], str) and s["family"]):
        raise ConfigError("score.family: null or a name shared by the runs pre-registered on one sealed range")
    if not (isinstance(s["family_size"], int) and s["family_size"] >= 1):
        raise ConfigError("score.family_size: the number of runs pre-registered on the range (>= 1)")
    if s["headline"] not in ("ensemble", "mean_of_seeds"):
        raise ConfigError("score.headline: 'ensemble' or 'mean_of_seeds' (declared before training; never the best seed)")
    for bl in s["baselines"]:
        _check_baseline(bl)
        if isinstance(bl, dict) and "neuralq_dir" in bl:
            bad = of_train_env_problems(cfg["env"])
            if bad:
                raise ConfigError(f"score.baselines neuralq_dir: of_train trains with the default {bad}; "
                                  "its nets would run outside their training conditions")
    lv = cfg["live"]
    if lv["seed"] is None:
        lv["seed"] = seeds[0]
    if lv["enabled"] and lv["seed"] not in seeds:
        raise ConfigError("live.seed must be one of train.seeds")
    h = cfg["hpo"]
    if h is not None:
        if not all(isinstance(x, int) and 1 <= x < 700_000_000 for x in h["trial_seeds"]):
            raise ConfigError("hpo.trial_seeds: run seeds")
        for name, dist in h["space"].items():
            if name not in params:
                raise ConfigError(f"hpo.space.{name}: not a parameter of {kind}")
            if not (isinstance(dist, list) and dist and dist[0] in ("log", "float", "int", "cat")):
                raise ConfigError(f"hpo.space.{name}: ['log'|'float'|'int', lo, hi] or ['cat', v1, v2, ...]")
    return cfg


def _check_baseline(bl):
    if bl in ("flat", "hold", "delta30"):
        return
    if isinstance(bl, dict) and len(bl) >= 1:
        if "oracle" in bl and isinstance(bl["oracle"], list) and all(isinstance(x, (int, float)) for x in bl["oracle"]):
            return
        if "neuralq_dir" in bl and isinstance(bl["neuralq_dir"], str):
            return
        if "rule" in bl and isinstance(bl["rule"], list) and len(bl["rule"]) == 3:
            return
    raise ConfigError(f"score.baselines: 'flat', 'hold', 'delta30', {{'oracle': [th...]}}, {{'rule': [k, th, reverse]}} or "
                      f"{{'neuralq_dir': DIR, 'seeds': [...]}}; got {bl!r}")


def of_train_env_problems(env: dict) -> list[str]:
    """env keys that differ from of_train's (it supports market and warm-up only)."""
    return [k for k in ["latency_steps", "fee_dollars", "contracts", "decide_seconds", "episode_seconds", "sim_overrides"]
            if env[k] != DEFAULTS["env"][k]]


def cpu_budget() -> int:
    return int(os.environ.get("RAGRL_CPU_BUDGET", os.cpu_count() or 1))


def steps_per_episode(env: dict) -> int:
    """Decisions per episode, counted exactly as C++ steps_per_episode / ScalpEnv::step do (no module needed)."""
    d, e = float(env["decide_seconds"]), float(env["episode_seconds"])
    if not (d > 0 and e > 0):
        raise ConfigError("env.decide_seconds and env.episode_seconds must be > 0")
    elapsed, n = 0.0, 0
    while True:
        n += 1
        elapsed += d
        if elapsed >= e - 1e-9:
            return n


def canonical(cfg: dict) -> str:
    return json.dumps(cfg, sort_keys=True, separators=(",", ":"))


def sha256(cfg: dict) -> str:
    return hashlib.sha256(canonical(cfg).encode()).hexdigest()


def dump(cfg: dict, path):
    Path(path).write_text(json.dumps(cfg, indent=1, sort_keys=True) + "\n", encoding="utf-8")


def make_params(env: dict):
    from . import native
    R = native.load()
    return R.ScalpParams(env["market"], float(env["warmup_seconds"]), decide_seconds=float(env["decide_seconds"]),
                         episode_seconds=float(env["episode_seconds"]), latency_steps=int(env["latency_steps"]),
                         contracts=int(env["contracts"]), fee_dollars=float(env["fee_dollars"]),
                         sim_overrides={k: float(v) for k, v in env["sim_overrides"].items()})
