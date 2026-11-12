"""Paired evaluation: every policy plays the same episodes of one indexed source.

An indexed EpisodeSource is one lockstep round (slot i = Rng(base + first + i)),
so a FrozenPolicy (Python step loop, observations only) and a C++ built-in
(R.run_builtin: flat, hold, delta30, rules, the fair-value oracle, of_train
nets) evaluated on it play identical markets. Results are per-episode arrays.
"""
import hashlib

import numpy as np

from . import native
from .policies import FrozenPolicy, NeuralQ


class Builtin:
    """A C++ built-in policy, evaluated entirely in C++ (the oracle's fair value never reaches Python)."""

    def __init__(self, spec, name: str | None = None):
        self.spec = spec
        self.name = name or (spec if isinstance(spec, str) else "_".join(str(x) for x in spec))

    @staticmethod
    def oracle(th: float):
        return Builtin(("oracle", float(th)), f"oracle_{th:g}")

    def __repr__(self):
        return f"Builtin({self.spec!r})"


def run_policy(params, source, policy, threads: int = 1, actions_hash: bool = False) -> dict:
    """Plays `policy` on every episode of the indexed `source`; per-episode results."""
    R = native.load()
    if isinstance(policy, Builtin):
        res = R.run_builtin(params, source, policy.spec, threads=max(1, threads))
        return {"net_pnl_ticks": res["net_pnl_ticks"], "orders": res["orders"], "fees_ticks": res["fees_ticks"]}
    if isinstance(policy, NeuralQ) and not actions_hash:
        res = R.run_builtin(params, source, ("neuralq", policy.net), threads=max(1, threads))
        return {"net_pnl_ticks": res["net_pnl_ticks"], "orders": res["orders"], "fees_ticks": res["fees_ticks"]}
    if not isinstance(policy, FrozenPolicy):
        raise TypeError(f"not a policy: {policy!r}")
    env = R.VecScalpEnv(params, source, n_envs=source.count, threads=max(1, threads))
    if env.privileged:  # cannot happen (privileged must be asked for); kept as the evaluator's invariant
        raise RuntimeError("evaluation environments are never privileged")
    obs = env.reset()
    policy.reset(env.n_envs)
    h = hashlib.sha256() if actions_hash else None
    while True:
        a, _ = policy.step(obs)
        if h is not None:
            h.update(np.ascontiguousarray(a, dtype=np.int64).tobytes())
        obs, _, done = env.step(a)
        if done[0]:
            break
    st = env.episode_stats
    out = {"net_pnl_ticks": st["net_pnl_ticks"], "orders": st["orders"], "fees_ticks": st["fees_ticks"]}
    if h is not None:
        out["actions_sha256"] = h.hexdigest()
    return out


def paired_eval(params, source, policies: dict, threads: int = 1) -> dict:
    """{name: per-episode results} for every policy on the same episodes."""
    out = {}
    for name, pol in policies.items():
        out[name] = run_policy(params, source, pol, threads)
        if len(out[name]["net_pnl_ticks"]) != source.count:
            raise RuntimeError(f"{name}: {len(out[name]['net_pnl_ticks'])} episodes, expected {source.count}")
    return out


def _nq_baselines(cfg: dict):
    """[(key, source dir, seeds)] for the config's neuralq_dir baselines."""
    from pathlib import Path
    out = []
    for i, bl in enumerate(b for b in cfg["score"]["baselines"] if isinstance(b, dict) and "neuralq_dir" in b):
        d = Path(bl["neuralq_dir"])
        if not d.is_absolute():
            d = native.REPO / d
        seeds = bl.get("seeds") or sorted(int(p.stem[4:]) for p in d.glob("seed*.net"))
        out.append((f"nq{i}", d, [int(s) for s in seeds]))
    return out


def stage_baselines(cfg: dict, run_dir) -> dict:
    """Copies the baseline of_train nets (seedK.net + seedK.txt record) into RUN/baselines/nqI/ at run
    start and records their sha256 in RUN/baselines.json: select and seal load the copies, verified,
    so a rerun of of14_regen.sh between select and seal cannot change the paired references."""
    import shutil
    from pathlib import Path
    run_dir = Path(run_dir)
    out = {}
    for key, d, seeds in _nq_baselines(cfg):
        dst = run_dir / "baselines" / key
        dst.mkdir(parents=True, exist_ok=True)
        files = {}
        for s in seeds:
            for suffix in (".net", ".txt"):
                f = d / f"seed{s}{suffix}"
                if f.exists():
                    shutil.copy2(f, dst / f.name)
                    files[f.name] = native.sha256_file(dst / f.name)
                elif suffix == ".net":
                    raise FileNotFoundError(f"baseline net {f} missing (experiments/of14_regen.sh)")
        out[key] = {"source": str(d), "seeds": seeds, "sha256": files}
    from .registry import write_json
    write_json(run_dir / "baselines.json", out)
    return out


def baseline_policies(cfg: dict, run_dir=None) -> dict:
    """The config's baselines as {name: Builtin | FrozenPolicy} (NeuralQ nets: each seed and their mean-Q ensemble).
    With run_dir, NeuralQ nets come from the run's verified copies (stage_baselines); without, from their source."""
    from pathlib import Path

    from .policies import Ensemble
    out = {}
    env = cfg["env"]
    staged = None
    if run_dir is not None and _nq_baselines(cfg):
        from .registry import read_json
        bj = Path(run_dir) / "baselines.json"
        if not bj.exists():
            raise FileNotFoundError(f"{bj} missing: baseline nets are staged into the run at its start")
        staged = read_json(bj)
        for key, rec in staged.items():
            for name, sha in rec["sha256"].items():
                if native.sha256_file(Path(run_dir) / "baselines" / key / name) != sha:
                    raise RuntimeError(f"baseline {key}/{name} changed since the run started (sha256 mismatch)")
    nq = iter(_nq_baselines(cfg))
    for bl in cfg["score"]["baselines"]:
        if isinstance(bl, str):
            out[bl] = Builtin(bl)
        elif "oracle" in bl:
            for th in bl["oracle"]:
                b = Builtin.oracle(th)
                out[b.name] = b
        elif "rule" in bl:
            k, th, rev = bl["rule"]
            out[f"rule_{k}_{th:g}{'_rev' if rev else ''}"] = Builtin(("rule", int(k), float(th), bool(rev)))
        elif "neuralq_dir" in bl:
            key, d, seeds = next(nq)
            if staged is not None:
                d = Path(run_dir) / "baselines" / key
            nets = {s: NeuralQ.from_file(d / f"seed{s}.net", env["market"], env["warmup_seconds"]) for s in seeds}
            for s, n in nets.items():
                out[f"neuralq_ref_seed{s}"] = n
            out["neuralq_ref_ensemble"] = Ensemble(list(nets.values()))
    return out
