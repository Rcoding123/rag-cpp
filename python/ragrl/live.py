"""Live-viewer hook: writes the of_train_live JSONL schema (examples/of_train_live.cpp)
so tools/of_train_live.py (serve / bake) shows any agent's training.

    header  market, warmup, episodes, every, seed, hidden, gamma, nstep, lr, probe,
            probe_seed=960000000, imb_bins 10, delta_bins 8, fee_dollars, cell_states,
            rule_pnl, rule (the 30 s delta rule's probe track); extra: agent, score_kind
    ckpt    episode, elapsed, epsilon, returns, evals, probe_pnl, orders, long_share,
            short_share, policy[3*10*8*3], track{tod, mid, fv, pos, pnl, fill, qs, qf, ql}
    done    champion_validation, net

Probe episodes are the visualization block (960M+, look, never select), played
with privileged=True only to show the hidden fair value and fills; the policy
sees observations only. Policy-map states are the probes as a flat strategy
sees them, binned by L1 imbalance (obs[6] / 2: feature 6 is 2 x imbalance,
never clipped) x 30 s delta (obs[1]), exactly as of_train_live bins them.
Memory agents (frame stack, LSTM) see each map state as an episode's first step.
"""
import json
import math
import time

import numpy as np

from . import native
from .envs import delta_rule_actions, indexed_source
from .registry import finite

IMB_BINS, DELTA_BINS, CLIP = 10, 8, 4.0


def _bin(x, lo, hi, n):
    k = np.floor((x - lo) / (hi - lo) * n)
    return np.clip(k, 0.0, n - 1.0).astype(np.int64)


def _r(v, nd):
    out = []
    for x in np.asarray(v, dtype=np.float64).tolist():
        out.append(round(x, nd) if math.isfinite(x) else None)
    return out


def probe_states(params, probe: int):
    """Observations of the probe episodes under the flat policy (every decision; not the terminal one)."""
    R = native.load()
    env = R.VecScalpEnv(params, indexed_source("viz", 0, probe), n_envs=probe, threads=1)
    obs = env.reset()
    states = [obs]
    flat = np.ones(probe, dtype=np.int64)
    while True:
        obs, _, done = env.step(flat)
        if done[0]:
            break
        states.append(obs)
    # of_train_live's order: probe by probe, step by step
    return np.stack(states, axis=1).reshape(-1, 16)


def play_probes(params, probe: int, act_fn, threads: int = 1):
    """Plays act_fn(obs) -> (actions, display_scores or None) on the probes; returns results and the first probe's track."""
    R = native.load()
    env = R.VecScalpEnv(params, indexed_source("viz", 0, probe), n_envs=probe, threads=threads, privileged=True)
    obs = env.reset()
    priv = env.priv
    tr = {k: [] for k in ["tod", "mid", "fv", "pos", "pnl", "fill", "qs", "qf", "ql"]}
    pnl0, long_s, short_s, steps = 0.0, 0, 0, 0
    while True:
        a, q = act_fn(obs)
        tr["tod"].append(priv[0, 3])
        tr["mid"].append(priv[0, 1])
        tr["fv"].append(priv[0, 0])
        obs, rew, done = env.step(a)
        priv = env.final_priv if done[0] else env.priv
        o = env.final_obs if done[0] else obs
        pnl0 += rew[0]
        pos = np.rint(o[:, 13] * params.contracts).astype(int)
        tr["pos"].append(int(pos[0]))
        tr["pnl"].append(pnl0)
        tr["fill"].append(priv[0, 4])
        for k, j in (("qs", 0), ("qf", 1), ("ql", 2)):
            tr[k].append(q[0, j] if q is not None else float("nan"))
        long_s += int((pos > 0).sum())
        short_s += int((pos < 0).sum())
        steps += len(pos)
        if done[0]:
            break
    st = env.episode_stats
    track = {"tod": _r(tr["tod"], 1), "mid": _r(tr["mid"], 1), "fv": _r(tr["fv"], 2), "pos": tr["pos"], "pnl": _r(tr["pnl"], 2),
             "fill": _r(tr["fill"], 2), "qs": _r(tr["qs"], 3), "qf": _r(tr["qf"], 3), "ql": _r(tr["ql"], 3)}
    return {"pnl": st["net_pnl_ticks"], "orders": float(np.mean(st["orders"])), "long_share": long_s / steps,
            "short_share": short_s / steps, "track": track}


class LiveWriter:
    def __init__(self, path, params, cfg: dict, seed: int, episodes: int, header_extra: dict | None = None, threads: int = 1):
        self.path = path
        self.params = params
        lv = cfg["live"]
        self.probe = int(lv["probe"])
        self.every = int(lv["every_episodes"])
        self.threads = threads
        self.t0 = time.perf_counter()
        states = probe_states(params, self.probe)
        self.states = states
        self.bi = _bin(states[:, 6] / 2.0, -1.0, 1.0, IMB_BINS)
        self.bd = _bin(states[:, 1], -CLIP, CLIP, DELTA_BINS)
        cells = np.bincount(self.bi * DELTA_BINS + self.bd, minlength=IMB_BINS * DELTA_BINS)
        rule = play_probes(params, self.probe, lambda o: (delta_rule_actions(o), None), threads)
        head = {"type": "header", "market": params.market, "warmup": params.warmup_seconds, "episodes": int(episodes), "every": self.every,
                "seed": int(seed), "hidden": None, "gamma": None, "nstep": None, "lr": None, "probe": self.probe,
                "probe_seed": native.block("viz")["base"], "imb_bins": IMB_BINS, "delta_bins": DELTA_BINS, "fee_dollars": params.fee_dollars,
                "cell_states": cells.tolist(), "rule_pnl": _r(rule["pnl"], 2), "rule": rule["track"],
                "agent": cfg["agent"]["kind"], **(header_extra or {})}
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            f.write(json.dumps(finite(head), allow_nan=False) + "\n")

    def _emit(self, rec):
        with open(self.path, "a", encoding="utf-8", newline="\n") as f:
            f.write(json.dumps(finite(rec), allow_nan=False) + "\n")

    def policy_map(self, policy) -> list:
        pol = np.zeros(3 * IMB_BINS * DELTA_BINS * 3, dtype=np.int64)
        for held in (-1, 0, 1):
            x = self.states.copy()
            x[:, 13] = held
            x[:, 14] = 0.0
            policy.reset(len(x))
            a, _ = policy.step(x)
            idx = (((held + 1) * IMB_BINS + self.bi) * DELTA_BINS + self.bd) * 3 + a
            np.add.at(pol, idx, 1)
        return pol.tolist()

    def ckpt(self, policy, episode: int, returns, evals, epsilon: float = 0.0):
        policy.reset(self.probe)

        def act(o):
            a, s = policy.step(o)
            return a, policy.display_scores(o, s)

        pr = play_probes(self.params, self.probe, act, self.threads)
        pm = self.policy_map(policy)
        rec = {"type": "ckpt", "episode": int(episode), "elapsed": round(time.perf_counter() - self.t0, 1),
               "epsilon": float(epsilon), "returns": _r(returns, 2),
               "evals": [{"episode": int(e["episode"]), "challenger": e["challenger"], "champion": e["champion"],
                          "promoted": bool(e["promoted"])} for e in evals],
               "probe_pnl": _r(pr["pnl"], 2), "orders": round(pr["orders"], 2), "long_share": round(pr["long_share"], 4),
               "short_share": round(pr["short_share"], 4), "policy": pm, "track": pr["track"]}
        self._emit(rec)

    def done(self, champion_validation: float, net: str = ""):
        self._emit({"type": "done", "champion_validation": champion_validation if math.isfinite(champion_validation) else None,
                    "net": net})
