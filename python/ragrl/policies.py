"""Frozen policies: what every agent produces and what evaluation plays.

A FrozenPolicy sees observations only (float64 [N, 16], ScalpEnv::features):
    reset(n)             start n lockstep episodes (memory agents clear state)
    step(obs)            -> (actions int64[N] in {0 short, 1 flat, 2 long}, scores f64[N, 3])
    save(dir) -> meta    artifacts written into dir (policy.json via save_policy)
`scores` are Q-values (score_kind "q"), action probabilities ("prob") or votes
("vote"); an Ensemble averages members' scores and takes the first maximum, as
of_train's mean-Q ensemble does. Actions are the policy's greedy choice
(first maximum: Agent::act(o, false); SB3's deterministic predict).

Determinism contract: torch's CPU GEMM gives batch-LAYOUT-dependent last bits
(measured with the venv's torch: a row's scores alone, or as row 5 of a 5-row
batch, can differ from the same row inside a 300-row batch; padding to a
minimum row count does not remove it). Every evaluation here plays one indexed
source in one lockstep batch (n = source.count), so scoring the same policy on
the same source is bitwise repeatable (checked: select's determinism record,
R8), but the same policy on a different batch layout (e.g. one replay window,
or n_envs=1) may differ in the last bits of its scores and, at exact ties, in
an action. Compare such results paired, never bit for bit.
"""
import copy
import importlib
import json
from pathlib import Path

import numpy as np

from . import native

TYPES = {}


def register(cls):
    TYPES[cls.type] = cls
    return cls


def first_max(scores: np.ndarray) -> np.ndarray:
    return np.argmax(scores, axis=1).astype(np.int64)


class FrozenPolicy:
    type = "base"
    score_kind = "q"

    def reset(self, n: int):
        pass

    def step(self, obs: np.ndarray):
        raise NotImplementedError

    def display_scores(self, obs, scores):
        """What the live viewer plots as qs/qf/ql (Q-values, log-probabilities or model scores)."""
        if self.score_kind == "prob":
            return np.log(np.maximum(scores, 1e-12))
        return scores

    def save(self, d: Path) -> dict:
        raise NotImplementedError

    @classmethod
    def load(cls, d: Path, meta: dict):
        raise NotImplementedError


def save_policy(policy: FrozenPolicy, d: Path, extra: dict | None = None) -> dict:
    d = Path(d)
    d.mkdir(parents=True, exist_ok=True)
    meta = policy.save(d)
    files = sorted(p.name for p in d.iterdir() if p.is_file() and p.name not in ("policy.json", "val.json", "worker.log", "metrics.jsonl"))
    rec = {"type": policy.type, "score_kind": policy.score_kind, "meta": meta,
           "sha256": {f: native.sha256_file(d / f) for f in files if not f.endswith(".log")}, **(extra or {})}
    (d / "policy.json").write_text(json.dumps(rec, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    return rec


def load_policy(d: Path) -> FrozenPolicy:
    d = Path(d)
    rec = json.loads((d / "policy.json").read_text(encoding="utf-8"))
    for f, h in rec["sha256"].items():
        if native.sha256_file(d / f) != h:
            raise RuntimeError(f"{d / f}: artifact changed since it was saved (sha256)")
    return TYPES[rec["type"]].load(d, rec["meta"])


def _torch():
    import torch
    return torch


def ctor_data(policy) -> dict:
    """An SB3 policy's constructor parameters (sb3-contrib's recurrent policy omits its LSTM sizes: added here)."""
    data = policy._get_constructor_parameters()
    if hasattr(policy, "lstm_actor"):
        data.update(lstm_hidden_size=policy.lstm_output_dim, n_lstm_layers=policy.lstm_actor.num_layers,
                    shared_lstm=policy.shared_lstm, enable_critic_lstm=policy.enable_critic_lstm)
    return data


def _class_path(obj) -> str:
    c = type(obj)
    return f"{c.__module__}:{c.__qualname__}"


def _import(path: str):
    mod, name = path.split(":")
    return getattr(importlib.import_module(mod), name)


@register
class SB3ActorCritic(FrozenPolicy):
    """PPO-style policy (optionally on a frame stack, as VecFrameStack builds it); scores = action probabilities."""
    type = "sb3_ac"
    score_kind = "prob"

    def __init__(self, policy, n_stack: int = 1):
        self.policy = policy
        self.policy.set_training_mode(False)
        self.n_stack = int(n_stack)
        self._stack = None

    def reset(self, n):
        self._stack = np.zeros((n, 16 * self.n_stack), dtype=np.float32) if self.n_stack > 1 else None

    def _input(self, obs):
        x = obs.astype(np.float32)
        if self.n_stack == 1:
            return x
        if self._stack is None or self._stack.shape[0] != x.shape[0]:
            self.reset(x.shape[0])
        self._stack = np.roll(self._stack, shift=-16, axis=-1)
        self._stack[:, -16:] = x
        return self._stack

    def step(self, obs):
        th = _torch()
        with th.no_grad():
            dist = self.policy.get_distribution(th.as_tensor(self._input(obs)))
            probs = dist.distribution.probs
            act = th.argmax(probs, dim=1)
        return act.numpy().astype(np.int64), probs.numpy().astype(np.float64)

    def save(self, d):
        self.policy.save(str(d / "policy.pt"))
        return {"class": _class_path(self.policy), "n_stack": self.n_stack}

    @classmethod
    def load(cls, d, meta):
        return cls(_import(meta["class"]).load(str(d / "policy.pt"), device="cpu"), meta["n_stack"])


@register
class SB3Recurrent(FrozenPolicy):
    """RecurrentPPO (LSTM) policy; the actor's LSTM state is carried through the episode."""
    type = "sb3_recurrent"
    score_kind = "prob"

    def __init__(self, policy):
        self.policy = policy
        self.policy.set_training_mode(False)
        self._states = None
        self._first = True

    def reset(self, n):
        th = _torch()
        layers, _, hidden = self.policy.lstm_hidden_state_shape
        z = th.zeros((layers, n, hidden), dtype=th.float32)
        self._states = (z, z.clone())
        self._first = True

    def step(self, obs):
        th = _torch()
        n = obs.shape[0]
        if self._states is None or self._states[0].shape[1] != n:
            self.reset(n)
        starts = th.full((n,), 1.0 if self._first else 0.0, dtype=th.float32)
        with th.no_grad():
            dist, self._states = self.policy.get_distribution(th.as_tensor(obs.astype(np.float32)), self._states, starts)
            probs = dist.distribution.probs
            act = th.argmax(probs, dim=1)
        self._first = False
        return act.numpy().astype(np.int64), probs.numpy().astype(np.float64)

    def save(self, d):
        th = _torch()
        data = ctor_data(self.policy)
        th.save({"state_dict": self.policy.state_dict(), "data": data}, d / "policy.pt")
        return {"class": _class_path(self.policy)}

    @classmethod
    def load(cls, d, meta):
        saved = _torch().load(d / "policy.pt", map_location="cpu", weights_only=False)
        pol = _import(meta["class"])(**saved["data"])
        pol.load_state_dict(saved["state_dict"])
        return cls(pol)


@register
class SB3Q(FrozenPolicy):
    """DQN / QR-DQN policy: scores = Q-values (QR-DQN: the mean over quantiles)."""
    type = "sb3_q"
    score_kind = "q"

    def __init__(self, policy, quantile: bool = False):
        self.policy = policy
        self.policy.set_training_mode(False)
        self.quantile = bool(quantile)

    def step(self, obs):
        th = _torch()
        with th.no_grad():
            x = th.as_tensor(obs.astype(np.float32))
            q = self.policy.quantile_net(x).mean(dim=1) if self.quantile else self.policy.q_net(x)
            act = q.argmax(dim=1)
        return act.numpy().astype(np.int64), q.numpy().astype(np.float64)

    def save(self, d):
        self.policy.save(str(d / "policy.pt"))
        return {"class": _class_path(self.policy), "quantile": self.quantile}

    @classmethod
    def load(cls, d, meta):
        return cls(_import(meta["class"]).load(str(d / "policy.pt"), device="cpu"), meta["quantile"])


@register
class MLPStudent(FrozenPolicy):
    """A small torch MLP (imitation student) on observations; scores = probabilities."""
    type = "mlp_student"
    score_kind = "prob"

    def __init__(self, module, hidden):
        self.module = module.eval()
        self.hidden = list(hidden)

    @staticmethod
    def build(hidden):
        th = _torch()
        layers, w = [], 16
        for h in hidden:
            layers += [th.nn.Linear(w, h), th.nn.Tanh()]
            w = h
        layers.append(th.nn.Linear(w, 3))
        return th.nn.Sequential(*layers)

    def step(self, obs):
        th = _torch()
        with th.no_grad():
            logits = self.module(th.as_tensor(obs.astype(np.float32)))
            probs = th.softmax(logits, dim=1)
            act = th.argmax(logits, dim=1)
        return act.numpy().astype(np.int64), probs.numpy().astype(np.float64)

    def save(self, d):
        _torch().save(self.module.state_dict(), d / "student.pt")
        return {"hidden": self.hidden}

    @classmethod
    def load(cls, d, meta):
        m = cls.build(meta["hidden"])
        m.load_state_dict(_torch().load(d / "student.pt", map_location="cpu", weights_only=True))
        return cls(m, meta["hidden"])


@register
class LGBMThreshold(FrozenPolicy):
    """Supervised baseline: LightGBM predicts the H-second mid change (ticks) from the 13 features;
    long above +theta, short below -theta, else keep the position. Scores are votes."""
    type = "lgbm"
    score_kind = "vote"

    def __init__(self, booster, theta: float, horizon: int):
        self.booster = booster
        self.theta = float(theta)
        self.horizon = int(horizon)
        self._last_pred = None

    def predict(self, obs):
        return self.booster.predict(np.ascontiguousarray(obs[:, :13]), num_threads=1)

    def step(self, obs):
        from .envs import keep_action
        r = self.predict(obs)
        self._last_pred = r
        a = np.where(r > self.theta, 2, np.where(r < -self.theta, 0, keep_action(obs))).astype(np.int64)
        s = np.zeros((obs.shape[0], 3))
        s[np.arange(obs.shape[0]), a] = 1.0
        return a, s

    def display_scores(self, obs, scores):
        r = self._last_pred if self._last_pred is not None else self.predict(obs)
        return np.stack([-r - self.theta, np.zeros_like(r), r - self.theta], axis=1)

    def save(self, d):
        self.booster.save_model(str(d / "model.txt"))
        return {"theta": self.theta, "horizon": self.horizon}

    @classmethod
    def load(cls, d, meta):
        import lightgbm as lgb
        return cls(lgb.Booster(model_file=str(d / "model.txt")), meta["theta"], meta["horizon"])


@register
class NeuralQ(FrozenPolicy):
    """An of_train network (seedK.net + record), loaded through R.NeuralQNet (check_net enforced)."""
    type = "neuralq"
    score_kind = "q"

    def __init__(self, net, market: str, warmup: float):
        self.net = net
        self.market = market
        self.warmup = float(warmup)

    @classmethod
    def from_file(cls, path, market, warmup):
        return cls(native.load().NeuralQNet.load(str(path), market, float(warmup)), market, warmup)

    def step(self, obs):
        obs = np.ascontiguousarray(obs, dtype=np.float64)
        return self.net.act(obs), self.net.q(obs)

    def save(self, d):
        import shutil
        src = Path(self.net.path)
        if src.resolve() != (d / "seed.net").resolve():
            shutil.copy2(src, d / "seed.net")
            shutil.copy2(src.with_suffix(".txt"), d / "seed.txt")
        return {"market": self.market, "warmup": self.warmup, "source": str(src)}

    @classmethod
    def load(cls, d, meta):
        return cls.from_file(d / "seed.net", meta["market"], meta["warmup"])


@register
class Ensemble(FrozenPolicy):
    """Mean of the members' scores (Q-values, probabilities or votes), first maximum."""
    type = "ensemble"

    def __init__(self, members):
        kinds = {m.score_kind for m in members}
        if len(kinds) != 1:
            raise ValueError(f"an ensemble needs members of one score kind, got {kinds}")
        self.members = list(members)
        self.score_kind = kinds.pop()

    def reset(self, n):
        for m in self.members:
            m.reset(n)

    def step(self, obs):
        s = None
        for m in self.members:
            _, sm = m.step(obs)
            s = sm.copy() if s is None else s + sm
        s /= len(self.members)
        return first_max(s), s

    def save(self, d):
        metas = []
        for i, m in enumerate(self.members):
            sub = Path(d) / f"m{i}"
            metas.append(save_policy(m, sub))
        return {"members": len(metas)}

    @classmethod
    def load(cls, d, meta):
        return cls([load_policy(Path(d) / f"m{i}") for i in range(meta["members"])])


def freeze_module(module):
    """A detached copy of a torch module (a frozen snapshot of a learner)."""
    m = copy.deepcopy(module)
    for p in m.parameters():
        p.requires_grad_(False)
    return m
