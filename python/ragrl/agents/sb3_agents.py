"""SB3 / sb3-contrib agents: PPO (MLP, optional frame stack), RecurrentPPO (LSTM),
DQN made NeuralQ-like (Double DQN, n-step, Polyak-averaged evaluation weights,
of_train's replay ratio: 16 envs x gradient_steps 2 x batch 256 = 32 samples per
transition, as NeuralQ's batch 32 per step) and QR-DQN.

Determinism: torch threads fixed by the worker, use_deterministic_algorithms,
SB3 seed = run seed; the environment's episodes come from the train stream."""
import copy

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from ..envs import RagVecEnv
from ..policies import SB3ActorCritic, SB3Q, SB3Recurrent
from ..worker import Progress


def _act_fn(name):
    import torch.nn as nn
    return {"tanh": nn.Tanh, "relu": nn.ReLU}[name]


class Hooks(BaseCallback):
    def __init__(self, progress: Progress):
        super().__init__(0)
        self.progress = progress

    def _on_step(self) -> bool:
        dones = self.locals["dones"]
        infos = self.locals["infos"]
        fin = [infos[i]["episode"]["r"] for i in np.flatnonzero(dones)]
        self.progress.update(self.num_timesteps, fin)
        return True


def _frozen_copy(policy):
    """A detached copy of an SB3 policy: rebuilt from its constructor parameters, weights copied.
    The torch RNG is restored afterwards, so freezing (checkpoints, live probes) never changes training."""
    import torch as th
    from ..policies import ctor_data
    with th.random.fork_rng(devices=[]):
        p = type(policy)(**ctor_data(policy))
    p.load_state_dict(policy.state_dict())
    p.optimizer = None
    for x in p.parameters():
        x.requires_grad_(False)
    p.set_training_mode(False)
    return p


def _ppo_kwargs(p):
    arch = list(p["net_arch"])
    return dict(learning_rate=p["learning_rate"], n_steps=p["n_steps"], batch_size=p["batch_size"], n_epochs=p["n_epochs"],
                gamma=p["gamma"], gae_lambda=p["gae_lambda"], clip_range=p["clip_range"], ent_coef=p["ent_coef"], vf_coef=p["vf_coef"],
                max_grad_norm=p["max_grad_norm"]), {"net_arch": {"pi": arch, "vf": arch}, "activation_fn": _act_fn(p["activation"])}


def _learn(ctx, model, freeze, epsilon=lambda: 0.0):
    prog = Progress(ctx, freeze, epsilon)
    model.learn(total_timesteps=ctx.env_steps, callback=Hooks(prog), progress_bar=False)
    prog.finish(model.num_timesteps)
    ctx.extra["env_steps_done"] = int(model.num_timesteps)


def train_ppo(ctx):
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import VecFrameStack
    p = ctx.p
    n_stack = int(p.get("n_stack", 1))
    env = RagVecEnv(ctx.train_env(), ctx.reward_scale)
    if n_stack > 1:
        env = VecFrameStack(env, n_stack)
    kw, pk = _ppo_kwargs(p)
    model = PPO("MlpPolicy", env, policy_kwargs=pk, seed=ctx.seed, device="cpu", verbose=0, **kw)
    _learn(ctx, model, lambda: SB3ActorCritic(_frozen_copy(model.policy), n_stack))


def train_recurrent_ppo(ctx):
    from sb3_contrib import RecurrentPPO
    p = ctx.p
    env = RagVecEnv(ctx.train_env(), ctx.reward_scale)
    kw, pk = _ppo_kwargs(p)
    pk.update(lstm_hidden_size=p["lstm_hidden_size"], n_lstm_layers=p["n_lstm_layers"])
    model = RecurrentPPO("MlpLstmPolicy", env, policy_kwargs=pk, seed=ctx.seed, device="cpu", verbose=0, **kw)
    _learn(ctx, model, lambda: SB3Recurrent(_frozen_copy(model.policy)))


def _dqn_kwargs(p):
    return dict(learning_rate=p["learning_rate"], buffer_size=p["buffer_size"], learning_starts=p["learning_starts"],
                batch_size=p["batch_size"], gamma=p["gamma"], n_steps=p["n_steps_return"], train_freq=p["train_freq"],
                gradient_steps=p["gradient_steps"], target_update_interval=p["target_update_interval"], tau=p["tau"],
                exploration_fraction=p["exploration_fraction"], exploration_initial_eps=p["exploration_initial_eps"],
                exploration_final_eps=p["exploration_final_eps"], max_grad_norm=p["max_grad_norm"])


def _dqn_classes():
    import torch as th
    import torch.nn.functional as F
    from sb3_contrib import QRDQN
    from stable_baselines3 import DQN
    from stable_baselines3.common.utils import polyak_update

    class NeuralQLikeDQN(DQN):
        """DQN with an optional Double-DQN target (online argmax, target value) and a Polyak
        average of the online weights (NeuralQ's nn_polyak) used for evaluation only."""

        def __init__(self, *a, double_q=True, polyak_eval=1e-4, **k):
            self.double_q = bool(double_q)
            self.polyak_eval = float(polyak_eval)
            super().__init__(*a, **k)

        def _setup_model(self):
            super()._setup_model()
            self.q_net_avg = copy.deepcopy(self.q_net)
            for x in self.q_net_avg.parameters():
                x.requires_grad_(False)

        def _excluded_save_params(self):
            return [*super()._excluded_save_params(), "q_net_avg"]

        def train(self, gradient_steps, batch_size=100):
            self.policy.set_training_mode(True)
            self._update_learning_rate(self.policy.optimizer)
            losses = []
            for _ in range(gradient_steps):
                d = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)
                discounts = d.discounts if d.discounts is not None else self.gamma
                with th.no_grad():
                    nq_target = self.q_net_target(d.next_observations)
                    if self.double_q:
                        best = self.q_net(d.next_observations).argmax(dim=1, keepdim=True)
                        nq = th.gather(nq_target, dim=1, index=best)
                    else:
                        nq = nq_target.max(dim=1)[0].reshape(-1, 1)
                    target = d.rewards + (1 - d.dones) * discounts * nq
                q = th.gather(self.q_net(d.observations), dim=1, index=d.actions.long())
                loss = F.smooth_l1_loss(q, target)
                losses.append(loss.item())
                self.policy.optimizer.zero_grad()
                loss.backward()
                th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()
                if self.polyak_eval > 0:
                    polyak_update(self.q_net.parameters(), self.q_net_avg.parameters(), self.polyak_eval)
            self._n_updates += gradient_steps
            self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
            self.logger.record("train/loss", float(np.mean(losses)))

        def frozen(self):
            pol = _frozen_copy(self.policy)
            if self.polyak_eval > 0:
                pol.q_net.load_state_dict(self.q_net_avg.state_dict())
            return SB3Q(pol, quantile=False)

    class AveragedQRDQN(QRDQN):
        """QR-DQN with a Polyak average of the online quantile network used for evaluation only."""

        def __init__(self, *a, polyak_eval=1e-4, **k):
            self.polyak_eval = float(polyak_eval)
            super().__init__(*a, **k)

        def _setup_model(self):
            super()._setup_model()
            self.quantile_net_avg = copy.deepcopy(self.quantile_net)
            for x in self.quantile_net_avg.parameters():
                x.requires_grad_(False)

        def _excluded_save_params(self):
            return [*super()._excluded_save_params(), "quantile_net_avg"]

        def train(self, gradient_steps, batch_size=100):
            for _ in range(gradient_steps):  # one update at a time, averaging after each (same updates as QRDQN.train)
                super().train(1, batch_size)
                if self.polyak_eval > 0:
                    polyak_update(self.quantile_net.parameters(), self.quantile_net_avg.parameters(), self.polyak_eval)

        def frozen(self):
            pol = _frozen_copy(self.policy)
            if self.polyak_eval > 0:
                pol.quantile_net.load_state_dict(self.quantile_net_avg.state_dict())
            return SB3Q(pol, quantile=True)

    return NeuralQLikeDQN, AveragedQRDQN


def train_dqn(ctx):
    DQNc, _ = _dqn_classes()
    p = ctx.p
    env = RagVecEnv(ctx.train_env(), ctx.reward_scale)
    model = DQNc("MlpPolicy", env, double_q=p["double_q"], polyak_eval=p["polyak_eval"],
                 policy_kwargs={"net_arch": list(p["net_arch"])}, seed=ctx.seed, device="cpu", verbose=0, **_dqn_kwargs(p))
    _learn(ctx, model, model.frozen, lambda: float(model.exploration_rate))


def train_qrdqn(ctx):
    _, QRc = _dqn_classes()
    p = ctx.p
    env = RagVecEnv(ctx.train_env(), ctx.reward_scale)
    model = QRc("MlpPolicy", env, polyak_eval=p["polyak_eval"],
                policy_kwargs={"net_arch": list(p["net_arch"]), "n_quantiles": p["n_quantiles"]}, seed=ctx.seed, device="cpu",
                verbose=0, **_dqn_kwargs(p))
    _learn(ctx, model, model.frozen, lambda: float(model.exploration_rate))
