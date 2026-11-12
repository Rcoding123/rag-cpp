"""The agent zoo: name -> AgentSpec (defaults, n_envs, train function).

Every agent trains on EpisodeSource.train(run_seed) (of_train's Trainer
stream for that run seed) and offers frozen policies to ctx.ckpt (champion on
val_ckpt). Budgets are env steps (decisions); the default 1.2M = of_train's
2000 episodes x 600 decisions, rounded up to whole rounds of n_envs episodes.
"""
from dataclasses import dataclass, field
from typing import Callable


@dataclass(frozen=True)
class AgentSpec:
    name: str
    description: str
    defaults: dict
    n_envs: int
    train: Callable  # train(ctx) -> None; offers policies to ctx.ckpt
    prepare: Callable | None = None  # prepare(cfg, run_dir) in the parent before the workers
    choices_per_seed: Callable = field(default=lambda p: 1)  # validation choices beyond checkpoints (multiplicity)
    live_header: Callable = field(default=lambda p: {})


def _lazy(module, fn):
    def call(*a, **k):
        import importlib
        return getattr(importlib.import_module(f"ragrl.agents.{module}"), fn)(*a, **k)
    call.__name__ = fn
    return call


PPO_DEFAULTS = {"learning_rate": 3e-4, "n_steps": 300, "batch_size": 1200, "n_epochs": 4, "gamma": 0.99, "gae_lambda": 0.95,
                "clip_range": 0.2, "ent_coef": 0.01, "vf_coef": 0.5, "max_grad_norm": 0.5, "net_arch": [64, 64], "activation": "tanh"}
DQN_DEFAULTS = {"learning_rate": 1e-3, "buffer_size": 300_000, "learning_starts": 2000, "batch_size": 256, "gamma": 0.97,
                "n_steps_return": 5, "train_freq": 1, "gradient_steps": 2, "target_update_interval": 16_000, "tau": 1.0,
                "exploration_fraction": 0.5, "exploration_initial_eps": 1.0, "exploration_final_eps": 0.02, "max_grad_norm": 10.0,
                "net_arch": [64, 64], "double_q": True, "polyak_eval": 1e-4}


def _sb3_header(p):
    return {"hidden": (p.get("net_arch") or [None])[0], "gamma": p.get("gamma"), "lr": p.get("learning_rate"),
            "nstep": p.get("n_steps_return"), "score_kind": "q" if "buffer_size" in p else "log_prob"}


ZOO = {
    "ppo": AgentSpec("ppo", "SB3 PPO, MLP policy/value [64, 64] tanh", dict(PPO_DEFAULTS), 16, _lazy("sb3_agents", "train_ppo"),
                     live_header=_sb3_header),
    "ppo_framestack": AgentSpec("ppo_framestack", "SB3 PPO on VecFrameStack(n_stack) observations (flow memory)",
                                dict(PPO_DEFAULTS, n_stack=8), 16, _lazy("sb3_agents", "train_ppo"), live_header=_sb3_header),
    "recurrent_ppo": AgentSpec("recurrent_ppo", "sb3-contrib RecurrentPPO, MlpLstmPolicy (flow memory)",
                               dict(PPO_DEFAULTS, n_steps=120, batch_size=3000, lstm_hidden_size=64, n_lstm_layers=1), 50,
                               _lazy("sb3_agents", "train_recurrent_ppo"), live_header=_sb3_header),
    "dqn": AgentSpec("dqn", "SB3 DQN made NeuralQ-like: Double DQN, n-step 5, Polyak-averaged eval weights, of_train's replay ratio",
                     dict(DQN_DEFAULTS), 16, _lazy("sb3_agents", "train_dqn"), live_header=_sb3_header),
    "qrdqn": AgentSpec("qrdqn", "sb3-contrib QR-DQN (distributional), n-step 5, Polyak-averaged eval weights",
                       {**{k: v for k, v in DQN_DEFAULTS.items() if k != "double_q"}, "n_quantiles": 32}, 16,
                       _lazy("sb3_agents", "train_qrdqn"), live_header=_sb3_header),
    "lgbm_threshold": AgentSpec("lgbm_threshold",
                                "LightGBM regression of the H-second mid change on the 13 features; cost-aware threshold policy",
                                {"horizons": [5, 10, 30], "theta_multiples": [0.5, 1.0, 1.5, 2.0, 3.0], "spread_ticks": 1.0,
                                 "num_leaves": 31, "n_estimators": 400, "learning_rate": 0.05, "min_child_samples": 200,
                                 "subsample": 0.8, "colsample_bytree": 0.8},
                                20, _lazy("supervised", "train_lgbm"),  # every (H, theta) is a val_ckpt offer
                                live_header=lambda p: {"score_kind": "lgbm_prediction_minus_theta"}),
    "bc_dagger": AgentSpec("bc_dagger", "Imitation of the privileged fair-value oracle: behaviour cloning + DAgger, student sees obs only",
                           {"teacher_thresholds": [0.5, 0.55, 0.75, 1.0, 2.0], "bc_fraction": 0.25, "dagger_iterations": 5,
                            "beta0": 0.5, "hidden": [64, 64], "epochs": 10, "batch_size": 4096, "learning_rate": 1e-3},
                           20, _lazy("imitation", "train_bc_dagger"),
                           choices_per_seed=lambda p: len(p["teacher_thresholds"]),
                           live_header=lambda p: {"hidden": p["hidden"][0], "lr": p["learning_rate"], "score_kind": "log_prob"}),
    "neuralq_ref": AgentSpec("neuralq_ref", "The C++ NeuralQ reference (of_train: Double DQN, Polyak 1e-4, hidden 64, gamma 0.97, n-step 5)",
                             {"nets_dir": None, "run_of_train": True, "hidden": 64, "gamma": 0.97, "nstep": 5, "batch": 32, "lr": 1e-3},
                             1, _lazy("neuralq", "train_neuralq_ref"), prepare=_lazy("neuralq", "prepare_neuralq_ref"),
                             # of_train picked each champion from its 20 evaluations (eval_every = episodes / 20) on
                             # validation [0, 40): that choice counts in M (config.resolve keeps val_ckpt off [0, 40))
                             choices_per_seed=lambda p: 20,
                             live_header=lambda p: {"hidden": p["hidden"], "gamma": p["gamma"], "nstep": p["nstep"], "lr": p["lr"]}),
}
