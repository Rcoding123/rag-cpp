"""SB3-facing environments over `_ragof.VecScalpEnv`.

RagVecEnv: an SB3 VecEnv on a train stream (auto-reset in C++). float32
observations (ScalpEnv::features), terminal observations and episode results in
infos, never the privileged channel. TeacherVecEnv (imitation): the same train
stream with privileged=True, exposing .priv for labels only.
"""
import numpy as np
from gymnasium import spaces
from stable_baselines3.common.vec_env.base_vec_env import VecEnv

from . import native

OBS_DIM = 16
N_ACTIONS = 3
OBS_SPACE = spaces.Box(low=-10.0, high=10.0, shape=(OBS_DIM,), dtype=np.float32)
ACT_SPACE = spaces.Discrete(N_ACTIONS)


def train_env(params, run_seed: int, n_envs: int, threads: int = 1, privileged: bool = False):
    R = native.load()
    return R.VecScalpEnv(params, R.EpisodeSource.train(int(run_seed)), n_envs=n_envs, threads=threads, privileged=privileged)


def indexed_source(block: str, first: int, count: int, orchestrator: str = ""):
    R = native.load()
    if orchestrator:
        return R.EpisodeSource.block(block, int(first), int(count), orchestrator=orchestrator)
    return R.EpisodeSource.block(block, int(first), int(count))


class RagVecEnv(VecEnv):
    """A VecScalpEnv train stream as an SB3 VecEnv (no privileged outputs)."""

    def __init__(self, venv, reward_scale: float = 1.0):
        if venv.privileged:
            raise ValueError("RagVecEnv takes a non-privileged environment (the agent sees observations only)")
        if venv.source.kind != "train":
            raise ValueError("RagVecEnv trains on a train stream (EpisodeSource.train)")
        self.venv = venv
        self.reward_scale = float(reward_scale)
        self.steps = venv.params.steps_per_episode
        self._actions = None
        self.episodes_done = 0
        super().__init__(venv.n_envs, OBS_SPACE, ACT_SPACE)

    def reset(self):
        if self.venv.episodes_started:  # a second reset would skip stream episodes
            raise RuntimeError("RagVecEnv.reset: the train stream is already running (episodes auto-reset in C++)")
        return self.venv.reset().astype(np.float32)

    def step_async(self, actions):
        self._actions = np.asarray(actions, dtype=np.int64).reshape(-1)

    def step_wait(self):
        obs, rew, done = self.venv.step(self._actions)
        infos = [{} for _ in range(self.num_envs)]
        if done.any():
            final = self.venv.final_obs.astype(np.float32)
            st = self.venv.episode_stats
            for i in np.flatnonzero(done):
                infos[i] = {"terminal_observation": final[i], "TimeLimit.truncated": False,
                            "episode": {"r": float(st["net_pnl_ticks"][i]), "l": self.steps, "t": 0.0,
                                        "orders": float(st["orders"][i])}}
            self.episodes_done += int(done.sum())
        return obs.astype(np.float32), (rew * self.reward_scale).astype(np.float32), done.copy(), infos

    def close(self):
        pass

    def _idx(self, indices):
        if indices is None:
            return range(self.num_envs)
        return [indices] if isinstance(indices, int) else indices

    def get_attr(self, attr_name, indices=None):
        return [getattr(self, attr_name, None) for _ in self._idx(indices)]

    def set_attr(self, attr_name, value, indices=None):
        pass

    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        return [None for _ in self._idx(indices)]

    def env_is_wrapped(self, wrapper_class, indices=None):
        return [False for _ in self._idx(indices)]


def keep_action(obs: np.ndarray) -> np.ndarray:
    """Keep the position (short/flat/long as now): BuiltinPolicy::keep on obs[13] = position / contracts."""
    return (np.sign(obs[:, 13]) + 1).astype(np.int64)


def oracle_actions(obs: np.ndarray, priv: np.ndarray, th: float) -> np.ndarray:
    """The fair-value oracle (BuiltinPolicy::oracle) from the privileged gap column: teacher labels only."""
    gap = priv[:, 2]
    return np.where(gap > th, 2, np.where(gap < -th, 0, keep_action(obs))).astype(np.int64)


def delta_rule_actions(obs: np.ndarray, th: float = 1.5) -> np.ndarray:
    """of_train's 30 s delta rule (BuiltinPolicy::delta30) on observations."""
    x = obs[:, 1]
    return np.where(x > th, 2, np.where(x < -th, 0, keep_action(obs))).astype(np.int64)
