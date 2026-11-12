"""One training worker = one run seed in its own process (role: trainer).

    python -m ragrl train RUN_DIR --seed K

forbid_sealed() first; trains the agent on EpisodeSource.train(K); the
champion (val_ckpt) is saved to RUN/seeds/seedK/ and re-loaded from disk and
re-scored, which must reproduce its validation score bit for bit."""
import os
import time
from pathlib import Path

import numpy as np

from . import config, envs, ledger, native, registry
from .agents import ZOO
from .checkpoints import ValCheckpointer
from .evaluate import run_policy
from .policies import load_policy, save_policy


def setup_torch(threads: int):
    import torch
    torch.set_num_threads(int(threads))
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    torch.use_deterministic_algorithms(True)


class Ctx:
    def __init__(self, cfg, seed, run_dir: Path, seed_dir: Path, live_path=None):
        self.R = native.load()
        self.cfg = cfg
        self.seed = int(seed)
        self.run_dir = run_dir
        self.seed_dir = seed_dir
        t = cfg["train"]
        self.params = config.make_params(cfg["env"])
        self.p = cfg["agent"]["params"]
        self.steps_per_episode = self.params.steps_per_episode
        self.episodes = t["episodes"]
        self.env_steps = t["env_steps"]
        self.n_envs = t["n_envs"]
        self.env_threads = t["env_threads"]
        self.torch_threads = t["torch_threads"]
        self.eval_threads = t["eval_threads"]
        self.reward_scale = t["reward_scale"]
        self.every_steps = cfg["select"]["every_steps"]
        self.metrics = registry.Metrics(seed_dir / "metrics.jsonl", seed=self.seed)
        self.ckpt = ValCheckpointer(self.params, cfg["select"]["val_ckpt"], self.eval_threads, self.metrics, self.seed)
        self.extra = {}
        self.t0 = time.perf_counter()
        self.live = None
        if live_path is not None:
            from .live import LiveWriter
            self.live = LiveWriter(live_path, self.params, cfg, self.seed, self.episodes,
                                   header_extra=ZOO[cfg["agent"]["kind"]].live_header(self.p), threads=self.eval_threads)

    def train_env(self, privileged=False, n_envs=None):
        return envs.train_env(self.params, self.seed, n_envs or self.n_envs, self.env_threads, privileged)

    def elapsed(self):
        return time.perf_counter() - self.t0


class Progress:
    """Shared bookkeeping for step-driven trainers: val checkpoints every `every_steps`, live checkpoints every N episodes."""

    def __init__(self, ctx: Ctx, freeze, epsilon=lambda: 0.0):
        self.ctx = ctx
        self.freeze = freeze
        self.epsilon = epsilon
        self.next_ckpt = ctx.every_steps
        self.next_live = ctx.live.every if ctx.live else None
        self.returns, self.all_returns = [], []
        self.episodes = 0
        self.last_offer = None
        if ctx.live:
            ctx.live.ckpt(freeze(), 0, [], [], epsilon())

    def update(self, steps: int, finished_returns):
        for r in finished_returns:
            self.returns.append(float(r))
            self.all_returns.append(float(r))
            self.episodes += 1
        ctx = self.ctx
        if steps >= self.next_ckpt:
            self.offer(steps)
            while self.next_ckpt <= steps:
                self.next_ckpt += ctx.every_steps
        if ctx.live and self.episodes >= self.next_live:
            ctx.live.ckpt(self.freeze(), self.episodes, self.returns, ctx.ckpt.pop_evals(), self.epsilon())
            ctx.metrics.write("train", step=steps, episode=self.episodes, mean_return=float(np.mean(self.returns)) if self.returns else None,
                              elapsed=ctx.elapsed())
            self.returns = []
            while self.next_live <= self.episodes:
                self.next_live += ctx.live.every

    def offer(self, steps):
        self.ctx.ckpt.offer(self.freeze(), steps, self.episodes)
        self.last_offer = steps

    def finish(self, steps):
        if self.last_offer != steps:
            self.offer(steps)
        ctx = self.ctx
        if ctx.live:
            if self.returns or ctx.ckpt.pending:
                ctx.live.ckpt(self.freeze(), self.episodes, self.returns, ctx.ckpt.pop_evals(), self.epsilon())
            ctx.live.done(ctx.ckpt.champion_score, "")
        ctx.metrics.write("train", step=steps, episode=self.episodes,
                          mean_return=float(np.mean(self.all_returns)) if self.all_returns else None, elapsed=ctx.elapsed(), final=True)


def train_worker(run_dir, seed: int, out_dir=None, live_path=None):
    ledger.as_trainer()
    run_dir = Path(run_dir)
    cfg = registry.read_json(run_dir / "config.json")
    setup_torch(cfg["train"]["torch_threads"])
    seed_dir = Path(out_dir) if out_dir else run_dir / "seeds" / f"seed{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    if live_path is None and cfg["live"]["enabled"] and int(seed) == int(cfg["live"]["seed"]) and out_dir is None:
        live_path = run_dir / "live.jsonl"
    ctx = Ctx(cfg, seed, run_dir, seed_dir, live_path)
    spec = ZOO[cfg["agent"]["kind"]]
    spec.train(ctx)
    train_seconds = ctx.elapsed()
    champ = ctx.ckpt.champion
    if champ is None:
        raise RuntimeError("the agent offered no policy")
    save_policy(champ, seed_dir, {"seed": int(seed), "agent": cfg["agent"]["kind"]})
    # Round trip: the saved champion, loaded back, must reproduce its validation score exactly.
    loaded = load_policy(seed_dir)
    res = run_policy(ctx.params, ctx.ckpt.source(), loaded, ctx.eval_threads, actions_hash=True)
    again = float(np.mean(res["net_pnl_ticks"]))
    if again != ctx.ckpt.champion_score:
        raise RuntimeError(f"saved champion re-scores {again!r} on val_ckpt, in memory {ctx.ckpt.champion_score!r}")
    val = {"seed": int(seed), "val_ckpt_mean": again, "champion_step": ctx.ckpt.champion_step, "offers": ctx.ckpt.offers,
           "choices_per_seed": spec.choices_per_seed(ctx.p), "actions_sha256": res["actions_sha256"],
           "train_seconds": train_seconds, "eval_seconds": ctx.ckpt.eval_seconds, "env_steps": ctx.env_steps,
           "episodes": ctx.episodes, "extra": ctx.extra}
    registry.write_json(seed_dir / "val.json", registry.finite(val))
    ctx.metrics.write("timing", seconds=ctx.elapsed(), train_seconds=train_seconds, eval_seconds=ctx.ckpt.eval_seconds,
                      env_steps=ctx.env_steps, steps_per_second=ctx.env_steps / max(1e-9, train_seconds - ctx.ckpt.eval_seconds),
                      pid=os.getpid())
    return val
