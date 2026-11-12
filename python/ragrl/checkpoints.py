"""Champion selection on validation (val_ckpt), as rag::agent::Trainer does:
every offered policy is frozen, scored on the same validation episodes as the
champion, and promoted if its mean is higher. Records go to metrics.jsonl and
(for the live viewer) to pending evals."""
import time

import numpy as np

from .envs import indexed_source
from .evaluate import run_policy


class ValCheckpointer:
    def __init__(self, params, val_ckpt: dict, threads: int, metrics, seed: int):
        self.params = params
        self.first, self.count = int(val_ckpt["first"]), int(val_ckpt["count"])
        self.threads = threads
        self.metrics = metrics
        self.seed = seed
        self.champion = None
        self.champion_score = float("nan")
        self.champion_step = None
        self.offers = 0
        self.pending = []  # evals since the last live checkpoint
        self.eval_seconds = 0.0

    def source(self):
        return indexed_source("validation", self.first, self.count)

    def score(self, policy) -> float:
        t0 = time.perf_counter()
        res = run_policy(self.params, self.source(), policy, self.threads)
        self.eval_seconds += time.perf_counter() - t0
        return float(np.mean(res["net_pnl_ticks"]))

    def offer(self, policy, step: int, episode: int, extra: dict | None = None) -> bool:
        s = self.score(policy)
        promoted = self.champion is None or s > self.champion_score
        if promoted:
            self.champion, self.champion_score, self.champion_step = policy, s, step
        self.offers += 1
        self.metrics.write("val", step=step, episode=episode, challenger=s, champion=self.champion_score, promoted=promoted,
                           **(extra or {}))
        self.pending.append({"episode": episode, "challenger": s, "champion": self.champion_score, "promoted": promoted})
        return promoted

    def pop_evals(self):
        out, self.pending = self.pending, []
        return out
