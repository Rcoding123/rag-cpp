"""Supervised baseline: LightGBM regression of the H-second mid change (ticks)
on the 13 order-flow features, and a cost-aware threshold policy.

Data: the FLAT policy on the run seed's train-stream episodes (privileged=True
on the train block, for the mid only: the label; X is the observation). One
model per horizon H; (H, theta) with theta = multiple x round-trip cost
(spread + 2 fees, ticks) are offered to the val_ckpt checkpointer, which keeps
the best (every choice counts toward the multiplicity)."""
import numpy as np

from ..policies import LGBMThreshold

TICK_DOLLARS = 12.5  # ES


def collect_flat(ctx):
    env = ctx.train_env(privileged=True)
    n, T = env.n_envs, ctx.steps_per_episode
    rounds = ctx.episodes // n
    X = np.empty((rounds, T, n, 16))
    mid = np.empty((rounds, T + 1, n))
    flat = np.ones(n, dtype=np.int64)
    obs = env.reset()
    for r in range(rounds):
        for t in range(T):
            X[r, t] = obs
            mid[r, t] = env.priv[:, 1]
            obs, _, done = env.step(flat)
        assert done[0]
        mid[r, T] = env.final_priv[:, 1]
    return X, mid


def train_lgbm(ctx):
    import lightgbm as lgb
    p = ctx.p
    X, mid = collect_flat(ctx)
    rounds, T, n, _ = X.shape
    fee_ticks = ctx.params.fee_dollars / TICK_DOLLARS
    cost = p["spread_ticks"] + 2 * fee_ticks
    ctx.extra.update(rows=int(rounds * T * n), round_trip_cost_ticks=cost)
    step = 0
    for H in p["horizons"]:
        h = int(round(H / ctx.params.decide_seconds))
        if not 1 <= h < T:
            raise ValueError(f"horizon {H} s is not inside an episode")
        xs = X[:, : T - h + 1].reshape(-1, 16)[:, :13]
        y = (mid[:, h:] - mid[:, : T + 1 - h])[:, : T - h + 1].reshape(-1)
        lp = {"objective": "regression", "num_leaves": p["num_leaves"], "learning_rate": p["learning_rate"],
              "min_child_samples": p["min_child_samples"], "bagging_fraction": p["subsample"], "bagging_freq": 1,
              "feature_fraction": p["colsample_bytree"], "deterministic": True, "force_row_wise": True, "num_threads": 1,
              "seed": ctx.seed, "verbose": -1}
        booster = lgb.train(lp, lgb.Dataset(xs, y, params={"verbose": -1}), num_boost_round=p["n_estimators"])
        r2 = 1 - float(np.mean((booster.predict(xs, num_threads=1) - y) ** 2) / max(1e-12, np.var(y)))
        ctx.extra[f"train_r2_H{H}"] = r2
        for m in p["theta_multiples"]:
            step += 1
            pol = LGBMThreshold(booster, m * cost, H)
            ctx.ckpt.offer(pol, step, ctx.episodes, {"horizon": H, "theta": m * cost})
            if ctx.live:
                ctx.live.ckpt(pol, ctx.episodes, [], ctx.ckpt.pop_evals(), 0.0)
    if ctx.live:
        ctx.live.done(ctx.ckpt.champion_score, "")
    ctx.extra.update(horizon=ctx.ckpt.champion.horizon, theta=ctx.ckpt.champion.theta)
