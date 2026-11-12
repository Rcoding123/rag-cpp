"""VecScalpEnv's contract: parameters, determinism (threads, n_envs), the
train stream's auto-reset, indexed sources played once, input checks, and
arrays that never alias the environment's buffers."""
import threading

import numpy as np
import pytest


def short(R, market="default", warmup=150.0, seconds=20.0, **kw):
    return R.ScalpParams(market, warmup, episode_seconds=seconds, **kw)


def test_params_round_trip(R):
    p = R.ScalpParams("discovery_v4", 1050.0, latency_steps=2, fee_dollars=2.0, sim_overrides={"value_vol": 0.4})
    d = p.to_dict()
    assert d == {"market": "discovery_v4", "warmup_seconds": 1050.0, "decide_seconds": 1.0, "episode_seconds": 600.0, "start_second": -1.0,
                 "latency_steps": 2, "contracts": 1, "fee_dollars": 2.0, "sim_overrides": {"value_vol": 0.4}}
    assert R.ScalpParams(**d) == p and R.ScalpParams(**{**d, "fee_dollars": 1.5}) != p
    assert p.steps_per_episode == 600 and R.steps_per_episode(short(R)) == 20
    assert "value_vol" in R.SIM_OVERRIDE_KEYS and "dt" not in R.SIM_OVERRIDE_KEYS
    for bad in [dict(market="nope"), dict(warmup_seconds=30_000.0), dict(warmup_seconds=-1.0), dict(sim_overrides={"dt": 0.2}),
                dict(market="discovery_v4", contracts=2), dict(decide_seconds=0.0), dict(contracts=0)]:
        with pytest.raises(ValueError):
            R.ScalpParams(**bad)
    assert R.BUILD_INFO["avx512"] in (True, False) and "nanobind" in R.BUILD_INFO


def run_stream(R, n_envs, threads, rounds, seed=11):
    """First obs and per-step rewards of stream episodes 0 .. n_envs * rounds - 1."""
    env = R.VecScalpEnv(short(R), R.EpisodeSource.train(seed), n_envs=n_envs, threads=threads)
    o = env.reset()
    first, rew = {}, {}
    for _ in range(rounds):
        ks = env.episode_ids.tolist()
        for j, k in enumerate(ks):
            first[k] = o[j].tobytes()
            rew[k] = []
        for t in range(20):
            o, r, d = env.step(np.array([(t + k) % 3 for k in ks], dtype=np.int64))
            for j, k in enumerate(ks):
                rew[k].append(r[j])
    return first, rew


def test_stream_invariant_to_threads_and_n_envs(R):
    a = run_stream(R, 4, 1, 2)
    assert a == run_stream(R, 4, 4, 2)
    assert a == run_stream(R, 2, 2, 4)
    assert a == run_stream(R, 8, 3, 1)
    assert run_stream(R, 4, 1, 2, seed=12) != a


def test_auto_reset_and_final_obs(R):
    env = R.VecScalpEnv(short(R, seconds=5.0), R.EpisodeSource.train(2), n_envs=3, threads=1)
    o = env.reset()
    assert env.episode_ids.tolist() == [0, 1, 2]
    for t in range(5):
        o, r, d = env.step(np.ones(3, dtype=np.int64))
    assert d.all() and env.final_episode_ids.tolist() == [0, 1, 2] and env.episode_ids.tolist() == [3, 4, 5]
    fo = env.final_obs
    assert np.all(fo[:, 15] == 0) and np.all(o[:, 15] == 1)  # time_left: terminal vs the next episode's first obs
    assert set(env.episode_stats) == {"net_pnl_ticks", "orders", "round_trips", "fees_ticks"}
    assert env.rounds_finished == 1 and env.episodes_started == 6


def test_indexed_played_once(R):
    env = R.VecScalpEnv(short(R, seconds=3.0), R.EpisodeSource.block("validation", 0, 2), n_envs=50)
    assert env.n_envs == 2
    with pytest.raises(RuntimeError):
        env.step(np.ones(2, dtype=np.int64))  # before reset
    env.reset()
    for _ in range(3):
        env.step(np.ones(2, dtype=np.int64))
    with pytest.raises(RuntimeError):
        env.step(np.ones(2, dtype=np.int64))
    with pytest.raises(RuntimeError):
        env.reset()


def test_action_checks_and_dtypes(R):
    env = R.VecScalpEnv(short(R), R.EpisodeSource.train(3), n_envs=2)
    env.reset()
    with pytest.raises(ValueError):
        env.step(np.array([0, 3], dtype=np.int64))
    with pytest.raises(ValueError):
        env.step(np.array([1, 1, 1], dtype=np.int64))
    o, r, d = env.step(np.array([2, 0], dtype=np.int32))  # converted
    o, r, d = env.step(np.array([2.0, 0.0])[::1].astype(np.int64))
    assert o.shape == (2, 16) and o.dtype == np.float64 and r.shape == (2,) and d.dtype == np.bool_


def test_arrays_are_copies(R):
    env = R.VecScalpEnv(short(R), R.EpisodeSource.train(4), n_envs=2)
    o = env.reset()
    o[:] = 99.0  # writable, owned by Python
    assert np.array_equal(env.final_obs, np.zeros((2, 16)))
    o2, _, _ = env.step(np.ones(2, dtype=np.int64))
    assert not (o2 == 99.0).any()
    ids = env.episode_ids
    ids[:] = 7
    assert env.episode_ids.tolist() == [0, 1]


def test_concurrent_use_is_refused_or_serialized(R):
    """Two Python threads stepping one env (GIL released inside) never corrupt it: one is refused."""
    env = R.VecScalpEnv(R.ScalpParams("default", 150.0), R.EpisodeSource.train(5), n_envs=64, threads=1)
    env.reset()
    errors = []

    def work():
        for _ in range(50):
            try:
                env.step(np.ones(64, dtype=np.int64))
            except RuntimeError as e:
                errors.append(str(e))

    ts = [threading.Thread(target=work) for _ in range(2)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    assert all("already in use" in e for e in errors)
