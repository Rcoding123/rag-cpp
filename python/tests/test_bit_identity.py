"""The binding plays exactly ScalpEnv's episodes: observations, rewards,
privileged columns and episode results equal, byte for byte, to serial C++
loops (examples/of15_refdump.cpp), for of_train's train stream and for
indexed blocks; the C++ built-in rules and of_train networks match."""
import re
import subprocess

import numpy as np
import pytest

from conftest import REPO

D, P = 16, 7


def pattern(t, k):
    return (t * 7 + k * 3 + t // 13) % 3


def read_scalp(path, episodes, T):
    raw = np.fromfile(path, dtype=np.float64)
    pos, eps = 0, []
    for _ in range(episodes):
        obs, rew, priv = [raw[pos:pos + D]], [], [raw[pos + D:pos + D + P]]
        pos += D + P
        for _ in range(T):
            obs.append(raw[pos:pos + D])
            rew.append(raw[pos + D])
            priv.append(raw[pos + D + 1:pos + D + 1 + P])
            pos += D + 1 + P
        pnl, orders, fees = raw[pos:pos + 3]
        pos += 3
        eps.append(dict(obs=np.array(obs), rew=np.array(rew), priv=np.array(priv), pnl=pnl, orders=orders, fees=fees))
    assert pos == raw.size
    return eps


def dump(refdump, tmp_path, *args):
    out = tmp_path / ("ref_" + "_".join(args) + ".bin")
    subprocess.run([refdump, "scalp", *args, str(out)], check=True)
    return out


def same(a, b):
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    return a.shape == b.shape and a.tobytes() == b.tobytes()  # bytes: NaN fill prices compare equal too


def play_and_compare(env, ref, rounds, check_priv):
    """Plays `rounds` rounds with the pattern (k = episode id) and compares every step with ref[k]."""
    o = env.reset()
    for _ in range(rounds):
        ks = env.episode_ids.astype(np.int64)
        for j, k in enumerate(ks):
            assert same(o[j], ref[k]["obs"][0]), (k, "first obs")
            if check_priv:
                assert same(env.priv[j], ref[k]["priv"][0]), (k, "first priv")
        T = len(ref[ks[0]]["rew"])
        for t in range(T):
            o, r, d = env.step(np.array([pattern(t, k) for k in ks], dtype=np.int64))
            assert d.dtype == np.bool_ and bool(d.all()) == (t + 1 == T) and bool(d.any()) == (t + 1 == T)
            obs_now = env.final_obs if d[0] else o
            priv_now = (env.final_priv if d[0] else env.priv) if check_priv else None
            for j, k in enumerate(ks):
                assert same(r[j], ref[k]["rew"][t]), (k, t, "reward")
                assert same(obs_now[j], ref[k]["obs"][t + 1]), (k, t, "obs")
                if check_priv:
                    assert same(priv_now[j], ref[k]["priv"][t + 1]), (k, t, "priv")
        st = env.episode_stats
        assert list(env.final_episode_ids) == list(ks)
        for j, k in enumerate(ks):
            assert same(st["net_pnl_ticks"][j], ref[k]["pnl"])
            assert same(st["orders"][j], ref[k]["orders"])
            assert same(st["fees_ticks"][j], ref[k]["fees"])
            assert st["round_trips"][j] == ref[k]["orders"] // 2


def test_train_stream_v4_matches_trainer_stream(R, refdump, tmp_path):
    # of_train's Trainer stream for run seed 3 on discovery_v4 / 1050 s: 6 episodes as 3 envs x 2 rounds, 3 threads.
    p = R.ScalpParams("discovery_v4", 1050.0)
    ref = read_scalp(dump(refdump, tmp_path, "discovery_v4", "1050", "train", "3", "6"), 6, 600)
    env = R.VecScalpEnv(p, R.EpisodeSource.train(3), n_envs=3, threads=3, privileged=True)
    play_and_compare(env, ref, rounds=2, check_priv=True)
    assert env.episodes_started == 9 and env.rounds_finished == 2


@pytest.mark.parametrize("n_envs", [1, 4])
def test_train_stream_default_any_n_envs(R, refdump, tmp_path, n_envs):
    p = R.ScalpParams("default", 150.0)
    ref = read_scalp(dump(refdump, tmp_path, "default", "150", "train", "7", "4"), 4, 600)
    env = R.VecScalpEnv(p, R.EpisodeSource.train(7), n_envs=n_envs, threads=2, privileged=True)
    play_and_compare(env, ref, rounds=4 // n_envs, check_priv=True)


def test_validation_block_matches_evaluate(R, refdump, tmp_path):
    p = R.ScalpParams("default", 150.0)
    ref = read_scalp(dump(refdump, tmp_path, "default", "150", "seeds", "700000000", "4"), 4, 600)
    env = R.VecScalpEnv(p, R.EpisodeSource.block("validation", 0, 4), threads=4)
    assert env.n_envs == 4
    play_and_compare(env, ref, rounds=1, check_priv=False)


def test_viz_block_privileged_v4(R, refdump, tmp_path):
    p = R.ScalpParams("discovery_v4", 1050.0)
    ref = read_scalp(dump(refdump, tmp_path, "discovery_v4", "1050", "seeds", "960000002", "3"), 3, 600)
    env = R.VecScalpEnv(p, R.EpisodeSource.block("viz", 2, 3), threads=3, privileged=True)
    # viz episode i = Rng(960M + 2 + i); ids are block-relative, ref is indexed from 0
    o = env.reset()
    assert list(env.episode_ids) == [2, 3, 4]
    for j in range(3):
        assert same(o[j], ref[j]["obs"][0]) and same(env.priv[j], ref[j]["priv"][0])
    for t in range(600):
        o, r, d = env.step(np.array([pattern(t, j) for j in range(3)], dtype=np.int64))
        on, pn = (env.final_obs, env.final_priv) if d[0] else (o, env.priv)
        for j in range(3):
            assert same(r[j], ref[j]["rew"][t]) and same(on[j], ref[j]["obs"][t + 1]) and same(pn[j], ref[j]["priv"][t + 1])
    assert same(env.episode_stats["net_pnl_ticks"], [e["pnl"] for e in ref])


def test_builtin_delta30_equals_rule_on_obs(R):
    p = R.ScalpParams("default", 150.0, episode_seconds=300.0)
    src = R.EpisodeSource.block("validation", 100, 8)
    res = R.run_builtin(p, src, "delta30", threads=4)
    env = R.VecScalpEnv(p, src, threads=4)
    o = env.reset()
    while True:
        a = np.where(o[:, 1] > 1.5, 2, np.where(o[:, 1] < -1.5, 0, np.rint(o[:, 13]).astype(np.int64) + 1)).astype(np.int64)
        o, r, d = env.step(a)
        if d[0]:
            break
    assert same(res["net_pnl_ticks"], env.episode_stats["net_pnl_ticks"])
    assert same(res["orders"], env.episode_stats["orders"])
    assert same(res["fees_ticks"], env.episode_stats["fees_ticks"])
    # the rule as a tuple, and threads don't matter
    assert same(R.run_builtin(p, src, ("rule", 1, 1.5, False), threads=1)["net_pnl_ticks"], res["net_pnl_ticks"])


OF14 = REPO / "build-avx512" / "rl" / "of14"


@pytest.mark.skipif(not (OF14 / "seed1.net").exists(), reason="of14 networks not present (experiments/of14_regen.sh)")
def test_of14_networks_reproduce_validation_column(R):
    """of_train's champion validation scores (700M+0..39) in experiments/of14_train.txt, from the binding."""
    txt = (REPO / "experiments" / "of14_train.txt").read_text()
    want = {int(s): float(v) for s, v in re.findall(r"seed\s+(\d+): validation ([+-][\d.]+)", txt)}
    p = R.ScalpParams("discovery_v4", 1050.0)
    src = R.EpisodeSource.block("validation", 0, 40)
    for seed in (1, 5):
        net = R.NeuralQNet.load(str(OF14 / f"seed{seed}.net"), "discovery_v4", 1050.0)
        assert net.record["market"] == "discovery_v4"
        m = R.run_builtin(p, src, ("neuralq", net), threads=24)["net_pnl_ticks"].mean()
        assert f"{m:+.2f}" == f"{want[seed]:+.2f}", (seed, m, want[seed])
        if seed == 1:  # the Python loop (net.act on VecScalpEnv obs) plays the same episodes
            env = R.VecScalpEnv(p, src, threads=24)
            o = env.reset()
            while True:
                o, r, d = env.step(net.act(o))
                if d[0]:
                    break
            assert env.episode_stats["net_pnl_ticks"].mean() == m
    with pytest.raises(RuntimeError, match="market"):
        R.NeuralQNet.load(str(OF14 / "seed1.net"), "discovery", 1050.0)
    with pytest.raises(RuntimeError, match="warm-up"):
        R.NeuralQNet.load(str(OF14 / "seed1.net"), "discovery_v4", 150.0)


REPLAY_CHECK = """
csv, ref = sys.argv[1], sys.argv[2]
try:
    R.TickTape.from_csv(csv, time_unit="s")
    print("NOT REFUSED")
except RuntimeError:
    pass
R.open_sealed("real", "pytest replay identity")
tape = R.TickTape.from_csv(csv, time_unit="s")
rp = R.ReplayParams(window_seconds=60.0, warmup_seconds=30.0)
env = R.VecReplayEnv(tape, rp, [2, 3, 4], threads=3)
raw = np.fromfile(ref, dtype=np.float64)
T = 60
per = 16 + T * 17 + 3
assert raw.size == 3 * per
eps = [raw[i * per:(i + 1) * per] for i in range(3)]
o = env.reset()
ok = all(o[j].tobytes() == eps[j][:16].tobytes() for j in range(3))
for t in range(T):
    a = np.array([(t * 7 + k * 3 + t // 13) % 3 for k in range(3)], dtype=np.int64)
    o, r, d = env.step(a)
    on = env.final_obs if d[0] else o
    for j in range(3):
        base = 16 + t * 17
        ok &= on[j].tobytes() == eps[j][base:base + 16].tobytes()
        ok &= bool(r[j] == eps[j][base + 16])
    ok &= bool(d.all()) == (t + 1 == T)
st = env.episode_stats
ok &= all(st["net_pnl_ticks"][j] == eps[j][-3] and st["orders"][j] == eps[j][-2] for j in range(3))
print("REPLAY_OK" if ok else "REPLAY_MISMATCH", R.process_role())
"""


def test_replay_matches_replay_scalp_env(refdump, tmp_path, run_fresh):
    """VecReplayEnv (scorer process, open_sealed('real')) = serial ReplayScalpEnv::seek(k)."""
    csv = tmp_path / "ticks.csv"
    subprocess.run([refdump, "tape", "500", "9", str(csv)], check=True)
    ref = tmp_path / "replay.bin"
    subprocess.run([refdump, "replay", str(csv), "s", "60", "30", "2", "3", str(ref)], check=True)
    out = run_fresh(REPLAY_CHECK, str(csv), str(ref))
    assert "NOT REFUSED" not in out
    assert "REPLAY_OK scorer" in out
