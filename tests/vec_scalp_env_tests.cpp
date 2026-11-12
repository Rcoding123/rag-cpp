// vec_scalp_env_tests.cpp — OF15's batched ScalpEnv (vec_scalp_env.hpp) and
// seed ledger (seed_ledger.hpp): the same episodes, observations and rewards
// as serial ScalpEnv loops (of_train's Trainer stream, evaluate()/score(),
// OF8's days, OF6's stresses, the replay env), whatever the thread count, and
// the ledger's refusals. Each test uses its own Ledger (the process one is the
// Python module's).
#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <vector>

#include "rag/orderflow/vec_scalp_env.hpp"

using namespace rag::orderflow;
using rag::agent::Action;
using rag::agent::Rng;
using seeds::Block;
using seeds::Ledger;

namespace {

constexpr std::size_t D = VecScalpEnv::D;

// The fixed action pattern of examples/of15_refdump.cpp: decision t of the run's episode k.
std::int64_t pattern(std::uint64_t t, std::uint64_t k) { return static_cast<std::int64_t>((t * 7 + k * 3 + t / 13) % 3); }

struct Episode {
  std::vector<double> obs;  // (T + 1) x D
  std::vector<double> rew;
  double pnl = 0, orders = 0, fees = 0;
};

// One serial ScalpEnv episode on `rng` (the Trainer's loop: reset and step on one stream).
Episode serial(const ScalpParams& p, Rng& rng, std::uint64_t k) {
  ScalpEnv env(p);
  Episode e;
  auto o = env.reset(rng);
  auto x = env.features(o);
  e.obs.insert(e.obs.end(), x.begin(), x.end());
  for (std::uint64_t t = 0;; ++t) {
    const auto st = env.step(static_cast<Action>(pattern(t, k)), rng);
    x = env.features(st.obs);
    e.obs.insert(e.obs.end(), x.begin(), x.end());
    e.rew.push_back(st.reward);
    if (st.done) break;
  }
  e.pnl = env.net_pnl_ticks();
  e.orders = env.orders();
  e.fees = env.fees_ticks();
  return e;
}

bool same_bytes(const double* a, const double* b, std::size_t n) { return std::memcmp(a, b, n * sizeof(double)) == 0; }

// Plays one round of `env` with the pattern (k = the slot's episode id) and
// checks every slot against want[id] bit for bit.
void check_round(VecScalpEnv& env, const std::vector<Episode>& want, bool first_round) {
  const std::size_t n = env.size();
  if (first_round) env.reset();
  std::vector<std::uint64_t> ids(env.episode_ids(), env.episode_ids() + n);
  for (std::size_t j = 0; j < n; ++j) {
    ASSERT_LT(ids[j], want.size());
    ASSERT_TRUE(same_bytes(env.obs() + j * D, want[ids[j]].obs.data(), D)) << "first obs, slot " << j;
  }
  const std::size_t T = want[ids[0]].rew.size();
  std::vector<std::int64_t> a(n);
  for (std::size_t t = 0; t < T; ++t) {
    for (std::size_t j = 0; j < n; ++j) a[j] = pattern(t, ids[j]);
    env.step(a.data());
    const bool done = t + 1 == T;
    for (std::size_t j = 0; j < n; ++j) {
      const Episode& w = want[ids[j]];
      ASSERT_EQ(env.dones()[j], done ? 1 : 0);
      ASSERT_TRUE(same_bytes(env.rewards() + j, &w.rew[t], 1)) << "reward t " << t << " slot " << j;
      const double* o = done ? env.final_obs() : env.obs();
      ASSERT_TRUE(same_bytes(o + j * D, w.obs.data() + (t + 1) * D, D)) << "obs t " << t << " slot " << j;
    }
  }
  for (std::size_t j = 0; j < n; ++j) {
    EXPECT_EQ(env.final_episode_ids()[j], ids[j]);
    EXPECT_EQ(env.net_pnl_ticks()[j], want[ids[j]].pnl);
    EXPECT_EQ(env.orders()[j], want[ids[j]].orders);
    EXPECT_EQ(env.fees_ticks()[j], want[ids[j]].fees);
  }
}

RunParams short_default(double episode = 120) { return run_params("default", 150, 1.0, episode); }

}  // namespace

TEST(VecScalpEnv, TrainStreamMatchesTrainerStream) {
  const RunParams r = run_params("default", 150);  // of_train's ScalpParams{}
  std::vector<Episode> want;
  Rng stream(5);  // of_train / Trainer: run seed 5, one stream for reset and step
  for (std::uint64_t k = 0; k < 9; ++k) want.push_back(serial(r.p, stream, k));
  for (const std::size_t n : {std::size_t{1}, std::size_t{3}, std::size_t{8}}) {
    Ledger L;
    VecScalpEnv env(r, EpisodeSource::train(5), n, 4, false, L);
    const std::size_t rounds = n == 8 ? 1 : 9 / n;
    for (std::size_t round = 0; round < rounds; ++round) {
      if (round == 0) env.reset();
      for (std::size_t j = 0; j < n; ++j) ASSERT_EQ(env.episode_ids()[j], round * n + j);
      check_round(env, want, false);
    }
    EXPECT_EQ(env.episodes_started(), (rounds + 1) * n);  // auto-reset started the next round
    EXPECT_EQ(L.role(), seeds::Role::Trainer);
  }
}

TEST(VecScalpEnv, IndexedMatchesEvaluateAndBuiltinRules) {
  const RunParams r = short_default();
  std::vector<Episode> want(8);
  for (std::uint64_t i = 3; i < 8; ++i) {
    Rng rng(700'000'000 + i);  // evaluate() / of_train score(): episode i = Rng(seed + i)
    want[i] = serial(r.p, rng, i);
  }
  Ledger L;
  VecScalpEnv env(r, EpisodeSource::indexed("validation", 3, 5, {}, L), 99, 3, false, L);
  ASSERT_EQ(env.size(), 5u);  // n_envs = count
  check_round(env, want, true);
  EXPECT_THROW(env.reset(), std::runtime_error);  // played once
  const std::int64_t a[5] = {1, 1, 1, 1, 1};
  EXPECT_THROW(env.step(a), std::runtime_error);

  // run_builtin(delta30) = of_train's rule in a serial loop = the rule acting on VecScalpEnv's obs.
  const auto src = EpisodeSource::indexed("validation", 10, 6, {}, L);
  const auto res = run_builtin(r, src, BuiltinPolicy::delta30(), 4, L);
  VecScalpEnv v(r, src, 0, 2, false, L);
  v.reset();
  std::vector<std::int64_t> act(6);
  for (;;) {
    for (std::size_t j = 0; j < 6; ++j) {
      const double* o = v.obs() + j * D;
      act[j] = o[1] > 1.5 ? 2 : o[1] < -1.5 ? 0 : static_cast<std::int64_t>(o[kFeatures] + 1);
    }
    v.step(act.data());
    if (v.dones()[0]) break;
  }
  for (std::uint64_t i = 0; i < 6; ++i) {
    ScalpEnv e(r.p);
    Rng rng(700'000'010 + i);
    auto o = e.reset(rng);
    for (;;) {
      const auto st = e.step(o.f[1] > 1.5 ? 2 : o.f[1] < -1.5 ? 0 : static_cast<Action>(o.position + 1), rng);
      o = st.obs;
      if (st.done) break;
    }
    EXPECT_EQ(res.net_pnl_ticks[i], e.net_pnl_ticks());
    EXPECT_EQ(res.orders[i], e.orders());
    EXPECT_EQ(v.net_pnl_ticks()[i], e.net_pnl_ticks());
  }
}

TEST(VecScalpEnv, HoldNestsOnItsThreadAndRefusesOthers) {
  const RunParams r = short_default(5);
  Ledger L;
  VecScalpEnv env(r, EpisodeSource::train(3), 2, 1, false, L);
  std::vector<std::int64_t> a(2, 1);
  {
    const auto g = env.hold();  // a binding holds it across a call and the copy of its results
    env.reset();                // nests on the owning thread
    env.step(a.data());
    bool refused = false;
    std::thread other([&] {
      try {
        env.step(a.data());
      } catch (const std::runtime_error&) {
        refused = true;
      }
    });
    other.join();
    EXPECT_TRUE(refused);
  }
  std::thread other([&] { env.step(a.data()); });  // released: another thread may use it now
  other.join();
  EXPECT_EQ(env.rounds_finished(), 0u);
}

TEST(VecScalpEnv, ThreadCountInvariant) {
  const RunParams r = short_default(60);
  auto run = [&](unsigned threads, std::size_t n) {
    Ledger L;
    VecScalpEnv env(r, EpisodeSource::train(11), n, threads, false, L);
    std::vector<double> out;
    env.reset();
    std::vector<std::int64_t> a(n);
    for (std::uint64_t t = 0; t < 130; ++t) {  // two rounds and a bit (60 decisions each)
      for (std::size_t j = 0; j < n; ++j) a[j] = pattern(t, env.episode_ids()[j]);
      env.step(a.data());
      out.insert(out.end(), env.obs(), env.obs() + n * D);
      out.insert(out.end(), env.rewards(), env.rewards() + n);
    }
    return out;
  };
  const auto one = run(1, 8);
  const auto many = run(8, 8);
  ASSERT_EQ(one.size(), many.size());
  EXPECT_TRUE(same_bytes(one.data(), many.data(), one.size()));
  EXPECT_TRUE(same_bytes(run(3, 8).data(), one.data(), one.size()));
}

TEST(VecScalpEnv, DaysAndStressMatchOf8AndOf6) {
  const RunParams r = short_default();
  Ledger L;
  L.open_sealed(Block::Days, "unit test");
  L.open_sealed(Block::Stress, "unit test");
  // OF8: day d session k = Rng(930M + d * 100 + k), warm-up starting at day_session_start(k).
  std::vector<Episode> want(400);
  for (std::uint64_t d = 2; d < 4; ++d)
    for (unsigned k = 0; k < 3; ++k) {
      ScalpParams p = r.p;
      p.start_second = day_session_start(k, p.warmup_seconds);
      Rng rng(930'000'000ull + d * 100 + k);
      want[d * 100 + k] = serial(p, rng, d * 100 + k);
    }
  VecScalpEnv days(r, EpisodeSource::days(2, 2, 3, L), 0, 3, false, L);
  ASSERT_EQ(days.size(), 6u);
  check_round(days, want, true);
  // OF6: stress episode i = Rng(920M + i) under the scenario's change.
  std::vector<Episode> sw(4);
  for (std::uint64_t i = 1; i < 4; ++i) {
    ScalpParams p = r.p;
    p.latency_steps = 3;
    Rng rng(920'000'000ull + i);
    sw[i] = serial(p, rng, i);
  }
  VecScalpEnv stress(r, EpisodeSource::stress("latency 300 ms", 1, 3, {}, L), 0, 2, false, L);
  check_round(stress, sw, true);
  EXPECT_EQ(L.role(), seeds::Role::Scorer);
  EXPECT_THROW(VecScalpEnv(r, EpisodeSource::stress("no such stress", 0, 2, {}, L), 0, 1, false, L), std::invalid_argument);
  const RunParams v4 = run_params("discovery_v4", 1050, 1.0, 60);
  EXPECT_THROW(VecScalpEnv(v4, EpisodeSource::stress("nominal", 0, 2, {}, L), 0, 1, false, L), std::invalid_argument);
}

TEST(VecScalpEnv, PrivilegedColumnsSeparateAndExact) {
  const RunParams r = short_default(60);
  Ledger L;
  EXPECT_THROW(VecScalpEnv(r, EpisodeSource::indexed("validation", 0, 2, {}, L), 0, 1, true, L), std::runtime_error);
  VecScalpEnv plain(r, EpisodeSource::indexed("viz", 0, 2, {}, L), 0, 1, false, L);
  EXPECT_THROW((void)plain.priv(), std::runtime_error);
  VecScalpEnv env(r, EpisodeSource::indexed("viz", 0, 2, {}, L), 0, 2, true, L);
  plain.reset();
  env.reset();
  // A serial copy of slot 0's episode, for the hidden state at every decision.
  ScalpEnv ref(r.p);
  Rng rng(960'000'000);
  Rng copy(960'000'000);
  const double start = ScalpEnv::draw_market(r.p, copy).start;
  auto o = ref.reset(rng);
  std::int64_t a[2];
  for (std::uint64_t t = 0;; ++t) {
    const double* pv = env.priv();
    EXPECT_EQ(pv[0], ref.market().fair_value());
    EXPECT_EQ(pv[1], ref.market().mid());
    EXPECT_EQ(pv[2], ref.market().fair_value() - ref.market().mid());
    EXPECT_EQ(pv[3], start + static_cast<double>(ref.market().t_ns()) * 1e-9);
    EXPECT_EQ(pv[5], ref.orders());
    EXPECT_TRUE(same_bytes(env.obs(), plain.obs(), 2 * D));  // obs never depend on the privileged flag
    a[0] = a[1] = pattern(t, 0);
    const int pos0 = o.position;
    const double c0 = ref.net_pnl_ticks() + ref.fees_ticks() - pos0 * ref.market().mid();
    const auto orders0 = ref.orders();
    const auto st = ref.step(static_cast<Action>(a[0]), rng);
    o = st.obs;
    env.step(a);
    plain.step(a);
    const double* q = st.done ? env.final_priv() : env.priv();
    const double cash = ref.net_pnl_ticks() + ref.fees_ticks() - o.position * ref.market().mid();
    EXPECT_EQ(q[6], cash);
    if (ref.orders() - orders0 == 1 && o.position != pos0) {
      EXPECT_EQ(q[4], -(cash - c0) / (o.position - pos0));
      EXPECT_TRUE(std::isfinite(q[4]));
    } else {
      EXPECT_TRUE(std::isnan(q[4]));
    }
    if (st.done) break;
  }
  EXPECT_EQ(L.role(), seeds::Role::None);  // viz is neutral
}

TEST(VecScalpEnv, NeuralQBuiltinIsGreedyFirstMax) {
  const RunParams r = short_default(60);
  Rng init(3);
  auto net = std::make_shared<rag::agent::Mlp>(std::vector<std::uint32_t>{16, 8, 3}, init);
  Ledger L;
  const auto src = EpisodeSource::indexed("pairing", 0, 3, {}, L);
  const auto res = run_builtin(r, src, BuiltinPolicy::neuralq(net), 3, L);
  std::vector<double> scratch(net->scratch_size());
  for (std::uint64_t i = 0; i < 3; ++i) {
    ScalpEnv e(r.p);
    Rng rng(940'000'000 + i);
    auto o = e.reset(rng);
    for (;;) {
      double q[3];
      const auto x = e.features(o);
      net->forward(x.data(), q, scratch.data());
      const Action a = q[1] > q[0] ? (q[2] > q[1] ? 2 : 1) : (q[2] > q[0] ? 2 : 0);
      const auto st = e.step(a, rng);
      o = st.obs;
      if (st.done) break;
    }
    EXPECT_EQ(res.net_pnl_ticks[i], e.net_pnl_ticks());
  }
  EXPECT_THROW(BuiltinPolicy::neuralq(std::make_shared<rag::agent::Mlp>(std::vector<std::uint32_t>{15, 3}, init)), std::invalid_argument);
}

TEST(VecReplayEnv, MatchesReplayScalpEnvAndIsSealed) {
  std::string csv;
  {
    EsSimulator sim(SimParams{}, 9);
    write_sim_ticks(sim, 4000, [&](const char* s, std::size_t n) { csv.append(s, n); });
  }
  std::istringstream in(csv);
  auto tape = std::make_shared<const TickTape>(TickTape::from_stream(in, {.time_unit = "s"}));
  const ReplayParams p{.window_seconds = 60, .warmup_seconds = 30};
  Ledger L;
  EXPECT_THROW(VecReplayEnv(tape, p, {0}, 1, L), std::runtime_error);
  L.open_sealed(Block::Real, "unit test");
  EXPECT_THROW(VecReplayEnv(tape, p, {0, 999}, 1, L), std::out_of_range);
  const std::vector<std::int64_t> wins = {0, 3, 1};
  VecReplayEnv env(tape, p, wins, 3, L);
  env.reset();
  std::vector<ReplayScalpEnv> ref;
  std::vector<std::vector<double>> o0;
  for (auto k : wins) {
    ref.emplace_back(tape, p);
    const auto x = ref.back().features(ref.back().seek(static_cast<std::size_t>(k)));
    o0.emplace_back(x.begin(), x.end());
  }
  for (std::size_t j = 0; j < wins.size(); ++j) ASSERT_TRUE(same_bytes(env.obs() + j * D, o0[j].data(), D));
  Rng unused(0);
  std::int64_t a[3];
  for (std::uint64_t t = 0;; ++t) {
    for (std::size_t j = 0; j < 3; ++j) a[j] = pattern(t, j);
    env.step(a);
    bool done = false;
    for (std::size_t j = 0; j < 3; ++j) {
      const auto st = ref[j].step(static_cast<Action>(a[j]), unused);
      const auto x = ref[j].features(st.obs);
      ASSERT_TRUE(same_bytes(env.rewards() + j, &st.reward, 1));
      ASSERT_TRUE(same_bytes((st.done ? env.final_obs() : env.obs()) + j * D, x.data(), D));
      ASSERT_EQ(env.dones()[j], st.done ? 1 : 0);
      done = st.done;
    }
    if (done) {
      ASSERT_EQ(t + 1, 60u);
      break;
    }
  }
  for (std::size_t j = 0; j < 3; ++j) EXPECT_EQ(env.net_pnl_ticks()[j], ref[j].net_pnl_ticks());
}

TEST(SeedLedger, Refusals) {
  const RunParams r = short_default(30);
  {
    Ledger L;  // ranges, names, claims
    EXPECT_THROW(EpisodeSource::train(0), std::invalid_argument);
    EXPECT_THROW(EpisodeSource::train(700'000'000), std::invalid_argument);
    EXPECT_NO_THROW(EpisodeSource::train(699'999'999));
    EXPECT_THROW(EpisodeSource::indexed("validation", 199'999'999, 2, {}, L), std::invalid_argument);
    EXPECT_THROW(EpisodeSource::indexed("validation", 0, 0, {}, L), std::invalid_argument);
    EXPECT_THROW(EpisodeSource::indexed("train", 0, 1, {}, L), std::invalid_argument);
    EXPECT_THROW(EpisodeSource::indexed("stress", 0, 1, {}, L), std::invalid_argument);
    EXPECT_THROW(EpisodeSource::indexed("real", 0, 1, {}, L), std::invalid_argument);
    EXPECT_THROW(EpisodeSource::indexed("tick_test", 0, 1, {}, L), std::invalid_argument);
    EXPECT_THROW(EpisodeSource::indexed("test", 0, 1, {}, L), std::runtime_error);  // sealed
    EXPECT_THROW(EpisodeSource::indexed("test", 9'999'999, 2, {}, L), std::invalid_argument);  // 910M+ (tick test) not exposed
    EXPECT_THROW(EpisodeSource::indexed("confirm", 0, 1, {}, L), std::invalid_argument);  // no claim id
    EXPECT_THROW(EpisodeSource::indexed("confirm", 0, 1, "claim-1", L), std::runtime_error);  // claim, but sealed
    EXPECT_THROW(EpisodeSource::indexed("validation", 0, 1, "claim-1", L), std::invalid_argument);
    EXPECT_THROW(EpisodeSource::days(0, 1, 39, L), std::runtime_error);
    EXPECT_THROW(EpisodeSource::stress("nominal", 0, 1, {}, L), std::runtime_error);
    EXPECT_THROW(L.open_sealed(Block::Validation, "x"), std::invalid_argument);  // not sealed
    EXPECT_THROW(L.open_sealed(Block::Test, ""), std::invalid_argument);         // no reason
    EXPECT_EQ(L.role(), seeds::Role::None);
  }
  {
    Ledger L;  // a trainer never opens a sealed block
    VecScalpEnv env(r, EpisodeSource::train(1), 2, 1, true, L);  // privileged is fine on train
    EXPECT_EQ(L.role(), seeds::Role::Trainer);
    EXPECT_THROW(L.open_sealed(Block::Test, "score"), std::runtime_error);
    EXPECT_FALSE(L.is_open(Block::Test));
  }
  {
    Ledger L;  // selection (validation, pairing) also makes a trainer
    (void)run_builtin(r, EpisodeSource::indexed("pairing", 0, 1, {}, L), BuiltinPolicy::flat(), 1, L);
    EXPECT_THROW(L.open_sealed(Block::Test, "score"), std::runtime_error);
  }
  {
    Ledger L;  // a scorer never trains or selects again
    L.open_sealed(Block::Test, "score");
    EXPECT_EQ(L.role(), seeds::Role::Scorer);
    const auto test = EpisodeSource::indexed("test", 0, 2, {}, L);
    EXPECT_THROW(VecScalpEnv(r, test, 0, 1, true, L), std::runtime_error);  // privileged refused on test
    EXPECT_NO_THROW(VecScalpEnv(r, test, 0, 1, false, L));
    EXPECT_NO_THROW((void)run_builtin(r, test, BuiltinPolicy::oracle(0.5), 2, L));
    EXPECT_NO_THROW(VecScalpEnv(r, EpisodeSource::indexed("viz", 0, 1, {}, L), 0, 1, false, L));
    EXPECT_THROW(VecScalpEnv(r, EpisodeSource::train(1), 1, 1, false, L), std::runtime_error);
    EXPECT_THROW(VecScalpEnv(r, EpisodeSource::indexed("validation", 0, 1, {}, L), 0, 1, false, L), std::runtime_error);
    EXPECT_THROW((void)run_builtin(r, EpisodeSource::indexed("pairing", 0, 1, {}, L), BuiltinPolicy::flat(), 1, L), std::runtime_error);
    EXPECT_THROW(EpisodeSource::indexed("confirm", 0, 1, {}, L), std::invalid_argument);  // a claim id is needed even when open
    L.open_sealed(Block::Confirm, "orchestrator");
    EXPECT_NO_THROW(EpisodeSource::indexed("confirm", 0, 1, "claim-1", L));
    ASSERT_EQ(L.openings().size(), 2u);
    EXPECT_EQ(L.openings()[1].first, "confirm");
    // forbid_sealed closes what was open: a source made earlier no longer plays.
    L.forbid_sealed();
    EXPECT_THROW(VecScalpEnv(r, test, 0, 1, false, L), std::runtime_error);
    EXPECT_THROW(L.open_sealed(Block::Test, "again"), std::runtime_error);
  }
  {
    Ledger L;
    L.forbid_sealed();
    EXPECT_THROW(L.open_sealed(Block::Real, "replay"), std::runtime_error);
  }
  // Parameters: market, warm-up, overrides, ScalpEnv's guards.
  EXPECT_THROW(run_params("nope", 150), std::invalid_argument);
  EXPECT_THROW(run_params("default", 30'000), std::invalid_argument);
  EXPECT_THROW(run_params("default", -1), std::invalid_argument);
  EXPECT_THROW(run_params("default", 150, 1, 600, -1, 1, 1, 1.5, {{"dt", 0.2}}), std::invalid_argument);
  EXPECT_THROW(run_params("discovery_v4", 1050, 1, 600, -1, 1, 2), std::invalid_argument);  // contracts > 1 on v4
  EXPECT_EQ(run_params("default", 150, 1, 600, -1, 1, 1, 1.5, {{"value_vol", 0.5}}).p.sim.value_vol, 0.5);
  EXPECT_TRUE(run_params("discovery_v4", 1050) == run_params("discovery_v4", 1050));
  EXPECT_FALSE(run_params("discovery_v4", 1050) == run_params("discovery_v4", 150));
  EXPECT_EQ(steps_per_episode(run_params("default", 150).p), 600u);
}
