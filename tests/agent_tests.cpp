// agent_tests.cpp — the self-learning agent: every component in isolation,
// then whole-system properties (it learns, it learns the right thing, it
// notices when the world changes and recovers, it is deterministic).
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <map>
#include <set>
#include <vector>

#include "rag/agent/execution_dp.hpp"
#include "rag/agent/trainer.hpp"
#include "test_util.hpp"

using namespace rag::agent;
using rag::testutil::begin_alloc_count;
using rag::testutil::end_alloc_count;

namespace {

constexpr Action kTwap = ExecutionEnv::kTwapAction;
auto twap = [](const ExecutionObs&) { return kTwap; };

// Fraction of the optimal policy's gain over TWAP that `policy` captures,
// measured on the same held-out scenarios.
template <class Policy>
double captured(const ExecutionEnv& env, const ExecutionDP& dp, Policy&& policy, std::uint32_t n = 1500) {
  const double mine = compare(env, policy, twap, n, 77'000'000).mean;
  const double best = compare(env, [&](const ExecutionObs& o) { return dp.act(o); }, twap, n, 77'000'000).mean;
  return mine / best;
}

const ExecutionDP& default_dp() {
  static const ExecutionDP dp{ExecutionParams{}};
  return dp;
}

AgentConfig<3> execution_config(std::uint64_t seed = 1) {
  AgentConfig<3> c;
  c.eps_decay_episodes = 10000;
  c.seed = seed;
  return c;
}

}  // namespace

// ================================== Rng =====================================
TEST(AgentRng, DeterministicAndSeedsAreIndependent) {
  Rng a(1), b(1), c(2);
  for (int i = 0; i < 100; ++i) {
    const auto x = a.next();
    EXPECT_EQ(x, b.next());
    EXPECT_NE(x, c.next());
  }
  Rng r(3);
  r.normal();  // leaves a cached spare...
  r.reseed(3);
  Rng fresh(3);
  EXPECT_EQ(r.normal(), fresh.normal());  // ...which reseed() discards
}

TEST(AgentRng, DistributionsHaveTheRightMoments) {
  Rng r(42);
  constexpr int n = 200'000;
  double su = 0, sn = 0, sn2 = 0, sn3 = 0;
  std::array<int, 10> bucket{};
  for (int i = 0; i < n; ++i) {
    const double u = r.uniform();
    ASSERT_GE(u, 0.0);
    ASSERT_LT(u, 1.0);
    su += u;
    const auto k = r.below(10);
    ASSERT_LT(k, 10u);
    ++bucket[k];
    const double z = r.normal();
    sn += z; sn2 += z * z; sn3 += z * z * z;
  }
  EXPECT_NEAR(su / n, 0.5, 0.003);
  for (int c : bucket) EXPECT_NEAR(c, n / 10, n / 10 * 0.03);
  EXPECT_NEAR(sn / n, 0.0, 0.01);
  EXPECT_NEAR(sn2 / n, 1.0, 0.015);
  EXPECT_NEAR(sn3 / n, 0.0, 0.03);
  EXPECT_EQ(Rng(5).below(1), 0u);
}

// ================================ ChainEnv ==================================
TEST(AgentChainEnv, OptimalReturnTruncationAndSlip) {
  ChainEnv env({.length = 6, .max_steps = 20});
  Rng rng(1);
  env.reset(rng);
  double total = 0;
  int steps = 0;
  for (;;) {
    const auto st = env.step(1, rng);
    total += st.reward;
    ++steps;
    if (st.done) break;
  }
  EXPECT_EQ(total, -5.0);  // -(length - 1)
  EXPECT_EQ(steps, 5);
  env.reset(rng);
  for (int i = 0; i < 19; ++i) EXPECT_FALSE(env.step(0, rng).done);  // walking into the wall...
  EXPECT_TRUE(env.step(0, rng).done);                               // ...until max_steps
  ChainEnv slippy({.length = 1000, .max_steps = 100000, .slip = 0.25});
  slippy.reset(rng);
  int backwards = 0;
  std::uint32_t pos = 0;
  for (int i = 0; i < 4000; ++i) {
    const auto st = slippy.step(1, rng);
    backwards += st.obs.pos < pos;
    pos = st.obs.pos;
  }
  EXPECT_NEAR(backwards / 4000.0, 0.25, 0.03);
}

// ============================== ExecutionEnv ================================
TEST(AgentExecutionEnv, RewardsAddUpToImplementationShortfall) {
  // Total reward must equal cash received minus the arrival value of the
  // inventory, for any policy: the reward really measures execution quality.
  ExecutionParams p;
  Rng pol(9);
  for (int ep = 0; ep < 300; ++ep) {
    ExecutionEnv env(p);
    Rng rng(ep);
    auto o = env.reset(rng);
    double total = 0;
    std::uint32_t prev_inv = o.inventory;
    for (;;) {
      const auto st = env.step(pol.below(5), rng);
      total += st.reward;
      ASSERT_LE(st.obs.inventory, prev_inv);  // inventory only ever goes down
      prev_inv = st.obs.inventory;
      o = st.obs;
      if (st.done) break;
    }
    EXPECT_EQ(env.inventory(), 0u);  // always fully liquidated
    EXPECT_LE(o.t, p.horizon);
    EXPECT_NEAR(total, env.cash() - p.lots * p.start_price, 1e-9);
  }
}

TEST(AgentExecutionEnv, TwapSellsEvenlyAndActionsAreCapped) {
  ExecutionParams p;
  ExecutionEnv env(p);
  Rng rng(1);
  auto o = env.reset(rng);
  for (std::uint32_t t = 0; t < p.horizon; ++t) {
    EXPECT_EQ(env.lots_for(kTwap, o.inventory), 5u);
    const auto st = env.step(kTwap, rng);
    EXPECT_EQ(st.obs.inventory, p.lots - 5 * (t + 1));
    EXPECT_EQ(st.done, t + 1 == p.horizon);
    o = st.obs;
  }
  EXPECT_EQ(env.lots_for(4, 7), 7u);  // never sells more than is held
  EXPECT_EQ(env.lots_for(0, 50), 0u);
  EXPECT_EQ(env.lots_for(3, 50), 10u);
}

TEST(AgentExecutionEnv, SameSeedSameScenarioAndSignalPredictsMoves) {
  ExecutionParams p;
  auto path = [&](std::uint64_t seed, double alpha) {
    ExecutionParams q = p;
    q.alpha = alpha;
    ExecutionEnv env(q);
    Rng rng(seed);
    std::vector<double> prices;
    auto o = env.reset(rng);
    double cov = 0;
    for (;;) {
      const double before = env.price(), sig = o.signal;
      const auto st = env.step(1, rng);
      if (st.done) break;
      cov += sig * (env.price() - before);
      prices.push_back(env.price());
      o = st.obs;
    }
    return std::pair{prices, cov};
  };
  EXPECT_EQ(path(5, 1.0).first, path(5, 1.0).first);
  EXPECT_NE(path(5, 1.0).first, path(6, 1.0).first);
  double up = 0, down = 0;
  for (int s = 0; s < 400; ++s) { up += path(s, 1.0).second; down += path(s, -1.0).second; }
  EXPECT_GT(up, 0);    // the signal predicts the move...
  EXPECT_LT(down, 0);  // ...and set_alpha(-) reverses what it predicts
}

// =============================== DP optimum =================================
TEST(AgentExecutionDP, PredictedValueMatchesSimulation) {
  const auto& dp = default_dp();
  ExecutionEnv env{ExecutionParams{}};
  // Paired against TWAP (whose expected value is exactly -impact * sum q^2 = -5),
  // so the large scenario noise both share cancels.
  const auto gain = compare(env, [&](const ExecutionObs& o) { return dp.act(o); }, twap, 4000, 123);
  const double predicted = dp.expected_start_value() - (-5.0);
  EXPECT_NEAR(gain.mean, predicted, 4 * gain.std_error + 0.05) << "stderr " << gain.std_error;
  EXPECT_GT(gain.mean, 10.0);
}

TEST(AgentExecutionDP, BeatsEveryHeuristicAndHandlesEdges) {
  const auto& dp = default_dp();
  ExecutionEnv env{ExecutionParams{}};
  auto opt = [&](const ExecutionObs& o) { return dp.act(o); };
  Rng r(1);
  EXPECT_GT(compare(env, opt, [](const ExecutionObs& o) { return o.signal < 0 ? Action{3} : Action{1}; }, 1500, 9).mean, 5.0);
  EXPECT_GT(compare(env, opt, [&](const ExecutionObs&) { return Action{r.below(5)}; }, 1500, 9).mean, 15.0);
  EXPECT_GT(compare(env, opt, [](const ExecutionObs&) { return Action{4}; }, 1500, 9).mean, 15.0);
  EXPECT_EQ(dp.value(3, 0, 0.1), 0.0);   // nothing left to sell
  EXPECT_EQ(dp.value(20, 40, 0.1), 0.0); // past the horizon
  // Last step: anything unsold is dumped at 5x the impact, so the best action
  // minimizes impact * q^2 + terminal_impact * (inventory - q)^2 over the sizes
  // available (here 0, 3, 5, 10, 15). Brute-force that and compare.
  ExecutionEnv probe{ExecutionParams{}};
  for (std::uint32_t inv : {1u, 4u, 7u, 12u, 14u, 16u, 40u, 100u}) {
    double best = 1e300;
    for (Action a = 0; a < 5; ++a) {
      const double q = probe.lots_for(a, inv), rem = inv - q;
      best = std::min(best, 0.01 * q * q + 0.05 * rem * rem);
    }
    EXPECT_NEAR(-dp.q_value(19, inv, 0.0, dp.act({19, inv, 0.0})), best, 1e-12) << "inventory " << inv;
  }
}

TEST(AgentExecutionDP, MirrorSymmetry) {
  // Flipping alpha and the signal together leaves the problem unchanged.
  ExecutionParams p, m;
  m.alpha = -p.alpha;
  const ExecutionDP a(p, 61), b(m, 61);
  for (std::uint32_t t : {0u, 7u, 18u})
    for (std::uint32_t inv : {5u, 40u, 100u})
      for (double g : {-0.2, -0.05, 0.0, 0.13}) EXPECT_NEAR(a.value(t, inv, g), b.value(t, inv, -g), 1e-9);
}

// =============================== Encoders ===================================
TEST(AgentTileCoder, OneTilePerTilingAndGeneralizationFallsWithDistance) {
  TileCoder<2> tc({0.0, -1.0}, {1.0, 1.0}, {4, 5}, 8);
  EXPECT_EQ(tc.size(), 8u * 5 * 6);
  auto tiles = [&](double x, double y) {
    std::array<std::uint32_t, kMaxTilings> idx{};
    tc.encode({x, y}, idx.data());
    std::set<std::uint32_t> s(idx.begin(), idx.begin() + 8);
    for (std::uint32_t k = 0; k < 8; ++k) {
      EXPECT_GE(idx[k], k * 30u);  // tiling k uses its own block of indices
      EXPECT_LT(idx[k], (k + 1) * 30u);
    }
    return s;
  };
  auto shared = [&](auto a, auto b) {
    std::vector<std::uint32_t> out;
    std::set_intersection(a.begin(), a.end(), b.begin(), b.end(), std::back_inserter(out));
    return out.size();
  };
  const auto p = tiles(0.5, 0.0);
  EXPECT_EQ(shared(p, tiles(0.5, 0.0)), 8u);
  const auto near = shared(p, tiles(0.52, 0.02)), mid = shared(p, tiles(0.6, 0.1)), far = shared(p, tiles(0.95, 0.9));
  EXPECT_GE(near, mid);
  EXPECT_GT(near, 4u);
  EXPECT_EQ(far, 0u);
  EXPECT_EQ(tiles(-5.0, 7.0), tiles(0.0, 1.0));  // clamped onto the boundary
  EXPECT_THROW(TileCoder<1>({0.0}, {0.0}, {4}, 8), std::invalid_argument);
  EXPECT_THROW(TileCoder<1>({0.0}, {1.0}, {4}, 0), std::invalid_argument);
}

TEST(AgentRandomFourier, UnitVectorsWhoseDotApproximatesAGaussianKernel) {
  RandomFourierEncoder<2, 256> enc({0.3, 0.3}, 7);
  alignas(64) std::array<float, 256> a, b;
  Rng r(3);
  double err = 0;
  for (int i = 0; i < 400; ++i) {
    const std::array<double, 2> x{r.uniform(), r.uniform()}, y{r.uniform(), r.uniform()};
    enc.encode(x, a.data());
    enc.encode(y, b.data());
    EXPECT_NEAR(rag::simd::dot_scalar(a.data(), a.data(), 256), 1.0, 1e-5);
    const double d2 = ((x[0] - y[0]) * (x[0] - y[0]) + (x[1] - y[1]) * (x[1] - y[1])) / (0.3 * 0.3);
    err += std::abs(rag::simd::dot_scalar(a.data(), b.data(), 256) - std::exp(-0.5 * d2));
  }
  EXPECT_LT(err / 400, 0.08);
}

// ================================ Learners ==================================
TEST(AgentTileQ, UpdateMovesExactlyAlphaOfTheErrorAndOnlyThatAction) {
  TileQ q(100, 4, 3, 0.5);
  const std::uint32_t s[4] = {1, 30, 55, 99};
  EXPECT_EQ(q.q(s, 1), 0.0);
  EXPECT_DOUBLE_EQ(q.update(s, 1, 10.0), 10.0);
  EXPECT_DOUBLE_EQ(q.q(s, 1), 5.0);  // halfway, with alpha 0.5
  EXPECT_DOUBLE_EQ(q.q(s, 0), 0.0);
  q.update(s, 1, 10.0);
  EXPECT_DOUBLE_EQ(q.q(s, 1), 7.5);
  double out[3];
  q.q_values(s, out);
  EXPECT_DOUBLE_EQ(out[1], 7.5);
  EXPECT_DOUBLE_EQ(out[2], 0.0);
  q.set_alpha(0.1);
  EXPECT_DOUBLE_EQ(q.alpha(), 0.1);
}

TEST(AgentReplay, RingOverwritesOldestAndSamplesOnlyFilledSlots) {
  ReplayBuffer rb(4);
  Rng rng(1);
  for (int i = 0; i < 6; ++i) {
    Transition t{};
    t.reward = static_cast<float>(i);
    rb.push(t);
  }
  EXPECT_EQ(rb.size(), 4u);
  EXPECT_EQ(rb.newest().reward, 5.0f);
  std::set<float> seen;
  for (int i = 0; i < 200; ++i) seen.insert(rb.sample(rng).reward);
  EXPECT_EQ(seen, (std::set<float>{2, 3, 4, 5}));  // 0 and 1 were overwritten
  rb.clear();
  EXPECT_EQ(rb.size(), 0u);
  ReplayBuffer none(0);
  none.push(Transition{});  // capacity 0: a no-op, not a crash
  EXPECT_EQ(none.size(), 0u);
}

TEST(AgentEpisodicMemory, SameSituationUpdatesUnknownActionsStayUnknown) {
  EpisodicMemory<64, 3> m({.capacity = 16, .neighbors = 4, .same_situation = 0.99f, .learning_rate = 0.5});
  alignas(64) std::array<float, 64> k{};
  k[0] = 1.f;
  m.write(k.data(), 1, 10.0);
  m.write(k.data(), 1, 20.0);  // same situation: EMA 10 -> 15
  EXPECT_EQ(m.size(), 1u);
  double q[3];
  bool known[3];
  m.estimate(k.data(), q, known);
  EXPECT_TRUE(known[1]);
  EXPECT_DOUBLE_EQ(q[1], 15.0);
  EXPECT_FALSE(known[0]);
  EXPECT_FALSE(known[2]);
  EpisodicMemory<64, 3> empty;
  empty.estimate(k.data(), q, known);
  EXPECT_FALSE(known[0] || known[1] || known[2]);
}

TEST(AgentEpisodicMemory, KnnEstimateMatchesBruteForceAndLruEvicts) {
  constexpr std::size_t D = 64;
  RandomFourierEncoder<1, D> enc({0.05}, 11);
  EpisodicMemory<D, 2> m({.capacity = 64, .neighbors = 5, .same_situation = 0.9999f, .learning_rate = 1.0});
  std::vector<std::array<float, D>> keys(40);
  std::vector<double> vals(40);
  for (int i = 0; i < 40; ++i) {
    enc.encode({i / 40.0}, keys[i].data());
    vals[i] = i * 1.5;
    m.write(keys[i].data(), 0, vals[i]);
  }
  ASSERT_EQ(m.size(), 40u);
  alignas(64) std::array<float, D> q;
  enc.encode({0.3}, q.data());
  // Brute force: similarity-weighted mean of the 5 most similar keys.
  std::vector<std::pair<float, int>> sims;
  for (int i = 0; i < 40; ++i) sims.push_back({rag::simd::dot_scalar(q.data(), keys[i].data(), D), i});
  std::sort(sims.begin(), sims.end(), [](auto& a, auto& b) { return a.first > b.first || (a.first == b.first && a.second < b.second); });
  double num = 0, den = 0;
  for (int j = 0; j < 5; ++j) {
    const double w = std::max(1e-3, static_cast<double>(sims[j].first));
    num += w * vals[sims[j].second];
    den += w;
  }
  double est[2];
  bool known[2];
  m.estimate(q.data(), est, known);
  EXPECT_NEAR(est[0], num / den, 1e-4);
  EXPECT_FALSE(known[1]);

  // LRU: fill a capacity-3 memory, refresh the first slot, add a fourth
  // situation: the second (least recently used) is the one evicted.
  EpisodicMemory<D, 2> lru({.capacity = 3, .neighbors = 1, .same_situation = 0.9999f, .learning_rate = 1.0});
  for (int i : {0, 10, 20}) lru.write(keys[i].data(), 0, vals[i]);
  lru.write(keys[0].data(), 0, vals[0]);  // touch slot 0
  lru.write(keys[30].data(), 0, vals[30]);
  EXPECT_EQ(lru.size(), 3u);
  auto nearest_value = [&](int i) {
    lru.estimate(keys[i].data(), est, known);
    return est[0];
  };
  EXPECT_EQ(nearest_value(0), vals[0]);
  EXPECT_EQ(nearest_value(20), vals[20]);
  EXPECT_EQ(nearest_value(30), vals[30]);
  EXPECT_NE(nearest_value(10), vals[10]);  // evicted: its nearest memory is now another one

  EpisodicMemory<D, 2> copy = m;  // deep copy: the copy keeps working after m changes
  m.write(q.data(), 0, -1000.0);
  copy.estimate(q.data(), est, known);
  EXPECT_NEAR(est[0], num / den, 1e-4);
}

// ================================== Agent ===================================
TEST(Agent, LearnsTheCorridorExactlyWithBothLearners) {
  for (LearnerKind kind : {LearnerKind::TileQ, LearnerKind::Episodic}) {
    ChainEnv env({.length = 8, .max_steps = 40});
    AgentConfig<1> cfg;
    cfg.learner = kind;
    cfg.eps_decay_episodes = 300;
    cfg.bins = {14};
    Agent<ChainEnv> agent(env, cfg);
    Trainer<ChainEnv> tr(agent, env, {.eval_every = 100, .eval_episodes = 1, .detect_drift = false});
    tr.run(600);
    const auto s = evaluate(env, [&](const auto& o) { return agent.act(o, false); }, 1, 1);
    EXPECT_EQ(s.mean, -7.0) << (kind == LearnerKind::TileQ ? "TileQ" : "Episodic");
  }
}

TEST(Agent, LearnsToGoRightDespiteRandomSlips) {
  ChainEnv env({.length = 8, .max_steps = 60, .slip = 0.2});
  AgentConfig<1> cfg;
  cfg.eps_decay_episodes = 500;
  cfg.bins = {14};
  Agent<ChainEnv> agent(env, cfg);
  Trainer<ChainEnv> tr(agent, env, {.eval_every = 0, .detect_drift = false});
  tr.run(1000);
  for (std::uint32_t pos = 0; pos + 1 < 8; ++pos) EXPECT_EQ(agent.act({pos, 0}, false), 1u) << "at " << pos;
}

TEST(Agent, ExplorationAndStepSizeSchedules) {
  ChainEnv env;
  AgentConfig<1> cfg;
  cfg.eps_start = 1.0;
  cfg.eps_end = 0.1;
  cfg.eps_decay_episodes = 100;
  cfg.alpha = 0.2;
  cfg.alpha_min = 0.01;
  cfg.alpha_decay_episodes = 50;
  Agent<ChainEnv> a(env, cfg);
  EXPECT_DOUBLE_EQ(a.epsilon(), 1.0);
  for (int i = 0; i < 50; ++i) a.end_episode();
  EXPECT_NEAR(a.epsilon(), 0.55, 1e-12);
  EXPECT_NEAR(a.step_size(), 0.1, 1e-12);  // 0.2 / (1 + 50/50)
  for (int i = 0; i < 1000; ++i) a.end_episode();
  EXPECT_DOUBLE_EQ(a.epsilon(), 0.1);
  EXPECT_DOUBLE_EQ(a.step_size(), 0.01);  // 0.2 / (1 + 1050/50) = 0.009 is below the floor
  a.reexplore(0.6, 10);  // a temporary boost that decays back
  EXPECT_DOUBLE_EQ(a.epsilon(), 0.6);
  EXPECT_DOUBLE_EQ(a.step_size(), 0.2);  // plasticity restored
  for (int i = 0; i < 5; ++i) a.end_episode();
  EXPECT_NEAR(a.epsilon(), 0.3, 1e-12);
  for (int i = 0; i < 5; ++i) a.end_episode();
  EXPECT_DOUBLE_EQ(a.epsilon(), 0.1);
  EXPECT_THROW(Agent<ChainEnv>(env, AgentConfig<1>{.eps_start = 0.1, .eps_end = 0.5}), std::invalid_argument);
  EXPECT_THROW(Agent<ChainEnv>(env, AgentConfig<1>{.tilings = 0}), std::invalid_argument);
}

TEST(Agent, ActingNeverAllocatesAndGreedyActingIsPure) {
  ExecutionEnv env{ExecutionParams{}};
  for (LearnerKind kind : {LearnerKind::TileQ, LearnerKind::Episodic}) {
    AgentConfig<3> cfg = execution_config();
    cfg.learner = kind;
    cfg.memory.capacity = 512;
    Agent<ExecutionEnv> agent(env, cfg);
    Trainer<ExecutionEnv> tr(agent, env, {.eval_every = 0, .detect_drift = false});
    tr.run(50);
    const ExecutionObs o{3, 80, -0.05};
    begin_alloc_count();
    Action sink = 0;
    for (int i = 0; i < 1000; ++i) sink ^= agent.act(o, i % 2 == 0);
    const std::size_t allocs = end_alloc_count();
    EXPECT_EQ(allocs, 0u);
    (void)sink;
    // Greedy decisions are deterministic and leave the agent's RNG alone, so
    // evaluating an agent never changes how it goes on to learn.
    EXPECT_EQ(agent.act(o, false), agent.act(o, false));
  }
}

TEST(Agent, EvaluationDoesNotPerturbLearningAndRunsAreReproducible) {
  ExecutionEnv env{ExecutionParams{}};
  auto train = [&](bool with_evals) {
    ExecutionEnv e = env;
    Agent<ExecutionEnv> agent(e, execution_config(7));
    Trainer<ExecutionEnv> tr(agent, e, {.eval_every = with_evals ? 200u : 0u, .eval_episodes = 20, .keep_champion = false,
                                        .detect_drift = false});
    tr.run(1000);
    return agent.q_learner().weights();
  };
  const auto a = train(false), b = train(false), c = train(true);
  EXPECT_EQ(a, b);  // same seeds, same result, bit for bit
  EXPECT_EQ(a, c);  // held-out evaluations in between change nothing
}

TEST(Agent, TrainingInChunksEqualsTrainingInOneGoAndActAsMatchesGreedy) {
  ExecutionEnv env{ExecutionParams{}};
  auto train = [&](std::vector<std::uint32_t> chunks) {
    ExecutionEnv e = env;
    auto agent = std::make_unique<Agent<ExecutionEnv>>(e, execution_config(3));
    Trainer<ExecutionEnv> tr(*agent, e, {.eval_every = 250, .eval_episodes = 30});
    for (auto c : chunks) tr.run(c);
    return agent->q_learner().weights();
  };
  // Ending a run() must not touch the learner (it once restored the
  // champion into it, silently undoing learning between calls).
  EXPECT_EQ(train({1500}), train({500, 250, 750}));

  Agent<ExecutionEnv> agent(env, execution_config());
  Trainer<ExecutionEnv> tr(agent, env, {.eval_every = 0, .detect_drift = false});
  tr.run(300);
  const auto snap = agent.snapshot();
  Rng r(2);
  for (int i = 0; i < 300; ++i) {
    const ExecutionObs o{r.below(20), r.below(101), 0.3 * (r.uniform() - 0.5)};
    ASSERT_EQ(agent.act_as(snap, o), agent.act(o, false));
  }
}

TEST(Agent, SnapshotRestoresTheExactPolicy) {
  ExecutionEnv env{ExecutionParams{}};
  Agent<ExecutionEnv> agent(env, execution_config());
  Trainer<ExecutionEnv> tr(agent, env, {.eval_every = 0, .detect_drift = false});
  tr.run(500);
  const auto snap = agent.snapshot();
  std::vector<Action> before;
  Rng r(1);
  std::vector<ExecutionObs> probes;
  for (int i = 0; i < 200; ++i) probes.push_back({r.below(20), r.below(101), 0.3 * (r.uniform() - 0.5)});
  for (const auto& o : probes) before.push_back(agent.act(o, false));
  tr.run(500);  // keep learning: the policy moves on
  agent.restore(snap);
  for (std::size_t i = 0; i < probes.size(); ++i) EXPECT_EQ(agent.act(probes[i], false), before[i]);
}

// ======================= Learning on the real task ==========================
TEST(AgentLearning, BeatsTwapAndCapturesMostOfTheOptimalGain) {
  ExecutionEnv env{ExecutionParams{}};
  // Measured across seeds 1-6 at this budget: 62-71% (and 45-61% at half
  // the budget, which is why this trains for 40,000 episodes).
  AgentConfig<3> cfg = execution_config();
  cfg.eps_decay_episodes = 20000;
  Agent<ExecutionEnv> agent(env, cfg);
  Trainer<ExecutionEnv> tr(agent, env, {.eval_every = 5000, .eval_episodes = 300, .detect_drift = false});
  const auto rep = tr.run(40000);
  const double share = captured(env, default_dp(), [&](const ExecutionObs& o) { return tr.deployed_act(o); });
  EXPECT_GT(share, 0.55) << "captured " << 100 * share << "% of the optimal gain over TWAP";
  // For scale: the simple rule "sell faster when the signal is negative".
  const double rule = captured(env, default_dp(), [](const ExecutionObs& o) { return o.signal < 0 ? Action{3} : Action{1}; });
  EXPECT_GT(share, rule + 0.15) << "rule captures " << 100 * rule << "%";

  // The champion's score never goes down in a stationary world: a policy is
  // only promoted if it scores at least as well on the same scenarios.
  for (std::size_t i = 1; i < rep.evals.size(); ++i) EXPECT_GE(rep.evals[i].champion, rep.evals[i - 1].champion);
  for (const auto& e : rep.evals) EXPECT_LE(e.challenger, rep.champion_score + 1e-9);  // the best policy is deployed
}

TEST(AgentLearning, LearnedWhatTheSignalMeans) {
  // Nobody told the agent that the signal predicts prices. If it learned
  // that, it sells more when the signal says the price will fall.
  ExecutionEnv env{ExecutionParams{}};
  Agent<ExecutionEnv> agent(env, execution_config());
  Trainer<ExecutionEnv> tr(agent, env, {.eval_every = 0, .detect_drift = false});
  tr.run(20000);
  double sold_when_falling = 0, sold_when_rising = 0;
  for (std::uint32_t t = 2; t < 16; ++t)
    for (std::uint32_t inv : {30u, 50u, 70u}) {
      sold_when_falling += env.lots_for(agent.act({t, inv, -0.15}, false), inv);
      sold_when_rising += env.lots_for(agent.act({t, inv, +0.15}, false), inv);
    }
  EXPECT_GT(sold_when_falling, 2.0 * sold_when_rising)
      << "falling " << sold_when_falling << " vs rising " << sold_when_rising;
}

TEST(AgentDrift, DetectorIsQuietOnNoiseAndFiresOnAShift) {
  DriftDetector d;
  Rng r(4);
  int false_alarms = 0;
  for (int i = 0; i < 20000; ++i) false_alarms += d.add(50.0 * r.normal());
  EXPECT_EQ(false_alarms, 0);
  int fired_after = -1;
  for (int i = 0; i < 1000 && fired_after < 0; ++i)
    if (d.add(-30.0 + 50.0 * r.normal())) fired_after = i;
  EXPECT_GE(fired_after, 0);
  EXPECT_LT(fired_after, 200);
  for (int i = 0; i < 1500; ++i) ASSERT_FALSE(d.add(-500.0)) << "warmup after an alarm";  // rebuilding a baseline
}

TEST(AgentDrift, NoticesThatTheWorldChangedAndRecovers) {
  ExecutionParams p;
  ExecutionEnv env(p);
  Agent<ExecutionEnv> agent(env, execution_config());
  Trainer<ExecutionEnv> tr(agent, env, {.eval_every = 0});
  const auto before = tr.run(20000);
  EXPECT_TRUE(before.drift_alarms.empty()) << "false alarm in a stationary world";

  // The signal's meaning flips. A copy of the agent continues without drift
  // detection, for comparison.
  ExecutionEnv env_off = env;
  Agent<ExecutionEnv> stubborn = agent;
  env.set_alpha(-p.alpha);
  env_off.set_alpha(-p.alpha);
  ExecutionParams flipped = p;
  flipped.alpha = -p.alpha;
  const ExecutionDP dp(flipped);
  auto score = [&](Agent<ExecutionEnv>& a) { return captured(env, dp, [&](const ExecutionObs& o) { return a.act(o, false); }); };
  EXPECT_LT(score(agent), 0.0);  // its old knowledge is now actively harmful

  const auto after = tr.run(10000);
  ASSERT_FALSE(after.drift_alarms.empty()) << "the change went unnoticed";
  EXPECT_LT(after.drift_alarms[0] - 20000, 500u) << "noticed only after " << after.drift_alarms[0] - 20000 << " episodes";
  Trainer<ExecutionEnv> tr_off(stubborn, env_off, {.eval_every = 0, .detect_drift = false});
  tr_off.run(10000);
  const double adaptive = score(agent), without = score(stubborn);
  EXPECT_GT(adaptive, 0.15) << "recovered only " << 100 * adaptive << "%";
  EXPECT_GT(adaptive, without + 0.2) << "with detection " << 100 * adaptive << "%, without " << 100 * without << "%";
}

// ======================= Experiment E2 (n-step, ceiling) ======================
TEST(AgentNStep, LearnsTheCorridorWithMultiStepAndMonteCarloTargets) {
  for (std::uint32_t n : {1u, 3u, 40u}) {  // 40 >= max_steps: pure Monte Carlo
    ChainEnv env({.length = 8, .max_steps = 40});
    AgentConfig<1> cfg;
    cfg.eps_decay_episodes = 300;
    cfg.bins = {14};
    cfg.n_step = n;
    Agent<ChainEnv> agent(env, cfg);
    Trainer<ChainEnv> tr(agent, env, {.eval_every = 0, .detect_drift = false});
    tr.run(600);
    EXPECT_EQ(evaluate(env, [&](const auto& o) { return agent.act(o, false); }, 1, 1).mean, -7.0) << "n = " << n;
  }
}

TEST(AgentNStep, ThreeStepTargetIsTheSumOfThreeRewardsPlusBootstrap) {
  // One-tiling, one-bin coder over a 1-d state: every state shares one
  // weight per action, so Q(s, a) = w[a] and targets are easy to compute.
  ChainEnv env({.length = 50, .max_steps = 100});
  AgentConfig<1> cfg;
  cfg.tilings = 1;
  cfg.bins = {1};
  cfg.alpha = 1.0;  // Q jumps straight to each target
  cfg.alpha_decay_episodes = 0;
  cfg.replay_per_step = 0;
  cfg.n_step = 3;
  Agent<ChainEnv> agent(env, cfg);
  ChainEnv::Observation s{0, 0};
  double q[2];
  agent.observe(s, 1, -1.0, {1, 1}, false);
  agent.observe({1, 1}, 1, -2.0, {2, 2}, false);
  agent.q_values(s, q);
  EXPECT_EQ(q[1], 0.0);  // nothing emitted until 3 steps are pending
  agent.observe({2, 2}, 1, -4.0, {3, 3}, false);
  agent.q_values(s, q);
  EXPECT_DOUBLE_EQ(q[1], -7.0);  // -1 - 2 - 4 + gamma^3 * max Q(s_3) = -7 + max(0, 0)
  agent.observe({3, 3}, 1, -1.0, {4, 4}, true);  // episode ends: the 3 pending steps flush, no bootstrap
  agent.q_values(s, q);
  EXPECT_DOUBLE_EQ(q[1], -1.0);  // last flushed transition: just the final reward
}

TEST(AgentRepresentation, TileCodingCanRepresentANearOptimalPolicy) {
  // E2's key finding: fitting the default 8x8x8 x 8-tiling coder to the TRUE
  // optimal Q* (from the DP) by plain regression gives a greedy policy at
  // 99.5% of optimal. The ~70% plateau of RL is therefore a learning problem
  // (noisy targets, uneven action coverage), not a representation problem.
  ExecutionParams p;
  ExecutionEnv env(p);
  const auto& dp = default_dp();
  TileCoder<3> coder(ExecutionEnv::kLo, ExecutionEnv::kHi, {8, 8, 8}, 8);
  TileQ q(coder.size(), 8, 5, 0.1);
  Rng rng(1), pol(2);
  for (int ep = 0; ep < 20000; ++ep) {
    q.set_alpha(std::max(0.005, 0.1 / (1 + ep / 3000.0)));
    auto o = env.reset(rng);
    for (;;) {
      std::array<std::uint32_t, kMaxTilings> idx;
      coder.encode(env.features(o), idx.data());
      for (Action a = 0; a < 5; ++a) q.update(idx.data(), a, dp.q_value(o.t, o.inventory, o.signal, a));
      const auto st = env.step(pol.uniform() < 0.2 ? pol.below(5) : dp.act(o), rng);
      o = st.obs;
      if (st.done) break;
    }
  }
  auto greedy = [&](const ExecutionObs& o) {
    std::array<std::uint32_t, kMaxTilings> idx;
    coder.encode(env.features(o), idx.data());
    double v[5];
    q.q_values(idx.data(), v);
    return static_cast<Action>(std::max_element(v, v + 5) - v);
  };
  EXPECT_GT(captured(env, dp, greedy), 0.97);
}

// ===================== Experiment E3 (policy bank) ===========================
TEST(AgentPolicyBank, RecognizesAReturningRegimeAndRedeploysItsPolicy) {
  // Measured (EXPERIMENTS.md E3, 20k/10k schedule): returning to a known
  // regime, the bank recalls its old champion within 20-131 episodes and is
  // at the old level (59%) 500 episodes after the switch; without the bank
  // the agent is at -82% to -109% then, relearning from scratch.
  ExecutionParams pa;
  ExecutionEnv ea(pa);
  const auto& dpa = default_dp();
  auto run = [&](bool bank, std::vector<std::uint64_t>* recalls) {
    ExecutionEnv env(pa);
    AgentConfig<3> cfg = execution_config();
    cfg.eps_decay_episodes = 8000;
    Agent<ExecutionEnv> agent(env, cfg);
    Trainer<ExecutionEnv> tr(agent, env, {.eval_every = 1000, .eval_episodes = 300, .policy_bank = bank});
    tr.run(16000);
    const double before = captured(ea, dpa, [&](const ExecutionObs& o) { return tr.deployed_act(o); }, 600);
    env.set_alpha(-1.0);  // regime B
    tr.run(6000);
    env.set_alpha(1.0);   // back to A
    const auto rep = tr.run(500);
    if (recalls) *recalls = rep.bank_recalls;
    const double after = captured(ea, dpa, [&](const ExecutionObs& o) { return tr.deployed_act(o); }, 600);
    return std::pair{before, after};
  };
  std::vector<std::uint64_t> recalls;
  const auto [before, with_bank] = run(true, &recalls);
  const auto [unused, without_bank] = run(false, nullptr);
  (void)unused;
  ASSERT_FALSE(recalls.empty()) << "the returning regime was not recognized";
  EXPECT_GT(with_bank, 0.9 * before) << "before " << before << " after recall " << with_bank;
  EXPECT_LT(without_bank, 0.0) << "without the bank the agent is still relearning: " << without_bank;
}

TEST(AgentPolicyBank, NoRecallWhenNothingBankedFitsTheNewWorld) {
  // First change ever: the bank only holds the outgoing champion, which just
  // failed, so the trainer must fall back to ordinary re-exploration.
  ExecutionEnv env{ExecutionParams{}};
  AgentConfig<3> cfg = execution_config();
  cfg.eps_decay_episodes = 6000;
  Agent<ExecutionEnv> agent(env, cfg);
  Trainer<ExecutionEnv> tr(agent, env, {.eval_every = 1000, .eval_episodes = 200});
  tr.run(12000);
  env.set_alpha(-1.0);
  const auto rep = tr.run(3000);
  ASSERT_FALSE(rep.drift_alarms.empty());
  EXPECT_TRUE(rep.bank_recalls.empty());
  EXPECT_EQ(tr.bank_size(), 1u);
  EXPECT_GT(agent.epsilon(), 0.1);  // re-exploring, not exploiting a recalled policy
}
