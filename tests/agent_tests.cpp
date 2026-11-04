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
#include "rag/agent/model_based.hpp"
#include "rag/agent/ope.hpp"
#include "rag/agent/trainer.hpp"
#include "rag/agent/variance_reduction.hpp"
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
  d.reset();  // the caller confirmed the change: rebuild the baseline
  for (int i = 0; i < 1500; ++i) ASSERT_FALSE(d.add(-500.0)) << "warmup after a reset";
}

TEST(AgentDrift, RebaseAfterAFalseAlarmStaysArmed) {
  // E13: after a rejected alarm the detector must not go blind for `warmup`
  // samples; rebase() re-centres it and it keeps watching.
  DriftDetector d;
  Rng r(5);
  for (int i = 0; i < 3000; ++i) d.add(10.0 * r.normal());
  int alarm = -1;
  for (int i = 0; i < 200 && alarm < 0; ++i)
    if (d.add(-8.0 + 10.0 * r.normal())) alarm = i;  // a modest dip, judged false by the caller
  ASSERT_GE(alarm, 0);
  d.rebase();
  int quiet = 0;
  for (int i = 0; i < 300; ++i) quiet += !d.add(-8.0 + 10.0 * r.normal());
  EXPECT_GE(quiet, 298) << "re-centred on the new level (measured: 0-1 alarms over 10 seeds)";
  int fired = -1;
  for (int i = 0; i < 100 && fired < 0; ++i)
    if (d.add(-60.0 + 10.0 * r.normal())) fired = i;  // a real change right after: caught at once
  EXPECT_GE(fired, 0);
  EXPECT_LT(fired, 20);
  // Two-sided: an improvement is a change too.
  DriftParams two;
  two.two_sided = true;
  DriftDetector u(two);
  for (int i = 0; i < 3000; ++i) u.add(10.0 * r.normal());
  int up = -1;
  for (int i = 0; i < 100 && up < 0; ++i)
    if (u.add(30.0 + 10.0 * r.normal())) up = i;
  EXPECT_GE(up, 0);
  EXPECT_TRUE(u.last_alarm_was_upward());
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

// ================= Experiment E5 (actor-critic learner) ======================
TEST(AgentActorCritic, LearnsTheCorridor) {
  ChainEnv env({.length = 8, .max_steps = 40});
  AgentConfig<1> cfg;
  cfg.learner = LearnerKind::ActorCritic;
  cfg.bins = {14};
  cfg.actor_alpha = 0.2;  // a tiny task: larger steps
  cfg.ac_decay_episodes = 2000;
  Agent<ChainEnv> agent(env, cfg);
  Trainer<ChainEnv> tr(agent, env, {.eval_every = 0, .detect_drift = false});
  tr.run(800);
  EXPECT_EQ(evaluate(env, [&](const auto& o) { return agent.act(o, false); }, 1, 1).mean, -7.0);
}

TEST(AgentActorCritic, BeatsQLearningAtEqualBudget) {
  // Measured at 10k episodes, seeds 1-3: actor-critic 75-78%, TileQ (n = 3)
  // 47-50%. At 40k: 80-86% vs 69-70%; at 200k: 86-89% vs 75-79%.
  ExecutionEnv env{ExecutionParams{}};
  auto train = [&](LearnerKind kind) {
    AgentConfig<3> cfg;
    cfg.learner = kind;
    cfg.eps_decay_episodes = 5000;
    ExecutionEnv e = env;
    auto agent = std::make_unique<Agent<ExecutionEnv>>(e, cfg);
    Trainer<ExecutionEnv> tr(*agent, e, {.eval_every = 0, .detect_drift = false});
    tr.run(10000);
    return captured(env, default_dp(), [&](const ExecutionObs& o) { return agent->act(o, false); });
  };
  const double ac = train(LearnerKind::ActorCritic), q = train(LearnerKind::TileQ);
  EXPECT_GT(ac, 0.65) << "actor-critic " << ac;
  EXPECT_GT(ac, q + 0.1) << "actor-critic " << ac << " vs TileQ " << q;
}

TEST(AgentActorCritic, PolicyActingIsAllocationFreeAndSnapshotsIncludeTheCritic) {
  ExecutionEnv env{ExecutionParams{}};
  AgentConfig<3> cfg;
  cfg.learner = LearnerKind::ActorCritic;
  Agent<ExecutionEnv> agent(env, cfg);
  Trainer<ExecutionEnv> tr(agent, env, {.eval_every = 0, .detect_drift = false});
  tr.run(300);
  const ExecutionObs o{4, 70, -0.08};
  begin_alloc_count();
  Action sink = 0;
  for (int i = 0; i < 1000; ++i) sink ^= agent.act(o, i % 2 == 0);
  EXPECT_EQ(end_alloc_count(), 0u);
  (void)sink;
  // Sampling covers more than one action while exploring; greedy is deterministic.
  std::set<Action> sampled;
  for (int i = 0; i < 200; ++i) sampled.insert(agent.act(o, true));
  EXPECT_GT(sampled.size(), 1u);
  EXPECT_EQ(agent.act(o, false), agent.act(o, false));
  // The critic is part of the learned state: restoring a snapshot restores value().
  const auto snap = agent.snapshot();
  const double v = agent.value(o);
  tr.run(300);
  agent.restore(snap);
  EXPECT_EQ(agent.value(o), v);
  EXPECT_EQ(agent.act_as(snap, o), agent.act(o, false));
}

// ============ Experiment E6 (robustness: never deploy below baseline) =========
TEST(AgentSafety, NeverDeploysAPolicyWorseThanTheBaseline) {
  // Weak-signal market (alpha 0.25): there is little to gain and the
  // Q-learner overfits noise. Measured: its champion scored -35% of the
  // optimal gain, i.e. worse than TWAP. With TWAP as baseline, the deployed
  // policy must not be significantly worse than TWAP.
  ExecutionParams p;
  p.alpha = 0.25;
  ExecutionEnv env(p);
  AgentConfig<3> cfg;
  cfg.eps_decay_episodes = 10000;
  Agent<ExecutionEnv> agent(env, cfg);
  Trainer<ExecutionEnv> tr(agent, env, {.eval_every = 2500, .eval_episodes = 600, .detect_drift = false});
  tr.set_baseline(twap);
  tr.run(20000);
  const auto deployed = compare(env, [&](const ExecutionObs& o) { return tr.deployed_act(o); }, twap, 1500, 61'000'000);
  EXPECT_GT(deployed.mean, -2.0 * deployed.std_error - 1e-9) << "deployed is worse than the baseline: " << deployed.mean;
  // In the default market the learned policy does beat TWAP and gets deployed.
  ExecutionEnv good{ExecutionParams{}};
  Agent<ExecutionEnv> agent2(good, execution_config());
  Trainer<ExecutionEnv> tr2(agent2, good, {.eval_every = 2500, .eval_episodes = 600, .detect_drift = false});
  tr2.set_baseline(twap);
  tr2.run(20000);
  EXPECT_TRUE(tr2.learned_policy_deployed());
  EXPECT_GT(compare(good, [&](const ExecutionObs& o) { return tr2.deployed_act(o); }, twap, 1000, 61'000'000).mean, 3.0);
}

// ======= Experiment E7 (warm start from a model, fine-tune, guard) ==========
TEST(AgentWarmStart, DistilledModelPlusGuardIsBestOfBoth) {
  // Measured (3 seeds): a model that is right gives a 99% policy when
  // distilled into the tiles, but RL fine-tuning wrecks it (35-41% after
  // 2,500 episodes: noisy targets swamp action gaps < 1). With the plan as
  // the guard's baseline the deployed policy stays at 100%. When the model is
  // wrong (permanent impact it lacks: 25-28%), fine-tuning improves on it
  // (78-82% at 40k), and the deployed policy is never below the plan, while
  // TileQ from scratch is at -24% to -167% for its first ~5,000 episodes.
  for (double gamma : {0.0, 0.01}) {
    ExecutionParams p;
    p.permanent_impact = gamma;
    ExecutionEnv env(p);
    const ExecutionDP truth(p);
    ModelBasedExecutionAgent mb(p, {.cost_model = CostModel::Shrunk, .learn_permanent_impact = false, .seed = 1});
    {
      Rng r(43);
      ExecutionEnv e(p);
      train_model_based(mb, e, 300, r);
      mb.replan();
    }
    auto plan = [&](const ExecutionObs& o) { return mb.act(o, false); };
    const ExecutionDP& model = *mb.plan();
    AgentConfig<3> cfg;
    cfg.eps_start = 0.1;
    cfg.eps_decay_episodes = 10000;
    Agent<ExecutionEnv> agent(env, cfg);
    ASSERT_TRUE(agent.distill(ExecutionEnv(model.params()),
                              [&](const ExecutionObs& o, Action a) { return model.q_value(o.t, o.inventory, o.signal, a); }, 10000, 777));
    EXPECT_EQ(agent.episodes(), 0u);  // no real experience spent
    const double plan_share = captured(env, truth, plan);
    EXPECT_NEAR(captured(env, truth, [&](const ExecutionObs& o) { return agent.act(o, false); }), plan_share, 0.05);
    Trainer<ExecutionEnv> tr(agent, env, {.eval_every = 1250, .eval_episodes = 600, .detect_drift = false});
    tr.set_baseline(plan);
    double worst = 1e9;
    for (int chunk = 0; chunk < 8; ++chunk) {
      tr.run(2500);
      worst = std::min(worst, captured(env, truth, [&](const ExecutionObs& o) { return tr.deployed_act(o); }, 600));
    }
    const double final_share = captured(env, truth, [&](const ExecutionObs& o) { return tr.deployed_act(o); });
    if (gamma == 0.0) {
      EXPECT_GT(plan_share, 0.97);
      EXPECT_GT(final_share, 0.97);  // the guard kept the model's plan deployed
    } else {
      EXPECT_LT(plan_share, 0.4);
      EXPECT_GT(final_share, plan_share + 0.3);  // real rewards fixed what the model got wrong
    }
    EXPECT_GT(worst, plan_share - 0.1) << "gamma " << gamma;  // never much below the plan while learning
  }
}

// ================= Experiment E8 (online regret: learning costs) =============
TEST(AgentOnline, LowExplorationCostsLessOnlineAndLearnsAsWell) {
  // Every training episode counted as real. Measured (seeds 1-3), cumulative
  // gain over TWAP after 10,000 episodes, in units of the optimal gain per
  // episode: eps 1 -> 0.02: -585 to -1027 (it is still paying for its random
  // start); eps 0.1 -> 0.02: +2497 to +2971. Zero-initialized Q-values are
  // optimistic about the bad actions (never selling ends in a dump costing
  // up to 500), so greedy selection explores them without help.
  ExecutionParams p;
  ExecutionEnv env(p);
  const auto& dp = default_dp();
  const double v_twap = dp.policy_value(twap);
  const double gain = dp.expected_start_value() - v_twap;
  auto online = [&](double eps_start, double& share) {
    AgentConfig<3> cfg = execution_config();
    cfg.eps_start = eps_start;
    Agent<ExecutionEnv> agent(env, cfg);
    Trainer<ExecutionEnv> tr(agent, env, {.eval_every = 0, .detect_drift = false});
    const auto rep = tr.run(10000);
    double cum = 0;
    for (double r : rep.returns) cum += r - v_twap;
    share = captured(env, dp, [&](const ExecutionObs& o) { return agent.act(o, false); });
    return cum / gain;
  };
  double share_hi = 0, share_lo = 0;
  const double cum_hi = online(1.0, share_hi), cum_lo = online(0.1, share_lo);
  EXPECT_LT(cum_hi, 0.0) << "eps 1 should still be under water at 10k episodes";
  EXPECT_GT(cum_lo, 1500.0);
  EXPECT_GT(share_lo, share_hi - 0.1) << share_lo << " vs " << share_hi;
  // A model-based agent earns most of the optimal gain while it learns: measured 745-913 of 1000.
  ModelBasedExecutionAgent mb(p, {.cost_model = CostModel::Shrunk, .seed = 1});
  ExecutionEnv e(p);
  Rng rng(1);
  double cum_mb = 0;
  for (int i = 0; i < 1000; ++i) {
    auto o = e.reset(rng);
    for (;;) {
      const Action a = mb.act(o, true);
      mb.note_action(o, a);
      const auto st = e.step(a, rng);
      mb.observe(o, a, st.reward, st.obs, st.done);
      cum_mb += st.reward;
      o = st.obs;
      if (st.done) break;
    }
    mb.end_episode();
    cum_mb -= v_twap;
  }
  EXPECT_GT(cum_mb / gain, 600.0);
}

// ============== Experiment E10 (off-policy evaluation) ======================
TEST(OffPolicy, EstimatorArithmeticOnAHandMadeLog) {
  // Episode 1: (t0, a1, r2, b 0.5), (t1, a0, r4, b 0.25). Episode 2: (t0, a0, r10, b 0.5), then over.
  // Target: action 1 at t = 0, action 0 after. Weights: ep1 2 then 8; ep2 0.
  std::vector<LoggedEpisode<ExecutionObs>> logs = {
      {{ExecutionObs{0, 100, 0}, 1, 2.0, 0.5}, {ExecutionObs{1, 95, 0}, 0, 4.0, 0.25}},
      {{ExecutionObs{0, 100, 0}, 0, 10.0, 0.5}},
  };
  auto pi = [](const ExecutionObs& o) { return o.t == 0 ? Action{1} : Action{0}; };
  const auto e = off_policy_estimates(logs, pi, [](const ExecutionObs&, Action) { return 1.0; },
                                      [](const ExecutionObs&) { return 3.0; }, 2);
  EXPECT_DOUBLE_EQ(e.pdis, (2 * 2 + 8 * 4) / 2.0);       // 18
  EXPECT_DOUBLE_EQ(e.wpdis, 4.0 / 2.0 + 32.0 / 8.0);     // 6
  EXPECT_DOUBLE_EQ(e.dr, ((2 * 1 + 3) + (8 * 3 + 2 * 3) + 3) / 2.0);  // 19
  EXPECT_DOUBLE_EQ(e.min_ess, 1.0);
  EXPECT_EQ(e.uncovered_steps, 0u);
  // A target nobody followed leaves its steps uncovered.
  const auto none = off_policy_estimates(logs, [](const ExecutionObs&) { return Action{4}; },
                                         [](const ExecutionObs&, Action) { return 0.0; }, [](const ExecutionObs&) { return 0.0; }, 2);
  EXPECT_EQ(none.uncovered_steps, 2u);
  EXPECT_EQ(none.min_ess, 0.0);
}

TEST(AgentExecutionDP, PolicyValueTablesAreSelfConsistent) {
  // Evaluating the optimal policy reproduces the optimal value function; for
  // any policy, V(s) = Q(s, pi(s)); and a worse policy is worth less everywhere it differs.
  ExecutionParams p;
  p.permanent_impact = 0.005;
  p.risk_aversion = 0.1;
  const ExecutionDP dp(p, 61);
  const auto opt = dp.evaluate_policy([&](const ExecutionObs& o) { return dp.act(o); });
  const auto tw = dp.evaluate_policy(twap);
  EXPECT_NEAR(opt.start_value(), dp.expected_start_value(), 1e-9);
  Rng rng(3);
  for (int i = 0; i < 300; ++i) {
    const std::uint32_t t = rng.below(p.horizon), inv = 1 + rng.below(p.lots);
    const double g = (rng.uniform() - 0.5) * 4 * p.signal_sd;
    EXPECT_NEAR(opt.value(t, inv, g), dp.value(t, inv, g), 1e-6 * (1 + std::abs(dp.value(t, inv, g))));
    EXPECT_GE(opt.value(t, inv, g) + 1e-9, tw.value(t, inv, g));
  }
  // At grid points V(s) == Q(s, pi(s)) exactly (off the grid V is interpolated).
  const double lo = -4 * p.signal_sd, step = 8 * p.signal_sd / 60;
  for (std::uint32_t j = 0; j < 61; j += 6) {
    const double g = lo + j * step;
    for (std::uint32_t inv : {1u, 37u, 100u}) {
      EXPECT_NEAR(tw.value(7, inv, g), tw.q(7, inv, g, ExecutionEnv::kTwapAction), 1e-9);
    }
  }
}

TEST(AgentExecutionDP, BandedSolveSatisfiesTheBellmanEquationExactly) {
  // The solve uses a precomputed banded expectation operator; q_value() uses
  // the 31-point quadrature directly. At grid points the two must agree:
  // V(t, inv, g_j) == max_a Q(t, inv, g_j, a), for every market variant.
  // (Measured against the previous per-action quadrature implementation:
  // max |dV| 6e-13, no action changes on 360,000 random states; 9x faster.)
  std::vector<ExecutionParams> ps(4);
  ps[1].permanent_impact = 0.01;
  ps[1].risk_aversion = 0.3;
  ps[2].signal_rho = 0.99;
  ps[3].horizon = 7;
  ps[3].lots = 33;
  ps[3].alpha = -2;
  for (const auto& p : ps) {
    const ExecutionDP dp(p, 41);
    const double lo = -4 * p.signal_sd, step = 8 * p.signal_sd / 40;
    for (std::uint32_t t = 0; t < p.horizon; t += 3)
      for (std::uint32_t inv = 0; inv <= p.lots; inv += 4)
        for (std::uint32_t j = 0; j < 41; j += 5) {
          const double g = lo + j * step;
          double best = -1e300;
          for (Action a = 0; a < ExecutionEnv::kNumActions; ++a) best = std::max(best, dp.q_value(t, inv, g, a));
          ASSERT_NEAR(dp.value(t, inv, g), inv == 0 ? 0.0 : best, 1e-9 * (1 + std::abs(best))) << t << " " << inv << " " << j;
        }
  }
}

// ============ Experiment E16 (a neural Q-learner, from scratch) ==============
TEST(AgentNeural, LearnsFasterThanTileCodingOnExecution) {
  // Measured with the full-size network (64 hidden, batch 32): 66-81% of
  // the optimal gain at 5,000 episodes and 84-89% at 40,000, against 8-21%
  // and 65-69% for TileQ. Here a small network for speed: 28-57% after
  // 3,000 episodes (seeds 1-3) vs -72% to +19% for TileQ.
  ExecutionParams p;
  ExecutionEnv env(p);
  const auto& dp = default_dp();
  AgentConfig<3> nq = execution_config();
  nq.learner = LearnerKind::NeuralQ;
  nq.eps_decay_episodes = 3000;
  nq.nn_hidden = 32;
  nq.nn_batch = 8;
  AgentConfig<3> tq = execution_config();
  tq.eps_decay_episodes = 3000;
  double share[2];
  for (int k = 0; k < 2; ++k) {
    ExecutionEnv e(p);
    Agent<ExecutionEnv> agent(e, k ? tq : nq);
    Trainer<ExecutionEnv> tr(agent, e, {.eval_every = 0, .detect_drift = false});
    tr.run(3000);
    share[k] = captured(env, dp, [&](const ExecutionObs& o) { return agent.act(o, false); });
  }
  EXPECT_GT(share[0], 0.25);
  EXPECT_GT(share[0], share[1] + 0.15) << "neural " << share[0] << " vs tiles " << share[1];
}

TEST(AgentNeural, EvaluationDoesNotPerturbLearningAndSnapshotsAct) {
  // The trainer scores champions by restoring them into the learner and
  // back. For NeuralQ that must leave the online network, its Adam state
  // and the target network exactly as they were.
  AgentConfig<3> cfg = execution_config();
  cfg.learner = LearnerKind::NeuralQ;
  cfg.nn_hidden = 8;
  cfg.nn_batch = 4;
  cfg.nn_warmup = 200;
  ExecutionEnv e1{ExecutionParams{}}, e2{ExecutionParams{}};
  Agent<ExecutionEnv> a(e1, cfg), b(e2, cfg);
  Trainer<ExecutionEnv> ta(a, e1, {.eval_every = 25, .eval_episodes = 20, .detect_drift = false});
  Trainer<ExecutionEnv> tb(b, e2, {.eval_every = 0, .detect_drift = false});
  ta.run(200);
  tb.run(200);
  EXPECT_EQ(a.snapshot().net.weights(), b.snapshot().net.weights());
  // A snapshot acts like the network it was taken from, after the learner moves on.
  const auto snap = a.snapshot();
  Rng r(4);
  std::vector<ExecutionObs> obs(50);
  for (auto& o : obs) o = {r.below(20), r.below(101), 0.1 * r.normal()};
  std::vector<Action> before;
  for (const auto& o : obs) before.push_back(a.act(o, false));
  ta.run(100);
  for (std::size_t i = 0; i < obs.size(); ++i) EXPECT_EQ(a.act_as(snap, obs[i]), before[i]);
}

// =========== Experiment E17 (a control variate for the price noise) =========
TEST(VarianceReduction, ControlledRewardsKeepValuesAndCutOnlyTheInnovation) {
  // r~ = r - I' * (dS - alpha * g) has the same expectation given (s, a), so
  // every policy keeps its value. How much variance it removes depends on the
  // innovation's share. Measured (TWAP, 20,000 episodes): default market
  // sd 54.4 -> 52.9 (the signal path, sd 53, dominates); noise_sd 0.2:
  // 72.9 -> ~53, and TileQ at 40k went from 49-61% to 67-71%.
  auto twap_sd = [](const ExecutionParams& p, bool controlled) {
    const ExecutionEnv plain(p);
    const InnovationControlledEnv cv{ExecutionEnv(p), p.alpha};
    const auto e = controlled ? evaluate(cv, twap, 4000, 1) : evaluate(plain, twap, 4000, 1);
    return std::array<double, 2>{e.mean, e.std_error * std::sqrt(4000.0)};
  };
  ExecutionParams noisy;
  noisy.noise_sd = 0.2;
  for (const ExecutionParams& p : {ExecutionParams{}, noisy}) {
    const auto a = twap_sd(p, false), b = twap_sd(p, true);
    EXPECT_NEAR(a[0], b[0], 3 * a[1] / std::sqrt(4000.0));  // same value (paired scenarios)
    if (p.noise_sd > 0.1) {
      EXPECT_LT(b[1], 0.8 * a[1]);
    } else {
      EXPECT_GT(b[1], 0.9 * a[1]);
    }
  }
  // The optimal policy's paired gain over TWAP is preserved too.
  const auto& dp = default_dp();
  const InnovationControlledEnv cv{ExecutionEnv{ExecutionParams{}}, 1.0};
  const ExecutionEnv plain{ExecutionParams{}};
  const auto g1 = compare(plain, [&](const ExecutionObs& o) { return dp.act(o); }, twap, 2000, 1);
  const auto g2 = compare(cv, [&](const ExecutionObs& o) { return dp.act(o); }, twap, 2000, 1);
  EXPECT_NEAR(g1.mean, g2.mean, 3 * g1.std_error);
}

TEST(VarianceReduction, EstimatesTheDriftFromMarketData) {
  // Without an oracle, alpha_hat is a regression of observed moves on the
  // signal (measured: 0.999-1.001 after 40k training episodes).
  InnovationControlledEnv env{ExecutionEnv{ExecutionParams{}}};
  Rng rng(9);
  for (int ep = 0; ep < 300; ++ep) {
    env.reset(rng);
    for (;;) {
      const auto st = env.step(rng.below(5), rng);
      if (st.done) break;
    }
  }
  EXPECT_NEAR(env.alpha_hat(), 1.0, 0.05);
}
