// agent_coverage_tests.cpp — exact checks of agent behaviour the other test
// files leave unexercised: configuration limits, discounting, double
// Q-learning, the actor-critic update, episodic Monte Carlo returns, the
// CUSUM arithmetic, evaluation statistics, off-policy reductions, the
// model-based agents' estimators, and the trainer's drift bookkeeping.
//
// Expected values come from the documented formulas (TASK_SPECS.md, the
// header comments) worked by hand on cases small enough to do so: mostly a
// one-tile coder over ChainEnv, where Q(s, a) is a single weight per action.
#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include "rag/agent/execution_dp.hpp"
#include "rag/agent/market_making.hpp"
#include "rag/agent/model_based.hpp"
#include "rag/agent/ope.hpp"
#include "rag/agent/trainer.hpp"

using namespace rag::agent;

namespace {

// With tilings = 1 and bins = {1}, every ChainEnv position short of the goal
// activates tile 0, so Q(s, a) is one weight per action and every state
// shares it: targets can be worked out by hand.
constexpr std::array<std::uint32_t, kMaxTilings> kTile0{};

AgentConfig<1> one_tile_config() {
  AgentConfig<1> c;
  c.tilings = 1;
  c.bins = {1};
  c.alpha = 1.0;               // Q jumps straight to each target
  c.alpha_decay_episodes = 0;  // constant step size
  c.replay_per_step = 0;       // no replayed updates in between
  return c;
}

ChainEnv long_chain() { return ChainEnv({.length = 50, .max_steps = 100}); }

// One-step environment for exact statistics: the reward is the action plus
// `scale` times an integer drawn from the scenario's RNG, so every policy run
// on the same scenario sees the same draw.
struct DrawEnv {
  struct Observation {
    std::uint32_t step;
  };
  static constexpr std::size_t kNumActions = 4;
  static constexpr std::size_t kStateDims = 1;
  static constexpr std::array<double, 1> kLo{0.0}, kHi{1.0};
  double scale = 0.0;

  Observation reset(Rng&) noexcept { return {0}; }
  Step<Observation> step(Action a, Rng& rng) noexcept {
    return {{1}, static_cast<double>(a) + scale * std::floor(100.0 * rng.uniform()), true};
  }
  std::array<double, 1> features(const Observation&) const noexcept { return {0.0}; }
};
static_assert(Environment<DrawEnv>);

}  // namespace

// ============================== Configuration ===============================
TEST(AgentCoverage, OutOfRangeSettingsAreRejected) {
  // Documented ranges: tilings in [1, 32], alpha > 0, 0 < gamma <= 1,
  // 0 <= eps_end <= eps_start <= 1 (agent.hpp); bins > 0 (TileCoder); kernel
  // widths > 0 (RandomFourierEncoder); neighbors in [1, 32] (EpisodicMemory);
  // cost tables of lots + 1 entries (ExecutionDP).
  const ChainEnv env;
  EXPECT_THROW(Agent<ChainEnv>(env, AgentConfig<1>{.tilings = 33}), std::invalid_argument);
  EXPECT_NO_THROW(Agent<ChainEnv>(env, AgentConfig<1>{.tilings = 32}));
  EXPECT_THROW(Agent<ChainEnv>(env, AgentConfig<1>{.alpha = 0.0}), std::invalid_argument);
  EXPECT_THROW(Agent<ChainEnv>(env, AgentConfig<1>{.alpha = std::nan("")}), std::invalid_argument);
  EXPECT_THROW(Agent<ChainEnv>(env, AgentConfig<1>{.gamma = 0.0}), std::invalid_argument);
  EXPECT_THROW(Agent<ChainEnv>(env, AgentConfig<1>{.gamma = 1.5}), std::invalid_argument);
  EXPECT_NO_THROW(Agent<ChainEnv>(env, AgentConfig<1>{.gamma = 1.0}));
  EXPECT_THROW(Agent<ChainEnv>(env, AgentConfig<1>{.eps_start = 0.5, .eps_end = -0.1}), std::invalid_argument);
  EXPECT_THROW(Agent<ChainEnv>(env, AgentConfig<1>{.eps_start = 1.5}), std::invalid_argument);
  EXPECT_NO_THROW(Agent<ChainEnv>(env, AgentConfig<1>{.eps_start = 0.0, .eps_end = 0.0}));

  EXPECT_THROW(TileCoder<1>({0.0}, {1.0}, {0}, 8), std::invalid_argument);
  EXPECT_THROW(TileCoder<1>({0.0}, {1.0}, {4}, 33), std::invalid_argument);
  using Rff = RandomFourierEncoder<1, 64>;
  EXPECT_THROW(Rff({0.0}, 1), std::invalid_argument);
  EXPECT_THROW(Rff({-1.0}, 1), std::invalid_argument);
  using Memory = EpisodicMemory<64, 2>;
  EXPECT_THROW(Memory(EpisodicParams{.capacity = 8, .neighbors = 0}), std::invalid_argument);
  EXPECT_THROW(Memory(EpisodicParams{.capacity = 8, .neighbors = 33}), std::invalid_argument);
  EXPECT_NO_THROW(Memory(EpisodicParams{.capacity = 8, .neighbors = 32}));

  const ExecutionParams p;
  EXPECT_THROW(ExecutionDP(p, std::vector<double>(p.lots + 1), std::vector<double>(3), 21), std::invalid_argument);
}

// ================================ TileQ agent ===============================
TEST(AgentCoverage, DiscountedNStepTargetsByHand) {
  // gamma = 0.5, n = 3: y = r0 + r1 / 2 + r2 / 4 + max_a Q(s3, a) / 8, the
  // sum truncated (and the bootstrap dropped) at the end of the episode.
  // init_q = 8 makes the bootstrap visible.
  AgentConfig<1> cfg = one_tile_config();
  cfg.gamma = 0.5;
  cfg.n_step = 3;
  cfg.init_q = 8.0;
  Agent<ChainEnv> agent(long_chain(), cfg);
  const ChainEnv::Observation s{0, 0};
  double q[2];
  agent.q_values(s, q);
  EXPECT_EQ(q[0], 8.0);
  EXPECT_EQ(q[1], 8.0);
  agent.observe({0, 0}, 1, -1.0, {1, 1}, false);
  agent.observe({1, 1}, 1, -2.0, {2, 2}, false);
  agent.observe({2, 2}, 1, -4.0, {3, 3}, false);
  agent.q_values(s, q);
  EXPECT_DOUBLE_EQ(q[1], -1.0 - 1.0 - 1.0 + 8.0 / 8);  // -2
  EXPECT_DOUBLE_EQ(q[0], 8.0);
  agent.observe({3, 3}, 1, -8.0, {4, 4}, false);
  agent.q_values(s, q);
  EXPECT_DOUBLE_EQ(q[1], -2.0 - 2.0 - 2.0 + std::max(8.0, -2.0) / 8);  // -5: the bootstrap takes the max
  agent.observe({4, 4}, 0, -6.0, {3, 5}, true);
  // The episode ends: the three pending steps flush oldest first, without a
  // bootstrap: (-4, -8, -6) -> -9.5 and (-8, -6) -> -11 on action 1, then
  // (-6) -> -6 on action 0.
  agent.q_values(s, q);
  EXPECT_DOUBLE_EQ(q[1], -8.0 + 0.5 * -6.0);
  EXPECT_DOUBLE_EQ(q[0], -6.0);
}

TEST(AgentCoverage, DoubleQUpdatesOneEstimatorTowardTheOthersValue) {
  // van Hasselt (2010): a coin picks the estimator that learns; it chooses
  // a' = argmax of its own values and the other estimator values a'. The
  // agent acts on the average of the two.
  int q_learned = 0, q2_learned = 0;
  for (std::uint64_t seed = 1; seed <= 8; ++seed) {
    AgentConfig<1> cfg = one_tile_config();
    cfg.n_step = 1;
    cfg.double_q = true;
    cfg.seed = seed;
    Agent<ChainEnv> agent(long_chain(), cfg);
    const ChainEnv::Observation s{0, 0}, s1{1, 1};
    agent.observe(s, 0, 3.0, s1, true);  // terminal: the learner jumps to the reward
    agent.observe(s, 1, 1.0, s1, true);
    const auto before = agent.snapshot();
    const std::uint32_t* t = kTile0.data();
    EXPECT_EQ(before.q.q(t, 0) + before.q2.q(t, 0), 3.0);  // each reward went to exactly one estimator
    EXPECT_EQ(before.q.q(t, 1) + before.q2.q(t, 1), 1.0);
    double avg[2];
    agent.q_values(s, avg);
    for (Action a : {Action{0}, Action{1}}) EXPECT_EQ(avg[a], 0.5 * (before.q.q(t, a) + before.q2.q(t, a)));

    agent.observe(s, 1, 10.0, s1, false);  // non-terminal: bootstraps
    const auto after = agent.snapshot();
    const bool q_moved = after.q.weights() != before.q.weights();
    const bool q2_moved = after.q2.weights() != before.q2.weights();
    ASSERT_NE(q_moved, q2_moved) << "seed " << seed << ": exactly one estimator learns per update";
    const TileQ& learner = q_moved ? before.q : before.q2;
    const TileQ& judge = q_moved ? before.q2 : before.q;
    const TileQ& learned = q_moved ? after.q : after.q2;
    const Action best = learner.q(t, 1) > learner.q(t, 0) ? Action{1} : Action{0};  // first maximum on ties
    EXPECT_EQ(learned.q(t, 1), 10.0 + judge.q(t, best)) << "seed " << seed;
    EXPECT_EQ(learned.q(t, 0), learner.q(t, 0));
    // Here the double target always differs from plain Q-learning's.
    EXPECT_NE(learned.q(t, 1), 10.0 + std::max(learner.q(t, 0), learner.q(t, 1)));
    ++(q_moved ? q_learned : q2_learned);
  }
  EXPECT_GT(q_learned, 0);
  EXPECT_GT(q2_learned, 0);

  // Both estimators follow the annealed step size: 0.2 / (1 + 10 / 10).
  AgentConfig<1> cfg;
  cfg.double_q = true;
  cfg.alpha = 0.2;
  cfg.alpha_decay_episodes = 10;
  Agent<ChainEnv> agent(ChainEnv{}, cfg);
  for (int i = 0; i < 10; ++i) agent.end_episode();
  const auto snap = agent.snapshot();
  EXPECT_DOUBLE_EQ(agent.step_size(), 0.1);
  EXPECT_DOUBLE_EQ(snap.q.alpha(), 0.1);
  EXPECT_DOUBLE_EQ(snap.q2.alpha(), 0.1);
}

TEST(AgentCoverage, DistillIsTileQOnlyAndSpendsNoExperience) {
  // A teacher that knows the corridor: right is worth 2 more than left.
  const ChainEnv env({.length = 8, .max_steps = 40});
  int calls = 0;
  auto teacher = [&](const ChainEnv::Observation& o, Action a) {
    ++calls;
    const double to_go = 7.0 - o.pos;
    return a == 1 ? -to_go : -to_go - 2.0;
  };
  for (LearnerKind kind : {LearnerKind::Episodic, LearnerKind::ActorCritic}) {
    AgentConfig<1> cfg;
    cfg.learner = kind;
    Agent<ChainEnv> agent(env, cfg);
    const auto weights = agent.q_learner().weights();
    EXPECT_FALSE(agent.distill(env, teacher, 50, 1));  // "returns false (and does nothing) for other learners"
    EXPECT_EQ(calls, 0);
    EXPECT_EQ(agent.q_learner().weights(), weights);
  }
  AgentConfig<1> cfg;
  cfg.bins = {14};
  cfg.eps_start = 0.7;
  cfg.double_q = true;
  Agent<ChainEnv> agent(env, cfg);
  const double alpha = agent.q_learner().alpha();
  ASSERT_TRUE(agent.distill(env, teacher, 300, 5));
  EXPECT_GT(calls, 0);
  for (std::uint32_t pos = 0; pos + 1 < 8; ++pos) EXPECT_EQ(agent.act({pos, 0}, false), 1u) << "at " << pos;
  // Episode count, schedules and replay are untouched.
  EXPECT_EQ(agent.episodes(), 0u);
  EXPECT_EQ(agent.epsilon(), 0.7);
  EXPECT_DOUBLE_EQ(agent.q_learner().alpha(), alpha);
  EXPECT_EQ(agent.replay_size(), 0u);
  // With double Q-learning both estimators start from the distilled values.
  const auto snap = agent.snapshot();
  EXPECT_EQ(snap.q2.weights(), snap.q.weights());
  EXPECT_DOUBLE_EQ(snap.q2.alpha(), alpha);
}

// =============================== Actor-critic ===============================
TEST(AgentCoverage, ActorCriticUpdateIsTheSoftmaxPolicyGradient) {
  // TASK_SPECS 5: delta = r + gamma V(s') - V(s); the critic moves toward
  // r + gamma V(s'); every logit moves by step * delta * (1[b = a] - pi(b|s)) / T.
  // With one tile V(s') = V(s), so a non-terminal step has delta = r.
  auto config = [](double temperature, double reexplore_temperature) {
    AgentConfig<1> c;
    c.learner = LearnerKind::ActorCritic;
    c.tilings = 1;
    c.bins = {1};
    c.actor_alpha = 0.5;
    c.critic_alpha = 0.5;
    c.temperature = temperature;
    c.ac_reexplore_temperature = reexplore_temperature;
    c.ac_decay_episodes = 4;  // both step sizes scale by 1 / (1 + e / 4)
    return c;
  };
  const ChainEnv::Observation s{0, 0}, s1{1, 1}, s2{2, 2};
  double h[2];
  {
    Agent<ChainEnv> agent(long_chain(), config(1.0, 0.0));
    agent.observe(s, 1, -1.0, s1, false);  // delta = -1, pi = (1/2, 1/2)
    agent.q_values(s, h);
    EXPECT_DOUBLE_EQ(agent.value(s), -0.5);
    EXPECT_DOUBLE_EQ(h[0], 0.25);
    EXPECT_DOUBLE_EQ(h[1], -0.25);
    EXPECT_EQ(agent.act(s, false), 0u);  // greedy = the policy's mode
    agent.observe(s1, 0, -1.0, s2, true);  // terminal: delta = -1 - V(s) = -0.5
    const double pi0 = 1.0 / (1.0 + std::exp(-0.5));  // softmax(0.25, -0.25)
    agent.q_values(s, h);
    EXPECT_DOUBLE_EQ(agent.value(s), -0.75);
    EXPECT_NEAR(h[0], 0.25 + 0.5 * -0.5 * (1.0 - pi0), 1e-12);
    EXPECT_NEAR(h[1], -0.25 + 0.5 * -0.5 * (0.0 - (1.0 - pi0)), 1e-12);
  }
  // One non-terminal step (delta = -1) at the given temperature and step size.
  auto expect_step = [&](Agent<ChainEnv>& agent, double temperature, double step) {
    double before[2], after[2];
    agent.q_values(s, before);
    const double m = std::max(before[0], before[1]);
    const double e0 = std::exp((before[0] - m) / temperature), e1 = std::exp((before[1] - m) / temperature);
    const double pi[2] = {e0 / (e0 + e1), e1 / (e0 + e1)};
    agent.observe(s, 1, -1.0, s1, false);
    agent.q_values(s, after);
    for (Action b : {Action{0}, Action{1}})
      EXPECT_NEAR(after[b], before[b] + step * -1.0 * ((b == 1 ? 1.0 : 0.0) - pi[b]) / temperature, 1e-12)
          << "T " << temperature << " step " << step << " logit " << b;
  };
  {
    Agent<ChainEnv> agent(long_chain(), config(2.0, 0.0));
    expect_step(agent, 2.0, 0.5);
    agent.q_values(s, h);
    EXPECT_DOUBLE_EQ(h[0], 0.125);  // 0.5 * -1 * (0 - 1/2) / 2
  }
  {
    // ac_reexplore_temperature: during a boost the policy is sampled (and
    // its gradient taken) at T * (1 + 3 * boost_left / boost_total).
    Agent<ChainEnv> agent(long_chain(), config(1.0, 3.0));
    expect_step(agent, 1.0, 0.5);  // no boost yet
    agent.reexplore(0.5, 4);
    expect_step(agent, 4.0, 0.5);
    agent.end_episode();
    agent.end_episode();
    expect_step(agent, 2.5, 0.5 / (1.0 + 2.0 / 4));
    agent.end_episode();
    agent.end_episode();
    expect_step(agent, 1.0, 0.5 / (1.0 + 4.0 / 4));  // the boost is over
  }
  {
    // Without ac_reexplore_temperature a boost doesn't change the temperature.
    Agent<ChainEnv> agent(long_chain(), config(1.0, 0.0));
    agent.reexplore(0.5, 4);
    expect_step(agent, 1.0, 0.5);
  }
}

// ============================= Episodic memory ==============================
TEST(AgentCoverage, EpisodicAgentLearnsDiscountedReturnsAndIsOptimisticOnlyWhileExploring) {
  AgentConfig<1> cfg;
  cfg.learner = LearnerKind::Episodic;
  cfg.gamma = 0.5;
  cfg.eps_start = cfg.eps_end = 0.0;  // no random actions: only optimism about untried actions explores
  cfg.memory = {.capacity = 16, .neighbors = 1, .same_situation = 0.97f, .learning_rate = 0.2};
  cfg.kernel_width = {0.01};  // positions 1/9 apart are unrelated situations
  Agent<ChainEnv> agent(ChainEnv({.length = 10, .max_steps = 60}), cfg);
  const ChainEnv::Observation p0{0, 0}, p1{1, 1}, p2{2, 2};
  EXPECT_EQ(agent.value(p0), 0.0);  // nothing known yet
  agent.observe(p0, 1, -1.0, p1, false);
  agent.observe(p1, 1, -2.0, p2, false);
  agent.observe(p2, 0, -8.0, p1, true);
  EXPECT_EQ(agent.memory().size(), 0u);  // memories are written when the episode ends
  agent.end_episode();
  // Monte Carlo returns-to-go: G2 = -8, G1 = -2 + G2 / 2 = -6, G0 = -1 + G1 / 2 = -4.
  ASSERT_EQ(agent.memory().size(), 3u);
  double q[2];
  agent.q_values(p0, q);
  EXPECT_DOUBLE_EQ(q[1], -4.0);
  EXPECT_EQ(q[0], 0.0);  // untried here: reads as 0
  agent.q_values(p1, q);
  EXPECT_DOUBLE_EQ(q[1], -6.0);
  agent.q_values(p2, q);
  EXPECT_DOUBLE_EQ(q[0], -8.0);
  EXPECT_EQ(q[1], 0.0);
  EXPECT_DOUBLE_EQ(agent.value(p2), -8.0);  // the best *known* action, not the 0 an untried one reads as
  // Untried actions are tried while exploring and avoided when acting for real.
  EXPECT_EQ(agent.act(p2, true), 1u);
  EXPECT_EQ(agent.act(p2, false), 0u);
  EXPECT_EQ(agent.act(p0, true), 0u);
  EXPECT_EQ(agent.act(p0, false), 1u);
  const auto snap = agent.snapshot();
  EXPECT_EQ(agent.act_as(snap, p0), 1u);  // known beats untried, though untried reads 0 > -4
  EXPECT_EQ(agent.act_as(snap, p2), 0u);
  // The same situation again updates its slot by the EMA: -4 + 0.2 * (-9 + 4) = -5.
  agent.observe(p0, 1, -9.0, p1, true);
  agent.end_episode();
  EXPECT_EQ(agent.memory().size(), 3u);
  agent.q_values(p0, q);
  EXPECT_DOUBLE_EQ(q[1], -5.0);
  agent.end_episode();  // an episode with no steps writes nothing
  EXPECT_EQ(agent.memory().size(), 3u);
}

// ================================ Learners ==================================
TEST(AgentCoverage, TileQBlendIsPolyakAveraging) {
  TileQ a(10, 2, 3, 0.5), b(10, 2, 3, 0.5);
  const std::uint32_t s[2] = {1, 7};
  a.update(s, 0, 8.0);   // Q_a(s, 0) = 4
  b.update(s, 0, -8.0);  // Q_b(s, 0) = -4
  b.update(s, 2, 2.0);   // Q_b(s, 2) = 1
  EXPECT_EQ(a.actions(), 3u);
  const TileQ a0 = a;
  a.blend(b, 0.0);
  EXPECT_EQ(a.weights(), a0.weights());
  a.blend(b, 0.25);  // (1 - w) * a + w * b
  EXPECT_DOUBLE_EQ(a.q(s, 0), 0.75 * 4.0 + 0.25 * -4.0);
  EXPECT_DOUBLE_EQ(a.q(s, 2), 0.25 * 1.0);
  EXPECT_EQ(a.q(s, 1), 0.0);
  a.blend(b, 1.0);
  EXPECT_EQ(a.weights(), b.weights());
  // Different sizes: only the overlapping weights are blended.
  const TileQ small(1, 2, 3, 0.5, 4.0);  // 3 weights of init_q / tilings = 2
  TileQ c = a0;
  c.blend(small, 1.0);
  for (std::size_t i = 0; i < c.weights().size(); ++i) EXPECT_EQ(c.weights()[i], i < 3 ? 2.0 : a0.weights()[i]) << i;
}

TEST(AgentCoverage, TileCoderOffsetsAreAsymmetricAsDocumented) {
  // Tiling k is shifted by k * (1, 3) / tilings of a cell (mod 1): with 4
  // tilings, (0, 0), (0.25, 0.75), (0.5, 0.5), (0.75, 0.25). 4 x 2 cells,
  // plus one overhanging cell per dimension: 15 cells per tiling.
  TileCoder<2> tc({0.0, 0.0}, {1.0, 1.0}, {4, 2}, 4);
  ASSERT_EQ(tc.size(), 60u);
  std::array<std::uint32_t, kMaxTilings> idx{};
  tc.encode({0.45, 0.2}, idx.data());  // in cell units: (1.8, 0.4)
  const std::uint32_t stride = 5;      // cells along dimension 0
  EXPECT_EQ(idx[0], 0 * 15 + 1 + 0 * stride);  // (1.8, 0.4)
  EXPECT_EQ(idx[1], 1 * 15 + 2 + 1 * stride);  // (2.05, 1.15)
  EXPECT_EQ(idx[2], 2 * 15 + 2 + 0 * stride);  // (2.3, 0.9)
  EXPECT_EQ(idx[3], 3 * 15 + 2 + 0 * stride);  // (2.55, 0.65)
  tc.encode({1.0, 1.0}, idx.data());  // the top corner is the last cell of every tiling
  for (std::uint32_t k = 0; k < 4; ++k) EXPECT_EQ(idx[k], k * 15 + 14);
  tc.encode({0.0, 0.0}, idx.data());
  for (std::uint32_t k = 0; k < 4; ++k) EXPECT_EQ(idx[k], k * 15);
}

// ============================== Drift detector ==============================
TEST(AgentCoverage, CusumArithmeticByHand) {
  // lag 1: a sample joins the baseline one step later. Armed once the
  // baseline holds 2 samples. S = min(0, S + z + k), alarm when S < -h.
  const DriftParams p{.lag = 1, .allowance = 0.5, .threshold = 2.0, .baseline_rate = 0.001, .warmup = 2};
  DriftDetector d(p);
  EXPECT_FALSE(d.add(-1.0));
  EXPECT_FALSE(d.add(1.0));  // baseline {-1}: not armed
  EXPECT_EQ(d.evidence(), 0.0);
  EXPECT_FALSE(d.add(-1.0));  // baseline {-1, 1}: mean 0, sd 1, z = -1: S = -0.5
  EXPECT_DOUBLE_EQ(d.evidence(), 0.5);
  const double s3 = -0.5;
  const double z4 = (-2.0 + 1.0 / 3) / std::sqrt(8.0 / 9);  // baseline {-1, 1, -1}
  EXPECT_FALSE(d.add(-2.0));
  const double s4 = s3 + z4 + 0.5;  // -1.77: not yet below -2
  EXPECT_NEAR(d.evidence(), -s4, 1e-12);
  const double z5 = (-3.0 + 0.75) / std::sqrt(19.0 / 16);  // baseline {-1, 1, -1, -2}
  ASSERT_LT(s4 + z5 + 0.5, -2.0);
  EXPECT_TRUE(d.add(-3.0));
  EXPECT_EQ(d.evidence(), 0.0);  // an alarm clears the evidence...
  EXPECT_FALSE(d.last_alarm_was_upward());
  EXPECT_TRUE(d.add(-5.0));  // ...but not the baseline: still armed, no new warm-up

  // Two-sided: the mirror-image statistic U = max(0, U + z - k) on rises.
  DriftParams two = p;
  two.two_sided = true;
  DriftDetector up(two), one_sided(p);
  for (double x : {-1.0, 1.0, 1.0, 2.0}) {
    EXPECT_FALSE(up.add(x));
    EXPECT_FALSE(one_sided.add(x));
  }
  EXPECT_NEAR(up.evidence(), 1.7677669529663689, 1e-12);  // the larger statistic: here U (= 0.5, then 1.77)
  EXPECT_TRUE(up.add(3.0));       // U: 0.5, then 1.77, then 1.77 + 2.06 - 0.5 > 2
  EXPECT_TRUE(up.last_alarm_was_upward());
  EXPECT_FALSE(one_sided.add(3.0));  // a one-sided test ignores improvements
  EXPECT_EQ(one_sided.evidence(), 0.0);

  // rebase() after a (false) alarm re-centres on the held-back samples: a
  // sample at that level adds no evidence, where the old centre would.
  DriftDetector a(p), b(p);
  for (double x : {-1.0, 1.0, -1.0, -2.0, -3.0}) {
    a.add(x);
    b.add(x);
  }
  a.rebase();
  EXPECT_FALSE(a.add(-3.0));
  EXPECT_EQ(a.evidence(), 0.0);
  EXPECT_FALSE(b.add(-3.0));
  EXPECT_GT(b.evidence(), 0.5);
}

// =============================== Evaluation =================================
TEST(AgentCoverage, EvaluationStatisticsByHand) {
  const DrawEnv quiet{.scale = 0.0}, noisy{.scale = 1.0};
  Action next = 0;
  auto cycle = [&](const DrawEnv::Observation&) { return next++ % 4; };  // returns 0, 1, 2, 3
  const EvalStats s = evaluate(quiet, cycle, 4, 1);
  EXPECT_EQ(s.episodes, 4u);
  EXPECT_DOUBLE_EQ(s.mean, 1.5);
  EXPECT_DOUBLE_EQ(s.std_error, std::sqrt(5.0 / 3.0 / 4.0));  // sample variance 5/3, n = 4
  auto three = [](const DrawEnv::Observation&) { return Action{3}; };
  auto one = [](const DrawEnv::Observation&) { return Action{1}; };
  const EvalStats single = evaluate(quiet, three, 1, 1);
  EXPECT_EQ(single.mean, 3.0);
  EXPECT_EQ(single.std_error, 0.0);  // one episode: no spread
  const EvalStats none = evaluate(quiet, three, 0, 1);
  EXPECT_EQ(none.episodes, 0u);
  EXPECT_EQ(none.mean, 0.0);
  EXPECT_EQ(none.std_error, 0.0);

  // Common random numbers: on the same scenarios the draws cancel exactly...
  const EvalStats diff = compare(noisy, three, one, 50, 7);
  EXPECT_EQ(diff.mean, 2.0);
  EXPECT_EQ(diff.std_error, 0.0);
  // ...while each policy's own score carries them.
  const EvalStats a = evaluate(noisy, three, 50, 7), b = evaluate(noisy, one, 50, 7);
  EXPECT_GT(a.std_error, 1.0);
  EXPECT_NEAR(a.mean - b.mean, 2.0, 1e-9);
  EXPECT_EQ(compare(noisy, three, one, 1, 7).std_error, 0.0);
  EXPECT_EQ(compare(noisy, three, one, 0, 7).mean, 0.0);
  // episode_returns() replays the same scenarios one by one.
  const auto r = episode_returns(noisy, three, 50, 7);
  ASSERT_EQ(r.size(), 50u);
  double sum = 0;
  for (std::uint32_t i = 0; i < 50; ++i) {
    EXPECT_EQ(r[i], evaluate(noisy, three, 1, 7 + i).mean);
    sum += r[i];
  }
  EXPECT_NEAR(sum / 50, a.mean, 1e-9);
}

// =========================== Off-policy evaluation ==========================
TEST(AgentCoverage, OffPolicyEstimatesReduceToTheSampleMeanOnPolicy) {
  using Obs = ExecutionObs;
  auto target = [](const Obs& o) { return Action{o.t % 2}; };  // the logged actions below
  auto q0 = [](const Obs&, Action) { return 0.0; };
  auto v0 = [](const Obs&) { return 0.0; };
  auto q7 = [](const Obs&, Action) { return 7.0; };
  auto v7 = [](const Obs&) { return 7.0; };
  auto episode = [](std::vector<double> rewards, double propensity) {
    LoggedEpisode<Obs> ep;
    for (std::uint32_t t = 0; t < rewards.size(); ++t) ep.push_back({Obs{t, 100, 0.0}, Action{t % 2}, rewards[t], propensity});
    return ep;
  };
  const OpeEstimates empty = off_policy_estimates(std::vector<LoggedEpisode<Obs>>{}, target, q0, v0, 2);
  EXPECT_EQ(empty.pdis, 0.0);
  EXPECT_EQ(empty.wpdis, 0.0);
  EXPECT_EQ(empty.dr, 0.0);
  EXPECT_EQ(empty.min_ess, 0.0);
  EXPECT_EQ(empty.uncovered_steps, 2u);  // empty logs cover no step (fixed in the edge-case review)

  // A deterministic behaviour that is the target: every weight is 1, so all
  // three estimators are the mean logged return (3 + 7 + 5) / 3, with or
  // without a model (DR's model terms cancel on-policy). The third episode
  // ended early.
  const std::vector<LoggedEpisode<Obs>> on = {episode({1, 2}, 1.0), episode({3, 4}, 1.0), episode({5}, 1.0)};
  for (const auto& e : {off_policy_estimates(on, target, q0, v0, 2), off_policy_estimates(on, target, q7, v7, 2)}) {
    EXPECT_DOUBLE_EQ(e.pdis, 5.0);
    EXPECT_DOUBLE_EQ(e.wpdis, 5.0);
    EXPECT_DOUBLE_EQ(e.dr, 5.0);
    EXPECT_DOUBLE_EQ(e.min_ess, 3.0);
    EXPECT_EQ(e.uncovered_steps, 0u);
  }
  // Behaviour propensity 0.5 everywhere: weights 2 then 4 for every episode.
  // PDIS is inflated by them (2 * 3 + 4 * 4); the self-normalized WPDIS
  // divides them out and is the mean return (3 + 7 + 11) / 3; ESS stays 3.
  const std::vector<LoggedEpisode<Obs>> half = {episode({1, 2}, 0.5), episode({3, 4}, 0.5), episode({5, 6}, 0.5)};
  const OpeEstimates e = off_policy_estimates(half, target, q0, v0, 2);
  EXPECT_DOUBLE_EQ(e.pdis, 22.0);
  EXPECT_DOUBLE_EQ(e.wpdis, 7.0);
  EXPECT_DOUBLE_EQ(e.dr, 22.0);  // DR without a model is PDIS
  EXPECT_DOUBLE_EQ(e.min_ess, 3.0);
  // DR with Q = V = 7: per episode 2 (r0 - 7) + 1 * 7 + 4 (r1 - 7) + 2 * 7 =
  // 2 r0 + 4 r1 - 21 (V is weighted by the previous step's weight).
  EXPECT_DOUBLE_EQ(off_policy_estimates(half, target, q7, v7, 2).dr, 2 * 3.0 + 4 * 4.0 - 21.0);
}

// =============================== Market making ==============================
TEST(AgentCoverage, MarketMakingDpOneStepByHandAndMirrorSymmetry) {
  using E = MarketMakingEnv;
  MarketMakingParams p1;
  p1.horizon = 1;
  p1.fill_probs = {1.0, 0.5, 0.0};  // at distances 0.5, 1, 2
  p1.terminal_penalty = 0.5;
  const MarketMakingDP dp(p1);
  // Flat: both quotes at 0.5 fill for sure, +1, still flat.
  EXPECT_DOUBLE_EQ(dp.value(0, 0), 1.0);
  EXPECT_EQ(dp.act({0, 0}), E::action(0, 0));
  // 0.5 / 1: both fill w.p. 1/2 (+1.5, flat), else only the sure one (+0.5 - 0.5 * 1^2).
  EXPECT_DOUBLE_EQ(dp.q_value(0, 0, E::action(0, 1)), 0.75);
  // Long 3: sell at 0.5 for sure and never buy: 0.5 - 0.5 * 2^2. Short: the mirror image.
  EXPECT_DOUBLE_EQ(dp.value(0, 3), -1.5);
  EXPECT_EQ(dp.act({0, 3}), E::action(2, 0));
  EXPECT_DOUBLE_EQ(dp.value(0, -3), -1.5);
  EXPECT_EQ(dp.act({0, -3}), E::action(0, 2));
  // At the cap the bid can't fill, whatever its distance: 0.5 - 0.5 * 9^2.
  for (std::uint32_t bid = 0; bid < 3; ++bid) EXPECT_DOUBLE_EQ(dp.q_value(0, 10, E::action(bid, 0)), -40.0);
  EXPECT_EQ(dp.value(1, 0), 0.0);  // past the horizon

  // The full problem is mirror-symmetric: Q(t, q, (bid, ask)) = Q(t, -q, (ask, bid)).
  const MarketMakingParams p;
  const MarketMakingDP full(p);
  for (std::uint32_t t = 0; t < p.horizon; t += 7)
    for (std::int32_t q = -p.max_inventory; q <= p.max_inventory; ++q)
      for (Action a = 0; a < E::kNumActions; ++a) {
        const double v = full.q_value(t, q, a);
        ASSERT_NEAR(full.q_value(t, -q, E::action(a % 3, a / 3)), v, 1e-9 * (1 + std::abs(v))) << t << " " << q << " " << a;
      }
  // More volatility means more inventory risk: risk_penalty scales with v^2.
  MarketMakingEnv env(p);
  env.set_volatility(2.0);
  EXPECT_DOUBLE_EQ(env.risk_penalty(3), p.risk_aversion * 4.0 * 9.0);
  MarketMakingParams wild = p;
  wild.volatility = 2.0;
  EXPECT_LT(MarketMakingDP(wild).expected_start_value(), full.expected_start_value());
}

TEST(AgentCoverage, ModelBasedMarketMakerCountsFillsWithABetaPrior) {
  // E15: per quote distance, P(fill) = (fills + 1) / (attempts + 2), pooled
  // over bid and ask; a quote that couldn't fill (at the inventory cap) is
  // no evidence. Volatility: RMS of the observed mid moves (none on the
  // last step). It plans with MarketMakingDP on the estimates.
  using E = MarketMakingEnv;
  const MarketMakingParams p;  // inventory cap 10
  ModelBasedMarketMaker mb(p, {.explore_episodes = 2, .seed = 3});
  for (std::uint32_t k = 0; k < 3; ++k) EXPECT_DOUBLE_EQ(mb.estimated_fill(k), 0.5);  // the prior's mean
  EXPECT_EQ(mb.estimated_volatility(), 1.0);  // no move seen yet
  mb.observe({0, 0}, E::action(0, 0), true, false, 2.0, false);  // two attempts at distance 0, one fill
  EXPECT_DOUBLE_EQ(mb.estimated_fill(0), 2.0 / 4);
  mb.observe({1, 1}, E::action(1, 2), false, true, -1.0, false);
  EXPECT_DOUBLE_EQ(mb.estimated_fill(1), 1.0 / 3);
  EXPECT_DOUBLE_EQ(mb.estimated_fill(2), 2.0 / 3);
  mb.observe({2, 10}, E::action(2, 0), false, false, 0.5, false);  // long cap: only the ask counts
  EXPECT_DOUBLE_EQ(mb.estimated_fill(2), 2.0 / 3);
  EXPECT_DOUBLE_EQ(mb.estimated_fill(0), 2.0 / 5);
  mb.observe({3, -10}, E::action(0, 1), true, false, 9.0, true);  // short cap: only the bid; last step
  EXPECT_DOUBLE_EQ(mb.estimated_fill(0), 3.0 / 6);
  EXPECT_DOUBLE_EQ(mb.estimated_fill(1), 1.0 / 3);
  EXPECT_DOUBLE_EQ(mb.estimated_volatility(), std::sqrt((4.0 + 1.0 + 0.25) / 3));

  mb.end_episode();
  EXPECT_FALSE(mb.has_plan());  // still exploring
  mb.end_episode();
  ASSERT_TRUE(mb.has_plan());
  EXPECT_EQ(mb.episodes(), 2u);
  MarketMakingParams fitted = p;
  fitted.fill_probs = {3.0 / 6, 1.0 / 3, 2.0 / 3};
  fitted.volatility = std::sqrt(5.25 / 3);
  const MarketMakingDP dp(fitted);
  for (std::uint32_t t : {0u, 25u, 49u})
    for (std::int32_t q : {-10, -3, 0, 4, 10}) EXPECT_EQ(mb.act({t, q}, false), dp.act({t, q})) << t << " " << q;

  // What the maker observes from the env: no mid move on the last step.
  MarketMakingParams short_p = p;
  short_p.horizon = 3;
  MarketMakingEnv env(short_p);
  Rng rng(4);
  env.reset(rng);
  EXPECT_FALSE(env.step(E::kSymmetricAction, rng).done);
  EXPECT_NE(env.last_mid_move(), 0.0);
  env.step(E::kSymmetricAction, rng);
  EXPECT_TRUE(env.step(E::kSymmetricAction, rng).done);
  EXPECT_EQ(env.last_mid_move(), 0.0);
}

// ======================== Model-based execution agent =======================
namespace {

// A market with no price noise: every reward is exactly the model's formula,
// so least squares must recover the coefficients (up to the 1e-9 ridge).
ExecutionParams noise_free_market() {
  ExecutionParams m;
  m.noise_sd = 0.0;
  m.impact = 0.02;
  m.terminal_impact = 0.07;
  m.alpha = 0.6;
  m.permanent_impact = 0.004;
  return m;
}

// Feed `episodes` episodes of a fixed behaviour: even episodes trade slowly
// (0, 3 or 5 lots a step) and reach the deadline with inventory to dump; odd
// ones cycle through every action and sell out early. notify: call
// note_action() before each step, as train_model_based() does.
void drive(ModelBasedExecutionAgent& agent, const ExecutionParams& market, int episodes, bool notify) {
  ExecutionEnv env(market);
  Rng rng(11);
  for (std::uint32_t ep = 0; ep < static_cast<std::uint32_t>(episodes); ++ep) {
    auto o = env.reset(rng);
    for (;;) {
      const Action a = ep % 2 == 0 ? (ep / 2 + o.t) % 3 : (ep * 7 + o.t) % 5;
      if (notify) agent.note_action(o, a);
      const auto st = env.step(a, rng);
      agent.observe(o, a, st.reward, st.obs, st.done);
      o = st.obs;
      if (st.done) break;
    }
    agent.end_episode();
  }
}

}  // namespace

TEST(AgentCoverage, ModelBasedIdentifiesANoiseFreeMarketExactly) {
  const ExecutionParams m = noise_free_market();
  auto rel = [](double v) { return 1e-6 * (1 + std::abs(v)); };
  ModelBasedExecutionAgent full(m), no_holding(m, {.learn_holding_cost = false});
  drive(full, m, 40, true);
  drive(no_holding, m, 40, true);
  for (const ModelBasedExecutionAgent* a : {&full, &no_holding}) {
    const ModelEstimate e = a->current_estimate();
    EXPECT_NEAR(e.impact, m.impact, rel(m.impact));
    EXPECT_NEAR(e.alpha, m.alpha, rel(m.alpha));
    EXPECT_NEAR(e.permanent, m.permanent_impact, rel(m.permanent_impact));
    EXPECT_NEAR(e.terminal_impact, m.terminal_impact, rel(m.terminal_impact));
    EXPECT_NEAR(e.holding, 0.0, 1e-9);  // risk-neutral (and noise-free) market
  }
  EXPECT_EQ(no_holding.current_estimate().holding, 0.0);  // not estimated at all

  // PerSize in the q^3 + fixed-cost market: each traded size's cost is
  // exact; an untraded size is interpolated between its neighbours (or
  // extrapolated along the last segment); the dump is a * rem^2 + b.
  ExecutionParams hard = m;
  hard.permanent_impact = 0.0;
  hard.impact_exponent = 3.0;
  hard.impact = 0.002;
  hard.fixed_cost = 1.5;
  ModelBasedExecutionAgent per(hard, {.cost_model = CostModel::PerSize});
  drive(per, hard, 40, true);
  per.replan();
  const ExecutionEnv env(hard);
  const auto [trade, dump] = per.per_size_costs();
  std::vector<std::uint32_t> knots{0};
  for (std::uint32_t q = 1; q <= hard.lots; ++q)
    if (per.seen(q)) knots.push_back(q);
  ASSERT_GE(knots.size(), 5u);
  ASSERT_LT(knots.back(), hard.lots);  // some sizes are only reachable by extrapolation
  auto truth = [&](std::uint32_t q) { return env.trade_cost(q); };
  EXPECT_EQ(trade[0], 0.0);
  for (std::uint32_t q = 1; q <= hard.lots; ++q) {
    auto hi = std::upper_bound(knots.begin(), knots.end(), q);
    double expected;
    if (per.seen(q)) {
      expected = truth(q);
    } else if (hi != knots.end()) {
      const std::uint32_t a = *(hi - 1), b = *hi;
      expected = truth(a) + (truth(b) - truth(a)) * (q - a) / static_cast<double>(b - a);
    } else {
      const std::uint32_t b = knots.back(), a = knots[knots.size() - 2];
      expected = truth(b) + (truth(b) - truth(a)) * (q - b) / static_cast<double>(b - a);
    }
    EXPECT_NEAR(trade[q], expected, rel(expected)) << "q " << q;
  }
  for (std::uint32_t rem = 1; rem <= hard.lots; ++rem) EXPECT_NEAR(dump[rem], env.dump_cost(rem), rel(env.dump_cost(rem))) << rem;
}

TEST(AgentCoverage, ModelBasedObserveAloneSeparatesTheLastSaleFromTheDump) {
  // Regression test. note_action() was documented with "Trainers that only
  // call observe() still work: then the split comes from lots_for(action,
  // inventory)", but observe() ignored its action and took the last step's
  // sale from note_action() alone, so such a trainer booked the whole last
  // step as a dump (terminal impact 0.062 for a true 0.07; PerSize dump
  // costs ~59 for a true 0.07 at rem = 1). Both kinds of trainer must learn
  // the same, correct terminal costs.
  const ExecutionParams m = noise_free_market();
  for (CostModel cm : {CostModel::Quadratic, CostModel::PerSize}) {
    ModelBasedExecutionAgent noted(m, {.cost_model = cm}), plain(m, {.cost_model = cm});
    drive(noted, m, 40, true);
    drive(plain, m, 40, false);
    EXPECT_NEAR(plain.current_estimate().terminal_impact, m.terminal_impact, 1e-6);
    EXPECT_EQ(plain.current_estimate().terminal_impact, noted.current_estimate().terminal_impact);
    noted.replan();
    plain.replan();
    EXPECT_EQ(plain.per_size_costs().second, noted.per_size_costs().second);
  }
}

TEST(AgentCoverage, ModelBasedAgentIsWellDefinedBeforeAnyData) {
  ModelBasedExecutionAgent quad{ExecutionParams{}};
  const ModelEstimate e = quad.current_estimate();
  EXPECT_EQ(e.steps, 0u);
  for (double v : {e.impact, e.terminal_impact, e.alpha, e.rho, e.signal_sd, e.holding, e.permanent}) EXPECT_EQ(v, 0.0);
  // No plan, nothing to check.
  const ModelCheck none = quad.check_model(ExecutionEnv{ExecutionParams{}}, 10, 1);
  EXPECT_EQ(none.predicted_gain, 0.0);
  EXPECT_EQ(none.realized_gain, 0.0);
  EXPECT_EQ(none.z, 0.0);
  quad.replan();  // the tiny ridge keeps the normal equations solvable
  ASSERT_TRUE(quad.has_plan());
  EXPECT_TRUE(std::isfinite(quad.plan()->expected_start_value()));
  EXPECT_LT(quad.act({0, 100, 0.0}, false), ExecutionEnv::kNumActions);
  // PerSize with no size traded yet: every cost extrapolates from c[0] = 0.
  const ModelBasedExecutionAgent per(ExecutionParams{}, {.cost_model = CostModel::PerSize});
  const auto [trade, dump] = per.per_size_costs();
  ASSERT_EQ(trade.size(), 101u);
  for (double c : trade) EXPECT_EQ(c, 0.0);
  EXPECT_EQ(dump[0], 0.0);
  for (std::uint32_t rem = 1; rem < dump.size(); ++rem) {
    EXPECT_TRUE(std::isfinite(dump[rem]));
    EXPECT_GT(dump[rem], dump[rem - 1]);
  }
}

TEST(AgentCoverage, ModelBasedExplorationStaysInTheConfiguredRange) {
  // "Exploratory actions are drawn uniformly from [explore_lo, explore_hi]",
  // and the first explore_episodes are random even if a plan exists.
  ModelBasedExecutionAgent a(ExecutionParams{}, {.explore_episodes = 1000, .explore_lo = 1, .explore_hi = 3});
  const ExecutionObs o{3, 60, 0.05};
  std::array<int, 5> count{};
  for (int i = 0; i < 3000; ++i) ++count[a.act(o, true)];
  EXPECT_EQ(count[0] + count[4], 0);
  for (Action k = 1; k <= 3; ++k) EXPECT_NEAR(count[k], 1000, 100) << k;
  drive(a, ExecutionParams{}, 5, true);
  a.replan();
  ASSERT_TRUE(a.has_plan());
  EXPECT_EQ(a.episodes(), 5u);
  for (int i = 0; i < 200; ++i) {
    const Action x = a.act(o, true);
    ASSERT_GE(x, 1u);
    ASSERT_LE(x, 3u);
  }
  EXPECT_EQ(a.act(o, false), a.plan()->act(o));
}

// ================================= Trainer ==================================
TEST(AgentCoverage, TrainerReportsEveryEvaluationAndScoresTheDeployedPolicy) {
  const ChainEnv env({.length = 6, .max_steps = 30});
  AgentConfig<1> cfg;
  cfg.eps_decay_episodes = 100;
  cfg.bins = {10};
  for (bool keep : {true, false}) {
    ChainEnv e = env;
    Agent<ChainEnv> agent(e, cfg);
    Trainer<ChainEnv> tr(agent, e, {.eval_every = 50, .eval_episodes = 5, .keep_champion = keep, .detect_drift = false});
    EXPECT_FALSE(tr.has_champion());
    std::vector<EvalPoint> seen;
    const TrainReport rep = tr.run(200, [&](const EvalPoint& p) { seen.push_back(p); });
    EXPECT_TRUE(tr.has_champion());
    EXPECT_EQ(rep.returns.size(), 200u);
    ASSERT_EQ(rep.evals.size(), 4u);
    ASSERT_EQ(seen.size(), 4u);  // the callback sees every evaluation as it happens
    for (std::size_t i = 0; i < 4; ++i) {
      EXPECT_EQ(seen[i].episode, 50 * (i + 1));
      EXPECT_EQ(seen[i].episode, rep.evals[i].episode);
      EXPECT_EQ(seen[i].challenger, rep.evals[i].challenger);
      EXPECT_EQ(seen[i].champion, rep.evals[i].champion);
      EXPECT_EQ(seen[i].promoted, rep.evals[i].promoted);
    }
    EXPECT_TRUE(rep.evals[0].promoted);  // the first evaluation always crowns a champion
    EXPECT_EQ(rep.evals[0].champion, rep.evals[0].challenger);
    const double greedy = evaluate(e, [&](const ChainEnv::Observation& o) { return agent.act(o, false); }, 5, 9'000'000).mean;
    EXPECT_EQ(tr.score_current(), greedy);
    EXPECT_EQ(rep.champion_score, tr.deployed_score());
    if (keep) {
      EXPECT_EQ(rep.champion_score, rep.evals.back().champion);  // the champion is what is deployed
    } else {
      EXPECT_EQ(rep.champion_score, greedy);  // the learner itself is deployed
      for (std::uint32_t pos = 0; pos < 5; ++pos) EXPECT_EQ(tr.deployed_act({pos, 0}), agent.act({pos, 0}, false));
    }
  }
}

TEST(AgentCoverage, TrainerConfirmedChangeReexploresForgetsAndBoundsTheBank) {
  // A frozen champion re-scored on fixed scenarios reproduces its score
  // exactly, so in the slip-free corridor (every episode identical, standard
  // error 0) any change in its score is a confirmed change of the world.
  const ChainParams calm{.length = 6, .max_steps = 30}, slippery{.length = 6, .max_steps = 30, .slip = 0.4};
  for (bool forget : {true, false}) {
    ChainEnv env(calm);
    AgentConfig<1> cfg;
    cfg.eps_decay_episodes = 100;
    cfg.bins = {10};
    Agent<ChainEnv> agent(env, cfg);
    TrainConfig tc;
    tc.eval_every = 50;
    tc.eval_episodes = 10;
    tc.drift_after_episodes = 1'000'000;  // CUSUM checks never start: only evaluations detect
    tc.reexplore_epsilon = 0.5;
    tc.reexplore_episodes = 100;
    tc.forget_on_drift = forget;
    tc.bank_capacity = 2;
    Trainer<ChainEnv> tr(agent, env, tc);
    const TrainReport calm1 = tr.run(200);
    ASSERT_EQ(calm1.champion_score, -5.0) << "precondition: the corridor is learned";
    EXPECT_TRUE(calm1.drift_alarms.empty());
    for (const auto& p : calm1.evals) EXPECT_FALSE(p.world_changed);

    env = ChainEnv(slippery);
    const TrainReport slip = tr.run(50);  // the evaluation at episode 250 sees the change
    ASSERT_EQ(slip.drift_alarms, std::vector<std::uint64_t>{250});
    ASSERT_EQ(slip.evals.size(), 1u);
    EXPECT_TRUE(slip.evals[0].world_changed);
    EXPECT_FALSE(slip.evals[0].promoted);
    EXPECT_TRUE(slip.bank_recalls.empty());  // the bank holds only the champion that just failed
    EXPECT_EQ(tr.bank_size(), 1u);
    EXPECT_EQ(agent.epsilon(), 0.5);                   // exploring again...
    EXPECT_EQ(agent.step_size(), cfg.alpha);           // ...with a fresh step size...
    EXPECT_EQ(agent.replay_size() == 0, forget) << forget;  // ...and (if asked) no stale replay

    // Every later change banks the outgoing champion; the bank keeps the newest two.
    env = ChainEnv(calm);
    EXPECT_EQ(tr.run(50).drift_alarms.size(), 1u);
    EXPECT_EQ(tr.bank_size(), 2u);
    env = ChainEnv(slippery);
    EXPECT_EQ(tr.run(50).drift_alarms.size(), 1u);
    EXPECT_EQ(tr.bank_size(), 2u);
  }
}

TEST(AgentCoverage, TrainerTwoSidedDetectorConfirmsAnImprovedWorld) {
  // E13: when the world gets better, a one-sided CUSUM stays quiet; with
  // two_sided the upward suspicion is confirmed by the champion's re-score
  // rising. In the unchanged world every suspicion is rejected, because
  // the champion's re-score reproduces exactly.
  const ChainParams rough{.length = 6, .max_steps = 30, .slip = 0.3}, smooth{.length = 6, .max_steps = 30};
  for (bool two_sided : {true, false}) {
    ChainEnv env(rough);
    AgentConfig<1> cfg;
    cfg.eps_decay_episodes = 200;
    cfg.bins = {10};
    Agent<ChainEnv> agent(env, cfg);
    TrainConfig tc;
    tc.eval_every = 400;
    tc.eval_episodes = 50;
    tc.drift = {.lag = 10, .allowance = 0.3, .threshold = 5.0, .baseline_rate = 0.01, .warmup = 100, .two_sided = two_sided};
    tc.drift_after_episodes = 450;  // after the first champion
    Trainer<ChainEnv> tr(agent, env, tc);
    const TrainReport before = tr.run(1600);
    EXPECT_TRUE(before.drift_alarms.empty()) << "a suspicion in an unchanged world was confirmed";
    env = ChainEnv(smooth);
    const TrainReport after = tr.run(399);  // no evaluation in this window: only the CUSUM can fire
    ASSERT_TRUE(after.evals.empty());
    if (two_sided) {
      ASSERT_FALSE(after.drift_alarms.empty()) << "an improvement went unnoticed";
      EXPECT_GT(after.drift_alarms.front(), 1600u);
    } else {
      EXPECT_TRUE(after.drift_alarms.empty());
      EXPECT_EQ(tr.run(1).drift_alarms, std::vector<std::uint64_t>{2000});  // the next evaluation catches it
    }
  }
}
