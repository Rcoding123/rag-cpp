// market_making_tests.cpp — Experiment E12: a second task (market making)
// for the same, unchanged agent, trainer and safeguards.
#include <gtest/gtest.h>

#include <cmath>

#include "rag/agent/market_making.hpp"
#include "rag/agent/trainer.hpp"

using namespace rag::agent;

namespace {
auto symmetric = [](const MarketMakingObs&) { return MarketMakingEnv::kSymmetricAction; };

// Share of the optimal gain over symmetric quoting, paired.
template <class Policy>
double mm_share(const MarketMakingEnv& env, const MarketMakingDP& dp, Policy&& pi, std::uint32_t n = 1000) {
  const double opt = compare(env, [&](const MarketMakingObs& o) { return dp.act(o); }, symmetric, n, 5'000'000).mean;
  return compare(env, pi, symmetric, n, 5'000'000).mean / opt;
}
}  // namespace

TEST(MarketMaking, AccountingIdentityAndInventoryCap) {
  MarketMakingParams p;
  p.risk_aversion = 0;  // the penalty is part of the reward only
  MarketMakingEnv env(p);
  Rng rng(3);
  for (int e = 0; e < 200; ++e) {
    env.reset(rng);
    double sum = 0;
    for (;;) {
      const auto st = env.step(rng.below(9), rng);
      sum += st.reward;
      ASSERT_LE(std::abs(st.obs.q), p.max_inventory);
      if (st.done) break;
    }
    ASSERT_NEAR(sum, env.cash(), 1e-9);  // flat at the end: total reward == final cash
    ASSERT_EQ(env.inventory(), 0);
  }
  // At the cap the side that would breach it never fills.
  EXPECT_EQ(env.bid_fill(0, p.max_inventory), 0.0);
  EXPECT_EQ(env.ask_fill(0, -p.max_inventory), 0.0);
  EXPECT_GT(env.bid_fill(0, 0), env.bid_fill(MarketMakingEnv::action(2, 0), 0));  // wider quotes fill less
}

TEST(MarketMaking, DpMatchesSimulationAndSkewsAgainstInventory) {
  // Measured: DP 27.38 vs simulated 27.43 +- 0.05; symmetric quoting -3.27
  // (inventory risk outweighs the spread).
  const MarketMakingParams p;
  const MarketMakingEnv env(p);
  const MarketMakingDP dp(p);
  auto optimal = [&](const MarketMakingObs& o) { return dp.act(o); };
  const auto sim = evaluate(env, optimal, 20000, 1);
  EXPECT_NEAR(sim.mean, dp.expected_start_value(), 4 * sim.std_error);
  EXPECT_NEAR(dp.policy_value(optimal), dp.expected_start_value(), 1e-9);
  EXPECT_LT(dp.policy_value(symmetric), 0.0);
  EXPECT_GT(dp.expected_start_value(), 20.0);
  // The skew: long -> wide bid, tight ask; short -> the mirror image; flat -> symmetric.
  for (std::int32_t q : {3, 6}) {
    const Action longq = dp.act({25, q}), shortq = dp.act({25, -q});
    EXPECT_GT(env.bid_distance(longq), env.ask_distance(longq)) << q;
    EXPECT_LT(env.bid_distance(shortq), env.ask_distance(shortq)) << q;
  }
  EXPECT_EQ(dp.act({25, 0}), MarketMakingEnv::kSymmetricAction);
}

TEST(MarketMaking, TheSameAgentDiscoversTheSkew) {
  // Measured (seed 1): actor-critic 94% of the optimal gain after 1,000
  // episodes, 100% after 5,000; TileQ 85% after 5,000, 97% after 50,000. Both
  // reproduce the DP's quotes at mid-horizon. Nothing in the agent, the
  // trainer or the guard is specific to the task.
  const MarketMakingParams p;
  const MarketMakingEnv env(p);
  const MarketMakingDP dp(p);
  AgentConfig<2> cfg;
  cfg.learner = LearnerKind::ActorCritic;
  cfg.eps_decay_episodes = 5000;
  MarketMakingEnv train_env(p);
  Agent<MarketMakingEnv> agent(train_env, cfg);
  Trainer<MarketMakingEnv> tr(agent, train_env, {.eval_every = 1000, .eval_episodes = 500, .detect_drift = false});
  tr.set_baseline(symmetric);
  tr.run(5000);
  EXPECT_TRUE(tr.learned_policy_deployed());
  EXPECT_GT(mm_share(env, dp, [&](const MarketMakingObs& o) { return tr.deployed_act(o); }), 0.95);
  for (std::int32_t q : {4, 6}) {
    const Action longq = tr.deployed_act({25, q}), shortq = tr.deployed_act({25, -q});
    EXPECT_GT(env.bid_distance(longq), env.ask_distance(longq)) << q;
    EXPECT_LT(env.bid_distance(shortq), env.ask_distance(shortq)) << q;
  }
}
