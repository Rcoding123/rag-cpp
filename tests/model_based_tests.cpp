// model_based_tests.cpp — Experiment E1 (docs/ai/EXPERIMENTS.md): a
// model-based agent that learns the market's parameters and plans with the
// DP, under correct and incorrect model assumptions.
#include <gtest/gtest.h>

#include <cmath>

#include "rag/agent/model_based.hpp"
#include "rag/agent/trainer.hpp"

using namespace rag::agent;

namespace {

auto twap = [](const ExecutionObs&) { return ExecutionEnv::kTwapAction; };

// Share of the optimal gain over TWAP, both measured on the same scenarios.
template <class Policy>
double share(const ExecutionParams& p, Policy&& pi, std::uint32_t n = 1000) {
  ExecutionEnv env(p);
  const ExecutionDP dp(p);
  const double opt = compare(env, [&](const ExecutionObs& o) { return dp.act(o); }, twap, n, 31'000'000).mean;
  return compare(env, pi, twap, n, 31'000'000).mean / opt;
}

ModelBasedExecutionAgent trained(const ExecutionParams& market, std::uint32_t episodes, CostModel cm,
                                 std::uint64_t seed = 1) {
  ModelBasedExecutionAgent a(market, {.explore_episodes = std::min(10u, episodes), .cost_model = cm, .seed = seed});
  ExecutionEnv env(market);
  Rng rng(seed * 77);
  train_model_based(a, env, episodes, rng);
  a.replan();
  return a;
}

// The "hard" market: impact grows like q^3 and every order pays a fixed cost.
ExecutionParams hard_market() {
  ExecutionParams p;
  p.impact_exponent = 3.0;
  p.impact = 0.01 * 25 / 125.0;  // same cost as the default at the TWAP rate (5 lots)
  p.fixed_cost = 1.5;
  return p;
}

}  // namespace

TEST(ExecutionCosts, GeneralizedCostsKeepTheAccountingIdentity) {
  for (const ExecutionParams& p : {ExecutionParams{}, hard_market()}) {
    ExecutionEnv probe(p);
    EXPECT_EQ(probe.trade_cost(0), 0.0);  // no order, no cost (fixed cost included)
    EXPECT_EQ(probe.dump_cost(0), 0.0);
    Rng pol(4);
    for (int ep = 0; ep < 200; ++ep) {
      ExecutionEnv env(p);
      Rng rng(ep);
      auto o = env.reset(rng);
      double total = 0;
      for (;;) {
        const auto st = env.step(pol.below(5), rng);
        total += st.reward;
        o = st.obs;
        if (st.done) break;
      }
      EXPECT_NEAR(total, env.cash() - p.lots * p.start_price, 1e-9);
    }
  }
  ExecutionEnv hard(hard_market());
  EXPECT_NEAR(hard.trade_cost(10), 0.002 * 1000 + 1.5, 1e-12);
}

TEST(ExecutionCosts, DpWithCostTablesEqualsDpWithFormulas) {
  const ExecutionParams p = hard_market();
  const ExecutionEnv env(p);
  std::vector<double> trade(p.lots + 1), dump(p.lots + 1);
  for (std::uint32_t q = 0; q <= p.lots; ++q) {
    trade[q] = env.trade_cost(q);
    dump[q] = env.dump_cost(q);
  }
  const ExecutionDP a(p, 61), b(p, trade, dump, 61);
  for (std::uint32_t t : {0u, 9u, 19u})
    for (std::uint32_t inv : {7u, 50u, 100u}) EXPECT_EQ(a.value(t, inv, 0.03), b.value(t, inv, 0.03));
  EXPECT_THROW(ExecutionDP(p, std::vector<double>(3), dump), std::invalid_argument);
}

TEST(ModelBased, RecoversTheMarketParameters) {
  const ExecutionParams p;
  const auto a = trained(p, 300, CostModel::Quadratic);
  const auto m = a.model();
  EXPECT_NEAR(m.impact, p.impact, 0.03 * p.impact);
  EXPECT_NEAR(m.terminal_impact, p.terminal_impact, 0.05 * p.terminal_impact);
  EXPECT_NEAR(m.alpha, p.alpha, 0.1);
  EXPECT_NEAR(m.rho, p.signal_rho, 0.06);
  EXPECT_NEAR(m.signal_sd, p.signal_sd, 0.012);
}

TEST(ModelBased, NearOptimalAfterFiftyEpisodes) {
  // Measured: 97-100% after 20-50 episodes across seeds; the model-free
  // TileQ learner needs 40,000 episodes for 62-71%.
  for (std::uint64_t seed : {1u, 2u, 3u}) {
    auto a = trained(ExecutionParams{}, 50, CostModel::Quadratic, seed);
    EXPECT_GT(share(ExecutionParams{}, [&](const ExecutionObs& o) { return a.act(o, false); }), 0.95) << "seed " << seed;
  }
}

TEST(ModelBased, HiddenMarketParametersDoNotLeakIntoThePlan) {
  // The agent receives `structure` only for lots, horizon and action sizes.
  // Regression test for a bug: replan() once kept the structure's
  // impact_exponent and fixed_cost, planning with true information mixed
  // with a coefficient fitted under the quadratic assumption.
  const ExecutionParams market = hard_market();
  ExecutionParams garbage = market;
  garbage.impact = 123;
  garbage.terminal_impact = 9;
  garbage.alpha = -7;
  garbage.signal_rho = 0.1;
  garbage.signal_sd = 3;
  garbage.impact_exponent = 1.1;
  garbage.fixed_cost = 99;
  ExecutionParams clean = market;
  clean.impact_exponent = 2.0;
  clean.fixed_cost = 0.0;
  for (CostModel cm : {CostModel::Quadratic, CostModel::Shrunk}) {
    ModelBasedExecutionAgent a(garbage, {.cost_model = cm}), b(clean, {.cost_model = cm});
    ExecutionEnv env(market);  // identical experience for both
    Rng rng(5);
    for (int ep = 0; ep < 40; ++ep) {
      auto o = env.reset(rng);
      for (;;) {
        const Action act = static_cast<Action>((ep * 7 + o.t) % 5);  // a fixed behaviour policy
        a.note_action(o, act);
        b.note_action(o, act);
        const auto st = env.step(act, rng);
        a.observe(o, act, st.reward, st.obs, st.done);
        b.observe(o, act, st.reward, st.obs, st.done);
        o = st.obs;
        if (st.done) break;
      }
      a.end_episode();
      b.end_episode();
    }
    a.replan();
    b.replan();
    Rng probe(9);
    for (int i = 0; i < 300; ++i) {
      const ExecutionObs o{probe.below(20), probe.below(101), 0.3 * (probe.uniform() - 0.5)};
      ASSERT_EQ(a.act(o, false), b.act(o, false));
    }
  }
}

TEST(ModelBased, WrongModelFormIsCostlyAndAFlexibleModelFixesIt) {
  // Hard market (impact ~ q^3 plus a fixed cost per order), 300 episodes.
  // Measured: quadratic 55%, per-size 100%, shrunk 99%; model-free TileQ 63%
  // after 40,000 episodes.
  const ExecutionParams p = hard_market();
  auto quad = trained(p, 300, CostModel::Quadratic);
  auto per = trained(p, 300, CostModel::PerSize);
  auto shr = trained(p, 300, CostModel::Shrunk);
  const double q = share(p, [&](const ExecutionObs& o) { return quad.act(o, false); });
  const double s1 = share(p, [&](const ExecutionObs& o) { return per.act(o, false); });
  const double s2 = share(p, [&](const ExecutionObs& o) { return shr.act(o, false); });
  EXPECT_LT(q, 0.75) << "quadratic " << q;
  EXPECT_GT(s1, 0.95) << "per-size " << s1;
  EXPECT_GT(s2, 0.93) << "shrunk " << s2;
}

TEST(ModelBased, FlexibleModelsNeedDataAndShrinkageMakesThemSafe) {
  // Measured at 10 episodes in the hard market, seeds 1-10: per-size is
  // bimodal, 97-99% or a catastrophe (-1440% to -1809%, 4 of 10 seeds), as
  // the planner exploits wild estimates for rarely traded sizes. Shrunk:
  // 82-89% on every seed.
  const ExecutionParams p = hard_market();
  int per_failures = 0;
  double shrunk_worst = 1e9;
  for (std::uint64_t seed = 1; seed <= 10; ++seed) {
    auto per = trained(p, 10, CostModel::PerSize, seed);
    auto shr = trained(p, 10, CostModel::Shrunk, seed);
    per_failures += share(p, [&](const ExecutionObs& o) { return per.act(o, false); }) < 0.0;
    shrunk_worst = std::min(shrunk_worst, share(p, [&](const ExecutionObs& o) { return shr.act(o, false); }));
  }
  EXPECT_GE(per_failures, 2) << "per-size was expected to fail on some seeds with this little data";
  EXPECT_GT(shrunk_worst, 0.75) << "shrinkage should never be catastrophic";
}
