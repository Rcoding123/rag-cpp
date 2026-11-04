// model_based_tests.cpp — Experiment E1 (docs/ai/EXPERIMENTS.md): a
// model-based agent that learns the market's parameters and plans with the
// DP, under correct and incorrect model assumptions.
#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <vector>

#include "rag/agent/model_based.hpp"
#include "rag/agent/ope.hpp"
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
  // with a coefficient fitted under the quadratic assumption. Later the same
  // bug class nearly recurred with permanent_impact (E7), so every market
  // field the planner reads is scrambled here.
  const ExecutionParams market = hard_market();
  ExecutionParams garbage = market;
  garbage.impact = 123;
  garbage.terminal_impact = 9;
  garbage.alpha = -7;
  garbage.signal_rho = 0.1;
  garbage.signal_sd = 3;
  garbage.impact_exponent = 1.1;
  garbage.fixed_cost = 99;
  garbage.permanent_impact = 0.5;
  garbage.risk_aversion = 3;
  garbage.noise_sd = 2;
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

// ================= Experiment E4 (risk-averse execution) =====================
TEST(ModelBased, LearnsTheRiskPenalty) {
  // Measured: fitted holding cost 0.00049 for a true 0.0005 (lambda 0.2), and
  // 100% of the risk-adjusted optimal gain after 300 episodes.
  ExecutionParams p;
  p.risk_aversion = 0.2;
  auto a = trained(p, 300, CostModel::Quadratic);
  const double truth = p.risk_aversion * p.noise_sd * p.noise_sd;
  EXPECT_NEAR(a.model().holding, truth, 0.2 * truth);
  EXPECT_GT(share(p, [&](const ExecutionObs& o) { return a.act(o, false); }), 0.97);
  // In a risk-neutral market the fitted holding cost is ~0, and the plan is unaffected.
  // (Tolerance: learning the permanent-impact term too, E7, widens this seed's
  // error from 2% to 15%: front-loading plans make q*I' and I'^2 correlated.
  // The share is unaffected: at most 0.3 points across seeds.)
  auto n = trained(ExecutionParams{}, 300, CostModel::Quadratic);
  EXPECT_NEAR(n.model().holding, 0.0, 2e-5);
}

TEST(AgentRisk, FrontLoadsMoreWhenRiskAverse) {
  // Lots sold in the first 5 steps, averaged over scenarios. Measured for the
  // DP optimum: 26.5 at lambda 0, 72.9 at lambda 1.
  auto early = [](const ExecutionParams& p, auto&& policy) {
    double sold = 0;
    for (int i = 0; i < 400; ++i) {
      ExecutionEnv env(p);
      Rng rng(800 + i);
      auto o = env.reset(rng);
      while (o.t < 5) {
        const auto inv = o.inventory;
        const auto st = env.step(policy(o), rng);
        sold += inv - st.obs.inventory;
        o = st.obs;
        if (st.done) break;
      }
    }
    return sold / 400;
  };
  ExecutionParams neutral, averse;
  averse.risk_aversion = 1.0;
  const ExecutionDP dpn(neutral), dpa(averse);
  const double dp_n = early(neutral, [&](const ExecutionObs& o) { return dpn.act(o); });
  const double dp_a = early(averse, [&](const ExecutionObs& o) { return dpa.act(o); });
  EXPECT_GT(dp_a, dp_n + 30) << "DP: neutral " << dp_n << ", averse " << dp_a;
  // A model-free learner, told nothing but the (risk-adjusted) rewards, follows.
  auto learn = [&](const ExecutionParams& p) {
    ExecutionEnv env(p);
    AgentConfig<3> cfg;
    cfg.learner = LearnerKind::ActorCritic;
    auto agent = std::make_unique<Agent<ExecutionEnv>>(env, cfg);
    Trainer<ExecutionEnv> tr(*agent, env, {.eval_every = 0, .detect_drift = false});
    tr.run(8000);
    return early(p, [&](const ExecutionObs& o) { return agent->act(o, false); });
  };
  const double rl_n = learn(neutral), rl_a = learn(averse);
  EXPECT_GT(rl_a, rl_n + 20) << "learned: neutral " << rl_n << ", averse " << rl_a;
}

// ============ Experiment E7 (outside the model class: permanent impact) ======
TEST(ExecutionCosts, PermanentImpactKeepsTheAccountingIdentityAndTheDpIsExact) {
  ExecutionParams p;
  p.permanent_impact = 0.01;
  p.risk_aversion = 0.2;
  {  // Rewards (without the risk penalty) add up to the cash shortfall.
    ExecutionParams q = p;
    q.risk_aversion = 0;
    ExecutionEnv env(q);
    Rng rng(3);
    for (int e = 0; e < 100; ++e) {
      env.reset(rng);
      double sum = 0;
      for (;;) {
        const auto st = env.step(rng.below(5), rng);
        sum += st.reward;
        if (st.done) break;
      }
      ASSERT_NEAR(sum, env.cash() - q.lots * q.start_price, 1e-9);
    }
  }
  const ExecutionDP dp(p);
  ExecutionEnv env(p);
  // Exact policy evaluation of the optimal policy reproduces its value...
  EXPECT_NEAR(dp.policy_value([&](const ExecutionObs& o) { return dp.act(o); }), dp.expected_start_value(), 1e-9);
  // ...and of TWAP, a closed form: costs, permanent impact and risk penalty of the fixed schedule.
  double twap_value = 0;
  std::uint32_t inv = p.lots;
  for (std::uint32_t t = 0; t < p.horizon; ++t) {
    const std::uint32_t q = env.lots_for(ExecutionEnv::kTwapAction, inv);
    inv -= q;
    twap_value -= env.trade_cost(q) + p.permanent_impact * q * inv;
    if (t + 1 < p.horizon) twap_value -= env.holding_penalty(inv);
  }
  twap_value -= env.dump_cost(inv);
  EXPECT_NEAR(dp.policy_value(twap), twap_value, 1e-9);
  // Simulation agrees with both (paired, so tight).
  const auto gain = compare(env, [&](const ExecutionObs& o) { return dp.act(o); }, twap, 4000, 1);
  EXPECT_NEAR(gain.mean, dp.expected_start_value() - twap_value, 4 * gain.std_error);
}

namespace {
ModelBasedExecutionAgent fitted(const ExecutionParams& market, std::uint32_t episodes, bool permanent, std::uint64_t seed = 1) {
  ModelBasedExecutionAgent a(market, {.cost_model = CostModel::Shrunk, .learn_permanent_impact = permanent, .seed = seed});
  ExecutionEnv env(market);
  Rng rng(42 + seed);
  train_model_based(a, env, episodes, rng);
  a.replan();
  return a;
}
}  // namespace

TEST(ModelBased, LearnsPermanentImpactAndAWrongModelGetsWorseWithData) {
  // Measured (5 seeds): without the term, 43% at 20 episodes, 37% at 50, 25%
  // at 300 (the plan exploits the model's error, so its data grows more
  // one-sided). With it: 93% at 20, 99.8% at 50, 99.9% at 300.
  ExecutionParams p;
  p.permanent_impact = 0.01;
  auto pol = [](ModelBasedExecutionAgent& a) { return [&a](const ExecutionObs& o) { return a.act(o, false); }; };
  auto with = fitted(p, 300, true);
  EXPECT_NEAR(with.model().permanent, 0.01, 0.0005);
  EXPECT_GT(share(p, pol(with)), 0.97);
  auto few = fitted(p, 20, false), many = fitted(p, 300, false);
  const double s_few = share(p, pol(few)), s_many = share(p, pol(many));
  EXPECT_LT(s_many, 0.5);
  EXPECT_LT(s_many, s_few - 0.05) << "more data should make the misspecified model worse: " << s_few << " -> " << s_many;
  // Where there is no permanent impact, learning the term costs nothing.
  auto plain = fitted(ExecutionParams{}, 300, true);
  EXPECT_NEAR(plain.model().permanent, 0.0, 0.0005);
  EXPECT_GT(share(ExecutionParams{}, pol(plain)), 0.99);
}

TEST(ModelBased, CheckModelFlagsAMissingTerm) {
  // Paired check of predicted vs realized gain over TWAP, 100 real episodes.
  // Measured over 8 seeds: z in [-2.5, 2.2] for a correct model; 8.5-12.9
  // with the permanent term missing (gamma 0.005), 20-27 at gamma 0.01.
  ExecutionParams p;
  p.permanent_impact = 0.005;
  const ExecutionEnv market(p);
  for (std::uint64_t seed = 1; seed <= 3; ++seed) {
    const auto right = fitted(p, 300, true, seed).check_model(market, 100, 6'000'000 + 1000 * seed);
    const auto wrong = fitted(p, 300, false, seed).check_model(market, 100, 6'000'000 + 1000 * seed);
    EXPECT_LT(right.z, 4.0) << "seed " << seed;
    EXPECT_GT(wrong.z, 4.0) << "seed " << seed;
    EXPECT_GT(wrong.predicted_gain, wrong.realized_gain);  // a wrong model is optimistic about its plan
  }
}

// ======== Experiment E9 (a changing market: forgetting vs detection) =========
namespace {
// Run n episodes of the model-based agent on env, learning online.
void online(ModelBasedExecutionAgent& a, ExecutionEnv& env, Rng& rng, int n) {
  for (int i = 0; i < n; ++i) {
    auto o = env.reset(rng);
    for (;;) {
      const Action act = a.act(o, true);
      a.note_action(o, act);
      const auto st = env.step(act, rng);
      a.observe(o, act, st.reward, st.obs, st.done);
      o = st.obs;
      if (st.done) break;
    }
    a.end_episode();
  }
}
}  // namespace

TEST(ModelBased, DetectsAMarketFlipAndRecoversWithinThirtyEpisodes) {
  // Measured (replanning every 5 episodes, seeds 1-3): alarms 7-13 episodes
  // after the signal's meaning flips, 95-100% of the new optimum at +20.
  // Without forgetting the estimates average both regimes: -166% at +20,
  // still -123% at +100. Here replanning is every 10 episodes on a coarser
  // grid (test speed).
  ExecutionParams flipped;
  flipped.alpha = -1.0;
  for (bool detect : {true, false}) {
    ModelBasedExecutionAgent a(ExecutionParams{}, {.replan_every = 10,
                                                   .replan_max_gap = 10,
                                                   .dp_grid = 31,  // coarser planner: 4x faster replans for the test
                                                   .cost_model = CostModel::Shrunk,
                                                   .detect_drift = detect,
                                                   .seed = 1});
    ExecutionEnv env{ExecutionParams{}};
    Rng rng(101);
    online(a, env, rng, 300);
    EXPECT_TRUE(a.drift_alarms().empty()) << "false alarm in a stable market";
    env.set_alpha(-1.0);
    online(a, env, rng, 30);
    const double s = share(flipped, [&](const ExecutionObs& o) { return a.act(o, false); }, 600);
    if (detect) {
      ASSERT_EQ(a.drift_alarms().size(), 1u);
      EXPECT_GT(a.drift_alarms()[0], 300u);
      EXPECT_LE(a.drift_alarms()[0], 320u);
      EXPECT_GT(s, 0.9);
    } else {
      EXPECT_LT(s, 0.0) << "without forgetting the old regime dominates the estimates";
    }
  }
}

TEST(ModelBased, ChangeDetectionIsFreeInAStableMarketButConstantForgettingIsNot) {
  // Measured over 3,000 episodes, noise 0.2: detection gave 0-1 false
  // alarms and 99.9-100.3% of optimal (seeds 1-3); constant forgetting 0.95
  // (half-life ~14 episodes) gave 66-98% (seeds 1-6, mean ~89%), because it
  // plans on ~20 episodes' worth of data. The mechanism, checked cheaply
  // (no replanning): with forgetting, the estimates stay as noisy as after
  // ~20 episodes, however long the agent runs.
  ExecutionParams noisy;
  noisy.noise_sd = 0.2;
  ModelBasedExecutionAgent detect(noisy, {.cost_model = CostModel::Shrunk, .detect_drift = true, .seed = 1});
  ExecutionEnv e1(noisy);
  Rng r1(301);
  online(detect, e1, r1, 1000);
  EXPECT_LE(detect.drift_alarms().size(), 1u);
  EXPECT_GT(share(noisy, [&](const ExecutionObs& o) { return detect.act(o, false); }), 0.97);
  double err_forget = 0, err_full = 0;
  for (std::uint64_t seed = 1; seed <= 4; ++seed) {
    for (double f : {0.95, 1.0}) {
      ModelBasedExecutionAgent a(noisy, {.replan_every = 100000, .forget = f, .seed = seed});
      ExecutionEnv e(noisy);
      Rng r(400 + seed);
      online(a, e, r, 300);
      const double d = a.current_estimate().alpha - noisy.alpha;
      (f < 1 ? err_forget : err_full) += d * d;
    }
  }
  EXPECT_GT(err_forget, 4 * err_full) << err_forget << " vs " << err_full;
}

// ============== Experiment E10 (off-policy evaluation) ======================
namespace {
struct OpeRun {
  double dm, wpdis, dr;  // estimates of the target's value
  OpeEstimates is;
};
// Log n episodes of TWAP with 20% uniformly random actions, fit a model on
// them (passively), and estimate `target`'s value from the logs alone.
template <class Target>
OpeRun estimate_offline(const ExecutionParams& market, bool permanent_term, unsigned n, std::uint64_t seed, Target&& target) {
  constexpr double eps = 0.2;
  ExecutionEnv env(market);
  Rng rng(1000 + seed), pick(5 + seed);
  ModelBasedExecutionAgent mb(market, {.replan_every = 1'000'000, .cost_model = CostModel::Shrunk,
                                       .learn_permanent_impact = permanent_term, .seed = 1});
  std::vector<LoggedEpisode<ExecutionObs>> logs(n);
  for (auto& ep : logs) {
    auto o = env.reset(rng);
    for (;;) {
      const Action a = pick.uniform() < eps ? static_cast<Action>(pick.below(5)) : ExecutionEnv::kTwapAction;
      mb.note_action(o, a);
      const auto st = env.step(a, rng);
      mb.observe(o, a, st.reward, st.obs, st.done);
      ep.push_back({o, a, st.reward, (a == ExecutionEnv::kTwapAction ? 1 - eps : 0.0) + eps / 5});
      o = st.obs;
      if (st.done) break;
    }
    mb.end_episode();
  }
  mb.replan();
  const auto values = mb.plan()->evaluate_policy(target);
  const auto is = off_policy_estimates(
      logs, target, [&](const ExecutionObs& o, Action a) { return values.q(o.t, o.inventory, o.signal, a); },
      [&](const ExecutionObs& o) { return values.value(o.t, o.inventory, o.signal); }, market.horizon);
  return {values.start_value(), is.wpdis, is.dr, is};
}
}  // namespace

TEST(OffPolicy, ImportanceSamplingCollapsesAtLongHorizonsAndTheModelDecides) {
  // Measured (20 replications, errors in % of the optimal gain): for the DP
  // optimum from 1,000 logged episodes, DM rmse 4% with the right model,
  // WPDIS 98%, DR 134%; ~17 of 20 steps had no logged episode following the
  // target. With the permanent term missing (gamma 0.01) DM's bias was -62%
  // at 1,000 episodes and still -62% at 10,000: more data doesn't fix a
  // wrong model.
  ExecutionParams p;
  const ExecutionDP truth(p);
  auto optimal = [&](const ExecutionObs& o) { return truth.act(o); };
  const double gain = truth.expected_start_value() - truth.policy_value(twap);
  const double v_opt = truth.policy_value(optimal);
  for (std::uint64_t seed = 1; seed <= 2; ++seed) {
    const auto r = estimate_offline(p, true, 1000, seed, optimal);
    EXPECT_NEAR(r.dm, v_opt, 0.08 * gain) << "seed " << seed;
    EXPECT_GE(r.is.uncovered_steps, 10u);  // nobody followed the target for long
    EXPECT_LT(r.is.min_ess, 1.0);
  }
  ExecutionParams perm;
  perm.permanent_impact = 0.01;
  const ExecutionDP truth_perm(perm);
  auto optimal_perm = [&](const ExecutionObs& o) { return truth_perm.act(o); };
  const double gain_perm = truth_perm.expected_start_value() - truth_perm.policy_value(twap);
  const double v_perm = truth_perm.policy_value(optimal_perm);
  for (unsigned n : {1000u, 10000u}) {
    const auto r = estimate_offline(perm, false, n, 1, optimal_perm);
    EXPECT_LT((r.dm - v_perm) / gain_perm, -0.4) << n << " episodes";
  }
}

// ============ Experiment E11 (cheaper system identification) ================
TEST(ModelBased, FirstPlanComesRightAfterExploration) {
  // Regression test: the first plan used to wait for replan_every (20)
  // episodes as well, so explore_episodes below 20 had no effect and the
  // agent acted randomly for 20 episodes instead of 10.
  for (std::uint32_t explore : {0u, 3u, 10u}) {
    ModelBasedExecutionAgent a(ExecutionParams{}, {.explore_episodes = explore});
    ExecutionEnv env{ExecutionParams{}};
    Rng rng(1);
    for (std::uint32_t ep = 1; ep <= 12; ++ep) {
      train_model_based(a, env, 1, rng);
      EXPECT_EQ(a.has_plan(), ep >= std::max(1u, explore)) << "explore " << explore << ", after episode " << ep;
    }
    EXPECT_EQ(a.replans(), 1u);
  }
}

// =================== Experiment E14 (tail risk) =============================
TEST(TailRisk, CvarArithmetic) {
  const std::vector<double> r = {5, 1, 3, 2, 4};
  EXPECT_DOUBLE_EQ(cvar(r, 0.4), 1.5);   // worst 2: 1 and 2
  EXPECT_DOUBLE_EQ(cvar(r, 0.01), 1.0);  // at least one outcome
  EXPECT_DOUBLE_EQ(cvar(r, 1.0), 3.0);   // everything: the mean
  EXPECT_EQ(cvar({}, 0.05), 0.0);
}

TEST(TailRisk, VarianceIsNotTailRiskInExecution) {
  // Measured on cash outcomes (20,000 scenarios): the risk-neutral optimum has
  // a higher sd than TWAP (57.4 vs 54.5) but a far better worst 5% (-66 vs
  // -117). A strong mean-variance penalty (lambda 1) halves the sd (27.5) yet
  // improves the worst 5% only to -61, while the mean falls from +10 to -8.
  const ExecutionParams p;
  const ExecutionEnv cash(p);
  auto stats = [&](auto&& pi) {
    const auto r = episode_returns(cash, pi, 4000, 7'000'000);
    const double n = static_cast<double>(r.size());
    double m = 0, v = 0;
    for (double x : r) m += x / n;
    for (double x : r) v += (x - m) * (x - m) / (n - 1);
    return std::array<double, 3>{m, std::sqrt(v), cvar(r, 0.05)};
  };
  ExecutionParams averse = p;
  averse.risk_aversion = 1.0;
  const ExecutionDP neutral_dp(p), averse_dp(averse);
  const auto tw = stats(twap);
  const auto neutral = stats([&](const ExecutionObs& o) { return neutral_dp.act(o); });
  const auto strong = stats([&](const ExecutionObs& o) { return averse_dp.act(o); });
  EXPECT_GT(neutral[1], tw[1]);           // more volatile than TWAP...
  EXPECT_GT(neutral[2], tw[2] + 30.0);    // ...with a much better tail
  EXPECT_LT(strong[1], 0.6 * neutral[1]); // the penalty cuts the sd...
  EXPECT_LT(strong[2] - neutral[2], 10.0);  // ...but barely moves the tail
  EXPECT_LT(strong[0], neutral[0] - 10.0);  // and costs a lot of mean
}
