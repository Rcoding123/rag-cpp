// agent_edge_case_tests.cpp — edge cases of the agent stack (adversarial
// review): last steps, inventory caps, schedules, zero standard errors,
// early-ended episodes. Each test is small, exact where possible, and fast.
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <utility>
#include <vector>

#include "rag/agent/drift.hpp"
#include "rag/agent/evaluate.hpp"
#include "rag/agent/market_making.hpp"
#include "rag/agent/model_based.hpp"
#include "rag/agent/ope.hpp"
#include "rag/agent/trainer.hpp"

using namespace rag::agent;

namespace {

// Feed identical experience (a fixed random behaviour, not the agents' own
// actions) to several model-based agents; `with_note` says which of them
// also get note_action() before each step.
void feed_execution(std::vector<ModelBasedExecutionAgent*> agents, const std::vector<bool>& with_note, int episodes) {
  const ExecutionParams p;
  ExecutionEnv env(p);
  Rng rng(11), pick(12);
  for (int ep = 0; ep < episodes; ++ep) {
    auto o = env.reset(rng);
    for (;;) {
      // Sell slowly (actions 0-2) so inventory is left at the last step, where
      // any action may be taken: the last step then has both a sale and a dump.
      const Action a = static_cast<Action>(pick.below(o.t + 1 < p.horizon ? 3u : 5u));
      for (std::size_t i = 0; i < agents.size(); ++i)
        if (with_note[i]) agents[i]->note_action(o, a);
      const auto st = env.step(a, rng);
      for (auto* ag : agents) ag->observe(o, a, st.reward, st.obs, st.done);
      o = st.obs;
      if (st.done) break;
    }
    for (auto* ag : agents) ag->end_episode();
  }
}

MarketMakingDP plan_from_estimates(const MarketMakingParams& objective, const ModelBasedMarketMaker& mb) {
  MarketMakingParams est = objective;
  for (std::uint32_t k = 0; k < 3; ++k) est.fill_probs[k] = mb.estimated_fill(k);
  est.volatility = mb.estimated_volatility();
  return MarketMakingDP(est);
}

}  // namespace

// ============================ model_based.hpp ================================
TEST(ModelBasedEdgeCases, ObserveAloneSeparatesTheLastSaleFromTheDump) {
  // observe() is documented to work for trainers that never call
  // note_action(): the last step's sale is lots_for(action, inventory), and
  // observe() is given the action. Bug: it read only the value note_action()
  // stored, so without it the whole last-step inventory counted as dumped
  // (sale 0) and terminal_impact was fitted on the wrong remainder.
  const ExecutionParams p;
  ModelBasedExecutionAgent noted(p, {.dp_grid = 31, .cost_model = CostModel::Shrunk});
  ModelBasedExecutionAgent plain(p, {.dp_grid = 31, .cost_model = CostModel::Shrunk});
  feed_execution({&noted, &plain}, {true, false}, 40);
  const ModelEstimate a = noted.current_estimate(), b = plain.current_estimate();
  EXPECT_NEAR(a.terminal_impact, p.terminal_impact, 0.1 * p.terminal_impact);
  EXPECT_EQ(b.terminal_impact, a.terminal_impact);
  noted.replan();
  plain.replan();
  EXPECT_EQ(plain.per_size_costs().second, noted.per_size_costs().second);  // the PerSize dump fit reads the same split
}

TEST(ModelBasedEdgeCases, TheSecondPlanComesReplanEveryEpisodesAfterTheFirst) {
  // Documented schedule (TASK_SPECS §5, Config): 10 exploration episodes,
  // then re-plan after 20 episodes, the gap doubling up to 500. Bug: since
  // the first plan moved to the end of exploration (E11), the gap doubled
  // right after it, so the second plan came 40 episodes later (plans at 10,
  // 50, 130 instead of 10, 30, 70).
  ModelBasedExecutionAgent a(ExecutionParams{}, {.explore_episodes = 10, .replan_every = 20, .dp_grid = 31});
  ExecutionEnv env{ExecutionParams{}};
  Rng rng(1);
  std::vector<std::uint32_t> planned_after;
  for (std::uint32_t ep = 1; ep <= 70; ++ep) {
    const auto before = a.replans();
    train_model_based(a, env, 1, rng);
    if (a.replans() != before) planned_after.push_back(ep);
  }
  EXPECT_EQ(planned_after, (std::vector<std::uint32_t>{10, 30, 70}));
}

TEST(ModelBasedEdgeCases, GreedyActingBeforeTheFirstPlanLeavesTrainingUnchanged) {
  // CONVENTIONS: greedy evaluation must not consume the agent's RNG. Bug:
  // without a plan, act(o, false) drew a random action from the agent's own
  // stream, so scoring the agent during its exploration episodes changed
  // what it explored afterwards, and so its model.
  ModelBasedExecutionAgent probed(ExecutionParams{}, {.explore_episodes = 3, .dp_grid = 31});
  ModelBasedExecutionAgent untouched(ExecutionParams{}, {.explore_episodes = 3, .dp_grid = 31});
  const ExecutionObs o{0, 100, 0.05};
  const Action first = probed.act(o, false);
  int same = 0;
  for (int i = 0; i < 50; ++i) same += probed.act(o, false) == first;
  EXPECT_EQ(same, 50) << "greedy acting must be deterministic";
  ExecutionEnv e1{ExecutionParams{}}, e2{ExecutionParams{}};
  Rng r1(7), r2(7);
  train_model_based(probed, e1, 5, r1);
  train_model_based(untouched, e2, 5, r2);
  EXPECT_EQ(probed.current_estimate().impact, untouched.current_estimate().impact);
  EXPECT_EQ(probed.current_estimate().alpha, untouched.current_estimate().alpha);
}

// =========================== market_making.hpp ===============================
TEST(MarketMakingEdgeCases, DpAtTheInventoryCapMatchesABruteForceExpectation) {
  // Pins the cap accounting of MarketMakingDP::q_value against TASK_SPECS §2,
  // re-derived independently: with max_inventory 1 a tight bid / wide ask
  // reaches the cap at once, where the bid can't fill (and from -Q the ask
  // can't). Exact enumeration of the four fill outcomes per step.
  MarketMakingParams p;
  p.horizon = 4;
  p.max_inventory = 1;
  const MarketMakingDP dp(p);
  const Action skew = MarketMakingEnv::action(0, 2);  // bid at 0.5, ask at 2
  const double fb = p.fill_rate * std::exp(-p.decay * p.distances[0]), fa = p.fill_rate * std::exp(-p.decay * p.distances[2]);
  const double pen = p.risk_aversion * p.volatility * p.volatility;
  auto brute = [&](auto&& self, std::uint32_t t, std::int32_t q) -> double {
    if (t == p.horizon) return 0.0;
    const double pb = q == p.max_inventory ? 0.0 : fb, pa = q == -p.max_inventory ? 0.0 : fa;
    double v = 0;
    for (int b = 0; b < 2; ++b)
      for (int s = 0; s < 2; ++s) {
        const double pr = (b ? pb : 1 - pb) * (s ? pa : 1 - pa);
        if (pr == 0) continue;
        const std::int32_t q2 = q + b - s;
        const double spread = b * p.distances[0] + s * p.distances[2];
        const double after = t + 1 == p.horizon ? -p.terminal_penalty * q2 * q2 : -pen * q2 * q2 + self(self, t + 1, q2);
        v += pr * (spread + after);
      }
    return v;
  };
  EXPECT_NEAR(dp.policy_value([&](const MarketMakingObs&) { return skew; }), brute(brute, 0, 0), 1e-12);
  for (std::int32_t q = -1; q <= 1; ++q) EXPECT_NEAR(dp.q_value(p.horizon - 1, q, skew), brute(brute, p.horizon - 1, q), 1e-12) << q;
}

TEST(MarketMakingEdgeCases, TheModelBasedMarketMakerReplansReplanEveryEpisodesAfterItsFirstPlan) {
  // Same schedule bug as the execution agent: documented "three random
  // episodes, then re-plans after 5 episodes, doubling" (E15, Config), but
  // the second plan came 10 episodes after the first. After episode 8 the
  // plan must be the DP of the estimates as of episode 8.
  const MarketMakingParams p;
  ModelBasedMarketMaker mb(p, {.explore_episodes = 3, .replan_every = 5, .seed = 1});
  MarketMakingEnv env(p);
  Rng rng(1);
  train_model_based(mb, env, 8, rng);
  const MarketMakingDP now = plan_from_estimates(p, mb);
  int differ = 0;
  for (std::uint32_t t = 0; t < p.horizon; ++t)
    for (std::int32_t q = -p.max_inventory; q <= p.max_inventory; ++q) differ += mb.act({t, q}, false) != now.act({t, q});
  EXPECT_EQ(differ, 0);
}

TEST(MarketMakingEdgeCases, GreedyActingBeforeTheFirstPlanLeavesTrainingUnchanged) {
  const MarketMakingParams p;
  ModelBasedMarketMaker probed(p, {.seed = 3}), untouched(p, {.seed = 3});
  const Action first = probed.act({0, 0}, false);
  int same = 0;
  for (int i = 0; i < 50; ++i) same += probed.act({0, 0}, false) == first;
  EXPECT_EQ(same, 50) << "greedy acting must be deterministic";
  MarketMakingEnv e1(p), e2(p);
  Rng r1(9), r2(9);
  train_model_based(probed, e1, 4, r1);
  train_model_based(untouched, e2, 4, r2);
  for (std::uint32_t k = 0; k < 3; ++k) EXPECT_EQ(probed.estimated_fill(k), untouched.estimated_fill(k)) << k;
}

// ============================== evaluate.hpp =================================
TEST(EvaluateEdgeCases, CvarIncludesExactlyTheWorstAlphaFraction) {
  // 0.29 * 100 is 28.999999999999996 in floating point; truncating it
  // dropped one of the 29 worst outcomes (CVaR 14.5 instead of 15).
  std::vector<double> r;
  for (int i = 100; i >= 1; --i) r.push_back(i);
  EXPECT_DOUBLE_EQ(cvar(r, 0.29), 15.0);  // mean of 1..29
  EXPECT_DOUBLE_EQ(cvar(r, 0.57), 29.0);  // mean of 1..57 (0.57 * 100 = 56.99999999999999)
  EXPECT_DOUBLE_EQ(cvar(r, 0.05), 3.0);   // mean of 1..5
  EXPECT_DOUBLE_EQ(cvar(r, 0.015), 1.0);  // 1.5 outcomes: the worst one only
}

// ================================ drift.hpp ==================================
TEST(DriftEdgeCases, EvidenceIncludesTheUpwardTest) {
  // evidence() is "how close to an alarm (alarm at threshold)". With
  // two_sided, a rise accumulates in the upward statistic, which evidence()
  // ignored: it read 0 right up to an upward alarm.
  const DriftParams params{.lag = 2, .allowance = 0.3, .threshold = 5.0, .baseline_rate = 0.001, .warmup = 20, .two_sided = true};
  DriftDetector d(params);
  for (int i = 0; i < 40; ++i) ASSERT_FALSE(d.add(i % 2 ? -1.0 : 1.0));  // baseline: mean 0, sd 1
  ASSERT_FALSE(d.add(2.0));                                               // z = 2: upward evidence 1.7
  EXPECT_NEAR(d.evidence(), 1.7, 0.1);
  int alarm_at = -1;
  for (int i = 0; i < 10 && alarm_at < 0; ++i)
    if (d.add(2.0)) alarm_at = i;
  EXPECT_GE(alarm_at, 0);
  EXPECT_TRUE(d.last_alarm_was_upward());
  EXPECT_EQ(d.evidence(), 0.0);  // an alarm clears the evidence
}

// ================================= ope.hpp ===================================
TEST(OffPolicyEdgeCases, AnEpisodeThatEndedEarlyKeepsItsWeightWithRewardZero) {
  // Pins ope.hpp's contract for short episodes ("ended early with reward 0
  // afterwards"): an episode that followed the target and then ended still
  // counts in later steps' normalizers. Target: action 0 everywhere.
  // Episode 1: (r 1, b 0.5), (r 2, b 0.5) -> weights 2, 4.
  // Episode 2: (r 3, b 0.5), then over -> weight 2, carried to step 1.
  std::vector<LoggedEpisode<ExecutionObs>> logs = {
      {{ExecutionObs{0, 100, 0}, 0, 1.0, 0.5}, {ExecutionObs{1, 90, 0}, 0, 2.0, 0.5}},
      {{ExecutionObs{0, 100, 0}, 0, 3.0, 0.5}},
  };
  const auto e = off_policy_estimates(
      logs, [](const ExecutionObs&) { return Action{0}; }, [](const ExecutionObs&, Action) { return 0.0; },
      [](const ExecutionObs&) { return 0.0; }, 2);
  EXPECT_DOUBLE_EQ(e.pdis, (2 * 1 + 4 * 2 + 2 * 3) / 2.0);       // 8
  EXPECT_DOUBLE_EQ(e.wpdis, (2 * 1 + 2 * 3) / 4.0 + 4 * 2 / 6.0);  // 2 + 4/3, not 2 + 2
  EXPECT_DOUBLE_EQ(e.dr, e.pdis);                                 // a zero model: DR is PDIS
  EXPECT_DOUBLE_EQ(e.min_ess, 36.0 / 20.0);                       // step 1: (4 + 2)^2 / (16 + 4)
  EXPECT_EQ(e.uncovered_steps, 0u);
}

TEST(OffPolicyEdgeCases, EmptyLogsCoverNoStep) {
  // With no logged episode nothing followed the target anywhere. Bug: the
  // early return reported uncovered_steps = 0, i.e. full coverage.
  const std::vector<LoggedEpisode<ExecutionObs>> none;
  const auto e = off_policy_estimates(
      none, [](const ExecutionObs&) { return Action{0}; }, [](const ExecutionObs&, Action) { return 0.0; },
      [](const ExecutionObs&) { return 0.0; }, 20);
  EXPECT_EQ(e.uncovered_steps, 20u);
  EXPECT_EQ(e.min_ess, 0.0);
}

// ============================= execution_dp.hpp ==============================
TEST(ExecutionDpEdgeCases, AMarketWithoutASignalHasExactFiniteValues) {
  // signal_sd = 0: the signal grid had zero width, every lookup was 0/0 and
  // the whole value table NaN (act() then always sold nothing: -363 per
  // episode against TWAP's -4.9). Without a signal the task is
  // deterministic, so a DP over (t, inventory) alone is the exact reference.
  ExecutionParams p;
  p.signal_sd = 0.0;
  p.risk_aversion = 0.1;  // exercise the holding penalty too
  const ExecutionEnv env(p);
  const ExecutionDP dp(p, 61);
  auto solve = [&](auto&& choose) {
    std::vector<double> next(p.lots + 1, 0.0), cur(p.lots + 1, 0.0);
    for (std::uint32_t t = p.horizon; t-- > 0;) {
      const bool last = t + 1 == p.horizon;
      for (std::uint32_t inv = 1; inv <= p.lots; ++inv) {
        double best = -1e300;
        for (Action a = 0; a < ExecutionEnv::kNumActions; ++a) {
          if (!choose(a)) continue;
          const std::uint32_t q = env.lots_for(a, inv), rem = inv - q;
          double r = -env.trade_cost(q);
          r += last ? -env.dump_cost(rem) : -env.holding_penalty(rem) + next[rem];
          best = std::max(best, r);
        }
        cur[inv] = best;
      }
      std::swap(next, cur);
    }
    return next[p.lots];
  };
  const double optimum = solve([](Action) { return true; });
  const double twap_value = solve([](Action a) { return a == ExecutionEnv::kTwapAction; });
  EXPECT_NEAR(dp.expected_start_value(), optimum, 1e-9 * std::abs(optimum));
  EXPECT_NEAR(dp.policy_value([](const ExecutionObs&) { return ExecutionEnv::kTwapAction; }), twap_value, 1e-9 * std::abs(twap_value));
  EXPECT_GT(optimum, twap_value);
}

TEST(ExecutionDpEdgeCases, ASignalGridNeedsTwoPoints) {
  // Interpolation needs two grid points. With grid = 1 the step divided by
  // zero and `m - 2` wrapped around, so the expectation band was written
  // past the end of a one-element row (heap overflow). Rejected like a
  // wrong-sized cost table.
  const ExecutionParams p;
  EXPECT_THROW(ExecutionDP(p, 1u), std::invalid_argument);
  EXPECT_THROW(ExecutionDP(p, 0u), std::invalid_argument);
  EXPECT_NO_THROW(ExecutionDP(p, 2u));
}

// =============================== trainer.hpp =================================
TEST(TrainerEdgeCases, ASingleEvaluationScenarioHasZeroStdErrorButNoFalseAlarms) {
  // eval_episodes = 1 makes every standard error 0, so the champion
  // re-score check fires on any change at all. Pins that it still never
  // fires in an unchanged world (the re-score is bit-for-bit reproducible),
  // and that a changed one is caught at the next evaluation.
  MarketMakingParams p;
  p.horizon = 10;
  p.max_inventory = 3;
  MarketMakingEnv env(p);
  AgentConfig<2> cfg;
  cfg.learner = LearnerKind::ActorCritic;
  Agent<MarketMakingEnv> agent(env, cfg);
  Trainer<MarketMakingEnv> tr(agent, env, {.eval_every = 25, .eval_episodes = 1});
  const auto stable = tr.run(300);
  EXPECT_TRUE(stable.drift_alarms.empty());
  ASSERT_EQ(stable.evals.size(), 12u);
  for (const auto& ev : stable.evals) EXPECT_FALSE(ev.world_changed) << ev.episode;
  env.set_decay(0.2);  // a deeper book: the one scenario's fills change
  const auto changed = tr.run(25);
  ASSERT_EQ(changed.drift_alarms.size(), 1u);
  EXPECT_EQ(changed.drift_alarms[0], 325u);
}
