// agent_demo — a self-learning execution agent, end to end.
//
//   1. the task and its baselines (TWAP, a hand-written rule, the DP optimum)
//   2. learning from scratch, measured against the optimum on unseen scenarios
//   3. what the agent learned (its policy next to the optimal one)
//   4. two kinds of learner: parametric vs episodic memory
//   5. the world changes: drift detection, re-exploration, recovery
//   6. learning a model instead, and checking whether the model is right
#include <cstdio>
#include <string>

#include "rag/agent/execution_dp.hpp"
#include "rag/agent/model_based.hpp"
#include "rag/agent/trainer.hpp"

using namespace rag::agent;

namespace {

constexpr std::uint64_t kHeldOut = 50'000'000;  // scenarios never used for training
constexpr std::uint32_t kEvalN = 2000;

auto twap = [](const ExecutionObs&) { return ExecutionEnv::kTwapAction; };

template <class Policy>
EvalStats gain_vs_twap(const ExecutionEnv& env, Policy&& p) {
  return compare(env, p, twap, kEvalN, kHeldOut);
}

void rule(const char* title) { std::printf("\n== %s %.*s\n", title, static_cast<int>(70 - std::string(title).size()), "======================================================================"); }

}  // namespace

int main() {
  ExecutionParams p;
  ExecutionEnv env(p);
  const ExecutionDP dp(p);
  auto optimal = [&](const ExecutionObs& o) { return dp.act(o); };
  auto sign_rule = [](const ExecutionObs& o) { return o.signal < 0 ? Action{3} : Action{1}; };

  rule("1. The task");
  std::printf("Sell %u lots in %u steps. Fast selling costs impact (%.2f x lots^2 per step);\n"
              "slow selling risks the price. A noisy signal predicts the next move, but the\n"
              "agent is not told that: it sees (time, inventory, signal) and a reward.\n\n",
              p.lots, p.horizon, p.impact);
  const auto g_opt = gain_vs_twap(env, optimal);
  const auto g_rule = gain_vs_twap(env, sign_rule);
  std::printf("  %-34s %+7.2f\n", "TWAP (sell evenly)", 0.0);
  std::printf("  %-34s %+7.2f +- %.2f   (%4.1f%% of optimal)\n", "rule: sell faster if signal < 0", g_rule.mean, g_rule.std_error,
              100 * g_rule.mean / g_opt.mean);
  std::printf("  %-34s %+7.2f +- %.2f   (knows the model)\n", "DP optimum", g_opt.mean, g_opt.std_error);
  std::printf("  (gain over TWAP per episode on %u held-out scenarios, in price units)\n", kEvalN);

  rule("2. Learning from scratch");
  AgentConfig<3> cfg;
  cfg.eps_decay_episodes = 20000;
  Agent<ExecutionEnv> agent(env, cfg);
  Trainer<ExecutionEnv> trainer(agent, env, {.eval_every = 5000, .eval_episodes = 1000});
  std::printf("  %8s  %6s  %6s  %22s  %s\n", "episode", "eps", "alpha", "gain vs TWAP", "champion");
  trainer.run(40000, [&](const EvalPoint& e) {
    const auto g = gain_vs_twap(env, [&](const ExecutionObs& o) { return agent.act(o, false); });  // the learner
    std::printf("  %8llu  %6.3f  %6.4f  %+7.2f (%5.1f%% of opt)  %s\n", static_cast<unsigned long long>(e.episode), e.epsilon,
                agent.step_size(), g.mean, 100 * g.mean / g_opt.mean, e.promoted ? "promoted" : "kept");
  });
  auto deployed = [&](const ExecutionObs& o) { return trainer.deployed_act(o); };
  const auto g_agent = gain_vs_twap(env, deployed);
  std::printf("  deployed (champion): %+.2f +- %.2f = %.1f%% of the optimal gain\n", g_agent.mean, g_agent.std_error,
              100 * g_agent.mean / g_opt.mean);

  rule("3. What it learned");
  std::printf("Lots sold at step 10 with 50 lots left, by signal (in standard deviations):\n\n  %-8s", "signal");
  const double sigs[] = {-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2};
  for (double s : sigs) std::printf("%6.1f", s);
  for (int who = 0; who < 2; ++who) {
    std::printf("\n  %-8s", who ? "optimal" : "agent");
    for (double s : sigs) {
      const ExecutionObs o{10, 50, s * p.signal_sd};
      std::printf("%6u", env.lots_for(who ? dp.act(o) : trainer.deployed_act(o), 50));
    }
  }
  std::printf("\n\nThe agent discovered on its own that a negative signal means \"sell now\".\n");

  rule("4. Parametric vs episodic learning");
  std::printf("Share of the optimal gain after N training episodes:\n\n  %-10s %10s %10s\n", "episodes", "TileQ", "Episodic");
  for (std::uint32_t n : {500u, 1000u, 2000u}) {
    double share[2];
    for (int k = 0; k < 2; ++k) {
      AgentConfig<3> c;
      c.learner = k ? LearnerKind::Episodic : LearnerKind::TileQ;
      c.eps_decay_episodes = n / 2;
      ExecutionEnv e(p);
      Agent<ExecutionEnv> a(e, c);
      Trainer<ExecutionEnv> t(a, e, {.eval_every = 0, .detect_drift = false});
      t.run(n);
      share[k] = 100 * compare(e, [&](const ExecutionObs& o) { return a.act(o, false); }, twap, 500, kHeldOut).mean / g_opt.mean;
    }
    std::printf("  %-10u %9.1f%% %9.1f%%\n", n, share[0], share[1]);
  }
  std::printf("\nEpisodic memory reuses single good episodes immediately; the parametric\n"
              "learner generalizes better and wins with more data (65-72%% at 40k+).\n");

  rule("5. The world changes");
  ExecutionParams flipped = p;
  flipped.alpha = -p.alpha;
  const ExecutionDP dp2(flipped);
  ExecutionEnv env_off = env;
  Agent<ExecutionEnv> stubborn = agent;  // same knowledge, no drift detection
  env.set_alpha(flipped.alpha);
  env_off.set_alpha(flipped.alpha);
  // Same supervision (evaluation, champion), drift detection off.
  Trainer<ExecutionEnv> trainer_off(stubborn, env_off, {.eval_every = 5000, .eval_episodes = 1000, .detect_drift = false});
  const double opt2 = compare(env, [&](const ExecutionObs& o) { return dp2.act(o); }, twap, 1000, kHeldOut).mean;
  auto share2 = [&](Trainer<ExecutionEnv>& t) {
    return 100 * compare(env, [&](const ExecutionObs& o) { return t.deployed_act(o); }, twap, 1000, kHeldOut).mean / opt2;
  };
  std::printf("The signal's meaning flips: it now predicts the opposite move.\n");
  std::printf("Right after the flip the agent captures %.0f%% of the new optimum (negative: harmful).\n\n", share2(trainer));
  std::printf("  %10s %20s %20s\n", "episodes", "with drift detection", "without");
  for (std::uint32_t done = 0; done < 40000;) {
    const auto rep = trainer.run(5000);
    trainer_off.run(5000);
    done += 5000;
    std::string note;
    for (auto at : rep.drift_alarms) note += "  <- drift detected " + std::to_string(at - 40000) + " episodes after the flip";
    std::printf("  %10u %19.1f%% %19.1f%%%s\n", done, share2(trainer), share2(trainer_off), note.c_str());
  }
  std::printf("\nOn an alarm the trainer raises exploration and step size and clears the\n"
              "replay buffer; the champion is re-scored on the new world and replaced.\n");

  rule("6. Learning a model instead");
  std::printf("A model-based agent assumes the *form* of the market and learns only its\n"
              "numbers, by regression on what it observes; then it plans with the DP.\n\n");
  {
    ModelBasedExecutionAgent mb(p, {.cost_model = CostModel::Shrunk});
    ExecutionEnv e(p);
    Rng rng(7);
    train_model_based(mb, e, 50, rng);
    mb.replan();
    const ModelEstimate m = mb.model();
    std::printf("  after 50 episodes:  impact %.4f (true %.4f)  alpha %.3f (%.3f)  rho %.3f (%.3f)\n", m.impact, p.impact, m.alpha,
                p.alpha, m.rho, p.signal_rho);
    const auto g = gain_vs_twap(ExecutionEnv(p), [&](const ExecutionObs& o) { return mb.act(o, false); });
    std::printf("  its plan: %.1f%% of the optimal gain (the model-free agent needed 40,000 episodes for %.0f%%)\n",
                100 * g.mean / g_opt.mean, 100 * g_agent.mean / g_opt.mean);
  }
  ExecutionParams perm = p;
  perm.permanent_impact = 0.01;
  const ExecutionEnv pe(perm);
  const ExecutionDP pdp(perm);
  const double opt3 = compare(pe, [&](const ExecutionObs& o) { return pdp.act(o); }, twap, 1000, kHeldOut).mean;
  std::printf("\nNow a market with permanent impact: each lot sold lowers the price of\n"
              "everything still held. 300 episodes each:\n\n");
  for (bool term : {false, true}) {
    ModelBasedExecutionAgent mb(perm, {.cost_model = CostModel::Shrunk, .learn_permanent_impact = term});
    ExecutionEnv e(perm);
    Rng rng(7);
    train_model_based(mb, e, 300, rng);
    mb.replan();
    const ModelCheck check = mb.check_model(pe, 100, kHeldOut + 1'000'000);
    const double share = 100 * compare(pe, [&](const ExecutionObs& o) { return mb.act(o, false); }, twap, 1000, kHeldOut).mean / opt3;
    std::printf("  %-34s %5.0f%% of optimal   model check: z = %+5.1f  %s\n",
                term ? "model with a permanent-impact term" : "model without it", share, check.z,
                check.z > 4 ? "-> the model is wrong" : "(consistent)");
  }
  std::printf("\nThe check compares the gain over TWAP the model predicts for its own plan\n"
              "with the gain measured on real scenarios, paired. A wrong model is\n"
              "optimistic: its planner picks the actions its errors flatter. When the\n"
              "missing term isn't known, distill the plan into a model-free learner and\n"
              "fine-tune it under a guard (docs/ai/EXPERIMENTS.md, E7).\n");
}
