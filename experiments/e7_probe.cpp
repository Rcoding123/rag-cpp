// E7: permanent impact, outside the model-based agent's model class.
#include <cstdio>
#include <memory>
#include "rag/agent/model_based.hpp"
#include "rag/agent/trainer.hpp"
using namespace rag::agent;
int main(int argc, char** argv) {
  const bool skip_mf = argc > 1;
  auto twap = [](const ExecutionObs&) { return ExecutionEnv::kTwapAction; };
  // Accounting identity with permanent impact.
  {
    ExecutionParams p; p.permanent_impact = 0.01; ExecutionEnv env(p); Rng rng(3);
    double worst = 0;
    for (int e = 0; e < 200; ++e) {
      auto o = env.reset(rng); double sum = 0;
      for (;;) { auto s = env.step(rng.below(5), rng); sum += s.reward; if (s.done) break; }
      worst = std::max(worst, std::abs(sum - (env.cash() - p.lots * p.start_price)));
    }
    std::printf("accounting identity, worst error %.2e\n", worst);
  }
  std::printf("%-6s %8s %8s %8s | %7s %7s %7s | %7s %7s | model gap: predicted realized (z)\n", "gamma", "DPgain", "DPvalue", "TWAPval", "MB-quad", "MB-shr", "MB-per", "TileQ", "AC");
  for (double gamma : {0.0, 0.005, 0.01, 0.02}) {
    ExecutionParams p; p.permanent_impact = gamma;
    ExecutionEnv env(p); const ExecutionDP dp(p);
    const double opt = compare(env, [&](const ExecutionObs& o) { return dp.act(o); }, twap, 1500, 5'000'000).mean;
    auto share = [&](auto&& pi) { return 100 * compare(env, pi, twap, 1500, 5'000'000).mean / opt; };
    const double tw = evaluate(env, twap, 1500, 5'000'000).mean;
    double mb[3]; double pred = 0, real = 0, se = 0;
    for (int k = 0; k < 3; ++k) {
      const CostModel cm = k == 0 ? CostModel::Quadratic : k == 1 ? CostModel::Shrunk : CostModel::PerSize;
      ModelBasedExecutionAgent a(p, {.cost_model = cm, .seed = 1}); Rng r(42); ExecutionEnv e(p);
      train_model_based(a, e, 300, r); a.replan();
      auto pol = [&](const ExecutionObs& o) { return a.act(o, false); };
      mb[k] = share(pol);
      if (k == 1) {
        pred = a.plan()->expected_start_value();
        const auto ev = evaluate(env, pol, 1500, 6'000'000);
        real = ev.mean; se = ev.std_error;
        const auto m = a.model();
        std::printf("  [shrunk fit: impact %.4f term %.4f alpha %.3f holding %.5f]\n", m.impact, m.terminal_impact, m.alpha, m.holding);
      }
    }
    double mf[2] = {0, 0};
    if (!skip_mf)
      for (int k = 0; k < 2; ++k) {
        AgentConfig<3> cfg; cfg.learner = k ? LearnerKind::ActorCritic : LearnerKind::TileQ; cfg.eps_decay_episodes = 20000;
        ExecutionEnv e(p); auto agent = std::make_unique<Agent<ExecutionEnv>>(e, cfg);
        Trainer<ExecutionEnv> tr(*agent, e, {.eval_every = 5000, .eval_episodes = 500, .detect_drift = false});
        tr.run(40000);
        mf[k] = share([&](const ExecutionObs& o) { return tr.deployed_act(o); });
      }
    std::printf("%-6.3f %+8.2f %+8.2f %+8.2f | %6.0f%% %6.0f%% %6.0f%% | %6.0f%% %6.0f%% | %+.2f %+.2f (%.1f)\n", gamma, opt,
                dp.expected_start_value(), tw, mb[0], mb[1], mb[2], mf[0], mf[1], pred, real, (pred - real) / se);
  }
}
