// E7 H2: model plan -> distill into TileQ -> fine-tune on real rewards, guarded by the plan.
#include <cstdio>
#include <cstdlib>
#include <memory>
#include "rag/agent/model_based.hpp"
#include "rag/agent/trainer.hpp"
using namespace rag::agent;
int main(int argc, char** argv) {
  const std::uint64_t seed = argc > 1 ? std::strtoull(argv[1], nullptr, 10) : 1;
  auto twap = [](const ExecutionObs&) { return ExecutionEnv::kTwapAction; };
  for (double gamma : {0.0, 0.01}) {
    ExecutionParams p; p.permanent_impact = gamma;
    ExecutionEnv env(p); const ExecutionDP dp(p);
    const double opt = compare(env, [&](const ExecutionObs& o) { return dp.act(o); }, twap, 1500, 5'000'000).mean;
    auto share = [&](auto&& pi) { return 100 * compare(env, pi, twap, 1500, 5'000'000).mean / opt; };
    // The model lacks the permanent term (the unknown unknown).
    ModelBasedExecutionAgent mb(p, {.cost_model = CostModel::Shrunk, .learn_permanent_impact = false, .seed = seed});
    { Rng r(42 + seed); ExecutionEnv e(p); train_model_based(mb, e, 300, r); mb.replan(); }
    auto plan = [&](const ExecutionObs& o) { return mb.act(o, false); };
    const ExecutionDP& model = *mb.plan();
    std::printf("gamma %.3f seed %llu: model-based plan %.0f%%\n", gamma, (unsigned long long)seed, share(plan));
    const unsigned marks[] = {2500, 5000, 10000, 20000, 40000};
    for (int variant = 0; variant < 4; ++variant) {
      // 0 scratch; 1-3 warm (distill + fine-tune, eps 0.1) with alpha 0.1 / 0.03 / 0.01
      AgentConfig<3> cfg; cfg.seed = seed; cfg.eps_decay_episodes = 10000;
      if (variant) { cfg.eps_start = 0.1; cfg.alpha = variant == 1 ? 0.1 : variant == 2 ? 0.03 : 0.01; cfg.alpha_min = std::min(cfg.alpha_min, cfg.alpha); }
      ExecutionEnv e(p); auto agent = std::make_unique<Agent<ExecutionEnv>>(e, cfg);
      if (variant) agent->distill(ExecutionEnv(model.params()), [&](const ExecutionObs& o, Action a) { return model.q_value(o.t, o.inventory, o.signal, a); }, 20000, 777);
      Trainer<ExecutionEnv> tr(*agent, e, {.eval_every = 1250, .eval_episodes = 600, .train_seed = seed, .detect_drift = false});
      const char* names[] = {"TileQ scratch", "warm a 0.1", "warm a 0.03", "warm a 0.01"};
      std::printf("  %-14s start %5.0f%%", names[variant], share([&](const ExecutionObs& o) { return agent->act(o, false); }));
      unsigned done = 0;
      for (unsigned m : marks) { tr.run(m - done); done = m; std::printf("  %5u: %4.0f%%", m, share([&](const ExecutionObs& o) { return tr.deployed_act(o); })); }
      std::printf("\n");
    }
  }
}
