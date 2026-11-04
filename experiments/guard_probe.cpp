// E6b: deployed share with and without the baseline guard, several markets and seeds.
#include <cstdio>
#include <string>
#include "rag/agent/trainer.hpp"
#include "rag/agent/execution_dp.hpp"
using namespace rag::agent;
int main() {
  auto twap = [](const ExecutionObs&) { return ExecutionEnv::kTwapAction; };
  struct Case { const char* name; ExecutionParams p; };
  ExecutionParams base, weak = base, flat = base;
  weak.alpha = 0.25; flat.signal_rho = 0.0;
  for (const Case& c : {Case{"default", base}, Case{"alpha 0.25", weak}, Case{"rho 0", flat}}) {
    ExecutionEnv env(c.p); const ExecutionDP dp(c.p);
    const double opt = compare(env, [&](const ExecutionObs& o) { return dp.act(o); }, twap, 1500, 5'000'000).mean;
    for (std::uint64_t seed : {1, 2, 3, 4, 5}) {
      double share[2]; bool dep = false; double se = 0;
      for (int guard = 0; guard < 2; ++guard) {
        AgentConfig<3> cfg; cfg.eps_decay_episodes = 10000; cfg.seed = seed;
        ExecutionEnv e(c.p); auto agent = std::make_unique<Agent<ExecutionEnv>>(e, cfg);
        Trainer<ExecutionEnv> tr(*agent, e, {.eval_every = 2500, .eval_episodes = 600, .train_seed = seed, .detect_drift = false});
        if (guard) tr.set_baseline(twap);
        tr.run(20000);
        const auto d = compare(env, [&](const ExecutionObs& o) { return tr.deployed_act(o); }, twap, 1500, 5'000'000);
        share[guard] = 100 * d.mean / opt;
        if (guard) { dep = tr.learned_policy_deployed(); se = 100 * d.std_error / opt; }
      }
      std::printf("%-11s seed %llu  opt %+6.2f  unguarded %6.0f%%  guarded %6.0f%% (+-%.0f)  learned deployed: %s\n", c.name,
                  (unsigned long long)seed, opt, share[0], share[1], se, dep ? "yes" : "no");
    }
  }
}
