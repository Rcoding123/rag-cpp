// E16 control: does coarser tile coding (more averaging per weight) close the gap to the network?
#include <cstdio>
#include <memory>
#include "rag/agent/execution_dp.hpp"
#include "rag/agent/trainer.hpp"
using namespace rag::agent;
int main() {
  ExecutionParams p; ExecutionEnv env(p); const ExecutionDP dp(p);
  auto twap = [](const ExecutionObs&) { return ExecutionEnv::kTwapAction; };
  const double opt = compare(env, [&](const ExecutionObs& o) { return dp.act(o); }, twap, 1000, 77'000'000).mean;
  struct C { std::array<std::uint32_t, 3> bins; std::uint32_t tilings; };
  for (C c : {C{{8, 8, 8}, 8}, C{{4, 4, 4}, 8}, C{{3, 3, 3}, 8}, C{{4, 4, 4}, 32}, C{{2, 4, 4}, 16}})
    for (std::uint64_t seed = 1; seed <= 3; ++seed) {
      AgentConfig<3> cfg; cfg.eps_decay_episodes = 10000; cfg.seed = seed; cfg.bins = c.bins; cfg.tilings = c.tilings;
      ExecutionEnv e(p); auto agent = std::make_unique<Agent<ExecutionEnv>>(e, cfg);
      Trainer<ExecutionEnv> tr(*agent, e, {.eval_every = 0, .train_seed = seed, .detect_drift = false});
      std::printf("tiles %ux%ux%u x %2u, seed %llu:", c.bins[0], c.bins[1], c.bins[2], c.tilings, (unsigned long long)seed);
      unsigned done = 0;
      for (unsigned m : {5000u, 20000u, 40000u}) { tr.run(m - done); done = m;
        std::printf("  %5u: %5.1f%%", m, 100 * compare(env, [&](const ExecutionObs& o) { return agent->act(o, false); }, twap, 1000, 77'000'000).mean / opt); }
      std::printf("\n"); std::fflush(stdout);
    }
}
