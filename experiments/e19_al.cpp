// E19: advantage learning for TileQ (default tiles and the E18-tuned coarse tiles).
#include <cstdio>
#include <cstdlib>
#include <memory>
#include "rag/agent/execution_dp.hpp"
#include "rag/agent/trainer.hpp"
using namespace rag::agent;
int main(int argc, char** argv) {
  const bool coarse = argc > 1;
  ExecutionParams p; const ExecutionEnv env(p); const ExecutionDP dp(p);
  auto twap = [](const ExecutionObs&) { return ExecutionEnv::kTwapAction; };
  const double opt = compare(env, [&](const ExecutionObs& o) { return dp.act(o); }, twap, 1000, 77'000'000).mean;
  for (double k : {0.0, 0.5, 0.9})
    for (std::uint64_t seed = 1; seed <= 3; ++seed) {
      AgentConfig<3> cfg; cfg.eps_decay_episodes = 10000; cfg.seed = seed; cfg.advantage_learning = k;
      if (coarse) { cfg.bins = {2, 4, 4}; cfg.tilings = 16; }
      ExecutionEnv e(p); auto agent = std::make_unique<Agent<ExecutionEnv>>(e, cfg);
      Trainer<ExecutionEnv> tr(*agent, e, {.eval_every = 0, .train_seed = seed, .detect_drift = false});
      std::printf("%s AL k=%.1f seed %llu:", coarse ? "coarse 2x4x4x16" : "default tiles  ", k, (unsigned long long)seed);
      unsigned done = 0;
      for (unsigned m : {5000u, 20000u, 40000u}) { tr.run(m - done); done = m;
        std::printf("  %5u: %5.1f%%", m, 100 * compare(env, [&](const ExecutionObs& o) { return agent->act(o, false); }, twap, 1000, 77'000'000).mean / opt); }
      std::printf("\n"); std::fflush(stdout);
    }
}
