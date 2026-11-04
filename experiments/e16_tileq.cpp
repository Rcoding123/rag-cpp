// E16 control: TileQ with more replayed updates per step (compute closer to DQN's 32 per step).
#include <cstdio>
#include <cstdlib>
#include <memory>
#include "rag/agent/execution_dp.hpp"
#include "rag/agent/trainer.hpp"
using namespace rag::agent;
int main(int argc, char** argv) {
  const std::uint64_t seed = argc > 1 ? std::strtoull(argv[1], nullptr, 10) : 1;
  const unsigned rps = argc > 2 ? std::atoi(argv[2]) : 4;
  ExecutionParams p; ExecutionEnv env(p); const ExecutionDP dp(p);
  auto twap = [](const ExecutionObs&) { return ExecutionEnv::kTwapAction; };
  const double opt = compare(env, [&](const ExecutionObs& o) { return dp.act(o); }, twap, 1000, 77'000'000).mean;
  AgentConfig<3> cfg; cfg.eps_decay_episodes = 10000; cfg.seed = seed; cfg.replay_per_step = rps;
  ExecutionEnv e(p); auto agent = std::make_unique<Agent<ExecutionEnv>>(e, cfg);
  Trainer<ExecutionEnv> tr(*agent, e, {.eval_every = 0, .train_seed = seed, .detect_drift = false});
  std::printf("TileQ seed %llu replay/step %u:", (unsigned long long)seed, rps);
  unsigned done = 0;
  for (unsigned m : {3000u, 5000u, 10000u, 20000u, 40000u}) { tr.run(m - done); done = m;
    std::printf("  %5u: %5.1f%%", m, 100 * compare(env, [&](const ExecutionObs& o) { return agent->act(o, false); }, twap, 1000, 77'000'000).mean / opt); std::fflush(stdout); }
  std::printf("\n");
}
