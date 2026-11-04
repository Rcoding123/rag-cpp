// E16: the library's LearnerKind::NeuralQ through the standard Trainer.
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include "rag/agent/execution_dp.hpp"
#include "rag/agent/trainer.hpp"
using namespace rag::agent;
int main(int argc, char** argv) {
  const std::uint64_t seed = argc > 1 ? std::strtoull(argv[1], nullptr, 10) : 1;
  const unsigned total = argc > 2 ? std::atoi(argv[2]) : 20000;
  ExecutionParams p; ExecutionEnv env(p); const ExecutionDP dp(p);
  auto twap = [](const ExecutionObs&) { return ExecutionEnv::kTwapAction; };
  const double opt = compare(env, [&](const ExecutionObs& o) { return dp.act(o); }, twap, 1000, 77'000'000).mean;
  AgentConfig<3> cfg; cfg.learner = LearnerKind::NeuralQ; cfg.eps_decay_episodes = 10000; cfg.seed = seed;
  ExecutionEnv e(p); auto agent = std::make_unique<Agent<ExecutionEnv>>(e, cfg);
  Trainer<ExecutionEnv> tr(*agent, e, {.eval_every = 1000, .eval_episodes = 500, .train_seed = seed, .detect_drift = false});
  const auto t0 = std::chrono::steady_clock::now();
  std::printf("NeuralQ (library) seed %llu:", (unsigned long long)seed);
  unsigned done = 0;
  for (unsigned m : {2000u, 5000u, 10000u, 20000u, 40000u}) {
    if (m > total) break;
    tr.run(m - done); done = m;
    std::printf("  %5u: learner %4.0f%% deployed %4.0f%%", m,
                100 * compare(env, [&](const ExecutionObs& o) { return agent->act(o, false); }, twap, 1000, 77'000'000).mean / opt,
                100 * compare(env, [&](const ExecutionObs& o) { return tr.deployed_act(o); }, twap, 1000, 77'000'000).mean / opt);
    std::fflush(stdout);
  }
  std::printf("  (%.0f s)\n", std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
}
