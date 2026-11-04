#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include "rag/agent/execution_dp.hpp"
#include "rag/agent/trainer.hpp"
using namespace rag::agent;
int main(int argc, char** argv) {
  const unsigned hidden = std::atoi(argv[1]), batch = std::atoi(argv[2]), eps = std::atoi(argv[3]);
  ExecutionParams p; ExecutionEnv env(p); const ExecutionDP dp(p);
  auto twap = [](const ExecutionObs&) { return ExecutionEnv::kTwapAction; };
  const double opt = compare(env, [&](const ExecutionObs& o) { return dp.act(o); }, twap, 1000, 61'000'000).mean;
  for (std::uint64_t seed = 1; seed <= 3; ++seed) {
    AgentConfig<3> cfg; cfg.learner = LearnerKind::NeuralQ; cfg.eps_decay_episodes = eps; cfg.seed = seed; cfg.nn_hidden = hidden; cfg.nn_batch = batch;
    ExecutionEnv e(p); auto agent = std::make_unique<Agent<ExecutionEnv>>(e, cfg);
    Trainer<ExecutionEnv> tr(*agent, e, {.eval_every = 0, .train_seed = seed, .detect_drift = false});
    const auto t0 = std::chrono::steady_clock::now();
    tr.run(eps);
    std::printf("hidden %u batch %u, %u episodes, seed %llu: %.0f%% (%.1f s)\n", hidden, batch, eps, (unsigned long long)seed,
                100 * compare(env, [&](const ExecutionObs& o) { return agent->act(o, false); }, twap, 1000, 61'000'000).mean / opt,
                std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
  }
}
