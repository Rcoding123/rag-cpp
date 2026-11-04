#include <chrono>
#include <cstdio>
#include <vector>
#include "rag/agent/trainer.hpp"
using namespace rag::agent;
int main() {
  for (int k = 0; k < 2; ++k) {
    AgentConfig<3> cfg; cfg.learner = k ? LearnerKind::NeuralQ : LearnerKind::TileQ;
    ExecutionEnv env{ExecutionParams{}}; Agent<ExecutionEnv> agent(env, cfg);
    Rng r(1); std::vector<ExecutionObs> obs(256);
    for (auto& o : obs) o = {r.below(20), r.below(101), 0.1 * r.normal()};
    std::uint64_t sink = 0; const int n = 2'000'000;
    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < n; ++i) sink += agent.act(obs[i & 255], false);
    const double ns = std::chrono::duration<double, std::nano>(std::chrono::steady_clock::now() - t0).count() / n;
    std::printf("%s greedy decision: %.0f ns (%llu)\n", k ? "NeuralQ 64-64" : "TileQ", ns, (unsigned long long)sink);
  }
}
