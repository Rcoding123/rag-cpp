#include <cstdio>
#include "rag/agent/model_based.hpp"
#include "rag/agent/trainer.hpp"
using namespace rag::agent;
int main() {
  ExecutionParams p; p.impact_exponent = 3.0; p.impact = 0.01 * 25 / 125.0; p.fixed_cost = 1.5;
  ExecutionEnv env(p); const ExecutionDP dp(p);
  auto twap = [](const ExecutionObs&) { return ExecutionEnv::kTwapAction; };
  const double opt = compare(env, [&](const ExecutionObs& o) { return dp.act(o); }, twap, 1000, 31'000'000).mean;
  for (unsigned b : {10u, 20u}) for (CostModel cm : {CostModel::PerSize, CostModel::Shrunk}) {
    std::printf("%u episodes %-9s:", b, cm == CostModel::PerSize ? "per-size" : "shrunk");
    for (std::uint64_t seed = 1; seed <= 10; ++seed) {
      ModelBasedExecutionAgent a(p, {.explore_episodes = 10, .cost_model = cm, .seed = seed});
      Rng rng(seed * 77); train_model_based(a, env, b, rng); a.replan();
      std::printf(" %6.0f%%", 100 * compare(env, [&](const ExecutionObs& o) { return a.act(o, false); }, twap, 1000, 31'000'000).mean / opt);
    }
    std::printf("\n");
  }
}
