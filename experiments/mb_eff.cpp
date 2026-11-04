#include <cstdio>
#include "rag/agent/model_based.hpp"
#include "rag/agent/trainer.hpp"
using namespace rag::agent;
int main() {
  ExecutionParams base, hard; hard.impact_exponent = 3.0; hard.impact = 0.01 * 25 / 125.0; hard.fixed_cost = 1.5;
  auto twap = [](const ExecutionObs&) { return ExecutionEnv::kTwapAction; };
  for (auto [name, p] : {std::pair{"quadratic market", base}, std::pair{"e=3 + fixed 1.5", hard}}) {
    ExecutionEnv env(p);
    const ExecutionDP dp(p);
    const double opt = compare(env, [&](const ExecutionObs& o) { return dp.act(o); }, twap, 1000, 5'000'000).mean;
    std::printf("%s\n  %-10s", name, "episodes");
    const unsigned budgets[] = {5, 10, 20, 50, 100, 300};
    for (unsigned b : budgets) std::printf("%8u", b);
    for (CostModel cm : {CostModel::Quadratic, CostModel::PerSize, CostModel::Shrunk}) {
      std::printf("\n  %-10s", cm == CostModel::Quadratic ? "quadratic" : cm == CostModel::PerSize ? "per-size" : "shrunk");
      for (unsigned b : budgets) {
        double sum = 0;
        for (std::uint64_t seed = 1; seed <= 5; ++seed) {
          ModelBasedExecutionAgent a(p, {.explore_episodes = std::min(10u, b), .cost_model = cm, .seed = seed});
          Rng rng(seed * 77);
          train_model_based(a, env, b, rng);
          a.replan();
          sum += compare(env, [&](const ExecutionObs& o) { return a.act(o, false); }, twap, 1000, 5'000'000).mean / opt;
        }
        std::printf("%7.0f%%", 100 * sum / 5);
      }
    }
    std::printf("\n");
  }
}
