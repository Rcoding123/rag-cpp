// Calibrate the model-gap z-score: model right vs wrong, several eval sets and training seeds.
#include <cstdio>
#include "rag/agent/model_based.hpp"
#include "rag/agent/trainer.hpp"
using namespace rag::agent;
int main() {
  for (int wrong = 0; wrong < 2; ++wrong)
    for (unsigned n : {100u, 400u, 1500u}) {
      std::printf("%s n=%4u z:", wrong ? "model wrong (gamma 0.005, no term)" : "model right (gamma 0.005, with term)", n);
      for (std::uint64_t seed = 1; seed <= 6; ++seed) {
        ExecutionParams p; p.permanent_impact = 0.005; ExecutionEnv env(p);
        ModelBasedExecutionAgent a(p, {.cost_model = CostModel::Shrunk, .learn_permanent_impact = !wrong, .seed = seed});
        Rng r(42 + seed); ExecutionEnv e(p); train_model_based(a, e, 300, r); a.replan();
        const auto ev = evaluate(env, [&](const ExecutionObs& o) { return a.act(o, false); }, n, 6'000'000 + 100'000 * seed);
        std::printf(" %+6.1f", (a.plan()->expected_start_value() - ev.mean) / ev.std_error);
      }
      std::printf("\n");
    }
  // DP value vs simulation for the TRUE optimum (should be ~0 if the 61-point plan grid is unbiased)
  ExecutionParams p; const ExecutionDP dp61(p, 61), dp121(p);
  for (std::uint64_t s : {6'000'000ull, 7'000'000ull, 8'000'000ull}) {
    ExecutionEnv env(p);
    const auto a = evaluate(env, [&](const ExecutionObs& o) { return dp61.act(o); }, 1500, s);
    std::printf("true model, grid 61: predicted %+.2f realized %+.2f +- %.2f\n", dp61.expected_start_value(), a.mean, a.std_error);
  }
}
