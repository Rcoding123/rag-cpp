// E9: stationary cost of forgetting (share after 300 episodes; 5 seeds; min / mean).
#include <cstdio>
#include "rag/agent/model_based.hpp"
#include "rag/agent/trainer.hpp"
using namespace rag::agent;
int main() {
  auto twap = [](const ExecutionObs&) { return ExecutionEnv::kTwapAction; };
  struct Case { const char* name; ExecutionParams p; };
  ExecutionParams base, noisy = base, weak = base, hard = base;
  noisy.noise_sd = 0.2; weak.alpha = 0.25; hard.impact_exponent = 3.0; hard.impact = 0.01 * 25 / 125.0; hard.fixed_cost = 1.5;
  for (const Case& c : {Case{"default", base}, Case{"noise 0.2", noisy}, Case{"alpha 0.25", weak}, Case{"q^3 + fixed cost", hard}}) {
    ExecutionEnv env(c.p); const ExecutionDP dp(c.p);
    const double opt = compare(env, [&](const ExecutionObs& o) { return dp.act(o); }, twap, 1000, 5'000'000).mean;
    std::printf("%-17s", c.name);
    for (double f : {1.0, 0.99, 0.95, 0.9}) {
      double lo = 1e9, sum = 0;
      for (std::uint64_t seed = 1; seed <= 5; ++seed) {
        ModelBasedExecutionAgent a(c.p, {.replan_every = 5, .replan_max_gap = 5, .cost_model = CostModel::Shrunk, .forget = f, .seed = seed});
        ExecutionEnv e(c.p); Rng r(200 + seed); train_model_based(a, e, 300, r); a.replan();
        const double s = 100 * compare(env, [&](const ExecutionObs& o) { return a.act(o, false); }, twap, 1000, 5'000'000).mean / opt;
        lo = std::min(lo, s); sum += s;
      }
      std::printf("  f %.2f: %5.1f%% (min %5.1f%%)", f, sum / 5, lo);
    }
    std::printf("\n");
  }
}
