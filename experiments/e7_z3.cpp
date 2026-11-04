// Paired model check: predicted gain over TWAP (exact, under the learned model) vs realized (paired).
#include <cstdio>
#include "rag/agent/model_based.hpp"
#include "rag/agent/trainer.hpp"
using namespace rag::agent;
int main() {
  auto twap = [](const ExecutionObs&) { return ExecutionEnv::kTwapAction; };
  { // sanity: policy_value of the optimal policy == expected_start_value; TWAP vs simulation
    ExecutionParams p; p.permanent_impact = 0.01; p.risk_aversion = 0.2; const ExecutionDP dp(p); ExecutionEnv env(p);
    const auto tw = evaluate(env, twap, 20000, 1);
    const auto op = evaluate(env, [&](const ExecutionObs& o) { return dp.act(o); }, 20000, 1);
    std::printf("sanity: V*=%.4f policy_value(dp)=%.4f sim %.3f+-%.3f | TWAP exact %.4f sim %.3f+-%.3f\n", dp.expected_start_value(),
                dp.policy_value([&](const ExecutionObs& o) { return dp.act(o); }), op.mean, op.std_error, dp.policy_value(twap), tw.mean, tw.std_error);
  }
  for (int wrong = 0; wrong < 2; ++wrong)
    for (double gamma : {0.0, 0.005, 0.01})
      for (unsigned n : {100u, 400u}) {
        if (!wrong && gamma > 0.005) continue;
        std::printf("%s gamma %.3f n=%3u z:", wrong ? "no term  " : "with term", gamma, n);
        for (std::uint64_t seed = 1; seed <= 8; ++seed) {
          ExecutionParams p; p.permanent_impact = gamma; ExecutionEnv env(p);
          ModelBasedExecutionAgent a(p, {.cost_model = CostModel::Shrunk, .learn_permanent_impact = !wrong, .seed = seed});
          Rng r(42 + seed); ExecutionEnv e(p); train_model_based(a, e, 300, r); a.replan();
          const ExecutionDP& m = *a.plan();
          auto plan = [&](const ExecutionObs& o) { return m.act(o); };
          const double pred = m.policy_value(plan) - m.policy_value(twap);
          const auto real = compare(env, plan, twap, n, 6'000'000 + 100'000 * seed);
          std::printf(" %+6.1f", (pred - real.mean) / real.std_error);
        }
        std::printf("\n");
      }
}
