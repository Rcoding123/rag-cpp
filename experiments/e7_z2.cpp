#include <cstdio>
#include "rag/agent/model_based.hpp"
#include "rag/agent/trainer.hpp"
using namespace rag::agent;
int main() {
  auto twap = [](const ExecutionObs&) { return ExecutionEnv::kTwapAction; };
  for (double gamma : {0.005, 0.01})
  for (std::uint64_t seed = 1; seed <= 6; ++seed) {
    ExecutionParams p; p.permanent_impact = gamma; ExecutionEnv env(p); const ExecutionDP dp(p);
    const double opt = compare(env, [&](const ExecutionObs& o) { return dp.act(o); }, twap, 1000, 5'000'000).mean;
    ModelBasedExecutionAgent a(p, {.cost_model = CostModel::Shrunk, .learn_permanent_impact = false, .seed = seed});
    Rng r(seed == 1 ? 42 : 42 + seed); ExecutionEnv e(p); train_model_based(a, e, 300, r); a.replan();
    auto pol = [&](const ExecutionObs& o) { return a.act(o, false); };
    const auto ev = evaluate(env, pol, 1500, 6'000'000);
    const auto m = a.model();
    std::printf("gamma %.3f seed %llu: share %4.0f%%  predicted %+7.2f realized %+7.2f (z %+6.1f)  fit alpha %.3f holding %.5f impact %.4f term %.4f\n",
                gamma, (unsigned long long)seed, 100 * compare(env, pol, twap, 1000, 5'000'000).mean / opt, a.plan()->expected_start_value(), ev.mean,
                (a.plan()->expected_start_value() - ev.mean) / ev.std_error, m.alpha, m.holding, m.impact, m.terminal_impact);
  }
}
