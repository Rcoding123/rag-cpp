#include <cstdio>
#include "rag/agent/model_based.hpp"
#include "rag/agent/trainer.hpp"
using namespace rag::agent;
int main() {
  auto twap = [](const ExecutionObs&) { return ExecutionEnv::kTwapAction; };
  ExecutionParams noisy; noisy.noise_sd = 0.2; ExecutionEnv env0(noisy); const ExecutionDP dp(noisy);
  const double opt = compare(env0, [&](const ExecutionObs& o) { return dp.act(o); }, twap, 1000, 31'000'000).mean;
  for (unsigned gap : {5u, 25u})
    for (std::uint64_t seed = 1; seed <= 6; ++seed) {
      ModelBasedExecutionAgent a(noisy, {.replan_every = gap, .replan_max_gap = gap, .cost_model = CostModel::Shrunk, .forget = 0.95, .seed = seed});
      ExecutionEnv e(noisy); Rng r(300 + seed); train_model_based(a, e, 300, r);
      std::printf("gap %u seed %llu: %.1f%%\n", gap, (unsigned long long)seed, 100 * compare(env0, [&](const ExecutionObs& o) { return a.act(o, false); }, twap, 1000, 31'000'000).mean / opt);
    }
}
