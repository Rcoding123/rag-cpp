#include <chrono>
#include <cstdio>
#include "rag/agent/model_based.hpp"
#include "rag/agent/trainer.hpp"
using namespace rag::agent;
int main() {
  ExecutionParams p;
  ExecutionEnv env(p);
  const ExecutionDP dp(p);
  auto twap = [](const ExecutionObs&) { return ExecutionEnv::kTwapAction; };
  const double opt = compare(env, [&](const ExecutionObs& o) { return dp.act(o); }, twap, 2000, 5'000'000).mean;
  std::printf("truth: impact %.4f terminal %.4f alpha %.3f rho %.3f signal_sd %.4f\n", p.impact, p.terminal_impact, p.alpha, p.signal_rho, p.signal_sd);
  for (std::uint64_t seed : {1, 2, 3}) {
    ModelBasedExecutionAgent agent(p, {.seed = seed});
    Rng rng(seed * 1000);
    std::uint32_t done = 0;
    for (std::uint32_t target : {10u, 20u, 50u, 100u, 200u, 500u, 1000u}) {
      auto t0 = std::chrono::steady_clock::now();
      train_model_based(agent, env, target - done, rng);
      done = target;
      agent.replan();  // evaluate the plan from all data so far
      const auto m = agent.model();
      const double g = compare(env, [&](const ExecutionObs& o) { return agent.act(o, false); }, twap, 2000, 5'000'000).mean;
      std::printf("seed %llu ep %5u: impact %.4f term %.4f alpha %.3f rho %.3f sd %.4f -> %5.1f%% of optimal (%.0f ms)\n",
                  (unsigned long long)seed, target, m.impact, m.terminal_impact, m.alpha, m.rho, m.signal_sd, 100 * g / opt,
                  std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
    }
  }
}
