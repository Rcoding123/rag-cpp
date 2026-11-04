#include <cstdio>
#include <memory>
#include "rag/agent/execution_dp.hpp"
#include "rag/agent/trainer.hpp"
#include "rag/agent/variance_reduction.hpp"
using namespace rag::agent;
int main() {
  ExecutionParams p; p.noise_sd = 0.2; const ExecutionEnv plain(p); const ExecutionDP dp(p);
  auto twap = [](const ExecutionObs&) { return ExecutionEnv::kTwapAction; };
  const double opt = compare(plain, [&](const ExecutionObs& o) { return dp.act(o); }, twap, 1000, 77'000'000).mean;
  for (int cv = 0; cv < 2; ++cv)
    for (std::uint64_t seed = 1; seed <= 3; ++seed) {
      AgentConfig<3> cfg; cfg.eps_decay_episodes = 10000; cfg.seed = seed;
      double s5 = 0, s40 = 0;
      if (cv) {
        InnovationControlledEnv e{ExecutionEnv(p)}; auto a = std::make_unique<Agent<InnovationControlledEnv>>(e, cfg);
        Trainer<InnovationControlledEnv> tr(*a, e, {.eval_every = 0, .train_seed = seed, .detect_drift = false});
        tr.run(5000); s5 = compare(plain, [&](const ExecutionObs& o) { return a->act(o, false); }, twap, 1000, 77'000'000).mean / opt;
        tr.run(35000); s40 = compare(plain, [&](const ExecutionObs& o) { return a->act(o, false); }, twap, 1000, 77'000'000).mean / opt;
        std::printf("controlled (alpha_hat %.3f) seed %llu: 5k %4.0f%%  40k %4.0f%%\n", e.alpha_hat(), (unsigned long long)seed, 100 * s5, 100 * s40);
      } else {
        ExecutionEnv e(p); auto a = std::make_unique<Agent<ExecutionEnv>>(e, cfg);
        Trainer<ExecutionEnv> tr(*a, e, {.eval_every = 0, .train_seed = seed, .detect_drift = false});
        tr.run(5000); s5 = compare(plain, [&](const ExecutionObs& o) { return a->act(o, false); }, twap, 1000, 77'000'000).mean / opt;
        tr.run(35000); s40 = compare(plain, [&](const ExecutionObs& o) { return a->act(o, false); }, twap, 1000, 77'000'000).mean / opt;
        std::printf("plain seed %llu: 5k %4.0f%%  40k %4.0f%%\n", (unsigned long long)seed, 100 * s5, 100 * s40);
      }
    }
}
