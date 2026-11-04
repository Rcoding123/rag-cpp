// E8: does greedy (eps 0) keep up with eps 1->0.02 at 40k and 100k episodes? (final greedy share)
#include <cstdio>
#include <cstdlib>
#include <memory>
#include "rag/agent/trainer.hpp"
#include "rag/agent/execution_dp.hpp"
using namespace rag::agent;
int main(int argc, char** argv) {
  const std::uint64_t seed = argc > 1 ? std::strtoull(argv[1], nullptr, 10) : 1;
  auto twap = [](const ExecutionObs&) { return ExecutionEnv::kTwapAction; };
  ExecutionParams p; ExecutionEnv env(p); const ExecutionDP dp(p);
  const double opt = compare(env, [&](const ExecutionObs& o) { return dp.act(o); }, twap, 1000, 5'000'000).mean;
  struct V { const char* name; double e0, e1; };
  for (V v : {V{"eps 1->0.02/10k", 1.0, 0.02}, V{"eps 0.1->0.02/10k", 0.1, 0.02}, V{"eps 0 (greedy)", 0.0, 0.0}}) {
    AgentConfig<3> c; c.eps_start = v.e0; c.eps_end = v.e1; c.eps_decay_episodes = 10000; c.seed = seed;
    ExecutionEnv e(p); auto agent = std::make_unique<Agent<ExecutionEnv>>(e, c);
    Trainer<ExecutionEnv> tr(*agent, e, {.eval_every = 5000, .eval_episodes = 500, .train_seed = seed, .detect_drift = false});
    std::printf("seed %llu %-18s", (unsigned long long)seed, v.name);
    unsigned done = 0;
    for (unsigned m : {10000u, 40000u, 100000u}) {
      tr.run(m - done); done = m;
      std::printf("  %6u: learner %4.0f%% deployed %4.0f%%", m, 100 * compare(env, [&](const ExecutionObs& o) { return agent->act(o, false); }, twap, 1000, 5'000'000).mean / opt,
                  100 * compare(env, [&](const ExecutionObs& o) { return tr.deployed_act(o); }, twap, 1000, 5'000'000).mean / opt);
    }
    std::printf("\n");
  }
}
