#include <cstdio>
#include <cstdlib>
#include <optional>
#include "rag/agent/execution_dp.hpp"
#include "rag/agent/trainer.hpp"
using namespace rag::agent;
int main(int argc, char** argv) {
  const unsigned episodes = argc > 1 ? std::atoi(argv[1]) : 40000;
  const unsigned nstep = argc > 2 ? std::atoi(argv[2]) : 3;
  ExecutionParams p; ExecutionEnv env(p); const ExecutionDP dp(p);
  auto twap = [](const ExecutionObs&) { return ExecutionEnv::kTwapAction; };
  const double opt = compare(env, [&](const ExecutionObs& o) { return dp.act(o); }, twap, 2000, 5'000'000).mean;
  std::printf("%u episodes, n=%u\n", episodes, nstep);
  for (std::uint64_t seed = 1; seed <= 3; ++seed) {
    AgentConfig<3> cfg; cfg.eps_decay_episodes = episodes / 2; cfg.n_step = nstep; cfg.seed = seed;
    ExecutionEnv e(p);
    Agent<ExecutionEnv> agent(e, cfg);
    Trainer<ExecutionEnv> tr(agent, e, {.eval_every = 0, .train_seed = seed, .detect_drift = false});
    std::optional<Agent<ExecutionEnv>::Snapshot> avg;
    unsigned k = 0;
    for (unsigned ep = 0; ep < episodes; ep += 50) {
      tr.run(50);
      if (ep + 50 > episodes / 2) {  // tail-average the second half (exploration has decayed)
        auto s = agent.snapshot();
        if (!avg) avg = s; else avg->q.blend(s.q, 1.0 / (++k + 1));
      }
    }
    auto last = [&](const ExecutionObs& o) { return agent.act(o, false); };
    auto averaged = [&](const ExecutionObs& o) { return agent.act_as(*avg, o); };
    std::printf("  seed %llu: last iterate %5.1f%%   tail-averaged %5.1f%%   (%u snapshots)\n", (unsigned long long)seed,
                100 * compare(env, last, twap, 2000, 5'000'000).mean / opt, 100 * compare(env, averaged, twap, 2000, 5'000'000).mean / opt, k + 1);
  }
}
