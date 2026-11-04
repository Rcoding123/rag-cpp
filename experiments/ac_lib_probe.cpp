#include <cstdio>
#include <cstdlib>
#include "rag/agent/execution_dp.hpp"
#include "rag/agent/trainer.hpp"
using namespace rag::agent;
int main(int argc, char** argv) {
  const unsigned decay = argc > 1 ? std::atoi(argv[1]) : 3000;
  ExecutionParams p; ExecutionEnv env(p); const ExecutionDP dp(p);
  auto twap = [](const ExecutionObs&) { return ExecutionEnv::kTwapAction; };
  const double opt = compare(env, [&](const ExecutionObs& o) { return dp.act(o); }, twap, 2000, 5'000'000).mean;
  for (unsigned episodes : {10000u, 40000u}) {
    std::printf("alpha decay %u, %6u episodes:", decay, episodes);
    for (std::uint64_t seed = 1; seed <= 3; ++seed) {
      AgentConfig<3> cfg; cfg.learner = LearnerKind::ActorCritic; cfg.alpha_decay_episodes = decay; cfg.seed = seed;
      ExecutionEnv e(p); Agent<ExecutionEnv> agent(e, cfg);
      Trainer<ExecutionEnv> tr(agent, e, {.eval_every = episodes / 8, .eval_episodes = 1000, .train_seed = seed, .detect_drift = false});
      tr.run(episodes);
      std::printf("  learner %5.1f%% deployed %5.1f%% |", 100 * compare(env, [&](const ExecutionObs& o) { return agent.act(o, false); }, twap, 2000, 5'000'000).mean / opt,
                  100 * compare(env, [&](const ExecutionObs& o) { return tr.deployed_act(o); }, twap, 2000, 5'000'000).mean / opt);
    }
    std::printf("\n");
  }
}
