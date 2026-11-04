// E13d: actor-critic re-exploration on the execution task (signal flip at 40k).
#include <cstdio>
#include <memory>
#include "rag/agent/execution_dp.hpp"
#include "rag/agent/trainer.hpp"
using namespace rag::agent;
int main() {
  auto twap = [](const ExecutionObs&) { return ExecutionEnv::kTwapAction; };
  ExecutionParams flipped; flipped.alpha = -1;
  const ExecutionEnv eb(flipped); const ExecutionDP db(flipped);
  const double ob = compare(eb, [&](const ExecutionObs& o) { return db.act(o); }, twap, 1000, 5'000'000).mean;
  for (double temp : {0.0, 2.0})
    for (std::uint64_t seed = 1; seed <= 2; ++seed) {
      AgentConfig<3> cfg; cfg.learner = LearnerKind::ActorCritic; cfg.eps_decay_episodes = 10000; cfg.seed = seed; cfg.ac_reexplore_temperature = temp;
      ExecutionEnv env{ExecutionParams{}}; auto agent = std::make_unique<Agent<ExecutionEnv>>(env, cfg);
      Trainer<ExecutionEnv> tr(*agent, env, {.eval_every = 2500, .eval_episodes = 500, .train_seed = seed, .policy_bank = false});
      tr.run(40000);
      env.set_alpha(-1);
      std::printf("%s seed %llu:", temp == 0 ? "eps-uniform " : "temperature x3", (unsigned long long)seed);
      unsigned done = 0;
      for (unsigned m : {5000u, 20000u, 40000u}) { const auto r = tr.run(m - done); done = m;
        std::printf("  +%u: %4.0f%% (%zu alarms)", m, 100 * compare(eb, [&](const ExecutionObs& o) { return tr.deployed_act(o); }, twap, 1000, 5'000'000).mean / ob, r.drift_alarms.size()); }
      std::printf("\n");
    }
}
