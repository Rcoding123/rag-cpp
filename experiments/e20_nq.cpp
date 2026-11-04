// E20: NeuralQ with Double DQN and/or advantage learning (library learner, standard Trainer).
#include <cstdio>
#include <cstdlib>
#include <memory>
#include "rag/agent/execution_dp.hpp"
#include "rag/agent/trainer.hpp"
using namespace rag::agent;
int main(int argc, char** argv) {
  const bool dq = std::atoi(argv[1]) != 0; const double al = std::atof(argv[2]); const std::uint64_t seed = std::strtoull(argv[3], nullptr, 10);
  ExecutionParams p; const ExecutionEnv env(p); const ExecutionDP dp(p);
  auto twap = [](const ExecutionObs&) { return ExecutionEnv::kTwapAction; };
  const double opt = compare(env, [&](const ExecutionObs& o) { return dp.act(o); }, twap, 1000, 77'000'000).mean;
  AgentConfig<3> cfg; cfg.learner = LearnerKind::NeuralQ; cfg.eps_decay_episodes = 10000; cfg.seed = seed; cfg.double_q = dq; cfg.advantage_learning = al;
  ExecutionEnv e(p); auto agent = std::make_unique<Agent<ExecutionEnv>>(e, cfg);
  Trainer<ExecutionEnv> tr(*agent, e, {.eval_every = 1000, .eval_episodes = 500, .train_seed = seed, .detect_drift = false});
  std::printf("NeuralQ double %d AL %.1f seed %llu:", dq, al, (unsigned long long)seed);
  unsigned done = 0;
  for (unsigned m : {5000u, 10000u, 20000u}) { tr.run(m - done); done = m;
    std::printf("  %5u: learner %4.0f%% deployed %4.0f%%", m, 100 * compare(env, [&](const ExecutionObs& o) { return agent->act(o, false); }, twap, 1000, 77'000'000).mean / opt,
                100 * compare(env, [&](const ExecutionObs& o) { return tr.deployed_act(o); }, twap, 1000, 77'000'000).mean / opt); std::fflush(stdout); }
  std::printf("\n");
}
