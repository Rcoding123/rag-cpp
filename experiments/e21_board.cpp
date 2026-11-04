// E21: leaderboard. One learner, a list of seeds; prints CSV: learner,seed,episodes,share (deployed champion).
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <string>
#include "rag/agent/execution_dp.hpp"
#include "rag/agent/model_based.hpp"
#include "rag/agent/trainer.hpp"
using namespace rag::agent;
int main(int argc, char** argv) {
  const std::string who = argv[1];
  ExecutionParams p; const ExecutionEnv env(p); const ExecutionDP dp(p);
  auto twap = [](const ExecutionObs&) { return ExecutionEnv::kTwapAction; };
  const double opt = compare(env, [&](const ExecutionObs& o) { return dp.act(o); }, twap, 1000, 77'000'000).mean;
  for (int i = 2; i < argc; ++i) {
    const std::uint64_t seed = std::strtoull(argv[i], nullptr, 10);
    if (who == "model_based") {
      for (unsigned n : {20u, 50u}) {
        ModelBasedExecutionAgent mb(p, {.cost_model = CostModel::Shrunk, .seed = seed}); ExecutionEnv e(p); Rng r(1000 + seed);
        train_model_based(mb, e, n, r); mb.replan();
        std::printf("%s,%llu,%u,%.6f\n", who.c_str(), (unsigned long long)seed, n,
                    compare(env, [&](const ExecutionObs& o) { return mb.act(o, false); }, twap, 1000, 77'000'000).mean / opt);
        std::fflush(stdout);
      }
      continue;
    }
    AgentConfig<3> c; c.seed = seed; c.eps_decay_episodes = 10000;
    if (who == "tileq_tuned") { c.alpha = 0.002763; c.alpha_min = 0.002763; c.alpha_decay_episodes = 678; c.eps_start = 0.02; c.eps_end = 0.02; c.eps_decay_episodes = 15380; c.bins = {4, 4, 4}; c.tilings = 7; }
    if (who == "tileq_coarse_al") { c.bins = {2, 4, 4}; c.tilings = 16; c.advantage_learning = 0.5; }
    if (who == "actor_critic") c.learner = LearnerKind::ActorCritic;
    if (who == "neuralq_double") { c.learner = LearnerKind::NeuralQ; c.double_q = true; }
    ExecutionEnv e(p); auto agent = std::make_unique<Agent<ExecutionEnv>>(e, c);
    Trainer<ExecutionEnv> tr(*agent, e, {.eval_every = 1000, .eval_episodes = 500, .train_seed = seed, .detect_drift = false});
    unsigned done = 0;
    for (unsigned m : {20000u, 40000u}) {
      tr.run(m - done); done = m;
      std::printf("%s,%llu,%u,%.6f\n", who.c_str(), (unsigned long long)seed, m,
                  compare(env, [&](const ExecutionObs& o) { return tr.deployed_act(o); }, twap, 1000, 77'000'000).mean / opt);
      std::fflush(stdout);
    }
  }
}
