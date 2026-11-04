// E24: weight averaging (Polyak / EMA) for NeuralQ + Double DQN, E21 protocol.
// Usage: e24_polyak SEED TAU. Prints CSV:
//   seed,tau,episodes,deployed_share,final_greedy_share
#include <cstdio>
#include <cstdlib>
#include <memory>

#include "rag/agent/execution_dp.hpp"
#include "rag/agent/trainer.hpp"

using namespace rag::agent;

int main(int argc, char** argv) {
  if (argc < 3) return 2;
  const std::uint64_t seed = std::strtoull(argv[1], nullptr, 10);
  const double tau = std::atof(argv[2]);
  const ExecutionParams p;
  const ExecutionEnv env(p);
  const ExecutionDP dp(p);
  auto twap = [](const ExecutionObs&) { return ExecutionEnv::kTwapAction; };
  const double opt = compare(env, [&](const ExecutionObs& o) { return dp.act(o); }, twap, 1000, 77'000'000).mean;
  AgentConfig<3> c;
  c.seed = seed;
  c.eps_decay_episodes = 10000;
  c.learner = LearnerKind::NeuralQ;
  c.double_q = true;
  c.nn_polyak = tau;
  ExecutionEnv e(p);
  auto agent = std::make_unique<Agent<ExecutionEnv>>(e, c);
  Trainer<ExecutionEnv> tr(*agent, e, {.eval_every = 1000, .eval_episodes = 500, .train_seed = seed, .detect_drift = false});
  unsigned done = 0;
  for (unsigned m : {20000u, 40000u}) {
    tr.run(m - done);
    done = m;
    const double dep = compare(env, [&](const ExecutionObs& o) { return tr.deployed_act(o); }, twap, 1000, 77'000'000).mean / opt;
    const double fin = compare(env, [&](const ExecutionObs& o) { return agent->act(o, false); }, twap, 1000, 77'000'000).mean / opt;
    std::printf("%llu,%g,%u,%.6f,%.6f\n", (unsigned long long)seed, tau, m, dep, fin);
    std::fflush(stdout);
  }
}
