// E22: compile a trained NeuralQ + Double DQN policy into a lookup table.
// Usage: e22_table SEED [episodes=40000] [table_dir]. Prints CSV rows:
//   seed,policy,bins,range_sd,bytes,share,agreement,ns_per_decision
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <string>
#include <vector>

#include "rag/agent/execution_dp.hpp"
#include "rag/agent/policy_table.hpp"
#include "rag/agent/trainer.hpp"

using namespace rag::agent;

template <class F>
double ns_per_decision(F&& f, const std::vector<ExecutionObs>& obs) {
  volatile Action sink = 0;
  const auto t0 = std::chrono::steady_clock::now();
  for (int rep = 0; rep < 5; ++rep)
    for (const auto& o : obs) sink = sink + f(o);
  const auto t1 = std::chrono::steady_clock::now();
  return std::chrono::duration<double, std::nano>(t1 - t0).count() / (5.0 * static_cast<double>(obs.size()));
}

int main(int argc, char** argv) {
  const std::uint64_t seed = std::strtoull(argv[1], nullptr, 10);
  const unsigned episodes = argc > 2 ? static_cast<unsigned>(std::atoi(argv[2])) : 40000;
  ExecutionParams p;
  const ExecutionEnv env(p);
  const ExecutionDP dp(p);
  auto twap = [](const ExecutionObs&) { return ExecutionEnv::kTwapAction; };
  const double opt = compare(env, [&](const ExecutionObs& o) { return dp.act(o); }, twap, 1000, 77'000'000).mean;

  AgentConfig<3> c;
  c.seed = seed;
  c.eps_decay_episodes = 10000;
  c.learner = LearnerKind::NeuralQ;
  c.double_q = true;
  ExecutionEnv e(p);
  auto agent = std::make_unique<Agent<ExecutionEnv>>(e, c);
  Trainer<ExecutionEnv> tr(*agent, e, {.eval_every = 1000, .eval_episodes = 500, .train_seed = seed, .detect_drift = false});
  tr.run(episodes);
  auto net = [&](const ExecutionObs& o) { return tr.deployed_act(o); };

  // States the network visits on the held-out scenarios: for agreement and timing.
  std::vector<ExecutionObs> visited;
  {
    ExecutionEnv ev(p);
    Rng r(77'000'000);
    for (int ep = 0; ep < 1000; ++ep) {
      auto o = ev.reset(r);
      for (;;) {
        visited.push_back(o);
        const auto st = ev.step(net(o), r);
        o = st.obs;
        if (st.done) break;
      }
    }
  }
  const double net_share = compare(env, net, twap, 1000, 77'000'000).mean / opt;
  std::printf("%llu,network,0,0,%zu,%.6f,1,%.1f\n", (unsigned long long)seed, std::size_t{0}, net_share, ns_per_decision(net, visited));
  for (double range : {3.0, 4.0})
    for (std::uint32_t bins : {4u, 8u, 16u, 32u, 64u, 128u, 256u}) {
      if (range == 3.0 && bins != 64) continue;
      const ExecutionPolicyTable table(p, net, {.signal_bins = bins, .signal_range_sd = range});
      auto tab = [&](const ExecutionObs& o) { return table.act(o); };
      if (range == 4.0 && (bins == 64 || bins == 256) && argc > 3)  // E23: keep the tables for ensembling
        table.save(std::string(argv[3]) + "/seed" + std::to_string(seed) + "_bins" + std::to_string(bins) + ".tbl");
      std::size_t same = 0;
      for (const auto& o : visited) same += table.act(o) == net(o);
      std::printf("%llu,table,%u,%.0f,%zu,%.6f,%.6f,%.1f\n", (unsigned long long)seed, bins, range, table.bytes(),
                  compare(env, tab, twap, 1000, 77'000'000).mean / opt, static_cast<double>(same) / static_cast<double>(visited.size()),
                  ns_per_decision(tab, visited));
      std::fflush(stdout);
    }
}
