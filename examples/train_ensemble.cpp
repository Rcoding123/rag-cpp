// train_ensemble — the best learned execution policy (E25, DECISIONS A48),
// trained in parallel: N NeuralQ networks (Double DQN, weight averaging
// nn_polyak = 1e-4, seeds 1..N), one per thread; each deployed champion
// compiled to an ExecutionPolicyTable; the cell-wise median of the tables
// saved as the deployable policy. Scores every member and the ensemble
// against the exact optimum on 1,000 held-out scenarios.
//
//   train_ensemble [networks=4] [episodes=40000] [threads=networks] [out=ensemble.tbl]
//
// Every network's result is independent of the thread count (each thread
// owns its agent, environment and RNG streams).
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "rag/agent/execution_dp.hpp"
#include "rag/agent/policy_table.hpp"
#include "rag/agent/trainer.hpp"

using namespace rag::agent;

int main(int argc, char** argv) {
  const unsigned networks = argc > 1 ? static_cast<unsigned>(std::max(1, std::atoi(argv[1]))) : 4u;
  const unsigned episodes = argc > 2 ? static_cast<unsigned>(std::max(1, std::atoi(argv[2]))) : 40000u;
  const unsigned threads = argc > 3 ? static_cast<unsigned>(std::max(1, std::atoi(argv[3]))) : networks;
  const std::string out = argc > 4 ? argv[4] : "ensemble.tbl";

  const ExecutionParams p;
  const ExecutionEnv env(p);
  const ExecutionDP dp(p);
  auto twap = [](const ExecutionObs&) { return ExecutionEnv::kTwapAction; };
  constexpr std::uint32_t kScenarios = 1000;
  constexpr std::uint64_t kEvalSeed = 77'000'000;  // the held-out scenarios of E21-E25
  const double opt = compare(env, [&](const ExecutionObs& o) { return dp.act(o); }, twap, kScenarios, kEvalSeed).mean;

  std::printf("Training %u networks x %u episodes on %u threads (hardware threads: %u)...\n", networks, episodes, threads,
              std::thread::hardware_concurrency());
  const auto t0 = std::chrono::steady_clock::now();
  std::vector<std::unique_ptr<ExecutionPolicyTable>> tables(networks);
  std::vector<double> network_share(networks);
  std::vector<std::thread> pool;
  unsigned next = 0;
  std::mutex m;  // guards next
  auto worker = [&] {
    for (;;) {
      unsigned i;
      {
        std::lock_guard lock(m);
        if (next >= networks) return;
        i = next++;
      }
      AgentConfig<3> c;
      c.seed = i + 1;
      c.eps_decay_episodes = std::min(10000u, std::max(1u, episodes / 4));
      c.learner = LearnerKind::NeuralQ;
      c.double_q = true;
      c.nn_polyak = 1e-4;
      ExecutionEnv e(p);
      auto agent = std::make_unique<Agent<ExecutionEnv>>(e, c);
      Trainer<ExecutionEnv> tr(*agent, e,
                               {.eval_every = std::min(1000u, std::max(1u, episodes / 10)), .eval_episodes = 500, .train_seed = c.seed,
                                .detect_drift = false});
      tr.run(episodes);
      auto deployed = [&](const ExecutionObs& o) { return tr.deployed_act(o); };
      network_share[i] = compare(env, deployed, twap, kScenarios, kEvalSeed).mean / opt;
      tables[i] = std::make_unique<ExecutionPolicyTable>(p, deployed, PolicyTableParams{.signal_bins = 256, .signal_range_sd = 4.0});
    }
  };
  for (unsigned t = 0; t < std::min(threads, networks); ++t) pool.emplace_back(worker);
  for (auto& t : pool) t.join();
  const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

  std::vector<ExecutionPolicyTable> members;
  for (unsigned i = 0; i < networks; ++i) {
    const double table_share =
        compare(env, [&](const ExecutionObs& o) { return tables[i]->act(o); }, twap, kScenarios, kEvalSeed).mean / opt;
    std::printf("  network %2u: %5.1f%% of the optimal gain (its table: %5.1f%%)\n", i + 1, 100 * network_share[i], 100 * table_share);
    members.push_back(*tables[i]);
  }
  const auto ensemble = ExecutionPolicyTable::median(members);
  const double ens = compare(env, [&](const ExecutionObs& o) { return ensemble.act(o); }, twap, kScenarios, kEvalSeed).mean / opt;
  ensemble.save(out);
  std::printf("Median vote of %u: %.1f%% of the optimal gain. Saved %s (%zu KB). Training took %.0f s.\n", networks, 100 * ens,
              out.c_str(), ensemble.bytes() / 1000, secs);
  std::printf("Load it with ExecutionPolicyTable::load(\"%s\"); act() is a ~3 ns lookup.\n", out.c_str());
}
