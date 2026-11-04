// E23 check on fresh scenarios: the median-vote ensemble of all 8 seeds'
// tables against each member and against the exact optimum, on 4,000
// scenarios none of the earlier evaluations used. Paired differences vs the
// ensemble (common random numbers) with standard errors.
// Usage: e23_fresh TABLE_DIR BINS
#include <chrono>
#include <cstdio>
#include <string>
#include <vector>

#include "rag/agent/evaluate.hpp"
#include "rag/agent/execution_dp.hpp"
#include "rag/agent/policy_table.hpp"

using namespace rag::agent;

int main(int argc, char** argv) {
  if (argc < 3) return 2;
  const std::string dir = argv[1], bins = argv[2];
  const ExecutionParams p;
  const ExecutionEnv env(p);
  const ExecutionDP dp(p);
  constexpr std::uint64_t kSeed = 99'000'000;
  constexpr std::uint32_t kN = 4000;
  auto twap = [](const ExecutionObs&) { return ExecutionEnv::kTwapAction; };
  const double opt = compare(env, [&](const ExecutionObs& o) { return dp.act(o); }, twap, kN, kSeed).mean;
  std::vector<ExecutionPolicyTable> all;
  for (int s = 1; s <= 8; ++s) all.push_back(ExecutionPolicyTable::load(dir + "/seed" + std::to_string(s) + "_bins" + bins + ".tbl"));
  const auto ens = ExecutionPolicyTable::median(all);
  auto ens_act = [&](const ExecutionObs& o) { return ens.act(o); };
  std::printf("ensemble of 8: %.1f%% of the optimal gain (se %.1f)\n", 100 * compare(env, ens_act, twap, kN, kSeed).mean / opt,
              100 * compare(env, ens_act, twap, kN, kSeed).std_error / opt);
  for (int s = 0; s < 8; ++s) {
    const auto& t = all[static_cast<std::size_t>(s)];
    auto act = [&](const ExecutionObs& o) { return t.act(o); };
    const auto d = compare(env, ens_act, act, kN, kSeed);  // ensemble minus member, paired
    std::printf("seed %d: %.1f%%; ensemble better by %.1f points (se %.1f)\n", s + 1, 100 * compare(env, act, twap, kN, kSeed).mean / opt,
                100 * d.mean / opt, 100 * d.std_error / opt);
  }
  // Lookup latency on a quiet core: 10M lookups over pseudo-random states.
  std::vector<ExecutionObs> obs;
  Rng r(5);
  for (int i = 0; i < 4096; ++i) obs.push_back({static_cast<std::uint32_t>(r.below(20)), static_cast<std::uint32_t>(r.below(101)), 0.1 * r.normal()});
  volatile Action sink = 0;
  const auto t0 = std::chrono::steady_clock::now();
  for (int rep = 0; rep < 2500; ++rep)
    for (const auto& o : obs) sink = sink + ens.act(o);
  const auto t1 = std::chrono::steady_clock::now();
  std::printf("lookup: %.2f ns\n", std::chrono::duration<double, std::nano>(t1 - t0).count() / (2500.0 * 4096));
}
