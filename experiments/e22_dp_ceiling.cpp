// E22 ceiling: compile the exact optimal (DP) policy into tables of each size.
// What a table loses here is the cost of the grid alone, not of the network.
// Prints CSV: bins,range_sd,share_of_optimal_gain
#include <cstdio>

#include "rag/agent/evaluate.hpp"
#include "rag/agent/execution_dp.hpp"
#include "rag/agent/policy_table.hpp"

using namespace rag::agent;

int main() {
  const ExecutionParams p;
  const ExecutionEnv env(p);
  const ExecutionDP dp(p);
  auto twap = [](const ExecutionObs&) { return ExecutionEnv::kTwapAction; };
  auto opt = [&](const ExecutionObs& o) { return dp.act(o); };
  const double best = compare(env, opt, twap, 1000, 77'000'000).mean;
  for (double range : {3.0, 4.0})
    for (std::uint32_t bins : {4u, 8u, 16u, 32u, 64u, 128u, 256u}) {
      if (range == 3.0 && bins != 64) continue;
      const ExecutionPolicyTable table(p, opt, {.signal_bins = bins, .signal_range_sd = range});
      std::printf("%u,%.0f,%.6f\n", bins, range,
                  compare(env, [&](const ExecutionObs& o) { return table.act(o); }, twap, 1000, 77'000'000).mean / best);
    }
}
