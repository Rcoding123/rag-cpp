// E23: ensembles of compiled policies. Loads the tables e22_table saved
// (seed1..8, one bin count), evaluates every non-empty subset (255) with
// ExecutionPolicyTable::median (lower median for even sizes) and a plurality vote, and prints CSV rows:
//   bins,size,members,share_median,share_plurality
// Usage: e23_vote TABLE_DIR BINS
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "rag/agent/evaluate.hpp"
#include "rag/agent/execution_dp.hpp"
#include "rag/agent/policy_table.hpp"

using namespace rag::agent;

int main(int argc, char** argv) {
  if (argc < 3) return std::fprintf(stderr, "usage: e23_vote TABLE_DIR BINS\n"), 2;
  const std::string dir = argv[1];
  const std::string bins = argv[2];
  ExecutionParams p;
  const ExecutionEnv env(p);
  const ExecutionDP dp(p);
  auto twap = [](const ExecutionObs&) { return ExecutionEnv::kTwapAction; };
  const double opt = compare(env, [&](const ExecutionObs& o) { return dp.act(o); }, twap, 1000, 77'000'000).mean;
  std::vector<ExecutionPolicyTable> all;
  for (int s = 1; s <= 8; ++s) all.push_back(ExecutionPolicyTable::load(dir + "/seed" + std::to_string(s) + "_bins" + bins + ".tbl"));
  for (unsigned mask = 1; mask < 256u; ++mask) {
    const int size = __builtin_popcount(mask);
    std::vector<ExecutionPolicyTable> members;
    std::string names;
    for (int s = 0; s < 8; ++s)
      if (mask & (1u << s)) {
        members.push_back(all[static_cast<std::size_t>(s)]);
        names += std::to_string(s + 1);
      }
    const auto ens = ExecutionPolicyTable::median(members);
    // Plurality vote for comparison: the most common action, ties to the lower one.
    const ExecutionPolicyTable plural(p, [&](const ExecutionObs& o) {
      unsigned count[ExecutionEnv::kNumActions] = {};
      for (const auto& m : members) ++count[m.act(o)];
      Action best = 0;
      for (Action a = 1; a < ExecutionEnv::kNumActions; ++a) if (count[a] > count[best]) best = a;
      return best;
    }, {.signal_bins = all[0].signal_bins()});
    const double share = compare(env, [&](const ExecutionObs& o) { return ens.act(o); }, twap, 1000, 77'000'000).mean / opt;
    const double share_plural = compare(env, [&](const ExecutionObs& o) { return plural.act(o); }, twap, 1000, 77'000'000).mean / opt;
    std::printf("%s,%d,%s,%.6f,%.6f\n", bins.c_str(), size, names.c_str(), share, share_plural);
  }
}
