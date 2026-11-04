// E27: quantile vote. E26 found the networks trade too much, even after the
// median vote. Vote with a lower quantile of the members' actions instead:
// action = sorted votes[floor(q * (n - 1))] (q = 0.5 is the lower median).
// Scored on the held-out scenarios (77M) and on fresh ones (99M, 4,000).
// Usage: e27_quantile TABLE_DIR LABEL. Prints CSV: label,q,share_heldout,share_fresh
#include <algorithm>
#include <array>
#include <cstdio>
#include <string>
#include <vector>

#include "rag/agent/evaluate.hpp"
#include "rag/agent/execution_dp.hpp"
#include "rag/agent/policy_table.hpp"

using namespace rag::agent;

int main(int argc, char** argv) {
  if (argc < 3) return 2;
  const std::string dir = argv[1], label = argv[2];
  const ExecutionParams p;
  const ExecutionEnv env(p);
  const ExecutionDP dp(p);
  auto twap = [](const ExecutionObs&) { return ExecutionEnv::kTwapAction; };
  auto opt_act = [&](const ExecutionObs& o) { return dp.act(o); };
  const double opt1 = compare(env, opt_act, twap, 1000, 77'000'000).mean, opt2 = compare(env, opt_act, twap, 4000, 99'000'000).mean;
  std::vector<ExecutionPolicyTable> all;
  for (int s = 1; s <= 8; ++s) all.push_back(ExecutionPolicyTable::load(dir + "/seed" + std::to_string(s) + "_bins256.tbl"));
  for (double q : {0.0, 0.125, 0.25, 0.375, 0.5, 0.625}) {
    const ExecutionPolicyTable t(p, [&](const ExecutionObs& o) {
      std::array<Action, 8> v{};
      for (std::size_t i = 0; i < 8; ++i) v[i] = all[i].act(o);
      std::sort(v.begin(), v.end());
      return v[static_cast<std::size_t>(q * 7)];
    }, {.signal_bins = 256});
    auto act = [&](const ExecutionObs& o) { return t.act(o); };
    std::printf("%s,%.3f,%.4f,%.4f\n", label.c_str(), q, compare(env, act, twap, 1000, 77'000'000).mean / opt1,
                compare(env, act, twap, 4000, 99'000'000).mean / opt2);
  }
}
