// E26: where do the learned policies deviate from the optimum? For each
// saved table (E22 unaveraged, E25 averaged; seeds 1-8) and their median
// votes, along each policy's own trajectories on the 1,000 held-out
// scenarios: how often its action is larger (trades more) or smaller than
// the DP's, and the share of the optimal gain each kind of deviation loses.
// Usage: e26_errors TABLE_DIR LABEL. Prints CSV:
//   label,policy,agree,more,less,loss_more_share,loss_less_share
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
  const double opt = compare(env, [&](const ExecutionObs& o) { return dp.act(o); }, twap, 1000, 77'000'000).mean;
  std::vector<ExecutionPolicyTable> all;
  for (int s = 1; s <= 8; ++s) all.push_back(ExecutionPolicyTable::load(dir + "/seed" + std::to_string(s) + "_bins256.tbl"));
  // Performance-difference identity: V*(s0) - V^pi(s0) = E_pi[sum_t (V*(s_t) - Q*(s_t, pi(s_t)))]
  // along pi's own trajectories, so summing one-step regrets splits the gap exactly
  // (up to the DP's grid) into deviations that trade more and less than optimal.
  auto report = [&](const std::string& name, const ExecutionPolicyTable& t) {
    std::size_t n = 0, agree = 0, more = 0, less = 0;
    double loss_more = 0, loss_less = 0;
    ExecutionEnv ev(p);
    for (std::uint32_t i = 0; i < 1000; ++i) {
      Rng r(77'000'000 + i);
      auto o = ev.reset(r);
      for (;;) {
        const Action a = t.act(o), b = dp.act(o);
        ++n;
        if (a == b) ++agree;
        else {
          const double regret = dp.q_value(o.t, o.inventory, o.signal, b) - dp.q_value(o.t, o.inventory, o.signal, a);
          if (a > b) { ++more; loss_more += regret; } else { ++less; loss_less += regret; }
        }
        const auto st = ev.step(a, r);
        o = st.obs;
        if (st.done) break;
      }
    }
    const double dn = static_cast<double>(n);
    std::printf("%s,%s,%.4f,%.4f,%.4f,%.4f,%.4f\n", label.c_str(), name.c_str(), agree / dn, more / dn, less / dn,
                loss_more / 1000.0 / opt, loss_less / 1000.0 / opt);
  };
  for (int s = 0; s < 8; ++s) report("seed" + std::to_string(s + 1), all[static_cast<std::size_t>(s)]);
  report("median8", ExecutionPolicyTable::median(all));
  report("median3_123", ExecutionPolicyTable::median({all[0], all[1], all[2]}));
}
