// Differential check: banded DP vs the reference implementation.
#include <cmath>
#include <cstdio>
#include "rag/agent/execution_dp.hpp"
#include "execution_dp_ref.hpp"
using namespace rag::agent;
int main() {
  std::vector<ExecutionParams> ps;
  ExecutionParams b; ps.push_back(b);
  { auto p = b; p.permanent_impact = 0.01; p.risk_aversion = 0.3; ps.push_back(p); }
  { auto p = b; p.signal_rho = 0.0; ps.push_back(p); }
  { auto p = b; p.signal_rho = 0.99; ps.push_back(p); }
  { auto p = b; p.alpha = -2; p.signal_sd = 0.3; p.horizon = 7; p.lots = 33; ps.push_back(p); }
  { auto p = b; p.impact_exponent = 3; p.impact = 0.002; p.fixed_cost = 1.5; ps.push_back(p); }
  int bad = 0;
  for (std::size_t c = 0; c < ps.size(); ++c)
    for (unsigned grid : {31u, 61u, 121u}) {
      const auto& p = ps[c];
      const ExecutionDP fast(p, grid); const ExecutionDPRef ref(p, grid);
      Rng rng(c * 100 + grid);
      double maxdiff = 0; int act_mismatch = 0, real_mismatch = 0;
      for (int i = 0; i < 20000; ++i) {
        const ExecutionObs o{rng.below(p.horizon), rng.below(p.lots + 1), 1.5 * p.signal_sd * rng.normal()};
        maxdiff = std::max(maxdiff, std::abs(fast.value(o.t, o.inventory, o.signal) - ref.value(o.t, o.inventory, o.signal)));
        const Action a = fast.act(o), r = ref.act(o);
        if (a != r) { ++act_mismatch; if (std::abs(ref.q_value(o.t, o.inventory, o.signal, a) - ref.q_value(o.t, o.inventory, o.signal, r)) > 1e-9) ++real_mismatch; }
      }
      const auto twap = [](const ExecutionObs&) { return ExecutionEnv::kTwapAction; };
      const double pv = std::abs(fast.policy_value(twap) - ref.policy_value(twap));
      const double sv = std::abs(fast.expected_start_value() - ref.expected_start_value());
      const bool ok = maxdiff < 1e-9 && real_mismatch == 0 && pv < 1e-9 && sv < 1e-9;
      bad += !ok;
      std::printf("case %zu grid %3u: max |dV| %.2e  start %.1e  twap %.1e  action mismatches %d (non-ties %d) %s\n", c, grid, maxdiff, sv, pv, act_mismatch, real_mismatch, ok ? "ok" : "FAIL");
    }
  return bad;
}
