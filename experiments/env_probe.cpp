#include <chrono>
#include <cmath>
#include <cstdio>
#include "rag/agent/execution_dp.hpp"
using namespace rag::agent;
template <class P> void run(const char* name, P&& policy, const ExecutionParams& p, int n) {
  double s = 0, s2 = 0, worst_id = 0;
  for (int i = 0; i < n; ++i) {
    ExecutionEnv env(p); Rng rng(1000 + i);
    auto o = env.reset(rng); double total = 0;
    for (;;) { auto st = env.step(policy(o), rng); total += st.reward; o = st.obs; if (st.done) break; }
    worst_id = std::max(worst_id, std::abs(total - (env.cash() - p.lots * p.start_price)));
    s += total; s2 += total * total;
  }
  const double m = s / n, sd = std::sqrt(s2 / n - m * m);
  std::printf("%-12s mean %8.4f  sd %7.4f  stderr %.4f  (accounting identity max error %.2e)\n", name, m, sd, sd / std::sqrt(n), worst_id);
}
int main() {
  ExecutionParams p;
  auto t0 = std::chrono::steady_clock::now();
  ExecutionDP dp(p);
  auto t1 = std::chrono::steady_clock::now();
  std::printf("DP solve: %.0f ms, predicted optimal start value %.4f\n", std::chrono::duration<double, std::milli>(t1 - t0).count(), dp.expected_start_value());
  std::printf("TWAP analytic: %.4f\n", -p.impact * 25.0 * 20);
  const int n = 20000;
  run("TWAP", [](const ExecutionObs&) { return ExecutionEnv::kTwapAction; }, p, n);
  run("DP optimal", [&](const ExecutionObs& o) { return dp.act(o); }, p, n);
  run("dump now", [](const ExecutionObs&) { return Action{4}; }, p, n);
  Rng r(7); run("random", [&](const ExecutionObs&) { return Action{r.below(5)}; }, p, n);
  run("sign rule", [](const ExecutionObs& o) { return o.signal < 0 ? Action{3} : Action{1}; }, p, n);
}
