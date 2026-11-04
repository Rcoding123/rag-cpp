// E7: what does learning the permanent-impact term cost when it isn't needed?
#include <cstdio>
#include "rag/agent/model_based.hpp"
#include "rag/agent/trainer.hpp"
using namespace rag::agent;
int main() {
  auto twap = [](const ExecutionObs&) { return ExecutionEnv::kTwapAction; };
  struct Case { const char* name; ExecutionParams p; };
  ExecutionParams base, r02 = base, r1 = base, g01 = base;
  r02.risk_aversion = 0.2; r1.risk_aversion = 1.0; g01.permanent_impact = 0.01;
  for (const Case& c : {Case{"default", base}, Case{"lambda 0.2", r02}, Case{"lambda 1", r1}, Case{"gamma 0.01", g01}}) {
    ExecutionEnv env(c.p); const ExecutionDP dp(c.p);
    const double opt = compare(env, [&](const ExecutionObs& o) { return dp.act(o); }, twap, 1000, 5'000'000).mean;
    for (unsigned eps : {20u, 50u, 300u})
      for (int perm = 0; perm < 2; ++perm) {
        double lo = 1e9, hi = -1e9, sum = 0, hold = 0, pm = 0;
        for (std::uint64_t seed = 1; seed <= 5; ++seed) {
          ModelBasedExecutionAgent a(c.p, {.cost_model = CostModel::Shrunk, .learn_permanent_impact = perm == 1, .seed = seed});
          Rng r(42 + seed); ExecutionEnv e(c.p); train_model_based(a, e, eps, r); a.replan();
          const double s = 100 * compare(env, [&](const ExecutionObs& o) { return a.act(o, false); }, twap, 1000, 5'000'000).mean / opt;
          lo = std::min(lo, s); hi = std::max(hi, s); sum += s; hold += a.model().holding / 5; pm += a.model().permanent / 5;
        }
        std::printf("%-11s %4u eps  permanent %-3s  share mean %5.1f%% [%5.1f, %5.1f]  holding %.5f  permanent %.5f\n", c.name, eps,
                    perm ? "on" : "off", sum / 5, lo, hi, hold, pm);
      }
  }
}
