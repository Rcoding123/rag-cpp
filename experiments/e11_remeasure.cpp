// Re-measure E1's data efficiency and E8's online gain with the E11 defaults.
#include <cstdio>
#include "rag/agent/model_based.hpp"
#include "rag/agent/trainer.hpp"
using namespace rag::agent;
int main() {
  auto twap = [](const ExecutionObs&) { return ExecutionEnv::kTwapAction; };
  ExecutionParams p; const ExecutionEnv env0(p); const ExecutionDP dp(p);
  const double opt = compare(env0, [&](const ExecutionObs& o) { return dp.act(o); }, twap, 1000, 5'000'000).mean;
  for (CostModel cm : {CostModel::Quadratic, CostModel::Shrunk})
    for (unsigned n : {10u, 20u, 50u}) {
      double lo = 1e9, hi = -1e9;
      for (std::uint64_t seed = 1; seed <= 5; ++seed) {
        ModelBasedExecutionAgent a(p, {.cost_model = cm, .seed = seed}); ExecutionEnv e(p); Rng r(42 + seed);
        train_model_based(a, e, n, r); a.replan();
        const double s = 100 * compare(env0, [&](const ExecutionObs& o) { return a.act(o, false); }, twap, 1000, 5'000'000).mean / opt;
        lo = std::min(lo, s); hi = std::max(hi, s);
      }
      std::printf("%s after %2u episodes: %.1f-%.1f%%\n", cm == CostModel::Shrunk ? "Shrunk   " : "Quadratic", n, lo, hi);
    }
  const double v_twap = dp.policy_value(twap), gain = dp.expected_start_value() - v_twap;
  for (std::uint64_t seed = 1; seed <= 3; ++seed) {
    ModelBasedExecutionAgent a(p, {.cost_model = CostModel::Shrunk, .seed = seed}); ExecutionEnv e(p); Rng rng(seed); double cum = 0;
    double at1000 = 0;
    for (unsigned i = 0; i < 20000; ++i) {
      auto o = e.reset(rng);
      for (;;) { const Action act = a.act(o, true); a.note_action(o, act); const auto st = e.step(act, rng);
                 a.observe(o, act, st.reward, st.obs, st.done); cum += st.reward; o = st.obs; if (st.done) break; }
      a.end_episode(); cum -= v_twap;
      if (i + 1 == 1000) at1000 = cum / gain;
    }
    std::printf("online seed %llu: %+.0f of 1000, %+.0f of 20000\n", (unsigned long long)seed, at1000, cum / gain);
  }
}
