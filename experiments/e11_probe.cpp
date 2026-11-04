// E11: cheaper system identification. Online gain over TWAP (in optimal-episodes) and final share.
#include <cstdio>
#include "rag/agent/model_based.hpp"
#include "rag/agent/trainer.hpp"
using namespace rag::agent;
int main() {
  auto twap = [](const ExecutionObs&) { return ExecutionEnv::kTwapAction; };
  struct Case { const char* name; ExecutionParams p; };
  ExecutionParams base, hard = base, perm = base;
  hard.impact_exponent = 3.0; hard.impact = 0.01 * 25 / 125.0; hard.fixed_cost = 1.5; perm.permanent_impact = 0.01;
  struct V { const char* name; unsigned explore; Action lo, hi; double eps; };
  const V vs[] = {{"eps 0.05 (default)", 10, 0, 4, 0.05}, {"eps 0.02", 10, 0, 4, 0.02}, {"eps 0.01", 10, 0, 4, 0.01},
                  {"eps 0 (exploit after 10)", 10, 0, 4, 0.0}, {"eps 0, 3 explore eps", 3, 0, 4, 0.0}};
  for (const Case& c : {Case{"default", base}, Case{"q^3 + fixed", hard}, Case{"permanent 0.01", perm}}) {
    const ExecutionEnv env0(c.p); const ExecutionDP dp(c.p);
    const double v_twap = dp.policy_value(twap), gain = dp.expected_start_value() - v_twap;
    const double opt = compare(env0, [&](const ExecutionObs& o) { return dp.act(o); }, twap, 1000, 5'000'000).mean;
    std::printf("%s\n", c.name);
    for (const V& v : vs) {
      double g10 = 0, g100 = 0, g1000 = 0, sh = 0, sh50 = 0;
      for (std::uint64_t seed = 1; seed <= 5; ++seed) {
        ModelBasedExecutionAgent a(c.p, {.explore_episodes = v.explore, .eps = v.eps, .explore_lo = v.lo, .explore_hi = v.hi,
                                         .cost_model = CostModel::Shrunk, .seed = seed});
        ExecutionEnv e(c.p); Rng rng(seed); double cum = 0;
        for (unsigned i = 0; i < 1000; ++i) {
          auto o = e.reset(rng);
          for (;;) { const Action act = a.act(o, true); a.note_action(o, act); const auto st = e.step(act, rng);
                     a.observe(o, act, st.reward, st.obs, st.done); cum += st.reward; o = st.obs; if (st.done) break; }
          a.end_episode(); cum -= v_twap;
          if (i + 1 == 10) g10 += cum / gain / 5;
          if (i + 1 == 100) g100 += cum / gain / 5;
          if (i + 1 == 50) sh50 += 100 * compare(env0, [&](const ExecutionObs& o) { return a.act(o, false); }, twap, 600, 5000000).mean / opt / 5;
        }
        g1000 += cum / gain / 5;
        sh += 100 * compare(env0, [&](const ExecutionObs& o) { return a.act(o, false); }, twap, 1000, 5'000'000).mean / opt / 5;
      }
      std::printf("  %-34s online after 10: %+7.1f  100: %+7.1f  1000: %+7.1f | share at 50: %5.1f%%  at 1000: %5.1f%%\n", v.name, g10, g100, g1000, sh50, sh);
    }
  }
}
