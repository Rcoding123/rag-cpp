#include <cstdio>
#include <vector>

#include "rag/agent/execution_dp.hpp"
#include "rag/agent/market_making.hpp"
#include "rag/agent/model_based.hpp"
#include "rag/agent/ope.hpp"
#include "rag/agent/trainer.hpp"

using namespace rag::agent;

int main() {
  // 1. Market making at the cap: tight bid, wide ask drives inventory to +Q.
  for (auto [H, Q] : {std::pair{1u, 1}, std::pair{5u, 1}, std::pair{50u, 10}, std::pair{30u, 2}}) {
    MarketMakingParams p;
    p.horizon = H;
    p.max_inventory = Q;
    MarketMakingEnv env(p);
    MarketMakingDP dp(p);
    auto skew = [](const MarketMakingObs&) { return MarketMakingEnv::action(0, 2); };
    auto opt = [&](const MarketMakingObs& o) { return dp.act(o); };
    auto s1 = evaluate(env, skew, 40000, 11);
    auto s2 = evaluate(env, opt, 40000, 11);
    std::printf("MM H=%u Q=%d skew: dp %.4f sim %.4f +- %.4f | opt: dp %.4f sim %.4f +- %.4f\n", H, Q, dp.policy_value(skew),
                s1.mean, s1.std_error, dp.expected_start_value(), s2.mean, s2.std_error);
  }
  // 2. Execution with lots < horizon, horizon 1.
  for (auto [L, H] : {std::pair{5u, 20u}, std::pair{3u, 7u}, std::pair{100u, 1u}, std::pair{1u, 1u}, std::pair{2u, 3u}}) {
    ExecutionParams p;
    p.lots = L;
    p.horizon = H;
    ExecutionEnv env(p);
    ExecutionDP dp(p, 61);
    auto twap = [](const ExecutionObs&) { return ExecutionEnv::kTwapAction; };
    auto opt = [&](const ExecutionObs& o) { return dp.act(o); };
    auto s1 = evaluate(env, twap, 40000, 11);
    auto s2 = evaluate(env, opt, 40000, 11);
    std::printf("EX L=%u H=%u twap: dp %.5f sim %.5f +- %.5f | opt: dp %.5f sim %.5f +- %.5f\n", L, H, dp.policy_value(twap),
                s1.mean, s1.std_error, dp.expected_start_value(), s2.mean, s2.std_error);
  }
  // 3. On-policy OPE with early termination: weights all 1 (propensity 1), PDIS == mean return.
  {
    ExecutionParams p;
    ExecutionEnv env(p);
    Rng rng(3);
    std::vector<LoggedEpisode<ExecutionObs>> logs(200);
    double mean = 0;
    auto pi = [](const ExecutionObs& o) { return o.t < 4 ? Action{4} : Action{2}; };  // sells out early
    for (auto& ep : logs) {
      auto o = env.reset(rng);
      double tot = 0;
      for (;;) {
        const Action a = pi(o);
        const auto st = env.step(a, rng);
        ep.push_back({o, a, st.reward, 1.0});
        tot += st.reward;
        o = st.obs;
        if (st.done) break;
      }
      mean += tot / 200;
    }
    auto e = off_policy_estimates(logs, pi, [](const ExecutionObs&, Action) { return 0.0; }, [](const ExecutionObs&) { return 0.0; },
                                  p.horizon);
    std::printf("OPE on-policy: len %zu mean %.6f pdis %.6f wpdis %.6f dr %.6f ess %.3f unc %u\n", logs[0].size(), mean, e.pdis, e.wpdis,
                e.dr, e.min_ess, e.uncovered_steps);
  }
  // 4. cvar truncation.
  {
    std::vector<double> r;
    for (int i = 1; i <= 100; ++i) r.push_back(i);
    std::printf("cvar(1..100, 0.29) = %.4f (expected mean of 1..29 = 15)\n", cvar(r, 0.29));
    std::printf("0.29*100 = %.17g\n", 0.29 * 100.0);
  }
  // 5. Replan schedule.
  {
    ModelBasedExecutionAgent a(ExecutionParams{}, {.explore_episodes = 10, .replan_every = 20});
    ExecutionEnv env{ExecutionParams{}};
    Rng rng(1);
    for (int ep = 1; ep <= 80; ++ep) {
      const auto before = a.replans();
      train_model_based(a, env, 1, rng);
      if (a.replans() != before) std::printf("replan after episode %d\n", ep);
    }
  }
  return 0;
}
