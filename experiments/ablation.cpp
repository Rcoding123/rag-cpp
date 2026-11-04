// Ablation: where does the gap between "fit Q* exactly" (99.5%) and RL (~70%) come from?
//   targets:  exact Q*(s,a)  vs  one sampled return (take a, then follow the optimal policy)
//   coverage: update all 5 actions per visited state  vs  only the action the behaviour policy took
#include <cstdio>
#include <cstdlib>
#include "rag/agent/execution_dp.hpp"
#include "rag/agent/trainer.hpp"
using namespace rag::agent;
int main(int argc, char** argv) {
  ExecutionParams p; ExecutionEnv env(p); const ExecutionDP dp(p);
  auto twap = [](const ExecutionObs&) { return ExecutionEnv::kTwapAction; };
  const double opt = compare(env, [&](const ExecutionObs& o) { return dp.act(o); }, twap, 2000, 5'000'000).mean;
  const int episodes = argc > 1 ? std::atoi(argv[1]) : 40000;
  const double innov = std::sqrt(1 - p.signal_rho * p.signal_rho) * p.signal_sd;
  // One sampled return of (t, I, g, a) followed by the optimal policy.
  auto sample_return = [&](std::uint32_t t, std::uint32_t I, double g, Action a, Rng& r) {
    double G = 0;
    for (;;) {
      const std::uint32_t q = env.lots_for(a, I);
      G -= env.trade_cost(q);
      if (t + 1 >= p.horizon) return G - env.dump_cost(I - q);
      I -= q;
      if (I == 0) return G;
      G += I * (p.alpha * g + p.noise_sd * r.normal());
      g = p.signal_rho * g + innov * r.normal();
      ++t;
      a = dp.act({t, I, g});
    }
  };
  for (int sampled = 0; sampled < 2; ++sampled)
    for (int all = 1; all >= 0; --all) {
      TileCoder<3> coder(ExecutionEnv::kLo, ExecutionEnv::kHi, {8, 8, 8}, 8);
      TileQ q(coder.size(), 8, 5, 0.1);
      Rng rng(1), pol(2), mc(3);
      for (int ep = 0; ep < episodes; ++ep) {
        q.set_alpha(std::max(0.003, 0.1 / (1 + ep / 3000.0)));
        auto o = env.reset(rng);
        for (;;) {
          std::array<std::uint32_t, kMaxTilings> idx;
          coder.encode(env.features(o), idx.data());
          const Action act = pol.uniform() < 0.2 ? pol.below(5) : dp.act(o);
          for (Action a = 0; a < 5; ++a) {
            if (!all && a != act) continue;
            const double target = sampled ? sample_return(o.t, o.inventory, o.signal, a, mc) : dp.q_value(o.t, o.inventory, o.signal, a);
            q.update(idx.data(), a, target);
          }
          const auto st = env.step(act, rng);
          o = st.obs;
          if (st.done) break;
        }
      }
      auto greedy = [&](const ExecutionObs& o) {
        std::array<std::uint32_t, kMaxTilings> idx; coder.encode(env.features(o), idx.data());
        double v[5]; q.q_values(idx.data(), v);
        return static_cast<Action>(std::max_element(v, v + 5) - v);
      };
      std::printf("%-15s targets, %-22s %6d episodes: %5.1f%% of optimal\n", sampled ? "sampled-return" : "exact Q*",
                  all ? "all actions updated," : "taken action only,", episodes, 100 * compare(env, greedy, twap, 2000, 5'000'000).mean / opt);
    }
}
