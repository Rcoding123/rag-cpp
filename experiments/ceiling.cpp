// Representation ceiling: fit tile-coded Q to the TRUE optimal Q* (from the DP)
// by supervised regression on states the agent would visit, then score the
// greedy policy of the fit. No RL: if this is low, the representation is the limit.
#include <cstdio>
#include <cstdlib>
#include "rag/agent/execution_dp.hpp"
#include "rag/agent/trainer.hpp"
using namespace rag::agent;
int main(int argc, char** argv) {
  ExecutionParams p; ExecutionEnv env(p); const ExecutionDP dp(p);
  auto twap = [](const ExecutionObs&) { return ExecutionEnv::kTwapAction; };
  const double opt = compare(env, [&](const ExecutionObs& o) { return dp.act(o); }, twap, 2000, 5'000'000).mean;
  struct Cfg { unsigned b0, b1, b2, tilings; };
  for (Cfg c : {Cfg{8, 8, 8, 8}, Cfg{8, 10, 12, 8}, Cfg{12, 12, 12, 8}, Cfg{16, 16, 16, 16}, Cfg{20, 25, 24, 16}}) {
    TileCoder<3> coder(ExecutionEnv::kLo, ExecutionEnv::kHi, {c.b0, c.b1, c.b2}, c.tilings);
    TileQ q(coder.size(), c.tilings, 5, 0.1);
    Rng rng(1), pol(2);
    const int episodes = argc > 1 ? std::atoi(argv[1]) : 200000;
    for (int ep = 0; ep < episodes; ++ep) {
      q.set_alpha(std::max(0.005, 0.1 / (1 + ep / 5000.0)));
      auto o = env.reset(rng);
      for (;;) {  // visit states with a mix of optimal and random actions (like an exploring agent)
        std::array<std::uint32_t, kMaxTilings> idx;
        coder.encode(env.features(o), idx.data());
        for (Action a = 0; a < 5; ++a) q.update(idx.data(), a, dp.q_value(o.t, o.inventory, o.signal, a));
        const Action act = pol.uniform() < 0.2 ? pol.below(5) : dp.act(o);
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
    std::printf("tiles %2ux%2ux%2u x %2u tilings (%6zu weights/action): greedy on fitted Q* = %5.1f%% of optimal\n", c.b0, c.b1, c.b2,
                c.tilings, coder.size(), 100 * compare(env, greedy, twap, 2000, 5'000'000).mean / opt);
  }
}
