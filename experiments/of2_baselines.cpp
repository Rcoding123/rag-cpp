// OF2: baseline order-flow strategies on ScalpEnv (1 ES contract, 1 s
// decisions, 100 ms latency, $1.50/side fees), on sealed test episodes
// (10 minutes each, fresh market each). Per strategy: mean net P&L per
// episode in ticks and dollars, its t-statistic, per-episode Sharpe,
// orders per episode, share of winning episodes.
// The oracle sees the hidden fair value: an upper reference, not a strategy.
// Usage: of2_baselines [episodes=300]
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <string>
#include <vector>

#include "rag/orderflow/scalp_env.hpp"

using namespace rag::orderflow;
using rag::agent::Action;

int main(int argc, char** argv) {
  const unsigned episodes = argc > 1 ? static_cast<unsigned>(std::atoi(argv[1])) : 300;
  const ScalpParams p;
  auto hold = [](const ScalpObs& o) { return static_cast<Action>(o.position + 1); };
  auto signal_rule = [&](std::size_t k, double th, bool reverse) {
    return [=](const ScalpObs& o, const ScalpEnv&) -> Action {
      const double x = reverse ? -o.f[k] : o.f[k];
      return x > th ? 2 : x < -th ? 0 : hold(o);
    };
  };
  std::vector<std::pair<std::string, std::function<Action(const ScalpObs&, const ScalpEnv&)>>> strategies = {
      {"flat", [](const ScalpObs&, const ScalpEnv&) -> Action { return 1; }},
      {"random (1/s)", [](const ScalpObs& o, const ScalpEnv& e) -> Action { return static_cast<Action>((e.market().t_ns() / 1'000'000'000 + o.position) % 3); }},
      {"delta 30 s > 1.5", signal_rule(1, 1.5, false)},
      {"delta 30 s > 2.5", signal_rule(1, 2.5, false)},
      {"delta 120 s > 2", signal_rule(2, 2.0, false)},
      {"OFI 30 s > 2", signal_rule(4, 2.0, false)},
      {"imbalance L1 > 0.6", signal_rule(6, 1.2, false)},
      {"absorption > 2", signal_rule(12, 2.0, false)},
      {"fade return 30 s > 2", signal_rule(9, 2.0, true)},
      {"oracle (sees fair value)", [&](const ScalpObs& o, const ScalpEnv& e) -> Action {
         const double gap = e.market().fair_value() - e.market().mid();
         return gap > 2 ? 2 : gap < -2 ? 0 : hold(o);
       }},
  };
  std::printf("%-26s %10s %9s %7s %8s %8s %6s\n", "strategy", "ticks/ep", "$/ep", "t", "Sharpe", "orders", "win%");
  for (auto& [name, policy] : strategies) {
    std::vector<double> pnl;
    double orders = 0;
    for (unsigned i = 0; i < episodes; ++i) {
      ScalpEnv env(p);
      rag::agent::Rng r(900'000'000ull + i);  // sealed test episodes
      auto o = env.reset(r);
      for (;;) {
        const auto st = env.step(policy(o, env), r);
        o = st.obs;
        if (st.done) break;
      }
      pnl.push_back(env.net_pnl_ticks());
      orders += env.orders();
    }
    double m = 0, v = 0, wins = 0;
    for (double x : pnl) { m += x; wins += x > 0; }
    m /= pnl.size();
    for (double x : pnl) v += (x - m) * (x - m);
    const double sd = std::sqrt(v / (pnl.size() - 1));
    std::printf("%-26s %+10.2f %+9.1f %+7.2f %+8.3f %8.1f %5.1f%%\n", name.c_str(), m, m * kTickValue, m / (sd / std::sqrt(pnl.size())), m / sd,
                orders / episodes, 100 * wins / episodes);
    std::fflush(stdout);
  }
}
