// OF2: baseline order-flow strategies on ScalpEnv (1 ES contract, 1 s
// decisions, 100 ms latency, $1.50/side fees), on sealed test episodes
// (10 minutes each, fresh market each). Per strategy: mean net P&L per
// episode in ticks and dollars, its t-statistic, per-episode Sharpe,
// orders per episode, share of winning episodes.
// The oracle sees the hidden fair value: an upper reference, not a strategy.
// With fast price discovery (the discovery market) the gap rarely exceeds
// 2 ticks, so oracles at lower thresholds follow the original rows.
// Usage: of2_baselines [episodes=300] [market: default | efficient | discovery]
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <stdexcept>
#include <string>
#include <vector>

#include "rag/orderflow/scalp_env.hpp"

using namespace rag::orderflow;
using rag::agent::Action;

int main(int argc, char** argv) {
  const unsigned episodes = argc > 1 ? static_cast<unsigned>(std::atoi(argv[1])) : 300;
  ScalpParams p;
  try {
    if (argc > 2) p.sim = market_preset(argv[2]);
  } catch (const std::invalid_argument& e) {
    std::fprintf(stderr, "%s\n", e.what());
    return 2;
  }
  auto hold = [](const ScalpObs& o) { return static_cast<Action>(o.position + 1); };
  auto signal_rule = [&](std::size_t k, double th, bool reverse) {
    return [=](const ScalpObs& o, const ScalpEnv&) -> Action {
      const double x = reverse ? -o.f[k] : o.f[k];
      return x > th ? 2 : x < -th ? 0 : hold(o);
    };
  };
  auto oracle = [&](double th) {
    return [=](const ScalpObs& o, const ScalpEnv& e) -> Action {
      const double gap = e.market().fair_value() - e.market().mid();
      return gap > th ? 2 : gap < -th ? 0 : hold(o);
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
      {"oracle (sees fair value)", oracle(2.0)},
      {"oracle (gap > 0.5)", oracle(0.5)},
      {"oracle (gap > 0.75)", oracle(0.75)},
      {"oracle (gap > 1)", oracle(1.0)},
  };
  // NaN (flat: sd 0) prints as "-nan" with every C library, as in the committed tables.
  auto num = [](int width, int decimals, double x) {
    char b[48];
    if (std::isnan(x)) std::snprintf(b, sizeof b, "%*s", width, "-nan");
    else std::snprintf(b, sizeof b, "%+*.*f", width, decimals, x);
    return std::string(b);
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
    const auto n = static_cast<double>(pnl.size());
    double m = 0, v = 0, wins = 0;
    for (double x : pnl) {
      m += x;
      if (x > 0) wins += 1;
    }
    m /= n;
    for (double x : pnl) v += (x - m) * (x - m);
    const double sd = std::sqrt(v / (n - 1));
    std::printf("%-26s %+10.2f %+9.1f %s %s %8.1f %5.1f%%\n", name.c_str(), m, m * kTickValue, num(7, 2, m / (sd / std::sqrt(n))).c_str(),
                num(8, 3, m / sd).c_str(), orders / episodes, 100 * wins / episodes);
    std::fflush(stdout);
  }
}
