// OF4: tick-scalping baselines on TickScalpEnv (1-lot quotes at the touch,
// inventory +-1, 0.5 s decisions, $1.50/side), on sealed 5-minute test
// episodes. Per strategy: net P&L per episode (ticks, $), t, fills per
// episode, share of winning episodes, and the gross spread capture vs the
// adverse selection it suffered (P&L before fees).
// Usage: of4_tick_baselines [episodes=300] [market: default | efficient | discovery | discovery_v1]
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <stdexcept>
#include <string>
#include <vector>

#include "rag/orderflow/tick_env.hpp"

using namespace rag::orderflow;
using rag::agent::Action;

int main(int argc, char** argv) {
  const unsigned episodes = argc > 1 ? static_cast<unsigned>(std::atoi(argv[1])) : 300;
  TickParams p;
  try {
    if (argc > 2) p.sim = market_preset(argv[2]);
  } catch (const std::invalid_argument& e) {
    std::fprintf(stderr, "%s\n", e.what());
    return 2;
  }
  auto skew = [](std::size_t k, double th) {
    return [=](const TickObs& o) -> Action { return o.f[k] > th ? 1 : o.f[k] < -th ? 2 : 3; };
  };
  std::vector<std::pair<std::string, std::function<Action(const TickObs&)>>> strategies = {
      {"no quotes", [](const TickObs&) -> Action { return 0; }},
      {"quote both sides", [](const TickObs&) -> Action { return 3; }},
      {"both, stop at -2 ticks", [](const TickObs& o) -> Action { return o.open_pnl <= -2 ? 4 : 3; }},
      {"skew on imbalance L1 0.3", skew(6, 0.6)},
      {"skew on imbalance L1 0.6", skew(6, 1.2)},
      {"skew on OFI 5 s > 1", skew(3, 1.0)},
      {"skew on delta 5 s > 1", skew(0, 1.0)},
      {"imbalance-skew + stop", [&](const TickObs& o) -> Action { return o.open_pnl <= -2 ? 4 : (o.f[6] > 0.6 ? 1 : o.f[6] < -0.6 ? 2 : 3); }},
  };
  std::printf("%-26s %10s %9s %7s %7s %8s %6s\n", "strategy", "ticks/ep", "$/ep", "t", "fills", "gross", "win%");
  for (auto& [name, policy] : strategies) {
    std::vector<double> pnl;
    double fills = 0, gross = 0;
    for (unsigned i = 0; i < episodes; ++i) {
      TickScalpEnv env(p);
      rag::agent::Rng r(910'000'000ull + i);  // sealed tick-scalping test episodes
      auto o = env.reset(r);
      for (;;) {
        const auto st = env.step(policy(o), r);
        o = st.obs;
        if (st.done) break;
      }
      pnl.push_back(env.net_pnl_ticks());
      fills += env.fills();
      gross += env.net_pnl_ticks() + env.fees_ticks();
    }
    double m = 0, v = 0, wins = 0;
    for (double x : pnl) { m += x; wins += x > 0; }
    m /= static_cast<double>(pnl.size());
    for (double x : pnl) v += (x - m) * (x - m);
    const double sd = std::sqrt(v / static_cast<double>(pnl.size() - 1));
    std::printf("%-26s %+10.2f %+9.1f %+7.2f %7.1f %+8.2f %5.1f%%\n", name.c_str(), m, m * kTickValue, sd > 0 ? m / (sd / std::sqrt(pnl.size())) : 0.0,
                fills / episodes, gross / episodes, 100 * wins / episodes);
    std::fflush(stdout);
  }
}
