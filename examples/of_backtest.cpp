// of_backtest — walk-forward backtest of order-flow scalping strategies on
// recorded ticks (CSV format: rag/orderflow/tick_csv.hpp). The tape is cut
// into consecutive windows (default 10 minutes, each with a 150 s feature
// warm-up); every strategy trades every window with 1 contract, market
// orders, 100 ms latency, the recorded spread and depth, and fees.
//
//   of_backtest <ticks.csv> [--tick 0.25] [--time ns|us|ms|s] [--window 600] [--fee 1.5] [NET...]
//
// NETs are networks saved by experiments/of3_rl.cpp (ScalpEnv agents).
// Reports per strategy: windows, mean net ticks per window (and $), t-stat,
// Sharpe per window, share of winning windows, total, and the maximum
// drawdown of the cumulative P&L across windows.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "rag/agent/agent.hpp"
#include "rag/orderflow/replay_env.hpp"

using namespace rag::agent;
using namespace rag::orderflow;

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: of_backtest <ticks.csv> [--tick 0.25] [--time ns] [--window 600] [--fee 1.5] [NET...]\n");
    return 2;
  }
  TickCsvOptions co;
  ReplayParams rp;
  std::vector<std::string> nets;
  for (int i = 2; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--tick" && i + 1 < argc) co.tick_size = std::atof(argv[++i]);
    else if (a == "--time" && i + 1 < argc) co.time_unit = argv[++i];
    else if (a == "--window" && i + 1 < argc) rp.window_seconds = std::atof(argv[++i]);
    else if (a == "--fee" && i + 1 < argc) rp.fee_dollars = std::atof(argv[++i]);
    else nets.push_back(a);
  }
  auto tape = std::make_shared<const TickTape>(TickTape::from_csv(argv[1], co));
  const auto& st = tape->stats();
  std::printf("tape: %zu buckets (%.1f h), %zu trades, %zu quotes, %zu lines skipped, %zu gaps\n", tape->size(),
              static_cast<double>(tape->size()) * tape->bucket_seconds() / 3600, st.trades, st.quotes, st.skipped, st.gaps);
  ReplayScalpEnv probe(tape, rp);
  const std::size_t windows = probe.windows();
  std::printf("%zu windows of %.0f s\n", windows, rp.window_seconds);
  if (windows == 0) return 1;

  std::vector<std::unique_ptr<Agent<ReplayScalpEnv>>> agents;
  for (const auto& path : nets) {
    AgentConfig<ReplayScalpEnv::kStateDims> c;
    c.learner = LearnerKind::NeuralQ;
    c.bins.fill(1);
    c.double_q = true;
    c.nn_polyak = 1e-4;
    auto a = std::make_unique<Agent<ReplayScalpEnv>>(probe, c);
    auto s = a->snapshot();
    s.net = Mlp::load(path);
    a->restore(s);
    agents.push_back(std::move(a));
  }
  const auto n_of = [](const std::vector<double>& v) { return static_cast<double>(v.size()); };
  std::vector<std::pair<std::string, std::function<Action(const ScalpObs&)>>> strategies = {
      {"flat", [](const ScalpObs&) -> Action { return 1; }},
      {"delta 30 s rule", [](const ScalpObs& o) -> Action { return o.f[1] > 1.5 ? 2 : o.f[1] < -1.5 ? 0 : static_cast<Action>(o.position + 1); }},
  };
  for (std::size_t i = 0; i < agents.size(); ++i)
    strategies.push_back({nets[i].substr(nets[i].find_last_of("/\\") + 1), [&, i](const ScalpObs& o) { return agents[i]->act(o, false); }});

  std::printf("%-22s %8s %9s %8s %7s %8s %6s %9s %8s\n", "strategy", "windows", "ticks/win", "$/win", "t", "Sharpe", "win%", "total", "max DD");
  for (auto& [name, policy] : strategies) {
    std::vector<double> pnl;
    double orders = 0;
    for (std::size_t w = 0; w < windows; ++w) {
      ReplayScalpEnv env(tape, rp);
      Rng r(1);
      auto o = env.seek(w);
      for (;;) {
        const auto s = env.step(policy(o), r);
        o = s.obs;
        if (s.done) break;
      }
      pnl.push_back(env.net_pnl_ticks());
      orders += env.orders();
    }
    double m = 0, v = 0, wins = 0, cum = 0, peak = 0, dd = 0;
    for (double x : pnl) {
      m += x;
      wins += x > 0;
      cum += x;
      peak = std::max(peak, cum);
      dd = std::max(dd, peak - cum);
    }
    m /= n_of(pnl);
    for (double x : pnl) v += (x - m) * (x - m);
    const double sd = pnl.size() > 1 ? std::sqrt(v / (n_of(pnl) - 1)) : 0;
    std::printf("%-22s %8zu %+9.2f %+8.1f %+7.2f %+8.3f %5.1f%% %+9.1f %8.1f   (%.1f orders/win)\n", name.c_str(), pnl.size(), m, m * kTickValue,
                sd > 0 ? m / (sd / std::sqrt(n_of(pnl))) : 0.0, sd > 0 ? m / sd : 0.0, 100 * wins / n_of(pnl), cum, dd, orders / n_of(pnl));
  }
}
