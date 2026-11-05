// OF2: baseline order-flow strategies on ScalpEnv (1 ES contract, 1 s
// decisions, 100 ms latency, $1.50/side fees), on sealed test episodes
// (10 minutes each, fresh market each). Per strategy: mean net P&L per
// episode in ticks and dollars, its t-statistic, per-episode Sharpe,
// orders per episode, share of winning episodes.
// The oracle sees the hidden fair value: an upper reference, not a strategy.
// With fast price discovery (the discovery market) the gap rarely exceeds
// 2 ticks, so oracles at lower thresholds follow the original rows.
// Rows appended in OF10 v2 (fade and short-horizon rules, a per-lot table)
// leave the earlier rows byte-identical.
// Usage: of2_baselines [episodes=300] [market: default | default_warm | efficient | discovery | discovery_v1 | discovery_v2]
//                      [--warmup S] [--seed0 N]
// --warmup S (after the market; default 150 s, ScalpEnv's): the episodes'
// warm-up (ScalpParams::warmup_seconds; OF10 v3 RL on discovery uses 1050).
// Episodes then start >= S s into the session (ScalpEnv::draw_market). When
// given, a line before the table says so; without it the output is unchanged.
// --seed0 N (OF13): episode i uses Rng(N + i) instead of the sealed test
// episodes 900,000,000 + i (to re-check a simulator on fresh episodes); when
// given, a line before the table says so.
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <stdexcept>
#include <string>
#include <vector>

#include "rag/orderflow/scalp_env.hpp"
#include "rag/orderflow/scalp_run.hpp"

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
  std::uint64_t seed0 = 900'000'000ull;  // sealed test episodes
  bool warmup_given = false, seed0_given = false;
  for (int a = 3; a < argc; a += 2) {
    const std::string k = argv[a];
    if (a + 1 < argc && k == "--warmup" && !warmup_given) {
      if (const auto w = parse_warmup(argv[a + 1])) {
        p.warmup_seconds = *w;
        warmup_given = true;
        continue;
      }
    } else if (a + 1 < argc && k == "--seed0" && !seed0_given) {
      char* end = nullptr;
      seed0 = std::strtoull(argv[a + 1], &end, 10);
      if (end != argv[a + 1] && *end == '\0') {
        seed0_given = true;
        continue;
      }
    }
    std::fprintf(stderr, "usage: of2_baselines [episodes] [market] [--warmup S] [--seed0 N], S a number of seconds >= 0\n");
    return 2;
  }
  if (warmup_given) {
    if (const std::string why = warmup_error(p); !why.empty()) {
      std::fprintf(stderr, "--warmup: %s\n", why.c_str());
      return 2;
    }
    std::printf("warm-up %g s before each episode (--warmup; ScalpEnv's default %g s)\n", p.warmup_seconds, kDefaultWarmup);
  }
  if (seed0_given)
    std::printf("episodes Rng(%llu + i) (--seed0; default: the sealed test episodes 900000000 + i)\n", static_cast<unsigned long long>(seed0));
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
      // Appended (OF10 v2): fade and short-horizon rules, a tighter oracle. Rows above are unchanged.
      {"fade delta 5 s > 1.5", signal_rule(0, 1.5, true)},
      {"fade OFI 5 s > 1.5", signal_rule(3, 1.5, true)},
      {"fade return 5 s > 1", signal_rule(8, 1.0, true)},
      {"fade imbalance L1 > 0.6", signal_rule(6, 1.2, true)},
      {"delta 5 s > 1.5", signal_rule(0, 1.5, false)},
      {"imbalance L1-5 > 0.6", signal_rule(7, 1.2, false)},
      {"oracle (gap > 0.55)", oracle(0.55)},
  };
  // NaN (flat: sd 0) prints as "-nan" with every C library, as in the committed tables.
  auto num = [](int width, int decimals, double x) {
    char b[48];
    if (std::isnan(x)) std::snprintf(b, sizeof b, "%*s", width, "-nan");
    else std::snprintf(b, sizeof b, "%+*.*f", width, decimals, x);
    return std::string(b);
  };
  std::printf("%-26s %10s %9s %7s %8s %8s %6s\n", "strategy", "ticks/ep", "$/ep", "t", "Sharpe", "orders", "win%");
  struct PerLot {
    std::string name;
    double lots, gross;  // per episode
  };
  std::vector<PerLot> per_lot;
  for (auto& [name, policy] : strategies) {
    std::vector<double> pnl;
    double orders = 0, fees = 0;
    for (unsigned i = 0; i < episodes; ++i) {
      ScalpEnv env(p);
      rag::agent::Rng r(seed0 + i);  // default: the sealed test episodes
      auto o = env.reset(r);
      for (;;) {
        const auto st = env.step(policy(o, env), r);
        o = st.obs;
        if (st.done) break;
      }
      pnl.push_back(env.net_pnl_ticks());
      orders += env.orders();
      fees += env.fees_ticks();
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
    const double lots = fees / (p.fee_dollars / kTickValue);  // a +1 -> -1 flip is 2 lots
    per_lot.push_back({name, lots / episodes, (m * n + fees) / episodes});
  }
  // Appended (OF10 v2): per lot, not per order (a flip is 2 lots). gross: before fees,
  // after the spread; edge = gross + 0.5 tick (the half spread: ~0 without information).
  std::printf("\n%-26s %8s %9s %9s\n", "per lot", "lots/ep", "gross/lot", "edge/lot");
  for (const auto& r : per_lot) {
    const double g = r.lots > 0 ? r.gross / r.lots : std::nan("");
    std::printf("%-26s %8.1f %s %s\n", r.name.c_str(), r.lots, num(9, 3, g).c_str(), num(9, 3, g + 0.5).c_str());
  }
}
