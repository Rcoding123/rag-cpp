// OF8: trading-desk view of the scalping agents. A "day" is 39 sealed
// 10-minute sessions at successive times of day (open to close; each a fresh
// simulated market), so the intraday activity pattern is included. Reports
// daily net P&L (ticks, $ per contract), share of winning days, worst day,
// maximum drawdown of cumulative daily P&L, and the annualized Sharpe ratio
// (mean / sd of daily P&L x sqrt(252)). Strategies: the delta rule, each saved
// RL agent, and their ensemble (mean of the networks' Q-values, then argmax).
// Usage: of8_scalp_days DAYS [--market default|efficient|discovery] [--warmup S] NET [NET...]
// --warmup S (default 150 s): the episodes' warm-up (ScalpParams::
// warmup_seconds), for networks trained with of_train --warmup S (a network
// whose record DIR/seedK.txt names another warm-up or market is refused,
// exit 2; so is one with no record, trained at 150 s, when S is not 150).
// Session k's episode stays at k * 600 + 150 s of the session (where the
// default puts it) and its warm-up starts S s earlier, never before the open:
// with S = 1050 sessions 0 and 1 both run 1050 .. 1650 s (fresh markets,
// different seeds), the rest are unchanged in time of day
// (scalp_run.hpp, day_session_start). With --warmup a first line says so.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "rag/agent/agent.hpp"
#include "rag/orderflow/scalp_env.hpp"
#include "rag/orderflow/scalp_run.hpp"

using namespace rag::agent;
using namespace rag::orderflow;

int main(int argc, char** argv) {
  if (argc < 3) return 2;
  const unsigned days = static_cast<unsigned>(std::atoi(argv[1]));
  SimParams market;
  std::string market_name = "default";
  double warmup = kDefaultWarmup;
  bool warmup_given = false;
  int first_net = 2;
  while (first_net + 1 < argc && (std::string(argv[first_net]) == "--market" || std::string(argv[first_net]) == "--warmup")) {
    if (std::string(argv[first_net]) == "--market") {
      try {
        market = market_preset(argv[first_net + 1]);
        market_name = argv[first_net + 1];
      } catch (const std::invalid_argument& e) {
        std::fprintf(stderr, "%s\n", e.what());
        return 2;
      }
    } else {
      const auto w = parse_warmup(argv[first_net + 1]);
      if (!w) {
        std::fprintf(stderr, "--warmup wants a number of seconds >= 0, got %s\n", argv[first_net + 1]);
        return 2;
      }
      warmup = *w;
      warmup_given = true;
    }
    first_net += 2;
  }
  {
    ScalpParams g;
    g.sim = market;
    g.warmup_seconds = warmup;
    if (const std::string why = warmup_error(g); !why.empty()) {
      std::fprintf(stderr, "--warmup: %s\n", why.c_str());
      return 2;
    }
  }
  for (int i = first_net; i < argc; ++i) {
    if (std::string(argv[i]).rfind("--", 0) == 0) {  // options come first, each with a value
      std::fprintf(stderr, "%s: unknown option or missing value (options go before the networks)\n", argv[i]);
      return 2;
    }
    if (const std::string why = check_net(argv[i], market_name, warmup); !why.empty()) {
      std::fprintf(stderr, "%s\n", why.c_str());
      return 2;
    }
  }
  if (warmup_given) std::printf("warm-up %g s before each episode (--warmup), market %s\n", warmup, market_name.c_str());
  std::vector<std::unique_ptr<Agent<ScalpEnv>>> agents;
  ScalpEnv proto;
  for (int i = first_net; i < argc; ++i) {
    AgentConfig<ScalpEnv::kStateDims> c;
    c.learner = LearnerKind::NeuralQ;
    c.bins.fill(1);
    c.double_q = true;
    c.nn_polyak = 1e-4;
    auto a = std::make_unique<Agent<ScalpEnv>>(proto, c);
    auto s = a->snapshot();
    s.net = Mlp::load(argv[i]);
    a->restore(s);
    agents.push_back(std::move(a));
  }
  std::vector<std::pair<std::string, std::function<Action(const ScalpObs&)>>> strategies;
  strategies.push_back({"delta 30 s rule", [](const ScalpObs& o) -> Action {
                          return o.f[1] > 1.5 ? 2 : o.f[1] < -1.5 ? 0 : static_cast<Action>(o.position + 1);
                        }});
  for (std::size_t i = 0; i < agents.size(); ++i)
    strategies.push_back({"RL net " + std::to_string(i + 1), [&, i](const ScalpObs& o) { return agents[i]->act(o, false); }});
  if (agents.size() > 1)
    strategies.push_back({"RL mean-Q ensemble", [&](const ScalpObs& o) -> Action {
                            std::array<double, ScalpEnv::kNumActions> sum{}, q{};
                            for (auto& a : agents) {
                              a->q_values(o, q.data());
                              for (std::size_t k = 0; k < q.size(); ++k) sum[k] += q[k];
                            }
                            return static_cast<Action>(std::max_element(sum.begin(), sum.end()) - sum.begin());
                          }});
  constexpr unsigned kSessions = 39;  // 39 x 10 min = 6.5 h
  std::printf("%-20s %9s %9s %7s %8s %10s %10s %7s\n", "strategy", "ticks/day", "$/day", "sd", "win days", "worst day", "max DD", "Sharpe");
  for (auto& [name, policy] : strategies) {
    std::vector<double> daily;
    for (unsigned d = 0; d < days; ++d) {
      double day = 0;
      for (unsigned k = 0; k < kSessions; ++k) {
        ScalpParams p;
        p.sim = market;
        p.warmup_seconds = warmup;
        p.start_second = day_session_start(k, warmup);  // k * 600 with the default warm-up
        ScalpEnv env(p);
        Rng r(930'000'000ull + d * 100 + k);  // sealed day episodes
        auto o = env.reset(r);
        for (;;) {
          const auto st = env.step(policy(o), r);
          o = st.obs;
          if (st.done) break;
        }
        day += env.net_pnl_ticks();
      }
      daily.push_back(day);
    }
    double m = 0, v = 0, wins = 0, cum = 0, peak = 0, dd = 0;
    for (double x : daily) {
      m += x;
      wins += x > 0;
      cum += x;
      peak = std::max(peak, cum);
      dd = std::max(dd, peak - cum);
    }
    m /= static_cast<double>(daily.size());
    for (double x : daily) v += (x - m) * (x - m);
    const double sd = std::sqrt(v / static_cast<double>(daily.size() - 1));
    std::printf("%-20s %+9.1f %+9.0f %7.1f %7.0f%% %+10.1f %10.1f %7.2f\n", name.c_str(), m, m * kTickValue, sd, 100 * wins / static_cast<double>(daily.size()),
                *std::min_element(daily.begin(), daily.end()), dd, sd > 0 ? m / sd * std::sqrt(252.0) : 0.0);
    std::fflush(stdout);
  }
}
