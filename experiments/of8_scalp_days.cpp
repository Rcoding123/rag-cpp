// OF8: trading-desk view of the scalping agents. A "day" is 39 sealed
// 10-minute sessions at successive times of day (open to close; each a fresh
// simulated market), so the intraday activity pattern is included. Reports
// daily net P&L (ticks, $ per contract), share of winning days, worst day,
// maximum drawdown of cumulative daily P&L, and the annualized Sharpe ratio
// (mean / sd of daily P&L x sqrt(252)). Strategies: the delta rule, each saved
// RL agent, and their ensemble (mean of the networks' Q-values, then argmax).
// Usage: of8_scalp_days DAYS NET [NET...]
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "rag/agent/agent.hpp"
#include "rag/orderflow/scalp_env.hpp"

using namespace rag::agent;
using namespace rag::orderflow;

int main(int argc, char** argv) {
  if (argc < 3) return 2;
  const unsigned days = static_cast<unsigned>(std::atoi(argv[1]));
  std::vector<std::unique_ptr<Agent<ScalpEnv>>> agents;
  ScalpEnv proto;
  for (int i = 2; i < argc; ++i) {
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
        p.start_second = k * 600.0;
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
    m /= daily.size();
    for (double x : daily) v += (x - m) * (x - m);
    const double sd = std::sqrt(v / (daily.size() - 1));
    std::printf("%-20s %+9.1f %+9.0f %7.1f %7.0f%% %+10.1f %10.1f %7.2f\n", name.c_str(), m, m * kTickValue, sd, 100 * wins / daily.size(),
                *std::min_element(daily.begin(), daily.end()), dd, sd > 0 ? m / sd * std::sqrt(252.0) : 0.0);
    std::fflush(stdout);
  }
}
