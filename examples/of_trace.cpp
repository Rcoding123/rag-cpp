// of_trace — step-by-step trace of saved scalping agents, for visualization
// (tools/of_viz.py turns the CSV into one self-contained HTML page).
//
//   of_trace NET [NET...] [--market default|efficient|discovery] [--episodes 20]
//            [--seed 960000000] [--warmup 150] [--out of_trace.csv]
//
// --warmup S (default 150 s): the episodes' warm-up (ScalpParams::
// warmup_seconds), so networks trained with of_train --warmup S are replayed
// on the episodes they learned on. A network whose of_train record
// (DIR/seedK.txt) names another warm-up or market is refused (exit 2), and so
// is a network with no record (trained at 150 s) when S is not 150
// (scalp_run.hpp, check_net). With --warmup the CSV header records it.
//
// Runs ScalpEnv episodes with the saved networks (DIR/seedK.net from
// of_train; several nets act as their mean-Q ensemble, as in of_train) and,
// on the same episode seeds, the 30 s delta rule and the fair-value oracle
// (gap > 2 ticks, as in OF2). Each strategy has its own environment (our
// orders move the market, so the three paths can drift apart by a tick or
// so); the market columns are the agent's.
//
// One CSV row per 1 s decision. Market columns (time of day: session seconds
// since the 09:30 open; bid, ask, mid, hidden
// fair value, L1 sizes and imbalance), the 13 observation features and the
// Q-values describe the state the agent saw when it chose; position, fill,
// orders, fees and P&L are after the step. The fair value is for display
// only: the agent never sees it (only the oracle does).
//
// Seeds: 960,000,000+ is the visualization block (docs/ai/NEXT_SESSION.md
// section 4): look at these episodes as often as you like, but never select
// or score anything on them.
#include <algorithm>
#include <array>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "rag/agent/agent.hpp"
#include "rag/orderflow/scalp_env.hpp"
#include "rag/orderflow/scalp_run.hpp"

using namespace rag::agent;
using namespace rag::orderflow;

namespace {

constexpr std::uint64_t kVizFirst = 960'000'000, kVizEnd = 970'000'000;
constexpr double kOracleGap = 2.0;  // ticks, OF2's "oracle (sees fair value)"
constexpr std::array<const char*, kFeatures> kFeatureNames = {"f_delta5",  "f_delta30", "f_delta120", "f_ofi5",  "f_ofi30",
                                                              "f_relvol",  "f_imb_l1",  "f_imb_l5",   "f_ret5",  "f_ret30",
                                                              "f_ret120",  "f_vwap",    "f_absorb"};

Action hold(const ScalpObs& o) { return static_cast<Action>(o.position + 1); }
Action delta_rule(const ScalpObs& o) { return o.f[1] > 1.5 ? 2 : o.f[1] < -1.5 ? 0 : hold(o); }
Action oracle(const ScalpObs& o, const ScalpEnv& e) {
  const double gap = e.market().fair_value() - e.market().mid();
  return gap > kOracleGap ? 2 : gap < -kOracleGap ? 0 : hold(o);
}

// Strategy cash (ticks) recovered from the environment's public statistics:
// equity = cash + position * mid, net P&L = equity - fees.
double cash(const ScalpEnv& e, int position) { return e.net_pnl_ticks() + e.fees_ticks() - position * e.market().mid(); }

std::string fmt(double x, const char* f = "%.6g") {
  if (!std::isfinite(x)) return "";
  char b[48];
  std::snprintf(b, sizeof b, f, x);
  return b;
}

}  // namespace

int main(int argc, char** argv) {
  std::vector<std::string> nets;
  std::string market = "default", out = "of_trace.csv";
  unsigned episodes = 20;
  std::uint64_t first_seed = kVizFirst;
  double warmup = kDefaultWarmup;
  bool warmup_given = false;
  for (int i = 1; i < argc; ++i) {
    const std::string k = argv[i];
    if (k.rfind("--", 0) != 0) {
      nets.push_back(k);
      continue;
    }
    if (i + 1 >= argc) {
      std::fprintf(stderr, "%s needs a value\n", k.c_str());
      return 2;
    }
    const std::string v = argv[++i];
    if (k == "--market") market = v;
    else if (k == "--out") out = v;
    else if (k == "--warmup") {
      const auto w = parse_warmup(v);
      if (!w) {
        std::fprintf(stderr, "--warmup wants a number of seconds >= 0, got %s\n", v.c_str());
        return 2;
      }
      warmup = *w;
      warmup_given = true;
    }
    else if (k == "--episodes") episodes = static_cast<unsigned>(std::max(1, std::atoi(v.c_str())));
    else if (k == "--seed") {
      char* end = nullptr;
      errno = 0;
      first_seed = std::strtoull(v.c_str(), &end, 10);
      if (v.empty() || v[0] == '-' || *end != '\0' || errno == ERANGE || first_seed < kVizFirst || first_seed >= kVizEnd) {
        std::fprintf(stderr, "--seed wants an integer in the visualization block [960000000, 970000000) (docs/ai/NEXT_SESSION.md), got %s\n",
                     v.c_str());
        return 2;
      }
    } else {
      std::fprintf(stderr, "unknown option %s\n", k.c_str());
      return 2;
    }
  }
  if (nets.empty()) {
    std::fprintf(stderr,
                 "usage: of_trace NET [NET...] [--market default|efficient|discovery] [--episodes 20] [--seed 960000000] [--warmup 150] [--out "
                 "of_trace.csv]\n");
    return 2;
  }
  if (first_seed + episodes > kVizEnd) {
    std::fprintf(stderr, "--episodes runs past the visualization block\n");
    return 2;
  }
  ScalpParams p;
  try {
    p = scalp_params(market, warmup);
  } catch (const std::invalid_argument& e) {
    std::fprintf(stderr, "--market: %s\n", e.what());
    return 2;
  }
  if (const std::string why = warmup_error(p); !why.empty()) {
    std::fprintf(stderr, "--warmup: %s\n", why.c_str());
    return 2;
  }
  for (const auto& path : nets)  // the networks' own market and warm-up (of_train's records)
    if (const std::string why = check_net(path, market, warmup); !why.empty()) {
      std::fprintf(stderr, "%s\n", why.c_str());
      return 2;
    }

  // The agents: of_train's NeuralQ setup with the saved weights (greedy).
  const ScalpEnv probe(p);
  std::vector<std::unique_ptr<Agent<ScalpEnv>>> agents;
  for (const auto& path : nets) {
    AgentConfig<ScalpEnv::kStateDims> c;
    c.learner = LearnerKind::NeuralQ;
    c.bins.fill(1);
    c.double_q = true;
    c.nn_polyak = 1e-4;
    auto a = std::make_unique<Agent<ScalpEnv>>(probe, c);
    auto s = a->snapshot();
    try {
      s.net = Mlp::load(path);
    } catch (const std::exception& e) {
      std::fprintf(stderr, "%s: %s\n", path.c_str(), e.what());
      return 1;
    }
    if (s.net.inputs() != ScalpEnv::kStateDims || s.net.outputs() != ScalpEnv::kNumActions) {
      std::fprintf(stderr, "%s: network has %u inputs and %u outputs, ScalpEnv needs %zu and %zu\n", path.c_str(), s.net.inputs(), s.net.outputs(),
                   ScalpEnv::kStateDims, ScalpEnv::kNumActions);
      return 1;
    }
    a->restore(s);
    agents.push_back(std::move(a));
  }
  auto mean_q = [&](const ScalpObs& o) {
    std::array<double, ScalpEnv::kNumActions> sum{}, q{};
    for (const auto& a : agents) {
      a->q_values(o, q.data());
      for (std::size_t k = 0; k < q.size(); ++k) sum[k] += q[k] / static_cast<double>(agents.size());
    }
    return sum;
  };

  std::ofstream csv(out);
  if (!csv) {
    std::fprintf(stderr, "cannot write %s\n", out.c_str());
    return 1;
  }
  csv << "# of_trace market=" << market << " nets=" << nets.size() << " oracle_gap=" << kOracleGap << " fee_dollars=" << p.fee_dollars
      << " tick_value=" << kTickValue;
  if (warmup_given) csv << " warmup_seconds=" << p.warmup_seconds;  // opt-in: the default header is unchanged
  csv << "\n";
  csv << "episode,seed,step,tod_s,bid,ask,mid,fair_value,bid_sz1,ask_sz1,imb_l1";
  for (const char* n : kFeatureNames) csv << ',' << n;
  csv << ",open_pnl,time_left,q_short,q_flat,q_long,action,pos,fill_px,orders,fees,pnl,rule_pos,rule_pnl,oracle_pos,oracle_pnl\n";

  std::printf("%-8s %10s %10s %10s %8s\n", "episode", "agent", "rule", "oracle", "orders");
  double tot_agent = 0, tot_rule = 0, tot_oracle = 0;
  for (unsigned ep = 0; ep < episodes; ++ep) {
    const std::uint64_t seed = first_seed + ep;
    ScalpEnv ea(p), er(p), eo(p);
    Rng ra(seed), rr(seed), ro(seed);
    // Session second of the simulator's start, drawn as ScalpEnv::reset draws
    // it (the simulator's own clock counts from its construction).
    const double start_second = [&] {
      if (p.start_second >= 0) return p.start_second;
      Rng r(seed);
      return r.uniform() * std::max(0.0, p.sim.session_seconds - p.episode_seconds - p.warmup_seconds);
    }();
    ScalpObs oa = ea.reset(ra), orule = er.reset(rr), oor = eo.reset(ro);
    for (unsigned step = 0;; ++step) {
      const auto& m = ea.market();
      const Quote qt = m.quote();
      const double bsz = qt.bid_size[0], asz = qt.ask_size[0];
      const auto q = mean_q(oa);
      const auto act = static_cast<Action>(std::max_element(q.begin(), q.end()) - q.begin());
      const double cash0 = cash(ea, oa.position);
      const std::uint32_t orders0 = ea.orders();

      std::string row = std::to_string(ep) + ',' + std::to_string(seed) + ',' + std::to_string(step) + ',' +
                        fmt(start_second + static_cast<double>(m.t_ns()) * 1e-9, "%.1f") + ',' + std::to_string(m.bid()) + ',' + std::to_string(m.ask()) + ',' +
                        fmt(m.mid(), "%.1f") + ',' + fmt(m.fair_value(), "%.3f") + ',' + fmt(bsz, "%.0f") + ',' + fmt(asz, "%.0f") + ',' +
                        fmt(bsz + asz > 0 ? (bsz - asz) / (bsz + asz) : 0.0, "%.4f");
      for (double f : oa.f) row += ',' + fmt(f, "%.4f");
      row += ',' + fmt(oa.open_pnl, "%.2f") + ',' + fmt(oa.time_left, "%.4f");
      for (double x : q) row += ',' + fmt(x, "%.4f");

      const auto sa = ea.step(act, ra);
      const auto sr = er.step(delta_rule(orule), rr);
      const auto so = eo.step(oracle(oor, eo), ro);
      // The fill price of this step's market order, recovered from the cash
      // change; empty without an order, or with two (the end-of-episode close
      // after a change: their prices can't be separated).
      double fill = std::nan("");
      const int dpos = sa.obs.position - oa.position;
      if (ea.orders() - orders0 == 1 && dpos != 0) fill = -(cash(ea, sa.obs.position) - cash0) / dpos;
      oa = sa.obs;
      orule = sr.obs;
      oor = so.obs;
      row += ',' + std::to_string(act) + ',' + std::to_string(oa.position) + ',' + fmt(fill, "%.2f") + ',' + std::to_string(ea.orders()) + ',' +
             fmt(ea.fees_ticks(), "%.2f") + ',' + fmt(ea.net_pnl_ticks(), "%.2f") + ',' + std::to_string(orule.position) + ',' +
             fmt(er.net_pnl_ticks(), "%.2f") + ',' + std::to_string(oor.position) + ',' + fmt(eo.net_pnl_ticks(), "%.2f") + '\n';
      csv << row;
      if (sa.done) break;
    }
    tot_agent += ea.net_pnl_ticks();
    tot_rule += er.net_pnl_ticks();
    tot_oracle += eo.net_pnl_ticks();
    std::printf("%-8u %+10.2f %+10.2f %+10.2f %8u\n", ep, ea.net_pnl_ticks(), er.net_pnl_ticks(), eo.net_pnl_ticks(), ea.orders());
  }
  const double n = episodes;
  std::printf("%-8s %+10.2f %+10.2f %+10.2f   ticks/episode, market %s, seeds %llu+ (visualization block: not a score)\n", "mean", tot_agent / n,
              tot_rule / n, tot_oracle / n, market.c_str(), static_cast<unsigned long long>(first_seed));
  std::printf("wrote %s\n", out.c_str());
  return csv.good() ? 0 : 1;
}
