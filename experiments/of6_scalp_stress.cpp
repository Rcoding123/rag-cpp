// OF6: stress-test saved scalping agents (OF3, saved with of3_rl ... save_path;
// or of_train's DIR/seedK.net) in markets that differ from the one they
// learned in, against the best rule (30 s delta momentum) and flat, on 200
// sealed episodes per scenario (seeds 920M+).
// Usage: of6_scalp_stress [--market default|efficient|discovery] [--warmup S] NET [NET...]. Prints CSV:
//   scenario,strategy,ticks_per_ep,t,orders,win
// (with --warmup, a first line "# warm-up S s ..." precedes the CSV).
// Scenarios (rag/orderflow/scalp_run.hpp, stress_scenarios(); the measured
// effect of each new one is noted there):
//   legacy (every market but discovery; OF6's committed results, unchanged):
//     nominal, latency 300 ms, latency 1 s, fees $3/side, informed flow x0.5,
//     uninformed flow x1.5, volatility x1.5, quote fading x2, session open;
//   --market discovery (v3; fixed in code before any v3 RL network existed,
//   OF10 iteration 3, pre-registered): nominal, latency 300 ms, latency 1 s,
//     fees $3/side, uninformed flow x1.5, slower discovery x0.5
//     (reprice_cancel and reprice_take x0.5), flow impact x1.5 (value_impact),
//     news x3 (jump_rate), informed meta-orders x4 (informed_meta_rate),
//     thinner book x0.5 (maker_rate), session open. (Legacy quote fading x2
//     is a no-op there, fade = 0; volatility x1.5 and informed flow x0.5
//     barely move it; informed x2, measured, was too weak.)
// --warmup S (default 150 s): the episodes' warm-up (ScalpParams::
// warmup_seconds), so networks trained with of_train --warmup S are stressed
// on the episodes they learned on; random episodes then start >= S s into the
// session (ScalpEnv::draw_market). "Session open" is a cold open whatever S:
// the simulator starts at the open with the default 150 s burn, the episode
// at 150 s (scalp_run.hpp, session_open). A network whose of_train record
// (DIR/seedK.txt) names another warm-up or market is refused (exit 2), and so
// is a network with no record (OF3, older of_train: trained at 150 s) when S
// is not 150.
#include <cmath>
#include <cstdio>
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
  std::vector<std::string> nets(argv + 1, argv + argc);
  SimParams market;  // the scenarios change this market (default unless --market NAME comes first)
  std::string market_name = "default";
  double warmup = kDefaultWarmup;
  bool warmup_given = false;
  while (nets.size() >= 2 && (nets[0] == "--market" || nets[0] == "--warmup")) {
    if (nets[0] == "--market") {
      try {
        market = market_preset(nets[1]);
        market_name = nets[1];
      } catch (const std::invalid_argument& e) {
        std::fprintf(stderr, "%s\n", e.what());
        return 2;
      }
    } else {
      const auto w = parse_warmup(nets[1]);
      if (!w) {
        std::fprintf(stderr, "--warmup wants a number of seconds >= 0, got %s\n", nets[1].c_str());
        return 2;
      }
      warmup = *w;
      warmup_given = true;
    }
    nets.erase(nets.begin(), nets.begin() + 2);
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
  for (const auto& path : nets) {
    if (path.rfind("--", 0) == 0) {  // options come first, each with a value
      std::fprintf(stderr, "%s: unknown option or missing value (options go before the networks)\n", path.c_str());
      return 2;
    }
    if (const std::string why = check_net(path, market_name, warmup); !why.empty()) {
      std::fprintf(stderr, "%s\n", why.c_str());
      return 2;
    }
  }
  if (warmup_given) std::printf("# warm-up %g s before each episode (--warmup), market %s\n", warmup, market_name.c_str());
  const std::vector<StressScenario> scenarios = stress_scenarios(market_name == "discovery");
  for (const auto& sc : scenarios) {
    ScalpParams p;
    p.sim = market;
    p.warmup_seconds = warmup;
    sc.change(p);
    auto run = [&](const std::string& name, const std::function<Action(const ScalpObs&)>& policy) {
      std::vector<double> pnl;
      double orders = 0, wins = 0;
      for (unsigned i = 0; i < 200; ++i) {
        ScalpEnv env(p);
        Rng r(920'000'000ull + i);  // sealed stress episodes
        auto o = env.reset(r);
        for (;;) {
          const auto st = env.step(policy(o), r);
          o = st.obs;
          if (st.done) break;
        }
        pnl.push_back(env.net_pnl_ticks());
        orders += env.orders();
        wins += env.net_pnl_ticks() > 0;
      }
      double m = 0, v = 0;
      for (double x : pnl) m += x;
      m /= static_cast<double>(pnl.size());
      for (double x : pnl) v += (x - m) * (x - m);
      const double se = std::sqrt(v / static_cast<double>(pnl.size() - 1) / static_cast<double>(pnl.size()));
      std::printf("%s,%s,%.3f,%.2f,%.1f,%.1f\n", sc.name.c_str(), name.c_str(), m, se > 0 ? m / se : 0.0, orders / 200, 100 * wins / 200);
      std::fflush(stdout);
    };
    run("flat", [](const ScalpObs&) -> Action { return 1; });
    run("delta 30 s rule", [](const ScalpObs& o) -> Action { return o.f[1] > 1.5 ? 2 : o.f[1] < -1.5 ? 0 : static_cast<Action>(o.position + 1); });
    for (const auto& path : nets) {
      ScalpEnv env(p);
      AgentConfig<ScalpEnv::kStateDims> c;
      c.learner = LearnerKind::NeuralQ;
      c.bins.fill(1);
      c.double_q = true;
      c.nn_polyak = 1e-4;
      auto agent = std::make_unique<Agent<ScalpEnv>>(env, c);
      auto snap = agent->snapshot();
      snap.net = Mlp::load(path);
      agent->restore(snap);
      run("RL " + path.substr(path.find_last_of('/') + 1), [&](const ScalpObs& o) { return agent->act(o, false); });
    }
  }
}
