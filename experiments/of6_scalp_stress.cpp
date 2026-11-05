// OF6: stress-test saved scalping agents (OF3, saved with of3_rl ... save_path)
// in markets that differ from the one they learned in, against the best rule
// (30 s delta momentum) and flat, on 200 sealed episodes per scenario.
// Usage: of6_scalp_stress NET [NET...]. Prints CSV:
//   scenario,strategy,ticks_per_ep,t,orders,win
#include <cmath>
#include <cstdio>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "rag/agent/agent.hpp"
#include "rag/orderflow/scalp_env.hpp"

using namespace rag::agent;
using namespace rag::orderflow;

int main(int argc, char** argv) {
  std::vector<std::string> nets(argv + 1, argv + argc);
  struct Sc {
    std::string name;
    std::function<void(ScalpParams&)> change;
  };
  const std::vector<Sc> scenarios = {
      {"nominal", [](ScalpParams&) {}},
      {"latency 300 ms", [](ScalpParams& p) { p.latency_steps = 3; }},
      {"latency 1 s", [](ScalpParams& p) { p.latency_steps = 10; }},
      {"fees $3/side", [](ScalpParams& p) { p.fee_dollars = 3.0; }},
      {"informed flow x0.5", [](ScalpParams& p) { p.sim.informed_meta_rate *= 0.5; }},
      {"uninformed flow x1.5", [](ScalpParams& p) { p.sim.uninformed_meta_rate *= 1.5; }},
      {"volatility x1.5", [](ScalpParams& p) { p.sim.value_vol *= 1.5; }},
      {"quote fading x2", [](ScalpParams& p) { p.sim.fade *= 2; }},
      {"session open", [](ScalpParams& p) { p.start_second = 0; }},
  };
  for (const auto& sc : scenarios) {
    ScalpParams p;
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
      m /= pnl.size();
      for (double x : pnl) v += (x - m) * (x - m);
      const double se = std::sqrt(v / (pnl.size() - 1) / pnl.size());
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
