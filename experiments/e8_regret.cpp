// E8: online regret. Every training episode is real; what does learning cost vs running TWAP?
#include <cstdio>
#include <memory>
#include <string>
#include <vector>
#include "rag/agent/model_based.hpp"
#include "rag/agent/trainer.hpp"
using namespace rag::agent;
constexpr unsigned N = 20000;
struct Curve { std::string name; std::vector<double> ret; double final_share; };
int main(int argc, char** argv) {
  const double gamma = argc > 1 ? std::atof(argv[1]) : 0.0;
  auto twap = [](const ExecutionObs&) { return ExecutionEnv::kTwapAction; };
  ExecutionParams p; p.permanent_impact = gamma;
  ExecutionEnv env(p); const ExecutionDP dp(p);
  const double v_star = dp.expected_start_value(), v_twap = dp.policy_value(twap), gain = v_star - v_twap;
  const double paired_opt = compare(env, [&](const ExecutionObs& o) { return dp.act(o); }, twap, 1000, 5'000'000).mean;
  auto share = [&](auto&& pi) { return 100 * compare(env, pi, twap, 1000, 5'000'000).mean / paired_opt; };
  const std::uint64_t seed = argc > 2 ? std::strtoull(argv[2], nullptr, 10) : 1;
  std::printf("gamma %.3f: V* %+.2f  V_twap %+.2f  optimal gain/episode %.2f\n", gamma, v_star, v_twap, gain);
  std::vector<Curve> curves;
  auto model_free = [&](const char* name, AgentConfig<3> cfg) {
    cfg.seed = seed; ExecutionEnv e(p); auto agent = std::make_unique<Agent<ExecutionEnv>>(e, cfg);
    Trainer<ExecutionEnv> tr(*agent, e, {.eval_every = 0, .train_seed = seed, .detect_drift = false});
    auto rep = tr.run(N);
    curves.push_back({name, rep.returns, share([&](const ExecutionObs& o) { return agent->act(o, false); })});
  };
  { AgentConfig<3> c; c.eps_decay_episodes = 10000; model_free("TileQ eps 1->0.02/10k", c); }
  { AgentConfig<3> c; c.eps_start = 0.3; c.eps_decay_episodes = 10000; model_free("TileQ eps 0.3->0.02", c); }
  { AgentConfig<3> c; c.eps_start = 0.1; c.eps_decay_episodes = 10000; model_free("TileQ eps 0.1->0.02", c); }
  { AgentConfig<3> c; c.eps_start = 0.02; c.eps_decay_episodes = 1; model_free("TileQ eps 0.02 const", c); }
  { AgentConfig<3> c; c.eps_start = 0.0; c.eps_end = 0.0; c.eps_decay_episodes = 1; model_free("TileQ eps 0 (greedy)", c); }
  { AgentConfig<3> c; c.learner = LearnerKind::ActorCritic; c.eps_decay_episodes = 10000; model_free("ActorCritic", c); }
  for (int term = 1; term >= 0; --term) {
    if (gamma == 0 && term == 0) continue;
    ModelBasedExecutionAgent a(p, {.cost_model = CostModel::Shrunk, .learn_permanent_impact = term == 1, .seed = seed});
    ExecutionEnv e(p); Rng rng(seed); std::vector<double> ret;
    for (unsigned i = 0; i < N; ++i) {
      auto o = e.reset(rng); double total = 0;
      for (;;) { const Action act = a.act(o, true); a.note_action(o, act); const auto st = e.step(act, rng);
                 a.observe(o, act, st.reward, st.obs, st.done); total += st.reward; o = st.obs; if (st.done) break; }
      a.end_episode(); ret.push_back(total);
    }
    curves.push_back({term ? "Model-based" : "Model-based, no term", ret, share([&](const ExecutionObs& o) { return a.act(o, false); })});
  }
  std::printf("%-24s | cumulative gain over TWAP, in optimal-episodes (higher is better; optimum = episodes)  | break-even | final\n", "");
  std::printf("%-24s | %9s %9s %9s %9s %9s |            |\n", "", "100", "1000", "5000", "10000", "20000");
  for (const auto& c : curves) {
    double cum = 0; long last_nonpos = 0; std::printf("%-24s |", c.name.c_str());
    for (unsigned i = 0; i < N; ++i) {
      cum += c.ret[i] - v_twap;
      if (cum <= 0) last_nonpos = i + 1;
      if (i + 1 == 100 || i + 1 == 1000 || i + 1 == 5000 || i + 1 == 10000 || i + 1 == 20000) std::printf(" %+9.0f", cum / gain);
    }
    std::printf(" | %10s | %4.0f%%\n", last_nonpos == static_cast<long>(N) ? "never" : std::to_string(last_nonpos + 1).c_str(), c.final_share);
  }
}
