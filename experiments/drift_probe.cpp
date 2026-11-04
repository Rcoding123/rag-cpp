#include <cstdio>
#include <cstdlib>
#include "rag/agent/execution_dp.hpp"
#include "rag/agent/trainer.hpp"
using namespace rag::agent;
int main(int argc, char** argv) {
  const bool detect = argc > 1 && std::atoi(argv[1]) != 0;
  const int before = 30000, after = argc > 2 ? std::atoi(argv[2]) : 30000;
  ExecutionParams p;
  ExecutionEnv env(p);
  AgentConfig<3> cfg;
  cfg.eps_decay_episodes = 15000;
  cfg.alpha = 0.1; cfg.alpha_decay_episodes = 3000; cfg.alpha_min = 0.003;
  Agent<ExecutionEnv> agent(env, cfg);
  TrainConfig tc;
  tc.eval_every = 2500;
  tc.detect_drift = detect;
  tc.drift_after_episodes = 15000;
  if (argc > 3) tc.forget_on_drift = std::atoi(argv[3]) != 0;
  Trainer<ExecutionEnv> tr(agent, env, tc);
  auto twap = [](const ExecutionObs&) { return ExecutionEnv::kTwapAction; };
  auto greedy = [&](const ExecutionObs& o) { return agent.act(o, false); };
  auto gain = [&](const ExecutionEnv& e, const ExecutionDP& dp) {
    const double g = compare(e, greedy, twap, 2000, 5'000'000).mean;
    const double opt = compare(e, [&](const ExecutionObs& o) { return dp.act(o); }, twap, 2000, 5'000'000).mean;
    return 100 * g / opt;
  };
  ExecutionDP dp1(p);
  auto r1 = tr.run(before);
  std::printf("regime 1 after %d episodes: %.1f%% of optimal gain (alarms: %zu)\n", before, gain(env, dp1), r1.drift_alarms.size());
  env.set_alpha(-p.alpha);  // the signal now predicts the opposite move
  ExecutionParams p2 = p; p2.alpha = -p.alpha;
  ExecutionDP dp2(p2);
  std::printf("right after the flip: %.1f%% of the new optimal gain\n", gain(env, dp2));
  for (int chunk = 0; chunk < after; chunk += 2500) {
    auto r = tr.run(2500);
    std::printf("  +%5d episodes: %6.1f%%  eps %.3f  alpha %.4f %s\n", chunk + 2500, gain(env, dp2), agent.epsilon(), agent.step_size(),
                r.drift_alarms.empty() ? "" : ("ALARM at episode " + std::to_string(r.drift_alarms[0]) + " (" + std::to_string(r.drift_alarms[0] - before) + " after flip)").c_str());
  }
}
