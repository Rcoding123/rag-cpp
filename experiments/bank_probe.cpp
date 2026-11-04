#include <cstdio>
#include <string>
#include "rag/agent/execution_dp.hpp"
#include "rag/agent/trainer.hpp"
using namespace rag::agent;
int main() {
  ExecutionParams pa, pb; pb.alpha = -1.0;
  const ExecutionDP dpa(pa), dpb(pb);
  auto twap = [](const ExecutionObs&) { return ExecutionEnv::kTwapAction; };
  ExecutionEnv ea(pa), eb(pb);
  const double opt_a = compare(ea, [&](const ExecutionObs& o) { return dpa.act(o); }, twap, 1000, 5'000'000).mean;
  const double opt_b = compare(eb, [&](const ExecutionObs& o) { return dpb.act(o); }, twap, 1000, 5'000'000).mean;
  for (int bank = 1; bank >= 0; --bank) {
    ExecutionEnv env(pa);
    AgentConfig<3> cfg; cfg.eps_decay_episodes = 10000;
    Agent<ExecutionEnv> agent(env, cfg);
    Trainer<ExecutionEnv> tr(agent, env, {.eval_every = 1000, .eval_episodes = 500, .policy_bank = bank != 0});
    tr.run(20000);
    auto share = [&](bool regime_b) {
      const ExecutionEnv& e = regime_b ? eb : ea;
      return 100 * compare(e, [&](const ExecutionObs& o) { return tr.deployed_act(o); }, twap, 1000, 5'000'000).mean / (regime_b ? opt_b : opt_a);
    };
    std::printf("policy bank %s. After 20k episodes in A: %.0f%%\n", bank ? "ON" : "OFF", share(false));
    std::printf("  %-10s %8s %8s %8s   %s\n", "switch to", "+500", "+2000", "+10000", "events");
    for (int sw = 1; sw <= 4; ++sw) {
      const bool b = sw % 2 == 1;
      env.set_alpha(b ? -1.0 : 1.0);
      double s[3]; std::string ev;
      unsigned done = 0, marks[3] = {500, 2000, 10000};
      for (int m = 0; m < 3; ++m) {
        auto rep = tr.run(marks[m] - done); done = marks[m];
        for (auto e : rep.drift_alarms) ev += " drift@+" + std::to_string(e - (20000 + (sw - 1) * 10000));
        for (auto e : rep.bank_recalls) ev += " RECALL@+" + std::to_string(e - (20000 + (sw - 1) * 10000));
        s[m] = share(b);
      }
      std::printf("  %-10s %7.0f%% %7.0f%% %7.0f%%  %s\n", b ? "B" : "A", s[0], s[1], s[2], ev.c_str());
    }
  }
}
