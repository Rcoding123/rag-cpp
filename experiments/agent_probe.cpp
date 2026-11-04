#include <chrono>
#include <cstdio>
#include <cstdlib>
#include "rag/agent/execution_dp.hpp"
#include "rag/agent/trainer.hpp"
using namespace rag::agent;
int main(int argc, char** argv) {
  const int episodes = argc > 1 ? std::atoi(argv[1]) : 3000;
  const auto kind = argc > 2 && argv[2][0] == 'e' ? LearnerKind::Episodic : LearnerKind::TileQ;
  ExecutionParams p;
  ExecutionEnv env(p);
  ExecutionDP dp(p);
  AgentConfig<3> cfg;
  cfg.learner = kind;
  cfg.eps_decay_episodes = episodes / 2;
  if (argc > 3) cfg.alpha = std::atof(argv[3]);
  if (argc > 4) cfg.tilings = std::atoi(argv[4]);
  if (const char* b = std::getenv("BINS")) { int x, y, z; std::sscanf(b, "%d,%d,%d", &x, &y, &z); cfg.bins = {std::uint32_t(x), std::uint32_t(y), std::uint32_t(z)}; }
  if (const char* r = std::getenv("REPLAY")) cfg.replay_per_step = std::atoi(r);
  if (const char* s = std::getenv("SEED")) cfg.seed = std::atoi(s);
  if (const char* d = std::getenv("ADECAY")) cfg.alpha_decay_episodes = std::atoi(d);
  if (const char* m = std::getenv("AMIN")) cfg.alpha_min = std::atof(m);
  if (const char* q = std::getenv("DQ")) cfg.double_q = std::atoi(q) != 0;
  if (const char* n = std::getenv("NSTEP")) cfg.n_step = std::atoi(n);
  Agent<ExecutionEnv> agent(env, cfg);
  TrainConfig tc;
  tc.eval_every = std::max(1, episodes / 6);
  tc.drift_after_episodes = episodes;  // off for this probe
  tc.eval_episodes = 1000;
  Trainer<ExecutionEnv> tr(agent, env, tc);
  const std::uint64_t seed = 5'000'000;
  auto twap = [](const ExecutionObs&) { return ExecutionEnv::kTwapAction; };
  auto opt = [&](const ExecutionObs& o) { return dp.act(o); };
  const double gap = compare(env, opt, twap, 2000, seed).mean;
  auto t0 = std::chrono::steady_clock::now();
  auto rep = tr.run(episodes, [&](const EvalPoint& e) {
    const auto g = compare(env, [&](const ExecutionObs& o) { return agent.act(o, false); }, twap, 2000, seed);
    std::printf("  ep %6llu eps %.3f heldout %7.3f champ %7.3f %s | vs TWAP %+6.2f +- %.2f = %5.1f%% of optimal gain\n",
                (unsigned long long)e.episode, e.epsilon, e.challenger, e.champion, e.promoted ? "promoted" : "        ",
                g.mean, g.std_error, 100 * g.mean / gap);
  });
  auto t1 = std::chrono::steady_clock::now();
  const auto g = compare(env, [&](const ExecutionObs& o) { return agent.act(o, false); }, twap, 2000, seed);
  std::printf("final (champion) vs TWAP: %+.2f +- %.2f; optimal %+.2f -> %.1f%% captured; train %.1f s\n", g.mean, g.std_error,
              gap, 100 * g.mean / gap, std::chrono::duration<double>(t1 - t0).count());
}
