// OF3: RL on ScalpEnv. NeuralQ (Double DQN, weight averaging), trained on
// fresh simulated ES sessions, champion selected on validation episodes,
// then scored on the same 300 sealed test episodes as the OF2 baselines.
// Usage: of3_rl SEED EPISODES [gamma=0.97] [n_step=5] [eps_decay=frac of episodes 0.5] [save_path] [market: default | efficient]
// Prints progress and a final CSV line: seed,episodes,ticks_per_ep,t,sharpe,orders,win
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <string>
#include <vector>

#include "rag/agent/trainer.hpp"
#include "rag/orderflow/scalp_env.hpp"

using namespace rag::agent;
using namespace rag::orderflow;

int main(int argc, char** argv) {
  if (argc < 3) return 2;
  const std::uint64_t seed = std::strtoull(argv[1], nullptr, 10);
  const unsigned episodes = static_cast<unsigned>(std::atoi(argv[2]));
  const double gamma = argc > 3 ? std::atof(argv[3]) : 0.97;
  const unsigned nstep = argc > 4 ? static_cast<unsigned>(std::atoi(argv[4])) : 5;
  const double decay = argc > 5 ? std::atof(argv[5]) : 0.5;
  ScalpParams p;
  if (argc > 7 && std::string(argv[7]) == "efficient") p.sim = efficient_market();  // OF9
  ScalpEnv env(p);
  AgentConfig<ScalpEnv::kStateDims> c;
  c.seed = seed;
  c.learner = LearnerKind::NeuralQ;
  c.bins.fill(1);  // the tile grid is unused by NeuralQ; 8 bins in 16 dims would be 8^16 cells
  c.double_q = true;
  c.nn_polyak = 1e-4;
  c.gamma = gamma;
  c.n_step = nstep;
  c.eps_start = 1.0;
  c.eps_end = 0.02;
  c.eps_decay_episodes = std::max(1u, static_cast<unsigned>(decay * episodes));
  auto agent = std::make_unique<Agent<ScalpEnv>>(env, c);
  Trainer<ScalpEnv> tr(*agent, env,
                       {.eval_every = std::max(1u, episodes / 20), .eval_episodes = 40, .eval_seed = 700'000'000, .train_seed = seed, .detect_drift = false});
  for (unsigned done = 0; done < episodes;) {
    const unsigned chunk = std::min(episodes - done, std::max(1u, episodes / 10));
    const auto rep = tr.run(chunk);
    done += chunk;
    double mean_train = 0;
    for (double x : rep.returns) mean_train += x;
    std::fprintf(stderr, "seed %llu: %u episodes, training %.1f ticks/ep, deployed validation %.2f ticks/ep\n", (unsigned long long)seed, done,
                 mean_train / std::max<std::size_t>(1, rep.returns.size()), rep.champion_score);
  }
  if (argc > 6 && tr.champion()) tr.champion()->net.save(argv[6]);  // the deployed network, for stress tests
  std::vector<double> pnl;
  double orders = 0;
  for (unsigned i = 0; i < 300; ++i) {
    ScalpEnv test(p);
    Rng r(900'000'000ull + i);  // the sealed OF2 test episodes
    auto o = test.reset(r);
    for (;;) {
      const auto st = test.step(tr.deployed_act(o), r);
      o = st.obs;
      if (st.done) break;
    }
    pnl.push_back(test.net_pnl_ticks());
    orders += test.orders();
  }
  double m = 0, v = 0, wins = 0;
  for (double x : pnl) { m += x; wins += x > 0; }
  m /= pnl.size();
  for (double x : pnl) v += (x - m) * (x - m);
  const double sd = std::sqrt(v / (pnl.size() - 1));
  std::printf("%llu,%u,%.3f,%.2f,%.3f,%.1f,%.1f\n", (unsigned long long)seed, episodes, m, m / (sd / std::sqrt(pnl.size())), m / sd, orders / 300,
              100 * wins / 300);
}
