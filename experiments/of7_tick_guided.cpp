// OF7: rule-guided RL on TickScalpEnv. OF5's agent learned never to quote:
// its exploration (uniform random actions, eps 1 -> 0.02) cancels a resting
// quote at ~60% of decisions and rejoins it behind ~120 lots, so it is
// almost never filled (measured: 0.10 fills per episode on validation,
// against 8.4 for the delta-skew rule), and the replay holds no experience
// of what quoting is worth.
//
// Here the behaviour policy is mixed, and the agent learns off-policy from
// every step (Agent::observe), whatever chose the action:
//   * guided episodes (probability p_guide, annealed from guide_start to
//     guide_floor over the first guide_frac of training): the rule acts
//     (o.f[0] > 1: bid only, < -1: ask only, else both), and with
//     probability rule_eps per decision a uniformly random action replaces
//     it, so the replay also holds what deviating from the rule costs in the
//     states the rule visits (e.g. cancelling a quote that was near the front);
//   * the other episodes: the agent's own eps-greedy policy.
// Rewards are the environment's (mark-to-market net of fees): no shaping,
// no hidden state. The observation is TickScalpEnv's (no fair value, no
// future information). The champion is chosen on validation episodes
// (seeds 700'000'000 + i, disjoint from the sealed 900M-930M range) and
// scored once on the same 300 sealed episodes as OF4/OF5 (910'000'000 + i),
// paired with the rule on each episode.
//
// Usage: of7_tick_guided SEED EPISODES [gamma=0.99] [guide_frac=0.5] [guide_floor=0.1] [rule_eps=0.02]
//                        [val_episodes=200] [eval_every=0: EPISODES/20] [save_path]
// Prints progress to stderr and a final CSV line to stdout:
//   seed,episodes,val_ticks,val_fills,test_ticks,test_t,test_fills,test_win,rule_test_ticks,diff_vs_rule,diff_t
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <optional>
#include <vector>

#include "rag/agent/agent.hpp"
#include "rag/orderflow/tick_env.hpp"

using namespace rag::agent;
using namespace rag::orderflow;

namespace {

Action rule(const TickObs& o) { return o.f[0] > 1 ? 1 : o.f[0] < -1 ? 2 : 3; }

struct Score {
  double mean = 0, se = 0, fills = 0, win = 0;
  std::vector<double> pnl;
};

template <class Policy>
Score score(const TickParams& p, Policy&& policy, unsigned episodes, std::uint64_t seed) {
  Score s;
  for (unsigned i = 0; i < episodes; ++i) {
    TickScalpEnv env(p);
    Rng r(seed + i);
    auto o = env.reset(r);
    for (;;) {
      const auto st = env.step(policy(o), r);
      o = st.obs;
      if (st.done) break;
    }
    s.pnl.push_back(env.net_pnl_ticks());
    s.fills += env.fills();
  }
  double v = 0;
  for (double x : s.pnl) {
    s.mean += x;
    s.win += x > 0;
  }
  s.mean /= episodes;
  for (double x : s.pnl) v += (x - s.mean) * (x - s.mean);
  s.se = episodes > 1 ? std::sqrt(v / (episodes - 1) / episodes) : 0;
  s.fills /= episodes;
  s.win = 100 * s.win / episodes;
  return s;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 3) {
    std::fprintf(stderr, "usage: of7_tick_guided SEED EPISODES [gamma] [guide_frac] [guide_floor] [rule_eps] [val_episodes] [eval_every] [save_path]\n");
    return 2;
  }
  const std::uint64_t seed = std::strtoull(argv[1], nullptr, 10);
  const unsigned episodes = static_cast<unsigned>(std::atoi(argv[2]));
  const double gamma = argc > 3 ? std::atof(argv[3]) : 0.99;
  const double guide_frac = argc > 4 ? std::atof(argv[4]) : 0.5;
  const double guide_floor = argc > 5 ? std::atof(argv[5]) : 0.1;
  const double rule_eps = argc > 6 ? std::atof(argv[6]) : 0.02;
  const unsigned val_episodes = argc > 7 ? static_cast<unsigned>(std::atoi(argv[7])) : 200;
  const unsigned eval_every = argc > 8 && std::atoi(argv[8]) > 0 ? static_cast<unsigned>(std::atoi(argv[8])) : std::max(1u, episodes / 20);
  const double guide_start = 1.0;
  constexpr std::uint64_t kValSeed = 700'000'000, kTestSeed = 910'000'000;

  const TickParams p;
  TickScalpEnv env(p);
  AgentConfig<TickScalpEnv::kStateDims> c;
  c.seed = seed;
  c.learner = LearnerKind::NeuralQ;
  c.bins.fill(1);  // the tile grid is unused by NeuralQ
  c.double_q = true;
  c.nn_polyak = 1e-4;
  c.gamma = gamma;
  c.n_step = 5;
  // The agent's own episodes explore lightly: the rule's episodes are the
  // broad exploration, and uniform actions at a high rate are what kept
  // OF5's quotes from ever reaching the front of the queue.
  c.eps_start = 0.1;
  c.eps_end = 0.02;
  c.eps_decay_episodes = std::max(1u, static_cast<unsigned>(guide_frac * episodes));
  auto agent = std::make_unique<Agent<TickScalpEnv>>(env, c);

  const Score rule_val = score(p, rule, val_episodes, kValSeed);
  std::fprintf(stderr, "seed %llu: rule on validation %.3f ticks/ep (se %.3f), %.2f fills/ep\n", (unsigned long long)seed, rule_val.mean, rule_val.se,
               rule_val.fills);

  Rng train(seed);                   // training scenarios, one stream (as the Trainer in OF5)
  Rng behave(seed ^ 0xB0B0B0B0ull);             // guidance and the rule's deviations
  std::optional<Agent<TickScalpEnv>::Snapshot> champion;
  Score champ_val;
  unsigned champ_episode = 0;
  double sum_guided = 0, sum_agent = 0, fills_guided = 0, fills_agent = 0;
  unsigned n_guided = 0, n_agent = 0;
  const auto t0 = std::chrono::steady_clock::now();
  for (unsigned ep = 1; ep <= episodes; ++ep) {
    const double f = std::min(1.0, (ep - 1) / std::max(1.0, guide_frac * episodes));
    const double p_guide = guide_start + (guide_floor - guide_start) * f;
    const bool guided = behave.uniform() < p_guide;
    TickScalpEnv e(p);
    auto o = e.reset(train);
    double total = 0;
    for (;;) {
      Action a;
      if (guided)
        a = behave.uniform() < rule_eps ? behave.below(static_cast<std::uint32_t>(TickScalpEnv::kNumActions)) : rule(o);
      else
        a = agent->act(o, /*explore=*/true);
      const auto st = e.step(a, train);
      agent->observe(o, a, st.reward, st.obs, st.done);
      total += st.reward;
      o = st.obs;
      if (st.done) break;
    }
    agent->end_episode();
    (guided ? sum_guided : sum_agent) += total;
    (guided ? fills_guided : fills_agent) += e.fills();
    ++(guided ? n_guided : n_agent);
    if (ep % eval_every == 0 || ep == episodes) {
      const auto snap = agent->snapshot();
      const Score v = score(p, [&](const TickObs& ob) { return agent->act_as(snap, ob); }, val_episodes, kValSeed);
      const bool promote = !champion || v.mean > champ_val.mean;
      if (promote) {
        champion = snap;
        champ_val = v;
        champ_episode = ep;
      }
      const double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
      std::fprintf(stderr,
                   "seed %llu ep %u (%.0f s) p_guide %.2f eps %.3f | train: guided %.2f (%.1f fills, n %u) agent %.2f (%.1f fills, n %u) | "
                   "validation %.3f (se %.3f, %.2f fills)%s | champion %.3f from ep %u\n",
                   (unsigned long long)seed, ep, sec, p_guide, agent->epsilon(), n_guided ? sum_guided / n_guided : 0.0,
                   n_guided ? fills_guided / n_guided : 0.0, n_guided, n_agent ? sum_agent / n_agent : 0.0, n_agent ? fills_agent / n_agent : 0.0,
                   n_agent, v.mean, v.se, v.fills, promote ? " promoted" : "", champ_val.mean, champ_episode);
      sum_guided = sum_agent = fills_guided = fills_agent = 0;
      n_guided = n_agent = 0;
    }
  }
  if (argc > 9 && champion) champion->net.save(argv[9]);

  // Sealed test: the champion and the rule on the same 300 episodes, paired.
  const Score test = score(p, [&](const TickObs& ob) { return agent->act_as(*champion, ob); }, 300, kTestSeed);
  const Score rule_test = score(p, rule, 300, kTestSeed);
  double dm = 0, dv = 0;
  for (std::size_t i = 0; i < test.pnl.size(); ++i) dm += test.pnl[i] - rule_test.pnl[i];
  dm /= static_cast<double>(test.pnl.size());
  for (std::size_t i = 0; i < test.pnl.size(); ++i) dv += std::pow(test.pnl[i] - rule_test.pnl[i] - dm, 2);
  const double dse = std::sqrt(dv / static_cast<double>(test.pnl.size() - 1) / static_cast<double>(test.pnl.size()));
  std::fprintf(stderr, "seed %llu sealed: champion %.3f ticks/ep (t %.2f, %.2f fills, win %.1f%%), rule %.3f (%.2f fills), paired diff %.3f (t %.2f)\n",
               (unsigned long long)seed, test.mean, test.se > 0 ? test.mean / test.se : 0.0, test.fills, test.win, rule_test.mean, rule_test.fills, dm,
               dse > 0 ? dm / dse : 0.0);
  std::printf("%llu,%u,%.3f,%.2f,%.3f,%.2f,%.2f,%.1f,%.3f,%.3f,%.2f\n", (unsigned long long)seed, episodes, champ_val.mean, champ_val.fills, test.mean,
              test.se > 0 ? test.mean / test.se : 0.0, test.fills, test.win, rule_test.mean, dm, dse > 0 ? dm / dse : 0.0);
}
