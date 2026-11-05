// OF7: rule-guided RL on TickScalpEnv. OF5's agent learned never to quote.
// Measured cause: its exploration (uniform random actions, eps 1 -> 0.02)
// cancels a resting quote at most decisions and rejoins it behind ~120 lots,
// so it is almost never filled (0.10 fills per validation episode, against
// 8.4 for the delta-skew rule): the replay holds no experience of what
// quoting is worth.
//
// Training. The agent learns off-policy from every step (Agent::observe),
// whatever chose the action. An episode is
//   * guided with probability p_guide (annealed from 1 to guide_floor over
//     the first guide_frac of training): the rule acts (o.f[0] > 1: bid
//     only, < -1: ask only, else both), and with probability rule_eps per
//     decision a uniformly random action replaces it, so the replay also
//     holds what deviating from the rule costs in the states the rule visits;
//   * otherwise the agent's own: a uniformly random action with the agent's
//     epsilon (0.1 -> 0.02), else the anchored policy that scored best at
//     the latest evaluation (below; the rule before the first one), on the
//     online network.
// Rewards are the environment's (mark-to-market net of fees): no shaping,
// no hidden state; the observation is TickScalpEnv's (no fair value, no
// future information).
//
// Deployment. The learned Q-values are used in two ways:
//   * greedy: argmax_a Q(s, a), the plain RL policy;
//   * rule-anchored, margin m: the rule's action unless some action's
//     learned value beats it by more than m ticks (conservative policy
//     improvement: rare random deviations from the rule cost ~0.15 ticks
//     each, measured: 2% of decisions take it from +3.05 to +1.23 ticks per
//     episode, so a deviation must be clearly better to be worth taking).
// Every evaluation scores greedy and m in {0.1, 0.3, 1} on validation
// episodes (seeds 700'000'000 + i, disjoint from the sealed 900M-930M
// range); the champion is the best (snapshot, policy) pair, and the greedy
// champion is kept too. Both are scored once on the same 300 sealed
// episodes as OF4/OF5 (910'000'000 + i), paired with the rule per episode.
// Pairing shares the seed, not the path: the market reacts to our orders,
// and one 0.5 s deviation from the rule changes the episode's P&L by sd 3.8
// ticks (6.2 if early; measured on validation), so a policy that deviates
// a few times per episode is compared on nearly independent paths.
//
// Usage: of7_tick_guided SEED EPISODES [gamma=0.99] [guide_frac=0.5] [guide_floor=0.25] [rule_eps=0.02]
//                        [val_episodes=200] [eval_every=0: EPISODES/20] [save_path]
// Prints progress to stderr and final CSV lines to stdout:
//   seed,episodes,policy,val_ticks,val_fills,test_ticks,test_t,test_fills,test_win,deviation_pct,diff_vs_rule,diff_t,val_diff,val_diff_t
// (policy: rule, greedy, or the champion's margin; deviation_pct: sealed
// decisions where the policy did not take the rule's action, by action
// index, so a twin with the same effect (bid only at the long limit vs no
// quotes) counts too: smoke seed-2 champion 1.67% by index, 1.20% in
// effect; diff: paired with the rule on the sealed episodes; val_diff: the
// same on validation, where the champion was selected, so optimistic).
//
// The rule scores +2.90 ticks/ep (t 10.5, 8.3 fills) on these sealed
// episodes with the committed simulator (of4_results.txt, regenerated: the
// table first committed there, +1.66, came from an earlier calibration and
// no commit reproduces it).
//
// Measured (smoke: 2 seeds x 700 episodes, defaults, eval every 50):
//   greedy champions   -0.80 / -0.20 ticks/ep sealed, 4.6 / 8.0 fills: it
//                      quotes now (OF5: 0-0.1 fills) but loses, -3.7 / -3.1
//                      vs the rule (t -9.4 / -10.4);
//   anchored champions +2.58 / +2.57 (t 9.6 / 11.4), 8.3 / 9.0 fills,
//                      deviating at 0.4% / 1.7% of decisions: -0.31 / -0.33
//                      vs the rule (t -1.75 / -1.40), while validation had
//                      them +0.54 / +0.53 ahead (selection on noise).
// A pilot with plain greedy deployment and guidance annealed to 10%
// oscillated between quoting everything (up to 27 fills, -3 ticks/ep) and
// almost nothing; its validation champions scored -0.08 / -0.25 sealed with
// ~1 fill: the learned action gaps are below the Q-estimation noise.
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "rag/agent/agent.hpp"
#include "rag/orderflow/tick_env.hpp"

using namespace rag::agent;
using namespace rag::orderflow;

namespace {

constexpr std::size_t kA = TickScalpEnv::kNumActions;
constexpr double kGreedy = -1;  // "margin" of the plain greedy policy
constexpr std::array<double, 4> kPolicies = {kGreedy, 0.1, 0.3, 1.0};

Action rule(const TickObs& o) { return o.f[0] > 1 ? 1 : o.f[0] < -1 ? 2 : 3; }

Action argmax(const std::array<double, kA>& q) {
  Action best = 0;
  for (Action a = 1; a < kA; ++a)
    if (q[a] > q[best]) best = a;
  return best;
}

// Greedy (margin < 0), or the rule unless the best action beats it by more than `margin`.
Action choose(const std::array<double, kA>& q, const TickObs& o, double margin) {
  const Action best = argmax(q);
  if (margin < 0) return best;
  const Action r = rule(o);
  return q[best] - q[r] > margin ? best : r;
}

struct QOf {  // Q-values of a fixed network
  const Mlp* net;
  const TickScalpEnv* env;
  mutable std::vector<double> scratch;
  std::array<double, kA> operator()(const TickObs& o) const {
    std::array<double, kA> q{};
    const auto x = env->features(o);
    net->forward(x.data(), q.data(), scratch.data());
    return q;
  }
};

struct Score {
  double mean = 0, se = 0, fills = 0, win = 0, deviation = 0;
  std::vector<double> pnl;
};

template <class Policy>
Score score(const TickParams& p, Policy&& policy, unsigned episodes, std::uint64_t seed) {
  Score s;
  double decisions = 0;
  for (unsigned i = 0; i < episodes; ++i) {
    TickScalpEnv env(p);
    Rng r(seed + i);
    auto o = env.reset(r);
    for (;;) {
      const Action a = policy(o);
      s.deviation += a != rule(o);
      ++decisions;
      const auto st = env.step(a, r);
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
  s.deviation = 100 * s.deviation / decisions;
  return s;
}

std::string name(double margin) {
  if (margin < 0) return "greedy";
  char buf[32];
  std::snprintf(buf, sizeof buf, "anchored m=%.1f", margin);
  return buf;
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
  const double guide_floor = argc > 5 ? std::atof(argv[5]) : 0.25;
  const double rule_eps = argc > 6 ? std::atof(argv[6]) : 0.02;
  const unsigned val_episodes = argc > 7 ? static_cast<unsigned>(std::atoi(argv[7])) : 200;
  const unsigned eval_every = argc > 8 && std::atoi(argv[8]) > 0 ? static_cast<unsigned>(std::atoi(argv[8])) : std::max(1u, episodes / 20);
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
  // Light exploration in the agent's own episodes: the guided episodes are
  // the broad exploration, and random actions at a high rate are what kept
  // OF5's quotes from ever reaching the front of the queue.
  c.eps_start = 0.1;
  c.eps_end = 0.02;
  c.eps_decay_episodes = std::max(1u, static_cast<unsigned>(guide_frac * episodes));
  auto agent = std::make_unique<Agent<TickScalpEnv>>(env, c);

  const Score rule_val = score(p, rule, val_episodes, kValSeed);
  std::fprintf(stderr, "seed %llu: rule on validation %.3f ticks/ep (se %.3f), %.2f fills/ep\n", (unsigned long long)seed, rule_val.mean, rule_val.se,
               rule_val.fills);

  Rng train(seed);                   // training scenarios, one stream (as the Trainer in OF5)
  Rng behave(seed ^ 0xB0B0B0B0ull);  // guidance, deviations and exploration
  struct Champion {
    std::optional<Agent<TickScalpEnv>::Snapshot> snap;
    double margin = kGreedy;
    Score val;
    unsigned episode = 0;
  } best, greedy;  // best (snapshot, policy) pair; best greedy snapshot
  double behave_margin = std::numeric_limits<double>::infinity();  // the rule until the first evaluation
  double sum_guided = 0, sum_agent = 0, fills_guided = 0, fills_agent = 0;
  unsigned n_guided = 0, n_agent = 0;
  const auto t0 = std::chrono::steady_clock::now();
  for (unsigned ep = 1; ep <= episodes; ++ep) {
    const double f = std::min(1.0, (ep - 1) / std::max(1.0, guide_frac * episodes));
    const double p_guide = 1.0 + (guide_floor - 1.0) * f;
    const bool guided = behave.uniform() < p_guide;
    TickScalpEnv e(p);
    auto o = e.reset(train);
    double total = 0;
    for (;;) {
      Action a;
      if (guided) {
        a = behave.uniform() < rule_eps ? behave.below(static_cast<std::uint32_t>(kA)) : rule(o);
      } else if (behave.uniform() < agent->epsilon()) {
        a = behave.below(static_cast<std::uint32_t>(kA));
      } else {
        std::array<double, kA> q;
        agent->q_values(o, q.data());  // the online network
        a = choose(q, o, behave_margin);
      }
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
      const QOf q{&snap.net, &env, std::vector<double>(snap.net.scratch_size())};
      std::string line;
      double best_now = -std::numeric_limits<double>::infinity();
      for (const double m : kPolicies) {
        const Score v = score(p, [&](const TickObs& ob) { return choose(q(ob), ob, m); }, val_episodes, kValSeed);
        char buf[96];
        std::snprintf(buf, sizeof buf, " %s %.3f (%.1f fills, dev %.0f%%)", name(m).c_str(), v.mean, v.fills, v.deviation);
        line += buf;
        if (m >= 0 && v.mean > best_now) {
          best_now = v.mean;
          behave_margin = m;  // the agent's own episodes act as the best anchored policy now would
        }
        if (!best.snap || v.mean > best.val.mean) best = {snap, m, v, ep};
        if (m < 0 && (!greedy.snap || v.mean > greedy.val.mean)) greedy = {snap, m, v, ep};
      }
      const double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
      std::fprintf(stderr,
                   "seed %llu ep %u (%.0f s) p_guide %.2f eps %.3f | train: guided %.2f (%.1f fills, n %u) agent %.2f (%.1f fills, n %u) | "
                   "validation:%s | champion %s %.3f from ep %u\n",
                   (unsigned long long)seed, ep, sec, p_guide, agent->epsilon(), n_guided ? sum_guided / n_guided : 0.0,
                   n_guided ? fills_guided / n_guided : 0.0, n_guided, n_agent ? sum_agent / n_agent : 0.0, n_agent ? fills_agent / n_agent : 0.0,
                   n_agent, line.c_str(), name(best.margin).c_str(), best.val.mean, best.episode);
      sum_guided = sum_agent = fills_guided = fills_agent = 0;
      n_guided = n_agent = 0;
    }
  }
  if (argc > 9 && best.snap) best.snap->net.save(argv[9]);

  // Sealed test: the rule, the champion and the greedy champion on the same 300 episodes, paired with the rule.
  const Score rule_test = score(p, rule, 300, kTestSeed);
  auto paired = [](const Score& a, const Score& b) {  // mean and t of a - b, episode by episode
    double dm = 0, dv = 0;
    const auto n = static_cast<double>(a.pnl.size());
    for (std::size_t i = 0; i < a.pnl.size(); ++i) dm += (a.pnl[i] - b.pnl[i]) / n;
    for (std::size_t i = 0; i < a.pnl.size(); ++i) dv += std::pow(a.pnl[i] - b.pnl[i] - dm, 2);
    const double dse = std::sqrt(dv / (n - 1) / n);
    return std::array<double, 2>{dm, dse > 0 ? dm / dse : 0.0};
  };
  auto report = [&](const std::string& label, const Score& val, const Score& test) {
    // The validation difference is the one the champion was selected on (optimistic); the sealed one is not.
    const auto dv = paired(val, rule_val), dt = paired(test, rule_test);
    std::fprintf(stderr,
                 "seed %llu sealed %s: %.3f ticks/ep (t %.2f, %.2f fills, win %.1f%%, deviates from the rule at %.1f%% of decisions), vs rule %+.3f (t %.2f)"
                 " [validation vs rule %+.3f (t %.2f)]\n",
                 (unsigned long long)seed, label.c_str(), test.mean, test.se > 0 ? test.mean / test.se : 0.0, test.fills, test.win, test.deviation, dt[0],
                 dt[1], dv[0], dv[1]);
    std::printf("%llu,%u,%s,%.3f,%.2f,%.3f,%.2f,%.2f,%.1f,%.1f,%.3f,%.2f,%.3f,%.2f\n", (unsigned long long)seed, episodes, label.c_str(), val.mean,
                val.fills, test.mean, test.se > 0 ? test.mean / test.se : 0.0, test.fills, test.win, test.deviation, dt[0], dt[1], dv[0], dv[1]);
  };
  report("rule", rule_val, rule_test);
  for (const Champion* ch : {&best, &greedy}) {
    const QOf q{&ch->snap->net, &env, std::vector<double>(ch->snap->net.scratch_size())};
    const double m = ch->margin;
    const Score test = score(p, [&](const TickObs& ob) { return choose(q(ob), ob, m); }, 300, kTestSeed);
    report(ch == &best ? "champion " + name(m) : "greedy champion", ch->val, test);
  }
}
