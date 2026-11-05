// of_train — train order-flow scalping agents in parallel on a many-core
// machine: one seed per thread, then score every agent and their mean-Q
// ensemble on sealed test episodes against the 30 s delta rule.
//
//   of_train [--seeds 12] [--threads N] [--episodes 2000] [--market default|efficient|discovery]
//            [--hidden 64] [--gamma 0.97] [--nstep 5] [--batch 32] [--lr 0.001]
//            [--test 300] [--test-seed 900000000] [--out DIR]
// --test-seed: first seed of the scoring episodes (900M+ sealed; 950M+ is the
// once-per-claim fresh confirmation block, see docs/ai/NEXT_SESSION.md).
//
// --threads defaults to the number of hardware threads (capped at --seeds).
// Each agent is NeuralQ with Double DQN and weight averaging (the recipe of
// OF3); champion selected on validation episodes (seeds 700M+), scored once
// on sealed episodes (900M+, the OF2/OF3 test set). Networks are saved as
// DIR/seedK.net (load with Mlp::load, run with of_backtest on recorded ticks).
//
// Results don't depend on the thread count: each agent owns its environment,
// simulator and random streams. Time: ~0.75 s per episode per thread on a
// 2.8 GHz Xeon (much less on a modern desktop core); 12 seeds x 2,000
// episodes on 12 cores take about as long as one seed.
#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "rag/agent/trainer.hpp"
#include "rag/orderflow/scalp_env.hpp"

using namespace rag::agent;
using namespace rag::orderflow;

namespace {

struct Options {
  unsigned seeds = 12, threads = 0, episodes = 2000, test = 300, hidden = 64, nstep = 5, batch = 32;
  double gamma = 0.97, lr = 1e-3;
  std::uint64_t test_seed = 900'000'000;
  std::string market = "default", out = "of_train_out";
};

struct Score {
  double mean = 0, t = 0, win = 0, orders = 0;
};

template <class Policy>
Score score(const ScalpParams& p, Policy&& policy, unsigned episodes, std::uint64_t first_seed) {
  std::vector<double> pnl;
  double orders = 0;
  for (unsigned i = 0; i < episodes; ++i) {
    ScalpEnv env(p);
    Rng r(first_seed + i);  // sealed test episodes
    auto o = env.reset(r);
    for (;;) {
      const auto st = env.step(policy(o), r);
      o = st.obs;
      if (st.done) break;
    }
    pnl.push_back(env.net_pnl_ticks());
    orders += env.orders();
  }
  Score s;
  double v = 0;
  for (double x : pnl) {
    s.mean += x;
    s.win += x > 0;
  }
  const double n = static_cast<double>(pnl.size());
  s.mean /= n;
  for (double x : pnl) v += (x - s.mean) * (x - s.mean);
  const double se = std::sqrt(v / (n - 1) / n);
  s.t = se > 0 ? s.mean / se : 0;
  s.win = 100 * s.win / n;
  s.orders = orders / n;
  return s;
}

AgentConfig<ScalpEnv::kStateDims> config(const Options& o, std::uint64_t seed) {
  AgentConfig<ScalpEnv::kStateDims> c;
  c.seed = seed;
  c.learner = LearnerKind::NeuralQ;
  c.bins.fill(1);  // the tile grid is unused by NeuralQ
  c.double_q = true;
  c.nn_polyak = 1e-4;
  c.nn_hidden = o.hidden;
  c.nn_batch = o.batch;
  c.nn_lr = o.lr;
  c.gamma = o.gamma;
  c.n_step = o.nstep;
  c.eps_start = 1.0;
  c.eps_end = 0.02;
  c.eps_decay_episodes = std::max(1u, o.episodes / 2);
  return c;
}

}  // namespace

int main(int argc, char** argv) {
  Options o;
  for (int i = 1; i + 1 < argc; i += 2) {
    const std::string k = argv[i], v = argv[i + 1];
    if (k == "--seeds") o.seeds = static_cast<unsigned>(std::max(1, std::atoi(v.c_str())));
    else if (k == "--threads") o.threads = static_cast<unsigned>(std::max(1, std::atoi(v.c_str())));
    else if (k == "--episodes") o.episodes = static_cast<unsigned>(std::max(1, std::atoi(v.c_str())));
    else if (k == "--test") o.test = static_cast<unsigned>(std::max(2, std::atoi(v.c_str())));
    else if (k == "--hidden") o.hidden = static_cast<unsigned>(std::max(4, std::atoi(v.c_str())));
    else if (k == "--nstep") o.nstep = static_cast<unsigned>(std::max(1, std::atoi(v.c_str())));
    else if (k == "--batch") o.batch = static_cast<unsigned>(std::max(1, std::atoi(v.c_str())));
    else if (k == "--gamma") o.gamma = std::atof(v.c_str());
    else if (k == "--lr") o.lr = std::atof(v.c_str());
    else if (k == "--market") o.market = v;
    else if (k == "--out") o.out = v;
    else if (k == "--test-seed") {
      // Must be a whole number in a test block (900M+): a typo parsing to a
      // small seed would score on the training streams (Rng(1..seeds)).
      char* end = nullptr;
      errno = 0;
      o.test_seed = std::strtoull(v.c_str(), &end, 10);
      if (v.empty() || v[0] == '-' || *end != '\0' || errno == ERANGE || o.test_seed < 900'000'000) {
        std::fprintf(stderr, "--test-seed wants an integer >= 900000000 (a test block, docs/ai/NEXT_SESSION.md), got %s\n", v.c_str());
        return 2;
      }
    }
    else {
      std::fprintf(stderr, "unknown option %s\n", k.c_str());
      return 2;
    }
  }
  if (o.threads == 0) o.threads = std::max(1u, std::thread::hardware_concurrency());
  o.threads = std::min(o.threads, o.seeds);
  ScalpParams p;
  try {
    p.sim = market_preset(o.market);
  } catch (const std::invalid_argument& e) {
    std::fprintf(stderr, "--market: %s\n", e.what());
    return 2;
  }
  std::filesystem::create_directories(o.out);
  std::printf("training %u seeds x %u episodes on %u threads (hardware threads: %u), market %s, hidden %u\n", o.seeds, o.episodes, o.threads,
              std::thread::hardware_concurrency(), o.market.c_str(), o.hidden);
  std::fflush(stdout);

  const auto t0 = std::chrono::steady_clock::now();
  std::vector<std::unique_ptr<Agent<ScalpEnv>>> agents(o.seeds);
  std::vector<Score> scores(o.seeds);
  std::vector<double> validation(o.seeds);
  std::atomic<unsigned> next{0};
  std::mutex print;
  auto worker = [&] {
    for (unsigned i; (i = next++) < o.seeds;) {
      const std::uint64_t seed = i + 1;
      ScalpEnv env(p);
      auto agent = std::make_unique<Agent<ScalpEnv>>(env, config(o, seed));
      Trainer<ScalpEnv> tr(*agent, env,
                           {.eval_every = std::max(1u, o.episodes / 20), .eval_episodes = 40, .eval_seed = 700'000'000, .train_seed = seed,
                            .detect_drift = false});
      const auto rep = tr.run(o.episodes);
      validation[i] = rep.champion_score;
      if (const auto* ch = tr.champion()) {
        ch->net.save(o.out + "/seed" + std::to_string(seed) + ".net");
        agent->restore(*ch);  // the deployed policy, for scoring and the ensemble
      }
      scores[i] = score(p, [&](const ScalpObs& ob) { return agent->act(ob, false); }, o.test, o.test_seed);
      {
        std::lock_guard lock(print);
        std::printf("  seed %2llu: validation %+.2f, sealed test %+.2f ticks/episode (t %+.1f, %.1f orders, %.0f%% winning)\n",
                    static_cast<unsigned long long>(seed), validation[i], scores[i].mean, scores[i].t, scores[i].orders, scores[i].win);
        std::fflush(stdout);
      }
      agents[i] = std::move(agent);
    }
  };
  std::vector<std::thread> pool;
  for (unsigned t = 0; t < o.threads; ++t) pool.emplace_back(worker);
  for (auto& t : pool) t.join();
  const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

  const Score rule = score(
      p, [](const ScalpObs& ob) -> Action { return ob.f[1] > 1.5 ? 2 : ob.f[1] < -1.5 ? 0 : static_cast<Action>(ob.position + 1); }, o.test, o.test_seed);
  const Score ens = score(
      p,
      [&](const ScalpObs& ob) -> Action {
        std::array<double, ScalpEnv::kNumActions> sum{}, q{};
        for (auto& a : agents) {
          a->q_values(ob, q.data());
          for (std::size_t k = 0; k < q.size(); ++k) sum[k] += q[k];
        }
        return static_cast<Action>(std::max_element(sum.begin(), sum.end()) - sum.begin());
      },
      o.test, o.test_seed);
  double mean_single = 0;
  for (const auto& s : scores) mean_single += s.mean / static_cast<double>(scores.size());
  std::printf("\nsealed test, %u episodes of 10 min from seed %llu (1 ES contract):\n", o.test, static_cast<unsigned long long>(o.test_seed));
  std::printf("  delta 30 s rule      %+.2f ticks/episode (t %+.1f)\n", rule.mean, rule.t);
  std::printf("  RL, mean of seeds    %+.2f\n", mean_single);
  std::printf("  RL mean-Q ensemble   %+.2f (t %+.1f, %.1f orders, %.0f%% winning) of all %u seeds\n", ens.mean, ens.t, ens.orders, ens.win, o.seeds);
  std::printf("training took %.0f s; networks in %s/\n", secs, o.out.c_str());
  std::printf("note: these are simulator results; replay recorded ticks with of_backtest before trusting any edge.\n");
}
