// rag/agent/trainer.hpp — the supervisor around a learning agent: training
// loop, held-out evaluation, champion/challenger promotion, and drift
// detection.
//
// A self-learning system needs more than a learner. It also needs:
//
//   * Held-out evaluation. Training returns are measured while exploring,
//     and on scenarios the agent has adapted to. Evaluation replays a fixed
//     set of *unseen* scenarios (seeds the trainer never uses) greedily, so
//     every score is comparable with every other.
//   * A champion. Learning is noisy, and a later policy can be worse than an
//     earlier one. The trainer keeps the best policy seen so far and only
//     promotes a challenger that scores at least as well on the same
//     scenarios. The champion is re-scored each time, so after the world
//     changes it is judged on the new world, not its old record. The
//     champion is what gets *deployed* (deployed_act()); the learner keeps
//     its own weights and keeps improving. Overwriting the learner with the
//     champion would throw away whatever it was in the middle of learning.
//   * Drift detection, in two stages.
//       1. Suspect (fast, noisy): a CUSUM test on surprise = (return obtained)
//          - (return the agent predicted at the start of the episode). A
//          sustained negative shift means things go worse than the agent
//          expects. But that has two causes: the world changed, or the
//          agent's own learning wobbled (values are biased and still moving
//          while it learns).
//       2. Confirm (slow, exact): re-score the frozen champion on the fixed
//          held-out scenarios. In an unchanged world that score is
//          reproduced bit for bit; only a changed world moves it. A
//          confirmed drift makes the agent explore again, with a fresh step
//          size and an empty replay buffer, instead of exploiting stale
//          knowledge. Without a champion yet, suspicions are acted on.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <deque>
#include <functional>
#include <optional>
#include <vector>

#include "rag/agent/agent.hpp"
#include "rag/agent/env.hpp"
#include "rag/agent/rng.hpp"

namespace rag::agent {

struct EvalStats {
  double mean = 0;
  double std_error = 0;
  std::uint32_t episodes = 0;
};

// Run `episodes` greedy episodes of policy(obs) -> Action on copies of
// `proto`. Episode i always uses Rng(seed + i), so two policies evaluated
// with the same seed face exactly the same scenarios (common random numbers).
template <Environment Env, class Policy>
EvalStats evaluate(const Env& proto, Policy&& policy, std::uint32_t episodes, std::uint64_t seed) {
  double sum = 0, sum2 = 0;
  for (std::uint32_t i = 0; i < episodes; ++i) {
    Env env = proto;
    Rng rng(seed + i);
    auto o = env.reset(rng);
    double total = 0;
    for (;;) {
      const auto st = env.step(policy(o), rng);
      total += st.reward;
      o = st.obs;
      if (st.done) break;
    }
    sum += total;
    sum2 += total * total;
  }
  EvalStats s;
  s.episodes = episodes;
  s.mean = episodes ? sum / episodes : 0;
  const double var = episodes > 1 ? std::max(0.0, (sum2 - episodes * s.mean * s.mean) / (episodes - 1)) : 0;
  s.std_error = episodes ? std::sqrt(var / episodes) : 0;
  return s;
}

// Mean per-episode difference policy_a - policy_b on the same scenarios.
// Pairing cancels the scenario-to-scenario noise both policies share, which
// usually dwarfs the difference between them.
template <Environment Env, class PolicyA, class PolicyB>
EvalStats compare(const Env& proto, PolicyA&& a, PolicyB&& b, std::uint32_t episodes, std::uint64_t seed) {
  double sum = 0, sum2 = 0;
  for (std::uint32_t i = 0; i < episodes; ++i) {
    const double ra = evaluate(proto, a, 1, seed + i).mean;
    const double rb = evaluate(proto, b, 1, seed + i).mean;
    sum += ra - rb;
    sum2 += (ra - rb) * (ra - rb);
  }
  EvalStats s;
  s.episodes = episodes;
  s.mean = episodes ? sum / episodes : 0;
  const double var = episodes > 1 ? std::max(0.0, (sum2 - episodes * s.mean * s.mean) / (episodes - 1)) : 0;
  s.std_error = episodes ? std::sqrt(var / episodes) : 0;
  return s;
}

// Defaults chosen by simulation on Gaussian noise: about one false alarm
// per 200,000 episodes, and a shift of 0.55 standard deviations (what the
// ExecutionEnv regime flip produces) detected after a median of ~50 episodes.
struct DriftParams {
  std::uint32_t lag = 50;         // samples held back before they join the baseline
  double allowance = 0.3;         // CUSUM k, in standard deviations: ignore shifts smaller than ~2k
  double threshold = 16.0;        // CUSUM h, in standard deviations of accumulated evidence
  double baseline_rate = 0.001;   // EMA rate of the baseline (memory of ~1000 samples)
  std::uint32_t warmup = 1500;    // baseline samples needed before alarms (also after one)
};

// Flags a sustained downward shift in "surprise" (return - predicted return)
// with a one-sided CUSUM test (Page, 1954):
//     S = min(0, S + z + k),   alarm when S < -h,   z = (x - mean) / sd
// Each sample adds evidence; ordinary noise drains it back toward 0 because
// of the allowance k, while a persistent shift accumulates until it crosses h.
//
// The baseline (mean, sd) is built only from samples at least `lag` old, so
// a shift is measured against the world before it, not partly against
// itself; and it has a long memory, because a noisy baseline is itself a
// source of false alarms.
class DriftDetector {
 public:
  using Params = DriftParams;
  explicit DriftDetector(Params p = {}) : p_(p) {}

  // Feed one episode's surprise; returns true on an alarm.
  bool add(double surprise) {
    held_.push_back(surprise);
    if (held_.size() > p_.lag) {  // the oldest sample graduates into the baseline
      absorb(held_.front());
      held_.pop_front();
    }
    if (baseline_n_ < p_.warmup) return false;
    const double z = (surprise - mean_) / std::sqrt(std::max(var_, 1e-12));
    cusum_ = std::min(0.0, cusum_ + z + p_.allowance);
    if (cusum_ < -p_.threshold) {
      reset();
      return true;
    }
    return false;
  }

  void reset() {
    held_.clear();
    baseline_n_ = 0;
    mean_ = var_ = cusum_ = 0;
  }

  [[nodiscard]] double evidence() const noexcept { return -cusum_; }  // how close to an alarm (alarm at threshold)

 private:
  // Exponential moving mean and variance; exact running moments while young.
  void absorb(double x) {
    ++baseline_n_;
    const double rate = std::max(p_.baseline_rate, 1.0 / static_cast<double>(baseline_n_));
    const double d = x - mean_;
    mean_ += rate * d;
    var_ = (1 - rate) * (var_ + rate * d * d);
  }

  Params p_;
  std::deque<double> held_;
  std::uint64_t baseline_n_ = 0;
  double mean_ = 0, var_ = 0, cusum_ = 0;
};

struct TrainConfig {
  std::uint32_t eval_every = 250;      // episodes between champion/challenger evaluations
  std::uint32_t eval_episodes = 300;   // held-out scenarios per evaluation
  std::uint64_t eval_seed = 9'000'000; // held-out scenario seeds: eval_seed + i
  std::uint64_t train_seed = 1;        // training scenarios come from one stream seeded here
  bool keep_champion = true;           // deploy the best policy seen, not just the latest
  bool detect_drift = true;
  DriftParams drift{};
  // Drift is only meaningful once the agent has stopped exploring heavily.
  // 0 = once the agent's ε schedule has finished (AgentConfig::eps_decay_episodes).
  std::uint32_t drift_after_episodes = 0;
  double reexplore_epsilon = 0.5;
  std::uint32_t reexplore_episodes = 5000;
  bool forget_on_drift = true;  // clear the replay buffer when drift is detected
  // Policy bank: on a confirmed drift the outgoing champion is banked, and
  // every banked policy is scored on the new world. If one clearly beats the
  // current champion (by 2 standard errors), the world has returned to a
  // regime seen before: that policy is redeployed, the learner warm-starts
  // from it, and exploration stays light (recall_epsilon).
  bool policy_bank = true;
  std::uint32_t bank_capacity = 8;  // oldest entries are dropped beyond this
  double recall_epsilon = 0.05;
  std::uint32_t recall_episodes = 500;
};

struct EvalPoint {
  std::uint64_t episode;
  double challenger;  // current policy on the held-out scenarios
  double champion;    // best policy so far, re-scored on the same scenarios
  bool promoted;
  double epsilon;
  bool learned_deployed = true;  // false while a baseline (set_baseline) beats the champion
};

struct TrainReport {
  std::vector<EvalPoint> evals;
  std::vector<std::uint64_t> drift_alarms;    // confirmed drifts (episode numbers)
  std::vector<std::uint64_t> drift_rejected;  // suspicions the champion check rejected
  std::vector<std::uint64_t> bank_recalls;    // confirmed drifts answered by redeploying a banked policy
  std::vector<double> returns;              // every training episode's return
  double champion_score = 0;                // held-out score of the deployed policy
};

template <Environment Env>
class Trainer {
 public:
  Trainer(Agent<Env>& agent, Env& env, TrainConfig cfg = {})
      : agent_(agent), env_(env), cfg_(cfg), rng_(cfg.train_seed), drift_(cfg.drift) {}

  // One training episode: act with exploration, learn from every step.
  double run_episode() {
    auto o = env_.reset(rng_);
    const double predicted = agent_.value(o);
    double total = 0;
    for (;;) {
      const Action a = agent_.act(o, /*explore=*/true);
      const auto st = env_.step(a, rng_);
      agent_.observe(o, a, st.reward, st.obs, st.done);
      total += st.reward;
      o = st.obs;
      if (st.done) break;
    }
    agent_.end_episode();
    last_surprise_ = total - predicted;
    return total;
  }

  // Train for `episodes` more episodes. Can be called repeatedly; the
  // champion, drift detector and scenario stream carry over between calls.
  TrainReport run(std::uint32_t episodes, const std::function<void(const EvalPoint&)>& on_eval = {}) {
    TrainReport rep;
    rep.returns.reserve(episodes);
    for (std::uint32_t i = 0; i < episodes; ++i) {
      rep.returns.push_back(run_episode());
      const std::uint64_t drift_after =
          cfg_.drift_after_episodes ? cfg_.drift_after_episodes : agent_.config().eps_decay_episodes;
      if (cfg_.detect_drift && agent_.episodes() > drift_after && drift_.add(last_surprise_)) {
        if (confirm_drift()) {
          rep.drift_alarms.push_back(agent_.episodes());
          if (cfg_.forget_on_drift) agent_.forget_experience();
          if (recall_from_bank()) {
            check_baseline();
            rep.bank_recalls.push_back(agent_.episodes());
            agent_.reexplore(cfg_.recall_epsilon, cfg_.recall_episodes);
          } else {
            agent_.reexplore(cfg_.reexplore_epsilon, cfg_.reexplore_episodes);
            check_baseline();  // the old champion just failed: fall back to the baseline if it's better
          }
        } else {
          rep.drift_rejected.push_back(agent_.episodes());
        }
      }
      if (cfg_.eval_every && agent_.episodes() % cfg_.eval_every == 0) {
        rep.evals.push_back(check_champion());
        if (on_eval) on_eval(rep.evals.back());
      }
    }
    rep.champion_score = deployed_score();
    return rep;
  }

  // The policy to use in production: the champion, or the learner itself
  // when no evaluation has happened yet (or keep_champion is off). With a
  // baseline set, the baseline until a learned policy has beaten it.
  [[nodiscard]] Action deployed_act(const typename Env::Observation& o) const noexcept {
    if (baseline_ && !learned_beats_baseline_) return baseline_(o);
    return (cfg_.keep_champion && champion_) ? agent_.act_as(*champion_, o) : agent_.act(o, false);
  }

  // A safe fallback policy (e.g. TWAP). The learned champion is deployed only
  // once it beats the baseline on the held-out scenarios by more than 2
  // standard errors of the paired difference; it's re-checked at every
  // champion evaluation and after every confirmed drift. Measured: in a
  // weak-signal market a Q-learner's champion was 35% *worse* than TWAP;
  // with this guard the deployed policy stays at TWAP until learning pays.
  void set_baseline(std::function<Action(const typename Env::Observation&)> baseline) {
    baseline_ = std::move(baseline);
    learned_beats_baseline_ = false;
  }
  [[nodiscard]] bool learned_policy_deployed() const noexcept { return !baseline_ || learned_beats_baseline_; }
  [[nodiscard]] double deployed_score() {
    return evaluate(env_, [&](const auto& o) { return deployed_act(o); }, cfg_.eval_episodes, cfg_.eval_seed).mean;
  }
  [[nodiscard]] bool has_champion() const noexcept { return champion_.has_value(); }
  [[nodiscard]] std::size_t bank_size() const noexcept { return bank_.size(); }

  // Held-out greedy score of the agent as it is now.
  [[nodiscard]] double score_current() { return score_current_stats().mean; }

 private:
  EvalStats score_current_stats() {
    return evaluate(env_, [&](const auto& o) { return agent_.act(o, false); }, cfg_.eval_episodes, cfg_.eval_seed);
  }

  double score_snapshot(const typename Agent<Env>::Snapshot& s) { return score_snapshot_stats(s).mean; }
  EvalStats score_snapshot_stats(const typename Agent<Env>::Snapshot& s) {
    const auto mine = agent_.snapshot();
    agent_.restore(s);
    const EvalStats v = score_current_stats();
    agent_.restore(mine);
    return v;
  }

  // After a confirmed drift: bank the outgoing champion, then look for a
  // banked policy that fits the new world better than the current champion.
  bool recall_from_bank() {
    if (!cfg_.policy_bank || !champion_) return false;
    bank_.push_back(*champion_);
    if (bank_.size() > cfg_.bank_capacity) bank_.erase(bank_.begin());
    std::size_t best = bank_.size();
    EvalStats best_stats = champion_stats_;  // the champion as re-scored on the new world
    for (std::size_t i = 0; i + 1 < bank_.size(); ++i) {  // the newest entry is the champion itself
      const EvalStats s = score_snapshot_stats(bank_[i]);
      if (s.mean > best_stats.mean + 2.0 * champion_stats_.std_error) {
        best = i;
        best_stats = s;
      }
    }
    if (best == bank_.size()) return false;
    champion_ = bank_[best];
    champion_stats_ = best_stats;
    agent_.restore(*champion_);  // warm start: the learner continues from the recalled policy
    return true;
  }

  // Stage 2 of drift detection: did the frozen champion's held-out score
  // drop? (Exactly reproducible in an unchanged world.)
  bool confirm_drift() {
    if (!champion_) return true;
    const EvalStats now = score_snapshot_stats(*champion_);
    const bool dropped = now.mean < champion_stats_.mean - 2.0 * champion_stats_.std_error;
    if (dropped) champion_stats_ = now;
    return dropped;
  }

  // Paired held-out comparison of the champion with the baseline.
  void check_baseline() {
    if (!baseline_ || !champion_) return;
    const auto mine = agent_.snapshot();
    agent_.restore(*champion_);
    const EvalStats d = compare(env_, [&](const auto& o) { return agent_.act(o, false); }, baseline_, cfg_.eval_episodes,
                                cfg_.eval_seed);
    agent_.restore(mine);
    learned_beats_baseline_ = d.mean > 2.0 * d.std_error;
  }

  EvalPoint check_champion() {
    EvalPoint p = check_champion_inner();
    check_baseline();
    p.learned_deployed = learned_policy_deployed();
    return p;
  }

  EvalPoint check_champion_inner() {
    const EvalStats challenger = score_current_stats();
    EvalPoint p{agent_.episodes(), challenger.mean, 0.0, false, agent_.epsilon()};
    if (!champion_) {
      champion_ = agent_.snapshot();
      champion_stats_ = challenger;
      p.champion = p.challenger;
      p.promoted = true;
      return p;
    }
    champion_stats_ = score_snapshot_stats(*champion_);  // re-scored: the world may have changed
    p.champion = champion_stats_.mean;
    if (p.challenger >= p.champion) {
      champion_ = agent_.snapshot();
      champion_stats_ = challenger;
      p.champion = p.challenger;
      p.promoted = true;
    }
    return p;
  }

  Agent<Env>& agent_;
  Env& env_;
  TrainConfig cfg_;
  Rng rng_;
  DriftDetector drift_;
  std::optional<typename Agent<Env>::Snapshot> champion_;
  EvalStats champion_stats_{};
  std::vector<typename Agent<Env>::Snapshot> bank_;  // champions of past regimes
  std::function<Action(const typename Env::Observation&)> baseline_;
  bool learned_beats_baseline_ = false;
  double last_surprise_ = 0;
};

}  // namespace rag::agent
