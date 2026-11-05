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
#include "rag/agent/drift.hpp"
#include "rag/agent/env.hpp"
#include "rag/agent/evaluate.hpp"
#include "rag/agent/rng.hpp"

namespace rag::agent {

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
  // The champion's re-score moved by more than 2 standard errors. In an
  // unchanged world it reproduces exactly (frozen policy, fixed scenarios),
  // so this is a confirmed change, in either direction (E13).
  bool world_changed = false;
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
          handle_drift(rep);
        } else {
          drift_.rebase();  // a false alarm: stay armed
          rep.drift_rejected.push_back(agent_.episodes());
        }
      }
      if (cfg_.eval_every && agent_.episodes() % cfg_.eval_every == 0) {
        rep.evals.push_back(check_champion());
        if (rep.evals.back().world_changed) handle_drift(rep);
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
  // The deployed policy's snapshot (nullptr before the first evaluation);
  // e.g. champion()->net.save(path) for a NeuralQ agent.
  [[nodiscard]] const auto* champion() const noexcept { return champion_ ? &*champion_ : nullptr; }
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
  // A confirmed change: rebuild the detector's baseline, bank the outgoing
  // champion, and either recall a banked policy or re-explore.
  void handle_drift(TrainReport& rep) {
    drift_.reset();
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
  }

  bool confirm_drift() {
    if (!champion_) return true;
    const EvalStats now = score_snapshot_stats(*champion_);
    const bool dropped = now.mean < champion_stats_.mean - 2.0 * champion_stats_.std_error;
    // With a two-sided detector, a significant rise is a changed world too.
    const bool rose = cfg_.drift.two_sided && now.mean > champion_stats_.mean + 2.0 * champion_stats_.std_error;
    if (dropped || rose) champion_stats_ = now;
    return dropped || rose;
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
    const EvalStats before = champion_stats_;
    champion_stats_ = score_snapshot_stats(*champion_);  // re-scored: the world may have changed
    p.champion = champion_stats_.mean;
    if (cfg_.detect_drift && std::abs(champion_stats_.mean - before.mean) > 2.0 * before.std_error) {
      p.world_changed = true;  // keep the old champion this time, so the drift handler banks it
      return p;
    }
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
