// rag/agent/variance_reduction.hpp — remove known-zero-mean noise from
// rewards before a learner sees them (a control variate; E17).
//
// In ExecutionEnv every non-final step's reward contains I' * dS, the
// inventory held through the price move dS = alpha * g + noise. Given the
// state and action, the innovation dS - alpha * g has mean zero, so
//
//     r~ = r - I' * (dS - alpha_hat * g)
//
// has the same expectation as r when alpha_hat = alpha: every policy's value,
// and so the optimal policy, is unchanged, while the dominant source of
// reward noise is gone. What remains is the randomness of the next signal.
//
// alpha_hat is either supplied (an oracle) or estimated online by least
// squares of the observed moves on the signal (market data the agent sees;
// no knowledge of the market is assumed). InnovationControlledEnv wraps
// ExecutionEnv and satisfies the Environment concept, so any learner and the
// Trainer use it unchanged. Score policies on the plain ExecutionEnv.
#pragma once

#include <optional>

#include "rag/agent/env.hpp"

namespace rag::agent {

class InnovationControlledEnv {
 public:
  using Observation = ExecutionObs;
  static constexpr std::size_t kNumActions = ExecutionEnv::kNumActions;
  static constexpr std::size_t kStateDims = ExecutionEnv::kStateDims;
  static constexpr Action kTwapAction = ExecutionEnv::kTwapAction;
  static constexpr std::array<double, 3> kLo = ExecutionEnv::kLo, kHi = ExecutionEnv::kHi;

  explicit InnovationControlledEnv(ExecutionEnv env, std::optional<double> alpha = std::nullopt)
      : env_(std::move(env)), fixed_alpha_(alpha) {}

  Observation reset(Rng& rng) noexcept {
    last_ = env_.reset(rng);
    return last_;
  }

  Step<Observation> step(Action a, Rng& rng) noexcept {
    const double g = last_.signal;
    auto st = env_.step(a, rng);
    const double move = env_.last_move();
    if (move != 0.0 || !st.done) {  // a price move happened while inventory was held
      st.reward -= static_cast<double>(st.obs.inventory) * (move - alpha_hat() * g);
      gm_ += g * move;  // online regression of the move on the signal
      gg_ += g * g;
    }
    last_ = st.obs;
    return st;
  }

  [[nodiscard]] std::array<double, 3> features(const Observation& o) const noexcept { return env_.features(o); }

  // The drift coefficient used for the correction.
  [[nodiscard]] double alpha_hat() const noexcept {
    if (fixed_alpha_) return *fixed_alpha_;
    return gg_ > 0 ? gm_ / gg_ : 0.0;
  }
  [[nodiscard]] const ExecutionEnv& inner() const noexcept { return env_; }
  [[nodiscard]] ExecutionEnv& inner() noexcept { return env_; }

 private:
  ExecutionEnv env_;
  std::optional<double> fixed_alpha_;
  Observation last_{};
  double gm_ = 0, gg_ = 0;
};

static_assert(Environment<InnovationControlledEnv>);

}  // namespace rag::agent
