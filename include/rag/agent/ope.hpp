// rag/agent/ope.hpp — off-policy evaluation: estimate a target policy's value
// from episodes logged under a different (behaviour) policy, without running
// the target.
//
// Estimators (per-decision forms; Precup et al. 2000, Jiang & Li 2016):
//   PDIS   sum_t mean_i( w_it r_it ),            w_it = prod_{k<=t} pi(a_ik|s_ik) / b(a_ik|s_ik)
//   WPDIS  sum_t sum_i( w_it r_it ) / sum_i w_it  (self-normalized: biased, much lower variance)
//   DR     mean_i sum_t [ w_it (r_it - Q(s_it, a_it)) + w_i,t-1 V(s_it) ]
//          (doubly robust: a model's Q/V plus importance-weighted corrections)
// The direct method (DM) is just the model's own value of the target, e.g.
// ExecutionDP::policy_value() on a fitted model.
//
// Diagnostic: the effective sample size of the step-t weights,
//   ESS_t = (sum_i w_it)^2 / sum_i w_it^2,
// is how many logged episodes the estimate at step t effectively rests on.
// When it collapses toward 0, importance sampling can't be trusted.
//
// Measured on ExecutionEnv (E10): with horizon 20 and a deterministic target
// that differs from the behaviour in many states, no logged episode follows
// the target beyond a few steps (ESS ~ 0), and the IS family's errors stay
// at 45-2,968% of the optimal gain even with 10,000 logged episodes. The
// direct method is accurate when the model class is right (<= 2% at 1,000
// episodes) and biased when it isn't, by an amount more data doesn't
// shrink; DR repairs that bias only for targets close to the behaviour.
#pragma once

#include <algorithm>
#include <cstdint>
#include <limits>
#include <vector>

#include "rag/agent/env.hpp"

namespace rag::agent {

template <class Obs>
struct LoggedStep {
  Obs obs;
  Action action;
  double reward;
  double propensity;  // b(action | obs) under the behaviour policy, > 0
};
template <class Obs>
using LoggedEpisode = std::vector<LoggedStep<Obs>>;

struct OpeEstimates {
  double pdis = 0, wpdis = 0, dr = 0;
  double min_ess = 0;               // smallest effective sample size over steps (in episodes)
  std::uint32_t uncovered_steps = 0;  // steps at which no logged episode followed the target
};

// Estimates for a deterministic target policy pi(obs) -> Action. q(obs, a)
// and v(obs) are a model's action values and state values *for the target*
// (used by DR only; pass functions returning 0 to get plain IS). Episodes
// shorter than `horizon` ended early with reward 0 afterwards.
template <class Obs, class Target, class QModel, class VModel>
OpeEstimates off_policy_estimates(const std::vector<LoggedEpisode<Obs>>& logs, Target&& pi, QModel&& q, VModel&& v,
                                  std::uint32_t horizon) {
  OpeEstimates out;
  if (logs.empty()) return out;
  const double n = static_cast<double>(logs.size());
  std::vector<double> wsum(horizon, 0.0), w2sum(horizon, 0.0), wr(horizon, 0.0);
  double pdis = 0, dr = 0;
  for (const auto& ep : logs) {
    double w = 1.0, w_prev = 1.0;
    for (std::uint32_t t = 0; t < horizon; ++t) {
      if (t < ep.size()) {
        const auto& s = ep[t];
        w *= (pi(s.obs) == s.action ? 1.0 : 0.0) / s.propensity;
        pdis += w * s.reward;
        wr[t] += w * s.reward;
        dr += w * (s.reward - q(s.obs, s.action)) + w_prev * v(s.obs);
        w_prev = w;
      }
      wsum[t] += w;  // after the episode ended: weight carries over, reward 0
      w2sum[t] += w * w;
    }
  }
  out.pdis = pdis / n;
  out.dr = dr / n;
  out.min_ess = std::numeric_limits<double>::infinity();
  for (std::uint32_t t = 0; t < horizon; ++t) {
    if (wsum[t] > 0) {
      out.wpdis += wr[t] / wsum[t];
      out.min_ess = std::min(out.min_ess, wsum[t] * wsum[t] / w2sum[t]);
    } else {
      ++out.uncovered_steps;
      out.min_ess = 0;
    }
  }
  return out;
}

}  // namespace rag::agent
