// rag/agent/tuning.hpp — hyperparameter optimization with NLopt (optional
// dependency: libnlopt-cxx-dev; CMake target NLopt::nlopt). E18.
//
// maximize() searches a box of parameters, each optionally on a log scale
// and/or integer-valued, with a derivative-free NLopt algorithm (default
// Subplex, a robust Nelder-Mead variant that tolerates the plateaus integer
// parameters create). The objective receives parameters in natural units.
// Every trial is kept, for later analysis (analysis.hpp).
//
// Learning curves are noisy, so make the objective deterministic (fixed
// training seeds) and validate the winner on seeds the search never saw: in
// E18 the tuned score on the search's own seeds was ~10 points higher than
// on held-out seeds (the winner's curse).
#pragma once

#include <nlopt.hpp>

#include <cmath>
#include <functional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace rag::agent::tuning {

struct Param {
  std::string name;
  double lo, hi;          // bounds in natural units
  bool log_scale = false;  // search in log space (for step sizes, horizons, ...)
  bool integer = false;    // rounded before the objective sees it
};

struct Trial {
  std::vector<double> values;  // natural units
  double score;
};

struct Result {
  std::vector<double> best;  // natural units
  double best_score = -HUGE_VAL;
  std::vector<Trial> history;
  std::string status;        // NLopt's termination reason
};

namespace detail {
inline double to_search(const Param& p, double v) { return p.log_scale ? std::log(v) : v; }
inline double to_natural(const Param& p, double s) {
  double v = p.log_scale ? std::exp(s) : s;
  if (p.integer) v = std::round(v);
  return std::min(std::max(v, p.lo), p.hi);
}
}  // namespace detail

// Maximize objective(values) over `space`, starting from `start` (natural
// units), with at most `max_evals` evaluations.
inline Result maximize(const std::vector<Param>& space, const std::vector<double>& start,
                       const std::function<double(const std::vector<double>&)>& objective, int max_evals,
                       nlopt::algorithm algorithm = nlopt::LN_SBPLX) {
  const std::size_t n = space.size();
  if (start.size() != n) throw std::invalid_argument("tuning::maximize: start has the wrong size");
  std::vector<double> lo(n), hi(n), x(n), step(n);
  for (std::size_t i = 0; i < n; ++i) {
    lo[i] = detail::to_search(space[i], space[i].lo);
    hi[i] = detail::to_search(space[i], space[i].hi);
    x[i] = detail::to_search(space[i], start[i]);
    step[i] = 0.25 * (hi[i] - lo[i]);
    if (space[i].integer && !space[i].log_scale) step[i] = std::max(step[i], 1.0);  // move by whole units
  }
  Result r;
  struct Ctx {
    const std::vector<Param>* space;
    const std::function<double(const std::vector<double>&)>* f;
    Result* r;
  } ctx{&space, &objective, &r};
  nlopt::opt opt(algorithm, static_cast<unsigned>(n));
  opt.set_lower_bounds(lo);
  opt.set_upper_bounds(hi);
  opt.set_maxeval(max_evals);
  opt.set_initial_step(step);
  opt.set_max_objective(
      [](const std::vector<double>& s, std::vector<double>&, void* data) {
        auto& c = *static_cast<Ctx*>(data);
        std::vector<double> v(s.size());
        for (std::size_t i = 0; i < s.size(); ++i) v[i] = detail::to_natural((*c.space)[i], s[i]);
        const double score = (*c.f)(v);
        c.r->history.push_back({v, score});
        if (score > c.r->best_score) {
          c.r->best_score = score;
          c.r->best = v;
        }
        return score;
      },
      &ctx);
  double fx = 0;
  try {
    const nlopt::result res = opt.optimize(x, fx);
    r.status = res == nlopt::MAXEVAL_REACHED ? "max evaluations reached" : "converged";
  } catch (const std::exception& e) {
    r.status = e.what();  // e.g. roundoff-limited: the best trial so far is still valid
  }
  return r;
}

// Global, then local: CRS2 (controlled random search with local mutation)
// over the whole box, then Subplex from the best point it found. Use this
// when the score landscape is rugged or noisy: in E18 a local search alone
// found 57% or 70% depending only on its initial step sizes. Reproducible:
// NLopt's random generator is seeded with `seed`.
inline Result maximize_global(const std::vector<Param>& space, const std::vector<double>& start,
                              const std::function<double(const std::vector<double>&)>& objective, int global_evals,
                              int local_evals, unsigned long seed = 1) {
  nlopt::srand(seed);
  Result g = maximize(space, start, objective, global_evals, nlopt::GN_CRS2_LM);
  Result l = maximize(space, g.best.empty() ? start : g.best, objective, local_evals, nlopt::LN_SBPLX);
  Result r = g.best_score >= l.best_score ? g : l;
  r.history = g.history;
  r.history.insert(r.history.end(), l.history.begin(), l.history.end());
  r.status = "global: " + g.status + "; local: " + l.status;
  return r;
}

}  // namespace rag::agent::tuning
