// optional_libs_tests.cpp — E18: the optional optimization (NLopt) and data
// analysis (Eigen) helpers. Built only when both libraries are found.
#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#include "rag/agent/analysis.hpp"
#include "rag/agent/model_based.hpp"
#include "rag/agent/tuning.hpp"

using namespace rag::agent;

TEST(Tuning, MaximizesAKnownFunctionWithLogAndIntegerParameters) {
  // Peak at step = 0.01 (log scale), count = 7 (integer), weight = 0.3.
  const std::vector<tuning::Param> space = {
      {"step", 1e-4, 1.0, true, false}, {"count", 1, 20, false, true}, {"weight", 0.0, 1.0, false, false}};
  auto f = [](const std::vector<double>& v) {
    const double a = std::log10(v[0]) + 2, b = v[1] - 7, c = v[2] - 0.3;
    return -(a * a) - 0.1 * b * b - 4 * c * c;
  };
  const auto r = tuning::maximize(space, {0.5, 15, 0.9}, f, 300);
  ASSERT_EQ(r.best.size(), 3u);
  EXPECT_NEAR(std::log10(r.best[0]), -2.0, 0.1);
  EXPECT_EQ(r.best[1], 7.0);  // integer parameters arrive rounded
  EXPECT_NEAR(r.best[2], 0.3, 0.05);
  EXPECT_FALSE(r.history.empty());
  for (const auto& t : r.history) {
    EXPECT_EQ(t.values[1], std::round(t.values[1]));
    EXPECT_GE(t.values[0], 1e-4);
    EXPECT_LE(t.values[0], 1.0);
  }
}

TEST(Tuning, AGlobalStageEscapesTheLocalOptimumALocalSearchGetsStuckIn) {
  // Two peaks: 0.6 at x = -2 (where the search starts) and 1.0 at x = +2.
  const std::vector<tuning::Param> space = {{"x", -4.0, 4.0, false, false}};
  auto f = [](const std::vector<double>& v) {
    return std::exp(-(v[0] - 2) * (v[0] - 2)) + 0.6 * std::exp(-(v[0] + 2) * (v[0] + 2) / 0.5);
  };
  const auto local = tuning::maximize(space, {-2.0}, f, 60);
  const auto global = tuning::maximize_global(space, {-2.0}, f, 60, 30, 7);
  EXPECT_LT(local.best[0], 0.0) << "a local search from the lower peak should stay there";
  EXPECT_NEAR(global.best[0], 2.0, 0.05);
  EXPECT_GT(global.best_score, 0.99);
  EXPECT_GE(global.history.size(), 60u);  // NLopt may take one evaluation past maxeval per stage
  EXPECT_LE(global.history.size(), 92u);
  // Reproducible with the same seed.
  EXPECT_EQ(tuning::maximize_global(space, {-2.0}, f, 60, 30, 7).best, global.best);
}

TEST(Analysis, EigenAgreesWithTheModelBasedAgentsOwnSolver) {
  // The model-based agent solves its weighted normal equations with
  // hand-written Gaussian elimination. Rebuild the same regression from the
  // same transitions and solve it with Eigen's rank-revealing QR.
  ExecutionParams p;
  p.permanent_impact = 0.005;
  p.risk_aversion = 0.2;
  ModelBasedExecutionAgent mb(p, {.replan_every = 1'000'000, .cost_model = CostModel::Quadratic, .seed = 1});
  ExecutionEnv env(p);
  Rng rng(7), pick(8);
  std::vector<std::array<double, 4>> rows;
  std::vector<double> ys, ws;
  for (int ep = 0; ep < 200; ++ep) {
    auto o = env.reset(rng);
    for (;;) {
      const Action a = pick.below(5);
      mb.note_action(o, a);
      const auto st = env.step(a, rng);
      mb.observe(o, a, st.reward, st.obs, st.done);
      if (!(st.done && st.obs.t >= p.horizon)) {  // non-terminal steps feed the regression
        const double held = st.obs.inventory, q = static_cast<double>(o.inventory - st.obs.inventory);
        rows.push_back({held * o.signal, -held * held, -q * held, -q * q});  // alpha, holding, permanent, impact
        ys.push_back(st.reward);
        ws.push_back(1.0 / std::max(1.0, held * held));
      }
      o = st.obs;
      if (st.done) break;
    }
    mb.end_episode();
  }
  Eigen::MatrixXd X(static_cast<Eigen::Index>(rows.size()), 4);
  Eigen::VectorXd y(X.rows()), w(X.rows());
  for (Eigen::Index i = 0; i < X.rows(); ++i) {
    for (Eigen::Index j = 0; j < 4; ++j) X(i, j) = rows[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)];
    y(i) = ys[static_cast<std::size_t>(i)];
    w(i) = ws[static_cast<std::size_t>(i)];
  }
  const Eigen::VectorXd b = analysis::least_squares(X, y, w);
  const ModelEstimate m = mb.current_estimate();
  EXPECT_NEAR(m.alpha, b(0), 1e-6 * (1 + std::abs(b(0))));
  EXPECT_NEAR(m.holding, b(1), 1e-6);
  EXPECT_NEAR(m.permanent, b(2), 1e-6);
  EXPECT_NEAR(m.impact, b(3), 1e-6);
  // And both are close to the truth.
  EXPECT_NEAR(b(0), p.alpha, 0.1);
  EXPECT_NEAR(b(2), p.permanent_impact, 0.002);
  EXPECT_NEAR(b(3), p.impact, 0.002);
}

TEST(Analysis, QuadraticSensitivityRecoversKnownEffects) {
  // score = 3 z0 - 2 z1^2 + 0 * z2 + small noise, on standardized inputs.
  Rng rng(3);
  std::vector<std::vector<double>> rows;
  std::vector<double> y;
  for (int i = 0; i < 400; ++i) {
    const double a = rng.normal(), b = rng.normal(), c = rng.normal();
    rows.push_back({10 + 2 * a, -5 + 0.5 * b, 100 + 30 * c});  // arbitrary units: the fit standardizes
    y.push_back(3 * a - 2 * b * b + 0.01 * rng.normal());
  }
  const auto s = analysis::quadratic_sensitivity(rows, y);
  EXPECT_NEAR(s.linear[0], 3.0, 0.15);
  EXPECT_NEAR(s.quadratic[1], -2.0, 0.15);
  EXPECT_NEAR(s.linear[2], 0.0, 0.1);
  EXPECT_NEAR(s.quadratic[2], 0.0, 0.1);
  EXPECT_GT(s.r2, 0.99);
}
