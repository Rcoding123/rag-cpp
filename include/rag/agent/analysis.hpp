// rag/agent/analysis.hpp — data analysis with Eigen (optional dependency:
// libeigen3-dev; CMake target Eigen3::Eigen). E18.
//
//   least_squares()          rank-revealing weighted least squares
//                            (column-pivoting Householder QR); used to
//                            cross-check the model-based agent's hand-written
//                            normal-equations solver
//   quadratic_sensitivity()  which parameters matter: fit
//                            score ~ b0 + sum b_i z_i + sum c_i z_i^2 on
//                            standardized parameters (e.g. a tuning history)
#pragma once

#include <Eigen/Dense>

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace rag::agent::analysis {

// argmin_b sum_i w_i (y_i - x_i . b)^2. Empty weights mean all ones.
inline Eigen::VectorXd least_squares(const Eigen::MatrixXd& X, const Eigen::VectorXd& y, const Eigen::VectorXd& w = {}) {
  if (X.rows() != y.size()) throw std::invalid_argument("least_squares: X and y disagree");
  if (w.size() == 0) return X.colPivHouseholderQr().solve(y);
  if (w.size() != y.size()) throw std::invalid_argument("least_squares: wrong weight count");
  const Eigen::VectorXd sw = w.cwiseSqrt();
  return (sw.asDiagonal() * X).colPivHouseholderQr().solve(sw.asDiagonal() * y);
}

struct Sensitivity {
  std::vector<double> linear;     // effect of +1 standard deviation of each parameter
  std::vector<double> quadratic;  // curvature per parameter (negative: an interior optimum)
  double r2 = 0;                  // share of the score variance the surface explains
};

// Quadratic response surface over standardized parameters (no interactions).
// rows[i] holds one trial's parameter values; scores[i] its score.
inline Sensitivity quadratic_sensitivity(const std::vector<std::vector<double>>& rows, const std::vector<double>& scores) {
  const auto n = static_cast<Eigen::Index>(rows.size());
  if (n == 0 || scores.size() != rows.size()) throw std::invalid_argument("quadratic_sensitivity: empty or mismatched input");
  const auto k = static_cast<Eigen::Index>(rows[0].size());
  Eigen::MatrixXd raw(n, k);
  for (Eigen::Index i = 0; i < n; ++i)
    for (Eigen::Index j = 0; j < k; ++j) raw(i, j) = rows[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)];
  const Eigen::RowVectorXd mean = raw.colwise().mean();
  Eigen::RowVectorXd sd = ((raw.rowwise() - mean).array().square().colwise().sum() / std::max<double>(1.0, static_cast<double>(n - 1))).sqrt();
  for (Eigen::Index j = 0; j < k; ++j)
    if (sd(j) <= 0) sd(j) = 1;  // a parameter that never moved: no information, no division by zero
  const Eigen::MatrixXd z = (raw.rowwise() - mean).array().rowwise() / sd.array();
  Eigen::MatrixXd X(n, 1 + 2 * k);
  X.col(0).setOnes();
  X.middleCols(1, k) = z;
  X.rightCols(k) = z.array().square().matrix();
  Eigen::VectorXd y(n);
  for (Eigen::Index i = 0; i < n; ++i) y(i) = scores[static_cast<std::size_t>(i)];
  const Eigen::VectorXd b = least_squares(X, y);
  Sensitivity s;
  for (Eigen::Index j = 0; j < k; ++j) {
    s.linear.push_back(b(1 + j));
    s.quadratic.push_back(b(1 + k + j));
  }
  const double tss = (y.array() - y.mean()).square().sum();
  const double rss = (y - X * b).squaredNorm();
  s.r2 = tss > 0 ? 1 - rss / tss : 0;
  return s;
}

}  // namespace rag::agent::analysis
