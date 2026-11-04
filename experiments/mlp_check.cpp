// Gradient check (finite differences) and a regression sanity test for mlp.hpp.
#include <cmath>
#include <cstdio>
#include "rag/agent/mlp.hpp"
using namespace rag::agent;
int main() {
  Rng rng(3);
  Mlp net({3, 7, 5, 4}, rng);
  std::vector<double> sc(net.scratch_size()), out(4);
  const double x[3] = {0.3, -0.7, 1.1};
  const std::uint32_t k = 2; const double target = 0.4;
  net.accumulate(x, k, target);
  const auto g = net.gradient();
  double worst = 0;
  for (std::size_t i = 0; i < net.parameters(); ++i) {
    const double w0 = net.weights()[i], h = 1e-6;
    net.weights()[i] = w0 + h; net.forward(x, out.data(), sc.data()); const double lp = 0.5 * std::pow(out[k] - target, 2);
    net.weights()[i] = w0 - h; net.forward(x, out.data(), sc.data()); const double lm = 0.5 * std::pow(out[k] - target, 2);
    net.weights()[i] = w0;
    worst = std::max(worst, std::abs((lp - lm) / (2 * h) - g[i]));
  }
  std::printf("gradient check: %zu parameters, worst |numeric - analytic| = %.2e\n", net.parameters(), worst);
  // Regression: y = sin(3a) + b^2 - c with 2x32 hidden.
  Mlp f({3, 32, 32, 1}, rng); Rng d(5);
  auto fn = [](const double* v) { return std::sin(3 * v[0]) + v[1] * v[1] - v[2]; };
  for (int it = 0; it < 20000; ++it) {
    for (int b = 0; b < 32; ++b) { double v[3] = {2 * d.uniform() - 1, 2 * d.uniform() - 1, 2 * d.uniform() - 1}; f.accumulate(v, 0, fn(v)); }
    f.step(1e-3);
  }
  std::vector<double> s2(f.scratch_size()); double mse = 0;
  for (int i = 0; i < 2000; ++i) { double v[3] = {2 * d.uniform() - 1, 2 * d.uniform() - 1, 2 * d.uniform() - 1}; double o; f.forward(v, &o, s2.data()); mse += std::pow(o - fn(v), 2) / 2000; }
  std::printf("regression MSE after 20k Adam steps: %.5f (target variance ~0.9)\n", mse);
}
