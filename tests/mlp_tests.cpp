// mlp_tests.cpp — the from-scratch neural network (E16): backpropagation
// against finite differences, Adam on a regression, determinism.
#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <vector>

#include "rag/agent/mlp.hpp"

using namespace rag::agent;

TEST(Mlp, GradientMatchesFiniteDifferences) {
  Rng rng(3);
  Mlp net({3, 7, 5, 4}, rng);
  std::vector<double> scratch(net.scratch_size());
  std::array<double, 4> out{};
  const std::array<double, 3> x = {0.3, -0.7, 1.1};
  for (std::uint32_t k = 0; k < 4; ++k) {
    Mlp n = net;
    const double target = 0.4 - 0.2 * k;
    n.accumulate(x.data(), k, target);
    const std::vector<double> g = n.gradient();
    for (std::size_t i = 0; i < n.parameters(); ++i) {
      const double w0 = n.weights()[i], h = 1e-6;
      n.weights()[i] = w0 + h;
      n.forward(x.data(), out.data(), scratch.data());
      const double lp = 0.5 * (out[k] - target) * (out[k] - target);
      n.weights()[i] = w0 - h;
      n.forward(x.data(), out.data(), scratch.data());
      const double lm = 0.5 * (out[k] - target) * (out[k] - target);
      n.weights()[i] = w0;
      ASSERT_NEAR((lp - lm) / (2 * h), g[i], 1e-7) << "output " << k << " parameter " << i;
    }
  }
}

TEST(Mlp, AdamFitsASmoothFunctionAndIsDeterministic) {
  // Measured: MSE 0.0004 after 20,000 steps of 32 samples; here 4,000 steps.
  auto fn = [](const std::array<double, 3>& v) { return std::sin(3 * v[0]) + v[1] * v[1] - v[2]; };
  auto train = [&](std::uint64_t seed) {
    Rng init(seed), data(seed + 1);
    Mlp f({3, 32, 32, 1}, init);
    for (int it = 0; it < 4000; ++it) {
      for (int b = 0; b < 32; ++b) {
        const std::array<double, 3> v = {2 * data.uniform() - 1, 2 * data.uniform() - 1, 2 * data.uniform() - 1};
        f.accumulate(v.data(), 0, fn(v));
      }
      f.step(1e-3);
    }
    return f;
  };
  const Mlp f = train(5);
  Rng test(99);
  std::vector<double> scratch(f.scratch_size());
  double mse = 0;
  for (int i = 0; i < 2000; ++i) {
    const std::array<double, 3> v = {2 * test.uniform() - 1, 2 * test.uniform() - 1, 2 * test.uniform() - 1};
    std::array<double, 1> o{};
    f.forward(v.data(), o.data(), scratch.data());
    mse += (o[0] - fn(v)) * (o[0] - fn(v)) / 2000;
  }
  EXPECT_LT(mse, 0.02);
  EXPECT_EQ(train(5).weights(), f.weights());  // bit-identical rerun
}
