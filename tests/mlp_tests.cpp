// mlp_tests.cpp — the from-scratch neural network (E16): backpropagation
// against finite differences, Adam on a regression, determinism.
#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <vector>

#include "rag/agent/mlp.hpp"
#include "rag/agent/trainer.hpp"

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

TEST(Mlp, BlendTowardMovesWeightsLinearly) {
  Rng r1(1), r2(2);
  const Mlp a({3, 5, 2}, r1), b({3, 5, 2}, r2);
  const std::array<double, 3> x = {0.4, -0.2, 0.9};
  std::vector<double> scratch(a.scratch_size());
  auto out = [&](const Mlp& m) {
    std::array<double, 2> o{};
    m.forward(x.data(), o.data(), scratch.data());
    return o;
  };
  Mlp m = a;
  m.blend_toward(b, 0.0);
  EXPECT_EQ(out(m), out(a));
  m.blend_toward(b, 1.0);
  EXPECT_EQ(out(m), out(b));
  ASSERT_THROW((void)Agent<ExecutionEnv>(*std::make_unique<ExecutionEnv>(ExecutionParams{}),
                                         AgentConfig<3>{.learner = LearnerKind::NeuralQ, .nn_polyak = 1.5}),
               std::invalid_argument);
}

namespace {
// Train a small NeuralQ with a given averaging rate; return it.
std::unique_ptr<Agent<ExecutionEnv>> small_neural(ExecutionEnv& env, double polyak) {
  AgentConfig<3> c;
  c.learner = LearnerKind::NeuralQ;
  c.nn_hidden = 16;
  c.nn_warmup = 200;
  c.eps_decay_episodes = 200;
  c.nn_polyak = polyak;
  c.seed = 9;
  auto agent = std::make_unique<Agent<ExecutionEnv>>(env, c);
  Trainer<ExecutionEnv> tr(*agent, env, {.eval_every = 0, .train_seed = 9, .detect_drift = false});
  tr.run(300);
  return agent;
}
}  // namespace

TEST(NeuralQ, WeightAveragingLeavesTrainingUnchangedAndOnlyAffectsGreedyActing) {
  ExecutionEnv e0{ExecutionParams{}}, e1{ExecutionParams{}}, e2{ExecutionParams{}};
  auto plain = small_neural(e0, 0.0), full = small_neural(e1, 1.0), avg = small_neural(e2, 0.01);
  Rng r(4);
  int differ = 0;
  for (int i = 0; i < 200; ++i) {
    const ExecutionObs o{static_cast<std::uint32_t>(r.below(20)), static_cast<std::uint32_t>(r.below(101)), 0.1 * r.normal()};
    std::array<double, 5> q0{}, q2{};
    plain->q_values(o, q0.data());
    avg->q_values(o, q2.data());
    EXPECT_EQ(q0, q2) << "averaging must not change the online network's training";
    // nn_polyak = 1: the average is the online network after every update.
    EXPECT_EQ(full->act(o, false), plain->act(o, false));
    EXPECT_EQ(full->act_as(full->snapshot(), o), plain->act(o, false));
    // The deployed policy (snapshot) is the averaged one.
    EXPECT_EQ(avg->act_as(avg->snapshot(), o), avg->act(o, false));
    differ += avg->act(o, false) != plain->act(o, false);
  }
  EXPECT_GT(differ, 0) << "a slow average should act differently from the online network somewhere";
}

TEST(Mlp, SaveLoadRoundTripAndRejectsCorruptFiles) {
  Rng rng(12);
  const Mlp net({4, 9, 3}, rng);
  const auto path = std::filesystem::path(::testing::TempDir()) / "mlp_roundtrip.net";
  net.save(path);
  const Mlp back = Mlp::load(path);
  EXPECT_EQ(back.parameters(), net.parameters());
  std::vector<double> s1(net.scratch_size()), s2(back.scratch_size());
  std::array<double, 3> a{}, b{};
  const std::array<double, 4> x = {0.1, -1.2, 0.7, 2.0};
  net.forward(x.data(), a.data(), s1.data());
  back.forward(x.data(), b.data(), s2.data());
  EXPECT_EQ(a, b);
  std::vector<char> bytes;
  {
    std::ifstream f(path, std::ios::binary);
    bytes.assign(std::istreambuf_iterator<char>(f), {});
  }
  bytes[bytes.size() / 2] ^= 0x40;
  {
    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    f.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
  }
  EXPECT_THROW((void)Mlp::load(path), rag::io::FormatError);
  std::filesystem::remove(path);
}
