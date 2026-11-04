// rag/agent/mlp.hpp — a small multilayer perceptron with backpropagation and
// Adam, written from scratch (no ML library): the neural alternative to tile
// coding for value functions (E16).
//
//   sizes = {in, h1, ..., out}; ReLU on hidden layers, linear output.
//   He-uniform initialization from the project's seeded Rng, so training is
//   reproducible bit for bit.
//
// Training is per sample: accumulate() adds the gradient of
// 0.5 * (out[k] - target)^2 for one output k (the action taken, as in
// Q-learning), and step() applies the averaged gradient with Adam and clears
// it. forward() is const and uses caller-provided scratch, so a target
// network can be a plain copy evaluated alongside.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

#include "rag/agent/rng.hpp"

namespace rag::agent {

class Mlp {
 public:
  Mlp() = default;
  Mlp(std::vector<std::uint32_t> sizes, Rng& rng) : sizes_(std::move(sizes)) {
    std::size_t n = 0, units = 0;
    for (std::size_t l = 0; l + 1 < sizes_.size(); ++l) n += std::size_t{sizes_[l] + 1} * sizes_[l + 1];
    for (auto s : sizes_) units += s;
    w_.resize(n);
    g_.assign(n, 0.0);
    m_.assign(n, 0.0);
    v_.assign(n, 0.0);
    act_.resize(units);
    delta_.resize(units);
    std::size_t off = 0;
    for (std::size_t l = 0; l + 1 < sizes_.size(); ++l) {
      const double bound = std::sqrt(6.0 / sizes_[l]);  // He-uniform for ReLU
      for (std::uint32_t o = 0; o < sizes_[l + 1]; ++o) {
        for (std::uint32_t i = 0; i < sizes_[l]; ++i) w_[off++] = bound * (2 * rng.uniform() - 1);
        w_[off++] = 0.0;  // bias
      }
    }
  }

  [[nodiscard]] std::uint32_t inputs() const noexcept { return sizes_.front(); }
  [[nodiscard]] std::uint32_t outputs() const noexcept { return sizes_.back(); }
  [[nodiscard]] std::size_t parameters() const noexcept { return w_.size(); }

  // out[0..outputs) = network(x). `scratch` must hold scratch_size() doubles.
  [[nodiscard]] std::size_t scratch_size() const noexcept { return act_.size(); }
  void forward(const double* x, double* out, double* scratch) const noexcept {
    for (std::uint32_t i = 0; i < sizes_[0]; ++i) scratch[i] = x[i];
    propagate(scratch);
    const double* last = scratch + (act_.size() - sizes_.back());
    for (std::uint32_t k = 0; k < sizes_.back(); ++k) out[k] = last[k];
  }

  // Add the gradient of 0.5 * (out[k] - target)^2; returns out[k] - target.
  double accumulate(const double* x, std::uint32_t k, double target) {
    for (std::uint32_t i = 0; i < sizes_[0]; ++i) act_[i] = x[i];
    propagate(act_.data());
    const std::size_t L = sizes_.size() - 1;
    std::size_t out_base = act_.size() - sizes_[L];
    std::fill(delta_.begin(), delta_.end(), 0.0);
    const double err = act_[out_base + k] - target;
    delta_[out_base + k] = err;
    // Backward through the layers.
    std::size_t w_end = w_.size(), a_hi = out_base;
    for (std::size_t l = L; l-- > 0;) {
      const std::uint32_t in = sizes_[l], out = sizes_[l + 1];
      const std::size_t w_base = w_end - std::size_t{in + 1} * out;
      const std::size_t a_lo = a_hi - in;  // activations of layer l
      for (std::uint32_t o = 0; o < out; ++o) {
        const double d = delta_[a_hi + o];
        if (d == 0.0) continue;
        double* gw = &g_[w_base + std::size_t{o} * (in + 1)];
        const double* w = &w_[w_base + std::size_t{o} * (in + 1)];
        for (std::uint32_t i = 0; i < in; ++i) {
          gw[i] += d * act_[a_lo + i];
          delta_[a_lo + i] += d * w[i];
        }
        gw[in] += d;
      }
      if (l > 0)  // ReLU derivative on hidden activations
        for (std::uint32_t i = 0; i < in; ++i)
          if (act_[a_lo + i] <= 0.0) delta_[a_lo + i] = 0.0;
      w_end = w_base;
      a_hi = a_lo;
    }
    ++pending_;
    return err;
  }

  // Move this network's weights a fraction tau of the way toward other's
  // (Polyak / exponential moving average; E24). Shapes must match.
  void blend_toward(const Mlp& other, double tau) {
    if (tau >= 1) {  // exact copy (w + 1 * (o - w) can differ from o in the last bit)
      w_ = other.w_;
      return;
    }
    for (std::size_t i = 0; i < w_.size(); ++i) w_[i] += tau * (other.w_[i] - w_[i]);
  }

  // Apply the averaged accumulated gradient with Adam, then clear it.
  void step(double lr, double beta1 = 0.9, double beta2 = 0.999, double eps = 1e-8) {
    if (pending_ == 0) return;
    ++t_;
    const double inv = 1.0 / static_cast<double>(pending_);
    const double c1 = 1 - std::pow(beta1, static_cast<double>(t_)), c2 = 1 - std::pow(beta2, static_cast<double>(t_));
    for (std::size_t i = 0; i < w_.size(); ++i) {
      const double g = g_[i] * inv;
      m_[i] = beta1 * m_[i] + (1 - beta1) * g;
      v_[i] = beta2 * v_[i] + (1 - beta2) * g * g;
      w_[i] -= lr * (m_[i] / c1) / (std::sqrt(v_[i] / c2) + eps);
      g_[i] = 0.0;
    }
    pending_ = 0;
  }

  // Plain stochastic gradient descent on the averaged gradient (the ablation
  // baseline for Adam), then clear it.
  void sgd_step(double lr) {
    if (pending_ == 0) return;
    const double scale = lr / static_cast<double>(pending_);
    for (std::size_t i = 0; i < w_.size(); ++i) {
      w_[i] -= scale * g_[i];
      g_[i] = 0.0;
    }
    pending_ = 0;
  }

  // Parameter access (for gradient checks and soft target updates).
  [[nodiscard]] std::vector<double>& weights() noexcept { return w_; }
  [[nodiscard]] const std::vector<double>& weights() const noexcept { return w_; }
  [[nodiscard]] const std::vector<double>& gradient() const noexcept { return g_; }

 private:
  // In place: act[0..in) holds the input; fills every later layer.
  void propagate(double* act) const noexcept {
    std::size_t w_off = 0, a_in = 0;
    for (std::size_t l = 0; l + 1 < sizes_.size(); ++l) {
      const std::uint32_t in = sizes_[l], out = sizes_[l + 1];
      const std::size_t a_out = a_in + in;
      const bool hidden = l + 2 < sizes_.size();
      for (std::uint32_t o = 0; o < out; ++o) {
        const double* w = &w_[w_off + std::size_t{o} * (in + 1)];
        double s = w[in];
        for (std::uint32_t i = 0; i < in; ++i) s += w[i] * act[a_in + i];
        act[a_out + o] = hidden ? std::max(0.0, s) : s;
      }
      w_off += std::size_t{in + 1} * out;
      a_in = a_out;
    }
  }

  std::vector<std::uint32_t> sizes_;
  std::vector<double> w_, g_, m_, v_;  // weights (row per output unit, bias last), gradient, Adam moments
  std::vector<double> act_, delta_;    // training scratch
  std::uint64_t t_ = 0;
  std::uint32_t pending_ = 0;
};

}  // namespace rag::agent
