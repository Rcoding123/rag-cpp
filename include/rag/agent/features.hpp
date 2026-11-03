// rag/agent/features.hpp — perception: turning an observation into features
// a learner can generalize over.
//
// Two encoders, matching the two learners:
//
//   TileCoder            sparse, for the parametric learner (TileQ). Several
//                        overlapping grids ("tilings"), each offset by a
//                        fraction of a cell; a point activates exactly one
//                        cell per tiling. Nearby states share most active
//                        cells, so what is learned in one state spreads to its
//                        neighbours, by an amount that shrinks with distance.
//
//   RandomFourierEncoder dense, for the episodic memory. Maps a state into a
//                        D-dimensional unit vector whose dot products
//                        approximate a Gaussian similarity between states
//                        (Rahimi & Recht, 2007). Then "recall similar past
//                        situations" is exactly the nearest-neighbour search
//                        the RAG engine runs over text embeddings, and it
//                        reuses the same SIMD kernels and top-K selection.
#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <numbers>
#include <stdexcept>
#include <vector>

#include "rag/agent/rng.hpp"
#include "rag/common.hpp"
#include "rag/simd.hpp"

namespace rag::agent {

inline constexpr std::uint32_t kMaxTilings = 32;

template <std::size_t Dims>
class TileCoder {
 public:
  TileCoder(const std::array<double, Dims>& lo, const std::array<double, Dims>& hi,
            const std::array<std::uint32_t, Dims>& bins, std::uint32_t tilings)
      : lo_(lo), bins_(bins), tilings_(tilings) {
    if (tilings == 0 || tilings > kMaxTilings) throw std::invalid_argument("TileCoder: tilings must be in [1, 32]");
    cells_ = 1;
    for (std::size_t d = 0; d < Dims; ++d) {
      if (bins[d] == 0 || !(hi[d] > lo[d])) throw std::invalid_argument("TileCoder: need bins > 0 and hi > lo");
      scale_[d] = bins[d] / (hi[d] - lo[d]);
      stride_[d] = cells_;
      cells_ *= bins[d] + 1;  // +1: the offset tilings overhang the range by one cell
    }
    // Asymmetric offsets (Sutton & Barto, 9.5.4): tiling k is shifted by
    // k * (1, 3, 5, ...) / tilings of a cell, which avoids the diagonal
    // artefacts of shifting every dimension by the same amount.
    for (std::uint32_t k = 0; k < tilings; ++k)
      for (std::size_t d = 0; d < Dims; ++d)
        offset_[k][d] = std::fmod(static_cast<double>(k) * static_cast<double>(2 * d + 1) / tilings, 1.0);
  }

  [[nodiscard]] std::uint32_t tilings() const noexcept { return tilings_; }
  [[nodiscard]] std::size_t size() const noexcept { return std::size_t{cells_} * tilings_; }

  // Writes one active feature index per tiling into idx[0..tilings).
  // Inputs outside [lo, hi] are clamped onto the boundary cells.
  void encode(const std::array<double, Dims>& x, std::uint32_t* idx) const noexcept {
    std::array<double, Dims> u;
    for (std::size_t d = 0; d < Dims; ++d)
      u[d] = std::clamp((x[d] - lo_[d]) * scale_[d], 0.0, static_cast<double>(bins_[d]));
    for (std::uint32_t k = 0; k < tilings_; ++k) {
      std::uint32_t flat = 0;
      for (std::size_t d = 0; d < Dims; ++d) {
        const auto c = std::min(static_cast<std::uint32_t>(u[d] + offset_[k][d]), bins_[d]);
        flat += c * stride_[d];
      }
      idx[k] = k * cells_ + flat;
    }
  }

 private:
  std::array<double, Dims> lo_, scale_{};
  std::array<std::uint32_t, Dims> bins_, stride_{};
  std::uint32_t tilings_, cells_ = 1;
  std::array<std::array<double, Dims>, kMaxTilings> offset_{};
};

template <std::size_t Dims, std::size_t D>
  requires ValidDim<D>
class RandomFourierEncoder {
 public:
  static constexpr std::size_t kDim = D;

  // `width[d]`: the distance along dimension d at which similarity has fallen
  // to exp(-1/2) ~ 0.61 (the kernel's length scale).
  RandomFourierEncoder(const std::array<double, Dims>& width, std::uint64_t seed) {
    Rng rng(seed);
    for (std::size_t i = 0; i < D; ++i) {
      for (std::size_t d = 0; d < Dims; ++d) {
        if (!(width[d] > 0)) throw std::invalid_argument("RandomFourierEncoder: widths must be > 0");
        w_[i][d] = rng.normal() / width[d];
      }
      b_[i] = 2.0 * std::numbers::pi * rng.uniform();
    }
  }

  // Writes a unit-length embedding of x into out[0..D).
  void encode(const std::array<double, Dims>& x, float* out) const noexcept {
    double norm = 0;
    for (std::size_t i = 0; i < D; ++i) {
      double p = b_[i];
      for (std::size_t d = 0; d < Dims; ++d) p += w_[i][d] * x[d];
      const double z = std::cos(p);
      out[i] = static_cast<float>(z);
      norm += z * z;
    }
    const auto inv = static_cast<float>(norm > 0 ? 1.0 / std::sqrt(norm) : 0.0);
    for (std::size_t i = 0; i < D; ++i) out[i] *= inv;
  }

 private:
  std::array<std::array<double, Dims>, D> w_{};
  std::array<double, D> b_{};
};

}  // namespace rag::agent
