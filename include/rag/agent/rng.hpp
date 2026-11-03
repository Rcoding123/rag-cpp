// rag/agent/rng.hpp — fast, deterministic random numbers for simulation and
// exploration.
//
// xoshiro256** (Blackman & Vigna): 4 x 64-bit state, ~1 ns per draw, passes
// BigCrush. Seeded through splitmix64 so that nearby seeds (1, 2, 3, ...)
// give unrelated streams — evaluation runs use seed + episode as a cheap way
// to replay the *same* market for every policy (common random numbers).
// std::mt19937 is avoided on purpose: 2.5 KB of state per generator and
// implementation-defined distributions would make results differ between
// standard libraries.
#pragma once

#include <cmath>
#include <cstdint>
#include <numbers>

namespace rag::agent {

class Rng {
 public:
  explicit constexpr Rng(std::uint64_t seed = 1) noexcept { reseed(seed); }

  constexpr void reseed(std::uint64_t seed) noexcept {
    for (auto& s : s_) {
      seed += 0x9E3779B97F4A7C15ull;
      std::uint64_t z = seed;
      z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
      z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
      s = z ^ (z >> 31);
    }
    has_spare_ = false;
  }

  constexpr std::uint64_t next() noexcept {
    const std::uint64_t result = rotl(s_[1] * 5, 7) * 9;
    const std::uint64_t t = s_[1] << 17;
    s_[2] ^= s_[0];
    s_[3] ^= s_[1];
    s_[1] ^= s_[2];
    s_[0] ^= s_[3];
    s_[2] ^= t;
    s_[3] = rotl(s_[3], 45);
    return result;
  }

  // Uniform in [0, 1) with 53 random bits.
  constexpr double uniform() noexcept { return static_cast<double>(next() >> 11) * 0x1.0p-53; }

  // Uniform integer in [0, n) (Lemire's multiply-shift; bias < 2^-32).
  constexpr std::uint32_t below(std::uint32_t n) noexcept {
    return static_cast<std::uint32_t>((static_cast<std::uint64_t>(static_cast<std::uint32_t>(next() >> 32)) * n) >> 32);
  }

  // Standard normal (Marsaglia polar method; caches the second variate).
  double normal() noexcept {
    if (has_spare_) {
      has_spare_ = false;
      return spare_;
    }
    double u, v, s;
    do {
      u = 2.0 * uniform() - 1.0;
      v = 2.0 * uniform() - 1.0;
      s = u * u + v * v;
    } while (s >= 1.0 || s == 0.0);
    const double m = std::sqrt(-2.0 * std::log(s) / s);
    spare_ = v * m;
    has_spare_ = true;
    return u * m;
  }

 private:
  static constexpr std::uint64_t rotl(std::uint64_t x, int k) noexcept { return (x << k) | (x >> (64 - k)); }
  std::uint64_t s_[4]{};
  double spare_ = 0.0;
  bool has_spare_ = false;
};

}  // namespace rag::agent
