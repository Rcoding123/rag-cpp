// rag/fusion.hpp — Reciprocal Rank Fusion of dense and lexical rankings.
//
//   rrf(d) = sum_i  w_i / (k0 + rank_i(d))      (rank is 1-based)
//
// RRF uses ranks only, so it needs no score calibration between cosine
// similarity and BM25 (their scales differ). Inputs are short (<= K hits),
// so a linear id-match on a stack array beats any hash map: everything
// stays in a few L1 lines and there is no allocation.
#pragma once

#include <algorithm>
#include <array>
#include <span>

#include "rag/topk.hpp"

namespace rag {

struct RrfParams {
  float k0 = 60.f;  // Cormack et al. 2009 default
  float w_dense = 1.f;
  float w_lexical = 1.f;
};

inline constexpr std::size_t kMaxFuseInput = 256;

template <class TK>
constexpr void rrf_fuse(std::span<const Hit> dense, std::span<const Hit> lexical, const RrfParams& p, TK& out) noexcept {
  std::array<Hit, 2 * kMaxFuseInput> acc{};
  const std::size_t na = std::min(dense.size(), kMaxFuseInput);
  const std::size_t nb = std::min(lexical.size(), kMaxFuseInput);
  std::size_t n = 0;
  for (std::size_t r = 0; r < na; ++r) acc[n++] = {p.w_dense / (p.k0 + static_cast<float>(r + 1)), dense[r].id};
  for (std::size_t r = 0; r < nb; ++r) {
    const float s = p.w_lexical / (p.k0 + static_cast<float>(r + 1));
    const auto id = lexical[r].id;
    std::size_t j = 0;
    while (j < na && acc[j].id != id) ++j;  // ids are unique within each list
    if (j < na) acc[j].score += s;
    else acc[n++] = {s, id};
  }
  for (std::size_t i = 0; i < n; ++i) out.push(acc[i].score, acc[i].id);
}

}  // namespace rag
