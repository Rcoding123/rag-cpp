// rag/ivf_index.hpp — IVF-Flat approximate nearest-neighbour index
// (spherical k-means coarse quantiser + contiguous inverted lists).
//
// Built offline from a snapshot of the VectorStore (cold path, allocates);
// queried without allocation. Each inverted list stores its vectors
// *contiguously* (CSR layout, rows copied and reordered), so probing a list
// is a sequential stream the hardware prefetcher handles perfectly — no
// pointer chasing, no gathers.
//
// Query cost ~ nlist*D (centroids, L1/L2 resident) + (nprobe/nlist)*N*D.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <stdexcept>
#include <vector>

#include "rag/common.hpp"
#include "rag/simd.hpp"
#include "rag/topk.hpp"
#include "rag/vector_store.hpp"

namespace rag {

struct IvfParams {
  std::uint32_t nlist = 0;           // 0 => ~sqrt(n), rounded to a multiple of 4
  std::uint32_t train_iters = 10;
  std::uint32_t max_train_points_per_list = 256;
  std::uint64_t seed = 0x5EED;
};

template <std::size_t D>
  requires ValidDim<D>
class IvfIndex {
 public:
  static constexpr std::uint32_t kMaxProbe = 256;

  // Index rows [0, n) of `store`. Cold path.
  void build(const VectorStore<D>& store, std::uint32_t n, IvfParams p = {}) {
    if (n == 0) { covered_ = 0; nlist_ = 0; return; }
    nlist_ = p.nlist ? p.nlist : std::max<std::uint32_t>(4, static_cast<std::uint32_t>(std::sqrt(double(n))) & ~3u);
    nlist_ = std::min(nlist_, n);
    centroids_ = make_aligned_array<float>(std::size_t{nlist_} * D);
    train(store, n, p);

    // Assign every row to its nearest centroid, then lay lists out as CSR.
    std::vector<std::uint32_t> assign(n);
    for (std::uint32_t i = 0; i < n; ++i) assign[i] = nearest(store.row(i));
    offsets_.assign(nlist_ + 1, 0);
    for (auto a : assign) ++offsets_[a + 1];
    std::partial_sum(offsets_.begin(), offsets_.end(), offsets_.begin());
    ids_.resize(n);
    vecs_ = make_aligned_array<float>(std::size_t{n} * D);
    std::vector<std::uint32_t> cursor(offsets_.begin(), offsets_.end() - 1);
    for (std::uint32_t i = 0; i < n; ++i) {
      const std::uint32_t slot = cursor[assign[i]]++;
      ids_[slot] = i;
      std::copy_n(store.row(i), D, vecs_.get() + std::size_t{slot} * D);
    }
    covered_ = n;
  }

  // Probe the `nprobe` closest lists. Pushes (score, original row id).
  template <class TK>
  void search(const float* q, std::uint32_t nprobe, TK& out) const noexcept {
    if (nlist_ == 0) return;
    TopK<kMaxProbe> probes(std::min({nprobe, nlist_, kMaxProbe}));
    score_centroids(q, probes);
    for (const Hit& pr : probes.sorted()) {
      std::uint32_t i = offsets_[pr.id];
      const std::uint32_t e = offsets_[pr.id + 1];
      float s[4];
      for (; i + 4 <= e; i += 4) {
        simd::dot4(q, vecs_.get() + std::size_t{i} * D, D, D, s);
        out.push(s[0], ids_[i]); out.push(s[1], ids_[i + 1]);
        out.push(s[2], ids_[i + 2]); out.push(s[3], ids_[i + 3]);
      }
      for (; i < e; ++i) out.push(simd::dot(q, vecs_.get() + std::size_t{i} * D, D), ids_[i]);
    }
  }

  [[nodiscard]] std::uint32_t covered() const noexcept { return covered_; }
  [[nodiscard]] std::uint32_t nlist() const noexcept { return nlist_; }
  [[nodiscard]] std::uint32_t list_size(std::uint32_t l) const noexcept { return offsets_[l + 1] - offsets_[l]; }

 private:
  template <class TK>
  void score_centroids(const float* q, TK& out) const noexcept {
    std::uint32_t c = 0;
    float s[4];
    for (; c + 4 <= nlist_; c += 4) {
      simd::dot4(q, centroids_.get() + std::size_t{c} * D, D, D, s);
      for (int j = 0; j < 4; ++j) out.push(s[j], c + j);
    }
    for (; c < nlist_; ++c) out.push(simd::dot(q, centroids_.get() + std::size_t{c} * D, D), c);
  }

  std::uint32_t nearest(const float* v) const noexcept {
    TopK<1> best(1);
    score_centroids(v, best);
    return best.sorted()[0].id;
  }

  static std::uint64_t splitmix(std::uint64_t& s) noexcept {
    std::uint64_t z = (s += 0x9E3779B97F4A7C15ull);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
  }

  // Spherical k-means (Lloyd iterations on the unit sphere) over a sample.
  void train(const VectorStore<D>& store, std::uint32_t n, const IvfParams& p) {
    std::uint64_t rng = p.seed;
    std::vector<std::uint32_t> perm(n);
    std::iota(perm.begin(), perm.end(), 0u);
    const auto m = static_cast<std::uint32_t>(std::min<std::uint64_t>(n, std::uint64_t{nlist_} * p.max_train_points_per_list));
    for (std::uint32_t i = 0; i < m; ++i) std::swap(perm[i], perm[i + splitmix(rng) % (n - i)]);  // partial Fisher-Yates
    for (std::uint32_t c = 0; c < nlist_; ++c) std::copy_n(store.row(perm[c]), D, centroids_.get() + std::size_t{c} * D);

    std::vector<float> sums(std::size_t{nlist_} * D);
    std::vector<std::uint32_t> counts(nlist_);
    for (std::uint32_t it = 0; it < p.train_iters; ++it) {
      std::fill(sums.begin(), sums.end(), 0.f);
      std::fill(counts.begin(), counts.end(), 0u);
      for (std::uint32_t j = 0; j < m; ++j) {
        const float* v = store.row(perm[j]);
        const std::uint32_t c = nearest(v);
        ++counts[c];
        float* sc = sums.data() + std::size_t{c} * D;
        for (std::size_t d = 0; d < D; ++d) sc[d] += v[d];
      }
      for (std::uint32_t c = 0; c < nlist_; ++c) {
        float* dst = centroids_.get() + std::size_t{c} * D;
        if (counts[c] == 0) {  // empty cluster: reseed from a random sample point
          std::copy_n(store.row(perm[splitmix(rng) % m]), D, dst);
          continue;
        }
        std::copy_n(sums.data() + std::size_t{c} * D, D, dst);
        simd::normalize(dst, D);
      }
    }
  }

  std::uint32_t nlist_ = 0;
  std::uint32_t covered_ = 0;
  AlignedArray<float> centroids_;
  std::vector<std::uint32_t> offsets_;  // nlist + 1, read-only after build
  std::vector<std::uint32_t> ids_;      // slot -> original row id
  AlignedArray<float> vecs_;            // rows reordered by list
};

}  // namespace rag
