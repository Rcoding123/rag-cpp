// rag/vector_store.hpp — fixed-capacity, append-only embedding store with
// single-writer / multi-reader lock-free publication.
//
// Layout (Struct of Arrays, each array one aligned allocation):
//   f32_    [capacity][D]  fp32 rows, 64 B aligned      -> exact scoring / rerank
//   i8_     [capacity][D]  int8 rows, 64 B aligned      -> 4x less memory traffic
//   scale_  [capacity]     dequantisation scale per row
//
// Concurrency protocol (SWMR):
//   writer: fill row n in all three arrays  ->  size_.store(n + 1, release)
//   reader: m = size_.load(acquire)         ->  rows [0, m) are fully visible
// Rows are never modified after publication, so readers need no locks, no
// retries, and never observe a torn row. size_ sits on its own cache line so
// the writer's stores do not invalidate the lines holding the read-only
// pointers that every reader touches (no false sharing).
#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <span>

#include "rag/common.hpp"
#include "rag/simd.hpp"
#include "rag/topk.hpp"

namespace rag {

template <std::size_t D>
  requires ValidDim<D>
class VectorStore {
 public:
  static constexpr std::size_t kDim = D;

  explicit VectorStore(std::size_t capacity)
      : capacity_(capacity),
        // Lazy: rows are written by append() before size_ publishes them, so
        // readers never fault, and unused capacity costs no physical memory.
        f32_(make_aligned_array<float>(capacity * D, Init::Lazy)),
        i8_(make_aligned_array<std::int8_t>(capacity * D, Init::Lazy)),
        scale_(make_aligned_array<float>(capacity, Init::Lazy)) {}

  // Single writer only. Returns false when full. `v` should be L2-normalised.
  bool append(std::span<const float, D> v) noexcept {
    const std::uint32_t n = size_.load(std::memory_order_relaxed);  // writer owns size_
    if (n >= capacity_) [[unlikely]] return false;
    float* dst = f32_.get() + std::size_t{n} * D;
    for (std::size_t i = 0; i < D; ++i) dst[i] = v[i];
    scale_[n] = simd::quantize_i8(dst, i8_.get() + std::size_t{n} * D, D);
    size_.store(n + 1, std::memory_order_release);  // publish
    return true;
  }

  [[nodiscard]] std::uint32_t size() const noexcept { return size_.load(std::memory_order_acquire); }
  [[nodiscard]] std::size_t capacity() const noexcept { return capacity_; }
  [[nodiscard]] const float* row(std::uint32_t i) const noexcept { return f32_.get() + std::size_t{i} * D; }
  [[nodiscard]] const std::int8_t* qrow(std::uint32_t i) const noexcept { return i8_.get() + std::size_t{i} * D; }
  [[nodiscard]] float qscale(std::uint32_t i) const noexcept { return scale_[i]; }

  // Exact cosine (= dot, rows normalised) over rows [begin, end).
  // `end` must come from size() on this thread (acquire) or be <= it.
  template <class TK>
  void search_exact(const float* q, std::uint32_t begin, std::uint32_t end, TK& out) const noexcept {
    std::uint32_t i = begin;
    float s[4];
    for (; i + 4 <= end; i += 4) {  // register-blocked: one q load feeds 4 rows
      simd::dot4(q, row(i), D, D, s);
      out.push(s[0], i); out.push(s[1], i + 1); out.push(s[2], i + 2); out.push(s[3], i + 3);
    }
    for (; i < end; ++i) out.push(simd::dot1(q, row(i), D), i);  // bit-identical to a dot4 lane
  }

  // ---------------------------------------------------------------- batched
  // Score nq <= kMaxBatch queries in ONE pass over rows [begin, end). The
  // scans are DRAM-bound, so streaming each row once for all queries
  // multiplies throughput: the row is loaded from memory once and then
  // reused from L1 against every query.
  static constexpr std::size_t kMaxBatch = 8;

  // `qs`: nq_padded fp32 query rows, contiguous with stride D, 64 B aligned,
  // nq_padded = nq rounded up to a multiple of 4 (pad rows may hold anything;
  // their scores are discarded). outs[j] receives query j's hits.
  // Each score is bit-identical to search_exact() on the same query.
  template <class TK>
  void search_exact_batch(const float* qs, std::size_t nq, std::uint32_t begin, std::uint32_t end,
                          TK* const* outs) const noexcept {
    const std::size_t padded = (nq + 3) & ~std::size_t{3};
    float s[kMaxBatch];
    for (std::uint32_t i = begin; i < end; ++i) {
      const float* r = row(i);
      for (std::size_t j = 0; j < padded; j += 4) simd::dot4(r, qs + j * D, D, D, s + j);  // row x 4 queries
      for (std::size_t j = 0; j < nq; ++j) outs[j]->push(s[j], i);
    }
  }

  // Batched int8 scan: qqs[j] is query j's int8 vector.
  template <class TK>
  void search_int8_batch(const std::int8_t* const* qqs, std::size_t nq, std::uint32_t begin, std::uint32_t end,
                         TK* const* outs) const noexcept {
    for (std::uint32_t i = begin; i < end; ++i) {
      const std::int8_t* r = qrow(i);
      RAG_PREFETCH(r + 8 * D);
      const float sc = scale_[i];
      for (std::size_t j = 0; j < nq; ++j) outs[j]->push(static_cast<float>(simd::dot_i8(qqs[j], r, D)) * sc, i);
    }
  }

  // Approximate scan over int8 rows. Scores are dequantised dot products;
  // the query scale is a positive constant so it is folded out of ranking.
  template <class TK>
  void search_int8(const std::int8_t* qq, std::uint32_t begin, std::uint32_t end, TK& out) const noexcept {
    for (std::uint32_t i = begin; i < end; ++i) {
      RAG_PREFETCH(qrow(i) + 8 * D);  // stay ~8 rows (3 KiB @ D=384) ahead of the stream
      out.push(static_cast<float>(simd::dot_i8(qq, qrow(i), D)) * scale_[i], i);
    }
  }

  // Two-stage search: int8 scan keeps `cand` candidates, fp32 reranks them.
  // Recall is close to exact while the bulk scan moves 4x fewer bytes.
  template <class TKCand, class TK>
  void search_two_stage(const float* q, const std::int8_t* qq, std::uint32_t begin, std::uint32_t end,
                        TKCand& cand, TK& out) const noexcept {
    search_int8(qq, begin, end, cand);
    for (const Hit& h : cand.sorted()) out.push(simd::dot(q, row(h.id), D), h.id);
  }

 private:
  std::size_t capacity_;
  AlignedArray<float> f32_;
  AlignedArray<std::int8_t> i8_;
  AlignedArray<float> scale_;
  alignas(kCacheLine) std::atomic<std::uint32_t> size_{0};
  [[maybe_unused]] char pad_[kCacheLine - sizeof(std::atomic<std::uint32_t>)]{};
};

}  // namespace rag
