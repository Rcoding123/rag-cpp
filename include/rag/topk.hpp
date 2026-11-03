// rag/topk.hpp — fixed-capacity top-K selection on the stack.
//
// Latency notes:
//  * Storage is std::array<Hit, Cap>: no allocation, lives in the caller's frame.
//  * A cached admission threshold turns the common case (candidate worse
//    than the current K-th best) into one compare + a well-predicted branch;
//    after warm-up nearly every candidate takes the [[likely]] reject path.
//  * The heap is a min-heap on "goodness", so the root is the worst kept hit.
//  * Ordering is total (score desc, then id asc) so results are deterministic
//    even with tied scores — important for reproducible retrieval tests.
#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>

namespace rag {

struct Hit {
  float score;
  std::uint32_t id;
  friend constexpr bool operator==(const Hit&, const Hit&) = default;
};

// a ranks ahead of b
constexpr bool better(const Hit& a, const Hit& b) noexcept {
  return a.score > b.score || (a.score == b.score && a.id < b.id);
}

template <std::size_t Cap>
class TopK {
  static_assert(Cap > 0);

 public:
  static constexpr std::size_t kCapacity = Cap;

  constexpr explicit TopK(std::size_t k = Cap) noexcept { reset(k); }

  // `min_score`: candidates must score >= min_score to be admitted at all.
  // It is folded into the admission threshold, so it costs nothing per push.
  constexpr void reset(std::size_t k, float min_score = -std::numeric_limits<float>::infinity()) noexcept {
    k_ = std::min(k, Cap);
    size_ = 0;
    sorted_ = false;
    // k == 0: a NaN threshold makes `score >= threshold_` false for every
    // input, so nothing is ever admitted and the heap is never touched.
    threshold_ = k_ ? min_score : std::numeric_limits<float>::quiet_NaN();
  }

  // Scores strictly below threshold() can never be admitted (NaN when k == 0).
  [[nodiscard]] constexpr float threshold() const noexcept { return threshold_; }

  constexpr void push(float score, std::uint32_t id) noexcept {
    // Written as !(>=) so NaN scores are rejected too. Ties with the threshold
    // pass through to the id tie-break below.
    if (!(score >= threshold_)) [[likely]] return;
    if (sorted_) [[unlikely]] {  // push after sorted(): restore heap order first
      std::make_heap(heap_.begin(), heap_.begin() + size_, better);
      sorted_ = false;
    }
    const Hit h{score, id};
    if (size_ < k_) {
      heap_[size_++] = h;
      std::push_heap(heap_.begin(), heap_.begin() + size_, better);
      if (size_ == k_) threshold_ = heap_[0].score;
    } else if (better(h, heap_[0])) {
      std::pop_heap(heap_.begin(), heap_.begin() + size_, better);
      heap_[size_ - 1] = h;
      std::push_heap(heap_.begin(), heap_.begin() + size_, better);
      threshold_ = heap_[0].score;
    }
  }

  // Sort in place (best first) and return a view. Idempotent; a later
  // push() transparently rebuilds the heap.
  [[nodiscard]] constexpr std::span<const Hit> sorted() noexcept {
    if (!sorted_) {
      std::sort_heap(heap_.begin(), heap_.begin() + size_, better);
      sorted_ = true;
    }
    return {heap_.data(), size_};
  }

  [[nodiscard]] constexpr std::size_t size() const noexcept { return size_; }
  [[nodiscard]] constexpr std::size_t k() const noexcept { return k_; }

 private:
  std::array<Hit, Cap> heap_{};
  std::size_t size_ = 0;
  std::size_t k_ = 0;
  float threshold_ = 0.f;
  bool sorted_ = false;
};

}  // namespace rag
