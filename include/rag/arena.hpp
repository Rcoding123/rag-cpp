// rag/arena.hpp — monotonic (bump-pointer) arena for document text.
//
// Latency notes:
//  * Allocation is a pointer bump + alignment mask on the common path.
//  * Blocks are never moved or freed until the arena dies, so a pointer handed
//    out once stays valid forever. Readers on other threads can therefore hold
//    std::string_view into the arena without any synchronisation beyond the
//    release/acquire publication of the chunk that references it.
//  * Only the (single) writer thread touches blocks_ / cur_ / end_.
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <memory>
#include <string_view>
#include <vector>

#include "rag/common.hpp"

namespace rag {

class Arena {
 public:
  explicit Arena(std::size_t block_bytes = std::size_t{1} << 20) : block_bytes_(block_bytes) {}

  Arena(const Arena&) = delete;
  Arena& operator=(const Arena&) = delete;

  [[nodiscard]] void* allocate(std::size_t bytes, std::size_t align = alignof(std::max_align_t)) {
    auto* p = align_up(cur_, align);
    if (p + bytes <= end_) [[likely]] {
      cur_ = p + bytes;
      used_ += bytes;
      return p;
    }
    return allocate_slow(bytes, align);
  }

  // Copy a string into the arena; the returned view is stable for the arena's lifetime.
  [[nodiscard]] std::string_view copy(std::string_view s) {
    if (s.empty()) return {};
    auto* dst = static_cast<char*>(allocate(s.size(), 1));
    std::memcpy(dst, s.data(), s.size());
    return {dst, s.size()};
  }

  [[nodiscard]] std::size_t bytes_used() const noexcept { return used_; }
  [[nodiscard]] std::size_t blocks() const noexcept { return blocks_.size(); }

 private:
  static std::byte* align_up(std::byte* p, std::size_t a) noexcept {
    const auto v = reinterpret_cast<std::uintptr_t>(p);
    return reinterpret_cast<std::byte*>((v + a - 1) & ~(std::uintptr_t{a} - 1));
  }

  void* allocate_slow(std::size_t bytes, std::size_t align) {
    const std::size_t size = std::max(block_bytes_, bytes + align);
    blocks_.push_back(make_aligned_array<std::byte>(size));
    cur_ = blocks_.back().get();
    end_ = cur_ + size;
    return allocate(bytes, align);
  }

  std::size_t block_bytes_;
  std::vector<AlignedArray<std::byte>> blocks_;
  std::byte* cur_ = nullptr;
  std::byte* end_ = nullptr;
  std::size_t used_ = 0;
};

}  // namespace rag
