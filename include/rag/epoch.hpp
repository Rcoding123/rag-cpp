// rag/epoch.hpp — epoch-based reclamation (EBR) for immutable snapshots.
//
// Readers announce "I may be holding a pointer read at epoch e" in a
// per-reader, cache-line-padded slot; the writer frees a retired object only
// once no slot holds an epoch older than the one the object was retired in.
//
// Reader cost per critical section: one load of the global epoch and two
// stores to the reader's *own* cache line. There are no read-modify-write
// atomics and no shared-line writes, so readers never contend with each other.
// Writer cost per reclamation check: one load per slot (kMaxReaders lines).
//
// Correctness (all operations below are seq_cst, so they share one total order):
//   writer: old = ptr.exchange(new) ; e = ++epoch ; ... scan: every slot is 0 or >= e  => free(old)
//   reader: slot = epoch.load()     ; p = ptr.load() ; use p ; slot = 0
// If the writer's scan saw a reader's slot as 0, the reader's slot store comes
// later in the total order, so its ptr.load() also comes after the exchange and
// returns `new`. If the scan saw a value >= e, that reader loaded the epoch after
// the increment, which is again after the exchange. Either way that reader
// cannot be holding `old`. A slot holding a value < e blocks the free.
#pragma once

#include <array>
#include <atomic>
#include <cstdint>
#include <stdexcept>
#include <utility>

#include "rag/common.hpp"

namespace rag {

class EpochDomain {
 public:
  static constexpr std::size_t kMaxReaders = 128;

  struct alignas(kCacheLine) Slot {
    std::atomic<std::uint64_t> active{0};  // 0 = quiescent, else epoch at entry
    std::atomic<bool> in_use{false};       // claimed by a Reader handle
    std::uint32_t depth = 0;               // nesting depth (owning thread only)
  };
  static_assert(sizeof(Slot) == kCacheLine, "one slot per cache line: no false sharing between readers");

  // Registration handle owning one slot. Movable, not copyable. Claiming a slot
  // is a cold-path CAS scan, done once per reader thread.
  class Reader {
   public:
    Reader() = default;
    explicit Reader(Slot* s) noexcept : slot_(s) {}
    Reader(Reader&& o) noexcept : slot_(std::exchange(o.slot_, nullptr)) {}
    Reader& operator=(Reader&& o) noexcept {
      if (this != &o) { release(); slot_ = std::exchange(o.slot_, nullptr); }
      return *this;
    }
    Reader(const Reader&) = delete;
    Reader& operator=(const Reader&) = delete;
    ~Reader() { release(); }
    [[nodiscard]] Slot* slot() const noexcept { return slot_; }

   private:
    void release() noexcept {
      if (slot_) {
        slot_->active.store(0, std::memory_order_release);
        slot_->depth = 0;
        slot_->in_use.store(false, std::memory_order_release);
        slot_ = nullptr;
      }
    }
    Slot* slot_ = nullptr;
  };

  // RAII critical section. Nestable on the same Reader.
  class Guard {
   public:
    Guard(const EpochDomain& d, Slot& s) noexcept : s_(s) {
      if (s_.depth++ == 0) s_.active.store(d.epoch_.load(std::memory_order_seq_cst), std::memory_order_seq_cst);
    }
    Guard(const Guard&) = delete;
    Guard& operator=(const Guard&) = delete;
    ~Guard() {
      if (--s_.depth == 0) s_.active.store(0, std::memory_order_release);
    }

   private:
    Slot& s_;
  };

  [[nodiscard]] Reader register_reader() {
    for (Slot& s : slots_) {
      bool expected = false;
      if (!s.in_use.load(std::memory_order_relaxed) &&
          s.in_use.compare_exchange_strong(expected, true, std::memory_order_acq_rel))
        return Reader(&s);
    }
    throw std::runtime_error("EpochDomain: more than kMaxReaders live QueryContexts");
  }

  // Writer: start a new epoch after unpublishing an object; retire it with the result.
  std::uint64_t advance() noexcept { return epoch_.fetch_add(1, std::memory_order_seq_cst) + 1; }

  // Writer: true once no reader can still hold an object retired at epoch `e`.
  [[nodiscard]] bool quiescent_since(std::uint64_t e) const noexcept {
    for (const Slot& s : slots_) {
      const std::uint64_t a = s.active.load(std::memory_order_seq_cst);
      if (a != 0 && a < e) return false;
    }
    return true;
  }

  [[nodiscard]] std::size_t readers() const noexcept {
    std::size_t n = 0;
    for (const Slot& s : slots_) n += s.in_use.load(std::memory_order_relaxed);
    return n;
  }

 private:
  alignas(kCacheLine) std::atomic<std::uint64_t> epoch_{1};  // starts at 1: 0 means quiescent
  std::array<Slot, kMaxReaders> slots_{};
};

}  // namespace rag
