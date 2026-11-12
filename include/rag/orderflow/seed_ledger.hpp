// rag/orderflow/seed_ledger.hpp — the seed blocks of the order-flow
// experiments (docs/ai/NEXT_SESSION.md section 4) as code, and a process-wide
// guard that keeps sealed blocks away from code that trains or selects.
//
// The OF15 Python pipeline (python/ragof_module.cpp, the `_ragof` module)
// names episodes only through this table: a block name and a block-relative
// index, never a raw seed. The C++ programs (of_train, of2..of8) keep their
// own literals; this header does not change them.
//
//   block       seeds                         sealed  privileged outputs
//   train       run seeds 1 .. 700M - 1       no      yes (teacher / oracle training)
//   validation  700M + [0, 200M)              no      no
//   test        900M + [0, 10M)  (tick test 910M+ is not exposed)  yes  no
//   stress      920M + [0, 10M)               yes     no
//   days        930M + [0, 10M)               yes     no
//   pairing     940M + [0, 10M)               no      no
//   confirm     950M + [0, 10M)  orchestrator only, with a claim id   yes  no
//   viz         960M + ...       look, never select                no   yes
//   real        recorded ticks (replay; not seeds)                 yes  no
//
// Process roles (Ledger): a process starts as "none". Building an
// environment on train, validation or pairing makes it a "trainer"; a trainer
// can never open a sealed block. open_sealed() makes it a "scorer"; a scorer
// can no longer build train, validation or pairing environments (so it can't
// tune after seeing sealed episodes). forbid_sealed() closes sealed blocks for
// the rest of the process (every training / HPO / live worker calls it
// first). viz is neutral: probes for looking at training.
//
// This guards against accidents, not against a determined caller (a fresh
// process can always open a block); the Python pipeline adds the audit trail.
#pragma once

#include <array>
#include <cstdint>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace rag::orderflow::seeds {

enum class Block : int { Train, Validation, Test, Stress, Days, Pairing, Confirm, Viz, Real };
inline constexpr int kBlocks = 9;

struct BlockInfo {
  const char* name;
  std::uint64_t base, size;  // seeds base .. base + size - 1 (train: run seeds)
  bool sealed;               // closed until open_sealed() in this process
  bool privileged_ok;        // hidden simulator state may be exported (teacher / display only)
  bool indexed;              // EpisodeSource::indexed(name, first, count) accepts it
  bool needs_claim;          // confirm: orchestrator claim id required
};

inline constexpr std::array<BlockInfo, kBlocks> kBlockTable = {{
    {"train", 1, 699'999'999, false, true, false, false},
    {"validation", 700'000'000, 200'000'000, false, false, true, false},
    {"test", 900'000'000, 10'000'000, true, false, true, false},
    {"stress", 920'000'000, 10'000'000, true, false, false, false},
    {"days", 930'000'000, 10'000'000, true, false, false, false},
    {"pairing", 940'000'000, 10'000'000, false, false, true, false},
    {"confirm", 950'000'000, 10'000'000, true, false, true, true},
    {"viz", 960'000'000, ~0ull - 960'000'000, false, true, true, false},
    {"real", 0, 0, true, false, false, false},
}};

[[nodiscard]] inline const BlockInfo& info(Block b) { return kBlockTable[static_cast<std::size_t>(b)]; }

[[nodiscard]] inline Block block_of(const std::string& name) {
  for (int i = 0; i < kBlocks; ++i)
    if (name == kBlockTable[static_cast<std::size_t>(i)].name) return static_cast<Block>(i);
  throw std::invalid_argument("unknown seed block '" + name + "' (train, validation, test, stress, days, pairing, confirm, viz, real)");
}

// Episodes first .. first + count - 1 of block b exist (count > 0, no overflow).
inline void check_range(Block b, std::uint64_t first, std::uint64_t count) {
  const BlockInfo& bi = info(b);
  if (count == 0) throw std::invalid_argument(std::string("seed block ") + bi.name + ": count must be > 0");
  if (first >= bi.size || count > bi.size - first)
    throw std::invalid_argument(std::string("episodes [") + std::to_string(first) + ", " + std::to_string(first) + " + " + std::to_string(count) +
                                ") are outside seed block " + bi.name + " (size " + std::to_string(bi.size) + ")");
}

enum class Role : int { None, Trainer, Scorer };
[[nodiscard]] inline const char* role_name(Role r) { return r == Role::Trainer ? "trainer" : r == Role::Scorer ? "scorer" : "none"; }

class Ledger {
 public:
  // The one instance environments use by default (tests build their own).
  static Ledger& process() {
    static Ledger l;
    return l;
  }

  // Opens a sealed block for this process (the scoring step). Refused after
  // forbid_sealed() and in a trainer process; makes this process a scorer.
  void open_sealed(Block b, const std::string& reason) {
    std::lock_guard g(m_);
    const BlockInfo& bi = info(b);
    if (!bi.sealed) throw std::invalid_argument(std::string("seed block ") + bi.name + " is not sealed");
    if (reason.empty()) throw std::invalid_argument("open_sealed needs a reason (it is logged)");
    if (forbidden_) throw std::runtime_error(std::string("this process may not open sealed blocks (forbid_sealed was called); refused: ") + bi.name);
    if (role_ == Role::Trainer)
      throw std::runtime_error(std::string("this process trained or selected (built a train, validation or pairing environment); it may not open ") +
                               bi.name + ": score in a fresh process");
    role_ = Role::Scorer;
    open_ |= 1u << static_cast<int>(b);
    openings_.emplace_back(bi.name, reason);
  }
  // Sealed blocks stay closed for the rest of the process.
  void forbid_sealed() {
    std::lock_guard g(m_);
    forbidden_ = true;
    open_ = 0;
  }
  [[nodiscard]] bool is_open(Block b) const {
    std::lock_guard g(m_);
    return (open_ >> static_cast<int>(b)) & 1u;
  }
  // Throws unless b is usable here; sealed blocks must be open.
  void require_open(Block b) const {
    std::lock_guard g(m_);
    if (info(b).sealed && !((open_ >> static_cast<int>(b)) & 1u))
      throw std::runtime_error(std::string("seed block ") + info(b).name + " is sealed in this process (open_sealed in a scoring process only)");
  }
  // Called when episodes of block b are about to be played: checks and records the role.
  void use(Block b) {
    std::lock_guard g(m_);
    const BlockInfo& bi = info(b);
    if (bi.sealed) {
      if (!((open_ >> static_cast<int>(b)) & 1u))
        throw std::runtime_error(std::string("seed block ") + bi.name + " is sealed in this process (open_sealed in a scoring process only)");
      return;
    }
    if (b == Block::Train || b == Block::Validation || b == Block::Pairing) {
      if (role_ == Role::Scorer)
        throw std::runtime_error(std::string("this process opened a sealed block; it may not train or select (") + bi.name + " episodes refused)");
      role_ = Role::Trainer;
    }
  }
  [[nodiscard]] Role role() const {
    std::lock_guard g(m_);
    return role_;
  }
  [[nodiscard]] bool forbidden() const {
    std::lock_guard g(m_);
    return forbidden_;
  }
  // (block, reason) of every open_sealed() in this process, in order.
  [[nodiscard]] std::vector<std::pair<std::string, std::string>> openings() const {
    std::lock_guard g(m_);
    return openings_;
  }

 private:
  mutable std::mutex m_;
  bool forbidden_ = false;
  unsigned open_ = 0;
  Role role_ = Role::None;
  std::vector<std::pair<std::string, std::string>> openings_;
};

}  // namespace rag::orderflow::seeds
