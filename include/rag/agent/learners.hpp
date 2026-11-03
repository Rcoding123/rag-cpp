// rag/agent/learners.hpp — memory and learning.
//
//   TileQ          parametric value learner: Q(s, a) is a sum of one weight
//                  per active tile. Updated by temporal-difference (TD)
//                  learning, i.e. each experience nudges the weights toward
//                  "reward + value of where I ended up". Slow to learn,
//                  generalizes smoothly, constant memory.
//   ReplayBuffer   fixed-capacity ring of past transitions. Each is re-learned
//                  from many times, which squeezes far more out of every
//                  (expensive) interaction with the world.
//   EpisodicMemory instance-based learner: stores embeddings of situations it
//                  has been in, with the returns that followed each action, and
//                  estimates Q(s, a) by averaging its k most similar memories
//                  (model-free episodic control, Blundell et al. 2016). Learns
//                  from a single good episode, but generalizes only as far as
//                  its similarity measure reaches.
//
// Latency notes: weights are stored action-minor ([feature][action]), so
// reading all of a state's Q-values touches one cache line per tiling. The
// episodic memory runs ONE nearest-neighbour search per decision (each slot
// carries values for every action) with the SIMD kernels of rag/simd.hpp.
#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

#include "rag/agent/env.hpp"
#include "rag/agent/features.hpp"
#include "rag/agent/rng.hpp"
#include "rag/common.hpp"
#include "rag/simd.hpp"
#include "rag/topk.hpp"

namespace rag::agent {

// -------------------------------- TileQ --------------------------------------
class TileQ {
 public:
  TileQ() = default;
  TileQ(std::size_t features, std::uint32_t tilings, std::uint32_t actions, double alpha, double init_q = 0.0)
      : tilings_(tilings), actions_(actions), step_(alpha / tilings),
        w_(features * actions, init_q / tilings) {}

  // out[a] = Q(s, a) for the state whose active tiles are idx[0..tilings).
  void q_values(const std::uint32_t* idx, double* out) const noexcept {
    for (std::uint32_t a = 0; a < actions_; ++a) out[a] = 0.0;
    for (std::uint32_t k = 0; k < tilings_; ++k) {
      const double* w = w_.data() + std::size_t{idx[k]} * actions_;
      for (std::uint32_t a = 0; a < actions_; ++a) out[a] += w[a];
    }
  }

  [[nodiscard]] double q(const std::uint32_t* idx, Action a) const noexcept {
    double s = 0;
    for (std::uint32_t k = 0; k < tilings_; ++k) s += w_[std::size_t{idx[k]} * actions_ + a];
    return s;
  }

  // Move Q(s, a) toward `target` by the step size; returns the TD error.
  double update(const std::uint32_t* idx, Action a, double target) noexcept {
    const double err = target - q(idx, a);
    const double d = step_ * err;
    for (std::uint32_t k = 0; k < tilings_; ++k) w_[std::size_t{idx[k]} * actions_ + a] += d;
    return err;
  }

  // Change the step size (the agent anneals it as experience accumulates).
  void set_alpha(double alpha) noexcept { step_ = alpha / tilings_; }
  [[nodiscard]] double alpha() const noexcept { return step_ * tilings_; }

  [[nodiscard]] std::uint32_t actions() const noexcept { return actions_; }
  [[nodiscard]] const std::vector<double>& weights() const noexcept { return w_; }

 private:
  std::uint32_t tilings_ = 0, actions_ = 0;
  double step_ = 0;
  std::vector<double> w_;  // [feature][action]
};

// ------------------------------ ReplayBuffer ---------------------------------
struct Transition {
  std::array<std::uint32_t, kMaxTilings> s;     // active tiles of the state
  std::array<std::uint32_t, kMaxTilings> next;  // ... and of the next state
  float reward;
  std::uint16_t action;
  bool done;
};

class ReplayBuffer {
 public:
  explicit ReplayBuffer(std::size_t capacity = 0) : buf_(capacity) {}

  // Overwrites the oldest transition once full.
  void push(const Transition& t) noexcept {
    if (buf_.empty()) return;
    buf_[head_] = t;
    head_ = (head_ + 1) % buf_.size();
    size_ = std::min(size_ + 1, buf_.size());
  }

  [[nodiscard]] const Transition& sample(Rng& rng) const noexcept {
    return buf_[rng.below(static_cast<std::uint32_t>(size_))];
  }

  void clear() noexcept { head_ = size_ = 0; }

  [[nodiscard]] std::size_t size() const noexcept { return size_; }
  [[nodiscard]] std::size_t capacity() const noexcept { return buf_.size(); }
  [[nodiscard]] const Transition& newest() const noexcept { return buf_[(head_ + buf_.size() - 1) % buf_.size()]; }

 private:
  std::vector<Transition> buf_;
  std::size_t head_ = 0, size_ = 0;
};

// ----------------------------- EpisodicMemory --------------------------------
struct EpisodicParams {
  std::uint32_t capacity = 4096;  // slots; least recently used is evicted when full
  std::uint32_t neighbors = 8;    // k for the k-NN estimate
  float same_situation = 0.97f;   // cosine above which a new memory updates an old slot
  double learning_rate = 0.2;     // EMA step for updating a slot's value
};

template <std::size_t D, std::size_t A>
  requires ValidDim<D>
class EpisodicMemory {
 public:
  static constexpr std::uint32_t kMaxNeighbors = 32;

  explicit EpisodicMemory(const EpisodicParams& p = {})
      : p_(p), keys_(make_aligned_array<float>(std::size_t{std::max(p.capacity, 1u)} * D, Init::Lazy)),
        slots_(std::max(p.capacity, 1u)) {
    if (p.neighbors == 0 || p.neighbors > kMaxNeighbors) throw std::invalid_argument("EpisodicParams: neighbors in [1, 32]");
  }

  EpisodicMemory(const EpisodicMemory& o) : p_(o.p_), keys_(make_aligned_array<float>(std::size_t{std::max(o.p_.capacity, 1u)} * D, Init::Lazy)),
                                            slots_(o.slots_), size_(o.size_), tick_(o.tick_) {
    std::copy_n(o.keys_.get(), std::size_t{size_} * D, keys_.get());
  }
  EpisodicMemory& operator=(const EpisodicMemory& o) {
    if (this != &o) {
      EpisodicMemory tmp(o);
      std::swap(p_, tmp.p_);
      std::swap(keys_, tmp.keys_);
      std::swap(slots_, tmp.slots_);
      size_ = tmp.size_;
      tick_ = tmp.tick_;
    }
    return *this;
  }
  EpisodicMemory(EpisodicMemory&&) noexcept = default;
  EpisodicMemory& operator=(EpisodicMemory&&) noexcept = default;

  // Q-value estimates for every action: the similarity-weighted mean of the
  // values the k most similar memories hold for that action. known[a] is
  // false when none of them has ever tried a (the caller decides what an
  // unknown action is worth; optimism drives exploration).
  void estimate(const float* key, double* out, bool* known) const noexcept {
    TopK<kMaxNeighbors> nn(p_.neighbors);
    search(key, nn);
    std::array<double, A> sum{}, wsum{};
    for (const Hit& h : nn.sorted()) {
      const Slot& s = slots_[h.id];
      const double w = std::max(1e-3, static_cast<double>(h.score));  // similarity weight
      for (std::size_t a = 0; a < A; ++a)
        if (s.visits[a]) { sum[a] += w * s.value[a]; wsum[a] += w; }
    }
    for (std::size_t a = 0; a < A; ++a) {
      known[a] = wsum[a] > 0;
      out[a] = known[a] ? sum[a] / wsum[a] : 0.0;
    }
  }

  // Remember that taking `a` in situation `key` led to return `g`. Updates
  // the nearest slot if it is the same situation, else stores a new one.
  void write(const float* key, Action a, double g) {
    ++tick_;
    TopK<1> nn(1);
    search(key, nn);
    const auto best = nn.sorted();
    if (!best.empty() && best[0].score >= p_.same_situation) {
      Slot& s = slots_[best[0].id];
      s.value[a] = s.visits[a] ? s.value[a] + p_.learning_rate * (g - s.value[a]) : g;
      ++s.visits[a];
      s.last_used = tick_;
      return;
    }
    std::uint32_t slot;
    if (size_ < slots_.size()) {
      slot = size_++;
    } else {  // full: evict the least recently used slot
      slot = 0;
      for (std::uint32_t i = 1; i < size_; ++i)
        if (slots_[i].last_used < slots_[slot].last_used) slot = i;
    }
    std::copy_n(key, D, keys_.get() + std::size_t{slot} * D);
    slots_[slot] = Slot{};
    slots_[slot].value[a] = g;
    slots_[slot].visits[a] = 1;
    slots_[slot].last_used = tick_;
  }

  [[nodiscard]] std::uint32_t size() const noexcept { return size_; }
  [[nodiscard]] std::uint32_t capacity() const noexcept { return static_cast<std::uint32_t>(slots_.size()); }

 private:
  struct Slot {
    std::array<double, A> value{};
    std::array<std::uint32_t, A> visits{};
    std::uint64_t last_used = 0;
  };

  template <class TK>
  void search(const float* key, TK& out) const noexcept {
    std::uint32_t i = 0;
    float s[4];
    for (; i + 4 <= size_; i += 4) {
      simd::dot4(key, keys_.get() + std::size_t{i} * D, D, D, s);
      out.push(s[0], i); out.push(s[1], i + 1); out.push(s[2], i + 2); out.push(s[3], i + 3);
    }
    for (; i < size_; ++i) out.push(simd::dot1(key, keys_.get() + std::size_t{i} * D, D), i);
  }

  EpisodicParams p_;
  AlignedArray<float> keys_;  // [slot][D], unit vectors
  std::vector<Slot> slots_;
  std::uint32_t size_ = 0;
  std::uint64_t tick_ = 0;
};

}  // namespace rag::agent
