// rag/agent/policy_table.hpp — compile an execution policy into a lookup
// table (E22).
//
// The execution state is small: time (horizon values), inventory (lots + 1
// values) and one continuous signal. Sampling any policy (a trained neural
// network, a planner, an expensive ensemble) at the centre of every
// (t, inventory, signal bin) cell gives a table of one-byte actions whose
// lookup costs a few nanoseconds, whatever the original policy cost.
//
//   ExecutionPolicyTable table(params, [&](const ExecutionObs& o) { return trainer.deployed_act(o); });
//   Action a = table.act(obs);            // no allocation, no floating-point model
//
// Signals outside +-signal_range_sd standard deviations use the edge bins; a
// NaN signal uses bin 0. Times and inventories past the grid are clamped.
//
//   table.save(path); auto t = ExecutionPolicyTable::load(path);   // rag::io format, checksummed
//   auto ens = ExecutionPolicyTable::median({t1, t2, t3});         // ensemble at table cost
//
// median() combines tables of the same shape cell by cell with the (lower)
// median action. Actions are ordered by trade size (ExecutionParams::sizes is
// increasing by default), so the median is the natural vote for an ordinal
// choice.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <limits>
#include <stdexcept>
#include <vector>

#include "rag/agent/env.hpp"
#include "rag/io.hpp"

namespace rag::agent {

struct PolicyTableParams {
  std::uint32_t signal_bins = 64;  // bins over [-range, +range] signal standard deviations
  double signal_range_sd = 4.0;    // the signal's stationary sd is ExecutionParams::signal_sd
};

class ExecutionPolicyTable {
 public:
  template <class Policy>
  ExecutionPolicyTable(const ExecutionParams& p, Policy&& policy, PolicyTableParams tp = {})
      : horizon_(p.horizon), lots_(p.lots), bins_(tp.signal_bins) {
    if (p.horizon == 0 || tp.signal_bins == 0 || !(tp.signal_range_sd > 0) || !(p.signal_sd > 0) ||
        !std::isfinite(tp.signal_range_sd * p.signal_sd))
      throw std::invalid_argument("ExecutionPolicyTable: horizon, bins, range and signal_sd must be positive");
    lo_ = -tp.signal_range_sd * p.signal_sd;
    width_ = 2 * tp.signal_range_sd * p.signal_sd / static_cast<double>(bins_);
    if (!valid_grid(lo_, width_))
      throw std::invalid_argument("ExecutionPolicyTable: signal_range_sd * signal_sd gives a zero, infinite or subnormal bin width");
    inv_width_ = 1.0 / width_;
    const std::size_t cells = cell_count(horizon_, lots_, bins_);
    if (cells == 0 || cells > table_.max_size())
      throw std::invalid_argument("ExecutionPolicyTable: horizon x (lots + 1) x signal_bins is too large");
    table_.resize(cells);
    std::size_t k = 0;
    for (std::uint32_t t = 0; t < horizon_; ++t)
      for (std::uint64_t inv = 0; inv <= lots_; ++inv)  // 64-bit: lots_ may be UINT32_MAX
        for (std::uint32_t b = 0; b < bins_; ++b) {
          const Action a = policy(ExecutionObs{t, static_cast<std::uint32_t>(inv), bin_centre(b)});
          if (a >= ExecutionEnv::kNumActions) throw std::out_of_range("ExecutionPolicyTable: policy returned an invalid action");
          table_[k++] = static_cast<std::uint8_t>(a);
        }
  }

  [[nodiscard]] Action act(const ExecutionObs& o) const noexcept {
    const double x = (o.signal - lo_) * inv_width_;
    const std::uint32_t b = !(x > 0) ? 0u : x >= static_cast<double>(bins_) ? bins_ - 1 : static_cast<std::uint32_t>(x);
    const std::uint32_t t = std::min(o.t, horizon_ - 1), inv = std::min(o.inventory, lots_);
    return table_[(static_cast<std::size_t>(t) * (static_cast<std::size_t>(lots_) + 1) + inv) * bins_ + b];
  }

  // Cell-by-cell lower median of tables with the same grid.
  [[nodiscard]] static ExecutionPolicyTable median(const std::vector<ExecutionPolicyTable>& tables) {
    if (tables.empty()) throw std::invalid_argument("ExecutionPolicyTable::median: no tables");
    const auto& f = tables.front();
    for (const auto& t : tables)
      if (t.horizon_ != f.horizon_ || t.lots_ != f.lots_ || t.bins_ != f.bins_ || t.lo_ != f.lo_ || t.width_ != f.width_)
        throw std::invalid_argument("ExecutionPolicyTable::median: tables have different grids");
    ExecutionPolicyTable out = f;
    std::vector<std::uint8_t> votes(tables.size());
    for (std::size_t k = 0; k < out.table_.size(); ++k) {
      for (std::size_t i = 0; i < tables.size(); ++i) votes[i] = tables[i].table_[k];
      std::nth_element(votes.begin(), votes.begin() + static_cast<std::ptrdiff_t>((votes.size() - 1) / 2), votes.end());
      out.table_[k] = votes[(votes.size() - 1) / 2];
    }
    return out;
  }

  void save(const std::filesystem::path& path) const {
    io::Writer w(path);
    w.bytes(kMagic, sizeof kMagic);
    w.pod(horizon_);
    w.pod(lots_);
    w.pod(bins_);
    w.pod(lo_);
    w.pod(width_);
    w.vec(table_);
    w.commit();
  }

  // Rejects anything save() could not have written (io::FormatError).
  [[nodiscard]] static ExecutionPolicyTable load(const std::filesystem::path& path) {
    io::Reader r(path);
    char magic[sizeof kMagic];
    r.bytes(magic, sizeof magic);
    if (std::memcmp(magic, kMagic, sizeof magic) != 0) throw io::FormatError("ExecutionPolicyTable: not a policy table");
    ExecutionPolicyTable t;
    t.horizon_ = r.pod<std::uint32_t>();
    t.lots_ = r.pod<std::uint32_t>();
    t.bins_ = r.pod<std::uint32_t>();
    t.lo_ = r.pod<double>();
    t.width_ = r.pod<double>();
    t.table_ = r.vec<std::uint8_t>();
    r.finish();
    const double cells = static_cast<double>(t.horizon_) * (static_cast<double>(t.lots_) + 1) * static_cast<double>(t.bins_);
    if (t.horizon_ == 0 || t.bins_ == 0 || !valid_grid(t.lo_, t.width_) || cells != static_cast<double>(t.table_.size()))
      throw io::FormatError("ExecutionPolicyTable: inconsistent grid");
    for (auto a : t.table_)
      if (a >= ExecutionEnv::kNumActions) throw io::FormatError("ExecutionPolicyTable: invalid action");
    t.inv_width_ = 1.0 / t.width_;
    return t;
  }

  [[nodiscard]] double bin_centre(std::uint32_t b) const noexcept { return lo_ + (static_cast<double>(b) + 0.5) * width_; }
  [[nodiscard]] std::size_t bytes() const noexcept { return table_.size(); }
  [[nodiscard]] std::uint32_t signal_bins() const noexcept { return bins_; }

 private:
  static constexpr char kMagic[8] = {'R', 'A', 'G', 'P', 'T', 'B', 'L', '1'};
  ExecutionPolicyTable() = default;

  // A finite origin and a normal (finite, non-zero, non-subnormal) bin width:
  // then 1 / width is finite too and act()'s multiply bins what bin_centre()
  // describes.
  [[nodiscard]] static bool valid_grid(double lo, double width) noexcept {
    return std::isfinite(lo) && width > 0 && std::isnormal(width);
  }
  // horizon x (lots + 1) x bins, or 0 if that does not fit in a size_t.
  [[nodiscard]] static std::size_t cell_count(std::uint32_t horizon, std::uint32_t lots, std::uint32_t bins) noexcept {
    const std::uint64_t rows = std::uint64_t{horizon} * (std::uint64_t{lots} + 1);  // < 2^64
    if (bins == 0 || rows > std::numeric_limits<std::size_t>::max() / bins) return 0;
    return static_cast<std::size_t>(rows * bins);
  }

  std::uint32_t horizon_ = 1, lots_ = 0, bins_ = 1;
  double lo_ = 0, width_ = 1, inv_width_ = 1;
  std::vector<std::uint8_t> table_;
};

}  // namespace rag::agent
