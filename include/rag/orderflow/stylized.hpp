// rag/orderflow/stylized.hpp — the stylized facts of a tick stream, measured
// the same way for simulated and recorded ES ticks (for calibration).
//
//   StylizedMeter m(0.1);                       // bucket width in seconds
//   for each bucket: m.add_trade(t)...; m.end_bucket(quote);
//   const StylizedFacts f = m.facts();
//
// Facts (prices in ticks):
//   trades_per_sec, mean_trade_size, volume_per_min      activity
//   spread1_share, mean_l1_size                          book
//   ret1m_sd, ret1m_kurtosis (excess), ret1m_ac1,        1-minute mid returns
//   absret1m_ac1                                         volatility clustering
//   sign_ac1, sign_ac10, sign_ac100                      trade-sign autocorrelation (long memory)
//   impact_r2                                            R^2 of 10 s mid change on 10 s delta
//   delta_predict                                        corr(30 s delta, next 30 s mid change)
//   markout_1s, markout_5s, markout_30s                  mean mid change 1/5/30 s after an
//                                                        aggressive print, signed by its side
//                                                        (ticks; the adverse selection a passive
//                                                        fill suffers, as a curve)
//   mid_changes_per_min                                  how often the mid moves (queue depletion)
#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <deque>
#include <vector>

#include "rag/orderflow/simulator.hpp"

namespace rag::orderflow {

struct StylizedFacts {
  double seconds = 0, trades_per_sec = 0, mean_trade_size = 0, volume_per_min = 0;
  double spread1_share = 0, mean_l1_size = 0;
  double ret1m_sd = 0, ret1m_kurtosis = 0, ret1m_ac1 = 0, absret1m_ac1 = 0;
  double sign_ac1 = 0, sign_ac10 = 0, sign_ac100 = 0;
  double impact_r2 = 0, delta_predict = 0, markout_5s = 0;
  double markout_1s = 0, markout_30s = 0, mid_changes_per_min = 0;
};

namespace detail {
inline double mean_of(const std::vector<double>& v) {
  double s = 0;
  for (double x : v) s += x;
  return v.empty() ? 0 : s / static_cast<double>(v.size());
}
inline double corr(const std::vector<double>& a, const std::vector<double>& b) {
  const std::size_t n = std::min(a.size(), b.size());
  if (n < 3) return 0;
  double ma = 0, mb = 0;
  for (std::size_t i = 0; i < n; ++i) {
    ma += a[i];
    mb += b[i];
  }
  ma /= static_cast<double>(n);
  mb /= static_cast<double>(n);
  double sab = 0, saa = 0, sbb = 0;
  for (std::size_t i = 0; i < n; ++i) {
    sab += (a[i] - ma) * (b[i] - mb);
    saa += (a[i] - ma) * (a[i] - ma);
    sbb += (b[i] - mb) * (b[i] - mb);
  }
  return saa > 0 && sbb > 0 ? sab / std::sqrt(saa * sbb) : 0;
}
inline double autocorr(const std::vector<double>& v, std::size_t lag) {
  if (v.size() <= lag + 2) return 0;
  return corr(std::vector<double>(v.begin(), v.end() - static_cast<std::ptrdiff_t>(lag)),
              std::vector<double>(v.begin() + static_cast<std::ptrdiff_t>(lag), v.end()));
}
}  // namespace detail

class StylizedMeter {
 public:
  explicit StylizedMeter(double bucket_seconds = 0.1) : dt_(bucket_seconds) {}

  void add_trade(const Trade& t) {
    const double s = t.aggressor == Side::Buy ? 1.0 : -1.0;
    trades_ += 1;
    volume_ += t.size;
    delta_ += s * t.size;
    if (signs_.size() < kMaxSigns) signs_.push_back(s);
    pending_signs_ += s;  // markout: signed prints in this bucket
  }

  void end_bucket(const Quote& q) {
    const double mid = 0.5 * (q.bid + q.ask);
    mids_.push_back(mid);
    deltas_.push_back(delta_);
    marks_.push_back(pending_signs_);
    delta_ = 0;
    pending_signs_ = 0;
    spread1_ += (q.ask - q.bid) == 1;
    l1_ += 0.5 * (q.bid_size[0] + q.ask_size[0]);
  }

  [[nodiscard]] StylizedFacts facts() const {
    using namespace detail;
    StylizedFacts f;
    const std::size_t n = mids_.size();
    f.seconds = static_cast<double>(n) * dt_;
    if (n == 0 || f.seconds <= 0) return f;
    f.trades_per_sec = trades_ / f.seconds;
    f.mean_trade_size = trades_ > 0 ? volume_ / trades_ : 0;
    f.volume_per_min = volume_ / f.seconds * 60;
    f.spread1_share = spread1_ / static_cast<double>(n);
    f.mean_l1_size = l1_ / static_cast<double>(n);
    const auto per = [&](double sec) { return std::max<std::size_t>(1, static_cast<std::size_t>(std::lround(sec / dt_))); };
    const std::size_t b10 = per(10), b30 = per(30), b60 = per(60), b5 = per(5);
    std::vector<double> r1, abs1, r10, d10, d30, f30;
    for (std::size_t i = b60; i < n; i += b60) {
      r1.push_back(mids_[i] - mids_[i - b60]);
      abs1.push_back(std::abs(r1.back()));
    }
    auto window = [&](std::size_t end, std::size_t len) {
      double s = 0;
      for (std::size_t i = end - len; i < end; ++i) s += deltas_[i];
      return s;
    };
    for (std::size_t i = b10; i < n; i += b10) {
      r10.push_back(mids_[i] - mids_[i - b10]);
      d10.push_back(window(i, b10));
    }
    for (std::size_t i = b30; i + b30 < n; i += b30) {
      d30.push_back(window(i, b30));
      f30.push_back(mids_[i + b30] - mids_[i]);
    }
    if (r1.size() > 2) {
      const double m = mean_of(r1);
      double m2 = 0, m4 = 0;
      for (double x : r1) {
        m2 += (x - m) * (x - m);
        m4 += std::pow(x - m, 4);
      }
      m2 /= static_cast<double>(r1.size());
      m4 /= static_cast<double>(r1.size());
      f.ret1m_sd = std::sqrt(m2);
      f.ret1m_kurtosis = m2 > 0 ? m4 / (m2 * m2) - 3 : 0;
      f.ret1m_ac1 = autocorr(r1, 1);
      f.absret1m_ac1 = autocorr(abs1, 1);
    }
    f.sign_ac1 = autocorr(signs_, 1);
    f.sign_ac10 = autocorr(signs_, 10);
    f.sign_ac100 = autocorr(signs_, 100);
    const double c = corr(d10, r10);
    f.impact_r2 = c * c;
    f.delta_predict = corr(d30, f30);
    auto markout = [&](std::size_t h) {
      double mk = 0, cnt = 0;
      for (std::size_t i = 0; i + h < n; ++i)
        if (marks_[i] != 0) {
          mk += marks_[i] * (mids_[i + h] - mids_[i]);
          cnt += std::abs(marks_[i]);
        }
      return cnt > 0 ? mk / cnt : 0.0;
    };
    f.markout_1s = markout(per(1));
    f.markout_5s = markout(b5);
    f.markout_30s = markout(b30);
    double changes = 0;
    for (std::size_t i = 1; i < n; ++i) changes += mids_[i] != mids_[i - 1];
    f.mid_changes_per_min = changes / f.seconds * 60;
    return f;
  }

 private:
  static constexpr std::size_t kMaxSigns = 2'000'000;
  double dt_;
  double trades_ = 0, volume_ = 0, delta_ = 0, pending_signs_ = 0, spread1_ = 0, l1_ = 0;
  std::vector<double> mids_, deltas_, marks_, signs_;
};

// Stylized facts of the simulator with these parameters over `seconds` of trading.
inline StylizedFacts simulate_facts(const SimParams& p, double seconds, std::uint64_t seed = 1, double start_second = 3600) {
  EsSimulator sim(p, seed, start_second);
  StylizedMeter m(p.dt);
  const auto steps = static_cast<std::size_t>(seconds / p.dt);
  for (std::size_t s = 0; s < steps; ++s) {
    sim.step();
    for (const auto& t : sim.trades()) m.add_trade(t);
    m.end_bucket(sim.quote());
  }
  return m.facts();
}

}  // namespace rag::orderflow
