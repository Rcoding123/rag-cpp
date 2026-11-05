// rag/orderflow/tick_csv.hpp — read recorded ticks from a simple CSV and
// replay them, in time buckets, into anything with add_trade()/end_bucket()
// (FeatureBuilder, StylizedMeter, ...).
//
// Format: one event per line, time-ordered; '#' lines and blank lines are
// ignored. Prices are in index points (converted with tick_size) and times in
// `time_unit` (ns, us, ms or s, fractional allowed):
//
//   T,<time>,<price>,<size>,<B|S>                         trade; B/S = aggressor side
//   Q,<time>,<bid>,<ask>[,<bid size 1..5>,<ask size 1..5>] best bid/ask (and depth)
//
// e.g.  T,1730811600.125,5012.25,3,B
//       Q,1730811600.130,5012.00,5012.25,412,380,355,290,301,388,402,377,365,330
//
// Any vendor export (Rithmic, CQG, Sierra Chart, Databento, ...) can be
// turned into this with a few lines of scripting. Gaps longer than
// `max_gap_seconds` (overnight, halts) restart the bucket clock instead of
// emitting empty buckets.
#pragma once

#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "rag/orderflow/simulator.hpp"

namespace rag::orderflow {

struct TickCsvOptions {
  double tick_size = 0.25;        // ES: 0.25 index points
  std::string time_unit = "ns";   // ns | us | ms | s
  double bucket_seconds = 0.1;
  double max_gap_seconds = 60;
};

struct TickCsvStats {
  std::size_t lines = 0, trades = 0, quotes = 0, buckets = 0, skipped = 0, gaps = 0;
};

namespace detail {
inline std::vector<std::string> split_csv(const std::string& line) {
  std::vector<std::string> out;
  std::string cur;
  std::istringstream s(line);
  while (std::getline(s, cur, ',')) out.push_back(cur);
  return out;
}
inline double time_scale(const std::string& unit) {
  if (unit == "ns") return 1e-9;
  if (unit == "us") return 1e-6;
  if (unit == "ms") return 1e-3;
  if (unit == "s") return 1.0;
  throw std::invalid_argument("tick csv: time_unit must be ns, us, ms or s");
}
}  // namespace detail

// Replays the file into `sink`; returns counts. Lines that don't parse are
// counted in `skipped`, not fatal.
template <class Sink>
TickCsvStats replay_tick_csv(std::istream& in, Sink& sink, const TickCsvOptions& o = {}) {
  if (!(o.tick_size > 0) || !(o.bucket_seconds > 0)) throw std::invalid_argument("tick csv: tick_size and bucket_seconds must be positive");
  const double scale = detail::time_scale(o.time_unit);
  TickCsvStats st;
  Quote last{};
  bool have_quote = false, started = false;
  double bucket_end = 0;
  auto ticks = [&](const std::string& s) { return static_cast<std::int32_t>(std::llround(std::stod(s) / o.tick_size)); };
  std::string line;
  while (std::getline(in, line)) {
    ++st.lines;
    if (line.empty() || line[0] == '#' || line[0] == '\r') continue;
    const auto f = detail::split_csv(line);
    try {
      if (f.size() < 4 || (f[0] != "T" && f[0] != "Q")) throw std::invalid_argument("type");
      const double t = std::stod(f[1]) * scale;
      if (!started) {
        bucket_end = t + o.bucket_seconds;
        started = true;
      }
      if (t - bucket_end > o.max_gap_seconds) {  // a session gap: restart the clock
        ++st.gaps;
        bucket_end = t + o.bucket_seconds;
      }
      while (t >= bucket_end) {  // close every bucket that ended before this event
        if (have_quote) {
          sink.end_bucket(last);
          ++st.buckets;
        }
        bucket_end += o.bucket_seconds;
      }
      if (f[0] == "T") {
        if (f.size() < 5 || (f[4] != "B" && f[4] != "S")) throw std::invalid_argument("side");
        const auto size = static_cast<std::uint32_t>(std::stoul(f[3]));
        sink.add_trade(Trade{static_cast<std::int64_t>(t * 1e9), ticks(f[2]), size, f[4] == "B" ? Side::Buy : Side::Sell, false});
        ++st.trades;
      } else {
        Quote q{static_cast<std::int64_t>(t * 1e9), ticks(f[2]), ticks(f[3]), {}, {}};
        if (q.ask <= q.bid) throw std::invalid_argument("crossed");
        if (f.size() >= 4 + 2 * kDepth)
          for (std::size_t i = 0; i < kDepth; ++i) {
            q.bid_size[i] = static_cast<std::uint32_t>(std::stoul(f[4 + i]));
            q.ask_size[i] = static_cast<std::uint32_t>(std::stoul(f[4 + kDepth + i]));
          }
        last = q;
        have_quote = true;
        ++st.quotes;
      }
    } catch (const std::exception&) {
      ++st.skipped;
    }
  }
  if (have_quote) {
    sink.end_bucket(last);
    ++st.buckets;
  }
  return st;
}

template <class Sink>
TickCsvStats replay_tick_csv(const std::string& path, Sink& sink, const TickCsvOptions& o = {}) {
  std::ifstream in(path);
  if (!in) throw std::runtime_error("tick csv: cannot open " + path);
  return replay_tick_csv(in, sink, o);
}

// Writes `steps` steps of `sim` in this format (examples/of_calibrate.cpp
// export): a '#' title line, then per step its trades (at the step's end
// time, prices in index points at 0.25 per tick) and the quote with 5 levels
// 0.5 ms later; times in seconds since the simulator was built. `write(const
// char*, std::size_t)` receives the text.
template <class Write>
void write_sim_ticks(EsSimulator& sim, std::size_t steps, Write&& write) {
  char buf[512];  // a Q line is at most ~130 characters
  auto put = [&](int n) {
    if (n > 0) write(static_cast<const char*>(buf), static_cast<std::size_t>(n));
  };
  put(std::snprintf(buf, sizeof buf, "# ES simulator ticks: T,time_s,price,size,side / Q,time_s,bid,ask,bid sizes x5,ask sizes x5\n"));
  for (std::size_t s = 0; s < steps; ++s) {
    sim.step();
    const double t = static_cast<double>(sim.t_ns()) * 1e-9;
    for (const auto& tr : sim.trades())
      put(std::snprintf(buf, sizeof buf, "T,%.3f,%.2f,%u,%c\n", t, tr.price * 0.25, tr.size, tr.aggressor == Side::Buy ? 'B' : 'S'));
    const Quote q = sim.quote();
    int n = std::snprintf(buf, sizeof buf, "Q,%.3f,%.2f,%.2f", t + 0.0005, q.bid * 0.25, q.ask * 0.25);
    for (auto v : q.bid_size) n += std::snprintf(buf + n, sizeof buf - static_cast<std::size_t>(n), ",%u", v);
    for (auto v : q.ask_size) n += std::snprintf(buf + n, sizeof buf - static_cast<std::size_t>(n), ",%u", v);
    n += std::snprintf(buf + n, sizeof buf - static_cast<std::size_t>(n), "\n");
    put(n);
  }
}

}  // namespace rag::orderflow
