// rag/orderflow/scalp_run.hpp — pieces the ScalpEnv programs share
// (examples/of_train.cpp, examples/of_trace.cpp, experiments/of2_baselines.cpp,
// of6_scalp_stress.cpp, of8_scalp_days.cpp, of10_discovery.cpp): the
// --warmup option, the record of_train writes next to each network (so a
// stress, day or trace run can check it uses the network's market and
// warm-up), OF8's session starts and OF6's stress scenarios.
//
// --warmup S (OF10 iteration 3, opt-in; default 150 s = ScalpParams{}):
// seconds the simulator runs before an episode's first decision. On the v3
// discovery market a 150 s burn leaves the meta-order population ~5% short
// of the long-window level (of10_warmup_check.txt); 1050 s matches it, so RL
// reruns on discovery use --warmup 1050. A longer warm-up also moves the
// episodes: ScalpEnv::draw_market starts the warm-up uniformly in
// [0, session - episode - warm-up], so every episode starts >= S s into the
// session (and ends by the close). S may be at most session - episode
// (22,800 s): beyond that no episode fits (warmup_error).
#pragma once

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <optional>
#include <string>
#include <system_error>
#include <vector>

#include "rag/orderflow/scalp_env.hpp"

namespace rag::orderflow {

inline constexpr double kDefaultWarmup = 150;  // ScalpParams{}.warmup_seconds: every committed result

// A --warmup value: a finite number of seconds >= 0 (the whole string), else nullopt.
inline std::optional<double> parse_warmup(const std::string& v) {
  char* end = nullptr;
  errno = 0;
  const double x = std::strtod(v.c_str(), &end);
  if (v.empty() || *end != '\0' || errno == ERANGE || !std::isfinite(x) || x < 0) return std::nullopt;
  return x;
}

// Empty if p's warm-up leaves room for an episode before the close
// (warm-up <= session - episode), else the reason. Larger values would put
// every episode after the close (draw_market clamps the latest start to 0).
inline std::string warmup_error(const ScalpParams& p) {
  const double room = p.sim.session_seconds - p.episode_seconds;
  if (!(p.warmup_seconds >= 0) || p.warmup_seconds > room) {
    char b[160];
    std::snprintf(b, sizeof b, "warm-up %g s leaves no room for a %g s episode in a %g s session (at most %g s)", p.warmup_seconds,
                  p.episode_seconds, p.sim.session_seconds, room);
    return b;
  }
  return {};
}

// The ScalpParams of a run: the market preset, warmed up `warmup` seconds.
inline ScalpParams scalp_params(const std::string& market, double warmup = kDefaultWarmup) {
  ScalpParams p;
  p.sim = market_preset(market);
  p.warmup_seconds = warmup;
  return p;
}

// of_train's per-network record: DIR/seedK.txt next to DIR/seedK.net
// (key=value lines: market, warmup_seconds, then the training options and the
// seed). One record per network, written right after that network is saved
// (the old record removed first), so a record never vouches for a network
// another run left in the same directory: a seed with no champion in a rerun
// keeps its old network AND its old record.
inline std::string record_path_of_net(const std::string& net_path) {
  const std::size_t slash = net_path.find_last_of("/\\");
  const std::size_t dot = net_path.find_last_of('.');
  const bool ext = dot != std::string::npos && (slash == std::string::npos || dot > slash);
  return (ext ? net_path.substr(0, dot) : net_path) + ".txt";
}

// Removes the record of `net_path` (before the network is overwritten).
inline void remove_net_record(const std::string& net_path) {
  std::error_code ec;
  std::filesystem::remove(record_path_of_net(net_path), ec);
}

// Writes the record of `net_path`: market, warmup_seconds, then `extra` (more
// key=value lines, each ending in '\n'). False if the file can't be written.
inline bool write_net_record(const std::string& net_path, const std::string& market, double warmup, const std::string& extra = {}) {
  std::FILE* f = std::fopen(record_path_of_net(net_path).c_str(), "w");
  if (!f) return false;
  std::fprintf(f, "market=%s\nwarmup_seconds=%.17g\n%s", market.c_str(), warmup, extra.c_str());
  return std::fclose(f) == 0;
}

// The value of `key` in the record of `net_path`; nullopt if there is no
// record (networks from of3_rl or an older of_train) or no such key.
inline std::optional<std::string> net_record_value(const std::string& net_path, const std::string& key) {
  std::ifstream in(record_path_of_net(net_path));
  if (!in) return std::nullopt;
  for (std::string line; std::getline(in, line);) {
    if (!line.empty() && line.back() == '\r') line.pop_back();
    if (line.size() > key.size() && line.compare(0, key.size(), key) == 0 && line[key.size()] == '=') return line.substr(key.size() + 1);
  }
  return std::nullopt;
}

// Empty if `net_path` may run on `market` with `warmup` seconds of warm-up,
// else the reason. A network with no record is a legacy one (of3_rl, an older
// of_train): trained with ScalpEnv's default 150 s warm-up, so it is refused
// at any other warm-up; its market is unknown and not checked. A recorded
// network is refused on another warm-up or another market.
inline std::string check_net(const std::string& net_path, const std::string& market, double warmup) {
  char b[200];
  if (!std::filesystem::exists(record_path_of_net(net_path))) {
    if (warmup == kDefaultWarmup) return {};
    std::snprintf(b, sizeof b, ": no of_train record (%s): a legacy network, trained with the default %g s warm-up; this run uses %g s (--warmup)",
                  record_path_of_net(net_path).c_str(), kDefaultWarmup, warmup);
    return net_path + b;
  }
  const auto v = net_record_value(net_path, "warmup_seconds");
  const auto w = v ? parse_warmup(*v) : std::nullopt;
  if (!w) return net_path + ": missing or unreadable warmup_seconds in " + record_path_of_net(net_path);
  if (*w != warmup) {
    std::snprintf(b, sizeof b, ": trained with a %g s warm-up (%s), this run uses %g s (--warmup)", *w, record_path_of_net(net_path).c_str(), warmup);
    return net_path + b;
  }
  if (const auto m = net_record_value(net_path, "market"); m && *m != market)
    return net_path + ": trained on market " + *m + " (" + record_path_of_net(net_path) + "), this run uses market " + market + " (--market)";
  return {};
}

// OF8's k-th 10-minute session of a day: the session second the warm-up
// starts. The episode stays where a 150 s warm-up puts it (k * 600 + 150 s)
// and the warm-up starts `warmup` s before it, never before the open (then
// the episode starts `warmup` s into the session: with 1050 s, sessions 0 and
// 1 both start at 1050 s). With 150 s this is k * 600 exactly.
inline double day_session_start(unsigned k, double warmup = kDefaultWarmup) {
  return std::max(0.0, k * 600.0 + (kDefaultWarmup - warmup));
}

// OF6 stress scenarios: a change of the episode's ScalpParams.
struct StressScenario {
  std::string name;
  std::function<void(ScalpParams&)> change;
};

// The legacy list (default and efficient markets; OF6's committed results)
// and, for the v3 discovery market, a list FIXED HERE BEFORE ANY v3 RL
// NETWORK EXISTED (OF10 iteration 3, pre-registered). On discovery v3 three
// legacy stresses are empty or weak: quote fading x2 is a no-op (fade = 0;
// measured: output byte-identical to nominal), volatility x1.5 scales only
// value_vol (~22% of the 10 s value change; ~77% is flow impact; measured:
// R^2 0.31 +- 0.015 vs 0.34, 1-min sd 3.90 vs 3.96, sd(g) 0.332 vs 0.334),
// informed flow x0.5 acts on ~4% of the volume (measured: informed volume
// 2.1% vs 3.8%, gap statistics unchanged). The discovery list keeps nominal,
// the latencies, fees, uninformed flow x1.5 and the (cold) session open, and
// replaces those three with five stresses that bite there.
// Measured on diagnostic seeds 22-24, before any v3 network existed:
//   of_discovery --markets discovery --seeds 22-24 --episodes 400 --long off
//                --scalp-warmup 1050 --set <the change>
// episode regime (ScalpEnv episodes, 1050 s warm-up), 1,200 windows, pooled
// value (+- jackknife SE where the probe gives one), nominal -> stressed.
// Nominal: sd(g) 0.334, corr(g, r 30 s) - ideal floor 0.018, race to the tick
// median 0.30 s, event corr(I1, next mid change) 0.209 +- 0.003, leak
// corr(I1, g) 0.362 +- 0.004, markout 1 s 0.252 +- 0.008, R^2(10 s mid
// change ~ delta) 0.340 +- 0.008, flow-impact share of the 10 s value change
// 0.78, 1-min sd 3.96 +- 0.07, kurtosis 5.6 +- 0.6, trades/s 18.5, mean L1
// 106 lots, informed volume share 3.8%, active meta-orders 11.2.
// Informed meta-orders x2 (also measured) was too weak to keep: informed
// volume 6.4%, active meta-orders 16.3, but the market barely moved (sd(g)
// 0.330, race median 0.295 s, event corr(I1, next change) 0.209, markout
// 1 s 0.260 +- 0.009, kurtosis 5.2): x4 is the smallest factor tried that
// moves the facts.
// "Session open" (both lists) is a cold open: the simulator starts at the
// open with ScalpEnv's default 150 s burn, so the episode runs 150 .. 750 s
// whatever the run's --warmup. With the default warm-up this is exactly the
// legacy scenario (start_second = 0; OF6's committed results unchanged). It
// does NOT keep the run's warm-up: with 1050 s, start_second = 0 alone would
// put the episode at 1050 .. 1650 s, the earliest window training already
// draws (in distribution, hardly a stress). Activity (intraday factor
// 1 + 1.5 exp(-t / 1800 s) + exp(-(session - t) / 1800 s), episode mean):
// 2.17 at 150 .. 750 s vs 1.71 at 1050 .. 1650 s and ~1.13 over the 1050 s
// training windows; the 150 s burn also leaves the discovery meta-order
// population ~5% short (of10_warmup_check.txt), as a real open would. Not
// measured with the probe (of_discovery cannot place a window at the open):
// computed, not tuned.
inline void session_open(ScalpParams& p) {
  p.start_second = 0;
  p.warmup_seconds = kDefaultWarmup;
}

inline std::vector<StressScenario> stress_scenarios(bool discovery) {
  if (!discovery)
    return {
        {"nominal", [](ScalpParams&) {}},
        {"latency 300 ms", [](ScalpParams& p) { p.latency_steps = 3; }},
        {"latency 1 s", [](ScalpParams& p) { p.latency_steps = 10; }},
        {"fees $3/side", [](ScalpParams& p) { p.fee_dollars = 3.0; }},
        {"informed flow x0.5", [](ScalpParams& p) { p.sim.informed_meta_rate *= 0.5; }},
        {"uninformed flow x1.5", [](ScalpParams& p) { p.sim.uninformed_meta_rate *= 1.5; }},
        {"volatility x1.5", [](ScalpParams& p) { p.sim.value_vol *= 1.5; }},
        {"quote fading x2", [](ScalpParams& p) { p.sim.fade *= 2; }},
        {"session open", session_open},
    };
  return {
      {"nominal", [](ScalpParams&) {}},
      {"latency 300 ms", [](ScalpParams& p) { p.latency_steps = 3; }},
      {"latency 1 s", [](ScalpParams& p) { p.latency_steps = 10; }},
      {"fees $3/side", [](ScalpParams& p) { p.fee_dollars = 3.0; }},
      {"uninformed flow x1.5", [](ScalpParams& p) { p.sim.uninformed_meta_rate *= 1.5; }},
      // Stale quotes leave at half the rate, half as much of them sniped:
      // race to the tick median 0.30 -> 0.40 s, time in a race 6.4% -> 9.0%,
      // sd(g) 0.334 -> 0.353, corr(g, r 30 s) - floor 0.018 -> 0.026, sniper
      // volume 7.7% -> 3.2%, event corr(I1, next change) 0.209 -> 0.236,
      // markout 1 s 0.252 -> 0.223, trades/s 18.5 -> 16.6.
      {"slower discovery x0.5", [](ScalpParams& p) {
         p.sim.reprice_cancel *= 0.5;
         p.sim.reprice_take *= 0.5;
       }},
      // Taker flow moves value 1.5x as much: flow-impact share 0.78 -> 0.89,
      // 1-min sd 3.96 -> 5.70, R^2 0.340 -> 0.396, markout 1 s 0.252 -> 0.326,
      // sd(g) 0.334 -> 0.367, mid changes/min 18.4 -> 25.8, leak corr(I1, g)
      // 0.362 -> 0.410.
      {"flow impact x1.5", [](ScalpParams& p) { p.sim.value_impact *= 1.5; }},
      // Three times the news jumps: flow-impact share 0.78 -> 0.54, R^2
      // 0.340 -> 0.258, 1-min sd 3.96 -> 4.66, kurtosis 5.6 -> 8.8, event
      // corr(I1, next change) 0.209 -> 0.171, corr(I1, r 30 s) 0.076 -> 0.051,
      // markout 1 s 0.252 -> 0.308.
      {"news x3", [](ScalpParams& p) { p.sim.jump_rate *= 3; }},
      // Four times the informed meta-order starts: informed volume 3.8% ->
      // 10.3%, active meta-orders 11.2 -> 26.1, markout 1 s 0.252 -> 0.315,
      // kurtosis 5.6 -> 9.1, race median 0.30 -> 0.27 s, leak corr(I1, g)
      // 0.362 -> 0.327, trades/s 18.5 -> 20.0.
      {"informed meta-orders x4", [](ScalpParams& p) { p.sim.informed_meta_rate *= 4; }},
      // Half the maker arrivals (a thinner book; attacks the depletion
      // signal in L1 imbalance): mean L1 106 -> 35 lots, leak corr(I1, g)
      // 0.362 -> 0.214, event corr(I1, next change) 0.209 -> 0.161, markout
      // 1 s 0.252 -> 0.159, R^2 0.340 -> 0.285, sniper volume 7.7% -> 2.5%.
      {"thinner book x0.5", [](ScalpParams& p) { p.sim.maker_rate *= 0.5; }},
      {"session open", session_open},
  };
}

}  // namespace rag::orderflow
