// OF10: price-discovery diagnostics. Measures, for each market preset and
// seed, every acceptance criterion of the quote-jump fix (ORDERFLOW.md plan
// item 1) in one run:
//   (a) the hidden gap g = fair value - mid: does it predict the mid, and
//       how fast (horizons 0.1 s ... 60 s, the 2 s -> 30 s tail, slopes),
//       against the discreteness floor of an ideal market whose mid is
//       floor(value) + 0.5 (a 1-tick grid can't do better);
//   (b) book imbalance (L1, L1-5) and the 13 features: forward-return
//       correlations in clock time and event time (mid changes), tails,
//       and their correlation with the hidden gap (leaks);
//   (c) the markout curve after aggressive prints (count-, print- and
//       volume-weighted; sniper vs other prints);
//   (d) the stylized facts (stylized.hpp) and market health (activity,
//       quote jumps, who trades how much);
//   (e, f) with --rules N: the OF2 rules and fair-value oracles at several
//       thresholds on ScalpEnv, N episodes per seed, episode seeds
//       1,000,000 * seed + i (the simulator-design rule-check sub-block:
//       not a sealed block).
// Each (market, seed) runs EsSimulator(preset, seed, start 3600 s) for
// --seconds after a 150 s warm-up, one thread each.
//
// Usage: of_discovery [--markets default,efficient,discovery] [--seeds 1-3]
//          [--seconds 7200] [--threads 8] [--rules 0] [--csv FILE] [--out FILE]
//          [--set name=value,...]
// --set overrides fields of the discovery preset (tuning), e.g.
// --set reprice_band=0.6,reprice_snipe=0.05 (names: see kTunable).
// --out also writes the summary to FILE.
#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <map>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "rag/orderflow/features.hpp"
#include "rag/orderflow/scalp_env.hpp"
#include "rag/orderflow/stylized.hpp"

using namespace rag::orderflow;
using rag::agent::Action;

namespace {

struct Options {
  std::vector<std::string> markets{"default", "efficient", "discovery"};
  std::vector<std::uint64_t> seeds{1, 2, 3};
  double seconds = 7200, warmup = 150;
  unsigned threads = 8, rules = 0;
  std::string csv, out;
  std::vector<std::pair<std::string, double>> set;  // overrides of the discovery preset
};

// SimParams fields --set may change.
const std::map<std::string, double SimParams::*> kTunable = {
    {"reprice_band", &SimParams::reprice_band},       {"reprice_snipe", &SimParams::reprice_snipe},
    {"informed_meta_rate", &SimParams::informed_meta_rate}, {"uninformed_meta_rate", &SimParams::uninformed_meta_rate},
    {"informed_scale", &SimParams::informed_scale},   {"momentum_rate", &SimParams::momentum_rate},
    {"noise_rate", &SimParams::noise_rate},           {"meta_child_rate", &SimParams::meta_child_rate},
    {"activity_excite", &SimParams::activity_excite}, {"value_vol", &SimParams::value_vol},
    {"fade", &SimParams::fade},                       {"maker_rate", &SimParams::maker_rate},
    {"cancel_rate", &SimParams::cancel_rate},         {"taker_size_mean", &SimParams::taker_size_mean}};

SimParams market(const Options& o, const std::string& name) {
  SimParams p = market_preset(name);
  if (name == "discovery")
    for (const auto& [k, v] : o.set) {
      const auto it = kTunable.find(k);
      if (it == kTunable.end()) throw std::invalid_argument("--set: unknown field " + k);
      p.*(it->second) = v;
    }
  return p;
}

struct Row {
  std::string metric;
  double horizon;
  double value;
};
using Rows = std::vector<Row>;

// Running moments of (x, y): correlation and regression slope of y on x.
struct Mom {
  double n = 0, sx = 0, sy = 0, sxx = 0, syy = 0, sxy = 0;
  void add(double x, double y) {
    n += 1;
    sx += x;
    sy += y;
    sxx += x * x;
    syy += y * y;
    sxy += x * y;
  }
  [[nodiscard]] double cov() const { return n > 1 ? (sxy - sx * sy / n) / n : 0; }
  [[nodiscard]] double var_x() const { return n > 1 ? (sxx - sx * sx / n) / n : 0; }
  [[nodiscard]] double var_y() const { return n > 1 ? (syy - sy * sy / n) / n : 0; }
  [[nodiscard]] double corr() const {
    const double d = var_x() * var_y();
    return d > 0 ? cov() / std::sqrt(d) : 0;
  }
  [[nodiscard]] double slope() const { return var_x() > 0 ? cov() / var_x() : 0; }
};

double quantile(std::vector<double> v, double q) {
  if (v.empty()) return 0;
  std::sort(v.begin(), v.end());
  const auto i = static_cast<std::size_t>(std::floor(q * static_cast<double>(v.size() - 1)));
  return v[i];
}

const std::array<const char*, kFeatures> kFeatureNames = {"delta5", "delta30", "delta120", "ofi5",  "ofi30", "vol30", "imbL1",
                                                          "imbL5",  "ret5",    "ret30",    "ret120", "vwap",  "absorb"};
const std::array<double, 9> kHorizons = {0.1, 0.2, 0.5, 1, 2, 5, 10, 30, 60};  // seconds
const std::array<double, 7> kMarkouts = {0.1, 1, 2, 5, 10, 30, 60};
const std::array<std::size_t, 4> kEvents = {1, 2, 5, 10};  // j-th mid change

Rows diagnose(const SimParams& p, std::uint64_t seed, const Options& o) {
  const auto per = [&](double sec) { return static_cast<std::size_t>(std::max(1L, std::lround(sec / p.dt))); };
  const std::size_t warm = per(o.warmup), total = per(o.seconds);
  EsSimulator sim(p, seed, 3600);
  FeatureBuilder fb(p.dt);
  StylizedMeter meter(p.dt);
  std::vector<double> mid, val, gap, act;
  std::vector<double> net, prints, vol, sv, snet, sprints, svol, ssv;  // per bucket: signed count, prints, volume, signed volume (s: snipers)
  std::vector<std::vector<double>> feat(kFeatures), raw(kFeatures);
  for (auto* v : {&mid, &val, &gap, &act, &net, &prints, &vol, &sv, &snet, &sprints, &svol, &ssv}) v->reserve(total);
  for (std::size_t k = 0; k < kFeatures; ++k) {
    feat[k].reserve(total);
    raw[k].reserve(total);
  }
  std::array<std::uint64_t, EsSimulator::kTakerKinds> vol0{};
  std::uint64_t inf0 = 0, uninf0 = 0, rep0 = 0, rept0 = 0;
  double clamped = 0;
  for (std::size_t s = 0; s < warm + total; ++s) {
    sim.step();
    const auto& tr = sim.trades();
    const std::size_t first_snipe = tr.size() - sim.sniper_prints();
    std::array<double, 8> b{};
    for (std::size_t k = 0; k < tr.size(); ++k) {
      const Trade& t = tr[k];
      fb.add_trade(t);
      if (s >= warm) meter.add_trade(t);
      const double sg = sign(t.aggressor), sz = t.size;
      b[0] += sg;
      b[1] += 1;
      b[2] += sz;
      b[3] += sg * sz;
      if (k >= first_snipe) {
        b[4] += sg;
        b[5] += 1;
        b[6] += sz;
        b[7] += sg * sz;
      }
    }
    const Quote q = sim.quote();
    fb.end_bucket(q);
    if (s + 1 == warm) {
      for (std::size_t k = 0; k < EsSimulator::kTakerKinds; ++k) vol0[k] = sim.taker_volume(static_cast<EsSimulator::TakerKind>(k));
      inf0 = sim.meta_starts(true);
      uninf0 = sim.meta_starts(false);
      rep0 = sim.reprices();
      rept0 = sim.reprice_ticks();
    }
    if (s < warm) continue;
    meter.end_bucket(q);
    mid.push_back(sim.mid());
    val.push_back(sim.fair_value());
    gap.push_back(sim.fair_value() - sim.mid());
    act.push_back(sim.activity());
    if (sim.log_activity() >= p.activity_max - 1e-9) clamped += 1;
    net.push_back(b[0]);
    prints.push_back(b[1]);
    vol.push_back(b[2]);
    sv.push_back(b[3]);
    snet.push_back(b[4]);
    sprints.push_back(b[5]);
    svol.push_back(b[6]);
    ssv.push_back(b[7]);
    const auto& x = fb.values();
    const auto r = fb.raw();
    for (std::size_t k = 0; k < kFeatures; ++k) {
      feat[k].push_back(x[k]);
      raw[k].push_back(r[k]);
    }
  }

  Rows rows;
  auto put = [&rows](const std::string& m, double h, double v) { rows.push_back({m, h, v}); };
  const std::size_t n = mid.size(), hmax = per(60), s1 = per(1), s2 = per(2), s30 = per(30);
  if (n <= hmax + 2) return rows;
  const std::size_t m_end = n - hmax;  // samples with every horizon ahead of them
  // (a), (b): signals vs forward mid changes.
  std::vector<std::pair<std::string, const std::vector<double>*>> sig = {{"gap", &gap}, {"I1", &raw[6]}, {"I5", &raw[7]}};
  for (std::size_t k = 0; k < kFeatures; ++k) sig.push_back({std::string("f_") + kFeatureNames[k], &feat[k]});
  for (const auto& [name, xp] : sig) {
    const auto& x = *xp;
    for (double h : kHorizons) {
      const std::size_t hs = per(h);
      Mom mm;
      for (std::size_t i = 0; i < m_end; ++i) mm.add(x[i], mid[i + hs] - mid[i]);
      put("C_" + name, h, mm.corr());
      put("B_" + name, h, mm.slope());
    }
    Mom t1, t2, lk;
    for (std::size_t i = 0; i < m_end; ++i) {
      t1.add(x[i], mid[i + s30] - mid[i + s1]);
      t2.add(x[i], mid[i + s30] - mid[i + s2]);
      lk.add(x[i], gap[i]);
    }
    put("tail1_" + name, 30, t1.corr());  // corr(x, mid(t+30 s) - mid(t+1 s))
    put("tail2_" + name, 30, t2.corr());  // corr(x, mid(t+30 s) - mid(t+2 s))
    put("leak_" + name, 0, lk.corr());    // corr(x, hidden gap)
  }
  // Event time: the mid j changes later.
  std::vector<std::size_t> changes;
  for (std::size_t i = 1; i < n; ++i)
    if (mid[i] != mid[i - 1]) changes.push_back(i);
  for (std::size_t si = 0; si < 3; ++si) {
    const auto& x = *sig[si].second;
    for (std::size_t j : kEvents) {
      Mom mm;
      std::size_t c = 0;
      for (std::size_t i = 0; i < n; ++i) {
        while (c < changes.size() && changes[c] <= i) ++c;
        if (c + j - 1 >= changes.size()) break;
        mm.add(x[i], mid[changes[c + j - 1]] - mid[i]);
      }
      put("event_" + sig[si].first, static_cast<double>(j), mm.corr());
    }
  }
  // After a mid change: the mean later move in its direction (< 0: the price
  // bounces back), and how often the next change reverses it.
  {
    for (double h : {1.0, 5.0, 30.0}) {
      const std::size_t hs = per(h);
      double acc = 0, cnt = 0;
      for (std::size_t c : changes)
        if (c + hs < n) {
          acc += (mid[c] > mid[c - 1] ? 1.0 : -1.0) * (mid[c + hs] - mid[c]);
          cnt += 1;
        }
      put("after_move_drift", h, cnt > 0 ? acc / cnt : 0);
    }
    double rev = 0, cnt = 0;
    for (std::size_t c = 1; c < changes.size(); ++c) {
      const double a = mid[changes[c - 1]] - mid[changes[c - 1] - 1], b = mid[changes[c]] - mid[changes[c] - 1];
      if (a * b < 0) rev += 1;
      cnt += 1;
    }
    put("next_change_reverses", 0, cnt > 0 ? rev / cnt : 0);
  }
  // Gap statistics and the discreteness floor of an ideal market (mid = floor(value) + 0.5).
  {
    Mom gs, r30, fl, flt;
    for (std::size_t i = 0; i < m_end; ++i) {
      gs.add(gap[i], 0);
      r30.add(mid[i + s30] - mid[i], 0);
      const auto ref = [&](std::size_t k) { return std::floor(val[k]) + 0.5; };
      fl.add(val[i] - ref(i), ref(i + s30) - ref(i));
      flt.add(val[i] - ref(i), ref(i + s30) - ref(i + s2));
    }
    const double sdg = std::sqrt(gs.var_x()), sdr = std::sqrt(r30.var_x());
    put("gap_sd", 0, sdg);
    put("r30_sd", 30, sdr);
    put("grid_floor_F", 30, sdr > 0 ? sdg / sdr : 0);  // sd(g) / sd(r30)
    put("ideal_floor", 30, fl.corr());                 // corr(value - ref, ref(t+30) - ref(t))
    put("ideal_floor_tail2", 30, flt.corr());
    double mx = 0, within = 0;
    for (double g : gap) {
      mx = std::max(mx, std::abs(g));
      if (std::abs(g) <= 1) within += 1;
    }
    put("gap_max_abs", 0, mx);
    put("gap_share_within_1", 0, within / static_cast<double>(n));
    for (std::size_t lag : {std::size_t{1}, std::size_t{10}, std::size_t{100}}) {
      Mom a;
      for (std::size_t i = 0; i + lag < n; ++i) a.add(gap[i], gap[i + lag]);
      put("gap_ac", static_cast<double>(lag) * p.dt, a.corr());
    }
    // Jump catch-up: steps with |d value| > 2 until |gap| <= 1.
    std::vector<double> catchup;
    for (std::size_t i = 1; i < n; ++i)
      if (std::abs(val[i] - val[i - 1]) > 2) {
        std::size_t k = i;
        while (k < n && std::abs(gap[k]) > 1 && k - i < hmax * 10) ++k;
        catchup.push_back(static_cast<double>(k - i) * p.dt);
      }
    put("jumps", 0, static_cast<double>(catchup.size()));
    put("jump_catchup_median_s", 0, quantile(catchup, 0.5));
    put("jump_catchup_p95_s", 0, quantile(catchup, 0.95));
  }
  // (c) Markouts after aggressive prints.
  for (double h : kMarkouts) {
    const std::size_t hs = per(h);
    double cnt_n = 0, cnt_d = 0, pr_n = 0, pr_d = 0, vo_n = 0, vo_d = 0, sp_n = 0, sp_d = 0, ns_n = 0, ns_d = 0, spv_n = 0, spv_d = 0, nsv_n = 0, nsv_d = 0;
    for (std::size_t i = 0; i < m_end; ++i) {
      const double dm = mid[i + hs] - mid[i];
      if (net[i] != 0) {  // as StylizedMeter: net signed count per bucket
        cnt_n += net[i] * dm;
        cnt_d += std::abs(net[i]);
      }
      pr_n += net[i] * dm;
      pr_d += prints[i];
      vo_n += sv[i] * dm;
      vo_d += vol[i];
      sp_n += snet[i] * dm;
      sp_d += sprints[i];
      ns_n += (net[i] - snet[i]) * dm;
      ns_d += prints[i] - sprints[i];
      spv_n += ssv[i] * dm;
      spv_d += svol[i];
      nsv_n += (sv[i] - ssv[i]) * dm;
      nsv_d += vol[i] - svol[i];
    }
    const auto ratio = [](double a, double b) { return b > 0 ? a / b : 0.0; };
    put("markout", h, ratio(cnt_n, cnt_d));
    put("markout_print", h, ratio(pr_n, pr_d));
    put("markout_volume", h, ratio(vo_n, vo_d));
    put("markout_sniper_print", h, ratio(sp_n, sp_d));
    put("markout_other_print", h, ratio(ns_n, ns_d));
    put("markout_sniper_volume", h, ratio(spv_n, spv_d));
    put("markout_other_volume", h, ratio(nsv_n, nsv_d));
  }
  // Leak scan C: 30 s return autocorrelation, delta / OFI 30 s vs the next 30 s (overlapping, every step).
  {
    Mom ac, dl, of;
    for (std::size_t i = s30; i + s30 < n; ++i) {
      const double next = mid[i + s30] - mid[i];
      ac.add(mid[i] - mid[i - s30], next);
      dl.add(raw[1][i], next);
      of.add(raw[4][i], next);
    }
    put("ret30_ac1", 30, ac.corr());
    put("delta30_next30", 30, dl.corr());
    put("ofi30_next30", 30, of.corr());
  }
  // (d) Stylized facts and health.
  const StylizedFacts f = meter.facts();
  const std::vector<std::pair<const char*, double>> facts = {
      {"trades_per_sec", f.trades_per_sec}, {"mean_trade_size", f.mean_trade_size}, {"volume_per_min", f.volume_per_min},
      {"spread1_share", f.spread1_share},   {"mean_l1_size", f.mean_l1_size},       {"ret1m_sd", f.ret1m_sd},
      {"ret1m_kurtosis", f.ret1m_kurtosis}, {"ret1m_ac1", f.ret1m_ac1},             {"absret1m_ac1", f.absret1m_ac1},
      {"sign_ac1", f.sign_ac1},             {"sign_ac10", f.sign_ac10},             {"sign_ac100", f.sign_ac100},
      {"impact_r2", f.impact_r2},           {"delta_predict", f.delta_predict},     {"markout_1s", f.markout_1s},
      {"markout_5s", f.markout_5s},         {"markout_30s", f.markout_30s},         {"mid_changes_per_min", f.mid_changes_per_min}};
  for (const auto& [k, v] : facts) put(std::string("fact_") + k, 0, v);
  const double minutes = static_cast<double>(n) * p.dt / 60;
  double am = 0, ax = 0;
  for (double a : act) {
    am += a;
    ax = std::max(ax, a);
  }
  put("activity_mean", 0, am / static_cast<double>(n));
  put("activity_max", 0, ax);
  put("activity_clamped_share", 0, clamped / static_cast<double>(n));
  const double reps = static_cast<double>(sim.reprices() - rep0);
  put("reprices_per_min", 0, reps / minutes);
  put("ticks_per_reprice", 0, reps > 0 ? static_cast<double>(sim.reprice_ticks() - rept0) / reps : 0);
  put("informed_starts_per_min", 0, static_cast<double>(sim.meta_starts(true) - inf0) / minutes);
  put("uninformed_starts_per_min", 0, static_cast<double>(sim.meta_starts(false) - uninf0) / minutes);
  double tv = 0;
  std::array<double, EsSimulator::kTakerKinds> kv{};
  for (std::size_t k = 0; k < EsSimulator::kTakerKinds; ++k) {
    kv[k] = static_cast<double>(sim.taker_volume(static_cast<EsSimulator::TakerKind>(k)) - vol0[k]);
    tv += kv[k];
  }
  const std::array<const char*, EsSimulator::kTakerKinds> kinds = {"noise", "informed_meta", "uninformed_meta", "momentum", "sniper"};
  for (std::size_t k = 0; k < EsSimulator::kTakerKinds; ++k) put(std::string("volume_share_") + kinds[k], 0, tv > 0 ? kv[k] / tv : 0);
  return rows;
}

// ---- (e, f) rules on ScalpEnv ------------------------------------------------
using Policy = std::function<Action(const ScalpObs&, const ScalpEnv&)>;
std::vector<std::pair<std::string, Policy>> strategies() {
  auto hold = [](const ScalpObs& o) { return static_cast<Action>(o.position + 1); };
  auto signal_rule = [hold](std::size_t k, double th, bool reverse) -> Policy {
    return [=](const ScalpObs& o, const ScalpEnv&) -> Action {
      const double x = reverse ? -o.f[k] : o.f[k];
      return x > th ? 2 : x < -th ? 0 : hold(o);
    };
  };
  auto oracle = [hold](double th) -> Policy {
    return [=](const ScalpObs& o, const ScalpEnv& e) -> Action {
      const double g = e.market().fair_value() - e.market().mid();
      return g > th ? 2 : g < -th ? 0 : hold(o);
    };
  };
  return {{"flat", [](const ScalpObs&, const ScalpEnv&) -> Action { return 1; }},
          {"delta 30 s > 1.5", signal_rule(1, 1.5, false)},
          {"delta 30 s > 2.5", signal_rule(1, 2.5, false)},
          {"delta 120 s > 2", signal_rule(2, 2.0, false)},
          {"OFI 30 s > 2", signal_rule(4, 2.0, false)},
          {"imbalance L1 > 0.6", signal_rule(6, 1.2, false)},
          {"absorption > 2", signal_rule(12, 2.0, false)},
          {"fade return 30 s > 2", signal_rule(9, 2.0, true)},
          {"oracle (gap > 0.5)", oracle(0.5)},
          {"oracle (gap > 0.75)", oracle(0.75)},
          {"oracle (gap > 1)", oracle(1.0)},
          {"oracle (gap > 2)", oracle(2.0)}};
}

struct RuleRun {
  std::vector<double> net, orders, fees;
};

RuleRun run_rule(const SimParams& sim, std::uint64_t seed, const Policy& policy, unsigned episodes) {
  ScalpParams p;
  p.sim = sim;
  RuleRun rr;
  for (unsigned i = 0; i < episodes; ++i) {
    ScalpEnv env(p);
    rag::agent::Rng r(1'000'000ull * seed + i);  // the simulator-design rule-check sub-block
    auto o = env.reset(r);
    for (;;) {
      const auto st = env.step(policy(o, env), r);
      o = st.obs;
      if (st.done) break;
    }
    rr.net.push_back(env.net_pnl_ticks());
    rr.orders.push_back(env.orders());
    rr.fees.push_back(env.fees_ticks());
  }
  return rr;
}

// ---- output -------------------------------------------------------------------
std::FILE* g_out = nullptr;
void say(const char* fmt, ...) {
  std::va_list ap;
  va_start(ap, fmt);
  std::va_list ap2;
  va_copy(ap2, ap);
  std::vprintf(fmt, ap);
  if (g_out) std::vfprintf(g_out, fmt, ap2);
  va_end(ap2);
  va_end(ap);
}

std::vector<std::uint64_t> parse_seeds(const std::string& s) {
  std::vector<std::uint64_t> v;
  std::size_t start = 0;
  while (start < s.size()) {
    std::size_t end = s.find(',', start);
    if (end == std::string::npos) end = s.size();
    const std::string tok = s.substr(start, end - start);
    const std::size_t dash = tok.find('-');
    if (dash != std::string::npos) {
      const std::uint64_t a = std::strtoull(tok.substr(0, dash).c_str(), nullptr, 10), b = std::strtoull(tok.substr(dash + 1).c_str(), nullptr, 10);
      for (std::uint64_t x = a; x <= b; ++x) v.push_back(x);
    } else if (!tok.empty()) {
      v.push_back(std::strtoull(tok.c_str(), nullptr, 10));
    }
    start = end + 1;
  }
  return v;
}

std::vector<std::string> split(const std::string& s) {
  std::vector<std::string> v;
  std::size_t start = 0;
  while (start <= s.size()) {
    std::size_t end = s.find(',', start);
    if (end == std::string::npos) end = s.size();
    if (end > start) v.push_back(s.substr(start, end - start));
    start = end + 1;
  }
  return v;
}

// Per-seed values of one metric: rows[seed index][(metric, horizon)].
struct Table {
  std::vector<std::map<std::pair<std::string, double>, double>> seed;
  [[nodiscard]] double at(std::size_t s, const std::string& m, double h) const {
    const auto it = seed[s].find({m, h});
    return it == seed[s].end() ? std::nan("") : it->second;
  }
};

// One summary line: mean [min, max] over seeds, and PASS/FAIL if every seed passes.
void line(const Table& t, const std::string& label, const std::function<double(std::size_t)>& value, const char* crit = nullptr,
          const std::function<bool(std::size_t)>& pass = {}) {
  double m = 0, lo = 1e300, hi = -1e300;
  bool ok = true;
  const std::size_t ns = t.seed.size();
  for (std::size_t s = 0; s < ns; ++s) {
    const double v = value(s);
    m += v / static_cast<double>(ns);
    lo = std::min(lo, v);
    hi = std::max(hi, v);
    if (pass && !pass(s)) ok = false;
  }
  say("  %-44s %+9.3f  [%+8.3f, %+8.3f]", label.c_str(), m, lo, hi);
  if (crit) say("  %-30s %s", crit, ok ? "PASS" : "FAIL");
  say("\n");
}

}  // namespace

int main(int argc, char** argv) {
  Options o;
  for (int i = 1; i + 1 < argc; i += 2) {
    const std::string k = argv[i], v = argv[i + 1];
    if (k == "--markets") o.markets = split(v);
    else if (k == "--seeds") o.seeds = parse_seeds(v);
    else if (k == "--seconds") o.seconds = std::atof(v.c_str());
    else if (k == "--threads") o.threads = static_cast<unsigned>(std::max(1, std::atoi(v.c_str())));
    else if (k == "--rules") o.rules = static_cast<unsigned>(std::max(0, std::atoi(v.c_str())));
    else if (k == "--csv") o.csv = v;
    else if (k == "--out") o.out = v;
    else if (k == "--set") {
      for (const auto& kv : split(v)) {
        const std::size_t eq = kv.find('=');
        if (eq == std::string::npos) {
          std::fprintf(stderr, "--set wants name=value, got %s\n", kv.c_str());
          return 2;
        }
        o.set.push_back({kv.substr(0, eq), std::atof(kv.substr(eq + 1).c_str())});
      }
    } else {
      std::fprintf(stderr, "unknown option %s\n", k.c_str());
      return 2;
    }
  }
  if (o.seeds.empty() || o.markets.empty()) return 2;
  std::vector<SimParams> params;
  try {
    for (const auto& m : o.markets) params.push_back(market(o, m));
  } catch (const std::exception& e) {
    std::fprintf(stderr, "%s\n", e.what());
    return 2;
  }
  if (!o.out.empty()) g_out = std::fopen(o.out.c_str(), "w");
  const auto strat = strategies();
  // Tasks: one diagnosis per (market, seed), one rule run per (market, seed, rule).
  const std::size_t nm = o.markets.size(), ns = o.seeds.size(), nr = o.rules > 0 ? strat.size() : 0;
  std::vector<Rows> diag(nm * ns);
  std::vector<RuleRun> rule(nm * ns * nr);
  const std::size_t ntasks = nm * ns + nm * ns * nr;
  std::atomic<std::size_t> next{0};
  auto worker = [&] {
    for (std::size_t t; (t = next.fetch_add(1)) < ntasks;) {
      if (t < nm * ns) {
        diag[t] = diagnose(params[t / ns], o.seeds[t % ns], o);
      } else {
        const std::size_t u = t - nm * ns, ms = u / nr;
        rule[u] = run_rule(params[ms / ns], o.seeds[ms % ns], strat[u % nr].second, o.rules);
      }
    }
  };
  std::vector<std::thread> pool;
  for (unsigned i = 0; i < std::min<std::size_t>(o.threads, ntasks); ++i) pool.emplace_back(worker);
  for (auto& th : pool) th.join();

  if (!o.csv.empty()) {
    if (std::FILE* c = std::fopen(o.csv.c_str(), "w")) {
      std::fprintf(c, "market,seed,metric,horizon_s,value\n");
      for (std::size_t mi = 0; mi < nm; ++mi)
        for (std::size_t si = 0; si < ns; ++si) {
          for (const auto& r : diag[mi * ns + si])
            std::fprintf(c, "%s,%llu,%s,%g,%.6g\n", o.markets[mi].c_str(), static_cast<unsigned long long>(o.seeds[si]), r.metric.c_str(), r.horizon, r.value);
          for (std::size_t ri = 0; ri < nr; ++ri) {
            const auto& rr = rule[(mi * ns + si) * nr + ri];
            double m = 0;
            for (double x : rr.net) m += x / static_cast<double>(rr.net.size());
            std::fprintf(c, "%s,%llu,rule:%s,0,%.6g\n", o.markets[mi].c_str(), static_cast<unsigned long long>(o.seeds[si]), strat[ri].first.c_str(), m);
          }
        }
      std::fclose(c);
    }
  }

  say("OF10 price-discovery diagnostics: seeds");
  for (auto s : o.seeds) say(" %llu", static_cast<unsigned long long>(s));
  say(", %.0f s each after a %.0f s warm-up (session start 3600 s); mean [min, max] over seeds\n", o.seconds, o.warmup);
  if (!o.set.empty()) {  // tuned runs must say so (the market header prints only some fields)
    say("--set overrides of the discovery preset:");
    for (const auto& [k, x] : o.set) say(" %s=%g", k.c_str(), x);
    say("\n");
  }
  for (std::size_t mi = 0; mi < nm; ++mi) {
    const SimParams& p = params[mi];
    Table t;
    t.seed.resize(ns);
    for (std::size_t si = 0; si < ns; ++si)
      for (const auto& r : diag[mi * ns + si]) t.seed[si][{r.metric, r.horizon}] = r.value;
    auto v = [&](const std::string& m, double h) { return [&t, m, h](std::size_t s) { return t.at(s, m, h); }; };
    say("\n=== market %s (reprice_band %.2f, reprice_snipe %.2f, maker_info %.2f, informed_meta_rate %.2f, uninformed_meta_rate %.2f, momentum_rate %.2f,\n"
        "    noise_rate %.2f, activity_excite %.2f, value_vol %.2f)\n",
        o.markets[mi].c_str(), p.reprice_band, p.reprice_snipe, p.maker_info, p.informed_meta_rate, p.uninformed_meta_rate, p.momentum_rate, p.noise_rate,
        p.activity_excite, p.value_vol);
    say("    iid-window SE of a 30 s correlation (a lower bound: activity clustering inflates it, ~2x seen; judge by the seed range):"
        " one seed %.3f, mean of %zu seeds %.3f\n",
        1 / std::sqrt(o.seconds / 30), ns, 1 / std::sqrt(o.seconds / 30 * static_cast<double>(ns)));
    say("(a) hidden gap g = fair value - mid\n");
    line(t, "sd(g), ticks", v("gap_sd", 0), "<= 0.45", [&](std::size_t s) { return t.at(s, "gap_sd", 0) <= 0.45; });
    line(t, "ideal-grid floor corr(V-ref, ref 30 s)", v("ideal_floor", 30));
    line(t, "grid floor F = sd(g) / sd(r30)", v("grid_floor_F", 30));
    line(t, "corr(g, r 30 s)", v("C_gap", 30), "|.| <= max(0.10, floor+0.03)", [&](std::size_t s) {
      return std::abs(t.at(s, "C_gap", 30)) <= std::max(0.10, t.at(s, "ideal_floor", 30) + 0.03);
    });
    line(t, "corr(g, r 30 s), literal", v("C_gap", 30), "|.| < 0.10", [&](std::size_t s) { return std::abs(t.at(s, "C_gap", 30)) < 0.10; });
    line(t, "tail corr(g, mid(t+30) - mid(t+2 s))", v("tail2_gap", 30), "|.| <= 0.03", [&](std::size_t s) { return std::abs(t.at(s, "tail2_gap", 30)) <= 0.03; });
    line(t, "tail corr(g, mid(t+30) - mid(t+1 s))", v("tail1_gap", 30));
    line(t, "ideal-grid tail corr (2 s -> 30 s)", v("ideal_floor_tail2", 30));
    line(t, "slope B_g(2 s) / B_g(30 s)", [&](std::size_t s) { return t.at(s, "B_gap", 2) / t.at(s, "B_gap", 30); }, ">= 0.8",
         [&](std::size_t s) { return t.at(s, "B_gap", 2) >= 0.8 * t.at(s, "B_gap", 30); });
    line(t, "slope B_g(30 s) / B_g(1 s)", [&](std::size_t s) { return t.at(s, "B_gap", 30) / t.at(s, "B_gap", 1); });
    say("  corr(g, r h) by horizon h (s):");
    for (double h : kHorizons) {
      double m = 0;
      for (std::size_t s = 0; s < ns; ++s) m += t.at(s, "C_gap", h) / static_cast<double>(ns);
      say(" %g:%+.3f", h, m);
    }
    say("\n");
    line(t, "gap autocorrelation 0.1 s", v("gap_ac", 0.1));
    line(t, "gap autocorrelation 1 s", v("gap_ac", 1.0));
    line(t, "gap autocorrelation 10 s", v("gap_ac", 10.0));
    line(t, "share of steps with |g| <= 1", v("gap_share_within_1", 0));
    line(t, "max |g|", v("gap_max_abs", 0));
    line(t, "value jumps > 2 ticks (count)", v("jumps", 0));
    line(t, "jump catch-up (|g| <= 1) median, s", v("jump_catchup_median_s", 0));
    line(t, "jump catch-up p95, s", v("jump_catchup_p95_s", 0));
    line(t, "after a mid change: drift 1 s in its direction", v("after_move_drift", 1));
    line(t, "after a mid change: drift 5 s", v("after_move_drift", 5));
    line(t, "after a mid change: drift 30 s", v("after_move_drift", 30));
    line(t, "share of mid changes reversing the last one", v("next_change_reverses", 0));
    say("(b) book imbalance and features\n");
    for (const char* name : {"I1", "I5"}) {
      const std::string nm2 = name;
      line(t, "corr(" + nm2 + ", r 0.1 s)", v("C_" + nm2, 0.1));
      line(t, "corr(" + nm2 + ", r 1 s)", v("C_" + nm2, 1));
      line(t, "corr(" + nm2 + ", r 30 s)", v("C_" + nm2, 30), "|.| <= 0.05", [&t, nm2](std::size_t s) { return std::abs(t.at(s, "C_" + nm2, 30)) <= 0.05; });
      line(t, "tail corr(" + nm2 + ", mid(t+30) - mid(t+1 s))", v("tail1_" + nm2, 30));
      line(t, "tail corr(" + nm2 + ", mid(t+30) - mid(t+2 s))", v("tail2_" + nm2, 30), "|.| <= 0.03",
           [&t, nm2](std::size_t s) { return std::abs(t.at(s, "tail2_" + nm2, 30)) <= 0.03; });
      line(t, "slope B(2 s) / B(30 s)", [&t, nm2](std::size_t s) { return t.at(s, "B_" + nm2, 2) / t.at(s, "B_" + nm2, 30); }, ">= 0.8 or |C30| <= 0.02",
           [&t, nm2](std::size_t s) {
             return t.at(s, "B_" + nm2, 2) >= 0.8 * t.at(s, "B_" + nm2, 30) || std::abs(t.at(s, "C_" + nm2, 30)) <= 0.02;
           });
      say("  event time corr(%s, mid j changes later):", name);
      for (std::size_t j : kEvents) {
        double m = 0;
        for (std::size_t s = 0; s < ns; ++s) m += t.at(s, "event_" + nm2, static_cast<double>(j)) / static_cast<double>(ns);
        say(" j=%zu:%+.3f", j, m);
      }
      say("\n");
      line(t, "leak corr(" + nm2 + ", g)", v("leak_" + nm2, 0), nm2 == "I1" ? "|.| <= 0.15" : nullptr,
           [&t, nm2](std::size_t s) { return nm2 != "I1" || std::abs(t.at(s, "leak_" + nm2, 0)) <= 0.15; });
    }
    say("  %-10s %8s %8s %8s %8s %8s   flag (|tail2| > 0.03 or |leak| > 0.2)\n", "feature", "C 1 s", "C 30 s", "tail1", "tail2", "leak");
    for (std::size_t k = 0; k < kFeatures; ++k) {
      const std::string f = std::string("f_") + kFeatureNames[k];
      double c1 = 0, c30 = 0, t1 = 0, t2 = 0, lk = 0;
      bool flag = false;
      for (std::size_t s = 0; s < ns; ++s) {
        const double d = static_cast<double>(ns);
        c1 += t.at(s, "C_" + f, 1) / d;
        c30 += t.at(s, "C_" + f, 30) / d;
        t1 += t.at(s, "tail1_" + f, 30) / d;
        t2 += t.at(s, "tail2_" + f, 30) / d;
        lk += t.at(s, "leak_" + f, 0) / d;
        flag = flag || std::abs(t.at(s, "tail2_" + f, 30)) > 0.03 || std::abs(t.at(s, "leak_" + f, 0)) > 0.2;
      }
      say("  %-10s %+8.3f %+8.3f %+8.3f %+8.3f %+8.3f   %s\n", kFeatureNames[k], c1, c30, t1, t2, lk, flag ? "FLAG" : "ok");
    }
    line(t, "leak scan: 30 s return autocorrelation", v("ret30_ac1", 30), "|.| <= 0.03", [&](std::size_t s) { return std::abs(t.at(s, "ret30_ac1", 30)) <= 0.03; });
    line(t, "leak scan: corr(delta 30 s, next 30 s)", v("delta30_next30", 30));
    line(t, "leak scan: corr(OFI 30 s, next 30 s)", v("ofi30_next30", 30));
    say("(c) markout after aggressive prints, ticks (count-weighted as StylizedMeter)\n");
    say("  horizon (s)             ");
    for (double h : kMarkouts) say(" %7g", h);
    say("\n");
    for (const char* mk : {"markout", "markout_print", "markout_volume", "markout_sniper_print", "markout_other_print", "markout_sniper_volume",
                           "markout_other_volume"}) {
      say("  %-24s", mk);
      for (double h : kMarkouts) {
        double m = 0;
        for (std::size_t s = 0; s < ns; ++s) m += t.at(s, mk, h) / static_cast<double>(ns);
        say(" %+7.3f", m);
      }
      say("\n");
    }
    line(t, "markout 30 s - markout 5 s", [&](std::size_t s) { return t.at(s, "markout", 30) - t.at(s, "markout", 5); }, "|.| <= 0.10",
         [&](std::size_t s) { return std::abs(t.at(s, "markout", 30) - t.at(s, "markout", 5)) <= 0.10; });
    line(t, "markout 60 s - markout 5 s", [&](std::size_t s) { return t.at(s, "markout", 60) - t.at(s, "markout", 5); }, "|.| <= 0.15",
         [&](std::size_t s) { return std::abs(t.at(s, "markout", 60) - t.at(s, "markout", 5)) <= 0.15; });
    say("(d) stylized facts and health\n");
    struct Rng2 {
      const char* key;
      double lo, hi;
      const char* crit;
    };
    const std::vector<Rng2> ranges = {{"trades_per_sec", 15, 22, "15 .. 22"},
                                      {"mean_trade_size", 3.4, 4.4, "3.4 .. 4.4"},
                                      {"volume_per_min", -1e9, 1e9, nullptr},
                                      {"spread1_share", 0.95, 1.01, ">= 0.95"},
                                      {"mean_l1_size", -1e9, 1e9, nullptr},
                                      {"ret1m_sd", 3.0, 4.5, "3.0 .. 4.5"},
                                      {"ret1m_kurtosis", 4, 12, "4 .. 12"},
                                      {"ret1m_ac1", -0.1, 0.1, "|.| <= 0.1"},
                                      {"absret1m_ac1", 0.1, 1, ">= 0.1"},
                                      {"sign_ac1", 0.2, 1, ">= 0.2"},
                                      {"sign_ac10", -1e9, 1e9, nullptr},
                                      {"sign_ac100", 0.05, 1, ">= 0.05"},
                                      {"impact_r2", 0.3, 1, ">= 0.3"},
                                      {"delta_predict", -0.04, 0.04, "|.| <= 0.04"},
                                      {"markout_1s", -1e9, 1e9, nullptr},
                                      {"markout_5s", -1e9, 1e9, nullptr},
                                      {"markout_30s", -1e9, 1e9, nullptr},
                                      {"mid_changes_per_min", -1e9, 1e9, nullptr}};
    for (const auto& r : ranges) {
      const std::string key = std::string("fact_") + r.key;
      line(t, r.key, v(key, 0), r.crit, [&t, key, r](std::size_t s) {
        const double x = t.at(s, key, 0);
        return x >= r.lo && x <= r.hi;
      });
    }
    line(t, "activity mean", v("activity_mean", 0));
    line(t, "activity max", v("activity_max", 0));
    line(t, "share of time at the activity clamp", v("activity_clamped_share", 0), "< 0.05",
         [&](std::size_t s) { return t.at(s, "activity_clamped_share", 0) < 0.05; });
    line(t, "quote jumps per minute", v("reprices_per_min", 0));
    line(t, "ticks per quote jump", v("ticks_per_reprice", 0));
    line(t, "informed meta-order starts per minute", v("informed_starts_per_min", 0));
    line(t, "uninformed meta-order starts per minute", v("uninformed_starts_per_min", 0));
    for (const char* k : {"noise", "informed_meta", "uninformed_meta", "momentum", "sniper"}) line(t, std::string("volume share: ") + k, v(std::string("volume_share_") + k, 0));
    if (nr > 0) {
      say("(e, f) rules on ScalpEnv: %u episodes x %zu seeds (episode seeds 1,000,000 * seed + i), net ticks/episode after costs\n", o.rules, ns);
      say("  %-24s %9s %7s %8s %11s %11s   per-seed means\n", "strategy", "ticks/ep", "t", "orders", "gross/order", "edge/order");
      say("  (gross: before fees, after the spread; edge = gross + 0.5 tick, the half spread a market order pays: ~0 for no information)\n");
      for (std::size_t ri = 0; ri < nr; ++ri) {
        std::vector<double> all;
        double orders = 0, fees = 0;
        std::vector<double> seed_means;
        bool naive_win = false;
        for (std::size_t si = 0; si < ns; ++si) {
          const auto& rr = rule[(mi * ns + si) * nr + ri];
          double m = 0, var = 0;
          for (double x : rr.net) m += x / static_cast<double>(rr.net.size());
          for (double x : rr.net) var += (x - m) * (x - m);
          const double se = rr.net.size() > 1 ? std::sqrt(var / static_cast<double>(rr.net.size() - 1) / static_cast<double>(rr.net.size())) : 0;
          seed_means.push_back(m);
          if (m > 0.5 && se > 0 && m / se > 2) naive_win = true;
          all.insert(all.end(), rr.net.begin(), rr.net.end());
          for (double x : rr.orders) orders += x;
          for (double x : rr.fees) fees += x;
        }
        double m = 0, var = 0;
        for (double x : all) m += x / static_cast<double>(all.size());
        for (double x : all) var += (x - m) * (x - m);
        const double se = all.size() > 1 ? std::sqrt(var / static_cast<double>(all.size() - 1) / static_cast<double>(all.size())) : 0;
        const double gross = orders > 0 ? (m * static_cast<double>(all.size()) + fees) / orders : 0;
        const std::string& name = strat[ri].first;
        const bool is_oracle = name.rfind("oracle", 0) == 0, is_flat = name == "flat";
        const double edge = orders > 0 ? gross + 0.5 : 0;
        say("  %-24s %+9.2f %+7.2f %8.1f %+11.3f %+11.3f  ", name.c_str(), m, se > 0 ? m / se : 0.0, orders / static_cast<double>(all.size()), gross, edge);
        for (double x : seed_means) say(" %+7.2f", x);
        if (!is_oracle && !is_flat) say("   %s", naive_win || (m > 0.5 && se > 0 && m / se > 2) ? "FAIL (naive rule wins)" : "ok");
        if (name == "delta 30 s > 1.5") say(", edge/order %s", std::abs(edge) <= 0.15 ? "within +-0.15 PASS" : "outside +-0.15 FAIL");
        if (is_oracle) say("   %s", m <= 1.0 ? "<= +1 (oracle ceiling small)" : "> +1");
        say("\n");
      }
    }
  }
  if (g_out) std::fclose(g_out);
}
