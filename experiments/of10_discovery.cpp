// OF10: price-discovery diagnostics. Measures, for each market preset and
// seed, every acceptance criterion of the price-discovery fix (ORDERFLOW.md
// plan item 1) in one run:
//   (a) the hidden gap g = fair value - mid: does it predict the mid, and
//       how fast (horizons 0.1 s ... 60 s, the 2 s -> 30 s tail, slopes),
//       against an ideal market on the same value path whose mid is
//       floor(value) + 0.5 (a 1-tick grid can't do better); stale-quote
//       races (v2 depletion: how long, how they end);
//   (b) book imbalance (L1, L1-5) and the 13 features: forward-return
//       correlations in clock time and event time (mid changes), the
//       increments beyond the first change, tails, and leaks;
//   (c) markouts after aggressive prints: prevailing-mid convention (mid
//       before the bucket's prints) and StylizedMeter's (end of the bucket),
//       sniper vs other prints;
//   (d) the stylized facts (stylized.hpp), price-on-flow R^2 for all,
//       non-sniper and sniper prints, market health;
//   (e, f) --rules N: OF2 rules (with fade rules), hysteresis fair-value
//       oracles and H-second clairvoyants on ScalpEnv, N episodes per seed,
//       episode seeds 1,000,000 * seed + i (the simulator-design rule-check
//       sub-block, not a sealed block);
//       --tick N: tick rules, the fair-value quoter and a clairvoyant quoter
//       on TickScalpEnv, episode seeds 1,000,000 * seed + 500,000 + i;
//       --impact N: a paired TWAP impact test, N pairs per seed, seeds
//       1,000,000 * seed + 700,000 + i.
// Each (market, seed) runs EsSimulator(preset, seed, start 3600 s) for
// --seconds after a 150 s warm-up, one thread each.
//
// Standard errors: every correlation, slope, markout and R^2 is accumulated
// per 5-minute block; "SE" is the leave-one-block-out jackknife over all
// blocks of all seeds (pooled), next to the seed mean +- sd / sqrt(n). A
// pooled check is PASS when the value meets the bound, FAIL when it misses
// it by more than 2 SE, "fail(<2SE)" when it misses by less; "UNRESOLVED"
// is appended when 2 SE exceeds the check's tolerance.
//
// Usage: of_discovery [--markets default,efficient,discovery_v1,discovery]
//          [--seeds 1-3] [--seconds 7200] [--threads 8] [--rules 0]
//          [--tick 0] [--impact 0] [--theta [market=]X,...] [--csv FILE] [--out FILE]
//          [--set name=value,...]
// --set overrides fields of the discovery preset (tuning), e.g.
// --set reprice_cancel=8,reprice_take=0.1 (names: see kTunable).
// --theta fixes the oracle threshold theta* (pre-registered on seeds 1-3) per
// market (--theta default=1.00,discovery=0.70), or for every market when bare;
// a market without one uses the best theta of this run, marked so.
// --out also writes the summary to FILE; --out / --csv that can't be opened exit 2.
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
#include "rag/orderflow/tick_env.hpp"

using namespace rag::orderflow;
using rag::agent::Action;

namespace {

struct Options {
  std::vector<std::string> markets{"default", "efficient", "discovery_v1", "discovery"};
  std::vector<std::uint64_t> seeds{1, 2, 3};
  double seconds = 7200, warmup = 150, trace = -1;
  std::map<std::string, double> theta;  // pre-registered theta* per market ("" = every market)
  unsigned threads = 8, rules = 0, tick = 0, impact = 0;
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
    {"cancel_rate", &SimParams::cancel_rate},         {"taker_size_mean", &SimParams::taker_size_mean},
    {"reprice_cancel", &SimParams::reprice_cancel},   {"reprice_take", &SimParams::reprice_take},
    {"value_impact", &SimParams::value_impact},       {"impact_forecast_tau", &SimParams::impact_forecast_tau},
    {"impact_forecast_weight", &SimParams::impact_forecast_weight}, {"spread_fill_prob", &SimParams::spread_fill_prob},
    {"maker_decay", &SimParams::maker_decay},         {"jump_rate", &SimParams::jump_rate},
    {"meta_min", &SimParams::meta_min},               {"meta_alpha", &SimParams::meta_alpha},
    {"maker_info", &SimParams::maker_info},           {"maker_info_max", &SimParams::maker_info_max}};

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

// Additive moments: of (x, y) for correlations and slopes, num / den for ratios.
struct Mom {
  double n = 0, sx = 0, sy = 0, sxx = 0, syy = 0, sxy = 0, num = 0, den = 0;
  void add(double x, double y) {
    n += 1;
    sx += x;
    sy += y;
    sxx += x * x;
    syy += y * y;
    sxy += x * y;
  }
  void ratio(double a, double b) {
    num += a;
    den += b;
  }
  Mom& operator+=(const Mom& o) {
    n += o.n, sx += o.sx, sy += o.sy, sxx += o.sxx, syy += o.syy, sxy += o.sxy, num += o.num, den += o.den;
    return *this;
  }
  [[nodiscard]] Mom minus(const Mom& o) const {
    Mom m = *this;
    m.n -= o.n, m.sx -= o.sx, m.sy -= o.sy, m.sxx -= o.sxx, m.syy -= o.syy, m.sxy -= o.sxy, m.num -= o.num, m.den -= o.den;
    return m;
  }
  [[nodiscard]] double cov() const { return n > 1 ? (sxy - sx * sy / n) / n : 0; }
  [[nodiscard]] double var_x() const { return n > 1 ? (sxx - sx * sx / n) / n : 0; }
  [[nodiscard]] double var_y() const { return n > 1 ? (syy - sy * sy / n) / n : 0; }
  [[nodiscard]] double corr() const {
    const double d = var_x() * var_y();
    return d > 0 ? cov() / std::sqrt(d) : 0;
  }
  [[nodiscard]] double slope() const { return var_x() > 0 ? cov() / var_x() : 0; }
  [[nodiscard]] double rat() const { return den > 0 ? num / den : 0; }
};
using Block = std::map<std::string, Mom>;
using Get = std::function<Mom(const std::string&)>;
using Stat = std::function<double(const Get&)>;

std::string key(const std::string& m, double h) {
  char b[32];
  std::snprintf(b, sizeof b, "@%g", h);
  return m + b;
}

struct Row {
  std::string metric;
  double horizon;
  double value;
};
using Rows = std::vector<Row>;

struct Diag {
  std::vector<Block> blocks;  // per 5-minute block
  Rows rows;                  // scalar per-seed values (facts, quantiles, counts)
};

double quantile(std::vector<double> v, double q) {
  if (v.empty()) return std::nan("");
  std::sort(v.begin(), v.end());
  const auto i = static_cast<std::size_t>(std::floor(q * static_cast<double>(v.size() - 1)));
  return v[i];
}

const std::array<const char*, kFeatures> kFeatureNames = {"delta5", "delta30", "delta120", "ofi5",  "ofi30", "vol30", "imbL1",
                                                          "imbL5",  "ret5",    "ret30",    "ret120", "vwap",  "absorb"};
const std::array<double, 9> kHorizons = {0.1, 0.2, 0.5, 1, 2, 5, 10, 30, 60};  // seconds
const std::array<double, 7> kMarkouts = {0.1, 1, 2, 5, 10, 30, 60};
const std::array<std::size_t, 4> kEvents = {1, 2, 5, 10};  // j-th mid change

Diag diagnose(const SimParams& p, std::uint64_t seed, const Options& o) {
  const auto per = [&](double sec) { return static_cast<std::size_t>(std::max(1L, std::lround(sec / p.dt))); };
  const std::size_t warm = per(o.warmup), total = per(o.seconds), bsteps = per(300);
  EsSimulator sim(p, seed, 3600);
  FeatureBuilder fb(p.dt);
  StylizedMeter meter(p.dt);
  std::vector<double> mid, val, gap, act, imp;
  std::vector<double> net, prints, vol, sv, snet, sprints, svol, ssv;  // per bucket: signed count, prints, volume, signed volume (s: snipers)
  std::vector<std::vector<double>> feat(kFeatures), raw(kFeatures);
  for (auto* v : {&mid, &val, &gap, &act, &imp, &net, &prints, &vol, &sv, &snet, &sprints, &svol, &ssv}) v->reserve(total);
  for (std::size_t k = 0; k < kFeatures; ++k) {
    feat[k].reserve(total);
    raw[k].reserve(total);
  }
  std::array<std::uint64_t, EsSimulator::kTakerKinds> vol0{};
  std::uint64_t inf0 = 0, uninf0 = 0, rep0 = 0, rept0 = 0, races0 = 0, scan0 = 0;
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
      races0 = sim.races();
      scan0 = sim.stale_cancelled_lots();
    }
    if (s < warm) continue;
    meter.end_bucket(q);
    mid.push_back(sim.mid());
    val.push_back(sim.fair_value());
    gap.push_back(sim.fair_value() - sim.mid());
    act.push_back(sim.activity());
    imp.push_back(sim.flow_impact());
    if (o.trace >= 0 && std::abs(static_cast<double>(s - warm) * p.dt - o.trace) < 5)  // --trace T: the book around T s (debugging)
      std::fprintf(stderr, "seed %llu t %.1f value %.2f bid %d ask %d sizes %u %u | %u %u impact %.2f prints %zu activity %.2f\n",
                   static_cast<unsigned long long>(seed), static_cast<double>(s - warm) * p.dt, sim.fair_value(), q.bid, q.ask, q.bid_size[0],
                   q.bid_size[1], q.ask_size[0], q.ask_size[1], sim.flow_impact(), tr.size(), sim.activity());
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

  Diag d;
  Rows& rows = d.rows;
  auto put = [&rows](const std::string& m, double h, double v) { rows.push_back({m, h, v}); };
  const std::size_t n = mid.size(), hmax = per(60), s1 = per(1), s2 = per(2), s10 = per(10), s30 = per(30);
  if (n <= hmax + 2) return d;
  const std::size_t nb = (n + bsteps - 1) / bsteps;
  d.blocks.resize(nb);
  // Accumulate one Mom per block over samples [i0, i1): fn(i, mom of i's block).
  auto acc = [&](const std::string& k, std::size_t i0, std::size_t i1, const std::function<void(std::size_t, Mom&)>& fn) {
    std::vector<Mom> v(nb);
    for (std::size_t i = i0; i < i1; ++i) fn(i, v[i / bsteps]);
    for (std::size_t b = 0; b < nb; ++b) d.blocks[b][k] += v[b];
  };
  const std::size_t m_end = n - hmax;  // samples with every horizon ahead of them
  const double band = p.reprice_band > 0 ? p.reprice_band : 0.5;
  auto ref = [&](std::size_t k) { return std::floor(val[k]) + 0.5; };  // the ideal grid's mid
  // (a), (b): signals vs forward mid changes.
  std::vector<std::pair<std::string, const std::vector<double>*>> sig = {{"gap", &gap}, {"I1", &raw[6]}, {"I5", &raw[7]}};
  for (std::size_t k = 0; k < kFeatures; ++k) sig.push_back({std::string("f_") + kFeatureNames[k], &feat[k]});
  for (const auto& [name, xp] : sig) {
    const auto& x = *xp;
    for (double h : kHorizons) {
      const std::size_t hs = per(h);
      acc(key("C_" + name, h), 0, m_end, [&](std::size_t i, Mom& m) { m.add(x[i], mid[i + hs] - mid[i]); });
    }
    acc("T1_" + name, 0, m_end, [&](std::size_t i, Mom& m) { m.add(x[i], mid[i + s30] - mid[i + s1]); });
    acc("T2_" + name, 0, m_end, [&](std::size_t i, Mom& m) { m.add(x[i], mid[i + s30] - mid[i + s2]); });
    acc("L_" + name, 0, m_end, [&](std::size_t i, Mom& m) { m.add(x[i], gap[i]); });
  }
  // The ideal grid on the same value path.
  acc("ideal@30", 0, m_end, [&](std::size_t i, Mom& m) { m.add(val[i] - ref(i), ref(i + s30) - ref(i)); });
  acc("ideal_t2", 0, m_end, [&](std::size_t i, Mom& m) { m.add(val[i] - ref(i), ref(i + s30) - ref(i + s2)); });
  for (double lag : {0.1, 1.0, 10.0}) {
    const std::size_t l = per(lag);
    acc(key("gapac", lag), 0, n - l, [&](std::size_t i, Mom& m) { m.add(gap[i], gap[i + l]); });
    acc(key("ideal_gapac", lag), 0, n - l, [&](std::size_t i, Mom& m) { m.add(val[i] - ref(i), val[i + l] - ref(i + l)); });
  }
  acc("r30ac", s30, n - s30, [&](std::size_t i, Mom& m) { m.add(mid[i] - mid[i - s30], mid[i + s30] - mid[i]); });
  acc("ideal_r30ac", s30, n - s30, [&](std::size_t i, Mom& m) { m.add(ref(i) - ref(i - s30), ref(i + s30) - ref(i)); });
  // Is fair value a martingale? Flow impact and value changes, 10 s vs the next 10 s.
  acc("impac10", s10, n - s10, [&](std::size_t i, Mom& m) { m.add(imp[i] - imp[i - s10], imp[i + s10] - imp[i]); });
  acc("valac10", s10, n - s10, [&](std::size_t i, Mom& m) { m.add(val[i] - val[i - s10], val[i + s10] - val[i]); });
  acc("delta30_next30", s30, n - s30, [&](std::size_t i, Mom& m) { m.add(raw[1][i], mid[i + s30] - mid[i]); });
  acc("ofi30_next30", s30, n - s30, [&](std::size_t i, Mom& m) { m.add(raw[4][i], mid[i + s30] - mid[i]); });
  // Event time: the mid j changes later, and the increments beyond the first change.
  std::vector<std::size_t> changes, ideal_changes;
  for (std::size_t i = 1; i < n; ++i) {
    if (mid[i] != mid[i - 1]) changes.push_back(i);
    if (ref(i) != ref(i - 1)) ideal_changes.push_back(i);
  }
  {
    std::vector<std::size_t> next(n, changes.size());  // index into changes of the first change after i
    for (std::size_t i = 0, c = 0; i < n; ++i) {
      while (c < changes.size() && changes[c] <= i) ++c;
      next[i] = c;
    }
    const std::size_t nc = changes.size();
    auto ok = [&](std::size_t i, std::size_t j) { return next[i] + j - 1 < nc; };
    auto ev = [&](std::size_t i, std::size_t j) { return mid[changes[next[i] + j - 1]]; };
    for (std::size_t si = 0; si < 3; ++si) {
      const auto& x = *sig[si].second;
      const std::string& nm = sig[si].first;
      for (std::size_t j : kEvents)
        acc(key("E_" + nm, static_cast<double>(j)), 0, n, [&](std::size_t i, Mom& m) {
          if (ok(i, j)) m.add(x[i], ev(i, j) - mid[i]);
        });
      acc("EI15_" + nm, 0, n, [&](std::size_t i, Mom& m) {
        if (ok(i, 5)) m.add(x[i], ev(i, 5) - ev(i, 1));
      });
      acc("EI210_" + nm, 0, n, [&](std::size_t i, Mom& m) {
        if (ok(i, 10)) m.add(x[i], ev(i, 10) - ev(i, 2));
      });
    }
    // Outside stale windows (|g| <= band): is the signal more than depletion?
    acc("Ecalm_I1", 0, n, [&](std::size_t i, Mom& m) {
      if (ok(i, 1) && std::abs(gap[i]) <= band) m.add(raw[6][i], ev(i, 1) - mid[i]);
    });
  }
  // After a mid change: later drift in its direction; reversal share (ideal grid too).
  {
    for (double h : {1.0, 5.0, 30.0}) {
      const std::size_t hs = per(h);
      double a = 0, c = 0;
      for (std::size_t k : changes)
        if (k + hs < n) {
          a += (mid[k] > mid[k - 1] ? 1.0 : -1.0) * (mid[k + hs] - mid[k]);
          c += 1;
        }
      put("after_move_drift", h, c > 0 ? a / c : 0);
    }
    auto reversal = [](const std::vector<std::size_t>& ch, const std::function<double(std::size_t)>& f) {
      double rev = 0, cnt = 0;
      for (std::size_t c = 1; c < ch.size(); ++c) {
        const double a = f(ch[c - 1]) - f(ch[c - 1] - 1), b = f(ch[c]) - f(ch[c] - 1);
        if (a * b < 0) rev += 1;
        cnt += 1;
      }
      return cnt > 0 ? rev / cnt : 0.0;
    };
    put("next_change_reverses", 0, reversal(changes, [&](std::size_t k) { return mid[k]; }));
    put("ideal_next_change_reverses", 0, reversal(ideal_changes, [&](std::size_t k) { return ref(k); }));
    const double minutes = static_cast<double>(n) * p.dt / 60;
    put("mid_changes_per_min", 0, static_cast<double>(changes.size()) / minutes);
    put("ideal_changes_per_min", 0, static_cast<double>(ideal_changes.size()) / minutes);
  }
  // Gap statistics, jump catch-up, stale-quote races.
  {
    acc("gap", 0, n, [&](std::size_t i, Mom& m) { m.add(gap[i], 0); });
    acc("ideal_gap", 0, n, [&](std::size_t i, Mom& m) { m.add(val[i] - ref(i), 0); });
    double mx = 0, within = 0;
    for (double g : gap) {
      mx = std::max(mx, std::abs(g));
      if (std::abs(g) <= 1) within += 1;
    }
    put("gap_max_abs", 0, mx);
    const auto worst = std::max_element(gap.begin(), gap.end(), [](double a, double b) { return std::abs(a) < std::abs(b); });
    put("gap_max_at_s", 0, static_cast<double>(worst - gap.begin()) * p.dt);
    put("gap_share_within_1", 0, within / static_cast<double>(n));
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
    // A race: consecutive snapshots with value past the band (|g| > band) on
    // one side; it ends with a tick toward value, or with value coming back.
    std::vector<double> dur;
    double races = 0, ticked = 0, in_steps = 0;
    for (std::size_t i = 0; i < n;) {
      if (!(std::abs(gap[i]) > band)) {
        ++i;
        continue;
      }
      const double dir = gap[i] > 0 ? 1 : -1, m0 = mid[i];
      std::size_t k = i;
      while (k < n && std::abs(gap[k]) > band && dir * gap[k] > 0 && mid[k] == m0) ++k;
      races += 1;
      in_steps += static_cast<double>(k - i);
      if (k < n && dir * (mid[k] - m0) > 0) {
        ticked += 1;
        dur.push_back(static_cast<double>(k - i + 1) * p.dt);  // from the step value passed the band to the tick
      }
      i = std::max(k, i + 1);
    }
    const double minutes = static_cast<double>(n) * p.dt / 60;
    put("races_per_min", 0, races / minutes);
    put("race_tick_share", 0, races > 0 ? ticked / races : 0);
    put("race_time_share", 0, in_steps / static_cast<double>(n));
    put("race_median_s", 0, quantile(dur, 0.5));
    put("race_p95_s", 0, quantile(dur, 0.95));
    put("race_floor_share", 0, dur.empty() ? 0 : static_cast<double>(std::count_if(dur.begin(), dur.end(), [&p](double x) { return x <= 2.5 * p.dt; })) / static_cast<double>(dur.size()));
    const double sn = static_cast<double>(sim.taker_volume(EsSimulator::kSniper) - vol0[EsSimulator::kSniper]);
    const double ca = static_cast<double>(sim.stale_cancelled_lots() - scan0);
    put("stale_lots_cancel_share", 0, sn + ca > 0 ? ca / (sn + ca) : 0);
    put("sim_races_per_min", 0, static_cast<double>(sim.races() - races0) / minutes);
  }
  // (c) Markouts after aggressive prints.
  for (double h : kMarkouts) {
    const std::size_t hs = per(h);
    acc(key("MK", h), 1, m_end, [&](std::size_t i, Mom& m) {  // StylizedMeter: net signed count per bucket, end-of-bucket mid
      if (net[i] != 0) m.ratio(net[i] * (mid[i + hs] - mid[i]), std::abs(net[i]));
    });
    acc(key("MKp_all", h), 1, m_end, [&](std::size_t i, Mom& m) { m.ratio(net[i] * (mid[i + hs] - mid[i - 1]), prints[i]); });
    acc(key("MKp_snp", h), 1, m_end, [&](std::size_t i, Mom& m) { m.ratio(snet[i] * (mid[i + hs] - mid[i - 1]), sprints[i]); });
    acc(key("MKp_oth", h), 1, m_end, [&](std::size_t i, Mom& m) { m.ratio((net[i] - snet[i]) * (mid[i + hs] - mid[i - 1]), prints[i] - sprints[i]); });
    acc(key("MKpv_all", h), 1, m_end, [&](std::size_t i, Mom& m) { m.ratio(sv[i] * (mid[i + hs] - mid[i - 1]), vol[i]); });
  }
  // (d) Price on flow: 10 s windows (non-overlapping); 30 s delta vs the next 30 s.
  for (std::size_t i = s10; i < n; i += s10) {
    double dall = 0, dsn = 0;
    for (std::size_t k = i - s10; k < i; ++k) {
      dall += sv[k];
      dsn += ssv[k];
    }
    const double dm = mid[i] - mid[i - s10];
    Block& bl = d.blocks[(i - 1) / bsteps];
    bl["R2_all"].add(dall, dm);
    bl["R2_oth"].add(dall - dsn, dm);
    bl["R2_snp"].add(dsn, dm);
    bl["impact_share"].add(imp[i] - imp[i - s10], val[i] - val[i - s10]);
  }
  for (std::size_t i = s30; i + s30 < n; i += s30) {
    double dl = 0;
    for (std::size_t k = i - s30; k < i; ++k) dl += sv[k];
    d.blocks[(i - 1) / bsteps]["DP"].add(dl, mid[i + s30] - mid[i]);
  }
  acc("sniper_print_share", 0, n, [&](std::size_t i, Mom& m) { m.ratio(sprints[i], prints[i]); });
  // Stylized facts and health (per seed).
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
  return d;
}

// ---- (e, f) rules on ScalpEnv ------------------------------------------------
using Policy = std::function<Action(const ScalpObs&, const ScalpEnv&)>;
struct Strategy {
  std::string name;
  Policy policy;
  bool separate_streams = false;  // clairvoyants: the future seen in a copy of the market
  std::uint32_t latency_steps = 1;
  int kind = 0;  // 0 rule, 1 flat, 2 gap oracle, 3 reference oracle (latency 0), 4 clairvoyant
  double theta = 0;
};

std::vector<Strategy> strategies() {
  auto hold = [](const ScalpObs& o) { return static_cast<Action>(o.position + 1); };
  auto signal_rule = [hold](std::size_t k, double th, bool reverse) -> Policy {
    return [=](const ScalpObs& o, const ScalpEnv&) -> Action {
      const double x = reverse ? -o.f[k] : o.f[k];
      return x > th ? 2 : x < -th ? 0 : hold(o);
    };
  };
  // Hysteresis fair-value oracle: long while g > theta, short while g < -theta, else keep.
  auto oracle = [hold](double th) -> Policy {
    return [=](const ScalpObs& o, const ScalpEnv& e) -> Action {
      const double g = e.market().fair_value() - e.market().mid();
      return g > th ? 2 : g < -th ? 0 : hold(o);
    };
  };
  // H-second clairvoyant: steps a copy of the market (separate streams: the
  // same news and arrivals) H seconds ahead; long if the mid will be >= 1
  // tick higher, short if >= 1 lower, else keep.
  auto clairvoyant = [hold](double h) -> Policy {
    return [=](const ScalpObs& o, const ScalpEnv& e) -> Action {
      EsSimulator c = e.market();
      const auto steps = static_cast<int>(std::lround(h / c.params().dt));
      for (int i = 0; i < steps; ++i) c.step();
      const double dm = c.mid() - e.market().mid();
      return dm >= 1 ? 2 : dm <= -1 ? 0 : hold(o);
    };
  };
  std::vector<Strategy> v = {{"flat", [](const ScalpObs&, const ScalpEnv&) -> Action { return 1; }, false, 1, 1, 0},
                             {"delta 30 s > 1.5", signal_rule(1, 1.5, false), false, 1, 0, 0},
                             {"delta 30 s > 2.5", signal_rule(1, 2.5, false), false, 1, 0, 0},
                             {"delta 120 s > 2", signal_rule(2, 2.0, false), false, 1, 0, 0},
                             {"OFI 30 s > 2", signal_rule(4, 2.0, false), false, 1, 0, 0},
                             {"imbalance L1 > 0.6", signal_rule(6, 1.2, false), false, 1, 0, 0},
                             {"imbalance L1-5 > 0.6", signal_rule(7, 1.2, false), false, 1, 0, 0},
                             {"absorption > 2", signal_rule(12, 2.0, false), false, 1, 0, 0},
                             {"fade return 30 s > 2", signal_rule(9, 2.0, true), false, 1, 0, 0},
                             {"delta 5 s > 1.5", signal_rule(0, 1.5, false), false, 1, 0, 0},
                             {"fade delta 5 s > 1.5", signal_rule(0, 1.5, true), false, 1, 0, 0},
                             {"fade OFI 5 s > 1.5", signal_rule(3, 1.5, true), false, 1, 0, 0},
                             {"fade return 5 s > 1", signal_rule(8, 1.0, true), false, 1, 0, 0},
                             {"fade imbalance L1 > 0.6", signal_rule(6, 1.2, true), false, 1, 0, 0}};
  for (double th : {0.5, 0.55, 0.6, 0.7, 0.8, 1.0}) {
    char b[48];
    std::snprintf(b, sizeof b, "oracle gap > %.2f", th);
    v.push_back({b, oracle(th), false, 1, 2, th});
  }
  for (double th : {0.55, 0.7}) {
    char b[48];
    std::snprintf(b, sizeof b, "oracle gap > %.2f, lat 0", th);
    v.push_back({b, oracle(th), false, 0, 3, th});
  }
  for (double h : {1.0, 2.0, 5.0, 30.0}) {
    char b[48];
    std::snprintf(b, sizeof b, "clairvoyant %g s (sep.)", h);
    v.push_back({b, clairvoyant(h), true, 1, 4, h});
  }
  return v;
}

struct RuleRun {
  std::vector<double> net, lots;
};

constexpr double kLotFee = 1.5 / kTickValue;  // ticks per lot per side

RuleRun run_rule(const SimParams& sim, std::uint64_t seed, const Strategy& s, unsigned episodes) {
  ScalpParams p;
  p.sim = sim;
  p.sim.separate_streams = p.sim.separate_streams || s.separate_streams;
  p.latency_steps = s.latency_steps;
  RuleRun rr;
  for (unsigned i = 0; i < episodes; ++i) {
    ScalpEnv env(p);
    rag::agent::Rng r(1'000'000ull * seed + i);  // the simulator-design rule-check sub-block
    auto o = env.reset(r);
    for (;;) {
      const auto st = env.step(s.policy(o, env), r);
      o = st.obs;
      if (st.done) break;
    }
    rr.net.push_back(env.net_pnl_ticks());
    rr.lots.push_back(env.fees_ticks() / kLotFee);  // a +1 -> -1 flip is 2 lots
  }
  return rr;
}

// ---- (e, f) tick rules on TickScalpEnv ----------------------------------------
using TickPolicy = std::function<Action(const TickObs&, const TickScalpEnv&)>;
struct TickStrategy {
  std::string name;
  TickPolicy policy;
  double decide = 0.5;
  bool separate_streams = false;
  int kind = 0;  // 0 rule, 1 no quotes, 2 fair-value quoter (0.5 s), 3 reference (0.1 s), 4 clairvoyant
};

std::vector<TickStrategy> tick_strategies() {
  auto skew = [](std::size_t k, double th) -> TickPolicy {
    return [=](const TickObs& o, const TickScalpEnv&) -> Action { return o.f[k] > th ? 1 : o.f[k] < -th ? 2 : 3; };
  };
  // Passive fade: after a rise quote only the ask, after a fall only the bid.
  auto fade_quote = [](double th) -> TickPolicy {
    return [=](const TickObs& o, const TickScalpEnv&) -> Action { return o.f[8] > th ? 2 : o.f[8] < -th ? 1 : 3; };
  };
  // Fair-value quoter: bid iff g >= -0.5 + d, ask iff g <= 0.5 - d; flatten when the position is against value.
  auto fv_quoter = [](double dl) -> TickPolicy {
    return [=](const TickObs& o, const TickScalpEnv& e) -> Action {
      const double g = e.market().fair_value() - e.market().mid();
      if (o.position * g < -(0.5 - dl)) return 4;
      const bool bid = g >= -0.5 + dl, ask = g <= 0.5 - dl;
      return bid && ask ? 3 : bid ? 1 : ask ? 2 : 0;
    };
  };
  // Clairvoyant quoter: the mid H s ahead (a copy of the market) decides the side.
  auto clair = [](double h) -> TickPolicy {
    return [=](const TickObs& o, const TickScalpEnv& e) -> Action {
      EsSimulator c = e.market();
      const auto steps = static_cast<int>(std::lround(h / c.params().dt));
      for (int i = 0; i < steps; ++i) c.step();
      const double dm = c.mid() - e.market().mid();
      if (o.position * dm <= -1) return 4;
      return dm >= 1 ? 1 : dm <= -1 ? 2 : 3;
    };
  };
  std::vector<TickStrategy> v = {{"no quotes", [](const TickObs&, const TickScalpEnv&) -> Action { return 0; }, 0.5, false, 1},
                                 {"quote both sides", [](const TickObs&, const TickScalpEnv&) -> Action { return 3; }, 0.5, false, 0},
                                 {"both, stop at -2 ticks", [](const TickObs& o, const TickScalpEnv&) -> Action { return o.open_pnl <= -2 ? 4 : 3; }, 0.5, false, 0},
                                 {"skew on imbalance L1 0.3", skew(6, 0.6), 0.5, false, 0},
                                 {"skew on imbalance L1 0.6", skew(6, 1.2), 0.5, false, 0},
                                 {"skew on delta 5 s > 1", skew(0, 1.0), 0.5, false, 0},
                                 {"fade-quote ret 5 s 0.5", fade_quote(0.5), 0.5, false, 0},
                                 {"fade-quote ret 5 s 1", fade_quote(1.0), 0.5, false, 0}};
  for (double dl : {0.0, 0.15, 0.3}) {
    char b[48];
    std::snprintf(b, sizeof b, "fair-value quoter d %.2f", dl);
    v.push_back({b, fv_quoter(dl), 0.5, false, 2});
  }
  v.push_back({"fair-value quoter 0.15, 0.1 s", fv_quoter(0.15), 0.1, false, 3});
  v.push_back({"clairvoyant quoter 2 s (sep.)", clair(2.0), 0.5, true, 4});
  return v;
}

struct TickRun {
  std::vector<double> net, fills, gross;
  double adverse = 0, adverse_n = 0;  // sum of signed (mid 1 s after - fill price), fills
};

TickRun run_tick(const SimParams& sim, std::uint64_t seed, const TickStrategy& s, unsigned episodes) {
  TickParams p;
  p.sim = sim;
  p.sim.separate_streams = p.sim.separate_streams || s.separate_streams;
  p.decide_seconds = s.decide;
  TickRun tr;
  struct Pending {
    double sign, price;
    std::int64_t due;
  };
  std::vector<Pending> pend;
  for (unsigned i = 0; i < episodes; ++i) {
    TickScalpEnv env(p);
    rag::agent::Rng r(1'000'000ull * seed + 500'000 + i);
    auto o = env.reset(r);
    pend.clear();
    for (;;) {
      const auto st = env.step(s.policy(o, env), r);
      o = st.obs;
      const std::int64_t now = env.market().t_ns();
      for (auto it = pend.begin(); it != pend.end();) {
        if (it->due > now) {
          ++it;
          continue;
        }
        tr.adverse += it->sign * (env.market().mid() - it->price);
        tr.adverse_n += 1;
        it = pend.erase(it);
      }
      for (const auto& f : env.step_fills())  // our bid hit (Buy): we bought
        pend.push_back({f.side == Side::Buy ? 1.0 : -1.0, static_cast<double>(f.price), now + 1'000'000'000});
      if (st.done) break;
    }
    tr.net.push_back(env.net_pnl_ticks());
    tr.fills.push_back(env.fills());
    tr.gross.push_back(env.net_pnl_ticks() + env.fees_ticks());
  }
  return tr;
}

// ---- impact: paired TWAP ------------------------------------------------------
// Two markets on the same seed with separate streams; one buys 1 or 4 lots per
// second for 300 s by market orders; the mid difference at the end, and 60 s
// and 300 s later (the larger size).
struct ImpactRun {
  std::vector<double> end_small, end_big, after60, after300;
};

ImpactRun run_impact(const SimParams& sim, std::uint64_t seed, unsigned pairs) {
  SimParams p = sim;
  p.separate_streams = true;
  ImpactRun ir;
  const auto per = [&](double sec) { return static_cast<int>(std::lround(sec / p.dt)); };
  for (unsigned i = 0; i < pairs; ++i) {
    const std::uint64_t s = 1'000'000ull * seed + 700'000 + i;
    for (int big = 0; big < 2; ++big) {
      EsSimulator a(p, s, 3600), b(p, s, 3600);
      for (int k = 0; k < per(150); ++k) {
        a.step();
        b.step();
      }
      const auto lots = static_cast<std::uint32_t>(big ? 4 : 1);  // 300 or 1,200 lots in all
      for (int k = 0; k < per(300); ++k) {
        if (k % per(1) == 0) a.market_order(Side::Buy, lots);
        a.step();
        b.step();
      }
      (big ? ir.end_big : ir.end_small).push_back(a.mid() - b.mid());
      if (!big) continue;
      for (int k = 0; k < per(300); ++k) {
        a.step();
        b.step();
        if (k + 1 == per(60)) ir.after60.push_back(a.mid() - b.mid());
      }
      ir.after300.push_back(a.mid() - b.mid());
    }
  }
  return ir;
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

double mean(const std::vector<double>& v) {
  if (v.empty()) return std::nan("");
  double m = 0;
  for (double x : v) m += x / static_cast<double>(v.size());
  return m;
}
double sd(const std::vector<double>& v) {
  if (v.size() < 2) return 0;
  const double m = mean(v);
  double s = 0;
  for (double x : v) s += (x - m) * (x - m);
  return std::sqrt(s / static_cast<double>(v.size() - 1));
}

// One market's results: blocks of every seed, per-seed rows.
struct Market {
  std::vector<Block> blocks;
  std::vector<Block> seed_total;
  Block total;
  std::vector<std::map<std::pair<std::string, double>, double>> rows;  // per seed
  [[nodiscard]] double row(std::size_t s, const std::string& m, double h) const {
    const auto it = rows[s].find({m, h});
    return it == rows[s].end() ? std::nan("") : it->second;
  }
};

struct Value {
  double pooled = 0, se = 0;
  std::vector<double> seeds;
};

Mom find(const Block& b, const std::string& k) {
  const auto it = b.find(k);
  return it == b.end() ? Mom{} : it->second;
}

Value eval(const Market& mk, const Stat& f) {
  Value v;
  v.pooled = f([&](const std::string& k) { return find(mk.total, k); });
  std::vector<double> jk;
  for (const Block& b : mk.blocks) jk.push_back(f([&](const std::string& k) { return find(mk.total, k).minus(find(b, k)); }));
  const double nb = static_cast<double>(jk.size());
  if (jk.size() > 1) {
    const double m = mean(jk);
    double s = 0;
    for (double x : jk) s += (x - m) * (x - m);
    v.se = std::sqrt((nb - 1) / nb * s);
  }
  for (const Block& b : mk.seed_total) v.seeds.push_back(f([&](const std::string& k) { return find(b, k); }));
  return v;
}

Stat corr_of(const std::string& k) {
  return [k](const Get& g) { return g(k).corr(); };
}
Stat rat_of(const std::string& k) {
  return [k](const Get& g) { return g(k).rat(); };
}

// A bound on a pooled value: lo <= x <= hi; tol for UNRESOLVED.
struct Check {
  double lo = -1e300, hi = 1e300, tol = 0;
  const char* text = nullptr;
};
std::string verdict(const Value& v, const Check& c) {
  const double x = v.pooled;
  std::string s;
  if (x >= c.lo && x <= c.hi) s = "PASS";
  else if (x < c.lo ? x + 2 * v.se < c.lo : x - 2 * v.se > c.hi) s = "FAIL";
  else s = "fail(<2SE)";
  if (c.tol > 0 && 2 * v.se > c.tol) s += " UNRESOLVED";
  return s;
}

void stat_line(const Market& mk, const std::string& label, const Stat& f, const Check& c = {}, const char* seed_text = nullptr,
               const std::function<bool(double)>& seed_ok = {}) {
  const Value v = eval(mk, f);
  double lo = 1e300, hi = -1e300;
  bool all = true;
  for (double x : v.seeds) {
    lo = std::min(lo, x);
    hi = std::max(hi, x);
    if (seed_ok && !seed_ok(x)) all = false;
  }
  say("  %-46s %+8.3f +- %.3f  seeds %+8.3f +- %.3f [%+7.3f, %+7.3f]", label.c_str(), v.pooled, v.se, mean(v.seeds),
      sd(v.seeds) / std::sqrt(static_cast<double>(v.seeds.size())), lo, hi);
  if (c.text) say("  %-24s %s", c.text, verdict(v, c).c_str());
  if (seed_text) say("  seeds %s %s", seed_text, all ? "PASS" : "FAIL");
  say("\n");
}

// Per-seed scalar rows: seed mean +- se [min, max], PASS if every seed is in range.
void row_line(const Market& mk, const std::string& label, const std::function<double(std::size_t)>& value, const char* crit = nullptr,
              const std::function<bool(double)>& ok = {}) {
  std::vector<double> xs;
  for (std::size_t s = 0; s < mk.rows.size(); ++s) xs.push_back(value(s));
  double lo = 1e300, hi = -1e300;
  bool all = true;
  for (double x : xs) {
    lo = std::min(lo, x);
    hi = std::max(hi, x);
    if (ok && !ok(x)) all = false;
  }
  say("  %-46s %+8.3f          seeds +- %.3f [%+7.3f, %+7.3f]", label.c_str(), mean(xs), sd(xs) / std::sqrt(static_cast<double>(xs.size())), lo, hi);
  if (crit) say("  every seed %-14s %s", crit, all ? "PASS" : "FAIL");
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
    else if (k == "--tick") o.tick = static_cast<unsigned>(std::max(0, std::atoi(v.c_str())));
    else if (k == "--impact") o.impact = static_cast<unsigned>(std::max(0, std::atoi(v.c_str())));
    else if (k == "--theta") {
      for (const auto& kv : split(v)) {
        const std::size_t eq = kv.find('=');
        if (eq == std::string::npos) o.theta[""] = std::atof(kv.c_str());
        else o.theta[kv.substr(0, eq)] = std::atof(kv.substr(eq + 1).c_str());
      }
    }
    else if (k == "--trace") o.trace = std::atof(v.c_str());
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
  for (const auto& [m, t] : o.theta)
    if (!m.empty() && std::find(o.markets.begin(), o.markets.end(), m) == o.markets.end()) {
      std::fprintf(stderr, "--theta names market %s, which is not in --markets\n", m.c_str());
      return 2;
    }
  std::vector<SimParams> params;
  try {
    for (const auto& m : o.markets) params.push_back(market(o, m));
  } catch (const std::exception& e) {
    std::fprintf(stderr, "%s\n", e.what());
    return 2;
  }
  const auto strat = strategies();
  const auto tstrat = tick_strategies();
  for (const auto& [m, t] : o.theta)  // a pre-registered theta* must be one of the gap oracles
    if (std::none_of(strat.begin(), strat.end(), [t](const auto& s) { return s.kind == 2 && std::abs(s.theta - t) < 1e-9; })) {
      std::fprintf(stderr, "--theta %g%s%s is not on the gap-oracle grid\n", t, m.empty() ? "" : " for ", m.c_str());
      return 2;
    }
  std::FILE* csv_file = nullptr;
  if ((!o.out.empty() && !(g_out = std::fopen(o.out.c_str(), "w"))) || (!o.csv.empty() && !(csv_file = std::fopen(o.csv.c_str(), "w")))) {
    std::fprintf(stderr, "can't open %s for writing\n", g_out || o.out.empty() ? o.csv.c_str() : o.out.c_str());
    return 2;
  }
  // Tasks: a diagnosis per (market, seed); a rule / tick-rule run per (market, seed, rule); an impact run per (market, seed).
  const std::size_t nm = o.markets.size(), ns = o.seeds.size(), nr = o.rules > 0 ? strat.size() : 0, nt = o.tick > 0 ? tstrat.size() : 0,
                    ni = o.impact > 0 ? 1 : 0;
  std::vector<Diag> diag(nm * ns);
  std::vector<RuleRun> rule(nm * ns * nr);
  std::vector<TickRun> tick(nm * ns * nt);
  std::vector<ImpactRun> impact(nm * ns * ni);
  const std::size_t t1 = nm * ns, t2 = t1 + nm * ns * nr, t3 = t2 + nm * ns * nt, ntasks = t3 + nm * ns * ni;
  std::atomic<std::size_t> next{0};
  auto worker = [&] {
    for (std::size_t t; (t = next.fetch_add(1)) < ntasks;) {
      if (t < t1) {
        diag[t] = diagnose(params[t / ns], o.seeds[t % ns], o);
      } else if (t < t2) {
        const std::size_t u = t - t1, ms = u / nr;
        rule[u] = run_rule(params[ms / ns], o.seeds[ms % ns], strat[u % nr], o.rules);
      } else if (t < t3) {
        const std::size_t u = t - t2, ms = u / nt;
        tick[u] = run_tick(params[ms / ns], o.seeds[ms % ns], tstrat[u % nt], o.tick);
      } else {
        const std::size_t u = t - t3;
        impact[u] = run_impact(params[u / ns], o.seeds[u % ns], o.impact);
      }
    }
  };
  std::vector<std::thread> pool;
  for (unsigned i = 0; i < std::min<std::size_t>(o.threads, ntasks); ++i) pool.emplace_back(worker);
  for (auto& th : pool) th.join();

  // Per market: pool the blocks.
  std::vector<Market> mks(nm);
  for (std::size_t mi = 0; mi < nm; ++mi) {
    Market& mk = mks[mi];
    mk.rows.resize(ns);
    mk.seed_total.resize(ns);
    for (std::size_t si = 0; si < ns; ++si) {
      const Diag& d = diag[mi * ns + si];
      for (const auto& r : d.rows) mk.rows[si][{r.metric, r.horizon}] = r.value;
      for (const Block& b : d.blocks) {
        mk.blocks.push_back(b);
        for (const auto& [k, m] : b) {
          mk.seed_total[si][k] += m;
          mk.total[k] += m;
        }
      }
    }
  }
  if (std::FILE* csv = csv_file) {
    std::fprintf(csv, "market,seed,metric,horizon_s,value,se_jackknife\n");
    for (std::size_t mi = 0; mi < nm; ++mi) {
      const Market& mk = mks[mi];
      for (std::size_t si = 0; si < ns; ++si)
        for (const auto& [k, v] : mk.rows[si])
          std::fprintf(csv, "%s,%llu,%s,%g,%.6g,\n", o.markets[mi].c_str(), static_cast<unsigned long long>(o.seeds[si]), k.first.c_str(), k.second, v);
      for (const auto& [k, m] : mk.total) {  // moments: correlations (ratio markouts as ratio:)
        const bool r = m.den != 0;
        const Value v = eval(mk, r ? rat_of(k) : corr_of(k));
        for (std::size_t si = 0; si < ns; ++si)
          std::fprintf(csv, "%s,%llu,%s:%s,0,%.6g,\n", o.markets[mi].c_str(), static_cast<unsigned long long>(o.seeds[si]), r ? "ratio" : "corr", k.c_str(),
                       v.seeds[si]);
        std::fprintf(csv, "%s,all,%s:%s,0,%.6g,%.6g\n", o.markets[mi].c_str(), r ? "ratio" : "corr", k.c_str(), v.pooled, v.se);
      }
    }
    std::fclose(csv);
  }

  say("OF10 price-discovery diagnostics: seeds");
  for (auto s : o.seeds) say(" %llu", static_cast<unsigned long long>(s));
  say(", %.0f s each after a %.0f s warm-up (session start 3600 s)\n", o.seconds, o.warmup);
  say("Columns: pooled value +- jackknife SE (5-minute blocks); seed mean +- sd/sqrt(n) [min, max]; pooled check; per-seed check.\n");
  if (!o.set.empty()) {  // tuned runs must say so
    say("--set overrides of the discovery preset:");
    for (const auto& [k, x] : o.set) say(" %s=%g", k.c_str(), x);
    say("\n");
  }
  double default_l1 = std::nan("");
  for (std::size_t mi = 0; mi < nm; ++mi)
    if (o.markets[mi] == "default") {
      std::vector<double> l;
      for (std::size_t s = 0; s < ns; ++s) l.push_back(mks[mi].row(s, "fact_mean_l1_size", 0));
      default_l1 = mean(l);
    }
  for (std::size_t mi = 0; mi < nm; ++mi) {
    const SimParams& p = params[mi];
    const Market& mk = mks[mi];
    auto rv = [&mk](const std::string& m, double h) { return [&mk, m, h](std::size_t s) { return mk.row(s, m, h); }; };
    say("\n=== market %s (reprice_band %.2f, reprice_snipe %.2f, reprice_cancel %.2f, reprice_take %.2f, value_impact %.4f,\n"
        "    impact_forecast_tau %.1f, impact_forecast_weight %.2f, maker_info %.2f, informed_meta_rate %.2f, uninformed_meta_rate %.2f,\n"
        "    momentum_rate %.2f, noise_rate %.2f, cancel_rate %.2f, fade %.2f, activity_excite %.2f, value_vol %.2f)\n",
        o.markets[mi].c_str(), p.reprice_band, p.reprice_snipe, p.reprice_cancel, p.reprice_take, p.value_impact, p.impact_forecast_tau,
        p.impact_forecast_weight, p.maker_info, p.informed_meta_rate, p.uninformed_meta_rate, p.momentum_rate, p.noise_rate, p.cancel_rate, p.fade,
        p.activity_excite, p.value_vol);
    say("(a) hidden gap g = fair value - mid (ideal: the 1-tick grid floor(value) + 0.5 on the same value path)\n");
    stat_line(mk, "sd(g), ticks", [](const Get& g) { return std::sqrt(g("gap").var_x()); }, {-1e300, 0.40, 0, "<= 0.40"});
    stat_line(mk, "ideal sd(g)", [](const Get& g) { return std::sqrt(g("ideal_gap").var_x()); });
    stat_line(mk, "corr(g, r 30 s)", corr_of("C_gap@30"), {-1e300, 0.12, 0, "<= 0.12 (or floor+0.02)"});
    stat_line(mk, "ideal floor corr(V - ref, ref 30 s)", corr_of("ideal@30"));
    stat_line(mk, "corr(g, r 30 s) - ideal floor", [](const Get& g) { return g("C_gap@30").corr() - g("ideal@30").corr(); }, {-1e300, 0.02, 0.02, "<= 0.02"},
              "<= 0.03", [](double x) { return x <= 0.03; });
    stat_line(mk, "tail corr(g, mid(t+30) - mid(t+2 s))", corr_of("T2_gap"));
    stat_line(mk, "ideal tail (2 s -> 30 s)", corr_of("ideal_t2"));
    stat_line(mk, "tail2(g) - ideal tail2", [](const Get& g) { return g("T2_gap").corr() - g("ideal_t2").corr(); }, {-1e300, 0.02, 0.02, "<= 0.02"});
    stat_line(mk, "slope B_g(2 s) / B_g(30 s)", [](const Get& g) { return g("C_gap@2").slope() / g("C_gap@30").slope(); }, {0.8, 1e300, 0, ">= 0.8"});
    say("  corr(g, r h) by horizon h (s), pooled:");
    for (double h : kHorizons) say(" %g:%+.3f", h, eval(mk, corr_of(key("C_gap", h))).pooled);
    say("\n");
    stat_line(mk, "gap autocorrelation 1 s", corr_of("gapac@1"));
    stat_line(mk, "ideal gap autocorrelation 1 s", corr_of("ideal_gapac@1"));
    stat_line(mk, "gap autocorrelation 10 s", corr_of("gapac@10"));
    row_line(mk, "share of steps with |g| <= 1", rv("gap_share_within_1", 0));
    row_line(mk, "max |g|", rv("gap_max_abs", 0));
    row_line(mk, "max |g| at, s after the warm-up", rv("gap_max_at_s", 0));
    row_line(mk, "value jumps > 2 ticks (count)", rv("jumps", 0));
    row_line(mk, "jump catch-up (|g| <= 1) median, s", rv("jump_catchup_median_s", 0), "<= 1", [](double x) { return !(x > 1); });
    row_line(mk, "jump catch-up p95, s", rv("jump_catchup_p95_s", 0), "<= 3", [](double x) { return !(x > 3); });
    row_line(mk, "races (|g| > band) per minute", rv("races_per_min", 0));
    row_line(mk, "share of races ending in a tick", rv("race_tick_share", 0));
    row_line(mk, "stale episodes (simulator) per minute", rv("sim_races_per_min", 0));
    row_line(mk, "share of time in a race", rv("race_time_share", 0));
    row_line(mk, "race to the tick, median s (0.2 s = floor)", rv("race_median_s", 0), "<= 1.0", [](double x) { return std::isnan(x) || x <= 1.0; });
    row_line(mk, "share of races ticked at the 0.2 s floor", rv("race_floor_share", 0));
    row_line(mk, "race to the tick, p95 s", rv("race_p95_s", 0));
    row_line(mk, "stale lots: cancelled share (rest sniped)", rv("stale_lots_cancel_share", 0));
    row_line(mk, "mid changes per minute", rv("mid_changes_per_min", 0));
    row_line(mk, "ideal-grid mid changes per minute", rv("ideal_changes_per_min", 0));
    row_line(mk, "re-pricing ticks per minute", rv("reprices_per_min", 0));
    row_line(mk, "re-pricing jumps per mid change", [&mk](std::size_t s) { return mk.row(s, "reprices_per_min", 0) / mk.row(s, "mid_changes_per_min", 0); });
    row_line(mk, "after a mid change: drift 1 s", rv("after_move_drift", 1));
    row_line(mk, "after a mid change: drift 30 s", rv("after_move_drift", 30));
    row_line(mk, "share of mid changes reversing the last one", rv("next_change_reverses", 0));
    row_line(mk, "ideal grid: share reversing", rv("ideal_next_change_reverses", 0));
    say("(b) book imbalance and features\n");
    for (const char* name : {"I1", "I5"}) {
      const std::string s = name;
      const bool l1 = s == "I1";
      stat_line(mk, "event corr(" + s + ", next mid change)", corr_of("E_" + s + "@1"),
                {l1 ? 0.20 : 0.15, 0.6, 0.05, l1 ? "0.20 .. 0.6" : "0.15 .. 0.6"}, l1 ? ">= 0.15" : nullptr, [](double x) { return x >= 0.15; });
      say("  event corr(%s, mid j changes later), pooled:", name);
      for (std::size_t j : kEvents) say(" j=%zu:%+.3f", j, eval(mk, corr_of(key("E_" + s, static_cast<double>(j)))).pooled);
      say("\n");
      stat_line(mk, "event corr(" + s + ", mid(c5) - mid(c1))", corr_of("EI15_" + s), {-0.03, 0.03, 0.03, "|.| <= 0.03"});
      stat_line(mk, "event corr(" + s + ", mid(c10) - mid(c2))", corr_of("EI210_" + s), {-0.03, 0.03, 0.03, "|.| <= 0.03"});
      stat_line(mk, "corr(" + s + ", r 1 s)", corr_of("C_" + s + "@1"), l1 ? Check{0.15, 1e300, 0, ">= 0.15"} : Check{});
      stat_line(mk, "corr(" + s + ", r 30 s)", corr_of("C_" + s + "@30"), {-0.05, 0.05, 0.05, "|.| <= 0.05"});
      stat_line(mk, "tail corr(" + s + ", mid(t+30) - mid(t+2 s))", corr_of("T2_" + s), {-0.03, 0.03, 0.03, "|.| <= 0.03"});
      stat_line(mk, "leak corr(" + s + ", g)", corr_of("L_" + s), {-1e300, 0.5, 0, "<= 0.5"});
    }
    stat_line(mk, "event corr(I1, next change) when |g| <= band", corr_of("Ecalm_I1"));
    stat_line(mk, "event corr(g, next mid change)", corr_of("E_gap@1"));
    say("  %-10s %8s %8s %8s %8s %8s %8s   flag: |tail2| > 0.03 by > 2 SE\n", "feature", "C 1 s", "C 30 s", "tail1", "tail2", "+-SE", "leak");
    for (std::size_t k = 0; k < kFeatures; ++k) {
      const std::string f = std::string("f_") + kFeatureNames[k];
      const Value tl2 = eval(mk, corr_of("T2_" + f));
      say("  %-10s %+8.3f %+8.3f %+8.3f %+8.3f %8.3f %+8.3f   %s\n", kFeatureNames[k], eval(mk, corr_of("C_" + f + "@1")).pooled,
          eval(mk, corr_of("C_" + f + "@30")).pooled, eval(mk, corr_of("T1_" + f)).pooled, tl2.pooled, tl2.se, eval(mk, corr_of("L_" + f)).pooled,
          std::abs(tl2.pooled) - 2 * tl2.se > 0.03 ? "FLAG" : std::abs(tl2.pooled) > 0.03 ? "flag(<2SE)" : "ok");
    }
    stat_line(mk, "30 s return autocorrelation (overlapping)", corr_of("r30ac"), {-0.03, 0.03, 0.03, "|.| <= 0.03"});
    stat_line(mk, "ideal grid: 30 s return autocorrelation", corr_of("ideal_r30ac"));
    stat_line(mk, "value change 10 s vs next 10 s (martingale: 0)", corr_of("valac10"));
    stat_line(mk, "flow impact 10 s vs next 10 s", corr_of("impac10"));
    stat_line(mk, "corr(delta 30 s, next 30 s), overlapping", corr_of("delta30_next30"));
    stat_line(mk, "corr(OFI 30 s, next 30 s), overlapping", corr_of("ofi30_next30"));
    say("(c) markouts after aggressive prints, ticks\n");
    say("  %-34s", "horizon (s)");
    for (double h : kMarkouts) say(" %7g", h);
    say("\n");
    const std::vector<std::pair<const char*, const char*>> mks_rows = {{"MKp_all", "prevailing mid, per print"},
                                                                        {"MKp_oth", "prevailing, non-sniper prints"},
                                                                        {"MKp_snp", "prevailing, sniper prints"},
                                                                        {"MKpv_all", "prevailing, per lot"},
                                                                        {"MK", "StylizedMeter (end of bucket)"}};
    for (const auto& [k, label] : mks_rows) {
      say("  %-34s", label);
      for (double h : kMarkouts) say(" %+7.3f", eval(mk, rat_of(key(k, h))).pooled);
      say("\n");
    }
    stat_line(mk, "prevailing markout 1 s, all prints", rat_of("MKp_all@1"), {0, 1e300, 0, "> 0 by 2 SE"});
    stat_line(mk, "prevailing markout 1 s, non-sniper prints", rat_of("MKp_oth@1"), {0, 1e300, 0, "> 0 by 2 SE"});
    stat_line(mk, "prevailing markout 5 s - 1 s", [](const Get& g) { return g("MKp_all@5").rat() - g("MKp_all@1").rat(); }, {0, 1e300, 0, ">= -SE"});
    stat_line(mk, "prevailing markout 30 s - 5 s", [](const Get& g) { return g("MKp_all@30").rat() - g("MKp_all@5").rat(); },
              {-0.10, 0.10, 0.20, "|.| <= 0.10"});
    stat_line(mk, "prevailing markout 60 s - 5 s", [](const Get& g) { return g("MKp_all@60").rat() - g("MKp_all@5").rat(); },
              {-0.15, 0.15, 0.30, "|.| <= 0.15"});
    stat_line(mk, "StylizedMeter markout 1 s", rat_of("MK@1"), {-0.05, 1e300, 0, ">= -0.05"});
    say("(d) stylized facts and health\n");
    struct Rng2 {
      const char* key;
      double lo, hi;
      const char* crit;
    };
    const std::vector<Rng2> ranges = {{"trades_per_sec", 15, 22, "15 .. 22"}, {"mean_trade_size", 3.4, 4.4, "3.4 .. 4.4"},
                                      {"volume_per_min", -1e9, 1e9, nullptr}, {"spread1_share", 0.95, 1.01, ">= 0.95"},
                                      {"mean_l1_size", -1e9, 190, "<= 190"},  {"ret1m_sd", 3.0, 4.5, "3.0 .. 4.5"},
                                      {"ret1m_kurtosis", 4, 12, "4 .. 12"},   {"ret1m_ac1", -0.1, 0.1, "|.| <= 0.1"},
                                      {"absret1m_ac1", 0.1, 1, ">= 0.1"},     {"sign_ac1", 0.2, 1, ">= 0.2"},
                                      {"sign_ac10", -1e9, 1e9, nullptr},      {"sign_ac100", 0.05, 1, ">= 0.05"},
                                      {"impact_r2", -1e9, 1e9, nullptr},      {"delta_predict", -1e9, 1e9, nullptr},
                                      {"markout_1s", -1e9, 1e9, nullptr},     {"markout_5s", -1e9, 1e9, nullptr},
                                      {"markout_30s", -1e9, 1e9, nullptr},    {"mid_changes_per_min", -1e9, 1e9, nullptr}};
    for (const auto& r : ranges) {
      const std::string k = std::string("fact_") + r.key;
      row_line(mk, r.key, rv(k, 0), r.crit, [r](double x) { return x >= r.lo && x <= r.hi; });
    }
    {
      std::vector<double> k4;
      for (std::size_t s = 0; s < ns; ++s) k4.push_back(mk.row(s, "fact_ret1m_kurtosis", 0));
      say("  %-46s %+8.3f  seed mean in 5 .. 10: %s\n", "excess kurtosis", mean(k4), mean(k4) >= 5 && mean(k4) <= 10 ? "PASS" : "FAIL");
    }
    if (!std::isnan(default_l1))
      row_line(mk, "mean L1 size / default's", [&mk, default_l1](std::size_t s) { return mk.row(s, "fact_mean_l1_size", 0) / default_l1; }, "<= 1.5",
               [](double x) { return x <= 1.5; });
    stat_line(mk, "R^2(10 s mid change ~ 10 s delta), all prints", [](const Get& g) { return std::pow(g("R2_all").corr(), 2); }, {0.3, 1e300, 0, ">= 0.30"});
    stat_line(mk, "R^2 on non-sniper delta (incl. informed)", [](const Get& g) { return std::pow(g("R2_oth").corr(), 2); }, {0.25, 1e300, 0, ">= 0.25"});
    stat_line(mk, "R^2 on sniper delta", [](const Get& g) { return std::pow(g("R2_snp").corr(), 2); });
    stat_line(mk, "delta predictability (30 s delta, next 30 s)", corr_of("DP"), {-0.04, 0.04, 0.04, "|.| <= 0.04"});
    stat_line(mk, "flow-impact share of 10 s value change (slope)", [](const Get& g) {
      const Mom m = g("impact_share");
      return m.var_y() > 0 ? m.cov() / m.var_y() : 0;
    });
    stat_line(mk, "sniper share of prints", rat_of("sniper_print_share"));
    row_line(mk, "volume share: sniper", rv("volume_share_sniper", 0), "<= 0.15", [](double x) { return x <= 0.15; });
    for (const char* k : {"noise", "informed_meta", "uninformed_meta", "momentum"})
      row_line(mk, std::string("volume share: ") + k, rv(std::string("volume_share_") + k, 0));
    row_line(mk, "activity mean", rv("activity_mean", 0));
    row_line(mk, "share of time at the activity clamp", rv("activity_clamped_share", 0), "< 0.05", [](double x) { return x < 0.05; });
    row_line(mk, "informed meta-order starts per minute", rv("informed_starts_per_min", 0));
    row_line(mk, "uninformed meta-order starts per minute", rv("uninformed_starts_per_min", 0));

    if (nr > 0) {
      say("(e, f) rules on ScalpEnv: %u episodes x %zu seeds (episode seeds 1,000,000 * seed + i), net ticks/episode after costs\n", o.rules, ns);
      say("  lots: fees / $1.50 per lot (a flip is 2 lots); gross/lot before fees; edge/lot = gross/lot + 0.5 (the half spread: ~0 without\n"
          "  information); SEs: episodes (net), episode jackknife (edge). Clairvoyants run on the same preset with separate streams.\n");
      say("  %-30s %8s %6s %6s %8s %8s %7s   per-seed means\n", "strategy", "net/ep", "t", "lots", "gross/lt", "edge/lt", "+-SE");
      struct Res {
        double net, se, lots, edge;
        int kind;
        double theta;
      };
      std::vector<Res> res(nr);
      for (std::size_t ri = 0; ri < nr; ++ri) {
        std::vector<double> all, lots, seed_means;
        for (std::size_t si = 0; si < ns; ++si) {
          const auto& rr = rule[(mi * ns + si) * nr + ri];
          all.insert(all.end(), rr.net.begin(), rr.net.end());
          lots.insert(lots.end(), rr.lots.begin(), rr.lots.end());
          seed_means.push_back(mean(rr.net));
        }
        const double ne = static_cast<double>(all.size());
        double sn = 0, sl = 0;
        for (std::size_t i = 0; i < all.size(); ++i) {
          sn += all[i];
          sl += lots[i];
        }
        auto edge_of = [](double n_, double l_) { return l_ > 0 ? (n_ + l_ * kLotFee) / l_ + 0.5 : std::nan(""); };  // gross = net + fees
        std::vector<double> jk;
        for (std::size_t i = 0; i < all.size(); ++i) jk.push_back(edge_of(sn - all[i], sl - lots[i]));
        double esd = 0;
        if (sl > 0 && jk.size() > 1) {
          const double m = mean(jk);
          for (double x : jk) esd += (x - m) * (x - m);
          esd = std::sqrt((ne - 1) / ne * esd);
        }
        const double m = sn / ne, se = sd(all) / std::sqrt(ne), edge = edge_of(sn, sl);
        res[ri] = {m, se, sl / ne, edge, strat[ri].kind, strat[ri].theta};
        say("  %-30s %+8.2f %+6.2f %6.1f %+8.3f %+8.3f %7.3f  ", strat[ri].name.c_str(), m, se > 0 ? m / se : 0.0, sl / ne, edge - 0.5, edge, esd);
        for (double x : seed_means) say(" %+6.2f", x);
        if (strat[ri].kind == 0)
          say("   %s; |edge| %s", m > 0 && se > 0 && m / se > 2 ? "FAIL (wins)" : "ok",
              std::isnan(edge) || std::abs(edge) <= 0.15 ? "<= 0.15" : std::abs(edge) - 2 * esd > 0.15 ? "> 0.15 FLAG" : "> 0.15 (<2SE)");
        if (strat[ri].kind == 2 && sl / ne < 1) say("   vacuous (< 1 lot/episode)");
        say("\n");
      }
      // theta*: pre-registered for this market (--theta) or this run's best non-vacuous gap oracle.
      const auto th = o.theta.count(o.markets[mi]) > 0 ? o.theta.find(o.markets[mi]) : o.theta.find("");
      const double theta = th == o.theta.end() ? -1 : th->second;
      std::size_t best = nr;
      for (std::size_t ri = 0; ri < nr; ++ri)
        if (res[ri].kind == 2 && res[ri].lots >= 1) {
          if (theta >= 0 ? std::abs(res[ri].theta - theta) < 1e-9 : best == nr || res[ri].net > res[best].net) best = ri;
        }
      if (best < nr) {
        say("  theta* = %.2f (%s): oracle net %+.2f +- %.2f, %.1f lots/episode, edge/lot %+.3f\n", res[best].theta,
            theta >= 0 ? "pre-registered" : "best of this run: pre-register it for held-out runs", res[best].net, res[best].se, res[best].lots,
            res[best].edge);
        bool flag = false;
        for (std::size_t ri = 0; ri < nr; ++ri)
          if (res[ri].kind == 0 && res[ri].net > res[best].net + 2 * res[ri].se) {
            say("  RED FLAG: %s beats the oracle at theta* by > 2 SE\n", strat[ri].name.c_str());
            flag = true;
          }
        say("  (f) every rule <= oracle(theta*) + 2 SE: %s\n", flag ? "FAIL" : "PASS");
      } else {
        say("  (f) %s is vacuous (< 1 lot/episode): the bound is empty; see the clairvoyants\n",
            theta >= 0 ? "the gap oracle at the pre-registered theta*" : "every gap oracle");
      }
    }
    if (nt > 0) {
      say("(e, f) tick rules on TickScalpEnv: %u episodes x %zu seeds (episode seeds 1,000,000 * seed + 500,000 + i), net ticks/episode\n", o.tick, ns);
      say("  adverse: mean signed (mid ~1 s after the fill's decision step - fill price) per passive fill\n"
          "  (+0.5 at a 1-tick spread and an unchanged mid: below +0.5 is adverse selection, < 0: picked off past the half spread)\n");
      say("  %-30s %8s %6s %6s %9s %8s   per-seed means\n", "strategy", "net/ep", "t", "fills", "gross/fl", "adverse");
      double oracle_mm = -1e300;
      std::vector<std::pair<std::string, double>> rules_net;
      for (std::size_t ri = 0; ri < nt; ++ri) {
        std::vector<double> all, seed_means;
        double fills = 0, gross = 0, adv = 0, advn = 0;
        for (std::size_t si = 0; si < ns; ++si) {
          const auto& tr = tick[(mi * ns + si) * nt + ri];
          all.insert(all.end(), tr.net.begin(), tr.net.end());
          seed_means.push_back(mean(tr.net));
          for (double x : tr.fills) fills += x;
          for (double x : tr.gross) gross += x;
          adv += tr.adverse;
          advn += tr.adverse_n;
        }
        const double ne = static_cast<double>(all.size()), m = mean(all), se = sd(all) / std::sqrt(ne);
        say("  %-30s %+8.2f %+6.2f %6.1f %+9.3f %+8.3f  ", tstrat[ri].name.c_str(), m, se > 0 ? m / se : 0.0, fills / ne, fills > 0 ? gross / fills : 0.0,
            advn > 0 ? adv / advn : 0.0);
        for (double x : seed_means) say(" %+6.2f", x);
        if (tstrat[ri].kind == 0) say("   %s", m > 0.5 && se > 0 && m / se > 2 ? "FAIL (naive quoting wins)" : "ok");
        if (tstrat[ri].name == "quote both sides") say(", %s", m >= -4 && m <= 0 ? "in [-4, 0] PASS" : "outside [-4, 0] FAIL");
        say("\n");
        if (tstrat[ri].kind == 2) oracle_mm = std::max(oracle_mm, m);
        if (tstrat[ri].kind == 0) rules_net.push_back({tstrat[ri].name, m});
      }
      bool flag = false;
      for (const auto& [nm2, x] : rules_net)
        if (x > oracle_mm) {
          say("  RED FLAG: %s beats the 0.5 s fair-value quoter\n", nm2.c_str());
          flag = true;
        }
      say("  (f) best 0.5 s fair-value quoter %+.2f >= every tick rule: %s\n", oracle_mm, flag ? "FAIL" : "PASS");
    }
    if (ni > 0) {
      std::vector<double> es, eb, a60, a300;
      for (std::size_t si = 0; si < ns; ++si) {
        const auto& ir = impact[mi * ns + si];
        es.insert(es.end(), ir.end_small.begin(), ir.end_small.end());
        eb.insert(eb.end(), ir.end_big.begin(), ir.end_big.end());
        a60.insert(a60.end(), ir.after60.begin(), ir.after60.end());
        a300.insert(a300.end(), ir.after300.begin(), ir.after300.end());
      }
      const double sq = std::sqrt(static_cast<double>(es.size()));
      say("(impact) paired TWAP buy over 300 s by market orders (separate streams), %u pairs x %zu seeds, mid difference in ticks\n", o.impact, ns);
      say("  300 lots: %+.2f +- %.2f; 1,200 lots: %+.2f +- %.2f; exponent log(I1200 / I300) / log 4 = %.2f\n", mean(es), sd(es) / sq, mean(eb), sd(eb) / sq,
          mean(es) > 0 && mean(eb) > 0 ? std::log(mean(eb) / mean(es)) / std::log(4.0) : std::nan(""));
      say("  1,200 lots, 60 s after the end: %+.2f +- %.2f; 300 s after: %+.2f +- %.2f (remaining share %.2f)\n", mean(a60), sd(a60) / sq, mean(a300),
          sd(a300) / sq, mean(eb) != 0 ? mean(a300) / mean(eb) : std::nan(""));
    }
  }
  if (g_out) std::fclose(g_out);
}
