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
//       non-sniper and sniper prints, market health, activity health (the
//       lower and upper clamp, log activity against its stationary law);
//   (e, f) --rules N: OF2 rules (with fade rules), hysteresis fair-value
//       oracles, the PRE-REGISTERED oracles O1 / O2 (below) and H-second
//       clairvoyants on ScalpEnv, N episodes per seed, episode seeds
//       1,000,000 * seed + i (the simulator-design rule-check sub-block, not a
//       sealed block); --emo N: the expected-mid oracles (O2 and a loose 2 s
//       one) on the first N of those episodes (expensive: ~2M steps each);
//       --tick N: tick rules, the fair-value quoter and a clairvoyant quoter
//       on TickScalpEnv, episode seeds 1,000,000 * seed + 850,000 + i, with
//       quote-both's fills split into stale sweeps (value already past the
//       quote) and flow fills; --impact N: a paired TWAP impact test, N pairs
//       per seed, seeds 1,000,000 * seed + 700,000 + i.
//
// Regimes (iteration 3: RL trains in 10-minute episodes, so the criteria must
// hold there, not only on long windows):
//   --long legacy (default): EsSimulator(preset, seed, start 3600 s), --seconds
//       after a 150 s warm-up, one window per seed (the v1 / v2 tables);
//   --long session: --sessions full sessions per seed (simulator seeds
//       1,000,000 * seed + 900,000 + k, 0 .. 23,400 s, 60 s look-ahead), every
//       step of the block statistics (correlations, slopes, markouts, moments,
//       the AGREE rows) weighted by ScalpEnv's time-of-day coverage w(t) =
//       clamp((t - 150) / L) - clamp((t - 750) / L), L = session - 750: the
//       same expectation as the episode regime under a stationary market.
//       Per-window rows (after-move drift, reversals, races, jump catch-up,
//       volume and clamp shares, fact_*) cover the whole session UNWEIGHTED;
//       jump catch-up quantiles pool the jumps of every window of a seed;
//   --episodes N: N episode windows per seed built exactly as ScalpEnv builds
//       them (ScalpEnv::make_market from Rng(1,000,000 * seed + 800,000 + i):
//       fresh simulator, random time of day), --episode-warmup (150 s), then
//       600 s measured plus a 60 s look-ahead tail used only for forward
//       returns and markouts. Blocks: one per episode.
//   With both --episodes and --long session an AGREE table tests episode -
//   long = 0 for every row: |d| <= max(tolerance floor, 2 SE).
//   --episode-warmup 1050: the same windows after a 15-minute burn (the
//   simulator starts 900 s earlier): ground truth that the stationary draw
//   (activity_warm_start) equals a burned-in market.
//   --scalp-warmup W (default 150 = ScalpEnv's default; OF10 v3 RL reruns on
//   discovery use 1050): the ScalpEnv geometry is ScalpParams with
//   warmup_seconds = W. The episode regime then draws its windows exactly as
//   such a ScalpEnv does (ScalpEnv::make_market / draw_market: warm-up start
//   uniform on [0, session - 600 - W], so windows start >= W s into the
//   session) and burns W s (--episode-warmup defaults to W; it must be >= W),
//   and the session-weighted long regime uses that geometry's time-of-day
//   coverage w(t) (W in place of 150 above). Rules, EMO, tick rules and
//   impact keep ScalpEnv's default 150 s (of2_baselines --warmup gives the
//   rule baselines at W).
//
// Standard errors: every correlation, slope, markout and R^2 is accumulated
// per block (5 minutes on legacy long windows, 1 hour in full sessions: activity
// decorrelates over activity_tau = 5-10 min; one episode in the episode regime);
// "SE" is the leave-one-block-out jackknife over all blocks of all seeds
// (pooled), next to the seed mean +- sd / sqrt(n). A pooled check is PASS
// when the value meets the bound, FAIL when it misses it by more than 2 SE,
// "fail(<2SE)" when it misses by less; "UNRESOLVED" is appended when 2 SE
// exceeds the check's tolerance.
//
// Usage: of_discovery [--markets default,efficient,discovery_v1,discovery]
//          [--seeds 1-3] [--seconds 7200] [--threads 8] [--long legacy|session|off]
//          [--sessions 3] [--episodes 0] [--episode-warmup 150] [--scalp-warmup 150] [--rules 0]
//          [--emo 0] [--tick 0] [--impact 0] [--theta [market=]X,...]
//          [--csv FILE] [--out FILE] [--set name=value,...]
// --set overrides fields of the discovery preset (tuning), e.g.
// --set reprice_cancel=8,reprice_take=0.1 (names: see kTunable; activity_model
// and activity_warm_start take 0 / 1).
// --theta fixes the hysteresis gap-oracle threshold per market (legacy v1 / v2
// tables); a market without one uses the best theta of this run, marked so.
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
#include "rag/orderflow/scalp_run.hpp"
#include "rag/orderflow/stylized.hpp"
#include "rag/orderflow/tick_env.hpp"

using namespace rag::orderflow;
using rag::agent::Action;

namespace {

// (f) PRE-REGISTERED ORACLES (OF10 iteration 3). Fixed here before any v3 RL
// run; the thresholds come from costs, never from results (fee = $1.50 /
// $12.50 = 0.12 tick per lot per side):
//   O1, zero-latency fair-value taker: long while g > 0.62, short while
//       g < -0.62, else keep (0.62 = the half spread + one fee); latency 0, a
//       decision every 0.1 s. Shows that the information exists; NOT an upper
//       bound (it reads only the current gap, and value is not exactly a
//       martingale: a policy can in principle use other drifts).
//   O2, expected-mid oracle EMO(H = 10 s, K = 32, theta = 1.24), at the
//       agent's cadence and latency (decide 1 s, latency 100 ms): K copies of
//       the market, each forked with fresh randomness (EsSimulator::fork:
//       the full hidden state, new future news and arrivals), are stepped H
//       seconds; f = mean(mid_H - mid); long if f > theta, short if
//       f < -theta, else keep (theta = the 1-tick spread + 2 fees: a round
//       trip). A reference, NOT an upper bound: one fixed rule (its P&L moves
//       with H and theta). A learned policy above O2 + 2 SE, compared PAIRED on
//       the same episodes (the first --emo of each seed), triggers a leak
//       audit, as does one above O1; neither proves a bug by itself. The (e)
//       gate stays per rule: no naive rule > 0 by 2 SE.
//   An oracle with < 2 lots/episode is VACUOUS; the 2 s clairvoyant is then
//   the primary bound before any RL run.
constexpr double kO1Theta = 0.62, kO1Decide = 0.1, kO2Horizon = 10, kO2Theta = 1.24, kMinOracleLots = 2;
constexpr int kO2K = 32;

struct Options {
  std::vector<std::string> markets{"default", "efficient", "discovery_v1", "discovery"};
  std::vector<std::uint64_t> seeds{1, 2, 3};
  double seconds = 7200, warmup = 150, trace = -1, episode_warmup = 150, scalp_warmup = 150;  // episode_warmup defaults to scalp_warmup
  std::map<std::string, double> theta;  // pre-registered theta* per market ("" = every market)
  unsigned threads = 8, rules = 0, tick = 0, impact = 0, emo = 0, episodes = 0, sessions = 3;
  std::string long_mode = "legacy";  // legacy | session | off
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
    {"maker_info", &SimParams::maker_info},           {"maker_info_max", &SimParams::maker_info_max},
    {"activity_vol", &SimParams::activity_vol},       {"activity_tau", &SimParams::activity_tau},
    {"activity_min", &SimParams::activity_min},       {"activity_max", &SimParams::activity_max},
    {"activity_news_excite", &SimParams::activity_news_excite}, {"jump_sd", &SimParams::jump_sd},
    {"join_ahead_mean", &SimParams::join_ahead_mean}, {"cancel_back_bias", &SimParams::cancel_back_bias},
    {"maker_size_mean", &SimParams::maker_size_mean}, {"iceberg_rate", &SimParams::iceberg_rate},
    {"taker_size_logsd", &SimParams::taker_size_logsd}, {"momentum_scale", &SimParams::momentum_scale}};

SimParams market(const Options& o, const std::string& name) {
  SimParams p = market_preset(name);
  if (name == "discovery")
    for (const auto& [k, v] : o.set) {
      if (k == "activity_model") {
        p.activity_model = static_cast<int>(std::lround(v));
        continue;
      }
      if (k == "activity_warm_start") {
        p.activity_warm_start = v != 0;
        continue;
      }
      const auto it = kTunable.find(k);
      if (it == kTunable.end()) throw std::invalid_argument("--set: unknown field " + k);
      p.*(it->second) = v;
    }
  EsSimulator check(p);  // invalid combinations throw here, not in a worker thread
  return p;
}

// The weight of the samples Mom records: 1, or a long session's time-of-day
// weight (set per sample by the accumulation loops, reset to 1 after).
thread_local double g_w = 1;

// Additive moments: of (x, y) for correlations and slopes (x's third and
// fourth moments for skew and kurtosis), num / den for ratios.
struct Mom {
  double n = 0, sx = 0, sy = 0, sxx = 0, syy = 0, sxy = 0, num = 0, den = 0, sx3 = 0, sx4 = 0;
  void add(double x, double y) {
    const double w = g_w;
    n += w;
    sx += w * x;
    sy += w * y;
    sxx += w * x * x;
    syy += w * y * y;
    sxy += w * x * y;
    sx3 += w * x * x * x;
    sx4 += w * x * x * x * x;
  }
  void ratio(double a, double b) {
    num += g_w * a;
    den += g_w * b;
  }
  Mom& operator+=(const Mom& o) {
    n += o.n, sx += o.sx, sy += o.sy, sxx += o.sxx, syy += o.syy, sxy += o.sxy, num += o.num, den += o.den, sx3 += o.sx3, sx4 += o.sx4;
    return *this;
  }
  [[nodiscard]] Mom minus(const Mom& o) const {
    Mom m = *this;
    m.n -= o.n, m.sx -= o.sx, m.sy -= o.sy, m.sxx -= o.sxx, m.syy -= o.syy, m.sxy -= o.sxy, m.num -= o.num, m.den -= o.den, m.sx3 -= o.sx3,
        m.sx4 -= o.sx4;
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
  [[nodiscard]] double mean_x() const { return n > 0 ? sx / n : 0; }
  [[nodiscard]] double skew_x() const {
    if (!(n > 0)) return 0;
    const double m = sx / n, e2 = sxx / n, e3 = sx3 / n, v = e2 - m * m;
    return v > 0 ? (e3 - 3 * m * e2 + 2 * m * m * m) / std::pow(v, 1.5) : 0;
  }
  [[nodiscard]] double exkurt_x() const {
    if (!(n > 0)) return 0;
    const double m = sx / n, e2 = sxx / n, e3 = sx3 / n, e4 = sx4 / n, v = e2 - m * m;
    return v > 0 ? (e4 - 4 * m * e3 + 6 * m * m * e2 - 3 * m * m * m * m) / (v * v) - 3 : 0;
  }
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
  std::vector<Block> blocks;    // per block (5 minutes, or one episode)
  Rows rows;                    // scalar per-window values (facts, quantiles, counts)
  std::vector<double> catchup;  // jump catch-up times, s (quantiles pooled over a seed's windows)
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
const std::array<std::size_t, 3> kSignLags = {1, 10, 100};

// One measured window: the simulator runs `warm` steps, then `anchor` steps
// whose samples anchor every statistic, then `tail` steps used only for
// forward returns and markouts. block: steps per jackknife block. coverage:
// weight each step by ScalpEnv's time-of-day coverage (long sessions).
struct WindowSpec {
  double start = 3600;  // session second the simulator started at
  std::size_t warm = 0, anchor = 0, tail = 0, block = 0;
  const ScalpParams* coverage = nullptr;
};

// ScalpEnv's time-of-day coverage of session second t (episode = E s after a
// W s warm-up, start uniform on [0, L]), scaled to 1 in the bulk of the session.
double coverage_weight(const ScalpParams& sp, double t) {
  const double e = sp.episode_seconds, w = sp.warmup_seconds;
  const double l = std::max(1e-9, sp.sim.session_seconds - e - w);
  auto c = [l](double x) { return std::clamp(x / l, 0.0, 1.0); };
  return (c(t - w) - c(t - w - e)) / std::min(1.0, e / l);
}

Diag diagnose(const SimParams& p, EsSimulator sim, std::uint64_t seed, const WindowSpec& ws, const Options& o) {
  const auto per = [&](double sec) { return static_cast<std::size_t>(std::max(1L, std::lround(sec / p.dt))); };
  const std::size_t warm = ws.warm, na = ws.anchor, total = ws.anchor + ws.tail, bsteps = ws.block;
  FeatureBuilder fb(p.dt);
  StylizedMeter meter(p.dt);
  std::vector<double> mid, val, gap, act, imp;
  std::vector<double> net, prints, vol, sv, snet, sprints, svol, ssv;  // per bucket: signed count, prints, volume, signed volume (s: snipers)
  std::vector<double> lx, intr, spr, l1, moved, mavg, metas, ices, wt;  // activity, book, legacy loop, counts, weights
  std::vector<double> tsign;                                            // trades in the anchor window: sign, step
  std::vector<std::size_t> tstep;
  std::vector<std::vector<double>> feat(kFeatures), raw(kFeatures);
  for (auto* v : {&mid, &val, &gap, &act, &imp, &net, &prints, &vol, &sv, &snet, &sprints, &svol, &ssv, &lx, &intr, &spr, &l1, &moved, &mavg, &metas, &ices, &wt})
    v->reserve(total);
  for (std::size_t k = 0; k < kFeatures; ++k) {
    feat[k].reserve(total);
    raw[k].reserve(total);
  }
  std::array<std::uint64_t, EsSimulator::kTakerKinds> vol0{}, vol1{};
  std::uint64_t inf0 = 0, uninf0 = 0, rep0 = 0, rept0 = 0, races0 = 0, scan0 = 0;
  std::uint64_t inf1 = 0, uninf1 = 0, rep1 = 0, rept1 = 0, races1 = 0, scan1 = 0;
  double clamped = 0, clamped_low = 0, mid_pre = sim.mid(), prev_mid = sim.mid();
  for (std::size_t s = 0; s < warm + total; ++s) {
    sim.step();
    const auto& tr = sim.trades();
    const std::size_t first_snipe = tr.size() - sim.sniper_prints();
    const bool anchored = s >= warm && s < warm + na;
    std::array<double, 8> b{};
    for (std::size_t k = 0; k < tr.size(); ++k) {
      const Trade& t = tr[k];
      fb.add_trade(t);
      if (anchored) {
        meter.add_trade(t);
        tsign.push_back(sign(t.aggressor));
        tstep.push_back(s - warm);
      }
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
      mid_pre = prev_mid = sim.mid();
    }
    if (s + 1 == warm + na) {  // counts over the anchor window only
      for (std::size_t k = 0; k < EsSimulator::kTakerKinds; ++k) vol1[k] = sim.taker_volume(static_cast<EsSimulator::TakerKind>(k));
      inf1 = sim.meta_starts(true);
      uninf1 = sim.meta_starts(false);
      rep1 = sim.reprices();
      rept1 = sim.reprice_ticks();
      races1 = sim.races();
      scan1 = sim.stale_cancelled_lots();
    }
    if (s < warm) continue;
    if (anchored) meter.end_bucket(q);
    mid.push_back(sim.mid());
    val.push_back(sim.fair_value());
    gap.push_back(sim.fair_value() - sim.mid());
    act.push_back(sim.activity());
    imp.push_back(sim.flow_impact());
    lx.push_back(sim.log_activity());
    intr.push_back(sim.intraday_factor());
    spr.push_back(q.ask - q.bid);
    l1.push_back(0.5 * (q.bid_size[0] + q.ask_size[0]));
    moved.push_back(std::abs(sim.mid() - prev_mid));
    prev_mid = sim.mid();
    mavg.push_back(sim.move_avg());
    metas.push_back(static_cast<double>(sim.active_meta_orders()));
    ices.push_back(static_cast<double>(sim.active_icebergs()));
    wt.push_back(ws.coverage ? coverage_weight(*ws.coverage, ws.start + static_cast<double>(s + 1) * p.dt) : 1.0);
    if (o.trace >= 0 && std::abs(static_cast<double>(s - warm) * p.dt - o.trace) < 5)  // --trace T: the book around T s (debugging)
      std::fprintf(stderr, "seed %llu t %.1f value %.2f bid %d ask %d sizes %u %u | %u %u impact %.2f prints %zu activity %.2f\n",
                   static_cast<unsigned long long>(seed), static_cast<double>(s - warm) * p.dt, sim.fair_value(), q.bid, q.ask, q.bid_size[0],
                   q.bid_size[1], q.ask_size[0], q.ask_size[1], sim.flow_impact(), tr.size(), sim.activity());
    if (anchored && sim.log_activity() >= p.activity_max - 1e-9) clamped += 1;
    if (anchored && sim.log_activity() <= p.activity_min + 1e-9) clamped_low += 1;
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
  const std::size_t n = mid.size(), hmax = per(60), s1 = per(1), s2 = per(2), s10 = per(10), s30 = per(30), b60 = per(60);
  if (n <= hmax + 2 || na == 0) return d;
  const std::size_t nb = (na + bsteps - 1) / bsteps;
  d.blocks.resize(nb);
  // Accumulate one Mom per block over anchors [i0, min(i1, na)): fn(i, mom of i's block), weighted by i's weight.
  auto acc = [&](const std::string& k, std::size_t i0, std::size_t i1, const std::function<void(std::size_t, Mom&)>& fn) {
    std::vector<Mom> v(nb);
    for (std::size_t i = i0; i < std::min(i1, na); ++i) {
      g_w = wt[i];
      fn(i, v[i / bsteps]);
    }
    g_w = 1;
    for (std::size_t b = 0; b < nb; ++b) d.blocks[b][k] += v[b];
  };
  const std::size_t m_end = std::min(n - hmax, na);  // anchors with every horizon ahead of them
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
  std::vector<std::size_t> changes, ideal_changes, changes_a, ideal_changes_a;  // _a: in the anchor window
  for (std::size_t i = 1; i < n; ++i) {
    if (mid[i] != mid[i - 1]) changes.push_back(i);
    if (ref(i) != ref(i - 1)) ideal_changes.push_back(i);
  }
  for (std::size_t c : changes)
    if (c < na) changes_a.push_back(c);
  for (std::size_t c : ideal_changes)
    if (c < na) ideal_changes_a.push_back(c);
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
  const double minutes = static_cast<double>(na) * p.dt / 60;
  // After a mid change: later drift in its direction; reversal share (ideal grid too).
  {
    for (double h : {1.0, 5.0, 30.0}) {
      const std::size_t hs = per(h);
      double a = 0, c = 0;
      for (std::size_t k : changes_a)
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
    put("next_change_reverses", 0, reversal(changes_a, [&](std::size_t k) { return mid[k]; }));
    put("ideal_next_change_reverses", 0, reversal(ideal_changes_a, [&](std::size_t k) { return ref(k); }));
    put("mid_changes_per_min", 0, static_cast<double>(changes_a.size()) / minutes);
    put("ideal_changes_per_min", 0, static_cast<double>(ideal_changes_a.size()) / minutes);
  }
  // Gap statistics, jump catch-up, stale-quote races.
  {
    acc("gap", 0, n, [&](std::size_t i, Mom& m) { m.add(gap[i], 0); });
    acc("ideal_gap", 0, n, [&](std::size_t i, Mom& m) { m.add(val[i] - ref(i), 0); });
    double mx = 0, within = 0;
    std::size_t worst = 0;
    for (std::size_t i = 0; i < na; ++i) {
      if (std::abs(gap[i]) > mx) {
        mx = std::abs(gap[i]);
        worst = i;
      }
      if (std::abs(gap[i]) <= 1) within += 1;
    }
    put("gap_max_abs", 0, mx);
    put("gap_max_at_s", 0, static_cast<double>(worst) * p.dt);
    put("gap_share_within_1", 0, within / static_cast<double>(na));
    std::vector<double> catchup;
    for (std::size_t i = 1; i < na; ++i)
      if (std::abs(val[i] - val[i - 1]) > 2) {
        std::size_t k = i;
        while (k < n && std::abs(gap[k]) > 1 && k - i < hmax * 10) ++k;
        catchup.push_back(static_cast<double>(k - i) * p.dt);
      }
    put("jumps", 0, static_cast<double>(catchup.size()));
    put("jump_catchup_median_s", 0, quantile(catchup, 0.5));  // per window; the seed row pools (build_market)
    put("jump_catchup_p95_s", 0, quantile(catchup, 0.95));
    d.catchup = catchup;
    // A race: consecutive snapshots with value past the band (|g| > band) on
    // one side; it ends with a tick toward value, or with value coming back.
    std::vector<double> dur;
    double races = 0, ticked = 0, in_steps = 0;
    for (std::size_t i = 0; i < na;) {
      if (!(std::abs(gap[i]) > band)) {
        ++i;
        continue;
      }
      const double dir = gap[i] > 0 ? 1 : -1, m0 = mid[i];
      std::size_t k = i;
      while (k < n && std::abs(gap[k]) > band && dir * gap[k] > 0 && mid[k] == m0) ++k;
      races += 1;
      in_steps += static_cast<double>(std::min(k, na) - i);
      if (k < n && dir * (mid[k] - m0) > 0) {
        ticked += 1;
        dur.push_back(static_cast<double>(k - i + 1) * p.dt);  // from the step value passed the band to the tick
      }
      i = std::max(k, i + 1);
    }
    put("races_per_min", 0, races / minutes);
    put("race_tick_share", 0, races > 0 ? ticked / races : 0);
    put("race_time_share", 0, in_steps / static_cast<double>(na));
    put("race_median_s", 0, quantile(dur, 0.5));
    put("race_p95_s", 0, quantile(dur, 0.95));
    put("race_floor_share", 0, dur.empty() ? 0 : static_cast<double>(std::count_if(dur.begin(), dur.end(), [&p](double x) { return x <= 2.5 * p.dt; })) / static_cast<double>(dur.size()));
    const double sn = static_cast<double>(vol1[EsSimulator::kSniper] - vol0[EsSimulator::kSniper]);
    const double ca = static_cast<double>(scan1 - scan0);
    put("stale_lots_cancel_share", 0, sn + ca > 0 ? ca / (sn + ca) : 0);
    put("sim_races_per_min", 0, static_cast<double>(races1 - races0) / minutes);
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
  // (d) Price on flow: 10 s windows (non-overlapping) inside the anchor window; 30 s delta vs the next 30 s.
  for (std::size_t i = s10; i < n && i <= na; i += s10) {
    double dall = 0, dsn = 0;
    for (std::size_t k = i - s10; k < i; ++k) {
      dall += sv[k];
      dsn += ssv[k];
    }
    const double dm = mid[i] - mid[i - s10];
    Block& bl = d.blocks[(i - 1) / bsteps];
    g_w = wt[i - 1];
    bl["R2_all"].add(dall, dm);
    bl["R2_oth"].add(dall - dsn, dm);
    bl["R2_snp"].add(dsn, dm);
    bl["impact_share"].add(imp[i] - imp[i - s10], val[i] - val[i - s10]);
  }
  for (std::size_t i = s30; i + s30 < n && i <= na; i += s30) {
    double dl = 0;
    for (std::size_t k = i - s30; k < i; ++k) dl += sv[k];
    g_w = wt[i - 1];
    d.blocks[(i - 1) / bsteps]["DP"].add(dl, mid[i + s30] - mid[i]);
  }
  g_w = 1;
  acc("sniper_print_share", 0, n, [&](std::size_t i, Mom& m) { m.ratio(sprints[i], prints[i]); });
  // (d) Pooled stylized facts: the same definitions as StylizedMeter, as
  // moments, so many short windows pool (1-minute returns on the window's own
  // minute grid; autocorrelations only from pairs inside one window).
  acc("F_trades", 0, n, [&](std::size_t i, Mom& m) { m.ratio(prints[i], p.dt); });
  acc("F_lots", 0, n, [&](std::size_t i, Mom& m) { m.ratio(vol[i], prints[i]); });
  acc("F_spread1", 0, n, [&](std::size_t i, Mom& m) { m.ratio(spr[i] == 1 ? 1.0 : 0.0, 1); });
  acc("F_l1", 0, n, [&](std::size_t i, Mom& m) { m.ratio(l1[i], 1); });
  std::vector<double> r1;
  std::vector<std::size_t> r1_end;
  for (std::size_t k = 0; (k + 1) * b60 <= na; ++k) {
    const std::size_t e = (k + 1) * b60 - 1;
    r1.push_back(mid[e] - (k == 0 ? mid_pre : mid[k * b60 - 1]));
    r1_end.push_back(e);
  }
  for (std::size_t k = 0; k < r1.size(); ++k) {
    g_w = wt[r1_end[k]];
    Block& bl = d.blocks[r1_end[k] / bsteps];
    bl["R1"].add(r1[k], 0);
    if (k + 1 < r1.size()) {
      bl["R1ac"].add(r1[k], r1[k + 1]);
      bl["AR1ac"].add(std::abs(r1[k]), std::abs(r1[k + 1]));
    }
  }
  for (std::size_t lag : kSignLags)
    for (std::size_t t = 0; t + lag < tsign.size(); ++t) {
      g_w = wt[tstep[t]];
      d.blocks[tstep[t] / bsteps][key("SG", static_cast<double>(lag))].add(tsign[t], tsign[t + lag]);
    }
  g_w = 1;
  // Activity health: the clamps, log activity against its stationary law, drift inside the window.
  acc("A_x", 0, n, [&](std::size_t i, Mom& m) { m.add(lx[i], 0); });
  acc("A_low", 0, n, [&](std::size_t i, Mom& m) { m.ratio(lx[i] <= p.activity_min + 1e-9 ? 1.0 : 0.0, 1); });
  acc("A_near", 0, n, [&](std::size_t i, Mom& m) { m.ratio(lx[i] <= p.activity_min + 0.05 ? 1.0 : 0.0, 1); });
  acc("A_up", 0, n, [&](std::size_t i, Mom& m) { m.ratio(lx[i] >= p.activity_max - 1e-9 ? 1.0 : 0.0, 1); });
  acc("A_ratio", 0, n, [&](std::size_t i, Mom& m) { m.ratio(act[i], intr[i]); });
  acc("A_x_first", 0, 1, [&](std::size_t i, Mom& m) { m.add(lx[i], 0); });
  acc("A_x_m0", 0, b60, [&](std::size_t i, Mom& m) { m.add(lx[i], 0); });
  if (na >= b60) acc("A_x_m9", na - b60, na, [&](std::size_t i, Mom& m) { m.add(lx[i], 0); });
  if (static_cast<double>(na) * p.dt >= 4 * 3600) {
    const std::size_t h2 = per(7200);
    acc("A_x_first2h", 0, h2, [&](std::size_t i, Mom& m) { m.add(lx[i], 0); });
    acc("A_x_last2h", na - h2, na, [&](std::size_t i, Mom& m) { m.add(lx[i], 0); });
  }
  acc("CNT_meta", 0, n, [&](std::size_t i, Mom& m) { m.add(metas[i], 0); });
  acc("CNT_ice", 0, n, [&](std::size_t i, Mom& m) { m.add(ices[i], 0); });
  // The legacy activity loop (model 0): per-step move against log activity, the move EWMA.
  acc("L_slope", 1, n, [&](std::size_t i, Mom& m) { m.add(lx[i - 1], moved[i]); });
  acc("L_mavg", 0, n, [&](std::size_t i, Mom& m) { m.add(mavg[i], 0); });
  acc("L_moved", 0, n, [&](std::size_t i, Mom& m) { m.add(moved[i], 0); });
  // Per-window scalars (quantiles across episodes).
  {
    double pr = 0, am = 0, r2 = 0, rm = 0;
    for (std::size_t i = 0; i < na; ++i) {
      pr += prints[i];
      am += act[i];
    }
    for (double r : r1) rm += r / static_cast<double>(r1.size());
    for (double r : r1) r2 += (r - rm) * (r - rm);
    put("win_trades_per_sec", 0, pr / (static_cast<double>(na) * p.dt));
    put("win_ret1m_sd", 0, r1.size() > 1 ? std::sqrt(r2 / static_cast<double>(r1.size() - 1)) : std::nan(""));
    put("win_activity", 0, am / static_cast<double>(na));
  }
  // Stylized facts and health (per window).
  const StylizedFacts f = meter.facts();
  const std::vector<std::pair<const char*, double>> facts = {
      {"trades_per_sec", f.trades_per_sec}, {"mean_trade_size", f.mean_trade_size}, {"volume_per_min", f.volume_per_min},
      {"spread1_share", f.spread1_share},   {"mean_l1_size", f.mean_l1_size},       {"ret1m_sd", f.ret1m_sd},
      {"ret1m_kurtosis", f.ret1m_kurtosis}, {"ret1m_ac1", f.ret1m_ac1},             {"absret1m_ac1", f.absret1m_ac1},
      {"sign_ac1", f.sign_ac1},             {"sign_ac10", f.sign_ac10},             {"sign_ac100", f.sign_ac100},
      {"impact_r2", f.impact_r2},           {"delta_predict", f.delta_predict},     {"markout_1s", f.markout_1s},
      {"markout_5s", f.markout_5s},         {"markout_30s", f.markout_30s},         {"mid_changes_per_min", f.mid_changes_per_min}};
  for (const auto& [k, v] : facts) put(std::string("fact_") + k, 0, v);
  double am = 0, ax = 0;
  for (std::size_t i = 0; i < na; ++i) {
    am += act[i];
    ax = std::max(ax, act[i]);
  }
  put("activity_mean", 0, am / static_cast<double>(na));
  put("activity_max", 0, ax);
  put("activity_clamped_share", 0, clamped / static_cast<double>(na));
  put("activity_lower_clamped_share", 0, clamped_low / static_cast<double>(na));
  const double reps = static_cast<double>(rep1 - rep0);
  put("reprices_per_min", 0, reps / minutes);
  put("ticks_per_reprice", 0, reps > 0 ? static_cast<double>(rept1 - rept0) / reps : 0);
  put("informed_starts_per_min", 0, static_cast<double>(inf1 - inf0) / minutes);
  put("uninformed_starts_per_min", 0, static_cast<double>(uninf1 - uninf0) / minutes);
  double tv = 0;
  std::array<double, EsSimulator::kTakerKinds> kv{};
  for (std::size_t k = 0; k < EsSimulator::kTakerKinds; ++k) {
    kv[k] = static_cast<double>(vol1[k] - vol0[k]);
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
  int kind = 0;  // 0 rule, 1 flat, 2 gap oracle, 3 reference oracle (latency 0), 4 clairvoyant, 5 O1, 6 O2, 7 loose EMO
  double theta = 0;
  double decide = 1.0;  // seconds per decision
  bool emo = false;     // expensive: runs on --emo episodes only
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
  // Expected-mid oracle: K forks of the market (fresh randomness, the full
  // hidden state) stepped H s ahead; trade on the mean future mid change.
  auto emo = [hold](double h, int k_forks, double th) -> Policy {
    return [=](const ScalpObs& o, const ScalpEnv& e) -> Action {
      const EsSimulator& m = e.market();
      const auto steps = static_cast<int>(std::lround(h / m.params().dt));
      double f = 0;
      for (int k = 0; k < k_forks; ++k) {
        EsSimulator c = m;
        c.fork(static_cast<std::uint64_t>(m.t_ns()) * 64u + static_cast<std::uint64_t>(k));
        for (int i = 0; i < steps; ++i) c.step();
        f += c.mid() - m.mid();
      }
      f /= static_cast<double>(k_forks);
      return f > th ? 2 : f < -th ? 0 : hold(o);
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
  v.push_back({"O1 gap > 0.62, lat 0, 0.1 s", oracle(kO1Theta), false, 0, 5, kO1Theta, kO1Decide});
  v.push_back({"O2 EMO 10 s K32 > 1.24", emo(kO2Horizon, kO2K, kO2Theta), false, 1, 6, kO2Theta, 1.0, true});
  v.push_back({"EMO 2 s K32 > 1.24 (loose)", emo(2.0, kO2K, kO2Theta), false, 1, 7, kO2Theta, 1.0, true});
  return v;
}

struct RuleRun {
  std::vector<double> net, lots;
};

constexpr double kLotFee = 1.5 / kTickValue;  // ticks per lot per side

RuleRun run_rule(const SimParams& sim, std::uint64_t seed, const Strategy& s, unsigned i0, unsigned i1) {
  ScalpParams p;
  p.sim = sim;
  p.sim.separate_streams = p.sim.separate_streams || s.separate_streams;
  p.latency_steps = s.latency_steps;
  p.decide_seconds = s.decide;
  RuleRun rr;
  for (unsigned i = i0; i < i1; ++i) {
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
  // Passive fills split by cause: a stale sweep (value already past our quote when it filled) or flow.
  double stale_adv = 0, stale_n = 0, flow_adv = 0, flow_n = 0;
};

TickRun run_tick(const SimParams& sim, std::uint64_t seed, const TickStrategy& s, unsigned i0, unsigned i1) {
  TickParams p;
  p.sim = sim;
  p.sim.separate_streams = p.sim.separate_streams || s.separate_streams;
  p.decide_seconds = s.decide;
  TickRun tr;
  struct Pending {
    double sign, price;
    std::int64_t due;
    bool stale;
  };
  std::vector<Pending> pend;
  for (unsigned i = i0; i < i1; ++i) {
    TickScalpEnv env(p);
    rag::agent::Rng r(1'000'000ull * seed + 850'000 + i);  // iteration 3's tick sub-block (v1 / v2: + 500,000)
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
        const double mk = it->sign * (env.market().mid() - it->price);
        tr.adverse += mk;
        tr.adverse_n += 1;
        (it->stale ? tr.stale_adv : tr.flow_adv) += mk;
        (it->stale ? tr.stale_n : tr.flow_n) += 1;
        it = pend.erase(it);
      }
      const auto& hidden = env.step_fill_hidden_values();  // diagnostics: value at each fill
      for (std::size_t k = 0; k < env.step_fills().size(); ++k) {  // our bid hit (Buy): we bought
        const OwnFill& f = env.step_fills()[k];
        const double sg = f.side == Side::Buy ? 1.0 : -1.0;
        pend.push_back({sg, static_cast<double>(f.price), now + 1'000'000'000, sg * (hidden[k] - f.price) < 0});
      }
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
// Round trip (value_impact, review of iteration 3): buy N lots in one market
// order, wait kRoundTripWait s, sell N; net ticks per lot after fees, raw and
// paired against the twin's mid change (own impact alone).
constexpr std::array<std::uint32_t, 4> kRoundTripLots = {1, 10, 30, 60};
constexpr double kRoundTripWait = 10;
struct ImpactRun {
  std::vector<double> end_small, end_big, after60, after300;
  std::array<std::vector<double>, kRoundTripLots.size()> rt_raw, rt_paired;
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
    const double fee2 = 2 * ScalpParams{}.fee_dollars / kTickValue;  // both sides, per lot
    for (std::size_t q = 0; q < kRoundTripLots.size(); ++q) {
      EsSimulator a(p, s, 3600), b(p, s, 3600);
      for (int k = 0; k < per(150); ++k) {
        a.step();
        b.step();
      }
      const std::uint32_t n = kRoundTripLots[q];
      const double b0 = b.mid();
      const Fill buy = a.market_order(Side::Buy, n);
      for (int k = 0; k < per(kRoundTripWait); ++k) {
        a.step();
        b.step();
      }
      const Fill sell = a.market_order(Side::Sell, n);
      if (buy.filled != n || sell.filled != n) continue;  // the book was too thin: not a round trip
      const double raw = sell.avg_price - buy.avg_price - fee2;
      ir.rt_raw[q].push_back(raw);
      ir.rt_paired[q].push_back(raw - (b.mid() - b0));
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

using RowMap = std::map<std::pair<std::string, double>, double>;

// One market's results in one regime: blocks of every window of every seed,
// per-seed rows (means over the seed's windows), every window's rows.
struct Market {
  std::vector<Block> blocks;
  std::vector<Block> seed_total;
  Block total;
  std::vector<RowMap> rows;     // per seed
  std::vector<RowMap> windows;  // every window
  [[nodiscard]] double row(std::size_t s, const std::string& m, double h) const {
    const auto it = rows[s].find({m, h});
    return it == rows[s].end() ? std::nan("") : it->second;
  }
  [[nodiscard]] std::vector<double> across_windows(const std::string& m) const {
    std::vector<double> v;
    for (const auto& w : windows) {
      const auto it = w.find({m, 0});
      if (it != w.end() && !std::isnan(it->second)) v.push_back(it->second);
    }
    return v;
  }
};

// Pool per-seed lists of windows (diags[seed][window]) into a Market.
Market build_market(const std::vector<std::vector<Diag>>& diags) {
  Market mk;
  const std::size_t ns = diags.size();
  mk.rows.resize(ns);
  mk.seed_total.resize(ns);
  for (std::size_t si = 0; si < ns; ++si) {
    std::map<std::pair<std::string, double>, std::pair<double, double>> sum;  // sum, count (NaN skipped)
    for (const Diag& d : diags[si]) {
      RowMap w;
      for (const auto& r : d.rows) {
        w[{r.metric, r.horizon}] = r.value;
        auto& s = sum[{r.metric, r.horizon}];
        if (!std::isnan(r.value)) {
          s.first += r.value;
          s.second += 1;
        }
      }
      mk.windows.push_back(std::move(w));
      for (const Block& b : d.blocks) {
        mk.blocks.push_back(b);
        for (const auto& [k, m] : b) {
          mk.seed_total[si][k] += m;
          mk.total[k] += m;
        }
      }
    }
    for (const auto& [k, s] : sum) mk.rows[si][k] = s.second > 0 ? s.first / s.second : std::nan("");
    // Quantiles of the jumps pooled over the seed's windows, not a mean of
    // per-window quantiles (a 10-minute window has 0-1 jumps: its p95 is that jump).
    std::vector<double> cu;
    for (const Diag& d : diags[si]) cu.insert(cu.end(), d.catchup.begin(), d.catchup.end());
    mk.rows[si][{"jump_catchup_median_s", 0}] = quantile(cu, 0.5);
    mk.rows[si][{"jump_catchup_p95_s", 0}] = quantile(cu, 0.95);
  }
  return mk;
}

// Both regimes' blocks in one market (pooled checks).
Market merge(const Market& a, const Market& b) {
  Market m = a;
  m.blocks.insert(m.blocks.end(), b.blocks.begin(), b.blocks.end());
  for (const auto& [k, x] : b.total) m.total[k] += x;
  for (std::size_t s = 0; s < m.seed_total.size() && s < b.seed_total.size(); ++s)
    for (const auto& [k, x] : b.seed_total[s]) m.seed_total[s][k] += x;
  return m;
}

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
Stat mean_of(const std::string& k) {
  return [k](const Get& g) { return g(k).mean_x(); };
}
Stat sd_of(const std::string& k) {
  return [k](const Get& g) { return std::sqrt(g(k).var_x()); };
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

// Every criterion (a)-(d) of one market in one regime. legacy: the v1 / v2
// tables' per-seed StylizedMeter rows (one long window per seed) too;
// episode: the in-window drift rows (first step, minute 0-1 vs 9-10).
void print_criteria(const Market& mk, const SimParams& p, bool legacy, bool episode, double default_l1) {
  const std::size_t ns = mk.rows.size();
  auto rv = [&mk](const std::string& m, double h) { return [&mk, m, h](std::size_t s) { return mk.row(s, m, h); }; };
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
  if (legacy) row_line(mk, "max |g| at, s after the warm-up", rv("gap_max_at_s", 0));
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
    if (legacy) stat_line(mk, "leak corr(" + s + ", g)", corr_of("L_" + s), {-1e300, 0.5, 0, "<= 0.5"});
    else stat_line(mk, "leak corr(" + s + ", g)", corr_of("L_" + s), {-1e300, 0.40, 0, "<= 0.40"});
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
  {  // (c) as gated in iteration 3: within 2 honest SE of 0
    const Value v = eval(mk, [](const Get& g) { return g("MKp_all@30").rat() - g("MKp_all@5").rat(); });
    say("  %-46s %+8.3f +- %.3f  %s\n", "mk30 - mk5 within 2 SE of 0", v.pooled, v.se, std::abs(v.pooled) <= 2 * v.se ? "PASS" : "FAIL");
  }
  stat_line(mk, "prevailing markout 60 s - 5 s", [](const Get& g) { return g("MKp_all@60").rat() - g("MKp_all@5").rat(); },
            {-0.15, 0.15, 0.30, "|.| <= 0.15"});
  stat_line(mk, "StylizedMeter markout 1 s", rat_of("MK@1"), {-0.05, 1e300, 0, ">= -0.05"});
  say("(d) stylized facts and health\n");
  if (legacy) {
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
    std::vector<double> k4;
    for (std::size_t s = 0; s < ns; ++s) k4.push_back(mk.row(s, "fact_ret1m_kurtosis", 0));
    say("  %-46s %+8.3f  seed mean in 5 .. 10: %s\n", "excess kurtosis", mean(k4), mean(k4) >= 5 && mean(k4) <= 10 ? "PASS" : "FAIL");
    if (!std::isnan(default_l1))
      row_line(mk, "mean L1 size / default's", [&mk, default_l1](std::size_t s) { return mk.row(s, "fact_mean_l1_size", 0) / default_l1; }, "<= 1.5",
               [](double x) { return x <= 1.5; });
  }
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
  row_line(mk, "share of time at the UPPER activity clamp", rv("activity_clamped_share", 0), "< 0.05", [](double x) { return x < 0.05; });
  row_line(mk, "share of time at the LOWER activity clamp", rv("activity_lower_clamped_share", 0), "< 0.05", [](double x) { return x < 0.05; });
  row_line(mk, "informed meta-order starts per minute", rv("informed_starts_per_min", 0));
  row_line(mk, "uninformed meta-order starts per minute", rv("uninformed_starts_per_min", 0));
  say("(d) pooled facts (moments over every window; checks on the pooled value +- SE)\n");
  stat_line(mk, "trades per second", rat_of("F_trades"), {15, 22, 0, "15 .. 22"});
  stat_line(mk, "lots per trade", rat_of("F_lots"), {3.4, 4.4, 0, "3.4 .. 4.4"});
  stat_line(mk, "spread = 1 tick share", rat_of("F_spread1"), {0.95, 1e300, 0, ">= 0.95"});
  stat_line(mk, "mean L1 size", rat_of("F_l1"), {-1e300, 190, 0, "<= 190"});
  stat_line(mk, "1-min return sd", sd_of("R1"), {3.0, 4.5, 0, "3.0 .. 4.5"});
  stat_line(mk, "1-min excess kurtosis", [](const Get& g) { return g("R1").exkurt_x(); }, {5, 10, 0, "5 .. 10"});
  stat_line(mk, "1-min return ac1 (pairs within a window)", corr_of("R1ac"), {-0.1, 0.1, 0, "|.| <= 0.1"});
  stat_line(mk, "|1-min return| ac1 (pairs within a window)", corr_of("AR1ac"), {0.1, 1e300, 0, ">= 0.1"});
  stat_line(mk, "trade-sign ac lag 1", corr_of("SG@1"), {0.2, 1e300, 0, ">= 0.2"});
  stat_line(mk, "trade-sign ac lag 10", corr_of("SG@10"));
  stat_line(mk, "trade-sign ac lag 100", corr_of("SG@100"), {0.05, 1e300, 0, ">= 0.05"});
  for (const char* k : {"win_trades_per_sec", "win_ret1m_sd", "win_activity"}) {
    const auto v = mk.across_windows(k);
    say("  %-46s 10%% %+8.3f  50%% %+8.3f  90%% %+8.3f  (%zu windows)\n", k, quantile(v, 0.1), quantile(v, 0.5), quantile(v, 0.9), v.size());
  }
  say("(d) activity health (model %d, news kick %.2f, vol %.3f, tau %.0f s, clamps [%.1f, %.1f], warm start %s)\n", p.activity_model,
      p.activity_news_excite, p.activity_vol, p.activity_tau, p.activity_min, p.activity_max, p.activity_warm_start ? "on" : "off");
  const bool m1 = p.activity_model == 1;
  const ActivityLaw law = activity_law(p);
  stat_line(mk, "share of steps at the LOWER clamp", rat_of("A_low"), {-1e300, m1 ? 0.001 : 0.05, 0, m1 ? "< 0.001" : "< 0.05"});
  stat_line(mk, "share within 0.05 of the LOWER clamp", rat_of("A_near"));
  stat_line(mk, "share of steps at the UPPER clamp", rat_of("A_up"), {-1e300, m1 ? 0.001 : 0.05, 0, m1 ? "< 0.001" : "< 0.05"});
  stat_line(mk, "log activity mean", mean_of("A_x"));
  stat_line(mk, "log activity sd", sd_of("A_x"));
  stat_line(mk, "log activity skew", [](const Get& g) { return g("A_x").skew_x(); });
  if (m1) say("  stationary law (analytic): m0 %+.4f, sd %.4f, skew %+.3f\n", law.m0, law.sd, law.skew);
  stat_line(mk, "mean(activity) / mean(intraday) (model 1: 1)", rat_of("A_ratio"));
  stat_line(mk, "active meta-orders (mean)", mean_of("CNT_meta"));
  stat_line(mk, "active icebergs (mean)", mean_of("CNT_ice"));
  if (episode) {
    stat_line(mk, "log A, first measured step: mean", mean_of("A_x_first"));
    stat_line(mk, "log A, first measured step: sd", sd_of("A_x_first"));
    stat_line(mk, "log A, minute 0-1 of the window: mean", mean_of("A_x_m0"));
    stat_line(mk, "log A, minute 9-10 of the window: mean", mean_of("A_x_m9"));
    stat_line(mk, "drift: minute 9-10 - minute 0-1", [](const Get& g) { return g("A_x_m9").mean_x() - g("A_x_m0").mean_x(); });
  }
  if (find(mk.total, "A_x_first2h").n > 0) {
    stat_line(mk, "log A, first 2 h of the window: mean", mean_of("A_x_first2h"));
    stat_line(mk, "log A, last 2 h of the window: mean", mean_of("A_x_last2h"));
  }
  if (!m1) {  // why model 0 can't be warm-started: its loop is unstable
    const double inv = 1 / p.activity_tau + 1.0 / 1800;
    say("  legacy activity loop: x' = -x/tau + excite * (move - move_avg); move_avg' = (move - move_avg) / 1800 s\n");
    stat_line(mk, "mean per-step |mid move|, ticks", mean_of("L_moved"));
    stat_line(mk, "mean move_avg_ (starts at 0.1)", mean_of("L_mavg"));
    stat_line(mk, "beta = d move / d log A (per step)", [](const Get& g) { return g("L_slope").slope(); });
    stat_line(mk, "loop gain g = excite * beta / dt, per s", [&p](const Get& g) { return p.activity_excite * g("L_slope").slope() / p.dt; });
    stat_line(mk, "R = g / (1/tau + 1/1800) (> 1: unstable)", [&p, inv](const Get& g) { return p.activity_excite * g("L_slope").slope() / p.dt / inv; });
  }
}

// AGREE: episode - long = 0 for every row, |d| <= max(tolerance floor, 2 SE).
void print_agree(const Market& e, const Market& l) {
  struct AgreeRow {
    const char* label;
    Stat ep, lg;  // lg empty: the same statistic
    double tol;
    bool rel;           // tol relative to |long|
    bool core = false;  // a row of the iteration-3 success metric (activity, facts, counts)
  };
  auto kurt = [](const Get& g) { return g("R1").exkurt_x(); };
  auto r2 = [](const Get& g) { return std::pow(g("R2_all").corr(), 2); };
  auto gapf = [](const Get& g) { return g("C_gap@30").corr() - g("ideal@30").corr(); };
  auto mk305 = [](const Get& g) { return g("MKp_all@30").rat() - g("MKp_all@5").rat(); };
  auto mk605 = [](const Get& g) { return g("MKp_all@60").rat() - g("MKp_all@5").rat(); };
  const std::vector<AgreeRow> rows = {
      {"trades per second", rat_of("F_trades"), {}, 0.10, true, true},
      {"lots per trade", rat_of("F_lots"), {}, 0.05, true, true},
      {"spread = 1 tick share", rat_of("F_spread1"), {}, 0.02, false},
      {"mean L1 size", rat_of("F_l1"), {}, 0.10, true},
      {"1-min return sd", sd_of("R1"), {}, 0.10, true, true},
      {"1-min excess kurtosis", kurt, {}, 2.0, false, true},
      {"1-min return ac1", corr_of("R1ac"), {}, 0.03, false},
      {"|1-min return| ac1", corr_of("AR1ac"), {}, 0.03, false},
      {"trade-sign ac lag 1", corr_of("SG@1"), {}, 0.03, false},
      {"trade-sign ac lag 100", corr_of("SG@100"), {}, 0.03, false},
      {"R^2(10 s mid ~ delta)", r2, {}, 0.03, false},
      {"delta predictability", corr_of("DP"), {}, 0.03, false},
      {"corr(g, r 30 s) - floor", gapf, {}, 0.03, false},
      {"event corr(I1, next change)", corr_of("E_I1@1"), {}, 0.03, false},
      {"event corr(I1, c5 - c1)", corr_of("EI15_I1"), {}, 0.03, false},
      {"tail corr(I1, 2 -> 30 s)", corr_of("T2_I1"), {}, 0.03, false},
      {"leak corr(I1, g)", corr_of("L_I1"), {}, 0.03, false},
      {"markout 1 s", rat_of("MKp_all@1"), {}, 0, false},
      {"markout 30 s - 5 s", mk305, {}, 0, false},
      {"markout 60 s - 5 s", mk605, {}, 0, false},
      {"activity / intraday", rat_of("A_ratio"), {}, 0.05, true},
      {"log A mean", mean_of("A_x"), {}, 0.05, false, true},
      {"log A sd", sd_of("A_x"), {}, 0.05, true, true},
      {"log A at the first measured step vs long: mean", mean_of("A_x_first"), mean_of("A_x"), 0.05, false},
      {"log A at the first measured step vs long: sd", sd_of("A_x_first"), sd_of("A_x"), 0.05, true},
      {"LOWER clamp share", rat_of("A_low"), {}, 0.02, false, true},
      {"UPPER clamp share", rat_of("A_up"), {}, 0.02, false, true},
      // Counts: 2 SE alone (no 10% floor: it hid a ~5 SE meta-order deficit;
      // only activity is warm-started, meta-orders and icebergs start empty).
      {"active meta-orders", mean_of("CNT_meta"), {}, 0, false, true},
      {"active icebergs", mean_of("CNT_ice"), {}, 0, false, true}};
  say("AGREE: episode regime vs long windows; d = episode - long, SE = sqrt(SE_e^2 + SE_l^2); AGREE when |d| <= max(tol, 2 SE)\n");
  say("  %-48s %9s %9s %9s %7s %7s   %s\n", "statistic (* core)", "episode", "long", "d", "SE", "tol", "");
  int bad = 0, bad_core = 0;
  for (const auto& r : rows) {
    const Value a = eval(e, r.ep), b = eval(l, r.lg ? r.lg : r.ep);
    const double dd = a.pooled - b.pooled, se = std::sqrt(a.se * a.se + b.se * b.se), tol = r.rel ? r.tol * std::abs(b.pooled) : r.tol;
    const bool ok = std::abs(dd) <= std::max(tol, 2 * se);
    bad += ok ? 0 : 1;
    bad_core += ok || !r.core ? 0 : 1;
    say("  %-48s %+9.4f %+9.4f %+9.4f %7.4f %7.4f   %s\n", (std::string(r.core ? "* " : "  ") + r.label).c_str(), a.pooled, b.pooled, dd, se, tol,
        ok ? "AGREE" : "DISAGREE");
  }
  say("  core rows (activity, facts, meta-order / iceberg counts) AGREE: %s (%d disagree)\n", bad_core == 0 ? "PASS" : "FAIL", bad_core);
  say("  AGREE on every row: %s (%d of %zu disagree; at 2 SE alone ~%.1f would by chance)\n", bad == 0 ? "PASS" : "FAIL", bad, rows.size(),
      0.0455 * static_cast<double>(rows.size()));
}

}  // namespace

int main(int argc, char** argv) {
  Options o;
  bool episode_warmup_given = false;
  for (int i = 1; i + 1 < argc; i += 2) {
    const std::string k = argv[i], v = argv[i + 1];
    if (k == "--markets") o.markets = split(v);
    else if (k == "--seeds") o.seeds = parse_seeds(v);
    else if (k == "--seconds") o.seconds = std::atof(v.c_str());
    else if (k == "--threads") o.threads = static_cast<unsigned>(std::max(1, std::atoi(v.c_str())));
    else if (k == "--rules") o.rules = static_cast<unsigned>(std::max(0, std::atoi(v.c_str())));
    else if (k == "--emo") o.emo = static_cast<unsigned>(std::max(0, std::atoi(v.c_str())));
    else if (k == "--tick") o.tick = static_cast<unsigned>(std::max(0, std::atoi(v.c_str())));
    else if (k == "--impact") o.impact = static_cast<unsigned>(std::max(0, std::atoi(v.c_str())));
    else if (k == "--episodes") o.episodes = static_cast<unsigned>(std::max(0, std::atoi(v.c_str())));
    else if (k == "--sessions") o.sessions = static_cast<unsigned>(std::max(1, std::atoi(v.c_str())));
    else if (k == "--episode-warmup" || k == "--scalp-warmup") {
      const auto w = parse_warmup(v);  // the whole string: "abc" or "1050s" is an error, not 0 / 1050
      if (!w) {
        std::fprintf(stderr, "%s wants a number of seconds >= 0, got %s\n", k.c_str(), v.c_str());
        return 2;
      }
      if (k == "--scalp-warmup") o.scalp_warmup = *w;
      else {
        o.episode_warmup = *w;
        episode_warmup_given = true;
      }
    }
    else if (k == "--long") o.long_mode = v;
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
  if (o.long_mode != "legacy" && o.long_mode != "session" && o.long_mode != "off") {
    std::fprintf(stderr, "--long wants legacy, session or off\n");
    return 2;
  }
  {
    ScalpParams g;
    g.warmup_seconds = o.scalp_warmup;
    if (const std::string why = warmup_error(g); !why.empty()) {
      std::fprintf(stderr, "--scalp-warmup: %s\n", why.c_str());
      return 2;
    }
  }
  if (!episode_warmup_given) o.episode_warmup = o.scalp_warmup;
  if (!(o.episode_warmup >= o.scalp_warmup)) {
    std::fprintf(stderr, "--episode-warmup must be >= the ScalpEnv warm-up (--scalp-warmup, %g s): the windows stay ScalpEnv's; the simulator starts earlier\n",
                 o.scalp_warmup);
    return 2;
  }
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
  const std::size_t nm = o.markets.size(), ns = o.seeds.size(), nr = o.rules > 0 || o.emo > 0 ? strat.size() : 0, nt = o.tick > 0 ? tstrat.size() : 0;
  const bool legacy = o.long_mode == "legacy", session = o.long_mode == "session";
  // Results: [market][seed][window] per regime; rule / tick runs per (market, seed, strategy) in episode chunks.
  using Windows = std::vector<std::vector<std::vector<Diag>>>;
  Windows dlong(nm, std::vector<std::vector<Diag>>(ns)), dep(nm, std::vector<std::vector<Diag>>(ns));
  for (std::size_t mi = 0; mi < nm; ++mi)
    for (std::size_t si = 0; si < ns; ++si) {
      dlong[mi][si].resize(legacy ? 1 : session ? o.sessions : 0);
      dep[mi][si].resize(o.episodes);
    }
  constexpr unsigned kChunk = 10, kEmoChunk = 2;
  struct ChunkRef {
    std::size_t m, s, r;
    unsigned i0, i1;
  };
  std::vector<ChunkRef> rule_chunks, tick_chunks;
  for (std::size_t mi = 0; mi < nm; ++mi)
    for (std::size_t si = 0; si < ns; ++si) {
      for (std::size_t ri = 0; ri < nr; ++ri) {
        const unsigned n_ep = strat[ri].emo ? o.emo : o.rules, step = strat[ri].emo ? kEmoChunk : kChunk;
        for (unsigned i = 0; i < n_ep; i += step) rule_chunks.push_back({mi, si, ri, i, std::min(n_ep, i + step)});
      }
      for (std::size_t ri = 0; ri < nt; ++ri)
        for (unsigned i = 0; i < o.tick; i += kChunk) tick_chunks.push_back({mi, si, ri, i, std::min(o.tick, i + kChunk)});
    }
  std::vector<RuleRun> rule_out(rule_chunks.size());
  std::vector<TickRun> tick_out(tick_chunks.size());
  std::vector<ImpactRun> impact(o.impact > 0 ? nm * ns : 0);
  ScalpParams cover;  // the episode geometry the long sessions are weighted by (ScalpEnv's defaults, warm-up --scalp-warmup)
  cover.warmup_seconds = o.scalp_warmup;
  const auto per = [](const SimParams& p, double sec) { return static_cast<std::size_t>(std::max(1L, std::lround(sec / p.dt))); };
  std::vector<std::function<void()>> tasks;
  // Expensive tasks first (EMO chunks, long sessions), so the pool ends together.
  for (std::size_t c = 0; c < rule_chunks.size(); ++c)
    if (strat[rule_chunks[c].r].emo)
      tasks.push_back([&, c] {
        const auto& ch = rule_chunks[c];
        rule_out[c] = run_rule(params[ch.m], o.seeds[ch.s], strat[ch.r], ch.i0, ch.i1);
      });
  for (std::size_t mi = 0; mi < nm; ++mi)
    for (std::size_t si = 0; si < ns; ++si) {
      for (std::size_t k = 0; k < dlong[mi][si].size(); ++k)
        tasks.push_back([&, mi, si, k] {
          const SimParams& p = params[mi];
          const std::uint64_t seed = o.seeds[si];
          if (legacy) {
            dlong[mi][si][k] = diagnose(p, EsSimulator(p, seed, 3600), seed, {3600, per(p, o.warmup), per(p, o.seconds), 0, per(p, 300), nullptr}, o);
          } else {
            ScalpParams sp = cover;
            sp.sim = p;
            const std::uint64_t s = 1'000'000ull * seed + 900'000 + k;
            dlong[mi][si][k] = diagnose(p, EsSimulator(p, s, 0), s, {0, 0, per(p, p.session_seconds), per(p, 60), per(p, 3600), &sp}, o);
          }
        });
      for (unsigned i = 0; i < o.episodes; ++i)
        tasks.push_back([&, mi, si, i] {
          const SimParams& p = params[mi];
          ScalpParams sp = cover;
          sp.sim = p;
          rag::agent::Rng r(1'000'000ull * o.seeds[si] + 800'000 + i);
          const std::size_t w = per(p, o.episode_warmup), a = per(p, sp.episode_seconds), b = per(p, 60);
          if (o.episode_warmup == sp.warmup_seconds) {  // exactly ScalpEnv's episode
            dep[mi][si][i] = diagnose(p, ScalpEnv::make_market(sp, r), o.seeds[si], {0, w, a, b, a, nullptr}, o);
          } else {  // ground truth: the same window after a longer burn (ScalpEnv::make_market's draws, the start moved earlier)
            const ScalpEnv::MarketDraw d = ScalpEnv::draw_market(sp, r);
            // Never before the open (intraday() would extrapolate above its opening
            // peak): a window whose burn would start earlier is burned from 0 (shorter).
            const double extra = o.episode_warmup - sp.warmup_seconds;
            const bool early = d.start < extra;
            const double start = early ? 0.0 : d.start - extra;
            const std::size_t wg = early ? per(p, d.start + sp.warmup_seconds) : w;
            dep[mi][si][i] = diagnose(p, EsSimulator(p, d.seed, start), o.seeds[si], {start, wg, a, b, a, nullptr}, o);
          }
        });
    }
  for (std::size_t c = 0; c < rule_chunks.size(); ++c)
    if (!strat[rule_chunks[c].r].emo)
      tasks.push_back([&, c] {
        const auto& ch = rule_chunks[c];
        rule_out[c] = run_rule(params[ch.m], o.seeds[ch.s], strat[ch.r], ch.i0, ch.i1);
      });
  for (std::size_t c = 0; c < tick_chunks.size(); ++c)
    tasks.push_back([&, c] {
      const auto& ch = tick_chunks[c];
      tick_out[c] = run_tick(params[ch.m], o.seeds[ch.s], tstrat[ch.r], ch.i0, ch.i1);
    });
  for (std::size_t u = 0; u < impact.size(); ++u) tasks.push_back([&, u] { impact[u] = run_impact(params[u / ns], o.seeds[u % ns], o.impact); });
  std::atomic<std::size_t> next{0};
  auto worker = [&] {
    for (std::size_t t; (t = next.fetch_add(1)) < tasks.size();) tasks[t]();
  };
  std::vector<std::thread> pool;
  for (unsigned i = 0; i < std::min<std::size_t>(o.threads, tasks.size()); ++i) pool.emplace_back(worker);
  for (auto& th : pool) th.join();
  // Rule / tick results per (market, seed, strategy), chunks in episode order.
  std::vector<RuleRun> rule(nm * ns * nr);
  for (std::size_t c = 0; c < rule_chunks.size(); ++c) {
    const auto& ch = rule_chunks[c];
    auto& dst = rule[(ch.m * ns + ch.s) * nr + ch.r];
    dst.net.insert(dst.net.end(), rule_out[c].net.begin(), rule_out[c].net.end());
    dst.lots.insert(dst.lots.end(), rule_out[c].lots.begin(), rule_out[c].lots.end());
  }
  std::vector<TickRun> tick(nm * ns * nt);
  for (std::size_t c = 0; c < tick_chunks.size(); ++c) {
    const auto& ch = tick_chunks[c];
    auto& dst = tick[(ch.m * ns + ch.s) * nt + ch.r];
    const auto& src = tick_out[c];
    dst.net.insert(dst.net.end(), src.net.begin(), src.net.end());
    dst.fills.insert(dst.fills.end(), src.fills.begin(), src.fills.end());
    dst.gross.insert(dst.gross.end(), src.gross.begin(), src.gross.end());
    dst.adverse += src.adverse;
    dst.adverse_n += src.adverse_n;
    dst.stale_adv += src.stale_adv;
    dst.stale_n += src.stale_n;
    dst.flow_adv += src.flow_adv;
    dst.flow_n += src.flow_n;
  }

  // Per market and regime: pool the windows.
  std::vector<Market> mk_long(nm), mk_ep(nm);
  for (std::size_t mi = 0; mi < nm; ++mi) {
    if (!o.long_mode.empty() && o.long_mode != "off") mk_long[mi] = build_market(dlong[mi]);
    if (o.episodes > 0) mk_ep[mi] = build_market(dep[mi]);
  }
  const bool has_long = o.long_mode != "off", has_ep = o.episodes > 0;
  if (std::FILE* csv = csv_file) {
    std::fprintf(csv, "market,seed,metric,horizon_s,value,se_jackknife\n");
    for (std::size_t mi = 0; mi < nm; ++mi)
      for (int reg = 0; reg < 2; ++reg) {
        if (reg == 0 ? !has_long : !has_ep) continue;
        const Market& mk = reg == 0 ? mk_long[mi] : mk_ep[mi];
        // Legacy runs keep the v1 / v2 market names; the new regimes are suffixed.
        const std::string name = o.markets[mi] + (reg == 1 ? "/episode" : legacy ? "" : "/session");
        for (std::size_t si = 0; si < ns; ++si)
          for (const auto& [k, v] : mk.rows[si])
            std::fprintf(csv, "%s,%llu,%s,%g,%.6g,\n", name.c_str(), static_cast<unsigned long long>(o.seeds[si]), k.first.c_str(), k.second, v);
        for (const auto& [k, m] : mk.total) {  // moments: correlations (ratio markouts as ratio:)
          const bool r = m.den != 0;
          const Value v = eval(mk, r ? rat_of(k) : corr_of(k));
          for (std::size_t si = 0; si < ns; ++si)
            std::fprintf(csv, "%s,%llu,%s:%s,0,%.6g,\n", name.c_str(), static_cast<unsigned long long>(o.seeds[si]), r ? "ratio" : "corr", k.c_str(),
                         v.seeds[si]);
          std::fprintf(csv, "%s,all,%s:%s,0,%.6g,%.6g\n", name.c_str(), r ? "ratio" : "corr", k.c_str(), v.pooled, v.se);
        }
      }
    std::fclose(csv);
  }

  say("OF10 price-discovery diagnostics: seeds");
  for (auto s : o.seeds) say(" %llu", static_cast<unsigned long long>(s));
  if (legacy) say(", %.0f s each after a %.0f s warm-up (session start 3600 s)\n", o.seconds, o.warmup);
  else if (session)
    say(", long windows: %u full sessions per seed (0 .. 23,400 s; block statistics weighted by ScalpEnv's\n"
        "  time-of-day coverage, per-window rows unweighted)\n",
        o.sessions);
  else say(", no long windows\n");
  if (has_ep)
    say("Episode regime: %u windows per seed built as ScalpEnv builds them (ScalpEnv::make_market, Rng(1,000,000 * seed + 800,000 + i)),\n"
        "  %.0f s warm-up%s, 600 s measured + 60 s look-ahead; blocks = episodes\n",
        o.episodes, o.episode_warmup, o.episode_warmup == o.scalp_warmup ? " (ScalpEnv's)"
                               : " (ground truth: the simulator starts earlier, same windows; never before the open: a burn that would\n"
                                 "  start before 0 s starts at 0 s, shorter)");
  if (o.scalp_warmup != 150 && (has_ep || session))  // opt-in: default runs print exactly what they did
    say("--scalp-warmup %.0f: ScalpEnv geometry with warmup_seconds = %.0f (episode windows drawn as that ScalpEnv draws them, start\n"
        "  uniform on [0, session - 600 - %.0f] + %.0f s; long sessions weighted by its time-of-day coverage). Rules, EMO, tick rules and\n"
        "  impact runs keep ScalpEnv's default 150 s warm-up.\n",
        o.scalp_warmup, o.scalp_warmup, o.scalp_warmup, o.scalp_warmup);
  say("Columns: pooled value +- jackknife SE (blocks); seed mean +- sd/sqrt(n) [min, max]; pooled check; per-seed check.\n");
  if (!o.set.empty()) {  // tuned runs must say so
    say("--set overrides of the discovery preset:");
    for (const auto& [k, x] : o.set) say(" %s=%g", k.c_str(), x);
    say("\n");
  }
  double default_l1 = std::nan("");
  for (std::size_t mi = 0; mi < nm; ++mi)
    if (o.markets[mi] == "default" && legacy) {
      std::vector<double> l;
      for (std::size_t s = 0; s < ns; ++s) l.push_back(mk_long[mi].row(s, "fact_mean_l1_size", 0));
      default_l1 = mean(l);
    }
  for (std::size_t mi = 0; mi < nm; ++mi) {
    const SimParams& p = params[mi];
    say("\n=== market %s (reprice_band %.2f, reprice_snipe %.2f, reprice_cancel %.2f, reprice_take %.2f, value_impact %.4f,\n"
        "    impact_forecast_tau %.1f, impact_forecast_weight %.2f, maker_info %.2f, informed_meta_rate %.2f, uninformed_meta_rate %.2f,\n"
        "    momentum_rate %.2f, noise_rate %.2f, cancel_rate %.2f, fade %.2f, activity_excite %.2f, value_vol %.2f)\n",
        o.markets[mi].c_str(), p.reprice_band, p.reprice_snipe, p.reprice_cancel, p.reprice_take, p.value_impact, p.impact_forecast_tau,
        p.impact_forecast_weight, p.maker_info, p.informed_meta_rate, p.uninformed_meta_rate, p.momentum_rate, p.noise_rate, p.cancel_rate, p.fade,
        p.activity_excite, p.value_vol);
    {
      const ActivityLaw law = activity_law(p);
      say("    activity_model %d, activity_news_excite %.2f, activity_vol %.3f, activity_tau %.0f, activity_min %.1f, activity_max %.1f, warm start %s",
          p.activity_model, p.activity_news_excite, p.activity_vol, p.activity_tau, p.activity_min, p.activity_max, p.activity_warm_start ? "on" : "off");
      if (p.activity_model == 1) say("; stationary log A: m0 %+.4f, sd %.4f, skew %+.3f", law.m0, law.sd, law.skew);
      say("\n");
    }
    if (has_long) {
      if (!legacy) say("--- LONG WINDOWS: %u full sessions x %zu seeds, every step weighted by ScalpEnv's time-of-day coverage\n", o.sessions, ns);
      print_criteria(mk_long[mi], p, legacy, false, default_l1);
    }
    if (has_ep) {
      say("--- EPISODE REGIME: %u windows x %zu seeds (ScalpEnv-built, 10 min after the warm-up)\n", o.episodes, ns);
      print_criteria(mk_ep[mi], p, false, true, default_l1);
    }
    if (has_ep && session) {
      say("--- ");
      print_agree(mk_ep[mi], mk_long[mi]);
      const Stat mk305 = [](const Get& g) { return g("MKp_all@30").rat() - g("MKp_all@5").rat(); };
      const Value vl = eval(mk_long[mi], mk305), ve = eval(mk_ep[mi], mk305), v = eval(merge(mk_ep[mi], mk_long[mi]), mk305);
      const bool cl = std::abs(vl.pooled) <= 2 * vl.se, ce = std::abs(ve.pooled) <= 2 * ve.se;
      say("  (c) judged in BOTH regimes, mk30 - mk5 within 2 SE of 0: long %+.3f +- %.3f %s, episode %+.3f +- %.3f %s: (c) %s\n", vl.pooled, vl.se,
          cl ? "PASS" : "FAIL", ve.pooled, ve.se, ce ? "PASS" : "FAIL", cl && ce ? "PASS" : "FAIL");
      say("  (c) pooled over both regimes (extra evidence only, not the gate): mk30 - mk5 %+.3f +- %.3f, within 2 SE of 0: %s\n", v.pooled, v.se,
          std::abs(v.pooled) <= 2 * v.se ? "yes" : "no");
    }

    if (nr > 0) {
      say("(e, f) rules on ScalpEnv: %u episodes x %zu seeds (episode seeds 1,000,000 * seed + i; EMO oracles: the first %u), net ticks/episode after costs\n",
          o.rules, ns, o.emo);
      say("  lots: fees / $1.50 per lot (a flip is 2 lots); gross/lot before fees; edge/lot = gross/lot + 0.5 (the half spread: ~0 without\n"
          "  information); SEs: episodes (net), episode jackknife (edge). Clairvoyants run on the same preset with separate streams.\n");
      say("  %-30s %8s %6s %6s %8s %8s %7s   per-seed means\n", "strategy", "net/ep", "t", "lots", "gross/lt", "edge/lt", "+-SE");
      struct Res {
        double net, se, lots, edge;
        int kind;
        double theta;
        bool ran;
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
        if (all.empty()) {
          res[ri] = {std::nan(""), 0, 0, std::nan(""), strat[ri].kind, strat[ri].theta, false};
          continue;
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
        res[ri] = {m, se, sl / ne, edge, strat[ri].kind, strat[ri].theta, true};
        say("  %-30s %+8.2f %+6.2f %6.1f %+8.3f %+8.3f %7.3f  ", strat[ri].name.c_str(), m, se > 0 ? m / se : 0.0, sl / ne, edge - 0.5, edge, esd);
        for (double x : seed_means) say(" %+6.2f", x);
        if (strat[ri].kind == 0)
          say("   %s; |edge| %s", m > 0 && se > 0 && m / se > 2 ? "FAIL (wins)" : "ok",
              std::isnan(edge) || std::abs(edge) <= 0.15 ? "<= 0.15" : std::abs(edge) - 2 * esd > 0.15 ? "> 0.15 FLAG" : "> 0.15 (<2SE)");
        if (strat[ri].kind == 2 && sl / ne < 1) say("   vacuous (< 1 lot/episode)");
        if (strat[ri].kind >= 5 && sl / ne < kMinOracleLots) say("   VACUOUS (< 2 lots/episode)");
        say("\n");
      }
      // Legacy theta* (v1 / v2 tables): pre-registered for this market (--theta) or this run's best non-vacuous gap oracle.
      const auto th = o.theta.count(o.markets[mi]) > 0 ? o.theta.find(o.markets[mi]) : o.theta.find("");
      const double theta = th == o.theta.end() ? -1 : th->second;
      std::size_t best = nr;
      for (std::size_t ri = 0; ri < nr; ++ri)
        if (res[ri].ran && res[ri].kind == 2 && res[ri].lots >= 1) {
          if (theta >= 0 ? std::abs(res[ri].theta - theta) < 1e-9 : best == nr || res[ri].net > res[best].net) best = ri;
        }
      if (best < nr) {
        say("  theta* = %.2f (%s): oracle net %+.2f +- %.2f, %.1f lots/episode, edge/lot %+.3f\n", res[best].theta,
            theta >= 0 ? "pre-registered" : "best of this run: pre-register it for held-out runs", res[best].net, res[best].se, res[best].lots,
            res[best].edge);
        bool flag = false;
        for (std::size_t ri = 0; ri < nr; ++ri)
          if (res[ri].ran && res[ri].kind == 0 && res[ri].net > res[best].net + 2 * res[ri].se) {
            // The O1 / O2 rule: a RULE trading < 2 lots/episode makes the comparison vacuous.
            // Still no PASS for (f) (as O1 / O2: vacuous never passes), just not a red flag.
            // Oracle vacuity never demotes a flag: a real rule beating an oracle at theta*
            // that trades < 2 lots/episode means theta* broke the upper bound.
            if (res[ri].lots < kMinOracleLots)
              say("  not meaningful (VACUOUS: < 2 lots/episode): %s beats the oracle at theta* by > 2 SE (rule %.1f, oracle %.1f lots/episode)\n",
                  strat[ri].name.c_str(), res[ri].lots, res[best].lots);
            else if (res[best].lots < kMinOracleLots)
              say("  RED FLAG (VACUOUS ORACLE: theta* trades %.1f < 2 lots/episode, no upper bound): %s beats the oracle at theta* by > 2 SE\n",
                  res[best].lots, strat[ri].name.c_str());
            else say("  RED FLAG: %s beats the oracle at theta* by > 2 SE\n", strat[ri].name.c_str());
            flag = true;
          }
        say("  (f) every rule <= oracle(theta*) + 2 SE: %s\n", flag ? "FAIL" : "PASS");
      } else {
        say("  (f) %s is vacuous (< 1 lot/episode): the bound is empty; see the clairvoyants\n",
            theta >= 0 ? "the gap oracle at the pre-registered theta*" : "every gap oracle");
      }
      // Iteration 3: the pre-registered oracles (constants at the top of this file).
      say("  (f) PRE-REGISTERED oracles (fixed in code before any v3 RL run; thresholds from costs, fee 0.12 tick/lot/side):\n"
          "      O1 = zero-latency fair-value taker, |g| > %.2f (half spread + one fee), latency 0, decide %.1f s\n"
          "      O2 = EMO(H %.0f s, K %d, theta %.2f = spread + 2 fees), decide 1 s, latency 100 ms: a reference, not an upper bound\n",
          kO1Theta, kO1Decide, kO2Horizon, kO2K, kO2Theta);
      std::size_t o1 = nr, o2 = nr, flat = nr, cl2 = nr;
      for (std::size_t ri = 0; ri < nr; ++ri) {
        if (res[ri].kind == 5) o1 = ri;
        if (res[ri].kind == 6) o2 = ri;
        if (res[ri].kind == 1) flat = ri;
        if (res[ri].kind == 4 && std::abs(res[ri].theta - 2) < 1e-9) cl2 = ri;
      }
      auto describe = [&](const char* nm2, std::size_t ri, const char* switch_name) {
        if (ri == nr || !res[ri].ran) {
          say("      %s: not run (%s 0)\n", nm2, switch_name);
          return false;
        }
        const bool vac = res[ri].lots < kMinOracleLots;
        say("      %s: net %+.2f +- %.2f ticks/episode (t %+.2f), %.1f lots/episode, edge/lot %+.3f%s\n", nm2, res[ri].net, res[ri].se,
            res[ri].se > 0 ? res[ri].net / res[ri].se : 0.0, res[ri].lots, res[ri].edge, vac ? "  VACUOUS (< 2 lots/episode)" : "");
        return !vac;
      };
      const bool o1_ok = describe("O1", o1, "--rules"), o2_ok = describe("O2", o2, "--emo");
      const bool o2_pos = o2 < nr && res[o2].ran && res[o2].net > 2 * res[o2].se;
      say("  (f1) O1 and O2 trade >= 2 lots/episode and O2 > 0 by 2 SE: %s\n", o1_ok && o2_ok && o2_pos ? "PASS" : "FAIL");
      if (!o2_ok && cl2 < nr && res[cl2].ran)
        say("  O2 vacuous or not run: the primary bound before RL is the 2 s clairvoyant (net %+.2f +- %.2f, %.1f lots/episode)\n", res[cl2].net,
            res[cl2].se, res[cl2].lots);
      if (flat < nr && res[flat].ran) {
        // The (e) gate, per rule: no naive rule profits by 2 SE. O2 is one fixed
        // rule, not a bound: a naive rule above it only triggers a leak audit.
        bool wins = false;
        for (std::size_t ri = 0; ri < nr; ++ri)
          if (res[ri].ran && res[ri].kind == 0) {
            if (res[ri].net > 2 * res[ri].se) {
              say("  WINS: %s net %+.2f > 0 by > 2 SE\n", strat[ri].name.c_str(), res[ri].net);
              wins = true;
            }
            if (o2 < nr && res[o2].ran && res[ri].net > res[o2].net + 2 * res[ri].se)
              say("  leak audit: %s above O2 by > 2 SE (unpaired: O2 ran on fewer episodes)\n", strat[ri].name.c_str());
          }
        say("  (e) no naive rule net > 0 by 2 SE (per rule): %s\n", wins ? "FAIL" : "PASS");
      }
      say("  Later RL checks: RL net > 0 by 2 SE needs a fresh-seed confirmation; compare RL with O2 PAIRED on the same episodes\n"
          "  (O2 ran on the first %u of each seed): RL > O2 + 2 SE triggers a leak audit (O2 is one fixed rule, not an upper bound),\n"
          "  as does RL > O1 (O1 reads only the current gap: not a bound either); neither proves a bug by itself.\n",
          o.emo);
    }
    if (nt > 0) {
      say("(e, f) tick rules on TickScalpEnv: %u episodes x %zu seeds (episode seeds 1,000,000 * seed + 850,000 + i), net ticks/episode\n", o.tick, ns);
      say("  adverse: mean signed (mid ~1 s after the fill's decision step - fill price) per passive fill\n"
          "  (+0.5 at a 1-tick spread and an unchanged mid: below +0.5 is adverse selection, < 0: picked off past the half spread)\n");
      say("  %-30s %8s %6s %6s %9s %8s   per-seed means\n", "strategy", "net/ep", "t", "fills", "gross/fl", "adverse");
      double oracle_mm = -1e300;
      std::vector<std::pair<std::string, double>> rules_net;
      for (std::size_t ri = 0; ri < nt; ++ri) {
        std::vector<double> all, seed_means;
        double fills = 0, gross = 0, adv = 0, advn = 0, sa = 0, sn2 = 0, fa = 0, fn = 0;
        for (std::size_t si = 0; si < ns; ++si) {
          const auto& tr = tick[(mi * ns + si) * nt + ri];
          all.insert(all.end(), tr.net.begin(), tr.net.end());
          seed_means.push_back(mean(tr.net));
          for (double x : tr.fills) fills += x;
          for (double x : tr.gross) gross += x;
          adv += tr.adverse;
          advn += tr.adverse_n;
          sa += tr.stale_adv;
          sn2 += tr.stale_n;
          fa += tr.flow_adv;
          fn += tr.flow_n;
        }
        const double ne = static_cast<double>(all.size()), m = mean(all), se = sd(all) / std::sqrt(ne);
        say("  %-30s %+8.2f %+6.2f %6.1f %+9.3f %+8.3f  ", tstrat[ri].name.c_str(), m, se > 0 ? m / se : 0.0, fills / ne, fills > 0 ? gross / fills : 0.0,
            advn > 0 ? adv / advn : 0.0);
        for (double x : seed_means) say(" %+6.2f", x);
        if (tstrat[ri].kind == 0) say("   %s", m > 0.5 && se > 0 && m / se > 2 ? "FAIL (naive quoting wins)" : "ok");
        if (tstrat[ri].name == "quote both sides") {
          say(", %s", m >= -4 && m <= 0 ? "in [-4, 0] PASS" : "outside [-4, 0] FAIL");
          const double a = advn > 0 ? adv / advn : 0.0;
          say(", adverse in [0, 0.5] %s", a >= 0 && a <= 0.5 ? "PASS" : "FAIL");
          say("\n    quote both, by cause: stale sweeps (value past the quote at the fill) %.1f/episode, 1 s markout %+.3f;"
              " flow fills %.1f/episode, 1 s markout %+.3f",
              sn2 / ne, sn2 > 0 ? sa / sn2 : 0.0, fn / ne, fn > 0 ? fa / fn : 0.0);
        }
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
    if (!impact.empty()) {
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
      say("  round trip: buy N lots in one market order, sell N %.0f s later, net ticks/lot after fees (raw; paired = minus the twin's mid change)\n",
          kRoundTripWait);
      for (std::size_t q = 0; q < kRoundTripLots.size(); ++q) {
        std::vector<double> raw, pr;
        for (std::size_t si = 0; si < ns; ++si) {
          const auto& ir = impact[mi * ns + si];
          raw.insert(raw.end(), ir.rt_raw[q].begin(), ir.rt_raw[q].end());
          pr.insert(pr.end(), ir.rt_paired[q].begin(), ir.rt_paired[q].end());
        }
        const double sr = std::sqrt(static_cast<double>(std::max<std::size_t>(1, raw.size())));
        const bool wins = !pr.empty() && mean(pr) > 2 * sd(pr) / sr;
        say("    N %2u: raw %+.3f +- %.3f, paired %+.3f +- %.3f (%zu round trips)%s\n", kRoundTripLots[q], mean(raw), sd(raw) / sr, mean(pr),
            sd(pr) / sr, pr.size(), wins ? "  PROFITS FROM OWN IMPACT" : "");
      }
      if (p.value_impact > 0)
        say("  note: own-flow value_impact is linear in own lots (value_impact %.3f x 1,200 lots = %+.0f ticks of fair value) and own\n"
            "  market orders fill at pre-impact prices: a round trip of N lots gains ~%.3f N ticks/lot from its own impact; ScalpEnv\n"
            "  refuses contracts > 1 on presets with value_impact > 0\n",
            p.value_impact, p.value_impact * 1200, p.value_impact);
    }
  }
  if (g_out) std::fclose(g_out);
}
