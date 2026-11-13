// OF24 kernel: the OF23 turning-point candidates, their features and the policy network, in plain C++17.
// No Sierra Chart header is used here; sierra/RagAgent.cpp is the adapter (docs/ai/OF24_LIVE_SIM_PLAN.md).
// The candidate and feature definitions mirror experiments/of23_mine.py line for line (candidates(),
// prior_levels(), value_area of of20_levels.py); python/tests/test_of24.py checks them against it.
//
//   of24::Bars day;                      // completed 1-min RTH bars 0..m of the day, prices in ticks
//   of24::Levels lv = of24::prior_levels(prev_day);
//   of24::Cand c;
//   if (of24::candidate_at(day, &lv, dow, c)) { ... c.f[0..36], c.dir ... }   // decided at the close of bar m
//   of24::Policy p;  p.load(path);  auto out = p.forward(x);  p.update(x, action, reward);  p.save(path);
#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <limits>
#include <random>
#include <string>
#include <vector>

namespace of24 {

inline constexpr int kFirstM = 20, kLastM = 359, kGapWin = 30;
inline constexpr int kFeat = 37;   // the OF23 features, in of23_mine.FEATS order
inline constexpr int kBook = 6;    // live book features (zero when absent)
inline constexpr int kIn = kFeat + kBook;
inline constexpr int kHidden = 64;
inline constexpr int kExits = 12;
inline constexpr int kActions = kExits + 1;  // 0 = skip
inline const double kNaN = std::numeric_limits<double>::quiet_NaN();

// feature indices (of23_mine.FEATS)
enum F : int {
  c1_1, c1_5, c1_15, c2, c3, c4, c5, c6, c3_count, c3_name, dir_, d_ext_entry, d_ext_far, d_open, d_vwap, vwap_z,
  atr_, range_atr, close_loc, relvol, delta1, delta5, delta15, delta30, ret5, ret15, ret60, ret_open,
  slope5, slope15, slope60, d_level, fvg5_n, fvg15_n, d_fvg, minute_, dow_
};

struct ExitSpec { int target, stop, hold_min; bool trailing; };  // hold_min 0 = end of day
// of23_mine.MENU x trailing {off, on}: action a (1..12) -> kMenu[a - 1]
inline constexpr std::array<ExitSpec, kExits> kMenu = {{
    {8, 4, 30, false}, {8, 4, 30, true}, {12, 6, 60, false}, {12, 6, 60, true}, {16, 8, 60, false}, {16, 8, 60, true},
    {24, 8, 120, false}, {24, 8, 120, true}, {32, 12, 0, false}, {32, 12, 0, true}, {48, 16, 0, false}, {48, 16, 0, true}}};

struct Bars {
  std::vector<double> o, h, l, c, v, buy, sell;
  int size() const { return static_cast<int>(o.size()); }
  void push(double o_, double h_, double l_, double c_, double v_, double b_, double s_) {
    o.push_back(o_); h.push_back(h_); l.push_back(l_); c.push_back(c_); v.push_back(v_); buy.push_back(b_); sell.push_back(s_);
  }
  void clear() { o.clear(); h.clear(); l.clear(); c.clear(); v.clear(); buy.clear(); sell.clear(); }
};

struct Levels {  // pdh, pdl, pdc, ppoc, pvah, pval
  bool ok = false;
  std::array<double, 6> v{};
};

struct Cand {
  int m = 0;
  int dir = 0;
  std::array<double, kFeat> f{};
};

// The prior day's levels: high, low, close, and POC / 70% value area with each bar's volume spread evenly over
// the ticks of its range (of23_mine.prior_levels, of20_levels.value_area).
inline Levels prior_levels(const Bars& p) {
  Levels L;
  const int n = p.size();
  if (n == 0) return L;
  double lo_d = p.l[0], hi_d = p.h[0];
  for (int i = 0; i < n; ++i) { lo_d = std::min(lo_d, p.l[i]); hi_d = std::max(hi_d, p.h[i]); }
  long long lo = 0;
  bool any = false;
  for (int i = 0; i < n; ++i)
    if (p.v[i] > 0) { const long long x = static_cast<long long>(p.l[i]); lo = any ? std::min(lo, x) : x; any = true; }
  if (!any) return L;
  long long hi = lo;
  for (int i = 0; i < n; ++i)
    if (p.v[i] > 0) hi = std::max(hi, static_cast<long long>(p.h[i]));
  std::vector<double> vol(static_cast<size_t>(hi - lo + 1), 0.0);
  double sum_px = 0.0, sum_sz = 0.0;
  for (int i = 0; i < n; ++i) {
    if (!(p.v[i] > 0)) continue;
    const long long a = static_cast<long long>(p.l[i]), b = static_cast<long long>(p.h[i]);
    const double per = p.v[i] / static_cast<double>(b - a + 1);
    for (long long t = a; t <= b; ++t) {
      vol[static_cast<size_t>(t - lo)] += per;
      sum_px += static_cast<double>(t) * per;
      sum_sz += per;
    }
  }
  double total = 0.0, vmax = vol[0];
  for (double x : vol) { total += x; vmax = std::max(vmax, x); }
  const double vw = sum_px / sum_sz - static_cast<double>(lo);
  int poc = -1;
  double best = 0.0;
  for (int i = 0; i < static_cast<int>(vol.size()); ++i)
    if (vol[i] == vmax) {
      const double d = std::fabs(i - vw);
      if (poc < 0 || d < best) { poc = i; best = d; }
    }
  int a = poc, b = poc;
  double held = vol[poc];
  const int nv = static_cast<int>(vol.size());
  while (held < 0.70 * total) {
    const double up = b + 1 < nv ? vol[b + 1] : -1.0;
    const double dn = a - 1 >= 0 ? vol[a - 1] : -1.0;
    if (up < 0 && dn < 0) break;
    if (up >= dn) { ++b; held += up; } else { --a; held += dn; }
  }
  L.ok = true;
  L.v = {hi_d, lo_d, p.c[n - 1], static_cast<double>(poc + lo), static_cast<double>(b + lo), static_cast<double>(a + lo)};
  return L;
}

namespace detail {

struct TfBars { std::vector<double> o, h, l, c; };

inline TfBars tf_bars(const Bars& b, int tf) {
  TfBars t;
  const int n = b.size() / tf;
  for (int k = 0; k < n; ++k) {
    double hi = b.h[k * tf], lo = b.l[k * tf];
    for (int j = k * tf; j < (k + 1) * tf; ++j) { hi = std::max(hi, b.h[j]); lo = std::min(lo, b.l[j]); }
    t.o.push_back(b.o[k * tf]); t.h.push_back(hi); t.l.push_back(lo); t.c.push_back(b.c[(k + 1) * tf - 1]);
  }
  return t;
}

// +1 / -1 flags for a 5-bar swing low / high of tf bars confirmed exactly at 1-min bar m.
inline void swing_at(const std::vector<double>& hi, const std::vector<double>& lo, int tf, int m, bool& low, bool& high) {
  low = high = false;
  if ((m + 1) % tf != 0) return;
  const int k = (m + 1) / tf - 3;
  if (k < 2 || k + 2 >= static_cast<int>(lo.size())) return;
  low = high = true;
  for (int j = k - 2; j <= k + 2; ++j) {
    if (j == k) continue;
    if (!(lo[k] < lo[j])) low = false;
    if (!(hi[k] > hi[j])) high = false;
  }
}

struct Fvg { int dir; double bot, top; int m0; int closed; int retest; bool ok; };  // closed / retest -1 = none

inline std::vector<Fvg> fvgs(const Bars& b, const TfBars& t, int tf) {
  std::vector<Fvg> out;
  const int n1 = b.size(), nt = static_cast<int>(t.h.size());
  for (int i = 2; i < nt; ++i) {
    for (int d : {1, -1}) {
      double bot, top;
      if (d == 1 && t.l[i] > t.h[i - 2]) { bot = t.h[i - 2]; top = t.l[i]; }
      else if (d == -1 && t.h[i] < t.l[i - 2]) { bot = t.h[i]; top = t.l[i - 2]; }
      else continue;
      Fvg g{d, bot, top, (i + 1) * tf - 1, -1, -1, false};
      for (int k = i + 1; k < nt; ++k)
        if ((d == 1 && t.c[k] < bot) || (d == -1 && t.c[k] > top)) { g.closed = (k + 1) * tf - 1; break; }
      for (int m = g.m0 + 1; m < n1; ++m) {
        if (g.closed >= 0 && m > g.closed) break;
        if (b.l[m] <= top && b.h[m] >= bot) {
          g.retest = m;
          g.ok = !((d == 1 && b.c[m] < bot) || (d == -1 && b.c[m] > top));
          break;
        }
      }
      out.push_back(g);
    }
  }
  return out;
}

inline double ema_slope(const TfBars& t, int m, int tf, double atr) {
  const int k = (m + 1) / tf;
  if (k < 2 || !(atr > 0)) return kNaN;
  const double a = 2.0 / 9.0;
  double e = t.c[0], prev = t.c[0];
  for (int i = 1; i < k; ++i) { prev = e; e = e + a * (t.c[i] - e); }
  return (e - prev) / atr;
}

// numpy.quantile(x, q), linear interpolation
inline double quantile(std::vector<double> x, double q) {
  std::sort(x.begin(), x.end());
  const double pos = q * static_cast<double>(x.size() - 1);
  const size_t i = static_cast<size_t>(std::floor(pos));
  if (i + 1 >= x.size()) return x.back();
  const double t = pos - static_cast<double>(i), a = x[i], b = x[i + 1];
  return t >= 0.5 ? b - (b - a) * (1.0 - t) : a + (b - a) * t;
}

}  // namespace detail

// The candidate decided at the close of the last bar in `b` (m = b.size() - 1), if any. `lv` may be null or !ok
// (no prior-day levels). Mirrors of23_mine.candidates() for that m.
inline bool candidate_at(const Bars& b, const Levels* lv, int dow, Cand& out) {
  using namespace detail;
  const int n = b.size(), m = n - 1;
  if (m < kFirstM || m > kLastM) return false;
  const auto &o = b.o, &h = b.h, &l = b.l, &c = b.c, &v = b.v;
  for (int i = std::max(0, m - kGapWin + 1); i <= m; ++i)
    if (!(v[i] > 0)) return false;
  // ATR: mean true range of bars m-19..m (bar 0: high - low)
  double atr = 0.0;
  for (int i = m - 19; i <= m; ++i)
    atr += i == 0 ? h[0] - l[0] : std::max(h[i], c[i - 1]) - std::min(l[i], c[i - 1]);
  atr /= 20.0;
  if (!(atr > 0)) return false;
  const bool has_lv = lv != nullptr && lv->ok;

  double fl[2][8] = {};  // [0] long (+1), [1] short (-1); c1_1 c1_5 c1_15 c2 c3 c4 c5 c6
  bool any[2] = {false, false};
  auto add = [&](int d, int key) { fl[d > 0 ? 0 : 1][key] += 1.0; any[d > 0 ? 0 : 1] = true; };

  const TfBars t5 = tf_bars(b, 5), t15 = tf_bars(b, 15), t60 = tf_bars(b, 60);
  bool lo_, hi_;
  swing_at(h, l, 1, m, lo_, hi_);
  if (lo_) add(1, F::c1_1);
  if (hi_) add(-1, F::c1_1);
  swing_at(t5.h, t5.l, 5, m, lo_, hi_);
  if (lo_) add(1, F::c1_5);
  if (hi_) add(-1, F::c1_5);
  swing_at(t15.h, t15.l, 15, m, lo_, hi_);
  if (lo_) add(1, F::c1_15);
  if (hi_) add(-1, F::c1_15);

  if (m >= 30) {
    double ph = h[0], pl = l[0];
    for (int i = 0; i < m; ++i) { ph = std::max(ph, h[i]); pl = std::min(pl, l[i]); }
    if (h[m] > ph && c[m] < ph) add(-1, F::c2);
    if (l[m] < pl && c[m] > pl) add(1, F::c2);
  }
  std::vector<int> fired[2];
  if (has_lv && m >= 1) {
    for (int li = 0; li < 6; ++li) {
      const double L = lv->v[li];
      if (c[m - 1] < L && h[m] >= L + 1 && c[m] < L) fired[1].push_back(li);
      else if (c[m - 1] > L && l[m] <= L - 1 && c[m] > L) fired[0].push_back(li);
    }
    if (!fired[0].empty()) add(1, F::c3);
    if (!fired[1].empty()) add(-1, F::c3);
  }
  double cv = 0.0, cvp = 0.0, cvp2 = 0.0;
  for (int i = 0; i <= m; ++i) {
    const double tp = (h[i] + l[i] + c[i]) / 3.0;
    cv += v[i]; cvp += v[i] * tp; cvp2 += v[i] * tp * tp;
  }
  const double vw = cv > 0 ? cvp / cv : kNaN;
  const double sd = cv > 0 ? std::sqrt(std::max(cvp2 / cv - vw * vw, 0.0)) : kNaN;
  if (m >= 60 && sd > 0) {
    if (h[m] >= vw + 2 * sd && c[m] < vw + 2 * sd) add(-1, F::c4);
    if (l[m] <= vw - 2 * sd && c[m] > vw - 2 * sd) add(1, F::c4);
  }
  const std::vector<Fvg> g5 = fvgs(b, t5, 5), g15 = fvgs(b, t15, 15);
  {
    bool r[2] = {false, false};
    for (const auto* gs : {&g5, &g15})
      for (const Fvg& g : *gs)
        if (g.retest == m && g.ok) r[g.dir > 0 ? 0 : 1] = true;
    if (r[0]) add(1, F::c5);
    if (r[1]) add(-1, F::c5);
  }
  const double rng = h[m] - l[m];
  if (m >= 30 && rng >= 2 * atr && v[m] >= quantile(std::vector<double>(v.begin(), v.begin() + m), 0.9)) {
    double mh = h[m - 29], ml = l[m - 29];
    for (int i = m - 29; i < m; ++i) { mh = std::max(mh, h[i]); ml = std::min(ml, l[i]); }
    if (h[m] > mh && c[m] <= l[m] + rng / 3.0) add(-1, F::c6);
    if (l[m] < ml && c[m] >= h[m] - rng / 3.0) add(1, F::c6);
  }
  if (any[0] == any[1]) return false;  // none, or both directions on one bar
  const int s = any[0] ? 0 : 1, d = any[0] ? 1 : -1;
  auto& f = out.f;
  f.fill(0.0);
  for (int k = 0; k < 8; ++k) f[k] = fl[s][k];
  const double A = atr;
  double sh = h[0], sl = l[0];
  for (int i = 0; i <= m; ++i) { sh = std::max(sh, h[i]); sl = std::min(sl, l[i]); }
  const auto& lis = fired[s];
  f[F::c3_count] = static_cast<double>(lis.size());
  f[F::c3_name] = -1.0;
  if (!lis.empty()) {
    int nearest = lis[0];
    for (int li : lis)
      if (std::fabs(c[m] - lv->v[li]) < std::fabs(c[m] - lv->v[nearest])) nearest = li;
    f[F::c3_name] = nearest;
  }
  f[F::dir_] = d;
  f[F::d_ext_entry] = (d == -1 ? sh - c[m] : c[m] - sl) / A;
  f[F::d_ext_far] = (d == -1 ? c[m] - sl : sh - c[m]) / A;
  f[F::d_open] = (c[m] - o[0]) / A;
  f[F::d_vwap] = (c[m] - vw) / A;
  f[F::vwap_z] = sd > 0 ? d * (c[m] - vw) / sd : kNaN;
  f[F::atr_] = A;
  f[F::range_atr] = rng / A;
  f[F::close_loc] = rng > 0 ? (c[m] - l[m]) / rng : kNaN;
  {
    double pv = 0.0;
    const int s0 = std::max(0, m - 30);
    for (int i = s0; i < m; ++i) pv += v[i];
    const double mean = m > s0 ? pv / (m - s0) : 0.0;
    f[F::relvol] = mean > 0 ? v[m] / mean : kNaN;
  }
  const int dk[4] = {1, 5, 15, 30};
  for (int q = 0; q < 4; ++q) {
    double vv = 0.0, bs = 0.0, ss = 0.0;
    for (int i = std::max(0, m - dk[q] + 1); i <= m; ++i) { vv += v[i]; bs += b.buy[i]; ss += b.sell[i]; }
    f[F::delta1 + q] = vv > 0 ? d * (bs - ss) / vv : kNaN;
  }
  const int rk[3] = {5, 15, 60};
  for (int q = 0; q < 3; ++q) f[F::ret5 + q] = d * (c[m] - (m - rk[q] >= 0 ? c[m - rk[q]] : o[0])) / A;
  f[F::ret_open] = d * (c[m] - o[0]) / A;
  f[F::slope5] = d * ema_slope(t5, m, 5, A);
  f[F::slope15] = d * ema_slope(t15, m, 15, A);
  f[F::slope60] = d * ema_slope(t60, m, 60, A);
  if (has_lv) {
    double dm = std::fabs(c[m] - lv->v[0]);
    for (int li = 1; li < 6; ++li) dm = std::min(dm, std::fabs(c[m] - lv->v[li]));
    f[F::d_level] = dm / A;
  } else {
    f[F::d_level] = kNaN;
  }
  double dmin = kNaN;
  int q = 0;
  for (const auto* gs : {&g5, &g15}) {
    int cnt = 0;
    for (const Fvg& g : *gs)
      if (g.m0 <= m && (g.closed < 0 || g.closed > m)) {
        ++cnt;
        const double dist = (g.bot <= c[m] && c[m] <= g.top) ? 0.0 : std::min(std::fabs(c[m] - g.bot), std::fabs(c[m] - g.top)) / A;
        if (!(dmin <= dist)) dmin = dist;
      }
    f[F::fvg5_n + q++] = cnt;
  }
  f[F::d_fvg] = dmin;
  f[F::minute_] = m;
  f[F::dow_] = dow;
  out.m = m;
  out.dir = d;
  return true;
}

// ---------------------------------------------------------------- the policy (actor and critic, 2 x 64 tanh)
struct Net {
  int in = 0, out = 0;
  std::vector<double> W1, b1, W2, b2, W3, b3;  // row-major: W1[kHidden x in], W2[kHidden x kHidden], W3[out x kHidden]
  void init(int in_, int out_) {
    in = in_; out = out_;
    W1.assign(static_cast<size_t>(kHidden) * in, 0.0); b1.assign(kHidden, 0.0);
    W2.assign(static_cast<size_t>(kHidden) * kHidden, 0.0); b2.assign(kHidden, 0.0);
    W3.assign(static_cast<size_t>(out) * kHidden, 0.0); b3.assign(out, 0.0);
  }
  struct Act { std::vector<double> h1, h2, y; };
  void forward(const double* x, Act& a) const {
    a.h1.assign(kHidden, 0.0); a.h2.assign(kHidden, 0.0); a.y.assign(out, 0.0);
    for (int i = 0; i < kHidden; ++i) {
      double s = b1[i];
      for (int j = 0; j < in; ++j) s += W1[static_cast<size_t>(i) * in + j] * x[j];
      a.h1[i] = std::tanh(s);
    }
    for (int i = 0; i < kHidden; ++i) {
      double s = b2[i];
      for (int j = 0; j < kHidden; ++j) s += W2[static_cast<size_t>(i) * kHidden + j] * a.h1[j];
      a.h2[i] = std::tanh(s);
    }
    for (int i = 0; i < out; ++i) {
      double s = b3[i];
      for (int j = 0; j < kHidden; ++j) s += W3[static_cast<size_t>(i) * kHidden + j] * a.h2[j];
      a.y[i] = s;
    }
  }
  // One SGD step along -dy (dy = dLoss/dy), the whole gradient clipped to norm `clip`.
  void sgd(const double* x, const Act& a, const std::vector<double>& dy, double lr, double clip) {
    std::vector<double> d2(kHidden, 0.0), d1(kHidden, 0.0);
    for (int j = 0; j < kHidden; ++j) {
      double s = 0.0;
      for (int i = 0; i < out; ++i) s += W3[static_cast<size_t>(i) * kHidden + j] * dy[i];
      d2[j] = s * (1.0 - a.h2[j] * a.h2[j]);
    }
    for (int j = 0; j < kHidden; ++j) {
      double s = 0.0;
      for (int i = 0; i < kHidden; ++i) s += W2[static_cast<size_t>(i) * kHidden + j] * d2[i];
      d1[j] = s * (1.0 - a.h1[j] * a.h1[j]);
    }
    double n2 = 0.0;
    for (int i = 0; i < out; ++i) { n2 += dy[i] * dy[i]; for (int j = 0; j < kHidden; ++j) { const double g = dy[i] * a.h2[j]; n2 += g * g; } }
    for (int i = 0; i < kHidden; ++i) { n2 += d2[i] * d2[i]; for (int j = 0; j < kHidden; ++j) { const double g = d2[i] * a.h1[j]; n2 += g * g; } }
    for (int i = 0; i < kHidden; ++i) { n2 += d1[i] * d1[i]; for (int j = 0; j < in; ++j) { const double g = d1[i] * x[j]; n2 += g * g; } }
    const double norm = std::sqrt(n2), k = lr * (norm > clip ? clip / norm : 1.0);
    if (!std::isfinite(norm)) return;  // a non-finite gradient changes nothing
    for (int i = 0; i < out; ++i) { b3[i] -= k * dy[i]; for (int j = 0; j < kHidden; ++j) W3[static_cast<size_t>(i) * kHidden + j] -= k * dy[i] * a.h2[j]; }
    for (int i = 0; i < kHidden; ++i) { b2[i] -= k * d2[i]; for (int j = 0; j < kHidden; ++j) W2[static_cast<size_t>(i) * kHidden + j] -= k * d2[i] * a.h1[j]; }
    for (int i = 0; i < kHidden; ++i) { b1[i] -= k * d1[i]; for (int j = 0; j < in; ++j) W1[static_cast<size_t>(i) * in + j] -= k * d1[i] * x[j]; }
  }
};

// Replaces `path` by `tmp` in one step where the platform can (set OF24_REPLACE_FILE to override, e.g. MoveFileExA).
inline bool replace_file(const std::string& tmp, const std::string& path) {
#ifdef OF24_REPLACE_FILE
  return OF24_REPLACE_FILE(tmp.c_str(), path.c_str());
#else
  std::remove(path.c_str());
  return std::rename(tmp.c_str(), path.c_str()) == 0;
#endif
}

struct Policy {
  std::array<double, kFeat> mu{}, sd{};
  Net pol, val;
  long long updates = 0;
  double lr = 1e-4, clip = 1.0, ent = 0.01;

  Policy() { pol.init(kIn, kActions); val.init(kIn, 1); sd.fill(1.0); }

  // The network input: standardized OF23 features (missing = 0), then the book features as given.
  std::array<double, kIn> input(const Cand& c, const std::array<double, kBook>& book) const {
    std::array<double, kIn> x{};
    for (int i = 0; i < kFeat; ++i) {
      const double z = (c.f[i] - mu[i]) / sd[i];
      x[i] = std::isfinite(z) ? z : 0.0;
    }
    for (int i = 0; i < kBook; ++i) x[kFeat + i] = std::isfinite(book[i]) ? book[i] : 0.0;
    return x;
  }
  struct Out { std::array<double, kActions> logit{}, p{}; double value = 0.0; };
  Out forward(const std::array<double, kIn>& x) const {
    Net::Act a;
    Out o;
    pol.forward(x.data(), a);
    double mx = a.y[0];
    for (int i = 0; i < kActions; ++i) { o.logit[i] = a.y[i]; mx = std::max(mx, a.y[i]); }
    double z = 0.0;
    for (int i = 0; i < kActions; ++i) { o.p[i] = std::exp(a.y[i] - mx); z += o.p[i]; }
    for (double& p : o.p) p /= z;
    val.forward(x.data(), a);
    o.value = a.y[0];
    return o;
  }
  static bool finite(const Out& o) {
    for (double p : o.p)
      if (!std::isfinite(p)) return false;
    return std::isfinite(o.value);
  }
  static int greedy(const Out& o) {
    if (!finite(o)) return 0;
    return static_cast<int>(std::max_element(o.logit.begin(), o.logit.end()) - o.logit.begin());
  }
  template <class Rng>
  static int sample(const Out& o, Rng& rng) {
    if (!finite(o)) return 0;  // a broken network skips
    double u = std::uniform_real_distribution<double>(0.0, 1.0)(rng);
    for (int i = 0; i < kActions; ++i) { u -= o.p[i]; if (u < 0) return i; }
    return 0;
  }
  // One actor-critic step for a finished decision: reward already scaled (net ticks / 10; 0 for a skip).
  void update(const std::array<double, kIn>& x, int action, double reward) {
    Net::Act ap, av;
    pol.forward(x.data(), ap);
    val.forward(x.data(), av);
    const double adv = reward - av.y[0];
    double mx = ap.y[0];
    for (double y : ap.y) mx = std::max(mx, y);
    std::vector<double> p(kActions), dy(kActions);
    double z = 0.0;
    for (int i = 0; i < kActions; ++i) { p[i] = std::exp(ap.y[i] - mx); z += p[i]; }
    double H = 0.0;
    for (double& q : p) { q /= z; if (q > 0) H -= q * std::log(q); }
    for (int i = 0; i < kActions; ++i) {
      const double lp = p[i] > 0 ? std::log(p[i]) : -745.0;
      dy[i] = adv * (p[i] - (i == action ? 1.0 : 0.0)) + ent * p[i] * (lp + H);  // d/dlogit of -adv*log p_a - ent*H
    }
    pol.sgd(x.data(), ap, dy, lr, clip);
    val.sgd(x.data(), av, {av.y[0] - reward}, lr, clip);
    ++updates;
  }

  // Text file: "OF24W 1", "in <kIn> hidden <kHidden> actions <kActions> updates <n>", then named arrays.
  bool save(const std::string& path) const {
    const std::string tmp = path + ".tmp";
    {
      std::ofstream f(tmp, std::ios::trunc);
      if (!f) return false;
      f.precision(17);
      f << "OF24W 1\nin " << kIn << " hidden " << kHidden << " actions " << kActions << " updates " << updates << "\n";
      auto put = [&](const char* name, const double* a, size_t n) {
        f << name << " " << n << "\n";
        for (size_t i = 0; i < n; ++i) f << a[i] << (i + 1 == n ? "\n" : " ");
      };
      put("mu", mu.data(), mu.size()); put("sd", sd.data(), sd.size());
      for (const auto* net : {&pol, &val}) {
        const char* p = net == &pol ? "pol" : "val";
        put((std::string(p) + "_W1").c_str(), net->W1.data(), net->W1.size()); put((std::string(p) + "_b1").c_str(), net->b1.data(), net->b1.size());
        put((std::string(p) + "_W2").c_str(), net->W2.data(), net->W2.size()); put((std::string(p) + "_b2").c_str(), net->b2.data(), net->b2.size());
        put((std::string(p) + "_W3").c_str(), net->W3.data(), net->W3.size()); put((std::string(p) + "_b3").c_str(), net->b3.data(), net->b3.size());
      }
      if (!f) return false;
    }
    return replace_file(tmp, path);
  }
  bool load(const std::string& path) {
    std::ifstream f(path);
    std::string tag, k1, k2, k3, k4;
    int ver = 0, in = 0, hid = 0, act = 0;
    long long upd = 0;
    if (!(f >> tag >> ver) || tag != "OF24W" || ver != 1) return false;
    if (!(f >> k1 >> in >> k2 >> hid >> k3 >> act >> k4 >> upd) || in != kIn || hid != kHidden || act != kActions) return false;
    Policy t;
    auto get = [&](const std::string& name, double* a, size_t n) {
      std::string nm;
      size_t cnt = 0;
      if (!(f >> nm >> cnt) || nm != name || cnt != n) return false;
      for (size_t i = 0; i < n; ++i)
        if (!(f >> a[i])) return false;
      return true;
    };
    if (!get("mu", t.mu.data(), t.mu.size()) || !get("sd", t.sd.data(), t.sd.size())) return false;
    for (auto* net : {&t.pol, &t.val}) {
      const std::string p = net == &t.pol ? "pol" : "val";
      if (!get(p + "_W1", net->W1.data(), net->W1.size()) || !get(p + "_b1", net->b1.data(), net->b1.size()) ||
          !get(p + "_W2", net->W2.data(), net->W2.size()) || !get(p + "_b2", net->b2.data(), net->b2.size()) ||
          !get(p + "_W3", net->W3.data(), net->W3.size()) || !get(p + "_b3", net->b3.data(), net->b3.size()))
        return false;
    }
    for (double s : t.sd)
      if (!(s > 0)) return false;
    mu = t.mu; sd = t.sd; pol = t.pol; val = t.val; updates = upd;
    return true;
  }
};

// OF23 base costs in ticks for a finished trade (market entry; limit target or market/stop exit).
inline double net_ticks(double move_ticks, bool exit_was_target) { return move_ticks - 0.64 - (exit_was_target ? 0.12 : 0.64); }

}  // namespace of24
