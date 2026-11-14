// RagAgent: the OF23/OF24 agent as a Sierra Chart study. SIMULATED ORDERS ONLY.
// Design: docs/ai/OF24_LIVE_SIM_PLAN.md. Kernel (candidates, features, policy, learning): include/rag/live/of24_kernel.hpp.
//
// Chart: ES front month, 1-minute bars, at least 2 days loaded. On every completed RTH bar (08:30-15:00 Chicago) the
// study rebuilds the day's bars, asks the kernel for a turning-point candidate at that bar, and when flat samples an
// action from the policy: skip, or a market entry with one of 12 attached target/stop brackets and a time cap. After
// each closed trade it applies one online learning step and saves the weights. It reads the depth book and the
// Market By Order queues at the decision for six book inputs (zero when absent, as in a replay).
//
// Safety (SierraChart-KB conventions): sc.SendOrdersToTradeService = 0 every call; nothing is sent unless Trade
// Simulation Mode is on, the Armed input is Yes, the selected trade account equals the required Sim account and the
// chart is 1-minute with a valid tick size. One lot, one position. A rejected order ends trading until recalculation.
//
// Build (never the same DLL name twice; no digit suffix):
//   powershell -File <SierraChart-KB>\build\build.ps1 -Source C:\rag-cpp\sierra\RagAgent.cpp -Name RagAgent_va_64 -Extra "/IC:\rag-cpp\include"
#include "sierrachart.h"

#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <share.h>
#include <string>

static bool of24_move(const char* tmp, const char* path) { return MoveFileExA(tmp, path, MOVEFILE_REPLACE_EXISTING) != 0; }
#define OF24_REPLACE_FILE of24_move
#include "rag/live/of24_kernel.hpp"

SCDLLName("RagAgent OF24 (SIM ONLY)")

EXTERN_C IMAGE_DOS_HEADER __ImageBase;

namespace {

constexpr int kOpenMin = 8 * 60 + 30;   // 08:30 Chicago
constexpr int kDayBars = 390;
constexpr int kMboCap = 512;

struct Pending {            // the open trade's decision, kept for the learning step
  bool live = false;
  std::array<double, of24::kIn> x{};
  int action = 0, dir = 0;
  int entry_m = 0;          // RTH minute index of the entry bar (m + 1)
  int date = 0;             // Chicago date (SCDateTime::GetDate) of the entry
  uint64_t entry_id = 0, stop_id = 0, target_id = 0;
  double best_close_ticks = -1e9;   // best completed-bar close in profit ticks (trailing exits)
  double stop_at_ticks = 0;         // current stop distance below entry in profit ticks (negative = loss side)
  bool flatten_sent = false;
  int flatten_try_bar = -1;         // the completed bar at which a flatten was last tried (one try per bar)
  bool entry_filled = false;        // the entry was seen filled (a position, or its fill)
  SCString account, symbol;         // the owner of the trade: exits act only there
};

struct State {
  of24::Policy policy;
  bool policy_ok = false;
  FILE* log = nullptr;
  std::mt19937_64 rng{0x0F24ULL};
  int last_done = -1;       // index of the last completed bar already handled
  bool ended = false;       // a rejected order: no more orders until recalculation
  int day = 0;              // Chicago date of the running totals
  double day_net = 0.0;     // net ticks today (OF23 base costs)
  int day_trades = 0;
  bool day_stopped = false;
  Pending t;
  n_ACSIL::s_MarketOrderData mbo[kMboCap];
  std::string weights_live;
  int skips_unsaved = 0;
  std::string ledger_trades, ledger_days;  // the profit log (append-only, survives restarts)
  double cum_ticks = 0.0;                  // cumulative net ticks over every run of this mode
  long long cum_trades = 0;
  bool chain_sent = false;                  // the replay-training end was reached and the replay stopped
  int jumps = 0;                           // completed bars skipped because several arrived at once
  int orphan_try_bar = -1;
  bool replay_mode = false;
  std::string weights_start;
};

void logf(State& s, const char* fmt, ...);

std::string ymd(int date) {
  int y = 0, m = 0, d = 0;
  SCDateTime(date, 0).GetDateYMD(y, m, d);
  char b[16];
  std::snprintf(b, sizeof b, "%04d-%02d-%02d", y, m, d);
  return b;
}

void append_line(const std::string& path, const char* header, const char* line) {
  FILE* f = _fsopen(path.c_str(), "a", _SH_DENYWR);
  if (!f) return;
  std::fseek(f, 0, SEEK_END);
  if (std::ftell(f) == 0) std::fputs(header, f);
  std::fputs(line, f);
  std::fclose(f);
}

// The day's totals to the day ledger, once, when the day is left behind (a new date, or the end of replay training).
void close_day(State& s) {
  if (s.day == 0 || s.day_trades == 0) return;
  char line[256];
  std::snprintf(line, sizeof line, "%s,%d,%.2f,%.2f,%.2f,%.2f,%lld\n", ymd(s.day).c_str(), s.day_trades, s.day_net, s.day_net * 12.5,
                s.cum_ticks, s.cum_ticks * 12.5, s.policy.updates);
  append_line(s.ledger_days, "date,trades,net_ticks,net_usd,cum_ticks,cum_usd,updates\n", line);
  s.day_trades = 0;
}

void save_weights(SCStudyInterfaceRef sc, State& s) {
  if (!s.policy.save(s.weights_live)) {
    logf(s, "E,0,0,weights save failed\n");
    sc.AddMessageToLog("RagAgent: saving the learned weights failed", 1);
  }
  s.skips_unsaved = 0;
}

void logf(State& s, const char* fmt, ...) {
  if (!s.log) return;
  va_list a;
  va_start(a, fmt);
  std::vfprintf(s.log, fmt, a);
  va_end(a);
  std::fflush(s.log);
}

struct ChiTime { int date; int minute; int dow; };  // Chicago date, minute of day, weekday (Monday = 0)

ChiTime chicago(SCStudyInterfaceRef sc, const SCDateTime& chart_time) {
  const SCDateTime t = sc.ConvertDateTimeFromChartTimeZone(chart_time, TIMEZONE_CHICAGO);
  return {t.GetDate(), t.GetTimeInSeconds() / 60, (t.GetDayOfWeek() + 6) % 7};
}

// The RTH bars of Chicago date `date` up to chart bar `last` (inclusive), one per minute from 08:30; a minute with
// no chart bar carries the last close with zero volume (a gap, as in tools/scid_to_bars.py). Prices in ticks.
bool day_bars(SCStudyInterfaceRef sc, int date, int last, of24::Bars& out, int& last_minute) {
  out.clear();
  int first = -1;
  for (int i = last; i >= 0; --i) {
    const ChiTime c = chicago(sc, sc.BaseDateTimeIn[i]);
    if (c.date < date) break;
    if (c.date == date && c.minute >= kOpenMin && c.minute < kOpenMin + kDayBars) first = i;
  }
  if (first < 0) return false;
  const double ts = sc.TickSize;
  double carry = std::round(sc.Open[first] / ts);
  int next = 0;
  for (int i = first; i <= last; ++i) {
    const ChiTime c = chicago(sc, sc.BaseDateTimeIn[i]);
    if (c.date != date) break;
    const int m = c.minute - kOpenMin;
    if (m < 0 || m >= kDayBars) continue;
    if (m < next) return false;  // two chart bars in one minute: not a 1-minute chart
    for (; next < m; ++next) out.push(carry, carry, carry, carry, 0, 0, 0);
    out.push(std::round(sc.Open[i] / ts), std::round(sc.High[i] / ts), std::round(sc.Low[i] / ts), std::round(sc.Close[i] / ts),
             sc.Volume[i], sc.AskVolume[i], sc.BidVolume[i]);
    carry = std::round(sc.Close[i] / ts);
    next = m + 1;
  }
  last_minute = next - 1;
  return out.size() > 0;
}

// The prior session's bars: the most recent earlier Chicago date with RTH bars in the chart.
bool prev_day_bars(SCStudyInterfaceRef sc, int date, int from, of24::Bars& out) {
  for (int i = from; i >= 0; --i) {
    const ChiTime c = chicago(sc, sc.BaseDateTimeIn[i]);
    if (c.date >= date || c.minute < kOpenMin || c.minute >= kOpenMin + kDayBars) continue;
    int lm = 0;
    if (!day_bars(sc, c.date, i, out, lm)) return false;
    int traded = 0;
    for (double v : out.v) traded += v > 0;
    return out.size() == kDayBars && traded >= kDayBars - 5;  // a full session only (OF18's rule: at most 5 gap minutes)
  }
  return false;
}

// Six book inputs, signed by the trade direction; zeros where the data is absent.
std::array<double, of24::kBook> book_features(SCStudyInterfaceRef sc, State& s, int dir, int& levels, int& mbo_orders) {
  std::array<double, of24::kBook> f{};
  levels = mbo_orders = 0;
  if (sc.ReplayStatus != REPLAY_STOPPED) return f;  // depth in a replay may be recorded; MBO never is: keep replays to bars
  const int nb = sc.GetBidMarketDepthNumberOfLevels(), na = sc.GetAskMarketDepthNumberOfLevels();
  levels = nb < na ? nb : na;
  if (levels < 1) return f;
  double qb[10] = {}, qa[10] = {};
  int tick_b = 0, tick_a = 0;
  for (int i = 0; i < 10 && i < levels; ++i) {
    s_MarketDepthEntry eb{}, ea{};
    if (!sc.GetBidMarketDepthEntryAtLevel(eb, i) || !sc.GetAskMarketDepthEntryAtLevel(ea, i)) { levels = i; break; }
    if (!(eb.AdjustedPrice > 0) || !(ea.AdjustedPrice > 0)) { levels = i; break; }
    qb[i] = eb.Quantity;
    qa[i] = ea.Quantity;
    if (i == 0) { tick_b = static_cast<int>(std::lround(eb.AdjustedPrice / sc.TickSize)); tick_a = static_cast<int>(std::lround(ea.AdjustedPrice / sc.TickSize)); }
  }
  if (levels < 1 || tick_a <= tick_b) return f;
  auto imb = [&](int n) {
    double b = 0, a = 0;
    for (int i = 0; i < n; ++i) { b += qb[i]; a += qa[i]; }
    return b + a > 0 ? dir * (b - a) / (b + a) : 0.0;
  };
  f[0] = imb(1);
  if (levels >= 5) f[1] = imb(5);
  if (levels >= 10) f[2] = imb(10);
  auto big_share = [&](bool bid) {
    const int n = bid ? sc.GetBidMarketLimitOrdersForPrice(tick_b, kMboCap, s.mbo) : sc.GetAskMarketLimitOrdersForPrice(tick_a, kMboCap, s.mbo);
    if (n <= 0 || n >= kMboCap) return 0.0;  // none, or possibly truncated
    mbo_orders += n;
    double all = 0, big = 0;
    for (int i = 0; i < n; ++i) { const double q = static_cast<double>(s.mbo[i].OrderQuantity); all += q; if (q >= 10) big += q; }
    return all > 0 ? big / all : 0.0;
  };
  const double sb = big_share(true), sa = big_share(false);
  f[3] = dir > 0 ? sb : sa;   // large resting orders behind the entry (support for a long, resistance for a short)
  f[4] = dir > 0 ? sa : sb;   // and on the far side
  f[5] = tick_a - tick_b;
  return f;
}

bool own_fill(SCStudyInterfaceRef sc, const s_SCOrderFillData& f) {
  return f.IsSimulated && f.TradeAccount == sc.SelectedTradeAccount && f.Symbol == sc.GetTradeSymbol();
}

}  // namespace

SCSFExport scsf_RagAgent(SCStudyInterfaceRef sc) {
  sc.SendOrdersToTradeService = 0;  // every call: simulated orders only, never routed
  SCInputRef InArmed = sc.Input[0];
  SCInputRef InAccount = sc.Input[1];
  SCInputRef InWeights = sc.Input[2];
  SCInputRef InLearn = sc.Input[3];
  SCInputRef InGreedy = sc.Input[4];
  SCInputRef InDayStop = sc.Input[5];
  SCInputRef InReplay = sc.Input[6];
  SCInputRef InUseBook = sc.Input[7];
  SCInputRef InChainEnd = sc.Input[8];
  if (sc.SetDefaults) {
    sc.GraphName = "RagAgent OF24 (turning points, RL exits) - SIM ONLY";
    sc.StudyDescription = "OF23 turning-point candidates on 1-minute RTH bars; a policy network picks skip or one of 12 brackets; "
                          "learns online from its simulated trades. Simulation accounts only. docs/ai/OF24_LIVE_SIM_PLAN.md";
    sc.AutoLoop = 0;
    sc.UpdateAlways = 1;
    sc.GraphRegion = 0;
    sc.UsesMarketDepthData = 1;
    sc.CalculationPrecedence = LOW_PREC_LEVEL;
    sc.MaximumPositionAllowed = 1;
    sc.AllowMultipleEntriesInSameDirection = 0;
    sc.SupportReversals = 0;
    sc.AllowOnlyOneTradePerBar = 1;
    sc.SupportAttachedOrdersForTrading = 1;
    sc.AllowEntryWithWorkingOrders = 0;
    sc.CancelAllOrdersOnEntriesAndReversals = 0;
    sc.MaintainTradeStatisticsAndTradesData = 1;
    InArmed.Name = "Armed (send SIM orders)";
    InArmed.SetYesNo(0);
    InAccount.Name = "Required simulation trade account";
    InAccount.SetString("Sim9");
    InWeights.Name = "Starting weights file (experiments/of24_export.py)";
    InWeights.SetString("C:\\rag-cpp\\build-avx512\\of24\\of24_weights.txt");
    InLearn.Name = "Learn online from closed sim trades";
    InLearn.SetYesNo(1);
    InGreedy.Name = "Greedy actions (No = sample from the policy)";
    InGreedy.SetYesNo(0);
    InDayStop.Name = "Daily stop (net ticks lost; 0 = none)";
    InDayStop.SetInt(120);
    InDayStop.SetIntLimits(0, 10000);
    InReplay.Name = "Replay test mode (Yes = orders ONLY during a chart replay, separate weights; No = live only)";
    InReplay.SetYesNo(0);
    InUseBook.Name = "Read depth and Market By Order for the book inputs";
    InUseBook.SetYesNo(1);
    InChainEnd.Name = "Replay training: last session (YYYYMMDD); the study stops the replay after it (0 = never)";
    InChainEnd.SetInt(20261111);  // days from 2026-11-12 are the forward block: a replay never trades them by default
    InChainEnd.SetIntLimits(0, 21001231);
    sc.Subgraph[0].Name = "Long entry";
    sc.Subgraph[0].DrawStyle = DRAWSTYLE_ARROW_UP;
    sc.Subgraph[0].PrimaryColor = RGB(0, 200, 0);
    sc.Subgraph[0].LineWidth = 3;
    sc.Subgraph[0].DrawZeros = 0;
    sc.Subgraph[1].Name = "Short entry";
    sc.Subgraph[1].DrawStyle = DRAWSTYLE_ARROW_DOWN;
    sc.Subgraph[1].PrimaryColor = RGB(220, 0, 0);
    sc.Subgraph[1].LineWidth = 3;
    sc.Subgraph[1].DrawZeros = 0;
    sc.Subgraph[2].Name = "Candidate skipped";
    sc.Subgraph[2].DrawStyle = DRAWSTYLE_POINT;
    sc.Subgraph[2].PrimaryColor = RGB(150, 150, 150);
    sc.Subgraph[2].LineWidth = 3;
    sc.Subgraph[2].DrawZeros = 0;
    return;
  }
  void*& slot = sc.GetPersistentPointer(1);
  if (sc.LastCallToFunction) {
    State* s = static_cast<State*>(slot);
    if (s) {
      if (s->t.live) sc.AddMessageToLog("RagAgent removed with an open SIM trade: its bracket stays, flatten it by hand", 1);
      if (s->skips_unsaved > 0) s->policy.save(s->weights_live);
      if (s->log) std::fclose(s->log);
      delete s;
    }
    slot = nullptr;
    return;
  }
  if (sc.IsFullRecalculation && sc.UpdateStartIndex == 0 && slot) {
    // inputs changed or the chart reloaded: with an open trade keep its identity and the policy and restart the bar
    // cursor; otherwise start over, so a changed mode or weights path takes effect
    State* r = static_cast<State*>(slot);
    if (r->t.live) {
      r->last_done = -1;
      r->ended = false;
    } else {
      if (r->skips_unsaved > 0) r->policy.save(r->weights_live);
      if (r->log) std::fclose(r->log);
      delete r;
      slot = nullptr;
    }
  }
  const bool replay_mode = slot ? static_cast<State*>(slot)->replay_mode : InReplay.GetYesNo() != 0;
  if (!slot) {
    State* s = new (std::nothrow) State();
    if (!s) return;
    slot = s;
    const SCString dir = sc.DataFilesFolder() + "RagAgent";
    CreateDirectoryA(dir.GetChars(), nullptr);
    char mod[MAX_PATH] = "?";
    GetModuleFileNameA(reinterpret_cast<HMODULE>(&__ImageBase), mod, MAX_PATH);
    const SCDateTime now_utc = sc.ConvertDateTimeFromChartTimeZone(sc.CurrentSystemDateTime, TIMEZONE_UTC);
    char path[MAX_PATH];
    std::snprintf(path, sizeof path, "%s\\%s-c%d-%lld.csv", dir.GetChars(), replay_mode ? "replay" : "live", sc.ChartNumber,
                  static_cast<long long>(now_utc.ToUNIXTimeInMicroseconds() / 1000000));
    s->log = _fsopen(path, "w", _SH_DENYWR);
    s->replay_mode = replay_mode;
    s->rng.seed(static_cast<uint64_t>(now_utc.ToUNIXTimeInMicroseconds()));
    char wname[64];
    std::snprintf(wname, sizeof wname, "\\weights_%s_c%d.txt", replay_mode ? "replay" : "live", sc.ChartNumber);
    s->weights_live = std::string(dir.GetChars()) + wname;
    // continue learning: this mode's learned weights; live falls back to what the replays learned; then the start
    const char* from = replay_mode ? "replay-learned" : "live-learned";
    s->policy_ok = s->policy.load(s->weights_live);
    if (!s->policy_ok && !replay_mode) {
      char rname[64];
      std::snprintf(rname, sizeof rname, "\\weights_replay_c%d.txt", sc.ChartNumber);
      s->policy_ok = s->policy.load(std::string(dir.GetChars()) + rname);
      from = "replay-learned";
    }
    if (!s->policy_ok) { s->policy_ok = s->policy.load(InWeights.GetString()); from = "start"; }
    s->ledger_trades = std::string(dir.GetChars()) + (replay_mode ? "\\profit_trades_replay.csv" : "\\profit_trades_live.csv");
    s->ledger_days = std::string(dir.GetChars()) + (replay_mode ? "\\profit_days_replay.csv" : "\\profit_days_live.csv");
    if (FILE* f = _fsopen(s->ledger_trades.c_str(), "r", _SH_DENYNO)) {  // the running totals from the last ledger line
      char line[512], last[512] = "";
      while (std::fgets(line, sizeof line, f)) std::snprintf(last, sizeof last, "%s", line);
      std::fclose(f);
      long long n = 0;
      double cum = 0;
      const char* c = last;
      for (int k = 0; k < 11 && c; ++k) { c = std::strchr(c, ','); if (c) ++c; }  // field 12 = cum_ticks, 13 = trade count
      if (c && std::sscanf(c, "%lf,%lld", &cum, &n) == 2) { s->cum_ticks = cum; s->cum_trades = n; }
    }
    logf(*s, "H,module=%s,symbol=%s,account=%s,tick=%g,seconds_per_bar=%d,replay_mode=%d,weights=%s,weights_ok=%d,updates=%lld,learn=%d,greedy=%d,day_stop=%d\n",
         mod, sc.Symbol.GetChars(), InAccount.GetString(), sc.TickSize, sc.SecondsPerBar, replay_mode ? 1 : 0, from, s->policy_ok ? 1 : 0,
         s->policy.updates, InLearn.GetYesNo(), InGreedy.GetYesNo(), InDayStop.GetInt());
    SCString msg;
    msg.Format("RagAgent: module %s, weights %s (%s), log %s", mod, from, s->policy_ok ? "ok" : "MISSING: no decisions", path);
    sc.AddMessageToLog(msg, s->policy_ok ? 0 : 1);
  }
  State& s = *static_cast<State*>(slot);
  if (!(sc.TickSize > 0) || sc.ArraySize < 3) return;
  const int cur = sc.ArraySize - 1, done = cur - 1;
  const bool replaying_now = sc.ReplayStatus == REPLAY_RUNNING;
  if (sc.IsFullRecalculation || s.last_done < 0) {  // never act on history: start from the bar forming now
    s.last_done = done;
    return;
  }

  s_SCPositionData pos;
  const bool have_pos = sc.GetTradePosition(pos) > 0;
  const bool flat_idle = have_pos && pos.PositionQuantity == 0 && !pos.WorkingOrdersExist;
  const bool account_ok = sc.SelectedTradeAccount.GetLength() > 0 && sc.SelectedTradeAccount.CompareNoCase(InAccount.GetString()) == 0;
  // replay mode: only while a replay is running; live mode: only with no replay, connected and not downloading
  const bool feed_ok = replay_mode ? sc.ReplayStatus == REPLAY_RUNNING
                                   : sc.ReplayStatus == REPLAY_STOPPED && sc.ServerConnectionState == SCS_CONNECTED &&
                                         !sc.ChartIsDownloadingHistoricalData(sc.ChartNumber);
  const bool sim_ok = sc.GlobalTradeSimulationIsOn == 1 && account_ok;
  const bool may_send = InArmed.GetYesNo() != 0 && sim_ok && sc.SecondsPerBar == 60 && feed_ok && s.policy_ok && !s.ended;
  const ChiTime now = chicago(sc, sc.BaseDateTimeIn[cur]);  // the forming bar's start: the study's clock (also in a replay)
  const double ts = sc.TickSize;

  // ---- 0. replay training is one continuous replay over many days: stop it after the last session, before any later
  // bar is acted on (restarting a replay from inside the study did not take: OF24 replay chain, 2026-11-13)
  const int chain_end = InChainEnd.GetInt();
  if (replay_mode && chain_end > 0 && replaying_now) {
    int y = 0, mo = 0, d = 0;
    SCDateTime(now.date, 0).GetDateYMD(y, mo, d);
    const int ymd_now = y * 10000 + mo * 100 + d;
    const bool after_day = ymd_now > chain_end;
    const bool after_close = ymd_now == chain_end && now.minute >= kOpenMin + kDayBars + 1 && !s.t.live && flat_idle;
    if (after_day || after_close) {
      if (!s.chain_sent) {
        int r = 0;
        if (sc.GlobalTradeSimulationIsOn == 1 && account_ok && have_pos && (pos.PositionQuantity != 0 || pos.WorkingOrdersExist))
          r = sc.FlattenAndCancelAllOrders();
        close_day(s);
        if (s.skips_unsaved > 0) save_weights(sc, s);
        s.chain_sent = true;
        s.ended = true;
        sc.StopChartReplay(sc.ChartNumber);
        logf(s, "C,%d,%d,replay training done,flatten=%d,jumps=%d,cum_ticks=%.2f,cum_trades=%lld\n", now.date, now.minute - kOpenMin, r,
             s.jumps, s.cum_ticks, s.cum_trades);
        sc.AddMessageToLog("RagAgent: replay training reached its last session and stopped the replay (see profit_days_replay.csv)", 0);
      }
      return;
    }
  }

  // ---- 1. the open trade: closed? time cap? trailing stop?
  const bool owner_ok = sc.GlobalTradeSimulationIsOn == 1 && s.t.live && sc.SelectedTradeAccount == s.t.account && sc.GetTradeSymbol() == s.t.symbol;
  if (s.t.live && !owner_ok) {
    // the chart's account or symbol changed, or simulation was switched off: touch nothing
  } else if (s.t.live) {
    if (!s.t.entry_filled) {
      if (have_pos && pos.PositionQuantity * s.t.dir > 0) s.t.entry_filled = true;
      for (int i = sc.GetOrderFillArraySize() - 1; i >= 0 && !s.t.entry_filled; --i) {
        s_SCOrderFillData f;
        if (sc.GetOrderFillEntry(i, f) && own_fill(sc, f) && f.InternalOrderID == s.t.entry_id) s.t.entry_filled = true;
      }
      if (!s.t.entry_filled) {
        s_SCTradeOrder od;
        if (sc.GetOrderByOrderID(s.t.entry_id, od) != SCTRADING_ORDER_ERROR &&
            (od.OrderStatusCode == SCT_OSC_CANCELED || od.OrderStatusCode == SCT_OSC_ERROR)) {
          logf(s, "E,%d,%d,entry order ended unfilled (status %d)\n", s.t.date, s.t.entry_m, static_cast<int>(od.OrderStatusCode));
          s.t = Pending{};
        }
      }
    }
    if (!s.t.live || !s.t.entry_filled) {
      // nothing to manage yet
    } else if (flat_idle) {
      // closed: the entry fill and the last opposite fill after it
      s_SCOrderFillData fe, fx;
      bool got_e = false, got_x = false;
      for (int i = sc.GetOrderFillArraySize() - 1; i >= 0 && !(got_e && got_x); --i) {
        s_SCOrderFillData f;
        if (!sc.GetOrderFillEntry(i, f) || !own_fill(sc, f)) continue;
        if (!got_e && f.InternalOrderID == s.t.entry_id) { fe = f; got_e = true; }
        else if (!got_x && !got_e && f.BuySell == (s.t.dir > 0 ? BSE_SELL : BSE_BUY)) { fx = f; got_x = true; }
      }
      if (got_e && got_x) {
        const double move = s.t.dir * (fx.FillPrice - fe.FillPrice) / ts;
        const bool was_target = fx.InternalOrderID == s.t.target_id;
        const double net = of24::net_ticks(move, was_target);
        if (s.t.date != s.day) { close_day(s); s.day = s.t.date; s.day_net = 0; s.day_trades = 0; s.day_stopped = false; }
        s.day_net += net;
        ++s.day_trades;
        s.cum_ticks += net;
        ++s.cum_trades;
        {
          const of24::ExitSpec ex = of24::kMenu[s.t.action - 1];
          const int em = kOpenMin + s.t.entry_m;
          char line[400];
          std::snprintf(line, sizeof line, "%s,%02d:%02d,%s,T%d/S%d/%s%s,%.2f,%.2f,%.2f,%s,%.2f,%.2f,%.2f,%.2f,%lld,%.2f,%lld\n",
                        ymd(s.t.date).c_str(), em / 60, em % 60, s.t.dir > 0 ? "long" : "short", ex.target, ex.stop,
                        ex.hold_min ? std::to_string(ex.hold_min).c_str() : "EOD", ex.trailing ? "/trail" : "", fe.FillPrice, fx.FillPrice,
                        move, was_target ? "target" : fx.InternalOrderID == s.t.stop_id ? "stop" : "time", net, net * 12.5, s.day_net,
                        s.cum_ticks, s.cum_trades, s.cum_ticks * 12.5, s.policy.updates + (InLearn.GetYesNo() ? 1 : 0));
          append_line(s.ledger_trades,
                      "date,entry_ct,side,exit_rule,entry_px,exit_px,move_ticks,exit,net_ticks,net_usd,day_net_ticks,cum_ticks,cum_trades,cum_usd,updates\n",
                      line);
        }
        if (InLearn.GetYesNo()) {
          s.policy.update(s.t.x, s.t.action, net / 10.0);
          save_weights(sc, s);
        }
        const int stop_lim = InDayStop.GetInt();
        if (stop_lim > 0 && s.day_net <= -stop_lim) s.day_stopped = true;
        logf(s, "T,%d,%d,%d,%d,%.2f,%.2f,%.2f,%s,%.2f,%.2f,%d,%lld,%d\n", s.t.date, s.t.entry_m, s.t.dir, s.t.action, fe.FillPrice, fx.FillPrice, move,
             was_target ? "target" : fx.InternalOrderID == s.t.stop_id ? "stop" : "time", net, s.day_net, s.day_trades, s.policy.updates, s.day_stopped ? 1 : 0);
      } else {
        logf(s, "E,%d,%d,closed without matching fills (entry %d exit %d)\n", s.t.date, s.t.entry_m, got_e ? 1 : 0, got_x ? 1 : 0);
      }
      s.t = Pending{};
    } else if (!s.t.flatten_sent && s.t.flatten_try_bar != done && have_pos && pos.PositionQuantity * s.t.dir > 0) {
      const of24::ExitSpec ex = of24::kMenu[s.t.action - 1];
      const int m_now = now.date == s.t.date ? now.minute - kOpenMin : kDayBars;  // a later date: past the close
      const bool cap = m_now >= kDayBars || (ex.hold_min > 0 && m_now >= s.t.entry_m + ex.hold_min);
      const bool disarmed = InArmed.GetYesNo() == 0;
      if (cap || disarmed) {
        const int r = sc.FlattenAndCancelAllOrders();  // owner_ok: simulation on, the trade's own account and symbol
        s.t.flatten_try_bar = done;                    // a failed flatten is retried at the next completed bar
        if (r > 0) s.t.flatten_sent = true;
        else sc.AddMessageToLog("RagAgent: the time-cap flatten failed; retrying at the next bar", 1);
        logf(s, "X,%d,%d,%s,%d\n", s.t.date, m_now, disarmed ? "disarmed" : "time", r);
      }
    }
  } else if (sim_ok && have_pos && pos.PositionQuantity != 0 && s.orphan_try_bar != done &&
             (now.minute >= kOpenMin + kDayBars || now.minute < kOpenMin)) {
    // a position on the agent's own Sim account that this run does not track (a restart, a lost entry): out after the close
    const int r = sc.FlattenAndCancelAllOrders();
    s.orphan_try_bar = done;
    logf(s, "X,%d,%d,orphan,%d\n", now.date, now.minute - kOpenMin, r);
  }

  // ---- 2. a bar completed
  if (done > s.last_done) {
    const bool single = done == s.last_done + 1;  // several at once (a reload, a jump): no decision
    s.last_done = done;
    const ChiTime bt = chicago(sc, sc.BaseDateTimeIn[done]);
    const int m = bt.minute - kOpenMin;
    // trailing exits: OF23's rule on completed closes (stop to best close - S once the best close is >= S in profit)
    if (s.t.live && owner_ok && s.t.entry_filled && !s.t.flatten_sent && s.t.stop_id && of24::kMenu[s.t.action - 1].trailing && bt.date == s.t.date && m >= s.t.entry_m) {
      s_SCOrderFillData f;
      for (int i = sc.GetOrderFillArraySize() - 1; i >= 0; --i) {
        if (!sc.GetOrderFillEntry(i, f) || !own_fill(sc, f) || f.InternalOrderID != s.t.entry_id) continue;
        const int S = of24::kMenu[s.t.action - 1].stop;
        const double prof = s.t.dir * (sc.Close[done] - f.FillPrice) / ts;
        if (prof > s.t.best_close_ticks) s.t.best_close_ticks = prof;
        const double want = s.t.best_close_ticks - S;  // stop level in profit ticks
        if (s.t.best_close_ticks >= S && want > s.t.stop_at_ticks) {
          s_SCNewOrder mod;
          mod.InternalOrderID = s.t.stop_id;
          mod.Price1 = std::round((f.FillPrice + s.t.dir * want * ts) / ts) * ts;
          const int r = sc.ModifyOrder(mod);
          if (r > 0) s.t.stop_at_ticks = want;
          logf(s, "M,%d,%d,%.2f,%d\n", bt.date, m, mod.Price1, r);
        }
        break;
      }
    }
    if (bt.date != s.day) { close_day(s); s.day = bt.date; s.day_net = 0; s.day_trades = 0; s.day_stopped = false; }
    if (!single) ++s.jumps;
    // the new bar must be the very next minute, so the entry is at open(m + 1) as in OF23
    const bool next_ok = now.date == bt.date && now.minute == bt.minute + 1;
    if (single && next_ok && m >= of24::kFirstM && m <= of24::kLastM && s.policy_ok) {
      of24::Bars day, prev;
      int lm = 0;
      of24::Cand c;
      if (day_bars(sc, bt.date, done, day, lm) && lm == m) {
        const of24::Levels lv = prev_day_bars(sc, bt.date, done, prev) ? of24::prior_levels(prev) : of24::Levels{};
        if (of24::candidate_at(day, lv.ok ? &lv : nullptr, bt.dow, c) && !s.t.live && flat_idle) {
          int levels = 0, mbo_n = 0;
          std::array<double, of24::kBook> book{};
          if (InUseBook.GetYesNo()) book = book_features(sc, s, c.dir, levels, mbo_n);
          const auto x = s.policy.input(c, book);
          const auto o = s.policy.forward(x);
          if (!of24::Policy::finite(o)) {
            s.policy_ok = false;
            sc.AddMessageToLog("RagAgent: the policy gave a non-finite output; decisions stopped", 1);
            logf(s, "E,%d,%d,non-finite policy output\n", bt.date, m);
            return;
          }
          int a = InGreedy.GetYesNo() ? of24::Policy::greedy(o) : of24::Policy::sample(o, s.rng);
          const char* why = "ok";
          if (!may_send) { why = "not armed or guards"; }
          else if (s.day_stopped) { why = "daily stop"; }
          int r = 0;
          s_SCNewOrder ord;
          if (a > 0 && may_send && !s.day_stopped) {
            const of24::ExitSpec ex = of24::kMenu[a - 1];
            ord.OrderType = SCT_ORDERTYPE_MARKET;
            ord.OrderQuantity = 1;
            ord.Symbol = sc.GetTradeSymbol();
            ord.TradeAccount = sc.SelectedTradeAccount;
            ord.TextTag = "RagAgentOF24";
            ord.Target1Offset = ex.target * ts;
            ord.Stop1Offset = ex.stop * ts;
            ord.AttachedOrderTarget1Type = SCT_ORDERTYPE_LIMIT;
            ord.AttachedOrderStop1Type = SCT_ORDERTYPE_STOP;
            r = static_cast<int>(c.dir > 0 ? sc.BuyEntry(ord, cur) : sc.SellEntry(ord, cur));
            if (r > 0) {
              s.t = Pending{};
              s.t.live = true;
              s.t.x = x;
              s.t.action = a;
              s.t.dir = c.dir;
              s.t.entry_m = m + 1;
              s.t.date = bt.date;
              s.t.entry_id = ord.InternalOrderID;
              s.t.stop_id = ord.Stop1InternalOrderID;
              s.t.target_id = ord.Target1InternalOrderID;
              s.t.stop_at_ticks = -ex.stop;
              s.t.account = sc.SelectedTradeAccount;
              s.t.symbol = sc.GetTradeSymbol();
              sc.Subgraph[c.dir > 0 ? 0 : 1][cur] = static_cast<float>(c.dir > 0 ? sc.Low[done] - 2 * ts : sc.High[done] + 2 * ts);
            } else if (r <= SCT_SKIPPED_AUTO_TRADING_DISABLED) {
              why = "skipped by Sierra";  // one trade per bar, auto trading off, downloading: not an error, no retry on this bar
            } else {
              s.ended = true;  // a rejected entry: stop until the study is recalculated (never retry each call)
              why = "rejected";
            }
          } else if (a == 0 && may_send && !s.day_stopped) {
            if (InLearn.GetYesNo()) {
              s.policy.update(x, 0, 0.0);
              if (++s.skips_unsaved >= 20) save_weights(sc, s);  // skips are saved in batches; trades at once
            }
            sc.Subgraph[2][done] = static_cast<float>(c.dir > 0 ? sc.Low[done] - ts : sc.High[done] + ts);
          }
          // D,date,m,dir,action,p(action),value,families(c1_1 c1_5 c1_15 c2 c3 c4 c5 c6),book x6,depth levels,mbo orders,order result,why
          logf(s, "D,%d,%d,%d,%d,%.4f,%.4f,%g%g%g%g%g%g%g%g,%.3f,%.3f,%.3f,%.3f,%.3f,%.0f,%d,%d,%d,%s\n", bt.date, m, c.dir, a, o.p[a], o.value,
               c.f[0], c.f[1], c.f[2], c.f[3], c.f[4], c.f[5], c.f[6], c.f[7], book[0], book[1], book[2], book[3], book[4], book[5], levels, mbo_n, r, why);
        }
      }
    }
  }
}
