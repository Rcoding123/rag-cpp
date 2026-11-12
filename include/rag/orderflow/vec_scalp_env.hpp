// rag/orderflow/vec_scalp_env.hpp — N ScalpEnv episodes stepped together on
// a thread pool (OF15: the batched environment behind the `_ragof` Python
// module, python/ragof_module.cpp), episodes named only through the seed
// ledger (seed_ledger.hpp), plus the C++ built-in policies (flat, hold, rules,
// the fair-value oracle, an of_train network) evaluated with no Python loop.
//
// Same episodes, observations and rewards as ScalpEnv, bit for bit:
//   * EpisodeSource::train(s): episode k is the k-th reset on one Rng(s)
//     stream — of_train's Trainer stream for run seed s — whatever n_envs is:
//     slot j of round r is stream episode r * n_envs + j. The draws happen
//     serially (copy the stream, advance it with ScalpEnv::draw_market, the
//     only thing reset() draws) and every reset checks it consumed exactly
//     those draws, so a change to ScalpEnv::reset fails loudly here.
//   * EpisodeSource::indexed(name, first, count): slot i = Rng(base + first + i),
//     as rag::agent::evaluate and of_train's score(); one round, n_envs = count,
//     so every policy evaluated on a source plays the same episodes (paired).
//   * days (OF8: Rng(930M + d * 100 + k), start at day_session_start(k)) and
//     stress (OF6: Rng(920M + i) with a scenario's change).
// Results don't depend on the thread count. Every ScalpEnv episode has the
// same number of decisions, so all slots finish on the same step (checked).
//
// Privileged outputs (hidden fair value, ...) go to a separate buffer, only
// when asked for at construction and only on blocks with privileged_ok
// (train, viz). Observations are ScalpEnv::features() verbatim.
#pragma once

#include <algorithm>
#include <atomic>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <exception>
#include <functional>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "rag/agent/mlp.hpp"
#include "rag/orderflow/replay_env.hpp"
#include "rag/orderflow/scalp_env.hpp"
#include "rag/orderflow/scalp_run.hpp"
#include "rag/orderflow/seed_ledger.hpp"

namespace rag::orderflow {

// ------------------------------------------------------------------ thread pool
// A persistent pool: run(n, f) calls f(0..n-1) on the workers and the caller
// (an atomic work index), returns when all are done, and rethrows the first
// exception a call threw.
class Pool {
 public:
  explicit Pool(unsigned threads) {
    for (unsigned t = 1; t < threads; ++t) ws_.emplace_back([this] { loop(); });
  }
  Pool(const Pool&) = delete;
  Pool& operator=(const Pool&) = delete;
  ~Pool() {
    {
      std::lock_guard l(m_);
      stop_ = true;
      ++gen_;
    }
    cv_.notify_all();
    for (auto& w : ws_) w.join();
  }
  [[nodiscard]] unsigned threads() const noexcept { return static_cast<unsigned>(ws_.size()) + 1; }

  void run(std::size_t n, const std::function<void(std::size_t)>& f) {
    err_ = nullptr;
    if (ws_.empty() || n < 2) {
      for (std::size_t i = 0; i < n; ++i) f(i);
      return;
    }
    {
      std::lock_guard l(m_);
      f_ = &f;
      n_ = n;
      next_ = 0;
      busy_ = ws_.size();
      ++gen_;
    }
    cv_.notify_all();
    work();
    {
      std::unique_lock l(m_);
      done_cv_.wait(l, [&] { return busy_ == 0; });
    }
    if (err_) std::rethrow_exception(err_);
  }

 private:
  void work() {
    for (std::size_t i; (i = next_.fetch_add(1)) < n_;) {
      try {
        (*f_)(i);
      } catch (...) {
        std::lock_guard l(em_);
        if (!err_) err_ = std::current_exception();
      }
    }
  }
  void loop() {
    std::uint64_t seen = 0;
    for (;;) {
      {
        std::unique_lock l(m_);
        cv_.wait(l, [&] { return gen_ != seen; });
        seen = gen_;
        if (stop_) return;
      }
      work();
      std::lock_guard l(m_);
      if (--busy_ == 0) done_cv_.notify_one();
    }
  }
  std::vector<std::thread> ws_;
  std::mutex m_, em_;
  std::condition_variable cv_, done_cv_;
  const std::function<void(std::size_t)>* f_ = nullptr;
  std::size_t n_ = 0, busy_ = 0;
  std::atomic<std::size_t> next_{0};
  std::uint64_t gen_ = 0;
  bool stop_ = false;
  std::exception_ptr err_;
};

// ------------------------------------------------------------------ run parameters
// ScalpParams plus the market preset's name (network records, stress lists)
// and the simulator overrides applied on top of the preset (a whitelist).
struct RunParams {
  std::string market = "default";
  ScalpParams p;
  std::map<std::string, double> sim_overrides;
};

// The SimParams fields a run may override (all doubles, validated by EsSimulator).
inline const std::vector<std::pair<const char*, double SimParams::*>>& sim_override_fields() {
  static const std::vector<std::pair<const char*, double SimParams::*>> f = {
      {"value_vol", &SimParams::value_vol},
      {"jump_rate", &SimParams::jump_rate},
      {"jump_sd", &SimParams::jump_sd},
      {"noise_rate", &SimParams::noise_rate},
      {"informed_meta_rate", &SimParams::informed_meta_rate},
      {"uninformed_meta_rate", &SimParams::uninformed_meta_rate},
      {"momentum_rate", &SimParams::momentum_rate},
      {"maker_rate", &SimParams::maker_rate},
      {"cancel_rate", &SimParams::cancel_rate},
      {"fade", &SimParams::fade},
      {"reprice_cancel", &SimParams::reprice_cancel},
      {"reprice_take", &SimParams::reprice_take},
      {"value_impact", &SimParams::value_impact},
      {"sweep_follow", &SimParams::sweep_follow},
  };
  return f;
}

// Builds and validates a run's parameters (market preset, warm-up, ScalpEnv's
// guards, the simulator's own checks). Throws std::invalid_argument.
inline RunParams run_params(const std::string& market, double warmup_seconds, double decide_seconds = 1.0, double episode_seconds = 600,
                            double start_second = -1, std::uint32_t latency_steps = 1, std::uint32_t contracts = 1, double fee_dollars = 1.5,
                            const std::map<std::string, double>& sim_overrides = {}) {
  RunParams r;
  r.market = market;
  r.p.sim = market_preset(market);
  if (!std::isfinite(warmup_seconds) || warmup_seconds < 0) throw std::invalid_argument("warmup_seconds must be a finite number >= 0");
  if (!std::isfinite(decide_seconds) || decide_seconds <= 0) throw std::invalid_argument("decide_seconds must be > 0");
  if (!std::isfinite(episode_seconds) || episode_seconds < decide_seconds) throw std::invalid_argument("episode_seconds must be >= decide_seconds");
  if (!std::isfinite(start_second)) throw std::invalid_argument("start_second must be finite (< 0: random)");
  if (!std::isfinite(fee_dollars) || fee_dollars < 0) throw std::invalid_argument("fee_dollars must be >= 0");
  if (contracts < 1) throw std::invalid_argument("contracts must be >= 1");
  r.p.warmup_seconds = warmup_seconds;
  r.p.decide_seconds = decide_seconds;
  r.p.episode_seconds = episode_seconds;
  r.p.start_second = start_second;
  r.p.latency_steps = latency_steps;
  r.p.contracts = contracts;
  r.p.fee_dollars = fee_dollars;
  for (const auto& [k, v] : sim_overrides) {
    const auto& f = sim_override_fields();
    const auto it = std::find_if(f.begin(), f.end(), [&](const auto& e) { return k == e.first; });
    if (it == f.end()) {
      std::string names;
      for (const auto& e : f) names += std::string(names.empty() ? "" : ", ") + e.first;
      throw std::invalid_argument("sim_overrides: '" + k + "' is not overridable (" + names + ")");
    }
    if (!std::isfinite(v)) throw std::invalid_argument("sim_overrides: '" + k + "' must be finite");
    r.p.sim.*(it->second) = v;
  }
  r.sim_overrides = sim_overrides;
  if (const std::string why = warmup_error(r.p); !why.empty()) throw std::invalid_argument(why);
  (void)ScalpEnv(r.p);                  // contracts > 1 guards
  (void)EsSimulator(r.p.sim, 1, 0.0);   // the simulator's parameter checks
  return r;
}

inline bool operator==(const RunParams& a, const RunParams& b) {
  const ScalpParams &x = a.p, &y = b.p;
  return a.market == b.market && a.sim_overrides == b.sim_overrides && x.decide_seconds == y.decide_seconds && x.episode_seconds == y.episode_seconds &&
         x.warmup_seconds == y.warmup_seconds && x.start_second == y.start_second && x.latency_steps == y.latency_steps && x.contracts == y.contracts &&
         x.fee_dollars == y.fee_dollars;
}

// Decisions per ScalpEnv episode (ScalpEnv::step's own clock).
[[nodiscard]] inline std::uint32_t steps_per_episode(const ScalpParams& p) {
  double elapsed = 0;
  for (std::uint32_t n = 1;; ++n) {
    elapsed += p.decide_seconds;
    if (elapsed >= p.episode_seconds - 1e-9) return n;
  }
}

// ------------------------------------------------------------------ episode sources
// The only way to name episodes: a train stream (run seed) or block-relative
// indices. Sealed blocks must be open in the process (seeds::Ledger).
struct EpisodeSource {
  enum class Kind { Train, Indexed, Days, Stress };
  Kind kind = Kind::Train;
  seeds::Block block = seeds::Block::Train;
  std::uint64_t run_seed = 0;          // train
  std::uint64_t first = 0, count = 0;  // indexed / stress: episodes; days: first day and slots
  std::uint32_t n_days = 0, sessions_per_day = 0;
  std::string scenario, list, orchestrator;

  static EpisodeSource train(std::uint64_t run_seed) {
    const auto& bi = seeds::info(seeds::Block::Train);
    if (run_seed < bi.base || run_seed - bi.base >= bi.size)
      throw std::invalid_argument("train run_seed must be in [1, 700,000,000) (validation and sealed blocks start at 700M)");
    EpisodeSource s;
    s.run_seed = run_seed;
    return s;
  }
  static EpisodeSource indexed(const std::string& name, std::uint64_t first, std::uint64_t count, const std::string& orchestrator = {},
                             const seeds::Ledger& L = seeds::Ledger::process()) {
    EpisodeSource s;
    s.kind = Kind::Indexed;
    s.block = seeds::block_of(name);
    const auto& bi = seeds::info(s.block);
    if (!bi.indexed)
      throw std::invalid_argument(std::string("block ") + bi.name + " has no indexed episodes (train(run_seed), days(...), stress(...))");
    seeds::check_range(s.block, first, count);
    if (bi.needs_claim && orchestrator.empty())
      throw std::invalid_argument("the confirm block (950M+) is the orchestrator's: it needs orchestrator=<claim id>, once per claimed edge");
    if (!bi.needs_claim && !orchestrator.empty()) throw std::invalid_argument("orchestrator= is only for the confirm block");
    L.require_open(s.block);
    s.first = first;
    s.count = count;
    s.orchestrator = orchestrator;
    return s;
  }
  // OF8's days: day d session k = Rng(930M + d * 100 + k), warm-up starting at day_session_start(k).
  static EpisodeSource days(std::uint32_t first_day, std::uint32_t n_days, std::uint32_t sessions_per_day,
                            const seeds::Ledger& L = seeds::Ledger::process()) {
    if (n_days == 0 || sessions_per_day == 0 || sessions_per_day > 39) throw std::invalid_argument("days: n_days >= 1, sessions_per_day in [1, 39]");
    if (std::uint64_t{first_day} + n_days > 100'000) throw std::invalid_argument("days: day indices must be < 100,000 (seed block days is 10M)");
    L.require_open(seeds::Block::Days);
    EpisodeSource s;
    s.kind = Kind::Days;
    s.block = seeds::Block::Days;
    s.first = first_day;
    s.n_days = n_days;
    s.sessions_per_day = sessions_per_day;
    s.count = std::uint64_t{n_days} * sessions_per_day;
    return s;
  }
  // OF6's stress episodes Rng(920M + i) under a scenario of stress_scenarios(); list "legacy" or
  // "discovery", or "" for the market's own (refused on discovery_v4, which has no pre-registered list).
  static EpisodeSource stress(const std::string& scenario, std::uint64_t first, std::uint64_t count, const std::string& list = {},
                              const seeds::Ledger& L = seeds::Ledger::process()) {
    if (!list.empty() && list != "legacy" && list != "discovery") throw std::invalid_argument("stress list must be 'legacy' or 'discovery'");
    seeds::check_range(seeds::Block::Stress, first, count);
    L.require_open(seeds::Block::Stress);
    EpisodeSource s;
    s.kind = Kind::Stress;
    s.block = seeds::Block::Stress;
    s.first = first;
    s.count = count;
    s.scenario = scenario;
    s.list = list;
    return s;
  }

  [[nodiscard]] bool is_train() const noexcept { return kind == Kind::Train; }
  // Slot j's episode seed (indexed kinds).
  [[nodiscard]] std::uint64_t seed_of(std::uint64_t j) const {
    switch (kind) {
      case Kind::Indexed:
      case Kind::Stress:
        return seeds::info(block).base + first + j;
      case Kind::Days:
        return seeds::info(block).base + (first + j / sessions_per_day) * 100 + j % sessions_per_day;
      case Kind::Train:
        break;
    }
    throw std::logic_error("seed_of: train episodes come from the run stream");
  }
  // Slot j's parameters (days: the session start; stress: the scenario's change).
  [[nodiscard]] ScalpParams params_of(const RunParams& r, std::uint64_t j) const {
    ScalpParams p = r.p;
    if (kind == Kind::Days) p.start_second = day_session_start(static_cast<unsigned>(j % sessions_per_day), r.p.warmup_seconds);
    if (kind == Kind::Stress) stress_scenario(r).change(p);
    return p;
  }
  [[nodiscard]] StressScenario stress_scenario(const RunParams& r) const {
    std::string l = list;
    if (l.empty()) {
      if (r.market == "discovery_v4")
        throw std::invalid_argument("stress on discovery_v4: no pre-registered v4 stress list; name list='legacy' or 'discovery' explicitly");
      l = r.market == "discovery" ? "discovery" : "legacy";
    }
    for (auto& sc : stress_scenarios(l == "discovery"))
      if (sc.name == scenario) return sc;
    throw std::invalid_argument("stress scenario '" + scenario + "' is not in the " + l + " list");
  }
};

// ------------------------------------------------------------------ built-in policies
// of2_baselines' strategies, of_train's delta rule and of_train networks,
// acting on the observation (and, for the oracle, the hidden fair value: it
// stays in C++).
struct BuiltinPolicy {
  enum class Kind { Flat, Hold, Rule, Oracle, NeuralQ };
  Kind kind = Kind::Flat;
  std::size_t feature = 0;
  double threshold = 0;
  bool reverse = false;
  std::shared_ptr<const rag::agent::Mlp> net;

  static BuiltinPolicy flat() { return {}; }
  static BuiltinPolicy hold() { return {Kind::Hold, 0, 0, false, nullptr}; }
  static BuiltinPolicy rule(std::size_t k, double th, bool rev) {
    if (k >= kFeatures) throw std::invalid_argument("rule: feature index must be < 13");
    return {Kind::Rule, k, th, rev, nullptr};
  }
  static BuiltinPolicy delta30() { return rule(1, 1.5, false); }  // of_train's 30 s delta rule
  static BuiltinPolicy oracle(double th) { return {Kind::Oracle, 0, th, false, nullptr}; }
  static BuiltinPolicy neuralq(std::shared_ptr<const rag::agent::Mlp> n) {
    if (!n || n->inputs() != ScalpEnv::kStateDims || n->outputs() != ScalpEnv::kNumActions)
      throw std::invalid_argument("neuralq: the network must map 16 inputs to 3 Q-values");
    return {Kind::NeuralQ, 0, 0, false, std::move(n)};
  }

  // Keep the position: short, flat or long as now.
  [[nodiscard]] static rag::agent::Action keep(const ScalpObs& o) { return o.position > 0 ? 2 : o.position < 0 ? 0 : 1; }
  // scratch: net->scratch_size() doubles (NeuralQ only).
  [[nodiscard]] rag::agent::Action act(const ScalpObs& o, const ScalpEnv& e, double* scratch) const {
    switch (kind) {
      case Kind::Flat:
        return 1;
      case Kind::Hold:
        return keep(o);
      case Kind::Rule: {
        const double x = reverse ? -o.f[feature] : o.f[feature];
        return x > threshold ? 2 : x < -threshold ? 0 : keep(o);
      }
      case Kind::Oracle: {
        const double gap = e.market().fair_value() - e.market().mid();
        return gap > threshold ? 2 : gap < -threshold ? 0 : keep(o);
      }
      case Kind::NeuralQ: {
        const auto x = e.features(o);
        double q[ScalpEnv::kNumActions];
        net->forward(x.data(), q, scratch);
        return greedy(q);
      }
    }
    return 1;
  }
  // Agent::act(o, false): the first maximum.
  [[nodiscard]] static rag::agent::Action greedy(const double* q) {
    rag::agent::Action best = 0;
    for (rag::agent::Action a = 1; a < ScalpEnv::kNumActions; ++a)
      if (q[a] > q[best]) best = a;
    return best;
  }
};

// Per-episode results of a built-in policy on an indexed source.
struct BuiltinResult {
  std::vector<double> net_pnl_ticks, orders, fees_ticks;
};

// Plays every episode of an indexed source with `policy` (of2_baselines' loop).
inline BuiltinResult run_builtin(const RunParams& r, const EpisodeSource& src, const BuiltinPolicy& policy, unsigned threads,
                                 seeds::Ledger& L = seeds::Ledger::process()) {
  if (src.is_train()) throw std::invalid_argument("run_builtin: needs an indexed source (block, days or stress)");
  L.use(src.block);
  if (src.kind == EpisodeSource::Kind::Stress) (void)src.stress_scenario(r);
  const std::size_t n = src.count;
  BuiltinResult res;
  res.net_pnl_ticks.assign(n, 0);
  res.orders.assign(n, 0);
  res.fees_ticks.assign(n, 0);
  const unsigned t = threads == 0 ? std::max(1u, std::thread::hardware_concurrency()) : threads;
  Pool pool(static_cast<unsigned>(std::min<std::size_t>(t, n)));
  pool.run(n, [&](std::size_t j) {
    ScalpEnv env(src.params_of(r, j));
    rag::agent::Rng rng(src.seed_of(j));
    std::vector<double> scratch(policy.net ? policy.net->scratch_size() : 0);
    auto o = env.reset(rng);
    for (;;) {
      const auto st = env.step(policy.act(o, env, scratch.data()), rng);
      o = st.obs;
      if (st.done) break;
    }
    res.net_pnl_ticks[j] = env.net_pnl_ticks();
    res.orders[j] = env.orders();
    res.fees_ticks[j] = env.fees_ticks();
  });
  return res;
}

// ------------------------------------------------------------------ batched environment
// Refuses concurrent use of one environment from two threads (e.g. two Python
// threads with the GIL released). Reentrant on the owning thread, so a binding
// can hold it across a call AND the copy of its results (hold()), and the
// call's own guard nests inside.
class UseGuard {
 public:
  UseGuard(std::atomic<std::thread::id>& owner, const char* what) : owner_(owner) {
    const auto me = std::this_thread::get_id();
    std::thread::id expected{};
    if (owner_.compare_exchange_strong(expected, me))
      held_ = true;
    else if (expected != me)
      throw std::runtime_error(std::string(what) + " is already in use by another thread");
  }
  ~UseGuard() {
    if (held_) owner_.store(std::thread::id{});
  }
  UseGuard(const UseGuard&) = delete;
  UseGuard& operator=(const UseGuard&) = delete;

 private:
  std::atomic<std::thread::id>& owner_;
  bool held_ = false;
};

class VecScalpEnv {
 public:
  static constexpr std::size_t D = ScalpEnv::kStateDims;
  // Privileged columns: hidden fair value, mid, gap = fair value - mid, session second,
  // this decision's fill price (NaN: none, or two orders; recovered from the cash change,
  // so it can differ from the fill's average price in the last bits: display only),
  // orders so far, cash (ticks).
  static constexpr std::size_t P = 7;
  static constexpr const char* kPrivColumns[P] = {"fair_value", "mid", "gap", "session_s", "fill_price", "orders", "cash_ticks"};

  VecScalpEnv(const RunParams& r, const EpisodeSource& src, std::size_t n_envs = 8, unsigned threads = 1, bool privileged = false,
              seeds::Ledger& L = seeds::Ledger::process())
      : r_(r), src_(src), n_(src.is_train() ? n_envs : static_cast<std::size_t>(src.count)), priv_on_(privileged), stream_(src.run_seed) {
    if (privileged && !seeds::info(src.block).privileged_ok)
      throw std::runtime_error(std::string("privileged outputs are refused on block ") + seeds::info(src.block).name + " (train and viz only)");
    if (n_ == 0) throw std::invalid_argument("n_envs must be > 0");
    if (src.kind == EpisodeSource::Kind::Stress) (void)src.stress_scenario(r);
    L.use(src.block);
    unsigned t = threads == 0 ? std::max(1u, std::thread::hardware_concurrency()) : threads;
    pool_ = std::make_unique<Pool>(static_cast<unsigned>(std::min<std::size_t>(t, n_)));
    envs_.reserve(n_);
    for (std::size_t j = 0; j < n_; ++j) envs_.emplace_back(src.is_train() ? r.p : src.params_of(r, j));
    obs_.assign(n_ * D, 0);
    final_obs_.assign(n_ * D, 0);
    rew_.assign(n_, 0);
    done_.assign(n_, 0);
    ids_.assign(n_, 0);
    final_ids_.assign(n_, 0);
    pnl_.assign(n_, 0);
    orders_.assign(n_, 0);
    trips_.assign(n_, 0);
    fees_.assign(n_, 0);
    pos_.assign(n_, 0);
    start_.assign(n_, 0);
    rngs_.assign(n_, rag::agent::Rng(0));
    after_.assign(n_, rag::agent::Rng(0));
    if (priv_on_) {
      priv_.assign(n_ * P, 0);
      final_priv_.assign(n_ * P, 0);
    }
  }

  [[nodiscard]] std::size_t size() const noexcept { return n_; }
  [[nodiscard]] bool privileged() const noexcept { return priv_on_; }
  [[nodiscard]] const RunParams& params() const noexcept { return r_; }
  [[nodiscard]] const EpisodeSource& source() const noexcept { return src_; }
  [[nodiscard]] unsigned threads() const noexcept { return pool_->threads(); }
  [[nodiscard]] std::uint64_t episodes_started() const noexcept { return started_; }

  // Starts the next n_envs episodes in every slot (train: the next stream
  // episodes; indexed: the source's episodes, once).
  // Holds the environment for the calling thread (see UseGuard).
  [[nodiscard]] UseGuard hold() const { return UseGuard(owner_, "VecScalpEnv"); }

  void reset() {
    const UseGuard g(owner_, "VecScalpEnv");
    start_round();
  }

  // actions[j] in {0 short, 1 flat, 2 long}. Train sources reset finished
  // slots (all finish together): the observation returned at done is the
  // next episode's first; final_obs() holds the terminal one.
  void step(const std::int64_t* actions) {
    const UseGuard g(owner_, "VecScalpEnv");
    if (!live_) throw std::runtime_error(round_ == 0 ? "step before reset" : "the episodes are over (indexed sources are played once)");
    for (std::size_t j = 0; j < n_; ++j)
      if (actions[j] < 0 || actions[j] > 2) throw std::invalid_argument("action " + std::to_string(actions[j]) + " is not 0, 1 or 2");
    rag::agent::Rng unused(0);
    pool_->run(n_, [&](std::size_t j) {
      ScalpEnv& e = envs_[j];
      const double c0 = cash(e, pos_[j]);
      const auto orders0 = e.orders();
      rag::agent::Rng rr = unused;
      const auto st = e.step(static_cast<rag::agent::Action>(actions[j]), rr);
      const int dpos = st.obs.position - pos_[j];
      pos_[j] = st.obs.position;
      rew_[j] = st.reward;
      done_[j] = st.done ? 1 : 0;
      write(j, st.obs, obs_.data());
      if (priv_on_) {
        double fill = std::nan("");
        if (e.orders() - orders0 == 1 && dpos != 0) fill = -(cash(e, st.obs.position) - c0) / dpos;
        write_priv(j, priv_.data(), fill);
      }
      if (st.done) {
        std::copy_n(obs_.data() + j * D, D, final_obs_.data() + j * D);
        if (priv_on_) std::copy_n(priv_.data() + j * P, P, final_priv_.data() + j * P);
        pnl_[j] = e.net_pnl_ticks();
        orders_[j] = e.orders();
        trips_[j] = e.round_trips();
        fees_[j] = e.fees_ticks();
        final_ids_[j] = ids_[j];
      }
    });
    for (std::size_t j = 1; j < n_; ++j)
      if (done_[j] != done_[0]) throw std::logic_error("VecScalpEnv: slots finished on different steps (episodes must have equal length)");
    if (!done_[0]) return;
    ++finished_;
    live_ = false;
    if (src_.is_train()) start_round();
  }

  // Buffers (n x D obs, n rewards, ...), valid until the next call.
  [[nodiscard]] const double* obs() const noexcept { return obs_.data(); }
  [[nodiscard]] const double* final_obs() const noexcept { return final_obs_.data(); }
  [[nodiscard]] const double* rewards() const noexcept { return rew_.data(); }
  [[nodiscard]] const std::uint8_t* dones() const noexcept { return done_.data(); }
  [[nodiscard]] const std::uint64_t* episode_ids() const noexcept { return ids_.data(); }
  [[nodiscard]] const std::uint64_t* final_episode_ids() const noexcept { return final_ids_.data(); }
  // Statistics of the episodes that finished last.
  [[nodiscard]] const double* net_pnl_ticks() const noexcept { return pnl_.data(); }
  [[nodiscard]] const double* orders() const noexcept { return orders_.data(); }
  [[nodiscard]] const double* round_trips() const noexcept { return trips_.data(); }
  [[nodiscard]] const double* fees_ticks() const noexcept { return fees_.data(); }
  [[nodiscard]] std::uint64_t rounds_finished() const noexcept { return finished_; }
  // Privileged buffers (n x P); throw unless constructed with privileged = true.
  [[nodiscard]] const double* priv() const {
    if (!priv_on_) throw std::runtime_error("privileged outputs were not requested (privileged=True, train or viz blocks only)");
    return priv_.data();
  }
  [[nodiscard]] const double* final_priv() const {
    if (!priv_on_) throw std::runtime_error("privileged outputs were not requested (privileged=True, train or viz blocks only)");
    return final_priv_.data();
  }

 private:
  // Starts the next round: train streams draw serially, the resets run on the pool.
  void start_round() {
    if (!src_.is_train() && round_ > 0) throw std::runtime_error("an indexed source is played once (one round of paired episodes)");
    for (std::size_t j = 0; j < n_; ++j) {
      if (src_.is_train()) {
        rngs_[j] = stream_;
        start_[j] = ScalpEnv::draw_market(r_.p, stream_).start;  // advances the stream as reset() will
        after_[j] = stream_;
        ids_[j] = next_train_++;
      } else {
        ids_[j] = src_.seed_of(j) - seeds::info(src_.block).base;  // block-relative (days: d * 100 + k)
        rngs_[j] = rag::agent::Rng(src_.seed_of(j));
        rag::agent::Rng c = rngs_[j];
        start_[j] = ScalpEnv::draw_market(envs_[j].params(), c).start;
        after_[j] = c;
      }
    }
    pool_->run(n_, [&](std::size_t j) {
      rag::agent::Rng rr = rngs_[j];
      const ScalpObs o = envs_[j].reset(rr);
      rag::agent::Rng x = rr, y = after_[j];
      if (x.next() != y.next() || x.next() != y.next())
        throw std::logic_error("VecScalpEnv: ScalpEnv::reset drew more than draw_market; the train-stream identity no longer holds");
      pos_[j] = o.position;
      write(j, o, obs_.data());
      if (priv_on_) write_priv(j, priv_.data(), std::nan(""));
    });
    ++round_;
    started_ += n_;
    live_ = true;
  }
  static double cash(const ScalpEnv& e, int position) { return e.net_pnl_ticks() + e.fees_ticks() - position * e.market().mid(); }
  void write(std::size_t j, const ScalpObs& o, double* out) const {
    const auto x = envs_[j].features(o);
    std::copy(x.begin(), x.end(), out + j * D);
  }
  void write_priv(std::size_t j, double* out, double fill) const {
    const ScalpEnv& e = envs_[j];
    const auto& m = e.market();
    double* p = out + j * P;
    p[0] = m.fair_value();
    p[1] = m.mid();
    p[2] = m.fair_value() - m.mid();
    p[3] = start_[j] + static_cast<double>(m.t_ns()) * 1e-9;
    p[4] = fill;
    p[5] = e.orders();
    p[6] = cash(e, pos_[j]);
  }

  RunParams r_;
  EpisodeSource src_;
  std::size_t n_;
  bool priv_on_;
  std::unique_ptr<Pool> pool_;
  rag::agent::Rng stream_;
  std::uint64_t next_train_ = 0, round_ = 0, started_ = 0, finished_ = 0;
  bool live_ = false;
  mutable std::atomic<std::thread::id> owner_{};
  std::vector<ScalpEnv> envs_;
  std::vector<double> obs_, final_obs_, rew_, priv_, final_priv_, pnl_, orders_, trips_, fees_, start_;
  std::vector<std::uint8_t> done_;
  std::vector<std::uint64_t> ids_, final_ids_;
  std::vector<int> pos_;
  std::vector<rag::agent::Rng> rngs_, after_;
};

// ------------------------------------------------------------------ recorded ticks
// ReplayScalpEnv on M windows of one tape, stepped together (backtests on
// real data). Real data is sealed: the "real" block must be open. Windows are
// full (ReplayScalpEnv::windows()), so all slots finish together.
class VecReplayEnv {
 public:
  static constexpr std::size_t D = ReplayScalpEnv::kStateDims;

  VecReplayEnv(std::shared_ptr<const TickTape> tape, const ReplayParams& p, std::vector<std::int64_t> windows, unsigned threads = 1,
               seeds::Ledger& L = seeds::Ledger::process())
      : windows_(std::move(windows)), n_(windows_.size()) {
    L.use(seeds::Block::Real);
    if (n_ == 0) throw std::invalid_argument("VecReplayEnv: no windows");
    envs_.reserve(n_);
    for (std::size_t j = 0; j < n_; ++j) envs_.emplace_back(tape, p);
    const std::size_t w = envs_[0].windows();
    for (auto k : windows_)
      if (k < 0 || static_cast<std::size_t>(k) >= w)
        throw std::out_of_range("VecReplayEnv: window " + std::to_string(k) + " not on the tape (" + std::to_string(w) + " full windows)");
    pool_ = std::make_unique<Pool>(static_cast<unsigned>(std::min<std::size_t>(std::max(1u, threads), n_)));
    obs_.assign(n_ * D, 0);
    final_obs_.assign(n_ * D, 0);
    rew_.assign(n_, 0);
    done_.assign(n_, 0);
    pnl_.assign(n_, 0);
    orders_.assign(n_, 0);
    fees_.assign(n_, 0);
  }
  [[nodiscard]] std::size_t size() const noexcept { return n_; }
  [[nodiscard]] std::size_t windows_on_tape() const noexcept { return envs_[0].windows(); }
  [[nodiscard]] const std::vector<std::int64_t>& windows() const noexcept { return windows_; }
  [[nodiscard]] UseGuard hold() const { return UseGuard(owner_, "VecReplayEnv"); }

  void reset() {
    const UseGuard g(owner_, "VecReplayEnv");
    pool_->run(n_, [&](std::size_t j) {
      const ScalpObs o = envs_[j].seek(static_cast<std::size_t>(windows_[j]));
      write(j, o, obs_.data());
    });
    live_ = true;
  }
  void step(const std::int64_t* actions) {
    const UseGuard g(owner_, "VecReplayEnv");
    if (!live_) throw std::runtime_error("step before reset, or after the windows ended");
    for (std::size_t j = 0; j < n_; ++j)
      if (actions[j] < 0 || actions[j] > 2) throw std::invalid_argument("action " + std::to_string(actions[j]) + " is not 0, 1 or 2");
    rag::agent::Rng unused(0);
    pool_->run(n_, [&](std::size_t j) {
      rag::agent::Rng rr = unused;
      const auto st = envs_[j].step(static_cast<rag::agent::Action>(actions[j]), rr);
      rew_[j] = st.reward;
      done_[j] = st.done ? 1 : 0;
      write(j, st.obs, obs_.data());
      if (st.done) {
        std::copy_n(obs_.data() + j * D, D, final_obs_.data() + j * D);
        pnl_[j] = envs_[j].net_pnl_ticks();
        orders_[j] = envs_[j].orders();
        fees_[j] = envs_[j].fees_ticks();
      }
    });
    for (std::size_t j = 1; j < n_; ++j)
      if (done_[j] != done_[0]) throw std::logic_error("VecReplayEnv: windows finished on different steps");
    if (done_[0]) live_ = false;
  }
  [[nodiscard]] const double* obs() const noexcept { return obs_.data(); }
  [[nodiscard]] const double* final_obs() const noexcept { return final_obs_.data(); }
  [[nodiscard]] const double* rewards() const noexcept { return rew_.data(); }
  [[nodiscard]] const std::uint8_t* dones() const noexcept { return done_.data(); }
  [[nodiscard]] const double* net_pnl_ticks() const noexcept { return pnl_.data(); }
  [[nodiscard]] const double* orders() const noexcept { return orders_.data(); }
  [[nodiscard]] const double* fees_ticks() const noexcept { return fees_.data(); }

 private:
  void write(std::size_t j, const ScalpObs& o, double* out) const {
    const auto x = envs_[j].features(o);
    std::copy(x.begin(), x.end(), out + j * D);
  }
  std::vector<std::int64_t> windows_;
  std::size_t n_;
  std::unique_ptr<Pool> pool_;
  bool live_ = false;
  mutable std::atomic<std::thread::id> owner_{};
  std::vector<ReplayScalpEnv> envs_;
  std::vector<double> obs_, final_obs_, rew_, pnl_, orders_, fees_;
  std::vector<std::uint8_t> done_;
};

}  // namespace rag::orderflow
