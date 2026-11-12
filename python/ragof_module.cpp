// _ragof — the OF15 Python binding (nanobind; built only with -DRAG_PYTHON=ON).
//
// A thin wrapper over rag/orderflow/vec_scalp_env.hpp and seed_ledger.hpp:
// numpy arrays in and out (fresh copies, never views of internal buffers),
// the GIL released while C++ steps the environments. Python never passes a
// raw seed: episodes come from EpisodeSource (train(run_seed), block(name,
// first, count), days, stress), checked against the seed ledger, and sealed
// blocks open only through open_sealed() in a process that has not trained.
//
//   import _ragof as R
//   p = R.ScalpParams("discovery_v4", 1050.0)
//   env = R.VecScalpEnv(p, R.EpisodeSource.train(1), n_envs=8, threads=1)
//   obs = env.reset()                       # float64 [8, 16]
//   obs, reward, done = env.step(actions)   # actions int64 [8] in {0, 1, 2}
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/map.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <cstdint>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "rag/agent/mlp.hpp"
#include "rag/orderflow/scalp_run.hpp"
#include "rag/orderflow/seed_ledger.hpp"
#include "rag/orderflow/vec_scalp_env.hpp"

namespace nb = nanobind;
using namespace nb::literals;
using namespace rag::orderflow;

namespace {

constexpr std::size_t D = VecScalpEnv::D, P = VecScalpEnv::P;

// A fresh numpy array holding a copy of p[0 .. prod(shape)).
template <class T, class S = T>
nb::ndarray<nb::numpy, T> copy_array(const S* p, std::initializer_list<std::size_t> shape) {
  std::size_t n = 1;
  for (auto s : shape) n *= s;
  T* buf = new T[n > 0 ? n : 1];
  for (std::size_t i = 0; i < n; ++i) buf[i] = static_cast<T>(p[i]);
  nb::capsule owner(buf, [](void* q) noexcept { delete[] static_cast<T*>(q); });
  return nb::ndarray<nb::numpy, T>(buf, shape.size(), shape.begin(), owner);
}

using Actions = nb::ndarray<const std::int64_t, nb::ndim<1>, nb::c_contig, nb::device::cpu>;
using Obs2 = nb::ndarray<const double, nb::ndim<2>, nb::c_contig, nb::device::cpu>;

std::vector<std::int64_t> take_actions(const Actions& a, std::size_t n) {
  if (a.shape(0) != n) throw std::invalid_argument("actions: expected " + std::to_string(n) + " values, got " + std::to_string(a.shape(0)));
  return {a.data(), a.data() + n};
}

nb::dict params_dict(const RunParams& r) {
  nb::dict d;
  d["market"] = r.market;
  d["warmup_seconds"] = r.p.warmup_seconds;
  d["decide_seconds"] = r.p.decide_seconds;
  d["episode_seconds"] = r.p.episode_seconds;
  d["start_second"] = r.p.start_second;
  d["latency_steps"] = r.p.latency_steps;
  d["contracts"] = r.p.contracts;
  d["fee_dollars"] = r.p.fee_dollars;
  d["sim_overrides"] = r.sim_overrides;
  return d;
}

const char* kind_name(EpisodeSource::Kind k) {
  switch (k) {
    case EpisodeSource::Kind::Train: return "train";
    case EpisodeSource::Kind::Indexed: return "indexed";
    case EpisodeSource::Kind::Days: return "days";
    case EpisodeSource::Kind::Stress: return "stress";
  }
  return "?";
}

nb::dict source_dict(const EpisodeSource& s) {
  nb::dict d;
  d["kind"] = kind_name(s.kind);
  d["block"] = seeds::info(s.block).name;
  if (s.kind == EpisodeSource::Kind::Train) {
    d["run_seed"] = s.run_seed;
    return d;
  }
  d["first"] = s.first;
  d["count"] = s.count;
  if (s.kind == EpisodeSource::Kind::Days) {
    d["n_days"] = s.n_days;
    d["sessions_per_day"] = s.sessions_per_day;
  }
  if (s.kind == EpisodeSource::Kind::Stress) {
    d["scenario"] = s.scenario;
    d["list"] = s.list;
  }
  if (!s.orchestrator.empty()) d["orchestrator"] = s.orchestrator;
  return d;
}

// An of_train network, loaded with its record checked (check_net).
struct NeuralQNet {
  std::shared_ptr<const rag::agent::Mlp> net;
  std::string path;
  std::map<std::string, std::string> record;
};

BuiltinPolicy parse_policy(nb::handle h) {
  if (nb::isinstance<nb::str>(h)) {
    const std::string s = nb::cast<std::string>(h);
    if (s == "flat") return BuiltinPolicy::flat();
    if (s == "hold") return BuiltinPolicy::hold();
    if (s == "delta30") return BuiltinPolicy::delta30();
    throw std::invalid_argument("policy '" + s + "': flat, hold, delta30, ('rule', k, th, reverse), ('oracle', th) or ('neuralq', net)");
  }
  if (nb::isinstance<nb::tuple>(h)) {
    const nb::tuple t = nb::borrow<nb::tuple>(h);
    if (t.size() >= 1 && nb::isinstance<nb::str>(t[0])) {
      const std::string k = nb::cast<std::string>(t[0]);
      if (k == "rule" && t.size() == 4) return BuiltinPolicy::rule(nb::cast<std::size_t>(t[1]), nb::cast<double>(t[2]), nb::cast<bool>(t[3]));
      if (k == "oracle" && t.size() == 2) return BuiltinPolicy::oracle(nb::cast<double>(t[1]));
      if (k == "neuralq" && t.size() == 2) return BuiltinPolicy::neuralq(nb::cast<const NeuralQNet&>(t[1]).net);
    }
  }
  throw std::invalid_argument("policy: flat, hold, delta30, ('rule', k, th, reverse), ('oracle', th) or ('neuralq', net)");
}

nb::dict stats_dict(const double* pnl, const double* orders, const double* trips, const double* fees, std::size_t n) {
  nb::dict d;
  d["net_pnl_ticks"] = copy_array<double>(pnl, {n});
  d["orders"] = copy_array<double>(orders, {n});
  if (trips) d["round_trips"] = copy_array<double>(trips, {n});
  d["fees_ticks"] = copy_array<double>(fees, {n});
  return d;
}

}  // namespace

NB_MODULE(_ragof, m) {
  m.doc() = "OF15: batched ScalpEnv on the ES order-flow simulator, episodes named through the seed ledger";
  m.attr("OBS_DIM") = D;
  m.attr("N_ACTIONS") = ScalpEnv::kNumActions;
  m.attr("N_FEATURES") = kFeatures;
  {
    nb::list cols;
    for (const char* c : VecScalpEnv::kPrivColumns) cols.append(c);
    m.attr("PRIV_COLUMNS") = nb::tuple(cols);
    nb::list blocks;
    for (const auto& b : seeds::kBlockTable) {
      nb::dict d;
      d["name"] = b.name;
      d["base"] = b.base;
      d["size"] = b.size;
      d["sealed"] = b.sealed;
      d["privileged_ok"] = b.privileged_ok;
      d["indexed"] = b.indexed;
      d["needs_claim"] = b.needs_claim;
      blocks.append(d);
    }
    m.attr("BLOCKS") = nb::tuple(blocks);
    nb::list keys;
    for (const auto& f : sim_override_fields()) keys.append(f.first);
    m.attr("SIM_OVERRIDE_KEYS") = nb::tuple(keys);
    nb::dict build;
#if defined(_MSC_VER)
    build["compiler"] = "msvc " + std::to_string(_MSC_FULL_VER);
#elif defined(__clang__)
    build["compiler"] = std::string("clang ") + __clang_version__;
#elif defined(__GNUC__)
    build["compiler"] = std::string("gcc ") + __VERSION__;
#endif
#if defined(__AVX512F__)
    build["avx512"] = true;
#else
    build["avx512"] = false;
#endif
#if defined(__AVX2__)
    build["avx2"] = true;
#else
    build["avx2"] = false;
#endif
    build["nanobind"] = std::to_string(NB_VERSION_MAJOR) + "." + std::to_string(NB_VERSION_MINOR) + "." + std::to_string(NB_VERSION_PATCH);
    m.attr("BUILD_INFO") = build;
  }

  // ------------------------------------------------------------ ledger
  m.def(
      "open_sealed",
      [](const std::string& block, const std::string& reason) { seeds::Ledger::process().open_sealed(seeds::block_of(block), reason); }, "block"_a,
      "reason"_a, "Open a sealed block (test, stress, days, confirm, real) in this process: scoring only; refused after forbid_sealed() or training.");
  m.def(
      "forbid_sealed", [] { seeds::Ledger::process().forbid_sealed(); },
      "Close every sealed block for the rest of this process (training, HPO and live workers call it first).");
  m.def(
      "process_role", [] { return std::string(seeds::role_name(seeds::Ledger::process().role())); }, "'none', 'trainer' or 'scorer'.");
  m.def(
      "sealed_forbidden", [] { return seeds::Ledger::process().forbidden(); }, "True after forbid_sealed().");
  m.def(
      "is_open", [](const std::string& block) { return seeds::Ledger::process().is_open(seeds::block_of(block)); }, "block"_a);
  m.def(
      "sealed_openings", [] { return seeds::Ledger::process().openings(); }, "(block, reason) of every open_sealed() in this process.");

  // ------------------------------------------------------------ parameters
  nb::class_<RunParams>(m, "ScalpParams", "A run's ScalpParams: market preset, warm-up and the ScalpEnv options (validated).")
      .def(
          "__init__",
          [](RunParams* self, const std::string& market, double warmup, double decide, double episode, double start, std::uint32_t latency,
             std::uint32_t contracts, double fee, const std::map<std::string, double>& overrides) {
            new (self) RunParams(run_params(market, warmup, decide, episode, start, latency, contracts, fee, overrides));
          },
          "market"_a = "default", "warmup_seconds"_a = kDefaultWarmup, nb::kw_only(), "decide_seconds"_a = 1.0, "episode_seconds"_a = 600.0,
          "start_second"_a = -1.0, "latency_steps"_a = 1, "contracts"_a = 1, "fee_dollars"_a = 1.5,
          "sim_overrides"_a = std::map<std::string, double>{})
      .def_prop_ro("market", [](const RunParams& r) { return r.market; })
      .def_prop_ro("warmup_seconds", [](const RunParams& r) { return r.p.warmup_seconds; })
      .def_prop_ro("decide_seconds", [](const RunParams& r) { return r.p.decide_seconds; })
      .def_prop_ro("episode_seconds", [](const RunParams& r) { return r.p.episode_seconds; })
      .def_prop_ro("start_second", [](const RunParams& r) { return r.p.start_second; })
      .def_prop_ro("latency_steps", [](const RunParams& r) { return r.p.latency_steps; })
      .def_prop_ro("contracts", [](const RunParams& r) { return r.p.contracts; })
      .def_prop_ro("fee_dollars", [](const RunParams& r) { return r.p.fee_dollars; })
      .def_prop_ro("sim_overrides", [](const RunParams& r) { return r.sim_overrides; })
      .def_prop_ro("steps_per_episode", [](const RunParams& r) { return steps_per_episode(r.p); })
      .def("to_dict", &params_dict)
      .def("__eq__", [](const RunParams& a, const RunParams& b) { return a == b; })
      .def("__ne__", [](const RunParams& a, const RunParams& b) { return !(a == b); })
      .def("__repr__", [](const RunParams& r) {
        return "ScalpParams(" + nb::cast<std::string>(nb::repr(params_dict(r))) + ")";
      });
  m.def(
      "steps_per_episode", [](const RunParams& r) { return steps_per_episode(r.p); }, "params"_a);

  // ------------------------------------------------------------ episode sources
  nb::class_<EpisodeSource>(m, "EpisodeSource", "Names episodes: a train stream or block-relative indices; never raw seeds.")
      .def_static("train", &EpisodeSource::train, "run_seed"_a, "Episode k = the k-th reset on Rng(run_seed), of_train's Trainer stream.")
      .def_static(
          "block",
          [](const std::string& name, std::uint64_t first, std::uint64_t count, const std::string& orchestrator) {
            return EpisodeSource::indexed(name, first, count, orchestrator);
          },
          "name"_a, "first"_a, "count"_a, nb::kw_only(), "orchestrator"_a = "",
          "Episodes first .. first+count-1 of block name (validation, pairing, viz, test, confirm): episode i = Rng(base + first + i).")
      .def_static(
          "days", [](std::uint32_t first_day, std::uint32_t n_days, std::uint32_t spd) { return EpisodeSource::days(first_day, n_days, spd); },
          "first_day"_a, "n_days"_a, "sessions_per_day"_a = 39, "OF8's sealed days: Rng(930M + d * 100 + k), session start day_session_start(k).")
      .def_static(
          "stress",
          [](const std::string& scenario, std::uint64_t first, std::uint64_t count, const std::string& list) {
            return EpisodeSource::stress(scenario, first, count, list);
          },
          "scenario"_a, "first"_a, "count"_a, "list"_a = "", "OF6's sealed stresses: Rng(920M + i) under a scenario of the legacy or discovery list.")
      .def_prop_ro("kind", [](const EpisodeSource& s) { return std::string(kind_name(s.kind)); })
      .def_prop_ro("block_name", [](const EpisodeSource& s) { return std::string(seeds::info(s.block).name); })
      .def_prop_ro("count", [](const EpisodeSource& s) { return s.count; })
      .def("describe", &source_dict)
      .def("__repr__", [](const EpisodeSource& s) { return "EpisodeSource(" + nb::cast<std::string>(nb::repr(source_dict(s))) + ")"; });

  // ------------------------------------------------------------ batched environment
  nb::class_<VecScalpEnv>(m, "VecScalpEnv", "N ScalpEnv episodes stepped together in C++ (lockstep; train sources auto-reset).")
      .def(
          "__init__",
          [](VecScalpEnv* self, const RunParams& r, const EpisodeSource& s, std::size_t n_envs, unsigned threads, bool privileged) {
            new (self) VecScalpEnv(r, s, n_envs, threads, privileged);
          },
          "params"_a, "source"_a, "n_envs"_a = 8, "threads"_a = 1, "privileged"_a = false)
      .def("reset",
           [](VecScalpEnv& e) {
             const auto held = e.hold();  // held across reset and the copy (UseGuard)
             {
               nb::gil_scoped_release g;
               e.reset();
             }
             return copy_array<double>(e.obs(), {e.size(), D});
           })
      .def(
          "step",
          [](VecScalpEnv& e, const Actions& a) {
            const auto held = e.hold();  // held across step and the copies (UseGuard)
            const auto act = take_actions(a, e.size());
            {
              nb::gil_scoped_release g;
              e.step(act.data());
            }
            const std::size_t n = e.size();
            return nb::make_tuple(copy_array<double>(e.obs(), {n, D}), copy_array<double>(e.rewards(), {n}), copy_array<bool>(e.dones(), {n}));
          },
          "actions"_a)
      .def_prop_ro("final_obs", [](const VecScalpEnv& e) { const auto held = e.hold(); return copy_array<double>(e.final_obs(), {e.size(), D}); }, nb::rv_policy::move)
      .def_prop_ro("priv", [](const VecScalpEnv& e) { const auto held = e.hold(); return copy_array<double>(e.priv(), {e.size(), P}); }, nb::rv_policy::move)
      .def_prop_ro("final_priv", [](const VecScalpEnv& e) { const auto held = e.hold(); return copy_array<double>(e.final_priv(), {e.size(), P}); }, nb::rv_policy::move)
      .def_prop_ro("episode_ids", [](const VecScalpEnv& e) { const auto held = e.hold(); return copy_array<std::uint64_t>(e.episode_ids(), {e.size()}); }, nb::rv_policy::move)
      .def_prop_ro(
          "final_episode_ids", [](const VecScalpEnv& e) { const auto held = e.hold(); return copy_array<std::uint64_t>(e.final_episode_ids(), {e.size()}); }, nb::rv_policy::move)
      .def_prop_ro("episode_stats",
                   [](const VecScalpEnv& e) { const auto held = e.hold(); return stats_dict(e.net_pnl_ticks(), e.orders(), e.round_trips(), e.fees_ticks(), e.size()); })
      .def_prop_ro("n_envs", &VecScalpEnv::size)
      .def_prop_ro("threads", &VecScalpEnv::threads)
      .def_prop_ro("privileged", &VecScalpEnv::privileged)
      .def_prop_ro("params", [](const VecScalpEnv& e) { return e.params(); })
      .def_prop_ro("source", [](const VecScalpEnv& e) { return e.source(); })
      .def_prop_ro("rounds_finished", &VecScalpEnv::rounds_finished)
      .def_prop_ro("episodes_started", &VecScalpEnv::episodes_started);

  // ------------------------------------------------------------ of_train networks and built-in policies
  nb::class_<NeuralQNet>(m, "NeuralQNet", "An of_train network (seedK.net), refused unless its record matches market and warm-up.")
      .def_static(
          "load",
          [](const std::string& path, const std::string& market, double warmup) {
            if (const std::string why = check_net(path, market, warmup); !why.empty()) throw std::runtime_error(why);
            auto net = std::make_shared<const rag::agent::Mlp>(rag::agent::Mlp::load(path));
            if (net->inputs() != D || net->outputs() != ScalpEnv::kNumActions) throw std::runtime_error(path + ": not a 16 -> 3 ScalpEnv network");
            NeuralQNet q{std::move(net), path, {}};
            std::ifstream in(record_path_of_net(path));
            for (std::string line; std::getline(in, line);)
              if (const auto eq = line.find('='); eq != std::string::npos) q.record[line.substr(0, eq)] = line.substr(eq + 1);
            return q;
          },
          "path"_a, "market"_a, "warmup_seconds"_a)
      .def(
          "q",
          [](const NeuralQNet& q, const Obs2& x) {
            if (x.shape(1) != D) throw std::invalid_argument("obs must be [N, 16]");
            const std::size_t n = x.shape(0);
            std::vector<double> out(n * ScalpEnv::kNumActions), scratch(q.net->scratch_size());
            for (std::size_t i = 0; i < n; ++i) q.net->forward(x.data() + i * D, out.data() + i * ScalpEnv::kNumActions, scratch.data());
            return copy_array<double>(out.data(), {n, ScalpEnv::kNumActions});
          },
          "obs"_a)
      .def(
          "act",
          [](const NeuralQNet& q, const Obs2& x) {
            if (x.shape(1) != D) throw std::invalid_argument("obs must be [N, 16]");
            const std::size_t n = x.shape(0);
            std::vector<std::int64_t> out(n);
            std::vector<double> scratch(q.net->scratch_size());
            double v[ScalpEnv::kNumActions];
            for (std::size_t i = 0; i < n; ++i) {
              q.net->forward(x.data() + i * D, v, scratch.data());
              out[i] = BuiltinPolicy::greedy(v);
            }
            return copy_array<std::int64_t>(out.data(), {n});
          },
          "obs"_a, "Greedy action, first maximum (Agent::act(o, false)).")
      .def_prop_ro("record", [](const NeuralQNet& q) { return q.record; })
      .def_prop_ro("path", [](const NeuralQNet& q) { return q.path; })
      .def_prop_ro("parameters", [](const NeuralQNet& q) { return q.net->parameters(); });

  m.def(
      "run_builtin",
      [](const RunParams& r, const EpisodeSource& s, nb::handle policy, unsigned threads) {
        const BuiltinPolicy pol = parse_policy(policy);
        BuiltinResult res;
        {
          nb::gil_scoped_release g;
          res = run_builtin(r, s, pol, threads);
        }
        const std::size_t n = res.net_pnl_ticks.size();
        return stats_dict(res.net_pnl_ticks.data(), res.orders.data(), nullptr, res.fees_ticks.data(), n);
      },
      "params"_a, "source"_a, "policy"_a, "threads"_a = 24,
      "Per-episode results of a C++ policy on an indexed source: 'flat', 'hold', 'delta30', ('rule', k, th, reverse), ('oracle', th), "
      "('neuralq', net). The oracle's fair value stays in C++.");

  // ------------------------------------------------------------ recorded ticks (sealed: open_sealed('real'))
  nb::class_<TickTape>(m, "TickTape", "Recorded ticks, bucketed (TickTape::from_csv). Needs open_sealed('real').")
      .def_static(
          "from_csv",
          [](const std::string& path, const std::string& time_unit, double tick_size, double bucket_seconds, double max_gap_seconds) {
            seeds::Ledger::process().use(seeds::Block::Real);
            TickCsvOptions o;
            o.time_unit = time_unit;
            o.tick_size = tick_size;
            o.bucket_seconds = bucket_seconds;
            o.max_gap_seconds = max_gap_seconds;
            std::shared_ptr<TickTape> t;
            {
              nb::gil_scoped_release g;
              t = std::make_shared<TickTape>(TickTape::from_csv(path, o));
            }
            return t;
          },
          "path"_a, "time_unit"_a = "ns", "tick_size"_a = 0.25, "bucket_seconds"_a = 0.1, "max_gap_seconds"_a = 60.0)
      .def_prop_ro("size", &TickTape::size)
      .def_prop_ro("bucket_seconds", &TickTape::bucket_seconds)
      .def_prop_ro("stats", [](const TickTape& t) {
        const auto& s = t.stats();
        nb::dict d;
        d["lines"] = s.lines;
        d["trades"] = s.trades;
        d["quotes"] = s.quotes;
        d["buckets"] = s.buckets;
        d["skipped"] = s.skipped;
        d["gaps"] = s.gaps;
        return d;
      });

  nb::class_<ReplayParams>(m, "ReplayParams")
      .def(
          "__init__",
          [](ReplayParams* self, double decide, double window, double warmup, std::uint32_t latency, std::uint32_t contracts, double fee) {
            if (!(decide > 0) || !(window >= decide) || !(warmup >= 0) || contracts < 1 || !(fee >= 0))
              throw std::invalid_argument("ReplayParams: decide_seconds > 0, window_seconds >= decide_seconds, warmup_seconds >= 0, contracts >= 1");
            new (self) ReplayParams{decide, window, warmup, latency, contracts, fee};
          },
          nb::kw_only(), "decide_seconds"_a = 1.0, "window_seconds"_a = 600.0, "warmup_seconds"_a = 150.0, "latency_buckets"_a = 1, "contracts"_a = 1,
          "fee_dollars"_a = 1.5)
      .def_ro("decide_seconds", &ReplayParams::decide_seconds)
      .def_ro("window_seconds", &ReplayParams::window_seconds)
      .def_ro("warmup_seconds", &ReplayParams::warmup_seconds)
      .def_ro("latency_buckets", &ReplayParams::latency_buckets)
      .def_ro("contracts", &ReplayParams::contracts)
      .def_ro("fee_dollars", &ReplayParams::fee_dollars);

  nb::class_<VecReplayEnv>(m, "VecReplayEnv", "ReplayScalpEnv on M full windows of a tape, stepped together (scoring only).")
      .def(
          "__init__",
          [](VecReplayEnv* self, std::shared_ptr<TickTape> tape, const ReplayParams& p, const std::vector<std::int64_t>& windows, unsigned threads) {
            new (self) VecReplayEnv(std::move(tape), p, windows, threads);
          },
          "tape"_a, "params"_a, "windows"_a, "threads"_a = 1)
      .def("reset",
           [](VecReplayEnv& e) {
             const auto held = e.hold();  // held across reset and the copy (UseGuard)
             {
               nb::gil_scoped_release g;
               e.reset();
             }
             return copy_array<double>(e.obs(), {e.size(), D});
           })
      .def(
          "step",
          [](VecReplayEnv& e, const Actions& a) {
            const auto held = e.hold();  // held across step and the copies (UseGuard)
            const auto act = take_actions(a, e.size());
            {
              nb::gil_scoped_release g;
              e.step(act.data());
            }
            const std::size_t n = e.size();
            return nb::make_tuple(copy_array<double>(e.obs(), {n, D}), copy_array<double>(e.rewards(), {n}), copy_array<bool>(e.dones(), {n}));
          },
          "actions"_a)
      .def_prop_ro("final_obs", [](const VecReplayEnv& e) { const auto held = e.hold(); return copy_array<double>(e.final_obs(), {e.size(), D}); }, nb::rv_policy::move)
      .def_prop_ro("episode_stats", [](const VecReplayEnv& e) { const auto held = e.hold(); return stats_dict(e.net_pnl_ticks(), e.orders(), nullptr, e.fees_ticks(), e.size()); })
      .def_prop_ro("n_envs", &VecReplayEnv::size)
      .def_prop_ro("windows_on_tape", &VecReplayEnv::windows_on_tape)
      .def_prop_ro("windows", [](const VecReplayEnv& e) { return e.windows(); });
}
