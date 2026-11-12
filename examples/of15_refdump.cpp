// of15_refdump — serial reference episodes for the OF15 binding's bit-identity
// tests (python/tests/test_bit_identity.py). Plain ScalpEnv / ReplayScalpEnv
// loops, written independently of vec_scalp_env.hpp; built only with
// -DRAG_PYTHON=ON. Actions follow a fixed pattern: decision t of the run's
// k-th episode plays (t * 7 + k * 3 + t / 13) % 3.
//
//   of15_refdump scalp <market> <warmup> train <run_seed> <episodes> <out.bin>
//       one Rng(run_seed) stream for every reset (of_train's Trainer stream)
//   of15_refdump scalp <market> <warmup> seeds <seed0> <episodes> <out.bin>
//       episode i = Rng(seed0 + i) (evaluate(), of_train's score())
//     per episode, doubles: obs0[16] priv0[7], then per decision obs[16]
//     reward priv[7], then net P&L, orders, fees (ticks). priv = fair value,
//     mid, gap, session second, fill price (NaN: none or two orders), orders,
//     cash (of_train_live's probe track).
//   of15_refdump tape <seconds> <sim_seed> <out.csv>
//       default-market simulator ticks (time unit s) for replay tests
//   of15_refdump replay <csv> <time_unit> <window_s> <warmup_s> <first_window> <windows> <out.bin>
//       ReplayScalpEnv::seek(first_window + k), k = 0 .. windows - 1: obs0[16],
//       per decision obs[16] reward, then net P&L, orders, fees.
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <memory>
#include <string>
#include <vector>

#include "rag/orderflow/replay_env.hpp"
#include "rag/orderflow/scalp_run.hpp"
#include "rag/orderflow/tick_csv.hpp"

using namespace rag::orderflow;
using rag::agent::Action;
using rag::agent::Rng;

namespace {

Action pattern(unsigned long long t, unsigned long long k) { return static_cast<Action>((t * 7 + k * 3 + t / 13) % 3); }

struct Out {
  std::FILE* f;
  void put(const double* x, std::size_t n) { std::fwrite(x, sizeof(double), n, f); }
  void put(double x) { put(&x, 1); }
};

double cash(const ScalpEnv& e, int position) { return e.net_pnl_ticks() + e.fees_ticks() - position * e.market().mid(); }

void put_priv(Out& o, const ScalpEnv& e, double start, double fill, int position) {
  const double fv = e.market().fair_value(), mid = e.market().mid();
  const double p[7] = {fv, mid, fv - mid, start + static_cast<double>(e.market().t_ns()) * 1e-9, fill, static_cast<double>(e.orders()),
                       cash(e, position)};
  o.put(p, 7);
}

int scalp(int argc, char** argv) {
  if (argc != 8) return 2;
  const ScalpParams p = scalp_params(argv[2], std::atof(argv[3]));
  const std::string mode = argv[4];
  const unsigned long long s = std::strtoull(argv[5], nullptr, 10);
  const unsigned long long episodes = std::strtoull(argv[6], nullptr, 10);
  if (mode != "train" && mode != "seeds") return 2;
  Out out{std::fopen(argv[7], "wb")};
  if (!out.f) return 1;
  Rng stream(s);
  for (unsigned long long k = 0; k < episodes; ++k) {
    Rng own(s + k);
    Rng& r = mode == "train" ? stream : own;
    Rng peek = r;
    const double start = ScalpEnv::draw_market(p, peek).start;
    ScalpEnv env(p);
    auto o = env.reset(r);
    auto x = env.features(o);
    out.put(x.data(), x.size());
    put_priv(out, env, start, std::nan(""), o.position);
    for (unsigned long long t = 0;; ++t) {
      const double c0 = cash(env, o.position);
      const auto orders0 = env.orders();
      const auto st = env.step(pattern(t, k), r);
      const int dpos = st.obs.position - o.position;
      double fill = std::nan("");
      if (env.orders() - orders0 == 1 && dpos != 0) fill = -(cash(env, st.obs.position) - c0) / dpos;
      o = st.obs;
      x = env.features(o);
      out.put(x.data(), x.size());
      out.put(st.reward);
      put_priv(out, env, start, fill, o.position);
      if (st.done) break;
    }
    out.put(env.net_pnl_ticks());
    out.put(env.orders());
    out.put(env.fees_ticks());
  }
  std::fclose(out.f);
  return 0;
}

int tape(int argc, char** argv) {
  if (argc != 5) return 2;
  const double seconds = std::atof(argv[2]);
  EsSimulator sim(SimParams{}, std::strtoull(argv[3], nullptr, 10));
  std::FILE* f = std::fopen(argv[4], "wb");
  if (!f) return 1;
  write_sim_ticks(sim, static_cast<std::size_t>(std::lround(seconds / SimParams{}.dt)), [&](const char* s, std::size_t n) { std::fwrite(s, 1, n, f); });
  std::fclose(f);
  return 0;
}

int replay(int argc, char** argv) {
  if (argc != 9) return 2;
  TickCsvOptions o;
  o.time_unit = argv[3];
  auto t = std::make_shared<const TickTape>(TickTape::from_csv(argv[2], o));
  ReplayParams rp;
  rp.window_seconds = std::atof(argv[4]);
  rp.warmup_seconds = std::atof(argv[5]);
  const unsigned long long first = std::strtoull(argv[6], nullptr, 10), n = std::strtoull(argv[7], nullptr, 10);
  Out out{std::fopen(argv[8], "wb")};
  if (!out.f) return 1;
  Rng unused(0);
  for (unsigned long long k = 0; k < n; ++k) {
    ReplayScalpEnv env(t, rp);
    auto ob = env.seek(first + k);
    auto x = env.features(ob);
    out.put(x.data(), x.size());
    for (unsigned long long s = 0;; ++s) {
      const auto st = env.step(pattern(s, k), unused);
      x = env.features(st.obs);
      out.put(x.data(), x.size());
      out.put(st.reward);
      if (st.done) break;
    }
    out.put(env.net_pnl_ticks());
    out.put(env.orders());
    out.put(env.fees_ticks());
  }
  std::fclose(out.f);
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const std::string cmd = argc > 1 ? argv[1] : "";
    int rc = 2;
    if (cmd == "scalp") rc = scalp(argc, argv);
    else if (cmd == "tape") rc = tape(argc, argv);
    else if (cmd == "replay") rc = replay(argc, argv);
    if (rc == 2) std::fprintf(stderr, "usage: see examples/of15_refdump.cpp\n");
    return rc;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "of15_refdump: %s\n", e.what());
    return 1;
  }
}
