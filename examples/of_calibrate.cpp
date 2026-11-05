// of_calibrate — compare the ES simulator with recorded ES ticks, and fit its
// parameters so the simulated market has the same stylized facts.
//
//   of_calibrate sim [seconds=10800] [market]        facts of the simulator (market: default, efficient, discovery)
//   of_calibrate facts <ticks.csv> [options]         facts of recorded ticks
//   of_calibrate fit   <ticks.csv> [options]         fit the simulator to them (needs NLopt)
//   of_calibrate export <out.csv> [seconds=3600] [informed_scale=1]
//                                                    write simulator ticks in the CSV format
//                                                    (sample data; calibration self-tests)
//
// options: --tick 0.25  --time ns|us|ms|s  --evals 150  --seconds 10800
//          --seeds 2 (simulator runs averaged per evaluation, in parallel
//          threads: the facts of one short run are noisy, and the fit chases
//          the noise; on a 24-thread machine use e.g. --seeds 12 --seconds 21600)
// The CSV format is documented in rag/orderflow/tick_csv.hpp.
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "rag/orderflow/stylized.hpp"
#include "rag/orderflow/tick_csv.hpp"
#if defined(RAG_HAVE_NLOPT)
#include "rag/agent/tuning.hpp"
#endif

using namespace rag::orderflow;

namespace {

void print_facts(const char* label, const StylizedFacts& f) {
  std::printf("%s (%.1f hours)\n", label, f.seconds / 3600);
  std::printf("  activity     %.1f trades/s, %.2f contracts/trade, %.0f contracts/min\n", f.trades_per_sec, f.mean_trade_size, f.volume_per_min);
  std::printf("  book         spread = 1 tick %.1f%%, mean best-level size %.0f\n", 100 * f.spread1_share, f.mean_l1_size);
  std::printf("  returns      1-min sd %.2f ticks, excess kurtosis %.2f, lag-1 autocorr %+.3f, |r| lag-1 %+.3f\n", f.ret1m_sd, f.ret1m_kurtosis,
              f.ret1m_ac1, f.absret1m_ac1);
  std::printf("  trade signs  autocorr lag 1 %+.3f, lag 10 %+.3f, lag 100 %+.3f\n", f.sign_ac1, f.sign_ac10, f.sign_ac100);
  std::printf("  order flow   R^2(10 s mid change ~ 10 s delta) %.3f, corr(30 s delta, next 30 s) %+.3f\n", f.impact_r2, f.delta_predict);
  std::printf("  markouts     after an aggressive print: 1 s %+.3f, 5 s %+.3f, 30 s %+.3f ticks; mid changes %.1f per minute\n", f.markout_1s,
              f.markout_5s, f.markout_30s, f.mid_changes_per_min);
}

struct Options {
  double tick = 0.25, seconds = 10800;
  std::string time = "ns";
  int evals = 150, seeds = 2;
};

Options parse(int argc, char** argv, int from) {
  Options o;
  for (int i = from; i + 1 < argc; i += 2) {
    const std::string k = argv[i];
    if (k == "--tick") o.tick = std::atof(argv[i + 1]);
    else if (k == "--time") o.time = argv[i + 1];
    else if (k == "--evals") o.evals = std::atoi(argv[i + 1]);
    else if (k == "--seconds") o.seconds = std::atof(argv[i + 1]);
    else if (k == "--seeds") o.seeds = std::max(1, std::atoi(argv[i + 1]));
  }
  return o;
}

StylizedFacts read_facts(const std::string& path, const Options& o) {
  StylizedMeter m(0.1);
  const auto st = replay_tick_csv(path, m, TickCsvOptions{.tick_size = o.tick, .time_unit = o.time});
  std::printf("read %zu lines: %zu trades, %zu quotes, %zu buckets of 100 ms, %zu skipped, %zu session gaps\n", st.lines, st.trades, st.quotes,
              st.buckets, st.skipped, st.gaps);
  return m.facts();
}

#if defined(RAG_HAVE_NLOPT)
// Relative squared error on the facts that matter for trading, weighted.
double loss(const StylizedFacts& sim, const StylizedFacts& real) {
  auto rel = [](double a, double b, double floor) { const double d = (a - b) / std::max(std::abs(b), floor); return d * d; };
  auto abs2 = [](double a, double b, double scale) { const double d = (a - b) / scale; return d * d; };
  return 1.0 * rel(sim.trades_per_sec, real.trades_per_sec, 1) + 1.0 * rel(sim.volume_per_min, real.volume_per_min, 100) +
         1.0 * rel(sim.mean_l1_size, real.mean_l1_size, 10) + 2.0 * rel(sim.ret1m_sd, real.ret1m_sd, 0.5) +
         1.0 * abs2(sim.spread1_share, real.spread1_share, 0.05) + 1.0 * abs2(sim.sign_ac1, real.sign_ac1, 0.05) +
         1.0 * abs2(sim.sign_ac100, real.sign_ac100, 0.05) + 2.0 * abs2(sim.impact_r2, real.impact_r2, 0.05) +
         4.0 * abs2(sim.delta_predict, real.delta_predict, 0.03) + 2.0 * abs2(sim.markout_5s, real.markout_5s, 0.1) +
         1.0 * abs2(sim.markout_1s, real.markout_1s, 0.1) + 1.0 * abs2(sim.markout_30s, real.markout_30s, 0.2) +
         1.0 * rel(sim.mid_changes_per_min, real.mid_changes_per_min, 1) +
         0.5 * abs2(std::log1p(std::max(0.0, sim.ret1m_kurtosis)), std::log1p(std::max(0.0, real.ret1m_kurtosis)), 0.5) +
         1.0 * abs2(sim.ret1m_ac1, real.ret1m_ac1, 0.05);
}
// Facts averaged over several simulator seeds, run in parallel threads (on
// a many-core machine more seeds cost no extra wall-clock time).
StylizedFacts averaged_facts(const SimParams& p, double seconds, int seeds, std::uint64_t first_seed) {
  std::vector<StylizedFacts> runs(static_cast<std::size_t>(seeds));
  std::vector<std::thread> pool;
  for (int i = 0; i < seeds; ++i)
    pool.emplace_back([&, i] { runs[static_cast<std::size_t>(i)] = simulate_facts(p, seconds, first_seed + static_cast<std::uint64_t>(i)); });
  for (auto& t : pool) t.join();
  StylizedFacts a;
  for (const auto& f : runs) {
    a.seconds += f.seconds;
    a.trades_per_sec += f.trades_per_sec / seeds;
    a.mean_trade_size += f.mean_trade_size / seeds;
    a.volume_per_min += f.volume_per_min / seeds;
    a.spread1_share += f.spread1_share / seeds;
    a.mean_l1_size += f.mean_l1_size / seeds;
    a.ret1m_sd += f.ret1m_sd / seeds;
    a.ret1m_kurtosis += f.ret1m_kurtosis / seeds;
    a.ret1m_ac1 += f.ret1m_ac1 / seeds;
    a.absret1m_ac1 += f.absret1m_ac1 / seeds;
    a.sign_ac1 += f.sign_ac1 / seeds;
    a.sign_ac10 += f.sign_ac10 / seeds;
    a.sign_ac100 += f.sign_ac100 / seeds;
    a.impact_r2 += f.impact_r2 / seeds;
    a.delta_predict += f.delta_predict / seeds;
    a.markout_5s += f.markout_5s / seeds;
    a.markout_1s += f.markout_1s / seeds;
    a.markout_30s += f.markout_30s / seeds;
    a.mid_changes_per_min += f.mid_changes_per_min / seeds;
  }
  return a;
}
#endif

}  // namespace

int main(int argc, char** argv) {
  const std::string mode = argc > 1 ? argv[1] : "";
  if (mode == "export" && argc > 2) {
    const double seconds = argc > 3 ? std::atof(argv[3]) : 3600;
    SimParams p;
    if (argc > 4) p.informed_meta_rate *= std::atof(argv[4]);  // a market with a different informed share
    EsSimulator sim(p, 7, 3600);
    std::FILE* out = std::fopen(argv[2], "w");
    if (!out) return 1;
    std::fprintf(out, "# ES simulator ticks: T,time_s,price,size,side / Q,time_s,bid,ask,bid sizes x5,ask sizes x5\n");
    for (std::size_t s = 0, n = static_cast<std::size_t>(seconds / p.dt); s < n; ++s) {
      sim.step();
      const double t = static_cast<double>(sim.t_ns()) * 1e-9;
      for (const auto& tr : sim.trades())
        std::fprintf(out, "T,%.3f,%.2f,%u,%c\n", t, tr.price * 0.25, tr.size, tr.aggressor == Side::Buy ? 'B' : 'S');
      const Quote q = sim.quote();
      std::fprintf(out, "Q,%.3f,%.2f,%.2f", t + 0.0005, q.bid * 0.25, q.ask * 0.25);
      for (auto v : q.bid_size) std::fprintf(out, ",%u", v);
      for (auto v : q.ask_size) std::fprintf(out, ",%u", v);
      std::fprintf(out, "\n");
    }
    std::fclose(out);
    return 0;
  }
  if (mode == "sim") {
    const double seconds = argc > 2 ? std::atof(argv[2]) : 10800;
    const std::string market = argc > 3 ? argv[3] : "default";
    SimParams p;
    try {
      p = market_preset(market);
    } catch (const std::invalid_argument& e) {
      std::fprintf(stderr, "%s\n", e.what());
      return 2;
    }
    print_facts(("simulator, " + market + " parameters").c_str(), simulate_facts(p, seconds));
    return 0;
  }
  if ((mode == "facts" || mode == "fit") && argc > 2) {
    const Options o = parse(argc, argv, 3);
    const StylizedFacts real = read_facts(argv[2], o);
    print_facts("recorded ticks", real);
    print_facts("simulator, default parameters", simulate_facts(SimParams{}, o.seconds));
    if (mode == "facts") return 0;
#if defined(RAG_HAVE_NLOPT)
    using rag::agent::tuning::Param;
    const std::vector<Param> space = {{"noise_rate", 0.3, 10, true, false},        {"informed_meta_rate", 0.05, 3, true, false},
                                      {"uninformed_meta_rate", 0.05, 3, true, false}, {"value_vol", 0.05, 1.5, true, false},
                                      {"maker_rate", 1, 40, true, false},            {"cancel_rate", 0.02, 0.5, true, false},
                                      {"fade", 0, 2, false, false},                  {"momentum_rate", 0.05, 4, true, false}};
    auto apply = [&](const std::vector<double>& v) {
      SimParams p;
      p.noise_rate = v[0];
      p.informed_meta_rate = v[1];
      p.uninformed_meta_rate = v[2];
      p.value_vol = v[3];
      p.maker_rate = v[4];
      p.cancel_rate = v[5];
      p.fade = v[6];
      p.momentum_rate = v[7];
      p.taker_size_mean = std::max(1.0, real.mean_trade_size);
      return p;
    };
    const SimParams d;
    const std::vector<double> start = {d.noise_rate, d.informed_meta_rate, d.uninformed_meta_rate, d.value_vol,
                                       d.maker_rate, d.cancel_rate,        d.fade,                 d.momentum_rate};
    std::printf("\nfitting %zu parameters with %d simulator runs of %.1f h each...\n", space.size(), o.evals, o.seconds / 3600);
    const auto r = rag::agent::tuning::maximize_global(
        space, start, [&](const std::vector<double>& v) { return -loss(averaged_facts(apply(v), o.seconds, o.seeds, 1), real); }, o.evals * 2 / 3,
        o.evals - o.evals * 2 / 3, 1);
    const SimParams best = apply(r.best);
    std::printf("loss %.3f -> %.3f (%s)\n", loss(averaged_facts(SimParams{}, o.seconds, o.seeds, 1), real), -r.best_score, r.status.c_str());
    const StylizedFacts check = averaged_facts(best, o.seconds, o.seeds, 1000);
    std::printf("loss on fresh seeds: %.3f\n", loss(check, real));
    print_facts("simulator, fitted parameters (fresh seeds)", check);
    std::printf("\nSimParams fitted;\n");
    for (std::size_t i = 0; i < space.size(); ++i) std::printf("fitted.%s = %.6g;\n", space[i].name.c_str(), r.best[i]);
    std::printf("fitted.taker_size_mean = %.6g;\n", best.taker_size_mean);
    return 0;
#else
    std::fprintf(stderr, "fit needs NLopt (build with libnlopt-cxx-dev / RAG_OPTLIBS); facts mode works without it\n");
    return 1;
#endif
  }
  std::fprintf(stderr, "usage: of_calibrate sim [seconds] | facts <ticks.csv> [--tick 0.25 --time ns] | fit <ticks.csv> [--evals 150 --seconds 10800]\n");
  return 2;
}
