// of_train_live — train one scalping agent with of_train's recipe and stream
// checkpoints for a live training animation (tools/of_train_live.py).
//
//   of_train_live [--episodes 2000] [--every N] [--seed 1] [--market default|efficient|discovery]
//                 [--warmup 150] [--hidden 64] [--gamma 0.97] [--nstep 5] [--batch 32] [--lr 0.001]
//                 [--probe 5] [--out live.jsonl] [--net PATH]
//
// Appends one JSON line per checkpoint to --out (flushed, so a page polling
// the file follows training live): every --every episodes (default
// episodes / 40) and once before training. A checkpoint holds the training
// returns since the last one, the Trainer's validation points (700M+, as in
// of_train), and the current learner's greedy behaviour on fixed probe
// episodes from the visualization block (960,000,000+, docs/ai/NEXT_SESSION.md
// section 4): the first probe step by step (mid, hidden fair value for display
// only, position, fills, Q-values, P&L), the P&L of all probes, and a policy
// map: the learner's action on a fixed set of states (the probe episodes as
// seen by a flat strategy), binned by L1 imbalance x 30 s delta, given flat,
// long or short. The 30 s delta rule on the same probes is in the header.
//
// The probes are for looking only: never select or score on them. --net
// saves the champion (Trainer's choice on validation, as of_train) with its
// record, so of_trace can replay it.
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "rag/agent/trainer.hpp"
#include "rag/orderflow/scalp_run.hpp"

using namespace rag::agent;
using namespace rag::orderflow;

namespace {

constexpr std::uint64_t kProbeFirst = 960'000'000;  // visualization block
constexpr unsigned kImbBins = 10, kDeltaBins = 8;   // as tools/of_viz.py

struct Options {
  unsigned episodes = 2000, every = 0, probe = 5, hidden = 64, nstep = 5, batch = 32;
  std::uint64_t seed = 1;
  double gamma = 0.97, lr = 1e-3, warmup = kDefaultWarmup;
  std::string market = "default", out = "live.jsonl", net;
};

// of_train's agent (NeuralQ, Double DQN, weight averaging).
AgentConfig<ScalpEnv::kStateDims> config(const Options& o) {
  AgentConfig<ScalpEnv::kStateDims> c;
  c.seed = o.seed;
  c.learner = LearnerKind::NeuralQ;
  c.bins.fill(1);
  c.double_q = true;
  c.nn_polyak = 1e-4;
  c.nn_hidden = o.hidden;
  c.nn_batch = o.batch;
  c.nn_lr = o.lr;
  c.gamma = o.gamma;
  c.n_step = o.nstep;
  c.eps_start = 1.0;
  c.eps_end = 0.02;
  c.eps_decay_episodes = std::max(1u, o.episodes / 2);
  return c;
}

Action delta_rule(const ScalpObs& o) { return o.f[1] > 1.5 ? 2 : o.f[1] < -1.5 ? 0 : static_cast<Action>(o.position + 1); }
double cash(const ScalpEnv& e, int position) { return e.net_pnl_ticks() + e.fees_ticks() - position * e.market().mid(); }
double imbalance(const ScalpEnv& e) {
  const Quote q = e.market().quote();
  const double b = q.bid_size[0], a = q.ask_size[0];
  return b + a > 0 ? (b - a) / (b + a) : 0.0;
}
unsigned bin(double x, double lo, double hi, unsigned n) {
  const double k = std::floor((x - lo) / (hi - lo) * n);
  return static_cast<unsigned>(std::clamp(k, 0.0, n - 1.0));
}

// JSON numbers ("null" for NaN/inf).
std::string num(double x, const char* f = "%.4g") {
  if (!std::isfinite(x)) return "null";
  char b[48];
  std::snprintf(b, sizeof b, f, x);
  return b;
}
template <class V>
std::string arr(const V& v, const char* f = "%.4g") {
  std::string s = "[";
  for (std::size_t i = 0; i < v.size(); ++i) s += (i ? "," : "") + num(static_cast<double>(v[i]), f);
  return s + "]";
}

std::string json_string(const std::string& s) {
  std::string r = "\"";
  for (char c : s) {
    if (c == '"' || c == '\\') r += '\\';
    if (static_cast<unsigned char>(c) >= 0x20) r += c;
  }
  return r + "\"";
}

// One episode step by step (the probe the page replays).
struct Track {
  std::vector<double> tod, mid, fv, pnl, fill, qs, qf, ql;
  std::vector<int> pos;
  std::string json() const {
    return "{\"tod\":" + arr(tod, "%.1f") + ",\"mid\":" + arr(mid, "%.1f") + ",\"fv\":" + arr(fv, "%.2f") + ",\"pos\":" + arr(pos, "%.0f") +
           ",\"pnl\":" + arr(pnl, "%.2f") + ",\"fill\":" + arr(fill, "%.2f") + ",\"qs\":" + arr(qs, "%.3f") + ",\"qf\":" + arr(qf, "%.3f") +
           ",\"ql\":" + arr(ql, "%.3f") + "}";
  }
};

struct EpisodeResult {
  double pnl = 0;
  unsigned orders = 0, long_s = 0, short_s = 0, steps = 0;
};

// Runs one probe episode with `policy(obs, q_out) -> action`; fills `track` if given.
template <class Policy>
EpisodeResult run_probe(const ScalpParams& p, std::uint64_t seed, Policy&& policy, Track* track) {
  ScalpEnv env(p);
  Rng r(seed);
  const double start = ScalpEnv::draw_market(p, r).start;  // the session second reset() will draw
  Rng rr(seed);
  ScalpObs o = env.reset(rr);
  EpisodeResult res;
  for (;;) {
    std::array<double, ScalpEnv::kNumActions> q{};
    q.fill(std::nan(""));
    const Action a = policy(o, q);
    const double c0 = cash(env, o.position);
    const auto orders0 = env.orders();
    const double mid = env.market().mid(), fv = env.market().fair_value(), tod = start + static_cast<double>(env.market().t_ns()) * 1e-9;
    const auto st = env.step(a, rr);
    const int dpos = st.obs.position - o.position;
    if (track) {
      double fill = std::nan("");
      if (env.orders() - orders0 == 1 && dpos != 0) fill = -(cash(env, st.obs.position) - c0) / dpos;
      track->tod.push_back(tod);
      track->mid.push_back(mid);
      track->fv.push_back(fv);
      track->pos.push_back(st.obs.position);
      track->pnl.push_back(env.net_pnl_ticks());
      track->fill.push_back(fill);
      track->qs.push_back(q[0]);
      track->qf.push_back(q[1]);
      track->ql.push_back(q[2]);
    }
    res.long_s += st.obs.position > 0;
    res.short_s += st.obs.position < 0;
    ++res.steps;
    o = st.obs;
    if (st.done) break;
  }
  res.pnl = env.net_pnl_ticks();
  res.orders = env.orders();
  return res;
}

}  // namespace

int main(int argc, char** argv) {
  Options o;
  for (int i = 1; i + 1 < argc; i += 2) {
    const std::string k = argv[i], v = argv[i + 1];
    const int iv = std::atoi(v.c_str());
    if (k == "--episodes") o.episodes = static_cast<unsigned>(std::max(1, iv));
    else if (k == "--every") o.every = static_cast<unsigned>(std::max(1, iv));
    else if (k == "--probe") o.probe = static_cast<unsigned>(std::clamp(iv, 1, 1000));
    else if (k == "--seed") o.seed = static_cast<std::uint64_t>(std::max(1, iv));
    else if (k == "--hidden") o.hidden = static_cast<unsigned>(std::max(4, iv));
    else if (k == "--nstep") o.nstep = static_cast<unsigned>(std::max(1, iv));
    else if (k == "--batch") o.batch = static_cast<unsigned>(std::max(1, iv));
    else if (k == "--gamma") o.gamma = std::atof(v.c_str());
    else if (k == "--lr") o.lr = std::atof(v.c_str());
    else if (k == "--market") o.market = v;
    else if (k == "--out") o.out = v;
    else if (k == "--net") o.net = v;
    else if (k == "--warmup") {
      const auto w = parse_warmup(v);
      if (!w) {
        std::fprintf(stderr, "--warmup wants a number of seconds >= 0, got %s\n", v.c_str());
        return 2;
      }
      o.warmup = *w;
    } else {
      std::fprintf(stderr, "unknown option %s\n", k.c_str());
      return 2;
    }
  }
  if (argc % 2 == 0) {
    std::fprintf(stderr, "%s needs a value\n", argv[argc - 1]);
    return 2;
  }
  if (o.every == 0) o.every = std::max(1u, o.episodes / 40);
  ScalpParams p;
  try {
    p = scalp_params(o.market, o.warmup);
  } catch (const std::invalid_argument& e) {
    std::fprintf(stderr, "--market: %s\n", e.what());
    return 2;
  }
  if (const auto err = warmup_error(p); !err.empty()) {
    std::fprintf(stderr, "--warmup: %s\n", err.c_str());
    return 2;
  }
  std::ofstream out(o.out, std::ios::trunc);
  if (!out) {
    std::fprintf(stderr, "cannot write %s\n", o.out.c_str());
    return 1;
  }
  auto emit = [&](const std::string& line) {
    out << line << '\n';
    out.flush();
  };

  // The probes: the rule's results, and the policy-map states (each probe as a flat strategy sees it).
  struct MapState {
    ScalpObs obs;
    unsigned bi, bd;
  };
  std::vector<MapState> states;
  std::vector<double> rule_pnl;
  Track rule_track;
  for (unsigned i = 0; i < o.probe; ++i) {
    const std::uint64_t s = kProbeFirst + i;
    rule_pnl.push_back(run_probe(p, s, [](const ScalpObs& ob, auto&) { return delta_rule(ob); }, i == 0 ? &rule_track : nullptr).pnl);
    ScalpEnv env(p);
    Rng r(s);
    ScalpObs ob = env.reset(r);
    for (;;) {
      states.push_back({ob, bin(imbalance(env), -1, 1, kImbBins), bin(ob.f[1], -kClip, kClip, kDeltaBins)});
      const auto st = env.step(1, r);
      ob = st.obs;
      if (st.done) break;
    }
  }
  std::vector<unsigned> cell_states(kImbBins * kDeltaBins, 0);
  for (const auto& m : states) ++cell_states[m.bi * kDeltaBins + m.bd];
  char head[512];
  std::snprintf(head, sizeof head,
                "{\"type\":\"header\",\"market\":\"%s\",\"warmup\":%g,\"episodes\":%u,\"every\":%u,\"seed\":%llu,\"hidden\":%u,\"gamma\":%g,"
                "\"nstep\":%u,\"lr\":%g,\"probe\":%u,\"probe_seed\":%llu,\"imb_bins\":%u,\"delta_bins\":%u,\"fee_dollars\":%g,",
                o.market.c_str(), o.warmup, o.episodes, o.every, static_cast<unsigned long long>(o.seed), o.hidden, o.gamma, o.nstep, o.lr,
                o.probe, static_cast<unsigned long long>(kProbeFirst), kImbBins, kDeltaBins, p.fee_dollars);
  emit(std::string(head) + "\"cell_states\":" + arr(cell_states, "%.0f") + ",\"rule_pnl\":" + arr(rule_pnl, "%.2f") +
       ",\"rule\":" + rule_track.json() + "}");

  ScalpEnv env(p);
  Agent<ScalpEnv> agent(env, config(o));
  Agent<ScalpEnv> viewer(env, config(o));  // the learner's averaged (greedy) weights, copied at each checkpoint
  Trainer<ScalpEnv> tr(agent, env,
                       {.eval_every = std::max(1u, o.episodes / 20), .eval_episodes = 40, .eval_seed = 700'000'000, .train_seed = o.seed,
                        .detect_drift = false});
  std::vector<EvalPoint> evals;
  const auto t0 = std::chrono::steady_clock::now();
  double champion_score = std::nan("");

  auto checkpoint = [&](unsigned done, const std::vector<double>& returns) {
    viewer.restore(agent.snapshot());
    auto greedy = [&](const ScalpObs& ob, std::array<double, ScalpEnv::kNumActions>& q) {
      viewer.q_values(ob, q.data());
      return static_cast<Action>(std::max_element(q.begin(), q.end()) - q.begin());
    };
    Track track;
    std::vector<double> pnl;
    double orders = 0, long_s = 0, short_s = 0, steps = 0;
    for (unsigned i = 0; i < o.probe; ++i) {
      const auto r = run_probe(p, kProbeFirst + i, greedy, i == 0 ? &track : nullptr);
      pnl.push_back(r.pnl);
      orders += r.orders;
      long_s += r.long_s;
      short_s += r.short_s;
      steps += r.steps;
    }
    // policy[held + 1][imbalance bin][delta bin][action]: actions on the fixed states
    std::vector<unsigned> policy(3 * kImbBins * kDeltaBins * 3, 0);
    std::array<double, ScalpEnv::kNumActions> q{};
    for (const auto& m : states) {
      for (int held = -1; held <= 1; ++held) {
        ScalpObs ob = m.obs;
        ob.position = held;
        ob.open_pnl = 0;
        const Action a = greedy(ob, q);
        ++policy[((static_cast<unsigned>(held + 1) * kImbBins + m.bi) * kDeltaBins + m.bd) * 3 + a];
      }
    }
    std::string ev = "[";
    for (std::size_t i = 0; i < evals.size(); ++i) {
      const auto& e = evals[i];
      ev += (i ? "," : "") + std::string("{\"episode\":") + std::to_string(e.episode) + ",\"challenger\":" + num(e.challenger) +
            ",\"champion\":" + num(e.champion) + ",\"promoted\":" + (e.promoted ? "true" : "false") + "}";
    }
    ev += "]";
    evals.clear();
    const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    double mean = 0;
    for (double x : pnl) mean += x / static_cast<double>(pnl.size());
    emit("{\"type\":\"ckpt\",\"episode\":" + std::to_string(done) + ",\"elapsed\":" + num(secs, "%.1f") + ",\"epsilon\":" + num(agent.epsilon()) +
         ",\"returns\":" + arr(returns, "%.2f") + ",\"evals\":" + ev + ",\"probe_pnl\":" + arr(pnl, "%.2f") +
         ",\"orders\":" + num(orders / o.probe, "%.2f") + ",\"long_share\":" + num(long_s / steps, "%.4f") +
         ",\"short_share\":" + num(short_s / steps, "%.4f") + ",\"policy\":" + arr(policy, "%.0f") + ",\"track\":" + track.json() + "}");
    std::printf("episode %5u  eps %.3f  probe mean %+7.2f ticks  %4.1f orders  long %3.0f%% short %3.0f%%  (%.0f s)\n", done, agent.epsilon(), mean,
                orders / o.probe, 100 * long_s / steps, 100 * short_s / steps, secs);
    std::fflush(stdout);
  };

  std::printf("of_train_live: %u episodes, checkpoint every %u, market %s, warm-up %g s, seed %llu -> %s\n", o.episodes, o.every, o.market.c_str(),
              o.warmup, static_cast<unsigned long long>(o.seed), o.out.c_str());
  checkpoint(0, {});
  for (unsigned done = 0; done < o.episodes;) {
    const unsigned n = std::min(o.every, o.episodes - done);
    const auto rep = tr.run(n, [&](const EvalPoint& e) { evals.push_back(e); });
    done += n;
    champion_score = tr.has_champion() ? rep.champion_score : std::nan("");
    checkpoint(done, rep.returns);
  }
  bool saved = false;
  if (!o.net.empty() && tr.champion()) {
    remove_net_record(o.net);
    tr.champion()->net.save(o.net);
    char extra[256];
    std::snprintf(extra, sizeof extra, "episodes=%u\nhidden=%u\ngamma=%g\nnstep=%u\nbatch=%u\nlr=%g\nseed=%llu\ntool=of_train_live\n", o.episodes, o.hidden,
                  o.gamma, o.nstep, o.batch, o.lr, static_cast<unsigned long long>(o.seed));
    saved = write_net_record(o.net, o.market, o.warmup, extra);
    if (!saved) std::fprintf(stderr, "cannot write the record of %s\n", o.net.c_str());
  }
  emit("{\"type\":\"done\",\"champion_validation\":" + num(champion_score) + ",\"net\":" + json_string(saved ? o.net : std::string()) + "}");
  std::printf("done: champion validation %+.2f ticks/episode%s%s\n", champion_score, saved ? ", saved " : "", saved ? o.net.c_str() : "");
  return out.good() ? 0 : 1;
}
