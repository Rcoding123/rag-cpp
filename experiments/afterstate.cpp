// Afterstate decomposition: Q(s,a) = C[q] + W(t, I - q, g). W is a tile-coded
// value over afterstates (shared by every action that leads to the same
// remaining inventory); C is a per-order-size cost. Q-learning + replay.
#include <cstdio>
#include <cstdlib>
#include <vector>
#include "rag/agent/execution_dp.hpp"
#include "rag/agent/trainer.hpp"
using namespace rag::agent;
struct AfterAgent {
  ExecutionEnv env; TileCoder<3> coder; TileQ w; std::vector<double> c; Rng rng;
  double alpha = 0.1, alpha_min = 0.003; unsigned decay = 3000, eps_decay; std::uint64_t ep = 0;
  struct T { std::array<std::uint32_t, kMaxTilings> as; bool as_live; std::uint32_t q; ExecutionObs next; float r; bool done; };
  std::vector<T> replay; std::size_t head = 0;
  AfterAgent(ExecutionParams p, unsigned eps_decay, std::uint64_t seed)
      : env(p), coder(ExecutionEnv::kLo, ExecutionEnv::kHi, {8, 8, 8}, 8), w(coder.size(), 8, 1, 0.1), c(p.lots + 1, 0.0), rng(seed), eps_decay(eps_decay) {}
  double W(const ExecutionObs& o, std::uint32_t q, std::array<std::uint32_t, kMaxTilings>* idx_out = nullptr, bool* live = nullptr) const {
    const std::uint32_t rem = o.inventory - q;
    const bool l = !(rem == 0 && o.t + 1 < env.params().horizon);  // flat before the end: nothing follows
    if (live) *live = l;
    if (!l) return 0.0;
    std::array<std::uint32_t, kMaxTilings> idx;
    coder.encode({double(o.t) / env.params().horizon, double(rem) / env.params().lots, o.signal / env.params().signal_sd}, idx.data());
    if (idx_out) *idx_out = idx;
    return w.q(idx.data(), 0);
  }
  double Q(const ExecutionObs& o, Action a) const { const auto q = env.lots_for(a, o.inventory); return c[q] + W(o, q); }
  double V(const ExecutionObs& o) const { double v = -1e300; for (Action a = 0; a < 5; ++a) v = std::max(v, Q(o, a)); return v; }
  Action act(const ExecutionObs& o, bool explore) {
    const double eps = std::max(0.02, 1.0 - double(ep) / eps_decay);
    if (explore && rng.uniform() < eps) return rng.below(5);
    Action best = 0; for (Action a = 1; a < 5; ++a) if (Q(o, a) > Q(o, best)) best = a; return best;
  }
  void learn(const T& t) {
    const double target = t.r + (t.done ? 0.0 : V(t.next));
    const double cur = c[t.q] + (t.as_live ? w.q(t.as.data(), 0) : 0.0);
    const double err = target - cur;
    const double a = std::max(alpha_min, alpha / (1 + double(ep) / decay));
    c[t.q] += 0.1 * a * err;  // shared across all states: small step
    if (t.as_live) w.update(t.as.data(), 0, w.q(t.as.data(), 0) + err);  // w.update moves by alpha*err
  }
  void observe(const ExecutionObs& o, Action a, double r, const ExecutionObs& next, bool done) {
    T t; t.q = env.lots_for(a, o.inventory); W(o, t.q, &t.as, &t.as_live); t.next = next; t.r = float(r); t.done = done;
    w.set_alpha(std::max(alpha_min, alpha / (1 + double(ep) / decay)));
    learn(t);
    if (replay.size() < 100000) replay.push_back(t); else { replay[head] = t; head = (head + 1) % replay.size(); }
    for (int i = 0; i < 4; ++i) learn(replay[rng.below(std::uint32_t(replay.size()))]);
  }
};
int main(int argc, char** argv) {
  ExecutionParams p; ExecutionEnv env(p); const ExecutionDP dp(p);
  auto twap = [](const ExecutionObs&) { return ExecutionEnv::kTwapAction; };
  const double opt = compare(env, [&](const ExecutionObs& o) { return dp.act(o); }, twap, 2000, 5'000'000).mean;
  for (unsigned episodes : {2000u, 5000u, 10000u, 40000u}) {
    std::printf("%6u episodes:", episodes);
    for (std::uint64_t seed = 1; seed <= 3; ++seed) {
      AfterAgent ag(p, episodes / 2, seed);
      Rng rng(seed * 1000);
      for (unsigned e = 0; e < episodes; ++e) {
        auto o = env.reset(rng);
        for (;;) { const Action a = ag.act(o, true); auto st = env.step(a, rng); ag.observe(o, a, st.reward, st.obs, st.done); o = st.obs; if (st.done) break; }
        ++ag.ep;
      }
      std::printf(" %6.1f%%", 100 * compare(env, [&](const ExecutionObs& o) { return ag.act(o, false); }, twap, 2000, 5'000'000).mean / opt);
    }
    std::printf("\n");
  }
}
