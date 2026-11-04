// E16: a from-scratch neural Q-learner (DQN-style) on execution, vs tile coding.
#include <array>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <string>
#include <vector>
#include "rag/agent/execution_dp.hpp"
#include "rag/agent/mlp.hpp"
#include "rag/agent/trainer.hpp"
using namespace rag::agent;
constexpr std::uint32_t A = 5;
struct Tr { std::array<double, 3> s; Action a; double R; std::array<double, 3> s2; bool done; std::uint32_t steps; };
int main(int argc, char** argv) {
  const std::uint64_t seed = argc > 1 ? std::strtoull(argv[1], nullptr, 10) : 1;
  const unsigned hidden = argc > 2 ? std::atoi(argv[2]) : 64;
  const double lr = argc > 3 ? std::atof(argv[3]) : 1e-3;
  const unsigned total = argc > 4 ? std::atoi(argv[4]) : 40000;
  ExecutionParams p; ExecutionEnv env(p); const ExecutionDP dp(p);
  auto twap = [](const ExecutionObs&) { return ExecutionEnv::kTwapAction; };
  const double opt = compare(env, [&](const ExecutionObs& o) { return dp.act(o); }, twap, 1000, 77'000'000).mean;
  Rng init(seed);
  Mlp net({3, hidden, hidden, A}, init);
  std::vector<double> sc(net.scratch_size());
  std::array<double, A> q{};
  auto greedy = [&](const Mlp& m, const ExecutionObs& o) {
    const auto f = env.features(o); m.forward(f.data(), q.data(), sc.data());
    return static_cast<Action>(std::max_element(q.begin(), q.end()) - q.begin()); };
  auto share = [&](const Mlp& m) { return 100 * compare(env, [&](const ExecutionObs& o) { return greedy(m, o); }, twap, 1000, 77'000'000).mean / opt; };
  // E16a: supervised fit to Q* (representation check).
  if (argc > 5) {
    Mlp sup({3, hidden, hidden, A}, init); Rng r(1), pol(2);
    for (int ep = 0; ep < 20000; ++ep) {
      auto o = env.reset(r);
      for (;;) {
        const auto f = env.features(o);
        for (Action a = 0; a < A; ++a) sup.accumulate(f.data(), a, dp.q_value(o.t, o.inventory, o.signal, a));
        sup.step(ep < 15000 ? lr : lr / 10);
        const auto st = env.step(pol.uniform() < 0.2 ? pol.below(5) : dp.act(o), r);
        if (st.done) break; o = st.obs;
      }
    }
    std::printf("E16a hidden %u: greedy on fitted Q* = %.1f%% of optimal\n", hidden, share(sup));
    return 0;
  }
  // E16b: DQN with 3-step targets, replay, target network.
  const char* ev_t = std::getenv("E16_TARGET"); const unsigned sync = ev_t ? std::atoi(ev_t) : 2000;
  const char* ev_o = std::getenv("E16_OPT"); const bool sgd = ev_o && std::string(ev_o) == "sgd";
  const char* ev_b = std::getenv("E16_BATCH"); const int batch = ev_b ? std::atoi(ev_b) : 32;
  Mlp target = net;
  std::vector<Tr> replay; replay.reserve(100000); std::size_t head = 0;
  Rng rng(seed), explore(seed + 1000), pick(seed + 2000);
  ExecutionEnv e(p);
  std::vector<double> sc2(net.scratch_size());
  std::uint64_t updates = 0;
  const auto t0 = std::chrono::steady_clock::now();
  std::printf("DQN seed %llu hidden %u lr %g target %u opt %s batch %d:", (unsigned long long)seed, hidden, lr, sync, sgd ? "sgd" : "adam", batch);
  for (unsigned ep = 1; ep <= total; ++ep) {
    const double eps = std::max(0.02, 1.0 - 0.98 * ep / 10000.0);
    auto o = e.reset(rng);
    struct S { std::array<double, 3> f; Action a; double r; };
    std::vector<S> traj;
    for (;;) {
      const Action a = explore.uniform() < eps ? explore.below(5) : greedy(net, o);
      const auto st = e.step(a, rng);
      traj.push_back({e.features(o), a, st.reward});
      o = st.obs;
      if (st.done) break;
    }
    // 3-step returns at episode end; the bootstrap state is traj[t+3] (or terminal).
    const std::size_t T = traj.size();
    for (std::size_t t = 0; t < T; ++t) {
      Tr tr{traj[t].f, traj[t].a, 0, {}, true, 0};
      for (std::size_t k = t; k < std::min(T, t + 3); ++k) { tr.R += traj[k].r; ++tr.steps; }
      if (t + 3 < T) { tr.s2 = traj[t + 3].f; tr.done = false; }
      if (replay.size() < 100000) replay.push_back(tr); else { replay[head] = tr; head = (head + 1) % 100000; }
    }
    // Updates: one minibatch of 32 per environment step taken.
    if (replay.size() >= 2000)
      for (std::size_t u = 0; u < T; ++u) {
        for (int b = 0; b < batch; ++b) {
          const Tr& tr = replay[pick.below(static_cast<std::uint32_t>(replay.size()))];
          double y = tr.R;
          if (!tr.done) { std::array<double, A> qn; (sync ? target : net).forward(tr.s2.data(), qn.data(), sc2.data()); y += *std::max_element(qn.begin(), qn.end()); }
          net.accumulate(tr.s.data(), tr.a, y);
        }
        if (sgd) net.sgd_step(lr); else net.step(lr);
        if (sync && ++updates % sync == 0) target = net;
      }
    if (ep == 5000 || ep == 10000 || ep == 20000 || ep == 40000 || ep == total)
      std::printf("  %5u: %5.1f%%", ep, share(net)), std::fflush(stdout);
  }
  std::printf("  (%.0f s)\n", std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
}
