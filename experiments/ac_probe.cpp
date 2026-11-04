// Actor-critic probe: softmax policy (tile-coded logits) + tile-coded state-value critic, 1-step TD advantage.
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include "rag/agent/execution_dp.hpp"
#include "rag/agent/trainer.hpp"
using namespace rag::agent;
struct AC {
  TileCoder<3> coder; TileQ actor, critic; ExecutionEnv env; Rng rng; double a_actor, a_critic, temp;
  AC(const ExecutionParams& p, double aa, double ac, double t, std::uint64_t seed)
      : coder(ExecutionEnv::kLo, ExecutionEnv::kHi, {8, 8, 8}, 8), actor(coder.size(), 8, 5, aa), critic(coder.size(), 8, 1, ac), env(p), rng(seed), a_actor(aa), a_critic(ac), temp(t) {}
  void probs(const std::uint32_t* idx, double* pi) const {
    double h[5]; actor.q_values(idx, h); double m = *std::max_element(h, h + 5), z = 0;
    for (int a = 0; a < 5; ++a) { pi[a] = std::exp((h[a] - m) / temp); z += pi[a]; }
    for (int a = 0; a < 5; ++a) pi[a] /= z;
  }
  Action act(const ExecutionObs& o, bool explore) {
    std::array<std::uint32_t, kMaxTilings> idx; coder.encode(env.features(o), idx.data());
    double pi[5]; probs(idx.data(), pi);
    if (!explore) return static_cast<Action>(std::max_element(pi, pi + 5) - pi);
    double u = rng.uniform(); for (Action a = 0; a < 5; ++a) { u -= pi[a]; if (u <= 0) return a; } return 4;
  }
  void observe(const ExecutionObs& o, Action a, double r, const ExecutionObs& next, bool done, double scale) {
    std::array<std::uint32_t, kMaxTilings> s, s2; coder.encode(env.features(o), s.data()); coder.encode(env.features(next), s2.data());
    const double v = critic.q(s.data(), 0), target = r + (done ? 0.0 : critic.q(s2.data(), 0)), delta = target - v;
    critic.set_alpha(a_critic * scale); critic.update(s.data(), 0, target);
    double pi[5]; probs(s.data(), pi);
    actor.set_alpha(a_actor * scale);
    for (Action b = 0; b < 5; ++b) actor.update(s.data(), b, actor.q(s.data(), b) + delta * ((b == a) - pi[b]) / temp);
  }
};
int main(int argc, char** argv) {
  const double aa = argc > 1 ? std::atof(argv[1]) : 0.01, ac = argc > 2 ? std::atof(argv[2]) : 0.1, temp = argc > 3 ? std::atof(argv[3]) : 1.0;
  ExecutionParams p; ExecutionEnv env(p); const ExecutionDP dp(p);
  auto twap = [](const ExecutionObs&) { return ExecutionEnv::kTwapAction; };
  const double opt = compare(env, [&](const ExecutionObs& o) { return dp.act(o); }, twap, 2000, 5'000'000).mean;
  std::printf("actor %.4f critic %.3f temp %.2f:", aa, ac, temp);
  const unsigned budgets[] = {argc > 4 ? unsigned(std::atoi(argv[4])) : 10000u}; const unsigned nseeds = argc > 5 ? std::atoi(argv[5]) : 2;
  for (unsigned episodes : budgets) for (std::uint64_t seed = 1; seed <= nseeds; ++seed) {
    AC agent(p, aa, ac, temp, seed); Rng rng(seed * 1000);
    for (unsigned e = 0; e < episodes; ++e) {
      const double scale = 1.0 / (1 + e / 5000.0);
      auto o = env.reset(rng);
      for (;;) { const Action a = agent.act(o, true); auto st = env.step(a, rng); agent.observe(o, a, st.reward, st.obs, st.done, scale); o = st.obs; if (st.done) break; }
    }
    std::printf("  %uk/s%llu %5.1f%%", episodes / 1000, (unsigned long long)seed, 100 * compare(env, [&](const ExecutionObs& o) { return agent.act(o, false); }, twap, 2000, 5'000'000).mean / opt);
  }
  std::printf("\n");
}
