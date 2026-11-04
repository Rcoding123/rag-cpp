// Where does the return variance come from? Components of an episode's return under a policy.
#include <cmath>
#include <cstdio>
#include "rag/agent/execution_dp.hpp"
#include "rag/agent/trainer.hpp"
using namespace rag::agent;
int main() {
  ExecutionParams p; p.noise_sd = 0.2; const ExecutionDP dp(p);
  for (int k = 0; k < 2; ++k) {
    double s[4] = {}, s2[4] = {}; const int n = 20000;
    for (int i = 0; i < n; ++i) {
      ExecutionEnv env(p); Rng rng(1 + i); auto o = env.reset(rng);
      double c[4] = {};  // 0 innovation I'*noise, 1 drift I'*alpha*g, 2 costs (rest), 3 total
      for (;;) {
        const double g = o.signal;
        const Action a = k ? dp.act(o) : ExecutionEnv::kTwapAction;
        const auto st = env.step(a, rng);
        const double move = env.last_move(), held = st.obs.inventory;
        const double drift = (move != 0.0) ? held * p.alpha * g : 0.0, innov = (move != 0.0) ? held * (move - p.alpha * g) : 0.0;
        c[0] += innov; c[1] += drift; c[2] += st.reward - innov - drift; c[3] += st.reward;
        o = st.obs; if (st.done) break;
      }
      for (int j = 0; j < 4; ++j) { s[j] += c[j]; s2[j] += c[j] * c[j]; }
    }
    std::printf("%s: ", k ? "optimal" : "TWAP   ");
    const char* names[] = {"innovation I'*eps", "drift I'*alpha*g", "costs", "total"};
    for (int j = 0; j < 4; ++j) { const double m = s[j] / n; std::printf("  %s mean %+7.2f sd %6.2f |", names[j], m, std::sqrt(s2[j] / n - m * m)); }
    std::printf("\n");
  }
}
