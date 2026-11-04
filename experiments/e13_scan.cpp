#include <cstdio>
#include "rag/agent/market_making.hpp"
#include "rag/agent/trainer.hpp"
using namespace rag::agent;
int main() {
  auto sym = [](const MarketMakingObs&) { return MarketMakingEnv::kSymmetricAction; };
  struct C { const char* name; MarketMakingParams p; };
  MarketMakingParams base;
  std::vector<C> cs;
  for (double d : {0.3, 3.0}) { auto p = base; p.decay = d; cs.push_back({d < 1 ? "decay 0.3 (deep book)" : "decay 3 (thin book)", p}); }
  for (double r : {0.0, 0.5}) { auto p = base; p.risk_aversion = r; cs.push_back({r == 0 ? "risk 0" : "risk 0.5", p}); }
  { auto p = base; p.fill_rate = 0.3; cs.push_back({"fill 0.3", p}); }
  const MarketMakingEnv ea(base); const MarketMakingDP da(base);
  const double oa = compare(ea, [&](const MarketMakingObs& o) { return da.act(o); }, sym, 1000, 5'000'000).mean;
  for (const auto& c : cs) {
    const MarketMakingEnv eb(c.p); const MarketMakingDP db(c.p);
    const double ob = compare(eb, [&](const MarketMakingObs& o) { return db.act(o); }, sym, 1000, 5'000'000).mean;
    std::printf("%-22s gain %.1f | default optimum in it: %5.0f%% | its optimum in default: %5.0f%%  (t=25 quotes q=0: %u/%u, q=4: %u/%u)\n", c.name, ob,
                100 * compare(eb, [&](const MarketMakingObs& o) { return da.act(o); }, sym, 1000, 5'000'000).mean / ob,
                100 * compare(ea, [&](const MarketMakingObs& o) { return db.act(o); }, sym, 1000, 5'000'000).mean / oa,
                db.act({25, 0}) / 3, db.act({25, 0}) % 3, db.act({25, 4}) / 3, db.act({25, 4}) % 3);
  }
}
