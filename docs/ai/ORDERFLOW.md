# ES order-flow trading: simulator, agents, validation, plan

The order-flow trading work, in one place: what it is for, how it is
built, how to run it (Linux and the user's Windows server), every result
with its caveats, and what to do next. Experiment numbers (OF1–OF9) refer
to probes in `experiments/`; detailed numbers for each are in the probe's
results file and summarized here.

## 1. Objectives (from the user)

- **Market:** ES (E-mini S&P 500) futures. 1 tick = 0.25 index points =
  $12.50 per contract.
- **Two strategies, two agents:**
  1. **Order-flow scalping / momentum** (directional, market orders,
     seconds to minutes): decide from delta, volume, order-flow imbalance,
     absorption and market structure.
  2. **Tick scalping** (passive, limit orders): earn the spread with queue
     position, order-flow imbalance and order-arrival information, while
     avoiding adverse selection.
- **Data:** the user has ES tick data on their server and will feed real
  order flow later. Until then everything runs on the simulator, built so
  the same feature and strategy code runs on recorded ticks unchanged.
- **Hardware:** the user's server (AMD Ryzen 9 9900X, 12 cores / 24
  threads, AVX-512, 128 GB, Windows Server 2025) does the long training;
  GPUs can be rented. Tools train many seeds in parallel threads.
- **Standard of evidence:** sealed test seeds, paired comparisons with
  rules, stress scenarios, daily P&L / drawdown / Sharpe, and, before any
  edge is believed, calibration to and backtests on real ES ticks.

## 2. Architecture

```
            ┌──────────────── tick stream: Trade{price, size, aggressor}, Quote{bid, ask, 5 levels} ───────────────┐
 simulator.hpp (EsSimulator)                                             tick_csv.hpp (recorded ticks, CSV)
            │                                                                          │
            └──────────────► features.hpp (FeatureBuilder: 13 order-flow features) ◄───┘
                                   │                          │
                 scalp_env.hpp (ScalpEnv)        replay_env.hpp (ReplayScalpEnv)   tick_env.hpp (TickScalpEnv)
                 market orders, simulator        market orders, recorded ticks      limit orders + FIFO queue
                                   │                          │                          │
                     rag::agent learners (NeuralQ + Double DQN + weight averaging), Trainer, champion selection
                                   │
     tools: of_train (parallel seeds)  of_backtest (walk-forward on ticks)  of_calibrate (facts, fit)  probes OF1–OF9
```

| File | What it is |
|---|---|
| `include/rag/orderflow/simulator.hpp` | `EsSimulator`: L2 book (16 levels/side, 5 exposed), 100 ms steps; noise takers, informed and uninformed meta-orders (power-law sizes, child orders), momentum takers, makers with capped quote fading, optional informed makers (`maker_info`), icebergs (hidden reserve: absorption); activity log-OU + intraday U-shape + mean-zero excitation; fair value with activity-scaled volatility and jumps. Strategy orders: `market_order`, `submit_limit`/`cancel` with FIFO queue position (cancellations biased to the back, faster participants joining ahead). `efficient_market()` preset. |
| `include/rag/orderflow/features.hpp` | `FeatureBuilder`: delta 5/30/120 s, OFI 5/30 s, relative volume, L1 and L1-5 imbalance, returns 5/30/120 s, VWAP distance, absorption (volume traded at a holding best level beyond the size displayed). Running window sums; self-normalized by a running RMS (works on any market without hand scales). |
| `include/rag/orderflow/scalp_env.hpp` | `ScalpEnv`: target position short/flat/long every 1 s, market orders with 100 ms latency, real fills, $1.50/side fees, reward = mark-to-market change net of fees, ends flat, fresh market and random time of day per episode (10 min + 150 s warm-up). |
| `include/rag/orderflow/tick_env.hpp` | `TickScalpEnv`: quote none/bid/ask/both or flatten every 0.5 s; 1-lot at the touch, rejoining at the back when the price moves; inventory ±1; FIFO fills; queue position in the observation; 5-min episodes. |
| `include/rag/orderflow/tick_csv.hpp` | `replay_tick_csv`: recorded ticks (`T,time,price,size,B|S` / `Q,time,bid,ask[,5+5 sizes]`) replayed in buckets into any `add_trade`/`end_bucket` sink. |
| `include/rag/orderflow/replay_env.hpp` | `TickTape` + `ReplayScalpEnv`: the scalping environment on recorded ticks (same observations, so simulator-trained agents run unchanged); fills at the recorded touch walking recorded depth; `seek(k)` for walk-forward windows. |
| `include/rag/orderflow/stylized.hpp` | `StylizedMeter`/`simulate_facts`: the same stylized facts for simulated and recorded ticks (activity, spread, L1, 1-min sd/kurtosis/autocorrelations, trade-sign memory, price-on-flow R², 30 s delta predictability, markouts at 1/5/30 s, mid changes per minute). |
| `examples/of_train.cpp` | Parallel training of scalping agents (one seed per thread), sealed scoring, mean-Q ensemble. |
| `examples/of_backtest.cpp` | Walk-forward backtest of flat, the delta rule and saved agents on a tick CSV. |
| `examples/of_calibrate.cpp` | `sim`, `facts <csv>`, `export <csv>`, `fit <csv>` (NLopt; seeds averaged in parallel threads). |
| `experiments/of*.cpp` | Probes OF1–OF9 (below), results files next to them. |
| `experiments/research_synthesis.md` | Literature synthesis from the research crawl, with a ranked change list. |
| `tools/research_crawler.py` | Anonymous, polite crawler (arXiv, OpenAlex, Semantic Scholar, Hacker News) → `research_corpus/` (gitignored) indexable with `rag_pdf`. |

Tests: `tests/orderflow_tests.cpp` (book invariants, determinism, FIFO
queue, environments) and `tests/stylized_tests.cpp` (CSV reader, facts
round trip, replay fills, NeuralQ memory).

## 3. How to run

Linux (from the repository root, after `cmake -S . -B build && cmake --build build -j`):

```bash
./build/of_train --seeds 12 --episodes 2000 --market default --out nets/   # parallel training
./build/of_backtest es_ticks.csv --time ns nets/seed1.net nets/seed2.net   # recorded ticks
./build/of_calibrate facts es_ticks.csv --time ns                           # real vs simulated facts
./build/of_calibrate fit es_ticks.csv --time ns --seeds 12 --seconds 21600 --evals 300
./build/of_calibrate export sample.csv 3600                                 # example of the CSV format
```

Windows server (Developer PowerShell, AVX-512 build; see `docs/WINDOWS.md`):
the same tools are in `build-avx512\Release\` (`of_train.exe`, ...). The
tick-scalping trainer (OF7) is the target `of_tick_train`; run one process
per seed in parallel, e.g.
`1..12 | % { Start-Process -NoNewWindow .\build-avx512\Release\of_tick_train.exe "$_ 2000 0.99 0.5 0.25 0.02 200 100 tick_seed$_.net" }`
(Linux: `./build/of_tick_train SEED 2000 0.99 0.5 0.25 0.02 200 100 out.net`).

Hardware notes: the networks are small (2 × 64 units); a GPU does not
help at this size (per-update work is microseconds). Scale with cores:
one seed per thread (`of_train`), seeds averaged in parallel
(`of_calibrate --seeds`). GPUs pay off only with much larger models
(e.g. convolutional/recurrent models over raw book sequences), which
would need a GPU framework fed by this simulator: not built yet.

## 4. Results (simulator unless stated; sealed test seeds)

Seed blocks: training = trainer streams seeded by the run seed; validation
= 700M+i; sealed tests: scalping 900M+i (OF2, OF3, OF9), tick scalping
910M+i (OF4, OF5, OF7), stress 920M+i (OF6), days 930M+i (OF8).

**OF1 — stylized facts** (`of1_stylized_facts.cpp`, 3 × 6.5 h): ~20
trades/s, 3.8 lots/trade, ~4,500 contracts/min, spread 1 tick 97%,
1-min sd 3.4–4.4 ticks, excess kurtosis 5–10, |r| autocorrelation 0.15–0.3,
trade-sign autocorrelation 0.22 → 0.16 at lag 100, R²(10 s Δmid ~ delta)
≈ 0.5, corr(30 s delta, next 30 s) ≈ 0.1. Two calibration bugs found on
the way: uncapped quote fading and raw-move activity excitation both made
the book blow up (6,000 trades/s).

**OF2 — scalping baselines** (1 contract, 300 × 10 min): flat 0; 30 s
delta momentum +1.83 ticks/episode (t 3.9); OFI momentum −1.8; L1
imbalance −27.9 (trading costs); absorption −11.5; fading −4.5; oracle
(sees fair value) +11.1.

**OF3 — RL scalping** (NeuralQ + Double DQN + averaging, γ 0.97, n 5,
2,000 episodes): +4.84, +5.29, +5.12, +3.88 ticks/episode for 4 seeds
(t 10–13, 74–78% winning), 2.1–2.9× the best rule, 35–48% of the oracle.

**OF6 — stress** (2 saved agents, 9 scenarios): profitable in all 9
(t ≥ 4) including 1 s latency (+3.6/+4.2) and doubled fees (+2.1/+3.9);
beats the rule in 8/9 (not with informed flow ×0.5).

**OF8 — daily view** (20 days of 39 sessions): RL +171/+179 ticks/day,
100% winning days, annualized Sharpe ~80; mean-Q ensemble of 2 +206. **Not
believable as real trading**: the simulator is too predictable (see OF9).

**OF4 — tick-scalping baselines** (300 × 5 min, defaults): quote both
sides −2.77 ticks (gross +0.05: breakeven before fees); delta-skewed
quoting +2.90 (t 10.5, 8.3 fills, 74% winning). The first queue model
let naive quoting win every episode (+20.7); cancellations at the back
of the queue and latency competition fixed that. (The committed table
was once produced by a stale binary; corrected in f51c54b.)

**OF5 — plain RL tick scalping**: learned not to quote (0–0.1 fills, ~0).
Uniform exploration got 0.1 fills/episode: random modes keep cancelling
and rejoining behind ~120 lots, so no quoting experience is ever seen.

**OF7 — rule-guided tick scalping** (`of7_tick_guided.cpp`, reviewed):
off-policy learning from episodes where the delta-skew rule acts
(annealed 1 → 0.25, 2% deviations); greedy and rule-anchored
deployments. Smoke (2 × 700): greedy quotes but loses (−0.8/−0.2);
anchored +2.58/+2.57 vs the rule's +2.90 (not significantly different).
Review findings: pairing shares the seed, not the market path (our orders
change it), so validation leads are mostly selection noise; actions take
effect instantly (the rule's edge is +2.20 with 100 ms action latency).
Full run (2,000 episodes; seeds 1–2: seeds 3–4 were OOM-killed by the
tile-table bug, since fixed), `experiments/of7_results.csv`: anchored
champions +3.02 / +2.94 ticks vs the rule's +2.90, paired +0.13 (t 0.5)
/ +0.04 (t 0.2): **not significant**. Their validation leads (+0.97 t 2.2,
+1.09 t 2.8) vanished on sealed data, exactly the selection noise the
review predicted. Greedy champions −0.89 / −0.58 (−3.8 / −3.5 vs the
rule). RL matches the rule; it doesn't beat it.

**OF9 — efficient market** (`efficient_market()`: informed makers): 30 s
delta no longer predicts (corr −0.02 vs +0.12); markout flat after 5 s
(+0.57 → +0.56 vs +0.57 → +1.40 by default). Baselines: delta momentum
−7.2 ticks/episode (t −12), OFI momentum +1.4 (t 2.9: OFI sees the makers'
re-pricing), oracle +8.8. RL retraining in this market: see
`experiments/of9_*` when present.

**Calibration** (`of_calibrate`): recorded-tick CSV round trip exact;
in a recovery test (target = informed flow ×0.4) the fit matched
predictability, R² and clustering but not activity or markout, and
explained the predictability by momentum instead of informed flow:
summary facts don't identify mechanisms. Less informed flow made the
market *more* predictable (slower price discovery).

**Backtester** (`of_backtest`): replaying a simulated day exported to CSV
reproduces the agents' in-simulator Sharpe per window (0.74/0.70 vs
0.76/0.77); a 6.5 h day backtests in under a second.

## 5. Honest status

- The RL scalping agent beats every rule *in the simulator*, robustly.
  Its absolute P&L is a property of a simulator that is still too
  predictable (OF8). Nothing here shows it makes money in real ES.
- The tick-scalping agent does not beat the delta-skew rule. The
  literature agrees that slow passive agents rarely have a robust edge;
  the bar is "match or beat the rule at 100 ms latency on never-viewed
  seeds", not positive P&L.
- Both environments assume our orders don't need to be modelled beyond
  the simulator (scalping) or a FIFO queue without MBO calibration (tick).

## 6. Bugs found and fixed in this work (each has a test or a probe)

Uncapped quote fading and supercritical activity excitation (book blew
up); absorption feature measuring ordinary flow; tile tables of 8^16
cells for NeuralQ (bad_alloc) and 3.8 GB snapshots at 21 inputs (OOM);
unspecified argument-evaluation order in `reset()` (episodes differed
between compilers); a non-copyable environment; results committed from a
stale binary (OF4).

## 7. Plan for the next AI (ranked)

1. **Calibrate to the user's ES ticks:** convert their export to the CSV
   format, run `of_calibrate facts`, then `fit` with many seeds and long
   windows on their server; add richer targets first (markout curve at
   more horizons, queue depletion/refill rates; MBO if available). Re-run
   OF2/OF3/OF6/OF8 on the calibrated market.
2. **Backtest on real ticks:** `of_backtest` with the saved agents and the
   delta rule; walk-forward by day; train on recorded windows
   (`ReplayScalpEnv::reset`) and test on later days only.
3. **Paired evaluation on the same market path:** `SimParams::separate_streams`
   (opt-in; one RNG stream per exogenous process) exists, but measured it
   barely helps: rule vs the rule with one forced deviation, 200 episodes,
   paired-difference sd 5.41 → 5.16 ticks against an episode sd of ~4.8.
   Why: state-dependent draws (cancellations scale with queue size; the
   Poisson sampler uses a variable number of uniforms) desynchronize even
   a process's own stream, and activity (which scales fair-value
   volatility) is excited by price moves, so the fair-value path itself is
   endogenous (independent of our orders only with `activity_excite = 0`;
   tested). With the streams also re-seeded every step from (seed,
   process, step) (counter-based, now the behaviour of `separate_streams`),
   the correlation of the paired runs rises from 0.33 (shared stream) to
   0.61: paired comparisons have ~40% less variance. Use
   `separate_streams = true` for paired validation; the rest of the noise
   is the market's genuine reaction to our orders. Next: fewer
   state-dependent draw counts inside a step.
4. **Tick scalper:** action latency in `TickScalpEnv` (opt-in), tick
   diagnostics (no-trade share, spread capture vs carry, own-fill
   markouts), learn the advantage over the rule; success = match or beat
   the rule at 100 ms latency on fresh seeds (e.g. 940M+).
5. **Validation hygiene:** a seed-block ledger, fresh-seed confirmation of
   every claimed edge, CVaR in stress tests, domain randomization of
   simulator parameters during training.
6. Market-structure features (session high/low, volume-profile POC/value
   area) and the remaining items of `experiments/research_synthesis.md`.
