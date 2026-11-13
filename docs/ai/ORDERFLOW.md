# ES order-flow trading: simulator, agents, validation, plan

The order-flow trading work, in one place: what it is for, how it is
built, how to run it (on the user's Windows server; Windows only since
2026-11-04), every result with its caveats, and what to do next.
Experiment numbers (OF1–OF10) refer to probes in `experiments/`; detailed
numbers for each are in the probe's results file and summarized here.

> **Read first (found 2026-11-04):** every OF2–OF9 RL and rule result ran
> in a quiet market far below the stylized facts of OF1 (the activity
> warm-up, §5). They are comparisons within that regime, not ES-like
> evidence.

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
     tools: of_train (parallel seeds)  of_backtest (walk-forward on ticks)  of_calibrate (facts, fit)  probes OF1–OF10
            of_trace → tools/of_viz.py (episode viewer)   of_train_live → tools/of_train_live.py (live training viewer)
```

| File | What it is |
|---|---|
| `include/rag/orderflow/simulator.hpp` | `EsSimulator`: L2 book (16 levels/side, 5 exposed), 100 ms steps; noise takers, informed and uninformed meta-orders (power-law sizes, child orders), momentum takers, makers with capped quote fading, optional informed makers (`maker_info`), icebergs (hidden reserve: absorption); activity log-OU + intraday U-shape + mean-zero excitation; fair value with activity-scaled volatility and jumps. Strategy orders: `market_order`, `submit_limit`/`cancel` with FIFO queue position (cancellations biased to the back, faster participants joining ahead). Opt-in re-pricing (OF10): v1 rigid quote jumps (`reprice_band`, `reprice_snipe`), v2 stale-side depletion (`reprice_cancel`, `reprice_take`) with permanent impact of flow surprise (`value_impact`, `impact_forecast_*`). Opt-in activity model 1 (`activity_model`, `activity_warm_start`; OF10 v3): centred log-OU with compensated news kicks and an exact stationary law, warm start drawn on its own stream. Opt-in v4 book (OF13, all defaults off): `touch_cancel_mult` (thin, fast-turnover touch), `cancel_activity` (per-lot cancel hazard, depth independent of activity), `sweep_follow` (a sweep moves the hidden value to the new mid within the step), `stale_binomial` (stale lots leave independently), `max_stale_levels` and `max_step_ticks` (bounded per-step moves). Presets `efficient_market()`, `discovery_market_v1()`, `discovery_market_v2()`, `discovery_market()` (v3, frozen), `discovery_market_v4()` (v4, OF13), `default_warm()`; `market_preset(name)` for the tools. |
| `include/rag/orderflow/features.hpp` | `FeatureBuilder`: delta 5/30/120 s, OFI 5/30 s, relative volume, L1 and L1-5 imbalance, returns 5/30/120 s, VWAP distance, absorption (volume traded at a holding best level beyond the size displayed). Running window sums; self-normalized by a running RMS (works on any market without hand scales). |
| `include/rag/orderflow/scalp_env.hpp` | `ScalpEnv`: target position short/flat/long every 1 s, market orders with 100 ms latency, real fills, $1.50/side fees, reward = mark-to-market change net of fees, ends flat, fresh market and random time of day per episode (10 min + 150 s warm-up; a fresh market sits at the activity floor for ~1.5 h, so episodes are quiet: §5). |
| `include/rag/orderflow/tick_env.hpp` | `TickScalpEnv`: quote none/bid/ask/both or flatten every 0.5 s; 1-lot at the touch, rejoining at the back when the price moves; inventory ±1; FIFO fills; queue position in the observation; 5-min episodes. |
| `include/rag/orderflow/tick_csv.hpp` | `replay_tick_csv`: recorded ticks (`T,time,price,size,B|S` / `Q,time,bid,ask[,5+5 sizes]`) replayed in buckets into any `add_trade`/`end_bucket` sink. |
| `include/rag/orderflow/replay_env.hpp` | `TickTape` + `ReplayScalpEnv`: the scalping environment on recorded ticks (same observations, so simulator-trained agents run unchanged); fills at the recorded touch walking recorded depth; `seek(k)` for walk-forward windows. |
| `include/rag/orderflow/stylized.hpp` | `StylizedMeter`/`simulate_facts`: the same stylized facts for simulated and recorded ticks (activity, spread, L1, 1-min sd/kurtosis/autocorrelations, trade-sign memory, price-on-flow R², 30 s delta predictability, markouts at 1/5/30 s, mid changes per minute). |
| `examples/of_train.cpp` | Parallel training of scalping agents (one seed per thread), sealed scoring, mean-Q ensemble. |
| `examples/of_backtest.cpp` | Walk-forward backtest of flat, the delta rule and saved agents on a tick CSV. |
| `examples/of_calibrate.cpp` | `sim`, `facts <csv>`, `export <csv>`, `fit <csv>` (NLopt; seeds averaged in parallel threads). |
| `examples/of_trace.cpp` + `tools/of_viz.py` | Episode viewer. `of_trace` runs saved nets (mean-Q if several) on `ScalpEnv` episodes from the visualization block 960M+, with the 30 s delta rule and the fair-value oracle on the same seeds, and writes one CSV row per decision (book, hidden fair value for display, imbalance, the 13 features, Q-values, action, position, fills, fees, cumulative P&L). `of_viz.py` (stdlib only) turns the CSV into one self-contained HTML page: per-episode price/fills/fair value, positions, signals, Q-values, P&L; over all episodes a policy map, action shares, holding times, P&L per round trip. `--warmup W` replays nets trained with `of_train --warmup W`. |
| `examples/of_train_live.cpp` + `tools/of_train_live.py` | Live training viewer (A70). `of_train_live` trains one agent with `of_train`'s recipe in chunks and appends a JSON checkpoint every `--every` episodes to `--out` (training returns, validation points, epsilon, the current greedy learner on fixed probe episodes from the visualization block 960M+: one probe step by step, all probes' P&L and time in market, a policy map; the 30 s delta rule on the same probes); `--net` saves the champion (`of_trace` can replay it). `of_train_live.py` (stdlib only): `serve DIR` polls `DIR/live.jsonl` on 127.0.0.1, `bake` writes one self-contained HTML page, `page` writes just the polling page. Look only: never select or score on the probes. |
| `experiments/of13_regen.sh` + `tools/real_vs_sim.py` | OF13: real ES (the user's Sierra data) vs v4 with v3 as context (`of13_real_vs_v4.txt`, Table 4: the pre-registered gates G1–G8), then the efficiency probe and the naive rules on v4 at warm-up 150 s and 1,050 s (`of13_probe_w*.txt`, `of13_baselines_w*.txt`). `SEED0=… DIAG=… OUT=… PROBE_PREFIX=…` select another seed block. |
| `experiments/of*.cpp` | Probes OF1–OF10 (below), results files next to them. |
| `experiments/of10_discovery.cpp` (target `of_discovery`) | OF10 price-discovery diagnostics: every criterion (a)–(f) of plan item 1 per market and seed in one run: the hidden gap against an ideal 1-tick grid on the same value path, stale-quote races, imbalance and features in clock and event time, markouts (sniper vs other prints), stylized facts, rules / oracles / clairvoyants on `ScalpEnv` (`--rules`), tick rules on `TickScalpEnv` (`--tick`), a paired TWAP impact test (`--impact`). Block-jackknife SEs (5-min blocks). `experiments/of10_regen.sh` regenerates every OF10 file. |
| `experiments/research_synthesis.md` | Literature synthesis from the research crawl, with a ranked change list. |
| `tools/research_crawler.py` | Anonymous, polite crawler (arXiv, OpenAlex, Semantic Scholar, Hacker News) → `research_corpus/` (gitignored) indexable with `rag_pdf`. |

Tests: `tests/orderflow_tests.cpp` (book invariants, determinism, FIFO
queue, environments) and `tests/stylized_tests.cpp` (CSV reader, facts
round trip, replay fills, NeuralQ memory).

## 3. How to run

Windows only (the user's server; Linux CI was removed on 2026-11-04,
DECISIONS A60). From `C:\rag-cpp` in Developer PowerShell, AVX-512 build
(`docs/WINDOWS.md`): build with
`cmake --build --preset windows-avx512-release --parallel`; the tools are
in `build-avx512\Release\`.

```powershell
.\build-avx512\Release\of_train.exe --seeds 12 --episodes 2000 --market discovery --out nets\   # parallel training
.\build-avx512\Release\of_backtest.exe es_ticks.csv --time ns nets\seed1.net nets\seed2.net    # recorded ticks
.\build-avx512\Release\of_calibrate.exe facts es_ticks.csv --time ns                           # real vs simulated facts
.\build-avx512\Release\of_calibrate.exe fit es_ticks.csv --time ns --seeds 12 --seconds 21600 --evals 300
.\build-avx512\Release\of_calibrate.exe sim 10800 discovery                                    # simulated facts, one preset
.\build-avx512\Release\of_calibrate.exe export sample.csv 3600                                 # example of the CSV format
.\build-avx512\Release\of_trace.exe nets\seed1.net nets\seed2.net --market discovery --episodes 20 --out trace.csv
python tools\of_viz.py trace.csv trace.html                                                    # episode viewer (one HTML page)
.\build-avx512\Release\of_trace.exe DIR\seed1.net --market discovery_v4 --warmup 1050 --out trace.csv   # replay a v4 net (then of_viz.py)
.\build-avx512\Release\of_train_live.exe --market discovery_v4 --warmup 1050 --episodes 2000 --every 50 --out DIR\live.jsonl --net DIR\seed1.net
python tools\of_train_live.py serve DIR                                                        # live viewer on http://127.0.0.1:8770/
python tools\of_train_live.py bake DIR\live.jsonl out.html                                     # one self-contained page afterwards
bash experiments/of15_regen.sh [agent ...]                                                    # OF15 zoo on v4 (Git Bash; OF15_MARKET=v3 for v3)
$env:PYTHONPATH="python"; $env:RAGOF_DIR="<snapshot dir>"; .venv\Scripts\python.exe -m ragrl --help   # run/train/select/report/diag/seal/recheck/hpo/live/zoo/confirm
.venv\Scripts\python.exe tools\of_zoo_trace.py RUN [RUN ...] --out build-avx512\zoo --episodes 5    # replay ragrl runs on viz episodes (of_viz pages)
```
`_ragof` needs `cmake --preset windows-msvc-avx512 -DRAG_PYTHON=ON` and
the project venv `.venv` (A72). Serve `build-avx512\zoo` with a plain
static server (the `of-zoo` launch entry: `python -m http.server 8772
--directory build-avx512\zoo`), not with `of_train_live.py serve`, which
writes its own `index.html` into the directory (LESSONS 54).
Run the live viewer alongside every RL run (A70). `serve` writes the page
and serves DIR on port 8770 (`--port`); if you write the page with `page`
instead and serve it yourself, open `index.html?src=live.jsonl`.

**Markets** (`--market` in `of_train`, `of_trace`, `of8_scalp_days`;
second argument of `of2_baselines`, `of4_tick_baselines`, `of_calibrate
sim`; `--markets` list in `of_discovery`): `default` (OF1–OF8),
`efficient` (OF9), `discovery` (OF10 v3, `discovery_market()`, frozen),
`discovery_v4` (OF13 v4, `discovery_market_v4()`, matched to real ES; RL
uses `--warmup 1050`, A69; `TickScalpEnv` refuses it),
`discovery_v2` and `discovery_v1` (OF10 v2 and v1, kept to reproduce their
numbers) and `default_warm` (`SimParams{}` with activity model 1: the old
market at realistic activity, not calibrated). All go through
`market_preset(name)`.
`of_train --test-seed` must be ≥ 900,000,000 (a test block), or it stops
(A61).

OF10 diagnostics (Git Bash; builds first, warning-free, then regenerates
every OF10 file): `bash experiments/of10_regen.sh`. One probe run by hand:
`build-avx512/Release/of_discovery.exe --markets default,discovery --seeds 1-3 --episodes 400 --long session --sessions 16 --rules 100 --emo 40 --tick 100 --impact 30 --threads 16 --out out.txt`
(iteration 3: both regimes, the AGREE table, O1/O2; `--episode-warmup 1050`
replaces ScalpEnv's 150 s warm-up with a 15-min burn)
(`--set name=value,...` overrides fields of the discovery preset for
tuning). OF13 (real ES vs v4, Table 4 gates G1–G8, then the probe and
baselines on v4): `bash experiments/of13_regen.sh`. Long runs go from a copy of the executables, not from
`build-avx512\Release\` (A59).

The tick-scalping trainer (OF7) is the target `of_tick_train`; run one
process per seed in parallel, e.g.
`1..12 | % { Start-Process -NoNewWindow .\build-avx512\Release\of_tick_train.exe "$_ 2000 0.99 0.5 0.25 0.02 200 100 tick_seed$_.net" }`.

Hardware notes: the networks are small (2 × 64 units); a GPU does not
help at this size (per-update work is microseconds). Scale with cores:
one seed per thread (`of_train`), seeds averaged in parallel
(`of_calibrate --seeds`). GPUs pay off only with much larger models
(e.g. convolutional/recurrent models over raw book sequences), which
would need a GPU framework fed by this simulator: not built yet.

## 4. Results (simulator unless stated; sealed test seeds)

Seed blocks: training = trainer streams seeded by the run seed; validation
= 700M+i; sealed tests: scalping 900M+i (OF2, OF3, OF9), tick scalping
910M+i (OF4, OF5, OF7), stress 920M+i (OF6), days 930M+i (OF8). OF10
diagnostics use simulator seeds 1–18 (tuning, held out, final; ledger in
NEXT_SESSION.md §4).

**Caveat on OF2–OF9 (the warm-up finding, 2026-11-04).** `ScalpEnv` and
`TickScalpEnv` start a fresh simulator for every episode and warm it up
for 150 s. A fresh simulator sits at the activity floor (§5): the
skeptic's runs measured, on the default market, activity ~0.29, 4.2
trades/s and a 1-min sd of 2.5 ticks in 10-min windows after the warm-up
(6 seeds), against ~20 trades/s and
3.4–4.4 ticks in OF1's 6.5 h windows. So every OF2–OF9 rule, RL, stress
and daily result below was measured in this quiet regime, while the OF1
facts (and `of_calibrate sim`) describe long windows. The comparisons
between strategies stand within that regime; none of them shows behaviour
in an ES-like market. Present since the simulator's first commit
(`c808bd4`).

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
re-pricing), oracle +8.8. **RL retraining in this market** (4 seeds ×
2,000, `experiments/of9_rl_results.csv`): +9.7 to +11.2 ticks/episode
(t 21–24, 95–97% winning), *more* than by default and more than the
fair-value oracle: a red flag. Diagnosis: the informed makers tilt their
order rates toward the hidden value while the price adjusts slowly, so the
book imbalance leaks it (corr with the hidden gap 0.49 vs 0.20 by default;
L1 imbalance vs next 30 s return 0.32 vs 0.25) and the agent reads it. The
preset removed trade-flow predictability and moved it into the book. In
both presets price discovery is far too slow (the hidden gap predicts the
next 30 s return at 0.58–0.65); real ES prices information in seconds.
**This is the most important simulator fix** (plan item 1; OF10).

**OF10 — price discovery** (`of10_discovery.cpp`, target `of_discovery`;
plan item 1; iteration 3 done at `e229af4`, RL reruns next). Criteria (NEXT_SESSION.md §1): (a) the hidden
gap g = fair value − mid no longer predicts the next 30 s; (b) book
imbalance predicts the next tick or two and nothing after; (c) the markout
curve is flat after a few seconds; (d) the stylized facts stay ES-like;
(e) naive rules are near breakeven after costs; (f) nothing beats the
fair-value oracle. Tuning on diagnostic simulator seeds 1–3 × 2 h,
checked on 4–6 × 2 h, final check once on 7–12 × 6 h. Each window starts
at 3,600 s of the session after a 150 s warm-up. Rule checks run on
`ScalpEnv`/`TickScalpEnv` episodes (seeds 1,000,000 × seed + i). Both
versions are opt-in presets: `SimParams{}`, `efficient_market()` and
`separate_streams` stay bit-exact with `e0a4c7a` (golden hashes, A58).
Files: `of10_results.txt` (tuning), `of10_heldout.txt`, `of10_baselines.txt`
(OF2 rules, 900M), `of10_tick_baselines.txt` (OF4 rules, 910M),
`of10_calibrate_sim.txt` (3 h); v1 numbers at commit `10bfe03`, v2 at
`4ba0354`, v3 at `e229af4` (plus `of10_warmup_check.txt`; in v3
`of10_heldout.txt` is the seeds 4–6 sanity run, not held out).

*v1: rigid quote jumps* (`discovery_market_v1()`, `10bfe03`). When
|value − mid| > 0.65 tick the whole book jumps round(gap) ticks, and
snipers take a clip of the stale quotes. What it fixed: corr(g, next 30 s)
fell from 0.57 (default) and 0.64 (efficient) to 0.11 (tuning and held
out), against a tick-grid floor of 0.09–0.10 (a). What failed:
- (b) the book moves as a block, so imbalance predicts nothing: event-time
  corr(L1 imbalance, next mid change) +0.02, against +0.33 by default.
- (d) price-on-flow R² 0.32 came from the fixed sniper clip (re-measured
  in the v2 probe: R² 0.80 on sniper delta, 0.15 on the rest); 25.9
  trades/s in `of_calibrate sim` (3 h), above 15–22.
- (c) markout 30 s − 5 s +0.14, seeds −0.01 … +0.35: unresolved at 3 × 2 h.
- (e) every rule loses (delta 30 s −8.8 ticks/episode on 900M; quote both
  sides −6.7 on 910M).
- (f) vacuous: the gap never exceeds the band (max 0.65), so the gap > 0.75
  and > 1 oracles never trade; gap > 0.5 loses −7.4. Re-run with per-lot
  accounting in the v2 probe, two rules beat the oracle at θ* = 0.60 by
  more than 2 SE: FAIL.
- Probe bug: edge was counted per order, not per lot (a flip counted
  once); fixed in v2.

Rejected as the final design, kept as `discovery_v1` (A56).

*v2: stale-side depletion* (`discovery_market()`, `4ba0354`). When value
passes a quote, that level is withdrawn lot by lot within a few 100 ms
(92% cancelled, the rest taken by snipers in FIFO order), nobody joins a
stale price, and the price ticks only when the stale best level is empty;
the value side moves with it (shape kept, spread 1). The unexpected part
of taker flow moves fair value permanently (`value_impact 0.045`), and
informed makers (`maker_info 2.5`) tilt their rates by the within-tick gap
only. Fading is off (the flow's information reaches the price through
`value_impact`). Results: tuning / held out (pooled ± block-jackknife SE),
final from the `4ba0354` commit message:
- (a) **met.** corr(g, next 30 s) 0.111 / 0.092, 0.017 / 0.020 above the
  ideal grid's floor (0.094 / 0.072); after a value jump the mid is within
  a tick in 0.10–0.13 s (median), p95 0.2–0.3 s. Final: 0.10–0.11, within
  ~0.02 of the floor.
- (b) **met on long windows.** Event-time corr(L1 imbalance, next mid
  change) +0.25 / +0.24, then +0.09 for the second change; 30 s tail
  +0.000 / −0.008. Imbalance reads the within-tick gap, by design: corr(I1,
  g) 0.45 / 0.44 (default 0.19). In 10-min episode windows (skeptic, 6 ×
  600 s) the next-change correlation is +0.15, below the 0.20 target.
- (c) **marginal.** Markout 30 s − 5 s −0.075 ± 0.054 / −0.077 ± 0.048
  (seeds −0.23 … +0.12); final −0.015 ± 0.024.
- (d) **FAIL.** 20.1 trades/s on tuning but seeds 18.0–22.3; held out 23.1
  (20.4–27.9); `of_calibrate sim` 22.7. Excess kurtosis 4.5 / 4.1 / 4.2
  (target 5–10); 1-min sd 4.2 / 4.5 ticks with seeds above 4.5; L1 size
  1.06 / 1.40 × default's (one held-out seed 1.58, limit 1.5). The windows
  run hotter the longer they are: in the skeptic's 3 × 6 h, 33.1 trades/s,
  L1 191, 1-min sd 4.9.
- (e) **FAIL for passive quoting.** Quote both sides −7.2 / −6.7 / −6.6
  (910M) ticks per 5 min, against the target −4 … 0 (default −2.8): fills
  are picked off past the half spread (adverse −0.07). Directional rules no
  longer win: delta 30 s −2.0 (tuning) and −2.4 on 900M (t −4.7; default
  +1.8); OFI 30 s −4.5 / −5.0; L1 imbalance −25 to −28 (costs).
- (f) **formal pass only.** Every rule stays below the oracle at the
  pre-registered θ* = 0.70 + 2 SE, but that oracle trades 1.4–1.6 lots per
  episode and nets −0.5; the gap > 0.8 and > 1.0 oracles trade under 1 lot.
  Information is there: a clairvoyant that sees the mid 2 s ahead makes
  +7.6 / +6.9 ticks/episode, a fair-value quoter with 0.1 s latency +2.3 /
  +2.0. A bound that never trades bounds nothing (LESSONS 43).
- Impact (measured, not a criterion): a paired TWAP buy over 300 s moves
  the mid +13.7 ticks for 300 lots and +54.0 for 1,200 (exponent 0.99), all
  of it still there 300 s later. Default: +0.29 / +0.98, none left.
  Linear and fully permanent impact is not ES-like (real impact is concave
  and partly transient); to check in iteration 3.

**Skeptic's verdict on v2: not ready for RL.** Main finding (verified in
the code): the episodes RL trains on run at the activity floor (§5). The
v2 tuning used 2 h windows, which begin in that floor and leave it during
the window, while its rule checks (e, f) ran on quiet episodes.

*v3: stationary activity and a retuned depletion market*
(`discovery_market()`, `e229af4`; v2 kept verbatim as `discovery_v2`).
- **Warm-up fix.** The legacy activity loop has no stable steady state:
  loop gain 0.045/s, R = 11.5 on default sessions; 10 h runs swing between
  the clamps, and default 10-min episodes spent 41% of their steps at the
  lower clamp (4.5 trades/s). Initialising `move_avg_` correctly cannot fix
  that. Opt-in `activity_model 1`: a centred log-OU with compensated
  (mean-zero) news kicks, whose stationary law is exact; with
  `activity_warm_start` the first state is drawn from it on a dedicated
  stream (`kStreams` = 9). Model 0 is byte-identical (golden hashes);
  `default_warm()` = `SimParams{}` + model 1; v3 and `default_warm` have
  golden hashes (`GoldenRunsOfTheIteration3Presets`).
- **Preset.** v2's depletion and permanent flow impact without informed
  makers (`maker_info 0`), smaller and more frequent maker orders, slower
  depletion (`reprice_cancel 12`), more sniping (`reprice_take 0.15`),
  rarer larger jumps, `value_impact 0.05`; comment above
  `discovery_market()` in `simulator.hpp`.
- **Probe.** Every statistic in both regimes: 400 ScalpEnv-built 10-min
  episodes per seed (150 s warm-up) and 16 session-weighted full sessions
  per seed, an AGREE table between them, pre-registered oracles O1/O2, a
  quote-both decomposition by cause.
- **Acceptance** on seeds 1–3 (tuning), 13–18 (held out, used once) and
  19–21 (skeptic, fresh), both regimes:
  - (a) **met.** corr(g, r 30 s) − floor +0.017 … +0.019 (gate ≤ 0.02).
  - (b) **met, marginal.** Event corr(I1, next mid change) 0.206–0.216
    (one seed 0.193 on 19–21); tail ≤ 0.024; leak corr(I1, g) 0.37 (≤
    0.40). Outside the gate, corr(I1, r 30 s) ~0.07 misses the probe's own
    0.05 line.
  - (c) **met except one cell.** mk30 − mk5 fails only on tuning-seed
    episodes, +0.022 ± 0.009 (2.4 SE; long +0.006 ± 0.009); passes on
    13–18 and 19–21 (and 4–6). The skeptic judged it chance-level.
  - (d) **met.** 18.4–18.7 trades/s, 1-min sd 3.9–4.05 ticks, excess
    kurtosis 5.5–7.2, |r| ac1 0.19–0.22, R² 0.33–0.35; no clamp time.
  - (e) **met for naive rules, FAIL for quote-both.** No naive `ScalpEnv`
    rule nets > 0 after costs. Quoting both sides loses −24 to −27
    ticks/episode, adverse −0.17 … −0.20: ~60 fills/episode are stale
    sweeps (1 s markout −0.38) against ~25 flow fills (+0.30). Structural:
    snipers take the agent's lone stale lots before the price ticks, with
    a small L1 queue (~108 lots); no parameter lever removed it at 15–22
    trades/s.
  - (f1) **oracles bite.** O1 (|g| > 0.62, latency 0) +24 … +30
    ticks/episode on ~86–98 lots; O2 (EMO H 10 s, K 32, θ 1.24, decide 1 s,
    latency 100 ms) +5.4 … +7.2 on ~15–17.5 lots. (f2) untested: the RL
    smoke run collapsed to flat.
  - **AGREE FAIL:** active meta-orders 10.46–10.59 in episodes vs
    11.07–11.18 on long windows, 5–6 SE in all three seed sets: the
    meta-order population is not warm after 150 s. A 1,050 s burn gives
    11.13 (`of10_warmup_check.txt`), matching long windows.
- **Other findings.** Fair value is not a martingale: the 10 s value
  change predicts the next 10 s at +0.013 ± 0.003 (permanent-impact
  momentum; delta 5 s has gross +0.235/lot and loses only on costs). Own
  impact is linear and permanent: block round trips of ≥ 30 lots profit
  from their own impact (+0.15 tick/lot at 30), so `ScalpEnv` refuses
  `contracts > 1` when `value_impact > 0`. On `default_warm` (the old
  slow-discovery market at realistic activity) naive rules profit
  strongly (OF2 delta 5 s > 1.5 +15.19, t 22.8; imbalance L1–5 +7.31, t
  10.9; quote both +1.3): the slow-discovery leak is real, not an artefact
  of the activity bug. The legacy OF6 stresses are partly empty on
  discovery (fade ×2 is a no-op with fading off, vol ×1.5 weak, informed
  ×0.5 touches ~4% of volume).
- **Decisions for the reruns** (A62–A66): RL on discovery uses a
  pre-registered 1,050 s `ScalpEnv` warm-up (`--warmup`); the quote-both
  failure is waived for market-order environments only (OF3, OF6, OF8), and
  no tick-environment (OF4, OF5, OF7) rerun claims a calibrated market;
  OF6 gets a discovery-specific stress list fixed before any v3 net
  exists; O1/O2 are references, and RL > O2 + 2 SE or RL > O1 triggers a
  leak audit.

**OF11 (RL reruns, `1a73bd6`): no edge.** 24 seeds × 2,000 episodes at
warm-up 1,050 s: every seed loses on 300 sealed episodes (mean −0.54
ticks/episode) and the ensemble learns not to trade; OF6 (pre-registered
discovery stresses): every strategy ≤ 0 in all 11 scenarios; OF8: ensemble
−3.3 ticks/day (Sharpe −3.1), rule −88; no OF2 rule > 0 by 2 SE. Gate (f)
holds non-vacuously (EXPERIMENTS.md OF11).

**OF12 (real ES vs the simulator, the user's Sierra data):** with the same
code on both sides, v3 matches real ES on imbalance → next tick (+0.24 vs
+0.24) and on price-on-delta R², but its L1 queue is ~8× too deep (143 vs
17 lots) and its mid lags its own flow (corr(1-s delta, next 1 s mid move)
+0.15 vs −0.04 real); real ES also has far more volatility clustering and
mid flicker. Only 44 min of RTH book exist so far (EXPERIMENTS.md OF12).

**OF13 (simulator v4, `discovery_v4`, `31f51c4`):** a book-first rebuild
(opt-in; v3 frozen with a golden, A67): thin fast-turnover touch (L1 ~20
lots, depth rising with distance), memoryless noise flow whose sweeps move
mid and hidden value within the step, binomial stale depletion, bounded
per-step moves. All 25 pre-registered gate rows G1–G8 against real ES
pass on tuning (960,012,001–012), on held-out sessions (960,013,001–012,
once) and on the skeptic's (960,014,001–012) (A68); the efficiency gates
(a), (b), (c), AGREE (0 of 29 disagree), (e) and (f1) pass at the RL
warm-up 1,050 s (A69). **v4 has no meta-orders, no informed or momentum
takers and `value_impact` 0, so its fair value carries no edge after
costs: both oracles lose** (O1 −8.2 … −12.0, O2 −18.3 … −19.8
ticks/episode). For a 1-lot market-order scalper "RL finds no edge on v4"
is therefore close to true by construction; v4 tests whether RL overfits
the small signals that remain (I1 timing, mid flicker), not whether order
flow carries information (LESSONS 48). Quote-both loses ~−200
ticks/episode (A64 stays; `TickScalpEnv` refuses v4). Skeptic caveats:
I1 still predicts the next 1 s after latency (+0.08 vs ~0 real; top
decile +0.19 ticks/decision, below the 1.24-tick round trip); 100 ms mid
flicker reverts (−0.22 vs −0.02 real), giving small fade edges per lot;
the spread never widens (1 tick ~100% vs 93–95% real); the real G2/G4
reference is one calm morning (a busy second RTH morning, Nov 5, falls
outside G4 and G2 itself); margins shrink across blocks (1-min sd 8.68 →
7.99 → 7.48, volume 1.03 → 0.92 → 0.87); no volatility clustering
(EXPERIMENTS.md OF13).

**OF14 (RL on v4, `6d678c6`): no edge.** All 24 seeds lose on 300 sealed
episodes (mean −3.18 ticks/episode, worse than v3's −0.54: the agents
trade more and lose more); the ensemble stops trading; OF8 days: every
strategy loses (ensemble −50 ticks/day). No sign that RL exploits v4's
known quirks (EXPERIMENTS.md OF14).

**OF15 (the agent zoo through the `ragrl` pipeline, v4 and v3): no
edge.** A leak-proof pipeline (C++ seed ledger with process roles,
`VecScalpEnv` bound as `_ragof`, Python `ragrl`; A71–A73) trained 8
agent families (NeuralQ reference, PPO, frame-stack PPO, recurrent PPO,
DQN, QR-DQN, LightGBM threshold, BC/DAgger), 12 seeds × 2,000 episodes
each, and scored each config once, paired, on fresh sealed ranges (v4
901M+[0,300), v3 901M+[300,600)). All 16 configs NO_EDGE, no hard red
flag. The declared headline, the 12-seed ensemble, is flat for 5 of 8 v4
and 4 of 8 v3 agents (0 orders on test); the seeds that trade lose (mean
of seeds −0.10 … −3.80 on v4, +0.02 … −1.09 on v3), and on 600 fresh viz
episodes plus latency/fee stresses nothing is positive. PPO v4 holds one
~597 s position per episode (a coin flip); agents timing entries on I1
gain +0.1 … +0.3 ticks gross at 1 s but pay ~0.55 ticks slippage; fading
30 s returns loses (EXPERIMENTS.md OF15).

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

- **The warm-up finding (2026-11-04, skeptic's runs; verified in the
  code).** `EsSimulator::move_avg_` (the running mean of |mid move| that
  activity excitation is measured against) starts at 0.1 tick per step and
  adapts with an 1,800 s time constant. Real moves are smaller, so the
  excitation is negative step after step: activity collapses to
  exp(`activity_min`) = exp(−1.5) within seconds and stays there for
  ~1.5 h. Every `ScalpEnv`/`TickScalpEnv` episode starts a fresh simulator
  with a 150 s warm-up, so it runs in this floor. Skeptic's runs (seeds 1–6,
  600 s after the warm-up, `build-avx512/skeptic/s600.txt`, not committed):
  activity 0.29, 4.2 trades/s on the default market and 2.3 on discovery
  v2 (target 15–22), 1-min sd 2.4–2.5 ticks, excess kurtosis −0.6 / +0.6,
  sign autocorrelation 0.16 (default). On long windows the same presets
  look ES-like or hotter (3 × 6 h, `s6h.txt`: default 21.6 trades/s,
  discovery 33.1; to the close, `sclose.txt`: 20.8 / 31.2). So the OF1
  facts were measured on long windows and every OF2–OF9 RL and rule
  result in the quiet regime; OF10's (a)–(d) on 2 h windows that cross
  from one to the other. Present since `c808bd4` (the first simulator
  commit). **Resolved for the opt-in presets (iteration 3, `e229af4`):**
  the root cause is an unstable loop (gain 0.045/s, R = 11.5), not only
  the initial value; activity model 1 with a stationary warm start (in
  `discovery` and `default_warm`) has no clamp time and agrees between
  episodes and long windows on activity and the facts. The legacy model
  is unchanged, so **the old OF2–OF9 results still ran at ~4 trades/s**
  and stand only as results in that regime. Still not warm after 150 s:
  the meta-order population (~5% short, 5–6 SE); the RL reruns use a
  1,050 s warm-up (A63).
- **Price discovery (plan item 1): iteration 3 accepted for market-order
  RL.** v3 meets (a), (b) (marginally), (d), naive-rule (e) and (f1); (c)
  misses in one cell at 2.4 SE (chance-level). It fails passive quoting
  (−24 to −27 ticks/episode, stale sweeps), so tick-environment results
  on it are not calibrated (A64). **RL on it finds no edge (OF11).**
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
stale binary (OF4); the OF10 v1 probe counting edge per order instead of
per lot (fixed in v2); the activity warm-up (§5: an unstable activity
loop, fixed opt-in by activity model 1 in iteration 3); block round trips
profiting from their own permanent impact (`ScalpEnv` refuses
`contracts > 1` with `value_impact > 0`); OF13: v3's 151-tick one-step
sweep (v4 bounds every step, `max_step_ticks`, takers only, never the
strategy's own orders), and Poisson stale depletion leaving residue that
leaked book imbalance (`stale_binomial`).

## 7. Plan for the next AI (ranked)

**Where OF15 leaves it (2026-11-12).** OF15 closed the "maybe a better
learner would find it" question on the simulators: through a leak-proof
pipeline (seed ledger enforced in C++, one sealed opening per config,
paired CIs, red flags), 8 agent families × 2 markets give 16 NO_EDGE
results; what they learn is to stay flat (§4, EXPERIMENTS.md OF15). On
both simulators no RL agent finds an edge after costs with 1-lot market
orders at 1 s. More agents or HPO on the same markets is not the next
step. What remains, in order:

* **Real-tick replay through the pipeline:** `VecReplayEnv` (bound in
   `_ragof`) exists but is unused; feed it the user's recorded ES ticks
   (Sierra) and run the zoo walk-forward by day (item 2 below), with the replay
   block sealed in the ledger. **Pre-registered plan: `OF16_REAL_ES_PLAN.md`**
   (day manifest: 8 full front-month RTH test days, Sep and Nov 2026; days
   from 2026-11-12 on are the real-data confirm block; zero-shot scoring of
   the 16 OF15 runs first, training on real only after). Phase 1 done
   (OF16 phase 1: converter, ledger-gated tapes, `ragrl score-real`; on
   dev days the L1-imbalance signal is +0.07 ticks at 1 s against a 1.37-tick
   round trip, as in v4). Phase 2 done 2026-11-12: **no edge on real ES**
   (no headline of 32 meets the rule, best t 1.2 against 4.63; the agents
   trade on real as they did in sim, orders/window correlation 0.994, and
   lose a little more; A79). **OF17** (signal census of the 13 features on
   the 8 real days, A82): nothing pays the 1.28-tick round trip at 1 s to
   1 min (best gross +0.22); at 2-10 min none detected, but vwap_dist,
   delta120 and ret120 predict within a month with a sign that flips
   between Sep and Nov: a regime question for the ~224 days of `.scid`
   trades. **OF18** (A83): on 228 days, no walk-forward regime rule on 7
   trend features pays at 5-15 min (primary net −0.78, t −1.83; upper
   bounds 1.55 / 2.53 / 3.46 ticks per trade). **OF19** (A85): the
   per-price footprint rules on 47 days of true tick data, at the user's
   Sierra study defaults: no edge, no footprint condition beats its shell,
   unfinished and finished extremes revisited at indistinguishable rates.
   **OF20** (A86): absorption and exhaustion at key levels: no edge, no
   better than a plain touch of the level. **OF21** (A87): a forward test
   of exhaustion at those levels is pre-registered on days from
   2026-11-12 (looks at 60 / 125 / 250 full days).
   In parallel, simulator realism where the
   skeptic found gaps (spread that widens, I1 persistence after latency,
   100 ms mid flicker, volatility clustering) as an opt-in v5, judged on
   new real sessions (LESSONS 49).
* ~~**Red-flag calibration**~~ done 2026-11-12 (A77): R1 needs the
   annualized Sharpe's lower 2-SE bound above 5 (about t > 2.9 at 300
   episodes), the minimum-activity guard (R1v, A74) kept.
* **The 0xc0000374 heap crash**, once WMI is repaired (the user's
   call): ~150 lgbm seed-runs with the WMI bypass off and on. Keep
   per-attempt logs (LESSONS 53).
* **Throughput (no GPU, A75):** fewer gradient steps per env step for
   DQN/QR-DQN (`train_freq` 4), more torch threads and workers (24
   hardware threads, OF15 used 12), a faster v4 step.

1. **Fix slow price discovery, then calibrate to the user's ES ticks.**
   Iterations 1 (rigid quote jumps, rejected) and 2 (stale-side depletion:
   (a), (b) met, (c) marginal, (d), (e) fail) are done (OF10).
   **Iteration 3 is done** (`e229af4`, §4 OF10 v3): activity model 1 with
   a stationary warm start; v3 frozen with a golden hash and checked once
   on seeds 13–18; (a), (b), (d), naive (e), (f1) pass; quote-both fails
   structurally (stale sweeps); impact still linear and fully permanent.
   **Next, the RL reruns:** OF2, OF3, OF6, OF8, OF9 on `discovery` with
   the pre-registered 1,050 s `ScalpEnv` warm-up (`--warmup`, A63), on the
   sealed blocks as before; OF6 with the discovery stress list fixed
   before any v3 net exists (A65); compare RL with O1/O2 paired (A66). No
   tick-environment (OF4, OF5, OF7) rerun claims a calibrated market until
   the stale-sweep mechanism is redesigned (A64). 900M and 910M were
   viewed on the discovery market (v2 and an iteration-3 smoke;
   NEXT_SESSION.md §4), so confirm any edge once on 950M+. **Done (OF11, `1a73bd6`): no
   edge** (OF2/OF9 rules: `of11_baselines.txt`); 950M+ unused. **OF12
   compared v3 with the user's real ES data** (Sierra; EXPERIMENTS.md OF12):
   fix book depth (L1 ~17 lots, rising with distance) and same-second
   impact first; record RTH depth and tick-by-tick trades for a few weeks
   before fitting. **Done (OF13, `31f51c4`): v4 (`discovery_v4`)** passes
   the 25 real-ES gate rows and the efficiency gates on three seed blocks
   (§4; A67–A69). **Done (OF14, OF15): no RL edge on v4 or v3.** Before
   tightening any gate, record more RTH depth and tick-by-tick trades
   (Sierra settings): the real reference is one calm morning. Fix the
   spread-never-widens and I1-persistence caveats in a v5 if RL shows it
   exploits them. Then **calibrate to the user's ES ticks:** convert their export to the CSV
   format, run `of_calibrate facts`, then `fit` with many seeds and long
   windows on their server; add richer targets first (markout curve at
   more horizons, queue depletion/refill rates; MBO if available). Re-run
   OF2/OF3/OF6/OF8 on the calibrated market.
2. **Backtest on real ticks:** `of_backtest` with the saved agents and the
   delta rule; walk-forward by day; train on recorded windows
   (`ReplayScalpEnv::reset`) and test on later days only. With OF15 the
   same can run through `ragrl` on `VecReplayEnv` (the first bullet above).
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
   state-dependent draw counts inside a step. OF15 measured the cost
   between agents: PPO v4 at 100 vs 300 ms latency on the same episodes
   correlates 0.56, the delta rule 0.19 (LESSONS 52).
4. **Tick scalper:** action latency in `TickScalpEnv` (opt-in), tick
   diagnostics (no-trade share, spread capture vs carry, own-fill
   markouts), learn the advantage over the rule; success = match or beat
   the rule at 100 ms latency on fresh seeds (e.g. 940M+).
5. **Validation hygiene:** the seed-block ledger is done and enforced in
   C++ (`seed_ledger.hpp`, OF15, A71), with fresh-seed (950M+)
   confirmation of any claim behind an orchestrator flag. Remaining: a
   pre-registered stress list for v4 (none exists; OF15 ran no stress or
   days on either market, R11), CVaR in stress tests, domain
   randomization of simulator parameters during training.
6. Market-structure features (session high/low, volume-profile POC/value
   area) and the remaining items of `experiments/research_synthesis.md`.
