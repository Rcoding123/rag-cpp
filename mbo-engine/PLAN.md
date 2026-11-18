# ES MBO engine: plan

**Goal:** an analysis engine for CME E-mini S&P 500 (ES) futures, covering
order-book structure, price levels and options/gamma levels, plus RL agents
trained on full market-by-order (MBO) data. The agents are trained on a Linux
server and run live on Windows inside ATAS on a prop-firm account.

**Status (2026-11-18):** Phase 0 (infrastructure, data), Phase 1 (DBN reader) and
Phase 2 (per-order book, validated against mbp-1 and the status feed) are done; Phase 3
is in progress (bars, sessions done). The
owner approved engine work on 2026-11-18 ("proceed with the plan autonomously").
The no-spending rule still applies.

---

## Principles

These come from the rag-cpp order-flow research (OF1 to OF24, see
`docs/ai/EXPERIMENTS.md`, `docs/ai/OF24_AUDIT.md` and `docs/ai/LESSONS.md` in
this repo):

1. **No out-of-sample edge has been found yet.** Every in-sample profit in
   OF16 to OF24 turned negative on later, unseen days. OF24 scored -2.31
   ticks/trade on its 43 holdout days, against -2.44 for random exits. All of
   that work used Sierra Chart data and queue-free fills.
2. **Find a signal before training RL.** First find an order-book signal at
   short horizons that survives walk-forward testing after costs, then train
   RL on top of it (the OF24 audit recommendation).
3. **Score only on later, unseen days.** A replay of the training days
   measures memory, not skill. Always compare against random baselines and
   report confidence intervals, statistical power and detectable effect size.
4. **Use queue-aware fills only.** Passive fills cannot be judged honestly
   without per-order data. That is why we bought MBO.
5. **Training and live features must match exactly.** Training runs in
   Python/C++ on Linux and live runs in C# on Windows, and both must compute
   identical features. A parity test is mandatory before going live.
6. **Never spend money without an explicit go-ahead from the owner.**

## Architecture

```
Linux server (training, research)                       Windows (live)
-------------------------------------------------       ------------------------------
Databento DBN files (data/raw)
  -> C++ DBN reader -> per-order MBO book (queues)
  -> features / levels / gamma-by-strike (data/features)
  -> signal research (walk-forward)
  -> C++ replay env (queue-aware fills) x N envs
       via nanobind, on the batch cores
  -> RL training (CPU; rented GPU only if needed)
  -> agent.onnx  ---------------- copy ------------>    ATAS add-on (C#) + ONNX Runtime
```

The `.onnx` file is the hand-off. ONNX Runtime runs on Linux, Windows and
macOS, from Python, C++ and C#. ATAS is Windows-only: its newer ATAS X beta
adds macOS but not Linux.

## Phases

Each phase ends with acceptance checks. Nothing counts as done until those
checks pass on the server.

### Phase 0: infrastructure and data (DONE)
- The server is set up by `server/bootstrap.sh`. It provides a Python venv,
  C++ toolchain, SSH hardening, firewall, time sync, and a live/batch CPU
  split.
- Data downloaded and verified (`scripts/order_data.py`): 665 files,
  18.59 GB, all sha256-checked. See `HANDOFF.md` for the inventory.

### Phase 1: data layer (DONE 2026-11-18)
- Chose a small own reader, `engine/include/mbo/dbn.hpp`, instead of
  databento-cpp (v0.69.1): it needs zstd dev headers, OpenSSL and an HTTP stack
  we cannot install without sudo, and the MBO record is a fixed 56-byte struct.
  Layout checked against `databento/dbn` v0.74.0 (`record.rs`, `fsm.rs`) and
  against our own files (DBN v3). Links `libzstd.so.1` directly.
- Python `databento` stays the reference reader.
- Acceptance MET: `scripts/verify_dbn.py` compares count, per-action counts and
  per-field wrap-around sums (plain and position-weighted) of all 13 MBO fields.
  **92/92 files match, 1,029,905,106 records.** Truncated, corrupt, junk,
  header-only and mid-record-truncated inputs are rejected with errors.

### Phase 2: per-order book
- Build a book keyed by order_id with per-price FIFO queues, applying MBO
  actions A/C/M/T/F/R. It must handle the start-of-day synthetic snapshot,
  `R` (clear) and the `F_LAST` flag (only read the book at event boundaries).
- Expose queue position per order and book state at any timestamp.
- Acceptance (status 2026-11-18, `engine/tools/book_check.cpp`, results in
  `/srv/quant/results/phase2_book_check.jsonl`):
  - Book never crossed at `F_LAST` boundaries: **not true, and explained by the exchange's
    own status feed** (bought 2026-11-18, $0.0031, `es-status`, `ES.FUT` parent). 91,463 of
    855,090,264 boundaries are crossed (0.011%), in 112 runs. **All 91,463 fall in a state the
    exchange marks `PreOpen` with `is_trading=N`; 0 fall while status says trading.** Breakdown
    (codes decoded from the dbn enums): PreOpen/Scheduled/ChangeTradingSession 80,673;
    PreOpen/**MarketEvent** 5,811; PreOpen/Scheduled/NoCancel 4,304; PreOpen/Scheduled 675.
    Not the base rate: non-trading states cover 20.5% of calendar time, and 158 of 159 PreOpen
    periods contain a crossing. The five `PreOpen/MarketEvent` periods (4.98-5.01 s: 08-05
    15:18:07, 09-11 12:30:01, 09-24 04:03:39, 10-02 12:30:01, 11-10 13:30:01 UTC) are exactly the
    five 5-second crossed runs (CME Velocity Logic is the likely mechanism per the owner's CME
    link; the status feed says only "PreOpen, MarketEvent"). Databento's GLBX.MDP3 page states
    crossed books before the open and in the daily pause are normal. Scripts:
    `scripts/crossed_vs_status.py`, `scripts/crossed_status_context.py`; output
    `/srv/quant/results/crossed_vs_status.txt`. Consumers must treat `BAR_CROSSED` /
    non-trading status as "no valid touch".
  - Trades vs the touch captured at the previous boundary (buy aggressor at or
    above best ask, sell aggressor at or below best bid): 22 violations in
    34,474,229 checked trades (0.00006%), at most 4 per day. Not investigated further.
  - Internal consistency (level totals, links, index): 0 failures. Book
    anomaly counters (unknown cancel/modify, duplicate add, bad side/price,
    oversize cancel): **0** on all 92 days.
  - Top of book vs Databento `mbp-1` (bought 2026-11-18, $0.0779: ES.v.0, 2026-09-11 12:00-13:00
    UTC, 581,009 records, contains the 12:30:01-06 crossed run): **PASS at event boundaries.**
    mbp-1 is not a 1:1 projection of MBO (it publishes only top-level-changing records, reorders
    inside a multi-record event, shows some cancels as side N), so records are grouped by packet
    (ts_recv, sequence) and the book after the last record of each group is compared (best bid/ask
    price, size, order count; `scripts/verify_tob_mbp1.py`, `engine/tools/tob_dump.cpp`):
    521,656 groups ending at an F_LAST boundary: **0 mismatches**. 51 of 51 mbp-1 groups with a
    crossed/locked top of book match ours. 25 mismatches in 26,245 groups that end MID-event (no
    F_LAST; the book is undefined there): in the case inspected, mbp-1 publishes a trade with the
    pre-fill book and the post-event book under the following sequence (an `N` record), and our
    state equals mbp-1's later state. 13 of the 25 were confirmed by an automatic next-group test,
    the other 12 only classified structurally (mid-event), not individually explained. 28 mbp-1
    groups at `N` records are absent from our dump; all 28 match our state at that time.
    `mbp-10` was not bought (not needed).
  - Replay speed: 9.2-13.4M messages/s, at most 3.2 s per day-file, single
    thread, book built per file from the synthetic snapshot.
- Files are whole UTC calendar days, not trading sessions: each starts with a
  synthetic snapshot (`R` then adds, F_SNAPSHOT) and a session spans two files.
  One instrument_id per file (ES.v.0 switched from 42140870 to 10252 between 2026-08-04 and 2026-09-17; exact roll day not yet located).
- Queue priority and the FIFO anomaly (item investigated 2026-11-18). **ES is pure FIFO:** CME's
  matching-algorithm page (owner's link; algorithm F = "resting orders are matched in timestamp order
  only") and the exchange's own definition records (`match_algorithm = 'F'` for all 1,944 ES outright
  definitions in our data). CME's modify rule: priority is lost on a quantity increase, a price change
  or an account change (an account change is a same-price same-size modify in MBO); display-quantity
  (iceberg) orders are refreshed to the back of the queue after each match. The book therefore
  defaults to `ModifyPolicy::RequeueOnEqual` (size decrease keeps priority; increase, price change and
  equal-size modify requeue).
  Test (`engine/tools/fifo_check.cpp`): when a resting order is filled (`F`), every order ahead of it in
  our queue should also have been filled in the same packet. Message order, not timestamps, ranks the
  queue (Databento: messages of an instrument are never reordered and the start-of-day snapshot is
  published in priority order; verified: in the live part `sequence` is strictly non-decreasing on all
  3 days checked and `ts_recv` never decreases; the `sequence` decreases are all inside the snapshot,
  which is priority-sorted, so ranking by sequence would have been wrong; for adds at one price ts_event
  order equals message order in ~20M pairs, 0 exceptions). Layers removed one at a time, 2026-11-13:
  raw 5.1% of fills -> allowing same-packet fills 4.5% -> excluding AGGRESSOR fills (an `F` carrying the
  trade's own order_id: the aggressing order already rested on the book and was modified to cross; these
  were exactly the 5,869 fills whose price/side disagreed with the book) 4.13% -> excluding blockers
  removed (C) in the same packet 3.54% -> equal-size modify requeues 3.46% (CME rule alone 3.54%).
  **Residual FIFO violations, 7 days (2026-08-10, 08-26, 09-15, 10-01, 10-22, 11-10, 11-13): 3.0-6.2%
  of passive fills** (policy `eq`: 3.0-6.1%; always-requeue 5.2% and worse). Not explained by: the
  snapshot (involved in only 1-2% of the residual), timestamps (identical order), modifies (mostly
  never-modified orders), unknown fills (0), or a previously-partially-filled blocker (13.6%, a
  contributor only). Unexplained; candidates are order attributes MBO does not carry (minimum-quantity
  or other conditional orders) or display-quantity refreshes that MBO does not show as a requeue.
  **Size of the gap** (blocked contracts = unfilled quantity sitting ahead of a filled order): median
  1 contract, p90 3-6, p99 13-47; median 8-18% of the price level, p90 30-50%.
  **Phase 6 rule:** queue position from MBO is an upper bound on contracts ahead (the exchange skips
  some earlier orders, never the reverse in this test), so a simulated order is placed behind every
  earlier order at its price and fills only when the volume ahead (message-order queue) has traded
  or cancelled. That is the conservative direction (fills no earlier than the real exchange would);
  report results with and without a haircut for the 3-6% skipped-order effect, never assume better.
  The direction claim is reasoning from how the test is defined, not separately measured. The effect cuts
  both ways: a skipped order itself waits longer than its position suggests; our simulated orders are plain
  limit orders, so that case should not apply to them (also unmeasured).
- Port ideas, not code, from rag-cpp `simulator.hpp`. Its book is
  aggregated, not per-order.

### Phase 3: features and levels (IN PROGRESS: bar layer done 2026-11-18)
- DONE: `engine/include/mbo/bars.hpp`, `engine/tools/bar_extract.cpp`,
  `scripts/bars_to_parquet.py`: 100 ms buckets (UTC-aligned on `ts_recv`, sparse:
  only buckets with records) of touch, 5-level depth totals, aggressor
  volumes, trade count, add/cancel/modify counts, L1 OFI and rag-cpp absorption,
  all read at F_LAST boundaries. Output `data/features/bars/YYYYMMDD.parquet`,
  92 files, 41,736,426 rows, 621 MB. Row layout (120 bytes) is the parity
  contract for the C# side.
- Verified (`scripts/verify_bars.py`, all 92 days): per-bucket counts, volumes
  and last trade price equal an independent numpy recomputation from the Python
  reader; touch and 5-level depth at every bucket end equal a separate dict-based
  Python order book over the first 300,000 records of each day (12,938-62,927
  buckets per day). Both checks fail on a deliberately corrupted row.
  `bars_test` pins OFI/absorption on hand-computed values.
- NOT verified: the depth/OFI check covers only a record prefix per day, not the
  full day. Trades with side N (auction, implied) count in `n_trades` but in
  neither volume column.
- DONE: session calendar `scripts/sessions.py` (America/Chicago; trade date = 17:00 CT -> 16:00
  CT, halt 16:00-17:00 CT, RTH 08:30-15:00 CT tag). Verified against our 2026 data
  (`scripts/verify_sessions_vs_data.py`): 1,627/1,627 scheduled opens (00:00 UTC snapshot records
  excluded, as Databento's docs do) at 17:00 CT, 1,584/1,584 scheduled closes at 16:00 CT, no
  scheduled pause 15:00-16:00 CT (CME's 2020 FAQ mentions a 15:15-15:30 CT pause: gone by 2026),
  1,534/1,534 preliminary settlements at 15:00 CT map to the right `ts_ref` trade date; the
  only early close found is 2026-09-07 (12:00 CT). DST unit tests: `scripts/test_sessions.py`.
  **RTH = 08:30-15:00 CT** (owner-confirmed 2026-11-18: NYSE 09:30-16:00 ET, matches the 15:00 CT
  settlement; CME itself could not be checked, cmegroup.com 403). Shortened days: 12:00 CT close, no
  RTH. It is one constant (`sessions.RTH_OPEN/RTH_CLOSE`).
- Settlement: CME FAQ (archived copy, effective 2020-10-26) says ES settles at 15:00 CT (12:00 CT
  on shortened days). Value comes from statistics (stat_type 3), never computed. `stat_flags` bit0
  = final, bit1 = actual (Databento GLBX.MDP3 page); our data has prelim-actual (2) published
  15:00 and ~16:39 CT, and final-actual (3) the next morning (or Sunday after Friday). `ts_ref` is
  a date label (do not localise). Use the latest record with ts_recv <= t, so a level is never
  used before it was published.
- DONE: trades table `data/features/trades/YYYYMMDD.parquet` (`engine/tools/trades_extract.cpp`,
  `scripts/trades_to_parquet.py`; ts_recv, price, size, aggressor side, trade_date, rth, halt):
  34,474,311 trades, equal to the book-check total; `scripts/verify_trades.py` matches the Python
  reader on all 92 days (count, size per side, sum price*size, first/last ts).
- DONE: settlement table `data/features/levels/settlement.parquet` (`scripts/settlements.py`):
  5,076 official ES outright settlement records (22 contracts), flags decoded, timestamped by
  publication; `asof()` returns the prior-session settlement known at a time (final replaces
  preliminary once published). Prelim == final in all 1,555 pairs. Plausibility only (not a CME
  check): median |settle - VWAP 14:55-15:00 CT| = 3.6 ticks, p90 8.0, max 21.3.
- DONE: session levels `data/features/levels/session_levels.parquet` (`scripts/levels.py`): per
  (trade_date, instrument) ETH and RTH open/high/low/close/VWAP/volume/POC/VAH/VAL, overnight
  H/L (every ETH trade before the RTH open, evening part included), opening ranges of 5, 15, 30
  and 60 minutes (`orb{5,15,30,60}_{high,low}`; `orb_high/orb_low` = the 30-minute default),
  completeness flag, and prior-session
  levels (same contract; across a roll the old contract's levels plus `prior_basis_adj` from the
  two official settlements; `prior_settle` is always the contract's own). 78 rows.
  Verified independently (`scripts/verify_levels.py`: sessions from the exchange's scheduled
  opens, trades from the Python reader): ETH and RTH count/volume/OHLC/VWAP equal to 1e-9 on all
  78, brute-force POC equal, value area contains POC and holds >=70%. Unit tests
  `scripts/test_levels.py`. **Data limits:** trade date 2026-09-16 is split between ESU6 (stub,
  17:00-18:59 CT) and ESZ6 (from 19:00 CT; ES.v.0 switches at UTC midnight, so ESZ6's first two
  hours were not in the MBO files) and 2026-11-17 is a one-hour stub. **Fixed 2026-11-18:** the
  missing ESZ6 window was bought as `trades` (not MBO: a book cannot be built from a mid-day MBO
  window, and the whole-day MBO quotes $0.96, over the cap): `es-trades-esz6-roll`, ESZ6,
  2026-09-15 22:00 -> 2026-09-16 00:30 UTC, $0.0064. The 30-minute overlap with our MBO trades is
  identical (1,441/1,441: ts_recv, price, size, side, sequence); 3,665 new trades (volume 7,937)
  before 00:00 UTC are in `data/features/trades/20260915_esz6_roll.parquet` (`scripts/supplement_trades.py`).
  ESZ6 2026-09-16 is now `complete`; the ESU6 stub of that date (its session was cut at 00:00 UTC
  by the switch to ESZ6) and 2026-11-17 stay `complete=False`. No book or bars exist for the
  bought window (trades only).
  **Bug found and fixed by the independent check:** the first overnight range used 'clock < 08:30'
  and dropped the evening half of the session; now all ETH trades before the RTH open.
  **Multiple-testing rule:** all four opening-range lengths are stored because it is cheap, but
  ONLY ONE length may be used in any pre-registered test (choose it before looking at results),
  otherwise the choice becomes a free parameter and biases the result.
  2026-09-07 (Labor Day, 12:00 CT close) has no RTH by design.
- TODO: windowed features (delta 5/30/120 s, OFI windows, volume ratio,
  imbalance L1/L1-5, returns, VWAP distance) with the rag-cpp EWMA
  normalisation, built from the bar rows; session assignment (CME session
  runs 18:00-17:00 ET, files are UTC days, DST shifts the halt between 21:00
  and 23:00 UTC); levels (prior-session high/low/settlement, VWAP, opening
  range, volume profile): levels are done (above); only the windowed features remain.
- Order-flow features: OFI, imbalance, delta, absorption. Port from rag-cpp
  `include/rag/orderflow/features.hpp`.
- Levels: volume profile, prior-session high/low/settlement, VWAP, opening
  range.
- Output: Parquet in `data/features/`, one file per day.

### Phase 4: options and gamma levels
- Inputs:
  - `definition` gives strike, expiry, call/put and underlying.
  - `statistics` gives open interest (stat_type 9) and settlements.
  - Use the quarterly (`ES.OPT`) and the weekly/daily/EOM (`EW*`, `E1A` to
    `E5D`) products.
- Implied volatility from settlement prices (Black-76), then gamma per
  strike and total gamma by price.
- The dealer-positioning sign is an assumption. Make it explicit and
  configurable, because every published GEX number depends on it.
- Acceptance:
  - Open-interest totals per expiry are plausible.
  - Results are reproducible.
  - Strikes and underlyings map correctly across the September/December
    roll.

### Phase 5: signal research (before any RL)
- Short horizons first, with walk-forward splits and plans registered
  before running.
- Include costs (ES tick = $12.50; round-trip cost assumptions as in OF23).
- Compare against random-time controls.
- Use the rag-cpp tooling ideas: seed ledger, sealed test blocks, day-block
  bootstrap.
- Exit criterion: a signal with a confidence interval above zero after
  costs on unseen days, or a documented null.

### Phase 6: replay environment
- C++ env on the real MBO book. Our order joins the back of the queue and
  fills only when the volume ahead of it trades or cancels.
- Model latency, fees, partial fills and order types (market, limit, stop).
- Vectorise N envs on a thread pool behind nanobind with the GIL released.
  This follows the pattern in rag-cpp `vec_scalp_env.hpp` and
  `python/ragof_module.cpp`.
- Run on the batch cores (`quant-batch.slice`).
- Acceptance: deterministic for a given seed, and fills reconcile with the
  book.

### Phase 7: RL
- Start with offline / pessimistic RL that has a skip action and
  distributional outputs, as the OF24 audit recommends.
- Evaluate only on sealed later days.
- Rent a GPU only when profiling shows the network, not the simulator, is
  the bottleneck.

### Phase 8: deployment
- Export with `torch.onnx.export`.
- C# parity test: identical features and outputs on recorded data.
  Unverified: whether an ATAS add-on can load ONNX Runtime; test it first
  with a tiny model.
- ATAS add-on on Windows. Per earlier research, MBO in the ATAS API needs
  the Ultra plan and a Rithmic connection.
- Run in simulation or on a paper account first. Check the prop firm's rules
  on automated trading before any live use.

## Known gaps and decisions still open
- The C++ DBN reader choice (Phase 1).
- Whether to buy an `mbp-1`/`mbp-10` validation sample (Phase 2).
- The dealer-sign convention for gamma (Phase 4).
- History is only 2026-08-01 to 2026-11-16, a single regime. More history
  costs money and needs owner approval.
