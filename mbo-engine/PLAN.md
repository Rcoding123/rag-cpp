# ES MBO engine: plan

**Goal:** an analysis engine for CME E-mini S&P 500 (ES) futures, covering
order-book structure, price levels and options/gamma levels, plus RL agents
trained on full market-by-order (MBO) data. The agents are trained on a Linux
server and run live on Windows inside ATAS on a prop-firm account.

**Status (2026-11-18):** infrastructure and data are done (Phase 0). Engine
work (Phase 1 onward) has **not started** and waits for the owner's go-ahead.

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

### Phase 1: data layer
- Choose a C++ DBN reader (databento-cpp or the `dbn` crate via C ABI).
  **Research the options before choosing.** Python `databento` stays the
  reference reader.
- Acceptance: per-day record counts and field values match the Python
  reader exactly.

### Phase 2: per-order book
- Build a book keyed by order_id with per-price FIFO queues, applying MBO
  actions A/C/M/T/F/R. It must handle the start-of-day synthetic snapshot,
  `R` (clear) and the `F_LAST` flag (only read the book at event boundaries).
- Expose queue position per order and book state at any timestamp.
- Acceptance:
  - The book is never crossed at `F_LAST` boundaries.
  - Trades print at or inside the touch.
  - Top of book matches a Databento `mbp-1`/`mbp-10` sample for a few days.
    Buying that sample needs a quote and owner approval first.
  - Replay speed for one full day is measured and recorded.
- Port ideas, not code, from rag-cpp `simulator.hpp`. Its book is
  aggregated, not per-order.

### Phase 3: features and levels
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
