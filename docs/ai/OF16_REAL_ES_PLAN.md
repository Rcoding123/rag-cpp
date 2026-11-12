# OF16 plan: the OF15 agents on real ES (recorded book + trades)

Status: **phase 1 done 2026-11-12** (plumbing, review fixes, dev-day diagnostics: EXPERIMENTS.md OF16 phase 1, `experiments/of16_dev_diagnostics.txt`); **phase 2 done 2026-11-12: no edge on real ES** (no headline of 32 meets §4; best t 1.2 against t_req 4.63; EXPERIMENTS.md OF16 phase 2, `experiments/of16_real_results.txt`). Per §4 the simulator-trained RL line is closed; phase 2b does not run. Originally: **plan only, nothing run.** Written 2026-11-12 after OF15 (no RL
edge on either simulator). Committing this file, with the day manifest in
§2, is the pre-registration: day roles, policies, statistics and the
decision rule are fixed here, before any agent or baseline P&L on real data
is computed.

## 1. Question

Does anything the OF15 agents learned in the simulator make money **on
real ES**, after fees and slippage, at the same decision rate (1 s) and
latency (100 ms)? Second question, diagnostic only: is the one thing the
agents found in v4 (entering with the L1 imbalance, +0.1 to +0.3 ticks
gross at 1 s, OF15) present on real ES at a size that beats the ~0.5-tick
cost of crossing the spread?

Expected answer: no. The sim agents mostly learned to stay flat, and the
ones that trade lose. But this is the first test against the real market,
and the zero-shot step is cheap.

## 2. Data and the day manifest

The data is the user's Sierra Chart recording, read-only:
`C:\SierraChart\Data\MarketDepthData\ES*.depth` (full book, `<qBBHfII`) and
`C:\SierraChart\Data\ES*_FUT_CME.scid` (trades, `<q4f4I`). Nothing goes
into the repo. Each day's export goes to `build-avx512/real/` (gitignored),
and the manifest records each file's sha256.

The table comes from sampled depth timestamps on 2026-11-12. RTH means
08:30-15:00 CT, using CDT before 2026-11-01 and CST after. The front month
by RTH volume comes from OF13: ESU26 through 09-11, ESZ26 from 09-14.

| day | contract | RTH coverage | role | why |
|---|---|---|---|---|
| 2026-09-08, 09-09, 09-10, 09-11 | ESU26 (front) | full | **test** | full front-month sessions. Their 1-min and 1-s aggregates were 4 of 224 days in OF12/OF13 calibration (statistics only, no agent selection): see §6 |
| 2026-11-05, 11-10, 11-11 | ESZ26 (front) | full | **test** | never used for anything. 11-05's ETH book was read by OF13; its RTH was not |
| 2026-11-09 | ESZ26 | 09:03-15:00 | **test** | partial session: only the windows that fit |
| 2026-11-03 | ESZ26 | 08:30-09:14 | dev | OF12/OF13's real reference: already viewed |
| 2026-09-14 … 09-17 | ESU26 (expiring back month) | full | dev | after the roll: a thinner market, not representative. Used for plumbing and as a thin-market stress |
| 2026-09-07 | ESU26 | partial | excluded | Labor Day session |
| 2026-09-18, 11-04, 11-06 | — | none or a few minutes | excluded | no usable RTH |
| **2026-11-12 onward** | ESZ26 / ESH27 | — | **confirm** | days recorded after this plan is committed: real-data fresh confirmation, the analogue of 950M+, opened once per claim |

Episodes are cut as in `ReplayScalpEnv`: 100 ms buckets, a ~~150 s~~
1050 s feature warm-up from 08:30, then consecutive 600 s windows. That
gives ~~38~~ 37 windows per full day and ~~~35~~ 33 on 11-09, **~~301~~ 292
test windows**, close to the simulator's 300-episode test. Days on the
expiring back month, and any day whose front month is not the recorded
contract, never count as test days.

**Amendment 2026-11-12 (before any test opening).** The warm-up is 1050 s,
not 150 s: every OF15 run trained with `env.warmup_seconds` 1050, and
FeatureBuilder's running scales (EWMA over ~600 s) are only ~22% settled
after 150 s against ~83% after 1050 s, so at 150 s the agents would see
features 0-4 and 8-12 at roughly twice their training scale, the volume
ratio near +1.5 instead of ~0.2, a 150 s session VWAP and delta30 on the
wrong scale. `ragrl score-real` takes the warm-up from the run's config and
refuses a run whose warm-up, decision step or episode length differ from
the manifest's `episodes` block. Window k is played after a warm-up on the
tape from k × 600 s, so windows = (buckets − 10500) // 6000: a full day
(08:30-15:00, 234000 buckets) gives 37; 11-09 (book from ~09:03, at most
214200 buckets; any start before 09:12:30 gives the same) gives 33; total
7 × 37 + 33 = **292 test windows**. A window whose span (warm-up plus
window, 1650 s) overlaps a book gap longer than 5 s (a recording outage)
is not played and is listed; that rule is fixed here too, before any test
day is opened (2026-11-03, a dev day, has a 610 s outage from 09:00:34 CT).

## 3. Work, in order

**Phase 1: plumbing and dev-day checks (no test day is opened)**
1. Write a converter `tools/sierra_to_ticks.py`: one RTH day from
   `.depth` + `.scid` into the `tick_csv.hpp` format, with T lines for
   trades and Q lines for best bid/ask plus 5 levels at each book change.
   It reuses `tools/real_vs_sim.py`'s parsers. It also writes per-day
   checks: crossed or locked quotes, gaps, trade-through-quote counts, and
   the file sha256.
2. Add `experiments/configs/of16_real_manifest.json`, the §2 table as
   data. ragrl loads test-day tapes only in a scorer process (C++
   `open_sealed("real")`). Each test opening goes to `ledger_usage.jsonl`
   with the manifest hash. Dev tapes load freely and confirm tapes need
   the orchestrator claim, as for 950M.
3. Add `ragrl score-real RUN --days test`. It scores a run's headline
   ensemble and its 12 seeds on `VecReplayEnv` windows, alongside the
   baselines flat, delta30 and an L1-imbalance rule (sign of I1 at entry,
   1 s hold), using the OF15 paired statistics.
4. Tests:
   - A synthetic tape where the right answer is known.
   - A no-lookahead test: a fill uses the quote one bucket after the
     decision, never the decision bucket's end quote.
   - DST: a CDT day and a CST day cut the same RTH.
   - The converter round-trips a small `.depth` slice.
5. Dev days only:
   - **Covariate shift:** for each observation feature, the share of real
     values outside the 1-99% range seen in v4 training. Any feature above
     5% is reported, and policies depending on it are flagged as
     extrapolating. They are still scored.
   - **Baseline sanity:** flat = 0, and delta30 and hold behave like
     OF12's real statistics.
   - **The I1 markout on real:** the 1 s mid move after the imbalance
     sign, against the half-spread.

**Phase 2: zero-shot scoring, once (needs the user's go)**
- Policies: the 16 sealed OF15 runs, each scored by its ensemble and its
  mean of seeds, and the three baselines. Nothing is retrained or tuned on
  real data.
- Headlines (both pre-registered): ensemble and mean of seeds per run.
  That is M = 32, so t_req = 2 + sqrt(2 ln 32) ≈ 4.6 (A71 / OF15
  multiplicity bar).
- Statistics:
  - Per-window net ticks and orders per window.
  - The paired difference against flat and delta30.
  - Window-level t and a **day-clustered bootstrap**, because windows
    within a day are correlated. With 8 days that bootstrap is coarse;
    report it anyway.
  - The Sep and Nov subsets separately (§6).
- Stress, scored in the same single opening: latency 300 ms, fees $3 per
  side.
- Red flags: R1 (A77); any agent whose gross on real beats its gross on
  v4 by 2 SE (check the replay for leaks before believing it); and the
  imbalance rule profiting (check the bucket alignment first).

**Phase 2b: confirm, only if something clears t_req.** Score it once on
the next 8 full front-month RTH days recorded from 2026-11-12 onward. That
takes about 2 weeks of recording. Sierra must keep recording depth.

**Phase 3, optional: training on real**, decided after phase 2. Walk
forward by day: train on days already scored and test only on later fresh
days. There is little data: ~8 days is ~300 windows, ~180k decisions,
against 14M in a simulator run. So prefer the data-efficient learners:
LightGBM or BC on real labels, or fine-tuning a sim-pretrained PPO. Start
only if phase 1's I1 markout or phase 2's gross numbers show a signal
bigger than the cost of trading.

## 4. Decision rule (fixed now)

- **Edge claim:** headline net > 0 with t ≥ t_req on the test days AND the
  day-clustered 95% CI above 0 AND positive in both the Sep and Nov
  subsets. Then confirm once (phase 2b), and claim only if the confirm
  days are > 0 with t ≥ 2.
- **Otherwise: no edge on real ES.** That closes the "simulator-trained
  RL" line. The next work is phase 3 or simulator realism (ORDERFLOW §7),
  chosen by the phase 1 diagnostics.

## 5. Cost

Light. The export is single-threaded and IO-bound: ~2.5 GB of depth and
the matching `.scid` ranges, a few minutes per day. Scoring is
replay-only, with no simulator: 16 runs × 13 policies × 292 windows × 600
decisions ≈ 36M policy steps (amended 2026-11-12: 292 windows, §2) on small MLPs, minutes on the CPU. No GPU.

## 6. Known weaknesses (stated before the results)

- **Power.** About 292 windows over 8 days (amended 2026-11-12, §2). For a policy that trades (per
  window SD ~10-26 ticks), the SE is ~0.6-1.5 ticks per window and coarser
  with day clustering. Only a large edge is detectable; "no edge" means
  "none above roughly 2 ticks per window".
- **Calibration overlap.** v4 was calibrated on aggregates of 224 days,
  including the 4 Sep test days. The agents never saw real data, so this
  is a weak link. The Nov-only subset (11-05, 09, 10, 11) is fully
  untouched and is reported on its own.
- **Replay fills.** 1-lot market orders fill at the recorded touch one
  bucket later. The replay cannot model our own impact, which is
  negligible for 1 lot in ES, or queue position. Directional market-order
  policies only, which is all the zoo trades.
- **Regime.** Two contracts and two months (September and November 2026).
  The confirm days extend it.
