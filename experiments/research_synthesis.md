# Research synthesis: ES order-flow simulator, scalping agent, tick-scalping agent, validation

Experiment artifact for the `research/rl` branch. It is not part of `docs/`.

## How this was made

- **Corpus:** `research_corpus/` holds 595 items and 60 PDFs. The PDFs total 1,862 pages and are indexed in `corpus.ragidx`.
- **Search:** I ran 64 targeted questions through `build/rag_pdf ask research_corpus/corpus.ragidx "<q>" -k 6..8`. They covered queue-reactive models, OFI and impact, RL market-making rewards and inventory penalties, adverse selection, no-trade collapse, latency, exploration, backtest overfitting, walk-forward testing and sim-to-real.
- **Reading:** I extracted the 30 most relevant PDFs to text and read the cited pages in full. Each citation gives the title and year exactly as they appear in the corpus. Citations are tagged:
  - **[PDF]**: I read the full text.
  - **[abstract]**: the corpus has only the title and abstract, so the claim rests on the abstract alone.
- **Our results:** they come from `NOTES_RL.md` (OF1 to OF8, the OF4 correction and OF7).

### Where the corpus is thin (read this first)

- **Backtest-overfitting statistics.** The corpus has nothing on the deflated Sharpe ratio or on purged or embargoed cross-validation: no metadata match and no full-text passage. The keyword scan found 0 deflated-Sharpe hits. What it does have is three abstracts:
  - *Deep Reinforcement Learning for Cryptocurrency Trading: Practical Approach to Address Backtest Overfitting* (2022) [abstract]
  - *Interpretable Hypothesis-Driven Trading:A Rigorous Walk-Forward Validation Framework for Market Microstructure Signals* (2025) [abstract]
  - *The Red Queen's Trap: Limits of Deep Evolution in High-Frequency Trading* (2025) [abstract]

  The validation section below therefore leans on the evaluation practice of the RL papers that are in the corpus as full text.
- **Sim-to-real.** No paper studies transfer from a simulator to live trading. The only discussion is the limitations chapter of *Market Making Strategies with Reinforcement Learning* (2025) [PDF].
- **ES specifically.** One econometric paper on E-mini order flow, *Returns and Order Flow Imbalances: Intraday Dynamics and Macroeconomic News Effects* (2025), is abstract only. The closest large-tick futures evidence is:
  - Bund futures in *Deep Learning Meets Queue-Reactive: A Framework for Realistic Limit Order Book Simulation* (2025) [PDF];
  - Bund/DAX in *Queue-reactive Hawkes models for the order flow* (2019) [abstract];
  - KOSPI200 index futures in *Market Making under Order Stacking Framework: A Deep Reinforcement Learning Approach* (2022) [abstract].
- **Ranking quality.** Semantic Scholar mostly failed during the crawl (HTTP 429), so citation-based ranking is weak. Several high-ranked digest items come from low-quality venues, and I did not use them.

---

## 1. Simulator realism and calibration

### What the literature says

**Queue-reactive (QR) dynamics are the standard tractable target for large-tick books.**

*Deep Learning Meets Queue-Reactive: A Framework for Realistic Limit Order Book Simulation* (2025) [PDF] models each queue by event intensities λ_L(q), λ_C(q) and λ_M(q) (limit, cancel, market) that depend on the current queue size. These intensities have a closed-form maximum-likelihood estimate: count the events at queue size n and divide by the time spent at n (pp. 4-5). Calibration on Euro-Bund futures (about 4,000 lots per 5 minutes) gave three findings:
- Adding hour of day and last event type to the state improves likelihood, next-event accuracy and inter-event timing (p. 10).
- Independent queues miss cross-excitation between bid and ask. Modelling the book jointly fixes the event-transition matrix.
- The validation suite (pp. 16-20) covers:
  - impact during a TWAP buy that rises concavely and then relaxes;
  - a square-root impact law, fitted exponent 0.55, R² 0.89;
  - queue-size distribution Q-Q plots against real data;
  - mid-price prediction, balanced accuracy 0.63 against 0.54 for a dedicated deep-learning price-prediction baseline.

*Queue-reactive Hawkes models for the order flow* (2019) [abstract] calibrates on Bund and DAX futures. It reports that adding a Hawkes term to QR intensities "dramatically improves" inter-event-time statistics and queue-size distributions.

*Reinforcement Learning in Queue-Reactive Models: Application to Optimal Execution* (2025) [PDF] uses a calibrated QR model as an RL environment precisely because it produces both kinds of impact (p. 3):
- direct impact: liquidity consumed;
- indirect impact: flow reacting to queue imbalance.

**Who cancels, and where, decides queue value.**
- *Market Making via Reinforcement Learning* (2018) [PDF] replays L2 data. It must assume where in the queue cancellations land, and it assumes uniform (p. 3).
- *Cooperative Multi-Agent Reinforcement Learning for Market Making on Prediction-Market Order Books* (2026) [PDF] makes the same uniform assumption when it rebuilds market-by-order data from L2. It calls this a limitation (p. 33).
- *Reinforcement Learning for Trade Execution with Market and Limit Orders* (2025) [PDF] goes the other way: its noise and tactical traders "always cancel orders with the highest queue positions first" (p. 11).
- Our `cancel_back_bias = 3` sits between these two extremes. OF4 showed that this parameter, together with `join_ahead_mean`, determines whether passive quoting pays at all.

**Order value can be decomposed.**

*Deep Hawkes Process for High-Frequency Market Making* (2021) [PDF] values a resting order, following Moallemi and Yuan, as:

V = P(fill) × (half-spread premium − E[adverse move | fill])  (pp. 37-38)

*Optimal Market Making in Prediction Markets: Empirical Distribution Learning, Analytical Policy Characterization, and Reinforcement Learning* (2026) [PDF] derives a quote-or-cancel threshold from three quantities (p. 42):
- queue depletion speed;
- the hazard of adverse selection;
- the size of the adverse move.

Quoting patiently pays when the queue clears fast and adverse moves are small.

**Predictability in real index futures decays fast.**
- On the E-mini, price and flow impacts are significant at one second, and "shocks dissipate almost entirely within a second". News raises price impact and lowers flow impact. *Returns and Order Flow Imbalances: Intraday Dynamics and Macroeconomic News Effects* (2025) [abstract].
- *Deep order flow imbalance: Extracting alpha at multiple horizons from the limit order book* (2023) [abstract] finds that the effective forecast horizon is about two average price changes.
- *Market Informedness and Market-Maker Profitability: The Trade-Off Between Adverse Selection and Price Discovery* (2026) [abstract] finds informed flow most harmful when aggregate informedness is low. This matches our OF calibration finding that fewer informed traders make the market *more* predictable.
- *When AI Trading Agents Compete: Adverse Selection of Meta-Orders by Reinforcement Learning-Based Market Making* (2025) [PDF] shows a fast RL market maker in a Hawkes book learning to profit from the price drift of a TWAP meta-order. A trail of slow meta-orders is something fast traders compete away, so a simulator in which nobody competes for that trail overstates what a slow strategy can earn from it.

**Latency changes the answer, and order entry and cancellation have different effects.**
- In *Integrating Tick-level Data and Periodical Signal for High-frequency Market Making* (2023) [PDF], order-entry latency of 10 ms gives a cumulative profit of about 15,000, while 40 ms gives about 1,000. Longer *cancel* latency raised profit there (20 to 80 ms: 8,000 to 14,000), which the authors tie to inventory effects (p. 16).
- *Resolving Latency and Inventory Risk in Market Making with Reinforcement Learning* (2025) [abstract] trains with random delays of 30 to 100 ms.
- The prediction-market thesis (2026, p. 57) adds action delays so that quotes can go stale.

**Non-stationary direction is the main failure mode.**

*Deep Learning of Robust Market Making under Regime-Switching Order Flow* (2026) [PDF] shows four things:
- The direction bias of market orders in real books is "highly non-stationary, with slowly varying regimes" (p. 7).
- Persistent directional flow causes inventory saturation in both closed-form and RL market makers.
- PnL falls monotonically as regime duration grows (p. 18).
- A fill-intensity model can be calibrated by censored waiting-time estimation, and diffusion by a volatility signature plot (p. 2).

**Replay alone can't carry market impact; there is a partial fix.**
- *Reinforcement Learning for Trade Execution with Market and Limit Orders* (2025) [PDF, p. 11] points out that a strategy trained on replayed history can't see its own impact.
- *Consistent time travel for realistic interactions with historical data: reinforcement learning for market making* (2024) [PDF] re-indexes time in historical data. After each action it jumps to the point in history whose state matches the state after that action, so the market's reaction comes from real data. The authors report clearly higher gains than naive sequential replay.

### How this maps onto our simulator

| Our state | What the papers suggest |
|---|---|
| One RNG stream (`rng_` in `EsSimulator`). Any change in our orders shifts every later draw, including the fair-value path. The OF7 review found that "pairing shares the seed, not the path". | Paired-seed evaluation in the 2026 regime-switching paper relies on "same exogenous flow realizations across policies" (p. 14). |
| Maker arrivals `maker_rate·exp(−i/decay)` don't depend on queue size; cancellations are linear in queue size. | QR intensities λ(q) are estimated per queue size from L2 or market-by-order data, and validated with queue-size Q-Q plots and event-transition matrices. |
| Aggressor 5 s markout +0.545 ticks, and corr(30 s delta, next 30 s return) of 0.085 to 0.12. OF8 says the market is "too predictable". | E-mini impulse responses die out within about 1 s; the forecast horizon is about two price changes. |
| The tick env applies actions instantly. The delta-skew rule earns +2.90 at 0 ms and +2.20 at 100 ms. | Latency, modelled separately for entry and cancel, is first-order for passive profit. |
| Facts don't identify mechanisms. The v2 fit matched delta_predict and R² but got the mechanism wrong. | Calibrate to richer targets: λ(q) curves, the impact profile with a square-root exponent, markout curves, and queue-size distributions. |

### Ranked changes to try (simulator)

1. **Split the RNG into one stream per exogenous process.** Use separate streams for:
   - fair value, jumps and activity noise;
   - noise-taker arrivals;
   - meta-order starts, sizes and sides;
   - iceberg arrivals;
   - maker arrivals and cancellations.

   Two policies run on the same seed then share the same fundamental path and the same exogenous order arrivals, even though the book reacts to their orders. This is the cheapest fix for the OF7 MAJOR finding. Measure the paired-difference standard deviation before and after; it was 3.8 to 6.2 ticks against an episode standard deviation of 6.6. Change `simulator.hpp` (`Rng rng_` becomes several `Rng`s seeded from one seed). Evidence: the paired-seed protocol in *Deep Learning of Robust Market Making under Regime-Switching Order Flow* (2026).
2. **Add latency to `TickScalpEnv`, modelled separately for submitting and cancelling.** Default to 100 ms to match `ScalpEnv`, and train with a random 30 to 300 ms. Today `manage()` submits and cancels inside the same simulator step. Evidence: *Integrating Tick-level Data and Periodical Signal for High-frequency Market Making* (2023); *Resolving Latency and Inventory Risk in Market Making with Reinforcement Learning* (2025) [abstract]; the prediction-market thesis (2026).
3. **Recalibrate how fast predictability decays.** Measure the mid's impulse response to a 1 s order-flow shock, and the aggressor markout curve at 1, 5, 30 and 60 s. Target the E-mini fact that the response is "almost entirely within a second", plus the real markout curve once ES data is loaded. Two levers:
   - Add a fast-competitor class that detects a running meta-order from signed child flow and trades ahead of it, which removes the slow trail.
   - Shorten informed meta-order persistence (`meta_child_rate`, size cap).

   Re-run OF2, OF3 and OF6 afterwards. Evidence: *Returns and Order Flow Imbalances…* (2025) [abstract]; *Deep order flow imbalance…* (2023) [abstract]; *When AI Trading Agents Compete…* (2025); *Market Informedness and Market-Maker Profitability…* (2026) [abstract].
4. **Make maker intensities at the touch depend on queue size, and calibrate them with the QR counting estimator.** For each queue size n at the best level, estimate λ_L(n), λ_C(n) and λ_M(n) from L2 data, as event counts at size n divided by time at size n, optionally conditioned on hour of day and the last event type. Add "λ(n) curves at L1" as a calibration target in `stylized.hpp` / `of_calibrate`. Evidence: *Deep Learning Meets Queue-Reactive…* (2025); *Queue-reactive Hawkes models for the order flow* (2019) [abstract]; *Reinforcement Learning in Queue-Reactive Models…* (2025).
5. **Add an impact-profile test to the stylized-facts suite.** Run TWAP buys of 10% to 200% of 5-minute volume through the simulator. Check that impact rises concavely, relaxes after the buy, and has a peak-impact exponent near 0.5. This is cheap and catches mechanism errors that the current facts miss. Evidence: *Deep Learning Meets Queue-Reactive…* (2025, pp. 16-18); *When AI Trading Agents Compete…* (2025, p. 3).
6. **Treat cancel position and competition at join as sensitivity axes, not constants.** Report tick-env results over a grid: `cancel_back_bias` ∈ {1 (uniform, as in the replay papers), 3, back-first (as in the trade-execution paper)} and `join_ahead_mean` ∈ {60, 120, 200}. Calibrate both from market-by-order data when it is available; until then, every passive result is conditional on them. Evidence: *Market Making via Reinforcement Learning* (2018); *Cooperative Multi-Agent RL… Prediction-Market Order Books* (2026); *Reinforcement Learning for Trade Execution with Market and Limit Orders* (2025).
7. **Add a regime-switching scenario for the direction of market-order flow.** Use a slowly varying buy probability for noise and momentum takers, with a mean regime duration τ, as a stress family. It is the documented failure mode for market makers. Evidence: *Deep Learning of Robust Market Making under Regime-Switching Order Flow* (2026).
8. **Estimate passive fills in the replay backtester.** In `replay_env.hpp`, rebuild queue position from L2 using the same cancel-position assumption as the simulator, then sweep it. Prototype "consistent time travel" so our own orders can draw a market reaction from historical data. Evidence: *Consistent time travel…* (2024); *Market Making via Reinforcement Learning* (2018).

---

## 2. The scalping agent (directional, market orders)

### What the literature says

- **RL's value is in turning a signal into trades, and its turnover should scale with signal quality.** *Asynchronous Deep Double Duelling Q-Learning for Trading-Signal Execution in Limit Order Book Markets* (2023) [PDF] feeds a noisy oracle signal at three noise levels (pp. 9-10):
  - RL beats the rule at every level.
  - At the noisiest level, RL ends near zero mean while the rule loses.
  - RL turnover *rises* with signal-to-noise; the agent "learns to trust the signal more".
  - Significance comes from t-tests and bootstrapped confidence intervals over 31 test episodes.

  This is a clean template for a signal-strength sweep.
- **Mark-to-market in the state, and the risk of collapsing to "never trade".** *Deep Reinforcement Learning for Active High Frequency Trading* (2021) [PDF] reports three things:
  - Adding mark-to-market P&L to the state "strongly improves convergence"; adding the spread did not help.
  - Ensembles of 30 elements were used.
  - The agent sometimes "converges to sub-optimal policy such as not trading at all" because expected returns are negative during early exploration.

  Our 100-episode OF3 run showed exactly that, and we already include open P&L in the state.
- **Reward horizon and auxiliary tasks.** *DeepScalper: A Risk-Aware Reinforcement Learning Framework to Capture Fleeting Intraday Trading Opportunities* (2021) [PDF] trains on a *hindsight bonus*, r_t + w·(p_{t+h} − p_t)·pos_t, used only in training while evaluation uses raw P&L. It also adds a volatility-prediction auxiliary head. Both reduce variance across 5 seeds (pp. 5, 9). The prediction-market MARL paper (2026) [PDF] also adds an auxiliary mid-move prediction head.
- **Robustness by randomization, not only adversaries.** *Robust Market Making with Hawkes Order Flow and Price Impact via Adversarial Reinforcement Learning* (2026) [PDF] has three findings:
  - Training on randomly sampled market parameters per episode (truncated normals around nominal) is competitive with adversarial training once the environment has Hawkes flow and impact.
  - A recurrent encoder over the observation window improves the left tail.
  - Robustness should be judged by CVaR10% plus a bootstrap test that the gain wasn't bought with directional inventory (pp. 2, 6).

  *Market Making Strategies with Reinforcement Learning* (2025) [PDF, p. 138] also recommends domain randomization for sim-to-real.
- **Generalization is limited by the number of distinct market paths.** *Towards Generalizable Reinforcement Learning for Trade Execution* (2023) [PDF] attributes RL overfitting to "large context space and limited context samples", and recommends compact representations of the market context. In simulation we can generate unlimited paths. On recorded ES data this becomes the binding limit: number of distinct days.
- **Feature ideas, all [abstract].**
  - *Cross-impact of order flow imbalance in equity markets* (2023): an OFI integrated over several book levels explains impact better than best-level OFI.
  - *Order Flow Decomposition for Price Impact Analysis in Equity Limit Order Books* (2023): decomposing OFI by event type improves forward-looking prediction.

  Our best-level OFI30 rule lost (−1.82, t −3.0), so these are worth a feature test, at low priority.

### Our status

- NeuralQ beats the delta-momentum rule by 2.1 to 2.9x and stays profitable in 9 stress scenarios. A 2-net ensemble adds 15% to 20%.
- OF8 concludes that the absolute P&L isn't believable because the simulator is too predictable. Changing the simulator (section 1, item 3) therefore comes before changing the agent.

### Ranked changes to try (scalping agent)

1. **Signal-strength sweep after recalibration.** Sweep the informed share, meta-order persistence and fast-competitor intensity. At each level, compare RL, the delta rule and the oracle, and report RL turnover. Expected if the agent is sound: turnover falls and the RL edge shrinks toward zero as predictability falls, without collapsing below the rule. This is the test of whether the edge is skill or a simulator artifact. Evidence: *Asynchronous Deep Double Duelling Q-Learning…* (2023).
2. **Randomize the simulator's parameters during training.** Sample `SimParams` per episode from truncated normals around nominal: informed rate, `value_vol`, `fade`, `noise_rate`, latency, fees. Evaluate on the held-out OF6 grid plus the new regime and fast-competitor scenarios. Compare against nominal-trained nets on CVaR10%, not only on the mean. Evidence: *Robust Market Making with Hawkes Order Flow…* (2026); *Market Making Strategies with Reinforcement Learning* (2025).
3. **Train with random latency (30 to 300 ms per order)** in `ScalpEnv` instead of a fixed `latency_steps = 1`. OF6 showed RL losing a little at 1 s latency; training on a range should flatten that. Evidence: *Resolving Latency and Inventory Risk…* (2025) [abstract]; *Integrating Tick-level Data…* (2023).
4. **Add an auxiliary prediction head** (next-30 s mid move or realized volatility) on the NeuralQ trunk, as an ablation with 4 seeds. It changes `rag/agent`, so it costs more. Evidence: *DeepScalper* (2021); *Cooperative Multi-Agent RL… Prediction-Market Order Books* (2026).
5. **Hindsight bonus, training only** (h of 30 to 60 s, small w). Ablate it against the plain reward to see whether it cuts orders per episode without cutting net ticks. Evidence: *DeepScalper* (2021).
6. **Add a recurrent or history encoder for regime identification.** Only do this if the regime-switching stress (section 1, item 7) shows a gap; our features are already windowed aggregates. Evidence: *Robust Market Making with Hawkes Order Flow…* (2026).
7. **Feature test: multi-level integrated OFI and OFI split by event type**, run against the current 13 features. Evidence: *Cross-impact of order flow imbalance in equity markets* (2023) [abstract]; *Order Flow Decomposition…* (2023) [abstract].

---

## 3. The tick-scalping agent: why RL ends up not quoting, and how papers avoid it

### The same failure is documented, and named

- **JaxMARL-HFT.** In *JaxMARL-HFT: GPU-Accelerated Large-Scale Multi-Agent Reinforcement Learning for High-Frequency Trading* (2025) [PDF], the market makers "learn to trade very infrequently" (p. 7):
  - The reward isn't normalized by traded volume, so "an optimal policy seems to be to never trade, thus guaranteeing no losses".
  - An inventory penalty makes it worse.
  - Even converged agents lose about 0.2 ticks on average.
  - The authors blame part of it on cancelling and reposting every step: "a marked disadvantage in a price-time priority LOB".
- **The prediction-market MARL paper.** *Cooperative Multi-Agent Reinforcement Learning for Market Making on Prediction-Market Order Books* (2026) [PDF] calls this the **no-trade collapse**: 61.1% to 97.7% no-trade actions across seeds in JaxMARL-HFT. Its analysis (pp. 5-6, 14-17) is the closest match to our OF5:
  1. On realistic flow, "fills are disproportionately informed", so quoting has negative expected value early in training.
  2. Removing the no-trade action doesn't help. The agent "moved most of its action mass onto the widest symmetric rows … where nothing fills". *Inaction is a reward problem.*
  3. The basic reward (spread capture plus mark-to-market of inventory) has three faults:
     - it is dominated by price noise from the revaluation term;
     - it doesn't charge for inventory size;
     - it misassigns credit over time. Spread is credited at the fill, while the adverse move lands a few steps later.

     Inventory penalties make the first and third faults worse, because any penalty routed through the reward is smallest at zero inventory.

### Why our tick agent doesn't quote (OF5 and OF7 against these mechanisms)

| Mechanism in the literature | Our evidence |
|---|---|
| Cancelling and reposting destroys queue priority (JaxMARL-HFT 2025). Queue priority is "an economic resource the agent earns by holding" (regime-switching 2026, p. 11). | OF7: uniform exploration gives 0.10 fills per episode against 8.4 for the rule. Random mode changes cancel, then rejoin behind about 120 lots. 2% random deviations cut the rule from +3.05 to +1.23; 5% cut it to +0.03. |
| Fills are adversely selected, so random quoting has negative expected value (prediction-market MARL 2026). Exploration returns are negative, so the agent converges to not trading (active HFT 2021). | Sealed OF4: quoting both sides earns −2.77 ticks per 5 min. That is +0.05 gross, against 23.5 fills × 0.12 ticks of fees. "Stay out" is the greedy answer to the data the agent sees. |
| A mark-to-market reward is dominated by noise (prediction-market MARL 2026, p. 15). | Our reward is the change in equity at the mid, net of fees: the "basic reward". The per-episode standard deviation is about 6.6 ticks against a rule edge of about 3 ticks. |
| Even good designs end near zero or negative on realistic flow: JaxMARL-HFT about −0.2 ticks; the prediction-market MARL paper's best mean is −4.3 ticks per episode. | Our conservative queue model (bias 3, join 120) was set on purpose so that naive quoting loses. "Don't quote" may be the right answer for a slow 1-lot participant. The honest bar is the delta-skew rule (+2.90 at 0 ms, +2.20 at 100 ms) on fresh seeds. |

### How the papers avoid it

- **Control inventory outside the reward.** The prediction-market MARL paper (2026, p. 16) uses a hard inventory cap plus a *loss stop with an annealed threshold*: once the episode loss passes the threshold, the agent can only close. The threshold starts loose and tightens. It uses no inventory penalty. Result: no-trade share 4.4% against 19.2% for the JaxMARL-HFT baseline, and −9.6 against −46.1 ticks per held-out episode.

  The regime-switching paper (2026) also uses a hard bound, applied *after* the action is chosen: when the inventory is at its limit, the leg that would add to it is dropped. The reward and next state are computed on the quotes actually sent (p. 11).
- **Reward the quality of fills, not mark-to-market.**
  - The prediction-market MARL paper adds a **markout term** (mid move over the next 5 steps × the quantity just filled), with its coefficient **annealed up from zero** "so the quoter learns to quote before it is judged on the quality of its fills" (pp. 17-18). The markout was the most effective fix: the bare split scored −72.3 ticks, blend −41.0, markout −5.9, and markout plus curriculum −4.3.
  - The regime-switching paper (2026) uses a dampened fill PnL: each fill is revalued at the mid after the decision, minus a quadratic inventory penalty plus a one-sided "inventory wall" term (p. 12).
  - *Market Making via Reinforcement Learning* (2018) [PDF] uses **asymmetrically dampened PnL**, r = Ψ − max(0, η·Inv·Δm), which removes speculative upside but keeps downside. It improved learning stability and asymptotic performance over the plain PnL reward, which the authors found unstable and poor out of sample (pp. 4, 7).
  - *Multi-Level Market Making with Reinforcement Learning* (2026) [PDF] keeps the mark-to-market change in the reward as potential-based shaping: denser signal, same optimal policy. It adds a linear |Q| penalty (pp. 5-6).
- **Keep queue priority across decisions, with a compact action set.**
  - The regime-switching paper (2026, p. 11) leaves a resting order in place when the target price is unchanged ("smart quoting … essential").
  - It restricts actions to 6 one-tick offset pairs. Under stationary flow, 6 and 9 actions converged to the same result (+0.339 against +0.331). Under limited non-stationary fine-tuning, 6 actions won at every checkpoint (+0.097 against −0.005).
  - Multi-level MM (2026) benchmarks "only cancel resting orders when necessary … to retain queue positions".
  - The order-stacking paper (2022) [abstract, KOSPI200 index futures] quotes at several levels beyond the best "to reserve good queue position".
- **Use state features that price the queue.** The regime-switching paper's quote-exposure feature is O = 1/(1 + lots ahead) per side, with imbalance (O_b − O_a)/(1 + O_b + O_a) (p. 10). The Deep Hawkes paper decomposes order value as fill probability × (half-spread − adverse selection cost).
- **Use structured exploration and warm starts.**
  - The regime-switching paper uses Rainbow: NoisyNet parameter-space exploration, prioritized replay, n-step returns and a C51 distributional head. Its reason: fill rewards are "sparse and heavy-tailed" (p. 12). It also warm-starts fine-tuning from a trained stationary controller.
  - *Reinforcement Learning with Expert Trajectory For Quantitative Trading* (2021) [abstract] mixes expert and agent transitions in the TD error.
  - The prediction-market MARL paper uses an entropy bonus against the collapse, and argues for on-policy learning over DQN's "deadly triad" (p. 19).
- **Use one-sided and skewed actions.** The prediction-market MARL paper keeps an explicit no-trade row but adds skewed and one-sided rows. It cites Wang et al. that allowing abstention or one-sided quotes improves risk-adjusted returns (pp. 5-6, 13).

  The 2018 paper goes the opposite way: the agent must quote at all times unless at the inventory limit. JaxMARL-HFT notes that forcing quotes reduced performance with its SpreadSkew action space.
- **Measure the collapse.** Report the no-trade share and split portfolio value into spread capture and carry, alongside P&L. "Treat an improvement accompanied by a large rise in that share as withdrawal from trading" (prediction-market MARL paper, p. 30).

### Ranked changes to try (tick-scalping agent)

1. **Add diagnostics before changing the learner.** Per evaluation, report:
   - quoting share per mode, and the share of time with no quote;
   - fills per episode;
   - P&L split into spread capture at fill time (fill price against mid) and carry/markout;
   - own-fill markout curve at 1, 5 and 30 s.

   Put this in `tick_env.hpp` accessors and `of5`/`of7` reporting. It is cheap and makes every later experiment interpretable. Evidence: *Cooperative Multi-Agent RL… Prediction-Market Order Books* (2026); *Multi-Level Market Making with Reinforcement Learning* (2026).
2. **Train on a markout reward with an annealed coefficient; evaluate on true net P&L.** The training reward is:

   r = Σ_fills side·(m_{t+H} − p_fill) − fees − κ·(cost of flattening)

   with H of 5 to 10 s. Blend it with the current Δequity reward as `(1−β)·Δequity + β·markout_reward`, with β annealed from 0 to 1 over the first third of training. Keep the episode-end flatten. Alternative arm: asymmetric dampening, Δequity − max(0, η·pos·Δmid). This tests the "reward problem" diagnosis directly. Evidence: *Cooperative Multi-Agent RL… Prediction-Market Order Books* (2026); *Deep Learning of Robust Market Making under Regime-Switching Order Flow* (2026); *Market Making via Reinforcement Learning* (2018); *JaxMARL-HFT* (2025).
3. **Explore without destroying queue position.** Arms:
   - (a) *sticky* exploration: an exploratory mode is held for k decisions, k ≈ 10 to 40, that is 5 to 20 s;
   - (b) exploration only among modes that keep an existing resting order (3→1 or 3→2, not 3→0→3);
   - (c) parameter-space noise (NoisyNet-style), if the learner supports it;
   - (d) the OF7 rule-guided episodes, kept.

   Measure fills per episode during exploration against the rule's 8.4. This attacks the measured root cause of OF5. Evidence: *Deep Learning of Robust Market Making under Regime-Switching Order Flow* (2026); *JaxMARL-HFT* (2025); OF7.
4. **Control inventory outside the reward.** Keep the ±1 cap, which already exists. Add an annealed per-episode loss stop: past the threshold, only flatten or the closing side is allowed, and the threshold tightens over training. Never add a |q| penalty to the reward. Evidence: *Cooperative Multi-Agent RL… Prediction-Market Order Books* (2026); *JaxMARL-HFT* (2025).
5. **Add queue-value features:**
   - O_b = 1/(1 + lots ahead of our bid), and O_a for the ask;
   - exposure imbalance x = (O_b − O_a)/(1 + O_b + O_a);
   - log(1 + lots ahead), in addition to today's ratio of lots ahead to displayed size;
   - optionally a running estimate of each side's fill probability and adverse-selection cost.

   Evidence: *Deep Learning of Robust Market Making under Regime-Switching Order Flow* (2026); *Deep Hawkes Process for High-Frequency Market Making* (2021); *Optimal Market Making in Prediction Markets…* (2026).
6. **Learn on top of the delta-skew rule.** Keep the OF7 rule-anchored deployment and also learn the advantage of deviating, Q(s,a) − Q(s, rule(s)), so the learner starts at the rule's +2.90. Combine this with item 3 (d). Evidence: the regime-switching paper warm-starts from a trained controller and benchmarks against a calibrated analytic policy; *Reinforcement Learning with Expert Trajectory For Quantitative Trading* (2021) [abstract].
7. **Try a distributional value head (C51-style)**, or at least quantile or Huber targets, for heavy-tailed, sparse fill rewards. This is a learner change in `rag/agent`. Evidence: *Deep Learning of Robust Market Making under Regime-Switching Order Flow* (2026).
8. **Action-space variants:**
   - one-sided "lean" modes that quote the closing side only while holding;
   - a "stack" mode that rests a 1-lot one level behind the touch to build queue priority, with `max_position` 2 so stacking doesn't block the touch order.

   OF7 found keep-levels hurt the *rule* (+3.05 to about +2.5), so treat this as a learner option, not a default. Evidence: *Market Making under Order Stacking Framework…* (2022) [abstract]; *Optimal Market Making in Prediction Markets…* (2026, scale quotes, pp. 42-43); *Multi-Level Market Making…* (2026, TOP2 markouts are better).
9. **On-policy actor-critic with an entropy bonus** as a second learner, using the existing E5 actor-critic in `rag/agent`. Compare it with NeuralQ on no-trade share. Evidence: *Cooperative Multi-Agent RL… Prediction-Market Order Books* (2026, pp. 19, 29).
10. **Quoter and risk-manager split** with a blended reward and a markout term. This only makes sense if inventory grows beyond ±1; it is listed for completeness. Evidence: *Cooperative Multi-Agent RL… Prediction-Market Order Books* (2026). The bare split was the *worst* configuration there (−72.3) until blend and markout were added.

---

## 4. Validation

### What the literature says

- **Paired evaluation on common random numbers, with proper tests.** The regime-switching paper (2026) [PDF]:
  - evaluates every policy on "matched realizations of the order-flow process";
  - keeps training and evaluation episodes "strictly disjoint";
  - reports paired Wilcoxon and paired t-tests on per-seed differences, with N of 100 to 1,000 paired episodes (pp. 2, 14, 18-23).

  The signal-execution DDQN paper (2023) uses t-tests and bootstrapped confidence intervals over test episodes.
- **Tail-focused robustness metrics.** The Hawkes adversarial market-making paper (2026) [PDF] uses CVaR10% as the primary metric across 19 evaluation environments, plus a one-sided bootstrap test showing the tail gain did not come from directional terminal inventory (pp. 2, 6).
- **Metrics that expose degenerate policies.** The prediction-market MARL paper (2026, p. 30) reports four things for every configuration:
  - held-out portfolio value, averaged over the last five evaluations;
  - stability, as the standard deviation of the last ten;
  - no-trade share;
  - the split into spread capture and carry, and the distribution across held-out windows.

  It also admits a single seed per configuration as a limitation.
- **Overfitting and selection, [abstract] only.**
  - Treat backtest overfitting as a hypothesis test, estimate the probability of overfitting, and reject agents that are likely overfit: *Deep Reinforcement Learning for Cryptocurrency Trading: Practical Approach to Address Backtest Overfitting* (2022).
  - Use walk-forward testing with 34 rolling test periods and strict information-set discipline, and report an insignificant aggregate (p = 0.34) honestly, with strong regime dependence: *Interpretable Hypothesis-Driven Trading:A Rigorous Walk-Forward Validation Framework…* (2025).
  - Validation APY above 300% against more than 70% live capital decay, attributed to survivor bias from selecting under high variance: *The Red Queen's Trap…* (2025).
- **Limits of generalization.** Overfitting scales with the size of the context space against the number of context samples: *Towards Generalizable Reinforcement Learning for Trade Execution* (2023) [PDF]. For ES replay, count distinct days, not windows.
- **Sim-to-real.** *Market Making Strategies with Reinforcement Learning* (2025) [PDF, pp. 138-139] says a simulator must first match the real asset (spreads, volatility, returns), then be monitored and re-tuned as markets drift. It recommends mixing real and simulated experience, domain randomization, or online training. It cites no transfer result. *The corpus has no empirical sim-to-real transfer study for trading.*
- **Not in the corpus:** the deflated Sharpe ratio, probability of backtest overfitting via combinatorially symmetric cross-validation, and purged or embargoed k-fold. These are still worth implementing, but this synthesis can't cite a source for them from the corpus.

### Where we stand

- **What we already do well:** sealed seed blocks, champion selection on a separate validation block, stress suites, a desk view and a replay sanity check.
- **Known weaknesses, from the notes:**
  - E27 selection bias on a reused held-out set;
  - champion-selection optimism of 1 to 2 ticks in OF3;
  - the sealed 910M block was viewed repeatedly in OF7;
  - paired comparisons share the seed, not the path;
  - one corrected baseline (OF4 binary mix-up).

### Ranked changes to try (validation)

1. **Common random numbers plus paired tests.** After the RNG split (simulator item 1), compare policies on per-episode paired differences, with a paired t-test, a Wilcoxon test and a bootstrap confidence interval. Report the paired-difference standard deviation next to every comparison. Evidence: *Deep Learning of Robust Market Making under Regime-Switching Order Flow* (2026); *Asynchronous Deep Double Duelling Q-Learning…* (2023).
2. **Keep a seed-block ledger and confirm on fresh seeds.** Record every seed block and every time it was viewed, for example in `experiments/seed_ledger.csv`. A claim counts only when confirmed on a block that was never viewed before (940M, 950M and so on). Always report the gap between validation and test. Evidence: *The Red Queen's Trap…* (2025) [abstract]; our E27 and OF7 review.
3. **Account for selection.** For each reported champion, log how many configurations, seeds and checkpoints were tried. Split the test episodes into blocks and estimate how often the validation-best configuration lands below the median out of sample: a probability-of-overfitting estimate in the spirit of the 2022 paper. Reject agents whose rank isn't stable. Evidence: *Deep Reinforcement Learning for Cryptocurrency Trading: Practical Approach to Address Backtest Overfitting* (2022) [abstract]; the deflated Sharpe ratio itself is not in the corpus.
4. **Tail metrics in the stress suite.** Add CVaR10% per scenario to OF6 and `validation.hpp` (E28 already has CVaR5), plus a bootstrap test that the gains don't come from directional inventory. Evidence: *Robust Market Making with Hawkes Order Flow and Price Impact via Adversarial Reinforcement Learning* (2026).
5. **Degenerate-policy metrics for every passive agent:** no-trade share, fills, the spread-against-carry split and own-fill markouts (the same as tick item 1). Evidence: *Cooperative Multi-Agent RL… Prediction-Market Order Books* (2026).
6. **Sensitivity surfaces instead of point results.** Report tick and scalp results over the simulator parameters we can't yet calibrate: cancel bias, competition at join, informed share, latency. A claimed edge must hold across the plausible region. Evidence: the market-making papers above, which show results moving with cancel and queue assumptions.
7. **A sim-to-real ladder for recorded ES data**, in this order:
   1. stylized facts plus the new impact, λ(n) and markout targets (`of_calibrate facts`);
   2. replay of directional strategies (market orders only, already supported);
   3. replay of passive strategies with L2-inferred queue position under a swept cancel assumption;
   4. walk-forward over many distinct days, split by regime (open, news minutes, midday);
   5. paper trading.

   Report per-regime results, because the walk-forward paper (2025) and the E-mini paper (2025) both find strong regime and news effects. Evidence: *Market Making Strategies with Reinforcement Learning* (2025); *Towards Generalizable Reinforcement Learning for Trade Execution* (2023); *Interpretable Hypothesis-Driven Trading…* (2025) [abstract]; *Returns and Order Flow Imbalances…* (2025) [abstract]; *Consistent time travel…* (2024).

---

## Overall priority order across areas

| # | Change | Applies to | Main evidence |
|---|---|---|---|
| 1 | One RNG stream per exogenous process, for paired comparisons on the same market path | simulator / validation | Regime-Switching MM (2026) |
| 2 | Tick-env diagnostics: no-trade share, fills, spread-against-carry split, own-fill markouts | tick-scalping agent | Prediction-Market MARL MM (2026); Multi-Level MM (2026) |
| 3 | Separate submit and cancel latency in `TickScalpEnv` (100 ms default, randomized in training) | simulator | Tick-level MM (2023); Resolving Latency (2025, abstract) |
| 4 | Markout or fill-quality training reward with an annealed coefficient; evaluate on net P&L | tick-scalping agent | Prediction-Market MARL MM (2026); Regime-Switching MM (2026); MM via RL (2018); JaxMARL-HFT (2025) |
| 5 | Exploration that preserves queue position (sticky modes, order-keeping transitions, guided episodes) | tick-scalping agent | Regime-Switching MM (2026); JaxMARL-HFT (2025) |
| 6 | Recalibrate predictability decay (impulse response about 1 s, markout curve); add fast competitors to the meta-order trail; re-run OF2, OF3 and OF6 | simulator | E-mini OFI (2025, abstract); Deep OFI (2023, abstract); AI Agents Compete (2025); Market Informedness (2026, abstract) |
| 7 | Seed-block ledger, fresh-seed confirmation and selection accounting | validation | Backtest Overfitting (2022, abstract); Red Queen's Trap (2025, abstract) |
| 8 | Signal-strength sweep for the scalping agent (turnover against signal quality, RL against rule) | scalping agent | Async DDDQN Signal Execution (2023) |
| 9 | Inventory control outside the reward: annealed loss stop, no |q| penalty | tick-scalping agent | Prediction-Market MARL MM (2026); JaxMARL-HFT (2025) |
| 10 | Queue-value features: O = 1/(1 + lots ahead), exposure imbalance, fill probability and adverse-selection cost | tick-scalping agent | Regime-Switching MM (2026); Deep Hawkes MM (2021) |
| 11 | Queue-reactive maker intensities at L1, calibrated with the QR counting estimator; λ(n) calibration target | simulator | Deep Queue-Reactive (2025); QR Hawkes (2019, abstract); RL in QR (2025) |
| 12 | CVaR10% and an inventory-bias bootstrap in stress validation | validation | Robust Hawkes Adversarial MM (2026) |
| 13 | Randomize simulator parameters during scalping training | scalping agent | Robust Hawkes Adversarial MM (2026); MM Strategies thesis (2025) |
| 14 | Impact-profile and square-root-law test in the stylized-facts suite | simulator | Deep Queue-Reactive (2025) |
| 15 | Learn the advantage over the delta-skew rule, warm-started from the rule | tick-scalping agent | Regime-Switching MM (2026); RL with Expert Trajectory (2021, abstract) |
| 16 | Sensitivity surfaces over cancel bias, competition at join, informed share and latency | validation | MM via RL (2018); Prediction-Market MARL MM (2026); RL Trade Execution Market+Limit (2025) |
| 17 | Regime-switching direction of market-order flow as a stress family | simulator | Regime-Switching MM (2026) |
| 18 | Distributional (C51-style) value head for sparse, heavy-tailed fill rewards | tick-scalping agent | Regime-Switching MM (2026) |
| 19 | Auxiliary prediction head and training-only hindsight bonus for the scalping agent | scalping agent | DeepScalper (2021); Prediction-Market MARL MM (2026) |
| 20 | Sim-to-real ladder with L2-inferred queue replay and time-travel replay | validation | Consistent Time Travel (2024); Generalizable Trade Execution (2023); MM Strategies thesis (2025) |
| 21 | Order-stacking and one-sided lean modes, as learner options | tick-scalping agent | Order Stacking MM (2022, abstract); Prediction-Market MM thesis (2026); Multi-Level MM (2026) |
| 22 | On-policy actor-critic with an entropy bonus as a second tick learner | tick-scalping agent | Prediction-Market MARL MM (2026) |

**Caveat for the tick agent.** On realistic flow, no market-making paper in the corpus reports a robustly positive mean for a slow passive agent:
- JaxMARL-HFT (2025): about −0.2 ticks;
- prediction-market MARL MM (2026): best mean −4.3 ticks per episode;
- the regime-switching paper (2026): positive, but in a zero-intelligence simulator with unit lots.

The success criterion should be "match or beat the delta-skew rule at 100 ms latency on never-viewed seeds, with a meaningful no-trade share". Positive P&L by itself isn't the bar.
