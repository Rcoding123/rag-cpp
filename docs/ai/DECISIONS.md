# Decision log

Each entry gives the decision, the alternatives considered, and the evidence
behind it. Bugs that shaped a design are included, because they explain why
the code looks the way it does.

## RAG engine

| # | Decision | Why (evidence) |
|---|---|---|
| R1 | Hybrid retrieval: dense + BM25, fused by RRF (k0 = 60) | Rank fusion needs no score calibration between cosine and BM25. Exact identifiers (e.g. "ZX81Q") are found by BM25 even when embeddings blur them. |
| R2 | Zero allocation on the query path, with per-thread `QueryContext` scratch | Allocation is a source of latency spikes; enforced by a test counter. |
| R3 | `dense_min_score = 0` (strict) | **Bug found by coverage review:** stopword-only chunks have all-zero embeddings, scored 0, and still entered prompts. |
| R4 | int8 rows with fp32 rerank of 128 | Scans are DRAM-bound (~10 GB/s); int8 moves 4× fewer bytes: 15.4 ms → 2.95 ms on 100k rows, recall ≥ 0.95 in tests. |
| R5 | IVF-SQ8 (int8 inverted lists + fp32 rerank) | p99 1.08 → 0.43 ms at 60k chunks against fp32 IVF. |
| R6 | Batched queries, 8 per memory pass | Exact 84 → 460 queries/s. Bit-exactness needed `simd::dot1` (one `dot4` lane), because `dot()`'s 4-accumulator summation order differs. |
| R7 | `commit()` concurrent with queries via immutable snapshots + epoch-based reclamation | Readers pin with one store to their own cache line. A mutant that frees without EBR is caught by `rag_stress` under ASan within seconds. |
| R8 | Storage allocated lazily, query scratch pre-faulted | Huge-page first touch was 11× slower (1.17 s per 200 MB). Writer-filled storage never faults on readers. Load dropped from 5.7 s to < 1 s. |
| R9 | Index file: checksum plus structural validation of every field | **Bug found while testing:** `offset + len > total` overflowed, so a forged file could point a chunk out of bounds; checksums don't stop forgers. |
| R10 | Saves use a tmp file, fsync, rename, then fsync the directory | Rename alone is atomic but not durable after power loss. |
| R11 | Saved snapshot reused on load when the parameters match | IVF training dominated load time (4.2 s). |
| R12 | Arena bounds checked on integer addresses | **Bug found by fuzzing:** `nullptr + n` on a fresh arena. GCC's UBSan misses it, so CI also runs Clang's. |

## PDF support

| # | Decision | Why |
|---|---|---|
| P1 | Poppler (poppler-cpp), reading-order layout | Multi-column pages come out column by column. Optional dependency. |
| P2 | Normalize ligatures (U+FB00–FB06) | Real fonts make Poppler emit U+FB01, which never matches "fi" in queries. |
| P3 | Running header/footer removal: lines recurring at page edges on ≥ 50% of pages (and both pages of a 2-page document) | A real 2-page handout kept its header under the first rule (minimum 3 pages). |
| P4 | Only lines of ≤ 6 words may differ in their digits and still match | Otherwise "Q3 revenue grew 12% ..." vs "Q4 ... 15% ..." would be deleted as a header. |
| P5 | `join_pages` always separates pages, even empty ones | An empty first page would otherwise share a start offset with page 2. |

## Agent

| # | Decision | Why (evidence) |
|---|---|---|
| A1 | An exact DP optimum as the yardstick; report the share of the optimal gain | "Reward went up" can't be compared; "67% of optimal" can. |
| A2 | Paired evaluation on common random numbers | Episode return sd ≈ 55, much larger than the differences between policies. |
| A3 | Tile coding 8×8×8 with 8 tilings (not finer) | 12³ cells: 40%; one cell per time step: 15% (vs 65% at 30k episodes). Generalization beats resolution. |
| A4 | Step size annealed `alpha/(1+ep/3000)` with floor 0.003 | Reward noise ±2.5 per step vs action gaps < 1; the floor keeps the agent adaptable. |
| A5 | Double Q-learning off by default | Level at 30k episodes (64.8 vs 64.5%), worse at 60k (64% vs 72%). |
| A6 | Champion deployed separately from the learner | **Bug:** restoring the champion into the learner at the end of each `run()` undid learning (recovery stalled at 16%). |
| A7 | Drift detection: CUSUM suspects, champion re-score confirms | A learning dip at episode ~34k looked like drift; acting on it cost 18 points. A champion's held-out score is exactly reproducible in an unchanged world. |
| A8 | CUSUM k = 0.3, h = 16, baseline from samples ≥ 50 episodes old, rate 0.001 | Simulated: ~1 false alarm per 200k episodes; a 0.55σ shift detected after a median of 51 episodes. The first detector's baseline absorbed the shift it was meant to detect. |
| A9 | Re-explore with ε 0.5 over 5,000 episodes, and clear replay on drift | Short boosts (0.3 / 600) recovered 10% vs 43%. Clearing replay helps in the first 2,500 episodes only. |
| A10 | Model-based agent: grey-box regression plus a DP planner | 99–100% of optimal after 20–50 episodes vs 67% after 40,000 model-free (E1a). |
| A11 | Model-based `replan()` forces the assumed form (exponent 2, no fixed cost) | **Bug:** the true exponent and fixed cost leaked from `structure` into the planner. Regression test: identical experience with different hidden parameters must give identical plans. |
| A12 | Cost model choice: Shrunk (empirical Bayes, k = 30) is the safe default under uncertainty about the form | Quadratic: 53% when wrong. Per-size: 100% eventually but −1,400% to −1,800% on 4 of 10 seeds at 10 episodes. Shrunk: never below 82% (E1c). |
| A13 | `n_step = 3` by default | Best of n ∈ {1, 2, 3, 5, 10, 20}: faster early (47–50% vs 36–45% at 10k) and consistent at 40k (69–70% vs 62–71%); 75–77% at 200k (E2a). |
| A14 | Don't re-tune the representation to fix the plateau | Fitting the default tiles to the true Q* gives 99.5% (E2b). The gap is target noise and coverage (E2c). |
| A15 | Policy bank on by default: bank champions on confirmed drift, recall one that beats the champion by 2 standard errors | Returning regimes are recalled within 20–131 episodes, at their old level, vs −82% to −109% without the bank (E3). The first visit to a regime is unaffected. |
| A16 | Risk aversion as a per-step reward penalty λ·σ²·I², not as a change to cash | Keeps the cash accounting identity testable, and makes the DP exact for the mean-variance objective. The model-based agent fits the penalty as a regressor on −I² (κ recovered to ~2%, E4). |
| A17 | Model-based estimator: one weighted normal-equations solve over all parameters | Separate fits of alpha and costs are biased once a holding term exists, because inventory correlates with both. Gaussian elimination with partial pivoting and a 1e-9 ridge; ≤ 3 + lots unknowns. |
| A18 | `LearnerKind::ActorCritic` anneals with its own half-life, `ac_decay_episodes = 10000` | The TileQ half-life (3,000) froze the policy near-uniform on 1 of 3 seeds (−235% at 10k). 10,000 gave 75–78% at 10k and 80–86% at 40k (E5). |
| A19 | Optional baseline guard (`Trainer::set_baseline`): deploy the learned champion only after it beats the baseline by 2 standard errors of the paired difference, re-checked at every evaluation and drift | Unguarded TileQ deployed −35% to −106% policies in a weak-signal market (E6). Guarded: TWAP on every seed, identical results in the default market (E6b). Opt-in, because it needs a meaningful baseline. |
| A20 | `learn_permanent_impact = true` by default | Without the term, permanent impact drops the model-based agent to 22–55% and it degrades with more data; with it, 99.8–99.9%. The extra parameter costs ≤ 0.3 points elsewhere (E7b). |
| A21 | Check a learned model by its *paired* predicted vs realized gain over TWAP, one-sided z > 4 | Exact policy evaluation makes the prediction noise-free. Pairing cuts the noise ~5×: 100 episodes separate right (\|z\| ≤ 2.5) from wrong (z ≥ 8.5) models; unpaired needed ~1,500. One-sided because a planner exploiting model errors is always optimistic (E7c). |
| A22 | Warm start by distillation, and always under a guard whose baseline is the model's plan | Fine-tuning alone wrecks a correct model's 99% policy (35–57% for α 0.1–0.01: TD noise floor ≈ √(α/2)·σ > action gaps). With the guard: 100% when the model is right, ≥ the plan and 78–82% when it's wrong (E7d). |
| A23 | `evaluate()`/`compare()` live in `evaluate.hpp`, not `trainer.hpp` | The model-based agent needs paired evaluation for `check_model()` but shouldn't depend on the trainer and the model-free stack. `trainer.hpp` includes it, so no caller changes. |
| A24 | Library default keeps `eps_start = 1`; for online learning on ExecutionEnv-like tasks start at 0.1 | Online, ε 1 lost 1,124–1,504 optimal-episodes by episode 5,000 and broke even at ~12k; ε 0.1 broke even at 1.5k–2.1k and learned as well at 10k/40k/100k (E8). Sparse-reward tasks (the corridor) have no optimistic initial values, so the generic default stays exploratory. |
| A25 | Model-based adaptation by change detection + reset, not constant forgetting; both opt-in | Detection recovered to 95–100% within 20 episodes of a flip with 3 false alarms in 27,000 stable episodes and no stable-market cost; forgetting 0.95 recovered as fast but fell to 54–93% in a noisy stable market (E9). Off by default so stationary results and tests are unchanged. |
| A26 | `DriftDetector` lives in `drift.hpp` | Shared by the trainer and the model-based agent without the agent depending on the trainer (same reasoning as A23). |
| A27 | Off-policy estimators ship with diagnostics (`min_ess`, `uncovered_steps`), and the docs recommend model-based (DM) evaluation plus a model check over IS at this horizon | At horizon 20, IS/WPDIS/DR errors were 45–2,968% of the optimal gain even with 10,000 logs (ESS ≈ 0), against ≤ 4% for DM with the right model. DM's bias under a wrong model doesn't shrink with data (−62% at 1k and 10k), so the model must be checked (E10). |
