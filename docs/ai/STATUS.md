# Status, limitations and next steps

## Works and is verified

* **ES order-flow trading** (`include/rag/orderflow/`, see `ORDERFLOW.md`):
  a tick-level ES simulator calibrated to stylized facts, order-flow
  features, a scalping environment (RL beats every rule, robust to 9
  stresses, but the simulator is too predictable for the absolute P&L to
  mean anything), a tick-scalping environment with FIFO queue position
  (RL matches but doesn't beat the delta-skew rule), a backtester on
  recorded ticks, calibration tooling, a parallel trainer for many-core
  servers, and an anonymous research crawler. Known flaw, first on the
  plan: price discovery is too slow in the simulator (the hidden value
  gap predicts the next 30 s at ~0.6), so RL profits there are inflated;
  the "efficient" preset leaks value through the book (OF9).
  **Order-flow status (2026-11-05):** OF10 iteration 3 is done
  (`e229af4`): an opt-in stationary activity model fixes the warm-up (the
  legacy activity loop is unstable), and the v3 `discovery` preset passes
  the gap, imbalance (marginally), stylized-fact and naive-rule criteria
  with oracles that trade, on tuning, held-out (13–18, once) and fresh
  skeptic seeds. It fails passive quoting (−24 to −27 ticks/episode, stale
  sweeps), so it is accepted for market-order RL only, and meta-orders
  need a 1,050 s warm-up (A62–A66). The RL reruns (OF11, `1a73bd6`) find no edge on `discovery`:
  every seed negative on sealed episodes, ensemble flat, all stresses and
  trading days ≤ 0. OF12 compared v3 with the user's real ES (Sierra):
  imbalance and price-on-delta match; the book is ~8× too deep and the
  mid lags the flow by a second. OF13 (`31f51c4`) added simulator v4,
  the opt-in `discovery_v4` preset (v3 frozen, golden-pinned): all 25
  pre-registered real-ES gate rows (G1–G8) and the efficiency gates pass
  on tuning, held-out (once) and skeptic seeds at the 1,050 s warm-up
  (A67–A69). Both oracles lose after costs on v4, so "no RL edge" there
  is close to true by construction (LESSONS 48); caveats: I1 still
  predicts the next 1 s, the mid flickers, the spread never widens, and
  the real reference is one calm morning. A live training viewer
  (`of_train_live`, `tools/of_train_live.py`) watches RL runs (A70).
  OF14: RL on v4 finds no edge either
  (all 24 seeds negative, mean −3.18 ticks/episode). **Caveat:** the legacy
  activity model is unchanged, so every OF2–OF9 result still ran at ~4
  trades/s instead of 15–22 (ORDERFLOW.md §5).

* **Platforms:** Windows only since 2026-11-04 (Linux CI removed; GCC 13 and
  Clang 18 with sanitizers and fuzzing last passed at `10bfe03`, except a
  gcc-avx2 NLopt configure failure); Windows (Visual Studio 2022 / MSVC, AVX2;
  CI job `msvc-windows`: the AVX2 build's full test suite, demos, latency
  probe, and an AVX-512 compile; the AVX-512 build passed 285/285 on a
  Ryzen 9 9900X under Windows Server 2025, where `train_ensemble`
  reproduced E25's 97.8% in 141 s). Build steps and measured
  latency for Windows: `docs/WINDOWS.md`.
* **RAG engine:**
  * hybrid retrieval in four dense modes;
  * zero-allocation queries;
  * batched queries, bit-identical to single queries;
  * concurrent ingest, commit and query, verified under TSan and with
    `rag_stress`;
  * persistence, with checksum plus structural validation and crash-safe
    saves;
  * PDF ingestion with page citations.
* **Self-learning agent:**
  * model-free learners: tile-coded Q-learning with 3-step returns, and
    episodic memory;
  * a model-based agent (grey-box regression plus DP planning) at 99–100%
    of the optimum after 20–50 episodes;
  * an exact DP yardstick, also for generalized cost tables;
  * held-out evaluation and champion/challenger deployment;
  * two-stage drift detection with re-exploration, plus a change check at
    every evaluation that also catches markets that got better;
  * a policy bank that recognizes returning regimes;
  * an actor-critic learner (the best model-free one here) and risk-averse
    (mean-variance) objectives with an exact optimum;
  * an optional baseline guard: never deploy a policy that hasn't beaten
    TWAP;
  * a paired model check that flags a wrong model after 100 episodes, and
    warm-starting a model-free learner from a model's plan;
  * change detection for the model-based agent: recovery within 20
    episodes of a market flip, free while the market is stable;
  * off-policy evaluation (importance sampling, doubly robust, ESS
    diagnostics) and exact policy evaluation under a model;
  * a second task (market making, exact DP) on which the unchanged agent
    learns the optimal inventory skew, and a model-based market maker;
  * a from-scratch neural Q-learner (`NeuralQ`, no ML library) that breaks
    the tile-coding plateau on execution (91.5% [90.1, 92.7] at 40k
    episodes with Double DQN, 8 seeds; E21 ranks every learner);
  * `ExecutionPolicyTable`: any execution policy compiled into a lookup
    table (3 ns per decision, the network's gain within 0.25 points), and
    a median vote of 8 seeds' tables at 97.5% of the optimum (E22, E23);
  * optional NLopt/Eigen tooling: hyperparameter search validated on
    held-out seeds, sensitivity analysis, rank-revealing least squares.
* **Reviews:** four review agents (adversarial bug hunt, coverage, RAG
  property tests, and a review plus fuzzer for `policy_table.hpp`) found
  and fixed 17 bugs, six of them in the engine (including a heap overflow
  reachable through config and through a forged index file, R13–R17) and
  three in the policy table (A46); agent-header coverage is 99.7% of lines and 99.0% of
  regions. Their schedule fixes may shift the online-gain numbers of E8,
  E11 and E15 slightly; those were measured before the fixes.
* **Testing:**
  * the full suite runs on the user's AVX-512 server and in CI's
    `msvc-windows` (the sanitizer and fuzz jobs were Linux-only and are gone);
  * 287 ctest cases: 272 in `rag_tests`, 4 in `rag_optlib_tests`, 9 PDF tests (Poppler), the
    stress run and a `rag_pdf` CLI smoke test.
  * The TSan CI job skips the single-threaded learning suites (`Agent*`,
    `ModelBased`, `ExecutionCosts`, `OffPolicy`, `MarketMaking`, `Reference`, `ApiExamples`), which ran
    for over an hour under TSan and have no threads to check.

## Known limitations

* **No neural embedder.** `HashingEmbedder` is lexical-ish (feature-hashed
  unigrams and bigrams). The `Embedder` concept is ready for ONNX or
  llama.cpp encoders, but none could be fetched or tested here (Hugging Face
  is blocked by this environment's network policy).
* **No LLM call.** The engine produces prompts; `Generator` is a concept
  only.
* **`commit()` rebuilds BM25 (and IVF) from scratch:** O(total corpus) per
  commit. Incremental segments would be the scalable design.
* **Capacity is fixed** at construction (`max_chunks`), and there are no
  deletes or updates of chunks.
* **PDF scope:** no OCR for scanned pages, and encrypted PDFs are rejected.
* **Agent:**
  * model-free learning on execution is slow because returns are noisy
    relative to the gain (0.27 return-sds per episode): default TileQ
    reaches 65–69% at 40k episodes; coarse tiles + advantage learning
    76–80%; the neural `NeuralQ` with Double DQN 91.5% [90.1, 92.7]
    (8 seeds, E21); the model-based
    agent 99–100% in 20–50 episodes (E1, E16, E19);
  * `ExecutionEnv` and `MarketMakingEnv` are stylized on purpose (exact DP
    optima): no order-book dynamics, queue position or real data;
  * `NeuralQ` is CPU-only and unvectorized (1.7 µs per decision).
* **Optional tooling:** tuning and analysis need NLopt and Eigen
  (`RAG_OPTLIBS`); without them those targets are skipped.
* **Machine-specific numbers:** benchmarks come from one shared 4-vCPU VM.

## Open questions and suspicions (from the review agents, not confirmed as bugs)

* `simd::quantize_i8`: a vector whose largest component is below ~3.7e-37
  overflows the scale (unreachable for unit-norm embeddings).
* `IvfIndex::load` checks row ids are in range but not unique; a forged
  file can repeat ids (duplicate hits, no memory-safety issue).
* IVF-SQ8 reranks committed rows with `simd::dot` but uncommitted rows with
  `dot1`: a row can score an ulp differently before and after `commit()`.
* `RrfParams` and `CleanOptions::repeated_line_fraction` aren't validated
  (NaN or negative values lead to an undefined float-to-int cast).
* `DriftDetector::rebase()` counts the held samples twice in the new
  baseline (slower adaptation, no failing scenario found).
* With `keep_champion = false`, the baseline guard checks the champion but
  deploys the learner.
* `MarketMakingEnv::reset()` doesn't clear `last_bought`/`last_sold`/
  `last_mid_move`; `ExecutionEnv::features` divides by `signal_sd`.
* The online-gain numbers of E8, E11 and E15 were measured before the
  re-plan schedule fix (A40) and may shift slightly.

## Next steps (in rough priority order)

0. **ES order-flow trading** (the current focus): the ranked plan is in
   `ORDERFLOW.md` §7: calibrate the simulator to the user's ES ticks,
   backtest the agents on real ticks, paired evaluation on the same
   market path, the tick scalper at 100 ms latency on fresh seeds.

1. A neural `Embedder` (e.g. ONNX Runtime with a MiniLM model, 384 dims to
   match), on a machine that can download models.
2. Incremental BM25 segments with background merging.
3. Agent: Q-value averaging across seeds (vote on values, not actions;
   RESEARCH.md §6.4); test the hint that the networks over-trade (E23);
   a policy table for market making. (Weight averaging: done, E24;
   averaged networks voted: E25.)
4. Agent: the research's remaining ideas (RESEARCH.md §5): all-action
   counterfactual updates, value/advantage sharing across actions,
   Munchausen targets, weight averaging for the deployed policy.
5. Tune `NeuralQ` and actor-critic with `tune_agent`'s machinery (needs a
   larger compute budget: one NeuralQ evaluation is ~75 s per seed).
6. A richer market simulator (queue position, order book) for both tasks.
7. Resolve the open questions above.
