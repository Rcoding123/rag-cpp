# Status, limitations and next steps

## Works and is verified

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
  * two-stage drift detection with re-exploration;
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
    diagnostics) and exact policy evaluation under a model.
* **Testing:**
  * the CI matrix covers SIMD levels, sanitizers and fuzzing;
  * 202 ctest cases: 191 in `rag_tests`, 9 PDF tests (Poppler), the
    stress run and a `rag_pdf` CLI smoke test.
  * The TSan CI job skips the single-threaded learning suites (`Agent*`,
    `ModelBased`, `ExecutionCosts`, `OffPolicy`, `Reference`, `ApiExamples`), which ran
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
  * the tile-coded learner reaches 69–70% of the optimal gain at 40k
    episodes and 75–77% at 200k. The gap is target noise and coverage, not
    the representation, which can hold a 99.5% policy (E2);
  * `ExecutionEnv` is a stylized model: linear temporary impact, an AR(1)
    signal, no permanent impact, no order-book dynamics.
* **Machine-specific numbers:** benchmarks come from one shared 4-vCPU VM.

## Next steps (in rough priority order)

1. A neural `Embedder` (e.g. ONNX Runtime with a MiniLM model, 384 dims to
   match), on a machine that can download models.
2. Incremental BM25 segments with background merging.
3. Agent experiments are ongoing; see `EXPERIMENTS.md`.
4. A richer market simulator (queue position, permanent impact) for the
   agent.
