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
  * two learners (tile-coded Q-learning, episodic memory);
  * an exact DP yardstick;
  * held-out evaluation and champion/challenger deployment;
  * two-stage drift detection with re-exploration.
* **Testing:**
  * the CI matrix covers SIMD levels, sanitizers and fuzzing;
  * 168 ctest cases: 158 in `rag_tests`, 9 PDF tests (Poppler) and the
    stress run.

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
  * the tile-coded learner plateaus around 72% of the optimal gain;
  * results vary across seeds (62–71% at 40k episodes);
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
