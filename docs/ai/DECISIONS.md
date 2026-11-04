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
