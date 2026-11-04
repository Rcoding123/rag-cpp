# Project map

rag-cpp is a header-only C++20 library with two parts:

1. **A RAG engine** (`include/rag/`): hybrid retrieval (dense vectors + BM25,
   fused by reciprocal rank fusion) with a zero-allocation query path, SIMD
   kernels, lock-free concurrent ingestion and commits, persistence, and PDF
   ingestion with page citations.
2. **A self-learning agent** (`include/rag/agent/`): reinforcement learning
   with held-out evaluation, champion/challenger deployment and drift
   detection. It is demonstrated on an optimal-execution task whose exact
   optimum is computed by dynamic programming.

## Repository layout

```
rag-cpp/
├── include/
│   ├── rag.hpp                  umbrella header: #include "rag/engine.hpp"
│   └── rag/
│       ├── common.hpp           kCacheLine, AlignedArray (2 MiB huge-page alignment, Init::Prefault/Lazy), ValidDim<D>
│       ├── arena.hpp            bump allocator; blocks never move (stable string_views for readers)
│       ├── tokenizer.hpp        constexpr byte LUT + FNV-1a hashing; constexpr stopword table
│       ├── chunker.hpp          sliding token windows -> byte ranges (zero-copy chunks)
│       ├── embedder.hpp         Embedder concept, HashingEmbedder<D>, embedder_tag<E>()
│       ├── simd.hpp             dot (fp32), dot4 (4 rows), dot1 (one dot4 lane, bit-identical), dot_i8 (AVX-512 VNNI/AVX2/scalar), quantize_i8
│       ├── topk.hpp             TopK<Cap>: fixed-capacity heap, cached admission threshold, min-score floor
│       ├── vector_store.hpp     SoA fp32 + int8 rows, single-writer/multi-reader publication, batched scans
│       ├── ivf_index.hpp        IVF-Flat / IVF-SQ8 (spherical k-means, CSR lists), save/load with validation
│       ├── bm25.hpp             Bm25Builder -> frozen Bm25Index with precomputed per-posting impacts
│       ├── fusion.hpp           reciprocal rank fusion
│       ├── prompt.hpp           ChunkView, citation formatting, prompt assembly into a fixed buffer
│       ├── epoch.hpp            epoch-based reclamation (per-reader padded slots)
│       ├── io.hpp               checksummed binary I/O, bounds-checked reads, fsync + atomic rename
│       ├── engine.hpp           RagEngine: ingest / commit / query / query_batch / save / load / inspect
│       ├── text_clean.hpp       PDF-text cleanup: ligatures, dehyphenation, running headers/footers, reflow
│       ├── pdf.hpp              (optional, Poppler) PDF extraction + ingest_pdf()
│       └── agent/
│           ├── rng.hpp          xoshiro256** + splitmix seeding, Marsaglia normals
│           ├── env.hpp          Environment concept, ChainEnv, ExecutionEnv (costs, risk aversion)
│           ├── execution_dp.hpp exact optimum for ExecutionEnv (cost tables, policy_value, evaluate_policy)
│           ├── features.hpp     TileCoder, RandomFourierEncoder
│           ├── learners.hpp     TileQ, ReplayBuffer, EpisodicMemory
│           ├── agent.hpp        Agent<Env>: TileQ / Episodic / ActorCritic, n-step, distill, snapshots
│           ├── evaluate.hpp     evaluate / compare on common random numbers (EvalStats)
│           ├── drift.hpp        DriftDetector: one-sided CUSUM on surprise (DriftParams)
│           ├── ope.hpp          off-policy evaluation: PDIS, WPDIS, doubly robust, ESS diagnostics
│           ├── model_based.hpp  ModelBasedExecutionAgent: regression + DP planning + check_model
│           ├── market_making.hpp a second task: MarketMakingEnv + exact MarketMakingDP
│           └── trainer.hpp      evaluate, compare, DriftDetector (CUSUM), Trainer (champion, bank, baseline guard)
├── tests/
│   ├── rag_tests.cpp            engine unit/integration tests; global operator-new counter
│   ├── forgery_tests.cpp        every loader check, against files with valid checksums
│   ├── reference_tests.cpp      differential tests vs a naive spec-level reference
│   ├── text_clean_tests.cpp     PDF text cleanup + paged ingest
│   ├── agent_tests.cpp          agent: components through learning outcomes
│   ├── model_based_tests.cpp    generalized costs, model-based agent, risk aversion, permanent impact
│   ├── market_making_tests.cpp  the same agent on market making (E12)
│   ├── api_examples_tests.cpp   compiles and runs every snippet in docs/ai/API.md
│   ├── pdf_tests.cpp            (Poppler) PDF extraction on generated PDFs
│   ├── pdf_writer.hpp           minimal PDF writer for fixtures
│   ├── stress.cpp               rag_stress: concurrent writer/readers with exact invariants
│   ├── synthetic.hpp            synthetic vectors and text corpora
│   ├── test_util.hpp            temp files, checksum fix-up, allocation counter hooks
│   └── fuzz/                    libFuzzer targets (fuzz_load, fuzz_text) + make_seeds
├── bench/                       Google Benchmark suite (rag_bench), tail-latency probe (rag_latency)
├── examples/
│   ├── demo.cpp                 minimal RAG usage
│   ├── rag_pdf.cpp              CLI: index PDFs, ask questions with page citations
│   └── agent_demo.cpp           the self-learning agent end to end (model-free, drift, model-based + model check)
├── docs/
│   ├── agent.md                 the agent's architecture, with every measured lesson
│   └── ai/                      this hand-off pack:
│       ├── README.md            reading order and ground rules
│       ├── OBJECTIVES.md        goals, success criteria, non-goals, request history
│       ├── PROJECT_MAP.md       this file
│       ├── SPECS.md             toolchain, libraries + versions, build options, machine, CI
│       ├── API.md               compiled usage examples and config tables
│       ├── RAG_ENGINE.md        engine internals, invariants, file format
│       ├── TASK_SPECS.md        exact task dynamics, rewards, optima, learner updates, metrics
│       ├── LESSONS.md           24 cross-cutting lessons with their evidence
│       ├── EXPERIMENTS.md       experiment log E0-E14, including negative results
│       ├── DECISIONS.md         design decisions with evidence (R*, P*, A*)
│       ├── CONVENTIONS.md       coding, testing and verification rules
│       └── STATUS.md            what works, limitations, next steps
├── tools/make_context_pack.py   bundles docs (and optionally sources) into one file for an AI
├── llms.txt                     index of this pack for language models
└── .github/workflows/ci.yml     8 CI jobs
```

## Dependency layers (lower layers never include higher ones)

```
L0  common.hpp  rng.hpp
L1  arena  tokenizer  simd  topk  io  text_clean
L2  chunker  embedder  vector_store  bm25  fusion  prompt  epoch  features
L3  ivf_index                    learners      env  market_making
L4  engine                       execution_dp  agent  evaluate  drift  ope
L5  pdf (Poppler)                trainer       model_based
```

The agent reuses the engine's SIMD kernels (`simd.hpp`), top-K selection
(`topk.hpp`) and aligned buffers (`common.hpp`). It doesn't depend on
`engine.hpp`.

## Build targets

| Target | What | Needs |
|---|---|---|
| `rag_tests` | main test binary (rag, forgery, reference, text_clean, agent, model_based, market_making, api_examples) | GoogleTest |
| `rag_pdf_tests` | PDF tests | GoogleTest, poppler-cpp |
| `rag_stress` | concurrency stress test (also registered with ctest, 2 s per mode) | threads |
| `rag_bench` | microbenchmarks | Google Benchmark |
| `rag_latency` | p50/p99/p99.9 query latency | |
| `rag_demo`, `agent_demo`, `rag_pdf` | examples and CLI | `rag_pdf` needs poppler-cpp |
| `fuzz_load`, `fuzz_text`, `make_seeds` | with `-DRAG_FUZZ=ON` and Clang | libclang-rt |

CMake options: `RAG_ARCH` (`-march` value, default `native`), `RAG_SANITIZE`
(ASan+UBSan), `RAG_TSAN`, `RAG_HUGEPAGES` (default ON), `RAG_FUZZ`, `RAG_PDF`
(default ON; used if Poppler is found).

## CI (`.github/workflows/ci.yml`)

* **Test matrix:** gcc native / x86-64-v3 (AVX2) / x86-64 (scalar), clang
  native, gcc ASan+UBSan, clang ASan+UBSan, gcc TSan (concurrency tests
  repeated 25 times plus `rag_stress 10`). All build with `-Werror`.
* **fuzz-smoke:** each fuzzer for 60 s.
