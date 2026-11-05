# Objectives

What this project is for, what "done" means for each part, and what is
deliberately out of scope. The history at the end records how the
objectives grew, so a new contributor can tell requirements from
experiments.

## 1. The RAG engine

**Goal:** a header-only C++20 retrieval-augmented generation (RAG) engine
written the way low-latency trading systems are written: predictable tail
latency, no allocation on the hot path, explicit memory layout, and
measured rather than assumed performance.

Functional requirements:

* Ingest documents (plain text, source files, PDFs with page numbers),
  chunk them with overlap, embed and index them.
* Hybrid retrieval: dense vector search plus BM25, fused by reciprocal rank
  fusion, with four dense modes (exact, int8 + rerank, IVF, IVF-SQ8).
* Build a prompt into a fixed buffer, with citations (`[n] (source, p. X)`),
  ready for any LLM (the generator itself is a concept, not a dependency).
* Persist an index to disk and load it back, safely.

Non-functional requirements and how they're enforced:

| Requirement | Target | Enforced by |
|---|---|---|
| No heap allocation on the query path | 0 allocations per query | global `operator new` counter in tests (R2) |
| Concurrency | queries concurrent with ingest and commit | TSan job, `rag_stress` (R7) |
| Batched queries | bit-identical to single queries | tests (R6) |
| Crash safety | saves survive power loss; forged files are rejected | fsync + rename + directory fsync (R10); structural validation and forgery tests (R9) |
| Robustness | no UB or memory errors on any input | ASan/UBSan (GCC and Clang), libFuzzer harnesses |
| Latency | sub-millisecond p99 at 60k chunks | IVF-SQ8: p99 0.43 ms (R5; README) |

## 2. The self-learning agent

**Goal:** a self-learning agent whose architecture can be studied piece
by piece (environment, encoder, learners, exploration, supervision), on
tasks where the optimum is known exactly, so every claim is a measurement.

Objectives, in the order they were added:

1. Learn optimal execution (sell 100 lots in 20 steps with a predictive
   signal and market impact) from rewards alone, scored against an exact
   DP optimum.
2. Supervise learning the way production systems need: held-out
   evaluation, champion/challenger deployment, drift detection and
   re-exploration, a policy bank for returning regimes, and a guard against
   deploying anything worse than a simple baseline.
3. Compare learning paradigms on the same task: tile-coded Q-learning,
   episodic memory, actor-critic, and a model-based agent that learns the
   market and plans, including what happens when the model is wrong and how
   to detect it.
4. Show the architecture is task-agnostic: a second task (market making)
   on which the unchanged agent finds the optimal policy.
5. Measure the things means hide: online cost of learning, tail risk,
   off-policy evaluation from logs.

What "success" means for the agent: each experiment answers a stated
question with numbers, across seeds, against the exact optimum, and records
the result even when it's negative (EXPERIMENTS.md). Library defaults
change only when an experiment shows they should (DECISIONS.md).

## 3. ES order-flow trading (current focus)

**Goal:** RL agents that trade ES futures from order flow, validated to
a trading desk's standard before any real money. Two strategies:
order-flow scalping/momentum (market orders) and tick scalping (limit
orders, queue position). Develop on a realistic simulator now; calibrate
to and backtest on the user's ES tick data next; train on the user's
many-core server. Success criteria, standard of evidence, status and the
ranked plan: `ORDERFLOW.md`.

## 4. The hand-off pack

**Goal:** another AI, or a person, can take over the project from these
documents alone.

* `llms.txt` and `docs/ai/` describe everything; `tools/make_context_pack.py`
  bundles them into one file (optionally with every source file).
* Every API snippet in API.md compiles and runs as a test.
* The pack is kept current after every experiment and pushed with it.

## Non-goals

* A neural embedding model or an LLM client. Both are concepts
  (`Embedder`, `Generator`) so they can be plugged in; none could be
  downloaded and tested in this environment.
* Distributed or multi-node operation; deletes and updates of indexed
  chunks (STATUS.md lists these as limitations).
* A realistic market simulator. Both trading environments are stylized on
  purpose: they keep an exact optimum computable, which is what makes the
  agent's numbers meaningful. Real market data would make the price-impact
  model an untestable assumption.
* GPU support.

## How the objectives grew (the requests, in order)

| Request | What it produced |
|---|---|
| "Make a custom C++ RAG, with high performance snippets and architecture" | the engine (`include/rag/`) |
| "Make the repo" | https://github.com/Rcoding123/rag-cpp |
| "Can you test it?", then "keep building", "keep doing more tests" | CI matrix, concurrency, IVF-SQ8, persistence, batching, coverage, fuzzing, forgery, differential and stress tests |
| "Make notes on PDFs", "add PDF support" | Poppler extraction, text cleanup, page citations, the `rag_pdf` CLI |
| "Make a self-learning agent; I want to learn the architecture" | `include/rag/agent/`, `docs/agent.md` |
| "Keep experimenting with the RL… keep programming, testing and simulating" | experiments E1–E14 (EXPERIMENTS.md) |
| "Create documents, layouts etc. along with the RAG engine so I can feed all of it to another AI" | this hand-off pack |
| "Make the documents: the specs you're using, the libraries, objectives" | SPECS.md, OBJECTIVES.md, TASK_SPECS.md |
| "I need to run this on Windows too", "make sure it's optimized" | MSVC build, presets (AVX2, AVX-512), `docs/WINDOWS.md`; verified 285/285 on the user's Ryzen 9 9900X |
| "Aim for a proper trading strategy and validation" | `validation.hpp` (stress suite, sealed seeds), E28 |
| "ES futures… directional trading… order flow… two agents: scalping/momentum and tick scalping (queue, OFI, order arrival)" | `include/rag/orderflow/`, OF1–OF9, `ORDERFLOW.md` |
| "Start a web crawler for RL trading papers; make it anonymous" | `tools/research_crawler.py`, `experiments/research_synthesis.md` |
| "Make the code suitable for my server (and rented GPUs); the next AI should pick up the work" | `of_train` (parallel seeds), parallel calibration, `ORDERFLOW.md` §3 and §7 |
