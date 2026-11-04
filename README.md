# rag-cpp — header-only, low-latency Retrieval-Augmented Generation in C++20

[![CI](https://github.com/Rcoding123/rag-cpp/actions/workflows/ci.yml/badge.svg)](https://github.com/Rcoding123/rag-cpp/actions/workflows/ci.yml)

Hybrid retrieval (dense ANN + BM25, fused by RRF) with a zero-allocation query
path, SIMD similarity kernels (AVX-512 VNNI / AVX2 / scalar), and lock-free
single-writer / multi-reader ingestion.

```
INGEST (1 writer)                                   QUERY (N readers, 0 allocations)
text ─► Arena (stable string_views)                 query ─► tokenize   (stack, constexpr LUT)
     ─► tokenize ─► sliding-window chunks                 ─► embed      (stack, L1)
     ─► Embedder ─► VectorStore.append ─┐                 ├─► dense:  Exact | Int8+rerank | IVF | IVF-SQ8, + delta scan
     ─► Bm25Builder.add                 │                 └─► BM25:   impact-scored postings, dense accumulator
commit() ─► Snapshot ─► atomic swap     │                      ─► RRF fusion (stack)
                                        │                      ─► prompt into preallocated buffer ─► Generator
        size.store(n+1, release) ───────┴──► size.load(acquire): new chunks searchable immediately (dense)
```

| Header | Role | Key latency technique |
|---|---|---|
| `common.hpp` | aligned buffers | 64 B / 2 MiB alignment, `MADV_HUGEPAGE`; query scratch pre-faulted, storage faulted by the writer |
| `arena.hpp` | document text | bump-pointer, blocks never move → lock-free readers |
| `tokenizer.hpp` | terms | constexpr 256-byte class/lowercase LUT, constexpr FNV-1a, compile-time stopword table |
| `chunker.hpp` | chunking | zero-copy byte ranges, token-window with overlap |
| `embedder.hpp` | `Embedder` concept + hashing embedder | signed feature hashing, Lemire fast-range (no division) |
| `simd.hpp` | kernels | 4-accumulator FMA, 4-row register blocking, int8 sign-trick / VNNI `vpdpbusd` |
| `topk.hpp` | selection | stack `std::array` heap, cached threshold, `[[likely]]` reject |
| `vector_store.hpp` | dense storage | SoA fp32 + int8 + scale, SWMR via release/acquire, padded atomic |
| `ivf_index.hpp` | ANN | spherical k-means, CSR lists of contiguous reordered vectors, fp32 or int8 (SQ8) storage |
| `bm25.hpp` | lexical | precomputed per-posting BM25 impact, open-addressing dictionary, touched-list reset |
| `fusion.hpp` | RRF | linear id-match over stack array (K ≤ 64) |
| `prompt.hpp` | context assembly | fixed buffer, reserved question space, greedy token-budget packing |
| `epoch.hpp` | snapshot reclamation | per-reader padded epoch slots, seq_cst publish/scan |
| `text_clean.hpp` | page text cleanup | ligatures, dehyphenation, running header/footer removal, paragraph reflow |
| `pdf.hpp` | PDF ingestion (optional, Poppler) | page-aware extraction, `ingest_pdf()` with source + page citations |
| `io.hpp` | persistence | streaming checksum, bounds-checked reads, fsync + atomic rename |
| `engine.hpp` | pipeline | per-thread `QueryContext` holds all scratch; atomic snapshot swap |

## Build, test, benchmark

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release   # -DRAG_ARCH=x86-64-v3 for portable AVX2
cmake --build build -j
./build/rag_tests          # GoogleTest suite
./build/rag_bench          # Google Benchmark
./build/rag_latency 20000  # p50/p99/p99.9 per dense mode
./build/rag_demo
```

Sanitizer builds: `-DRAG_SANITIZE=ON` (ASan+UBSan), `-DRAG_TSAN=ON` (ThreadSanitizer).

Dependencies: the engine and the agent need only the C++20 standard library.
Optional, each picked up automatically when installed: GoogleTest (tests),
Google Benchmark, poppler-cpp (PDFs), and NLopt + Eigen for hyperparameter
search and data analysis (`./build/tune_agent`, `rag_optlib_tests`). On
Ubuntu 24.04:

```bash
sudo apt-get install -y libgtest-dev libbenchmark-dev libpoppler-cpp-dev libnlopt-cxx-dev libeigen3-dev
```

The full list with versions is in [docs/ai/SPECS.md](docs/ai/SPECS.md).

`-DRAG_HUGEPAGES=OFF` turns off transparent huge pages for the large arrays.
On the test VM, huge pages made scans about 10–25% faster. Faulting in fresh
huge pages took about 11 times longer, though, because the kernel compacts
memory synchronously. That cost is paid only by the writer as it fills
storage, never by queries.

## Verified (Xeon @ 2.8 GHz, 4 vCPU, AVX-512 VNNI, GCC 13.3 `-O3 -march=native`)

* All tests pass on GCC 13 Release, GCC 13 ASan+UBSan, and Clang 18; the
  two concurrency tests are clean under ThreadSanitizer.
* Zero `operator new` calls across 200 queries in every dense mode (checked by
  a global allocation counter in the test binary).

See `bench_results.txt` and `latency_results.txt` for the raw output behind
these tables. On this shared VM, scans and full queries vary by about ±15%
between runs. Single-kernel timings in the nanosecond range vary more, up to
about 2x for the AVX2 fp32 dot product, so compare them only within one run.

| Benchmark (D = 384) | Result |
|---|---|
| fp32 dot: scalar / AVX2 / AVX-512 | 424 / 31.6 / 16.1 ns |
| int8 dot: scalar / AVX2 / AVX-512 VNNI | 54.6 / 12.6 / 9.3 ns |
| Flat fp32 scan, 100k rows (154 MB) | 15.1 ms (10.2 GB/s, DRAM-bound) |
| int8 scan + fp32 rerank of 128, 100k rows | 3.59 ms (4.2x faster) |
| IVF-256, 100k rows, nprobe=16: fp32 lists | 1270 µs |
| IVF-256, 100k rows, nprobe=16: int8 lists + fp32 rerank of 128 | 383 µs (3.3x faster) |
| BM25 query, 60k chunks / 5.07M postings | 9.28 µs |
| tokenize + embed query | 0.72 µs |
| fp32 scan of 100k rows, batch of 8 queries | 396 queries/s vs 56 at batch 1 (7.1x) |
| int8 scan + rerank of 100k rows, batch of 8 | 546 queries/s vs 215 at batch 1 (2.5x) |

End-to-end query (60k chunks, k = 8, pinned core, 5000 queries):

| mode | p50 | p99 | p99.9 | QPS/core |
|---|---|---|---|---|
| exact | 10.1 ms | 15.2 ms | 20.2 ms | 94 |
| int8 + rerank | 2.7 ms | 4.2 ms | 6.5 ms | 380 |
| ivf (nprobe 16) | 0.62 ms | 1.08 ms | 1.81 ms | 1575 |
| ivf-sq8 (nprobe 16, rerank 128) | 0.28 ms | 0.43 ms | 0.57 ms | 3630 |

Caveats: the IVF benchmarks' recall@10 = 1.0 reflects well-separated
synthetic clusters. Real sentence embeddings need a larger nprobe for the same
recall, so measure on your own data. `HashingEmbedder` is a lexical-ish
stand-in. For semantic recall, plug a neural encoder into the `Embedder`
concept.

## PDFs

```bash
./build/rag_pdf index research.ragidx ~/papers/ handbook.pdf   # folders are searched recursively
./build/rag_pdf index project.ragidx .                          # text and source files work too (.md .txt .hpp .py ...)
./build/rag_pdf ask research.ragidx "how does the kill switch work?" -k 3
./build/rag_pdf ask research.ragidx "maker rebates" --prompt     # also print the LLM prompt
```

```text
Top 3 passages for: how does the kill switch work?

[1] handbook/02_risk.pdf, p. 2   (semantic #1, keyword #1)
    The kill switch cancels all resting orders and blocks new ones within one hundred ...
```

In code:

```cpp
#include "rag/pdf.hpp"                          // link the rag_pdf_support CMake target
auto r = rag::pdf::ingest_pdf(engine, "report.pdf", /*doc_id=*/7);
// r.pages, r.chunks, r.empty_pages; prompts now cite "[n] (report.pdf, p. 12)"
```

* **Extraction:** Poppler (`libpoppler-cpp-dev`), in reading order, so
  multi-column pages come out one column after another. CMake builds the PDF
  pieces only when Poppler is found; nothing else in the library depends on it.
* **Cleanup** (`rag/text_clean.hpp`, dependency-free):
  * typographic ligatures (`ﬁ` → `fi`), so "ﬁnance" matches "finance";
  * soft hyphens, zero-width characters, BOMs and Unicode spaces;
  * words hyphenated across line ends;
  * hard-wrapped lines rejoined into paragraphs;
  * running headers, footers and page numbers: lines that recur at the top or
    bottom of at least half the pages, and of both pages in a 2-page document.
    Short lines may differ only in their digits ("Page 3 of 10"); long lines
    must repeat exactly, so similar body sentences are kept.
* **Citations:** every chunk records its source and the pages it spans. A
  chunk crossing a page break cites `pp. 4-5`. Sources and pages are saved
  in the index file (format v2).
* **Scanned PDFs:** pages without a text layer are counted and reported. They
  need OCR first; this tool doesn't do OCR.
* **Encrypted PDFs:** rejected with a clear error unless they open without a
  password.
* **Speed:** a 400-page PDF indexes in 0.42 s, about the same as `pdftotext`
  alone. Loading the index and answering a question takes 24 ms.

## Hand-off pack for AI assistants

[`llms.txt`](llms.txt) and [`docs/ai/`](docs/ai/README.md) describe the project
for another language model:
* a project map;
* a compiled API cheat sheet;
* engine internals, an experiment log and a decision log;
* conventions and status.

`python3 tools/make_context_pack.py [--with-source]` bundles it all into one
file: about 17k tokens for the docs alone, about 120k with every source file.

## A self-learning agent

`include/rag/agent/` is a complete self-learning agent that teaches itself to
execute a trade, with no model of the market, from rewards alone:

```bash
./build/agent_demo
```

* **Result:** after 40,000 training episodes, tile-coded Q-learning captures
  67% of the gain over TWAP that the mathematically optimal policy (solved
  by dynamic programming) achieves on unseen scenarios.
* **Adaptation:** when the market's behaviour flips, it detects the change
  within 44 episodes and relearns. An agent without drift detection stays
  stuck doing the wrong thing. Regimes seen before are recalled from a
  policy bank within 20–131 episodes.
* **Speed:** learned decisions take 64 ns; a planned (model-based) decision
  371 ns; a full re-plan 3.4 ms; a model check (100 paired episodes) 3.6 ms.

| Learner | Share of the optimal gain | Notes |
|---|---|---|
| `TileQ` (3-step Q-learning + replay) | 69–70% at 40k episodes, 75–77% at 200k | the default |
| `ActorCritic` (softmax policy + TD critic) | 80–86% at 40k, 86–89% at 200k | strong model-free learner |
| `NeuralQ` (from-scratch MLP, DQN-style) | 84–89% at 40k; 87–94% deployed at 20k with Double DQN | no ML library; 1.7 µs per decision |
| `ModelBasedExecutionAgent` (regression + DP planning) | 99–100% after 20–50 episodes | needs a model class that fits; 22–55% with a missing term, which `check_model()` flags |
| Model plan → `distill` → fine-tune under a guard | 100% if the model is right, 78–82% if not | never deploys worse than the plan |

On a second task, market making, the same unchanged agent learns the
optimal inventory skew (actor-critic: 99–100% of the optimal gain after
5,000 episodes; `market_making.hpp`).

Production safeguards: champion/challenger deployment, two-stage drift
detection, and an optional baseline guard that never deploys a policy that
hasn't beaten TWAP by 2 standard errors. Every experiment, including the
negative results, is logged in
[docs/ai/EXPERIMENTS.md](docs/ai/EXPERIMENTS.md).

**[docs/agent.md](docs/agent.md)** walks through the architecture:
* environment, encoder, learners (TD learning, actor-critic, episodic
  memory, model-based planning), replay, exploration, risk aversion;
* held-out evaluation, champion/challenger, two-stage drift detection, the
  policy bank and the baseline guard;
* what each design choice measurably did;
* how the same architecture maps onto LLM agents.

## Batched queries

```cpp
std::vector<Engine::QueryContext> ctxs;  // one per in-flight query
for (int i = 0; i < 8; ++i) ctxs.push_back(engine.make_context());
std::array<Engine::QueryResult, 8> out;
engine.query_batch(queries, /*k=*/8, ctxs, out);
```

In `Exact` and `Int8Rerank` modes, each group of up to 8 queries shares
**one** pass over the vector matrix. Those scans are limited by DRAM
bandwidth, so each row is loaded once and then scored against all 8 queries
from L1. The fp32 kernel runs a register-blocked tile of 1 row by 4 queries,
reusing `dot4`.

Batch results are **bit-identical** to calling `query()` on each query,
because the batch and single-query kernels add up each row's products in the
same order (`simd::dot1` matches each `dot4` lane). IVF modes probe different
lists for each query, so they run one query at a time inside a batch.

End-to-end, 60k chunks, one core (`Pipeline/query*` in `bench_results.txt`):

| mode | one at a time | batches of 8 | gain |
|---|---|---|---|
| exact | 84 queries/s | 460 queries/s | 5.5x |
| int8 + rerank | 428 queries/s | 890 queries/s | 2.1x |

Batching trades latency for throughput: each query in a batch waits for the
whole batch (17.4 ms per batch of 8 in exact mode, against 11.9 ms for one
query). Use it when many queries arrive together, for offline evaluation, or
when throughput matters more than latency.

## Persistence

```cpp
engine.save("corpus.ragidx");                                    // writer thread; queries keep running
auto engine2 = RagEngine<HashingEmbedder<384>>::load("corpus.ragidx", cfg);
```

* **What's in the file:**
  * chunk text and metadata;
  * fp32 vectors;
  * BM25 builder state, so ingestion can continue after load;
  * the committed snapshot (frozen BM25 plus IVF), with the parameters it
    was built with.
* **Index reuse:** `load()` reuses the saved snapshot when the new config
  asks for the same BM25 and IVF parameters and the snapshot covers every
  chunk. Otherwise it rebuilds, and the results are identical either way.
  Pass a `LoadReport*` to see which path it took.
* **Safety:**
  * The trailing checksum catches corruption.
  * Every offset and ID is checked against the structure it indexes, so a
    file with a valid checksum but forged contents still can't cause an
    out-of-bounds read.
  * Array lengths are checked against the bytes left in the file before
    anything is allocated.
  * The file records the embedder's identity (`E::kName` and the dimension),
    so loading vectors made by a different model fails instead of returning
    wrong results.
* **Crash-safety:** a save writes `<path>.tmp`, fsyncs it, renames it over
  `<path>`, and fsyncs the directory. After a crash, `<path>` holds either the
  old index or the new one, never a partial file.

Measured on the 60k-chunk benchmark corpus (file about 276 MB with int8 IVF lists):

| | exact | ivf-sq8 |
|---|---|---|
| build from text (tokenize, embed, index) | 2.9 s | 5.6 s |
| save | 1.6 s | 1.9 s |
| load (indexes reused) | 0.65 s | 0.97 s |

With the built-in hashing embedder, building from text is cheap. With a
neural encoder, embedding dominates the build, so loading instead can save
minutes or hours.

## Thread-safety contract

* **Writer side:** `ingest()` and `commit()` run on one thread at a time.
* **Readers:** `query()` runs on any number of threads, concurrently with both
  `ingest()` and `commit()`. Each thread owns a `QueryContext`, up to 128 live
  at once.
* **How `commit()` stays safe:**
  1. It builds a new immutable `Snapshot` (BM25 plus IVF) while queries keep
     running.
  2. It publishes the snapshot with one atomic exchange.
  3. It retires the old snapshot through epoch-based reclamation
     (`epoch.hpp`). Each query pins a snapshot by writing to its own
     cache-line slot, so pinning costs no read-modify-write and no shared-line
     write.
* **Consistent reads:** `engine.pin(ctx)` holds one snapshot across several
  reads. Each `QueryResult` reports the `snapshot_version` it was served from.
* **Lifetime:** destroy every `QueryContext` before the engine.

## Testing

| Layer | What it checks |
|---|---|
| `tests/rag_tests.cpp` | Unit and integration tests per component. Includes SIMD kernels against scalar references (bit-exact for int8) and recall floors for every approximate mode. A global `operator new` counter proves the query paths make zero heap allocations. |
| `tests/reference_tests.cpp` | Differential tests against a deliberately naive reference written from each stage's spec (`std::string`, `std::map`, `double`). Covers tokenizer, chunker, stopwords, embedder, BM25, exact dense ranking and RRF on randomized corpora and configs. |
| `tests/forgery_tests.cpp` | Every structural check in the index loader, using files with a **valid** checksum and one forged field each. Each case requires its check's specific error message. |
| `tests/fuzz/` | libFuzzer targets (`-DRAG_FUZZ=ON`, Clang). `fuzz_load` loads arbitrary payloads with a fixed-up checksum, then queries, ingests, commits and reloads them. `fuzz_text` checks tokenizer, chunker, engine and prompt invariants on arbitrary bytes. |
| `tests/stress.cpp` | `rag_stress <seconds>`: one writer ingests, commits, saves and reloads while threads run single queries, batches and long pins. Committed documents must rank first for their unique token, batch and single results must agree on the same snapshot, pinned snapshots must never change, and everything retired must be reclaimed. |

What this found:
* **Integer overflow in `load()`:** the chunk-text bounds check could wrap
  around and let a forged file point a chunk outside its buffer. Found by
  coverage review and the forgery tests.
* **Arena UB:** a fresh arena's first allocation computed `nullptr + n`.
  Found by the fuzzer; GCC's UBSan misses this, so CI now also runs Clang's.
* **Irrelevant dense hits:** chunks with zero or negative similarity (for
  example, stopword-only chunks) filled the dense list and reached prompts.
  Hence `dense_min_score`.

Checking the checks:
* **Mutation testing:** eight planted bugs (BM25 constants, the stopword
  list, bigram weight, whitespace trimming, the RRF rank offset, a dropped
  scan row, an inclusive score floor) were all caught.
* **Reclamation mutant:** freeing snapshots immediately instead of through
  epoch reclamation is caught by `rag_stress` under ASan within seconds.

## Continuous integration

`.github/workflows/ci.yml` runs on every push and pull request. All test
jobs build with `-Werror`:

| Job | Covers |
|---|---|
| gcc-native | the runner's best ISA (AVX-512 when present), plus a demo and benchmark smoke run |
| gcc-avx2 | the `-march=x86-64-v3` AVX2 + FMA kernels |
| gcc-scalar | the `-march=x86-64` scalar fallback |
| clang-native | Clang 18 |
| gcc-asan-ubsan | AddressSanitizer + UndefinedBehaviorSanitizer |
| clang-asan-ubsan | the same with Clang, whose UBSan also checks arithmetic on null pointers |
| gcc-tsan | ThreadSanitizer: concurrency tests repeated 25 times, then `rag_stress` for 10 s per mode |
| fuzz-smoke | both fuzzers for 60 s each, seeded by `make_seeds`; crashing inputs uploaded as artifacts |
