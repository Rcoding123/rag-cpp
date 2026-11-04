# API cheat sheet

Every snippet here is compiled and run by `tests/api_examples_tests.cpp`. If
you change an API, update both files.

## 1. Index and query

```cpp
#include "rag.hpp"
using Engine = rag::RagEngine<rag::HashingEmbedder<384>>;

rag::EngineConfig cfg;
cfg.max_chunks = 100'000;                       // capacity; memory is committed lazily
cfg.chunker = {.window_tokens = 128, .overlap_tokens = 32};
cfg.dense_mode = rag::DenseMode::IvfInt8;       // Exact | Int8Rerank | Ivf | IvfInt8
Engine engine(cfg);

engine.ingest("Iceberg orders show a small visible size and refill from a hidden reserve.", /*doc_id=*/1);
engine.ingest("VWAP algorithms follow the intraday volume curve.", 2);
engine.commit();  // freeze BM25 (+ IVF) into a new snapshot; safe while queries run

auto ctx = engine.make_context();  // one per query thread; reuse it
const auto r = engine.query("how do iceberg orders work?", /*k=*/3, ctx);
const rag::ChunkView top = engine.chunk(r.hits[0].id);   // text, doc_id, ordinal, source, page_first, page_last
std::string_view prompt = r.prompt.prompt;               // ready for an LLM
```

`QueryResult` fields:
* `hits`: fused ranking of `Hit{score, id}`, best first;
* `dense`, `lexical`: each retriever's own list;
* `prompt`: a `PromptResult` with `.prompt`, `.chunks_used`,
  `.context_tokens` and `.truncated`;
* `snapshot_version`.

Every span points into `ctx` and stays valid until the next query on that
context.

`EngineConfig` fields, in declaration order (designated initializers must
follow it):

| Field | Default | Meaning |
|---|---|---|
| `max_chunks` | 65,536 | capacity |
| `chunker` | `{128, 32}` | `window_tokens`, `overlap_tokens` |
| `bm25` | `{1.2, 0.75}` | `k1`, `b` |
| `rrf` | `{60, 1, 1}` | `k0`, `w_dense`, `w_lexical` |
| `prompt` | `{1024, system}` | `max_context_tokens`, system text |
| `dense_mode` | `Exact` | dense retrieval mode |
| `hybrid` | `true` | `false` = dense only |
| `dense_min_score` | `0` | strict cosine floor |
| `retrieve_depth` | 32 | per-retriever list depth fed to RRF |
| `rerank_candidates` | 128 | shortlist size for int8 modes |
| `nprobe` | 8 | IVF lists probed per query |
| `ivf` | `{nlist 0 = auto, train_iters 10, ...}` | IVF parameters |
| `prompt_capacity` | 32 KiB | prompt buffer size |

## 2. Page-aware ingestion and citations

```cpp
const std::string text = "Page one talks about spreads.\n\nPage two talks about maker rebates.";
const std::uint32_t page_starts[] = {0, static_cast<std::uint32_t>(text.find("Page two"))};
engine.ingest(text, 7, rag::IngestOptions{.source = "fees.pdf", .page_starts = page_starts});
// prompts cite "[n] (fees.pdf, p. 2)" or "(fees.pdf, pp. 1-2)" for chunks spanning a page break
```

For real PDFs (Poppler; link `rag_pdf_support`):

```cpp
#include "rag/pdf.hpp"
auto res = rag::pdf::ingest_pdf(engine, "report.pdf", /*doc_id=*/7);  // res.pages, res.chunks, res.empty_pages
```

CLI: `rag_pdf index out.ragidx docs/ a.pdf`, then
`rag_pdf ask out.ragidx "question" -k 5 --prompt`. It also indexes text and
source files verbatim (`.md .txt .hpp .cpp .py ...`, cited by chunk), so
`rag_pdf index project.ragidx .` makes this repository searchable.

## 3. Save, inspect, load

```cpp
engine.save(path);                                   // crash-safe: tmp file, fsync, rename
const rag::IndexInfo info = Engine::inspect(path);   // header only: version, dim, chunks, text_bytes
Engine::LoadReport report;
auto loaded = Engine::load(path, cfg, {}, &report);  // unique_ptr; throws io::FormatError / invalid_argument
// report.reused_indexes: true if the saved BM25/IVF matched cfg and were reused
```

## 4. Batched queries

```cpp
std::vector<std::string_view> queries = {"alpha", "zeta", "gamma delta"};
std::vector<Engine::QueryContext> ctxs;              // one context per query in flight
for (std::size_t i = 0; i < queries.size(); ++i) ctxs.push_back(engine.make_context());
std::vector<Engine::QueryResult> out(queries.size());
engine.query_batch(queries, /*k=*/2, ctxs, out);     // bit-identical to query(); one memory pass per 8
```

## 5. Text cleanup (no Poppler needed)

```cpp
#include "rag/text_clean.hpp"
const auto cleaned = rag::text::clean_pages(pages);  // ligatures, headers/footers, hyphenation, reflow
const auto joined = rag::text::join_pages(cleaned);  // joined.text + joined.page_starts
```

## 6. Train an agent

```cpp
#include "rag/agent/execution_dp.hpp"
#include "rag/agent/trainer.hpp"
using namespace rag::agent;

ExecutionEnv env{ExecutionParams{}};
AgentConfig<3> cfg;                 // 3 = ExecutionEnv::kStateDims
cfg.eps_decay_episodes = 2000;
Agent<ExecutionEnv> agent(env, cfg);
Trainer<ExecutionEnv> trainer(agent, env, {.eval_every = 500, .eval_episodes = 200});
const TrainReport report = trainer.run(4000);   // report.evals, .drift_alarms, .drift_rejected, .returns
const Action a = trainer.deployed_act(ExecutionObs{.t = 5, .inventory = 60, .signal = -0.1});

// Measure against the exact optimum on unseen scenarios, paired vs TWAP:
const ExecutionDP dp{ExecutionParams{}};
auto twap = [](const ExecutionObs&) { return ExecutionEnv::kTwapAction; };
EvalStats mine = compare(env, [&](const ExecutionObs& o) { return trainer.deployed_act(o); }, twap, 500, 1);
EvalStats best = compare(env, [&](const ExecutionObs& o) { return dp.act(o); }, twap, 500, 1);
// share of the optimal gain = mine.mean / best.mean
```

`AgentConfig` fields, in order:

| Field | Default | Meaning |
|---|---|---|
| `learner` | `TileQ` | or `Episodic` |
| `eps_start`, `eps_end`, `eps_decay_episodes` | 1.0, 0.02, 1500 | exploration schedule |
| `bins` | 0 = 8 per dimension | tile grid |
| `tilings` | 8 | number of offset grids |
| `alpha`, `alpha_min`, `alpha_decay_episodes` | 0.1, 0.003, 3000 | annealed step size |
| `gamma` | 1.0 | discount |
| `init_q` | 0 | initial Q-values |
| `double_q` | false | double Q-learning |
| `replay_capacity`, `replay_per_step` | 100,000, 4 | experience replay |
| `memory` | | `EpisodicParams` |
| `kernel_width` | 0 = 10% of each range | episodic similarity scale |
| `seed` | 1 | RNG seed |

`TrainConfig` fields, in order:

| Field | Default | Meaning |
|---|---|---|
| `eval_every`, `eval_episodes`, `eval_seed` | 250, 300, 9,000,000 | held-out evaluation |
| `train_seed` | 1 | training scenario stream |
| `keep_champion` | true | deploy the best policy seen |
| `detect_drift` | true | drift detection on/off |
| `drift` | | `DriftParams`: lag 50, allowance 0.3, threshold 16, baseline_rate 0.001, warmup 1500 |
| `drift_after_episodes` | 0 = after the ε schedule | when drift checks start |
| `reexplore_epsilon`, `reexplore_episodes` | 0.5, 5000 | response to drift |
| `forget_on_drift` | true | clear replay on drift |

A new environment only needs to satisfy the `Environment` concept: an
`Observation` type, `kNumActions`, `kStateDims`, `kLo`/`kHi`, and
`reset(rng)`, `step(action, rng)` and `features(obs)`.
