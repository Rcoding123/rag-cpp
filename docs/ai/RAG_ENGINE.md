# RAG engine internals

The engine is `rag::RagEngine<Embedder E, std::size_t MaxK = 64>` in
`include/rag/engine.hpp`. Read this before changing anything in `include/rag/`.

## Data flow

```
INGEST (one writer thread)
  text ─► Arena copy ─► tokenize (Token{offset,len,hash}) ─► chunk_tokens (window/overlap)
       ─► per chunk: Embedder::embed ─► VectorStore::append (fp32 + int8 + scale)
                     Bm25Builder::add (term hashes, stopwords removed)
                     ChunkMeta {text, len, doc_id, ordinal, page_first, page_last, source}
  commit() ─► new Snapshot{Bm25Index frozen, IvfIndex (if an IVF mode)} ─► atomic exchange ─► EBR retire old

QUERY (any number of threads, each with its own QueryContext; zero heap allocations)
  prepare:  tokenize ─► terms (no stopwords) ─► embed ─► has_signal = |qvec|² > 0
  pin snapshot (EpochDomain::Guard), live = store.size() (acquire)
  dense:    Exact      flat fp32 scan of [0, live)
            Int8Rerank int8 scan -> shortlist (rerank_candidates) -> fp32 rerank
            Ivf        IVF over [0, covered) + exact scan of the delta [covered, live)
            IvfInt8    IVF-SQ8 shortlist -> fp32 rerank + exact delta scan
            (admission floor: score > dense_min_score, default 0)
  lexical:  Bm25Index::search over the snapshot (impact sums into a dense accumulator)
  fuse:     RRF(k0 = 60) over the dense and lexical lists
  prompt:   build_prompt into ctx.prompt_buf with citations "[n] (source, p. X)"
```

## Key data structures

| Structure | Layout | Why |
|---|---|---|
| `VectorStore<D>` | separate aligned arrays: fp32 rows, int8 rows, per-row scale; `size_` on its own cache line | scans stream contiguous memory; readers never false-share with the writer |
| `TopK<Cap>` | `std::array` heap + cached threshold + `sorted_` flag | reject path is one compare; NaN rejected; deterministic ties (score desc, id asc) |
| `Bm25Index` | open-addressing term table (power of two, load ≤ 0.5) + SoA postings (doc u32, weight f32) | query = gather-add of precomputed impacts |
| `IvfIndex<D>` | centroids + CSR `offsets/ids` + contiguous reordered rows (fp32 and/or int8) | probing a list is a sequential read |
| `Arena` | list of blocks that never move | `string_view`s into it stay valid for readers |
| `EpochDomain` | 128 reader slots of 64 B each (active epoch, in_use, depth) | per-reader pinning with no shared writes |

## Concurrency contract (enforced by TSan tests and `rag_stress`)

* `ingest()` and `commit()` run on **one writer thread at a time**.
* `query()` / `query_batch()` run on any number of threads, **concurrently
  with ingest and commit**. Each thread owns a `QueryContext`; at most 128
  can be live.
* **Publication:** the writer fills a row and its `ChunkMeta`, then does
  `size_.store(n+1, release)`. Readers do `size()` (acquire) and see only
  complete rows.
* **Snapshots:** `commit()` builds a new `Snapshot` off to the side, publishes
  it with `snap_.exchange(seq_cst)`, retires the old one with the current
  epoch, and frees it once no reader slot holds an older epoch
  (`EpochDomain::quiescent_since`).
* **Lifetime:** destroy every `QueryContext` before the engine.

## Invariants that tests rely on (don't break these)

1. **Bit-exact batching:** `query_batch` returns bit-identical results to
   `query`. This holds because `simd::dot1` matches one lane of `simd::dot4`
   exactly, and the exact scan uses `dot1` for tail rows. Use `dot1`, not
   `dot`, wherever a score must match a `dot4` lane.
2. **Zero allocation:** the query path makes no heap allocations. A global
   `operator new` counter in `rag_tests.cpp` enforces it.
3. **Scratch reset:** `Bm25Scratch::acc` is all zeros between queries.
4. **Ordering:** `TopK` orders by score descending, then id ascending.
5. **Validate before mutating:** `ingest()` checks `IngestOptions` before
   touching anything. Invalid page offsets throw and leave the engine
   unchanged.
6. **Safe loading:** loading never trusts a field. Every offset, length and
   id is checked against the structure it indexes, and `tests/forgery_tests.cpp`
   covers each check with a valid-checksum forged file. Add a forgery case
   for every new field.

## Index file format (version 2, little-endian)

```
FileHeader   40 B  magic "RAGCPPIX", version=2, dim, embedder_tag, n_chunks, reserved, text_bytes
ChunkRecord  40 B × n   text_offset u64, len, doc_id, ordinal, page_first, page_last, source idx (0xFFFFFFFF = none), reserved u64
text blob    text_bytes
sources      vec<u64> offsets (n_sources + 1, from 0), vec<char> names
rows         n × D × f32
Bm25Builder  vec<Triple{term u64, doc u32, tf u32}>, vec<u32> doc_len, u32 num_docs
IndexRecord  40 B  k1, b, nlist, train_iters, max_train, storage, has_ivf, seed, covered
Bm25Index    vec<Slot{key u64, offset u32, len u32}>, u64 nterms, u32 ndocs, vec<u32> doc, vec<f32> weight
IvfIndex     u32 nlist, u32 covered, u8 flags(1 = fp32, 2 = int8), centroids, vec offsets, vec ids, rows...
checksum     u64 (8-byte-word streaming hash over everything before it)
```

* **Reuse on load:** `load()` reuses the saved snapshot if the BM25 and IVF
  parameters match the new `EngineConfig` and it covers every chunk;
  otherwise it rebuilds. `LoadReport::reused_indexes` says which happened.
* **Saving:** writes `<path>.tmp`, fsyncs it, renames it over `<path>`, then
  fsyncs the directory.
* **Format changes:** bump `io::kFormatVersion` for any layout change, and
  extend `tests/forgery_tests.cpp` (its `Layout` walker parses the file).

## Measured performance (Xeon 2.8 GHz VM, AVX-512 VNNI, D = 384, one core)

| Operation | Result |
|---|---|
| fp32 / int8 dot product (384 dims) | 16–32 ns / 7.7–9.6 ns |
| flat fp32 scan, 100k rows | 15.4 ms (~10 GB/s, DRAM-bound) |
| int8 scan + rerank, 100k rows | 2.95–3.6 ms |
| IVF-256 nprobe 16, 100k rows: fp32 lists / int8 lists | 1.1–1.3 ms / 0.32–0.38 ms |
| BM25 query, 5M postings | 7–9 µs |
| end-to-end query, 60k chunks (p50 / p99) | exact 10.1 / 15.2 ms; int8 2.7 / 4.2 ms; ivf 0.62 / 1.08 ms; ivf-sq8 0.28 / 0.43 ms |
| batch of 8 queries | exact 84 → 460 queries/s; int8 428 → 890 queries/s |
| PDF indexing | 400 pages in 0.42 s (≈ `pdftotext` alone) |

Results vary ±15% between runs on this shared VM, and up to 2× for
nanosecond-scale kernels.

## Extension points

* **Embedder:** implement the `Embedder` concept (`kDim`, `kName`, `embed()`
  noexcept, no allocation) to plug in a neural encoder. `kName` goes into
  index files; change it whenever the embedder's output changes.
* **Generator:** anything with `generate(std::string_view)` satisfies the
  concept; see `RagEngine::answer()`.
* **New dense mode:** add it to `DenseMode`, `dense_one()`, `dense_batch()`
  and `uses_ivf()`/`effective_ivf()` if it builds an index, plus the engine
  test parameter lists.
