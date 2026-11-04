// stress.cpp — long-running concurrency stress test with exact invariants.
//
//   rag_stress [seconds=5] [dense_mode 0..3 = all]
//
// One writer thread continuously ingests documents, commits, and every few
// commits saves the index and reloads it into a second engine. At the same
// time:
//   * query threads issue "docN" queries. Every document contains a unique
//     token docN, so once document N is committed, BM25 must rank a chunk of
//     document N first. That is an exact check, not a heuristic, and it is
//     made against the snapshot version the query reports.
//   * a batch thread runs query_batch() and, whenever the batch and a
//     follow-up single query saw the same snapshot, requires identical
//     lexical results.
//   * a pin thread holds snapshots for random durations and requires a
//     pinned snapshot never to change underneath it.
// Throughout: hit ids are in range, snapshot versions never go backwards
// per thread, and at the end every retired snapshot has been reclaimed.
// Exits non-zero on the first violated invariant. Run it under -DRAG_TSAN=ON
// to also check for data races.
#include <array>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include "pid.hpp"

#include "rag.hpp"

using namespace rag;
using Engine = RagEngine<HashingEmbedder<128>, 16>;
using Clock = std::chrono::steady_clock;

namespace {

std::atomic<bool> g_failed{false};
std::atomic<bool> g_stop{false};

#define STRESS_CHECK(cond, ...)                                              \
  do {                                                                       \
    if (!(cond)) {                                                           \
      if (!g_failed.exchange(true)) {                                        \
        std::fprintf(stderr, "INVARIANT FAILED: %s (%s:%d): ", #cond, __FILE__, __LINE__); \
        std::fprintf(stderr, __VA_ARGS__);                                   \
        std::fprintf(stderr, "\n");                                          \
      }                                                                      \
      g_stop = true;                                                         \
      return;                                                                \
    }                                                                        \
  } while (0)

std::string doc_text(std::uint32_t n, std::mt19937& rng) {
  static const char* words[] = {"order", "book", "spread", "venue", "latency", "queue", "tick", "vwap",
                                "imbalance", "auction", "the", "of", "and", "quote", "fill", "cancel"};
  std::string s = "doc" + std::to_string(n);
  for (int i = 0; i < 20; ++i) {
    s += ' ';
    s += words[rng() % std::size(words)];
  }
  return s;
}

struct Counters {
  std::atomic<std::uint64_t> queries{0}, batches{0}, batch_checks{0}, pins{0}, exact_checks{0};
  std::atomic<std::uint64_t> commits{0}, saves{0}, docs{0};
};

}  // namespace

int main(int argc, char** argv) {
  const double seconds = argc > 1 ? std::atof(argv[1]) : 5.0;
  const int only_mode = argc > 2 ? std::atoi(argv[2]) : -1;
  const auto tmp = std::filesystem::temp_directory_path() / ("rag-stress-" + std::to_string(test_pid()) + ".ragidx");
  int rc = 0;

  for (int m = 0; m < 4 && !g_failed; ++m) {
    if (only_mode >= 0 && m != only_mode) continue;
    const auto mode = static_cast<DenseMode>(m);
    EngineConfig cfg;
    cfg.max_chunks = 1u << 15;
    cfg.chunker = {64, 8};  // one chunk per document
    cfg.dense_mode = mode;
    cfg.retrieve_depth = 16;
    cfg.ivf.nlist = 8;
    cfg.ivf.train_iters = 3;
    cfg.nprobe = 3;
    cfg.prompt_capacity = 4096;
    Engine e(cfg);

    // committed_docs: documents [0, committed_docs) are in the published snapshot
    // with version >= committed_version.
    std::atomic<std::uint32_t> committed_docs{0};
    std::atomic<std::uint64_t> committed_version{0};
    Counters c;
    g_stop = false;

    std::thread writer([&] {
      std::mt19937 rng(1);
      std::uint32_t n = 0;
      const auto end = Clock::now() + std::chrono::duration<double>(seconds);
      while (!g_stop && Clock::now() < end && e.size() + 4 < cfg.max_chunks) {
        for (int i = 0; i < 40; ++i, ++n) e.ingest(doc_text(n, rng), n);
        c.docs = n;
        e.commit();
        ++c.commits;
        // Publish after commit: any snapshot with version >= this one covers [0, n).
        committed_version.store(c.commits.load(), std::memory_order_release);
        committed_docs.store(n, std::memory_order_release);
        if (c.commits % 10 == 0) {  // save + reload while queries keep running
          e.save(tmp);
          const auto copy = Engine::load(tmp, cfg);
          STRESS_CHECK(copy->size() == e.size(), "reloaded %u chunks, expected %u", copy->size(), e.size());
          ++c.saves;
        }
      }
      g_stop = true;
    });

    auto query_thread = [&](unsigned seed) {
      auto ctx = e.make_context();
      std::mt19937 rng(seed);
      std::uint64_t last_version = 0;
      while (!g_stop) {
        const std::uint32_t done = committed_docs.load(std::memory_order_acquire);
        const std::uint64_t at_least = committed_version.load(std::memory_order_acquire);
        if (done == 0) { std::this_thread::yield(); continue; }
        const std::uint32_t target = static_cast<std::uint32_t>(rng() % done);
        const std::string q = "doc" + std::to_string(target);
        const auto r = e.query(q, 8, ctx);
        const std::uint32_t n = e.size();
        STRESS_CHECK(r.snapshot_version >= last_version, "version went backwards %llu -> %llu",
                     (unsigned long long)last_version, (unsigned long long)r.snapshot_version);
        last_version = r.snapshot_version;
        for (const Hit& h : r.hits) STRESS_CHECK(h.id < n, "hit id %u >= size %u", h.id, n);
        for (const Hit& h : r.lexical) STRESS_CHECK(h.id < n, "lexical id %u >= size %u", h.id, n);
        // Exact check: document `target` was committed in a snapshot no newer
        // than the one this query used, so BM25 must rank it first.
        if (r.snapshot_version >= at_least) {
          STRESS_CHECK(!r.lexical.empty(), "no lexical hit for committed %s (v%llu)", q.c_str(),
                       (unsigned long long)r.snapshot_version);
          STRESS_CHECK(e.chunk(r.lexical[0].id).doc_id == target, "%s ranked doc %u first", q.c_str(),
                       e.chunk(r.lexical[0].id).doc_id);
          ++c.exact_checks;
        }
        ++c.queries;
      }
    };

    std::thread batcher([&] {
      std::vector<Engine::QueryContext> ctxs;
      for (int i = 0; i < 8; ++i) ctxs.push_back(e.make_context());
      auto single = e.make_context();
      std::array<Engine::QueryResult, 8> out;
      std::array<std::string, 8> qs;
      std::array<std::string_view, 8> views;
      std::mt19937 rng(99);
      while (!g_stop) {
        const std::uint32_t done = committed_docs.load(std::memory_order_acquire);
        if (done == 0) { std::this_thread::yield(); continue; }
        for (int i = 0; i < 8; ++i) {
          qs[i] = "doc" + std::to_string(rng() % done) + " vwap";
          views[i] = qs[i];
        }
        e.query_batch(views, 6, ctxs, out);
        const std::uint32_t n = e.size();
        for (int i = 0; i < 8; ++i) {
          for (const Hit& h : out[i].hits) STRESS_CHECK(h.id < n, "batch hit id %u >= %u", h.id, n);
          const auto s = e.query(views[i], 6, single);
          if (s.snapshot_version == out[i].snapshot_version) {  // same snapshot => same lexical answer
            STRESS_CHECK(s.lexical.size() == out[i].lexical.size(), "batch/single lexical size differ");
            for (std::size_t k = 0; k < s.lexical.size(); ++k)
              STRESS_CHECK(s.lexical[k] == out[i].lexical[k], "batch/single lexical differ at %zu", k);
            ++c.batch_checks;
          }
        }
        ++c.batches;
      }
    });

    std::thread pinner([&] {
      auto ctx = e.make_context();
      std::mt19937 rng(7);
      while (!g_stop) {
        const auto pin = e.pin(ctx);
        const auto version = pin->version;
        const auto docs = pin->bm25.num_docs();
        const auto covered = pin->covered;
        std::this_thread::sleep_for(std::chrono::microseconds(rng() % 2000));
        // Commits happen meanwhile; the pinned snapshot must be untouched.
        STRESS_CHECK(pin->version == version && pin->bm25.num_docs() == docs && pin->covered == covered,
                     "pinned snapshot changed under the pin");
        ++c.pins;
      }
    });

    std::vector<std::thread> readers;
    for (unsigned t = 0; t < 2; ++t) readers.emplace_back(query_thread, 100 + t);
    writer.join();
    for (auto& t : readers) t.join();
    batcher.join();
    pinner.join();

    const std::size_t left = e.reclaim();
    if (!g_failed && (left != 0 || e.live_readers() != 0)) {
      std::fprintf(stderr, "INVARIANT FAILED: %zu snapshots unreclaimed, %zu readers registered\n", left, e.live_readers());
      g_failed = true;
    }
    std::printf("%-11s docs=%llu commits=%llu saves=%llu queries=%llu exact_checks=%llu batches=%llu "
                "batch_checks=%llu pins=%llu\n",
                m == 0 ? "exact" : m == 1 ? "int8+rerank" : m == 2 ? "ivf" : "ivf-sq8", (unsigned long long)c.docs.load(),
                (unsigned long long)c.commits.load(), (unsigned long long)c.saves.load(), (unsigned long long)c.queries.load(),
                (unsigned long long)c.exact_checks.load(), (unsigned long long)c.batches.load(),
                (unsigned long long)c.batch_checks.load(), (unsigned long long)c.pins.load());
    if (!g_failed && (c.exact_checks == 0 || c.batch_checks == 0 || c.pins == 0 || c.saves == 0)) {
      std::fprintf(stderr, "stress run too short to exercise every check\n");
      rc = 2;
    }
  }
  std::error_code ec;
  std::filesystem::remove(tmp, ec);
  if (g_failed) return 1;
  std::printf("%s\n", rc ? "INCOMPLETE" : "OK");
  return rc;
}
