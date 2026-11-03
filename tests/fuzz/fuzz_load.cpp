// fuzz_load — Engine::load on arbitrary files.
//
// The fuzzer's bytes are the file payload; a correct checksum is appended,
// so mutations get past the integrity check and reach the structural
// validation behind it. That is the attack surface that matters: anyone who
// can hand the process a file can also compute a checksum.
//
// Invariants: load() either throws or returns an engine whose every chunk and
// index entry is in bounds; queries on a loaded engine must not crash (ASan
// turns any out-of-bounds read into a crash).
#include <cstdint>
#include <fstream>
#include <vector>

#include "fuzz_common.hpp"

using namespace rag;
using namespace rag::fuzz;

extern "C" int LLVMFuzzerTestOneInput(const std::uint8_t* data, std::size_t size) {
  static const auto path = scratch_file("load");
  {
    io::Checksum c;
    c.update(data, size);
    const std::uint64_t h = c.value();
    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    f.write(reinterpret_cast<const char*>(data), static_cast<std::streamsize>(size));
    f.write(reinterpret_cast<const char*>(&h), sizeof h);
  }
  // Vary the dense mode with the input length (byte 0 is always the magic
  // 'R' in valid files), so every reuse/rebuild path of load() is reached.
  const DenseMode modes[] = {DenseMode::Exact, DenseMode::Int8Rerank, DenseMode::Ivf, DenseMode::IvfInt8};
  const auto cfg = small_config(modes[size % 4]);
  std::unique_ptr<Engine> e;
  try {
    e = Engine::load(path, cfg);
  } catch (const std::exception&) {
    return 0;  // rejected: fine
  }
  const std::uint32_t n = e->size();
  FUZZ_CHECK(n <= cfg.max_chunks);
  FUZZ_CHECK(e->committed() == n);
  for (std::uint32_t i = 0; i < n; ++i) {
    const auto c = e->chunk(i);
    volatile char sink = 0;
    for (char ch : c.text) sink = static_cast<char>(sink ^ ch);  // touch every byte: ASan checks bounds
    (void)sink;
  }
  auto ctx = e->make_context();
  for (const char* q : {"alpha beta", "t0w1 t1w2 f3", "the", "order book imbalance"}) {
    const auto r = e->query(q, 8, ctx);
    for (const Hit& h : r.hits) FUZZ_CHECK(h.id < n);
    for (const Hit& h : r.lexical) FUZZ_CHECK(h.id < n);
    FUZZ_CHECK(r.prompt.prompt.size() <= cfg.prompt_capacity);
  }
  // A loaded engine must keep working: ingest, commit, save, reload.
  e->ingest("fresh document after load", 7);
  e->commit();
  e->save(path);
  const auto again = Engine::load(path, cfg);
  FUZZ_CHECK(again->size() == e->size());
  return 0;
}
