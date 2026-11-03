// forgery_tests.cpp — every structural check in the index loader, exercised
// with files that carry a *valid* checksum. Corruption is caught by the
// checksum; these tests cover the other threat: a well-formed but hostile or
// buggy file. Each case edits one field of a real saved index, recomputes the
// checksum, and requires load() to fail with that check's specific message —
// so a test can't pass by tripping some other check by accident.
#include <gtest/gtest.h>

#include <functional>
#include <limits>
#include <string>
#include <vector>

#include "rag.hpp"
#include "synthetic.hpp"
#include "test_util.hpp"

using namespace rag;
using namespace rag::testutil;

namespace {

constexpr std::size_t kD = 384;
using Engine = RagEngine<HashingEmbedder<kD>>;

// Byte offsets of every section in a saved index, found by walking the file
// the same way load() reads it (see RagEngine::save for the layout).
struct Layout {
  std::size_t n = 0, text_bytes = 0;
  std::size_t records = 0, text = 0, src_offsets = 0, src_names = 0, rows = 0;
  std::size_t triples_len = 0, triples = 0, ntriples = 0;
  std::size_t doclen_len = 0, builder_ndocs = 0;
  std::size_t index_record = 0;
  std::size_t bm25_table_len = 0, bm25_ndocs = 0;
  std::size_t ivf = 0;

  explicit Layout(const std::vector<char>& b) {
    n = peek<std::uint32_t>(b, 24);
    text_bytes = static_cast<std::size_t>(peek<std::uint64_t>(b, 32));
    records = 40;
    text = records + 40 * n;  // ChunkRecord: 40 bytes
    src_offsets = text + text_bytes;
    src_names = src_offsets + 8 + 8 * static_cast<std::size_t>(peek<std::uint64_t>(b, src_offsets));
    rows = src_names + 8 + static_cast<std::size_t>(peek<std::uint64_t>(b, src_names));
    triples_len = rows + n * kD * sizeof(float);
    ntriples = static_cast<std::size_t>(peek<std::uint64_t>(b, triples_len));
    triples = triples_len + 8;
    doclen_len = triples + 16 * ntriples;
    builder_ndocs = doclen_len + 8 + 4 * static_cast<std::size_t>(peek<std::uint64_t>(b, doclen_len));
    index_record = builder_ndocs + 4;
    bm25_table_len = index_record + 40;
    const auto table = static_cast<std::size_t>(peek<std::uint64_t>(b, bm25_table_len));
    bm25_ndocs = bm25_table_len + 8 + 16 * table + 8;
    const auto postings = static_cast<std::size_t>(peek<std::uint64_t>(b, bm25_ndocs + 4));
    ivf = bm25_ndocs + 4 + 8 + 4 * postings + 8 + 4 * postings;
  }
};

class Forgery : public ::testing::Test {
 protected:
  void SetUp() override {
    cfg_.max_chunks = 256;
    cfg_.chunker = {16, 4};
    Engine e(cfg_);
    const auto corpus = synth::make_text_corpus(12, 3, 40);
    // Half the documents carry a source name and page offsets, so every
    // record field is populated somewhere in the file.
    for (std::uint32_t d = 0; d < corpus.docs.size(); ++d) {
      const std::uint32_t pages[] = {0, static_cast<std::uint32_t>(corpus.docs[d].size() / 2)};
      if (d % 2) e.ingest(corpus.docs[d], d, IngestOptions{.source = d % 4 == 1 ? "a.pdf" : "b.pdf", .page_starts = pages});
      else e.ingest(corpus.docs[d], d);
    }
    e.commit();
    path_ = temp_index("forgery");
    e.save(path_);
    good_ = read_file(path_);
    ASSERT_NO_THROW((void)Engine::load(path_, cfg_));  // the baseline is valid
  }

  // Apply `edit` to a copy of the good file, fix the checksum, and return the
  // message load() fails with.
  std::string load_error(const std::function<void(std::vector<char>&, const Layout&)>& edit,
                         const EngineConfig* cfg = nullptr) {
    auto b = good_;
    const Layout L(b);
    edit(b, L);
    fix_checksum(b);
    write_file(path_, b);
    return error_of([&] { (void)Engine::load(path_, cfg ? *cfg : cfg_); });
  }

  EngineConfig cfg_;
  std::filesystem::path path_;
  std::vector<char> good_;
};

#define EXPECT_ERROR(msg, needle) EXPECT_NE((msg).find(needle), std::string::npos) << "got: " << (msg)

TEST_F(Forgery, LayoutWalkerMatchesTheFile) {
  const Layout L(good_);
  EXPECT_GT(L.n, 0u);
  EXPECT_EQ(peek<std::uint32_t>(good_, L.builder_ndocs), L.n);  // builder doc count == chunk count
  EXPECT_EQ(peek<std::uint32_t>(good_, L.index_record + 32), L.n);  // snapshot covers every chunk
  EXPECT_EQ(peek<std::uint32_t>(good_, L.bm25_ndocs), L.n);
  EXPECT_EQ(peek<std::uint32_t>(good_, L.ivf), 0u);       // Exact mode: no IVF lists
  EXPECT_EQ(L.ivf + 4 + 4 + 1 + 8, good_.size());         // ivf header, then the checksum
}

TEST_F(Forgery, Header) {
  EXPECT_ERROR(load_error([](auto& b, auto&) { poke<std::uint32_t>(b, 8, 99); }), "unsupported index format version");
  EXPECT_ERROR(load_error([](auto& b, auto&) { poke<std::uint32_t>(b, 12, 768); }), "dimension does not match");
  EXPECT_ERROR(load_error([](auto& b, auto&) { b[16] ^= 1; }), "different embedder");
  EXPECT_ERROR(load_error([](auto& b, auto&) { poke<std::uint32_t>(b, 24, 100000); }), "more chunks than cfg.max_chunks");
}

TEST_F(Forgery, ChunkTextBoundsIncludingOverflow) {
  // The overflow case: offset + len wraps around 2^64 to a small number. A
  // naive `offset + len > text_bytes` check passes it, and the chunk's text
  // pointer would then point far outside the buffer.
  EXPECT_ERROR(load_error([](auto& b, const Layout& L) {
                 poke<std::uint64_t>(b, L.records, std::numeric_limits<std::uint64_t>::max());
                 poke<std::uint32_t>(b, L.records + 8, 2);
               }),
               "chunk text out of range");
  EXPECT_ERROR(load_error([](auto& b, const Layout& L) {
                 poke<std::uint64_t>(b, L.records, L.text_bytes);  // starts at the end, len > 0
               }),
               "chunk text out of range");
  EXPECT_ERROR(load_error([](auto& b, const Layout& L) {
                 poke<std::uint32_t>(b, L.records + 8, static_cast<std::uint32_t>(L.text_bytes + 1));
               }),
               "chunk text out of range");
  // text_bytes larger than the file: records still in range of the bogus size.
  EXPECT_ERROR(load_error([](auto& b, auto&) { poke<std::uint64_t>(b, 32, std::uint64_t{1} << 40); }),
               "truncated text section");
}

TEST_F(Forgery, ChunkSourceAndPages) {
  // Record layout: text_offset u64 @0, len @8, doc_id @12, ordinal @16,
  // page_first @20, page_last @24, source @28.
  EXPECT_EQ(peek<std::uint64_t>(good_, Layout(good_).src_offsets), 3u);  // 2 sources + 1 terminator
  EXPECT_ERROR(load_error([](auto& b, const Layout& L) { poke<std::uint32_t>(b, L.records + 28, 2); }),
               "chunk source out of range");
  EXPECT_ERROR(load_error([](auto& b, const Layout& L) {
                 poke<std::uint32_t>(b, L.records + 20, 5);
                 poke<std::uint32_t>(b, L.records + 24, 4);
               }),
               "bad chunk page range");  // first > last
  EXPECT_ERROR(load_error([](auto& b, const Layout& L) { poke<std::uint32_t>(b, L.records + 20, 3); }),
               "bad chunk page range");  // first set, last unset
  EXPECT_ERROR(load_error([](auto& b, const Layout& L) { poke<std::uint64_t>(b, L.src_offsets + 8, 1); }),
               "bad source table");  // doesn't start at 0
  EXPECT_ERROR(load_error([](auto& b, const Layout& L) {
                 poke<std::uint64_t>(b, L.src_offsets + 8 + 8 * 2, 1000);  // last offset past the names blob
               }),
               "bad source table");
}

TEST_F(Forgery, BuilderState) {
  EXPECT_ERROR(load_error([](auto& b, const Layout& L) { poke<std::uint64_t>(b, L.triples_len, std::uint64_t{1} << 50); }),
               "array length exceeds file size");
  EXPECT_ERROR(load_error([](auto& b, const Layout& L) { poke<std::uint32_t>(b, L.triples + 8, 1u << 30); }),
               "Bm25Builder::load: invalid posting");  // doc id beyond the doc count
  EXPECT_ERROR(load_error([](auto& b, const Layout& L) { poke<std::uint32_t>(b, L.triples + 12, 0); }),
               "Bm25Builder::load: invalid posting");  // term frequency 0
  EXPECT_ERROR(load_error([](auto& b, const Layout& L) { poke<std::uint32_t>(b, L.builder_ndocs, static_cast<std::uint32_t>(L.n + 1)); }),
               "doc length table does not match");
}

TEST_F(Forgery, SavedSnapshot) {
  EXPECT_ERROR(load_error([](auto& b, const Layout& L) { poke<std::uint32_t>(b, L.index_record + 32, static_cast<std::uint32_t>(L.n + 1)); }),
               "covers more chunks than the file holds");
  // A BM25 doc count that disagrees with the snapshot's coverage. Raising it
  // keeps every posting's doc id valid, so only the cross-check can catch it.
  EXPECT_ERROR(load_error([](auto& b, const Layout& L) { poke<std::uint32_t>(b, L.bm25_ndocs, static_cast<std::uint32_t>(L.n + 5)); }),
               "saved snapshot is inconsistent");
  EXPECT_ERROR(load_error([](auto& b, const Layout& L) { poke<std::uint32_t>(b, L.ivf + 4, 7); }),  // empty IVF claiming rows
               "IvfIndex::load: inconsistent empty index");
}

TEST_F(Forgery, ParameterMismatchRebuildsInsteadOfFailing) {
  // A changed parameter in the index record is not an error: load() just
  // refuses to reuse the saved indexes and rebuilds them.
  auto b = good_;
  const Layout L(b);
  poke<float>(b, L.index_record, 9.f);  // k1
  fix_checksum(b);
  write_file(path_, b);
  Engine::LoadReport rep;
  auto e = Engine::load(path_, cfg_, {}, &rep);
  EXPECT_FALSE(rep.reused_indexes);
  EXPECT_GT(e->size(), 0u);
}

// ------------------------- BM25 index, crafted streams ------------------------
struct SlotBytes {  // mirrors Bm25Index::Slot
  std::uint64_t key;
  std::uint32_t offset, len;
};
static_assert(sizeof(SlotBytes) == 16);

std::string bm25_error(const std::vector<SlotBytes>& table, std::uint64_t nterms, std::uint32_t ndocs,
                       const std::vector<std::uint32_t>& doc, const std::vector<float>& weight) {
  const auto p = temp_index("bm25");
  {
    io::Writer w(p);
    w.vec(table);
    w.pod(nterms);
    w.pod(ndocs);
    w.vec(doc);
    w.vec(weight);
    w.commit();
  }
  return error_of([&] {
    io::Reader r(p);
    Bm25Index x;
    x.load(r);
    r.finish();
  });
}

std::vector<SlotBytes> table_of(std::size_t cap, std::initializer_list<SlotBytes> used) {
  std::vector<SlotBytes> t(cap, SlotBytes{0, 0, 0});
  std::size_t i = 0;
  for (const auto& s : used) t[i++] = s;
  return t;
}

TEST(ForgeryBm25, EveryStructuralCheck) {
  const std::vector<std::uint32_t> doc{0, 1};
  const std::vector<float> w{1.f, 2.f};
  // Valid baseline.
  EXPECT_EQ(bm25_error(table_of(4, {{11, 0, 2}}), 1, 2, doc, w), "no exception");
  EXPECT_EQ(bm25_error({}, 0, 0, {}, {}), "no exception");  // empty index
  // Capacity not a power of two (lookup masks would index out of range).
  EXPECT_ERROR(bm25_error(table_of(3, {{11, 0, 2}}), 1, 2, doc, w), "inconsistent index");
  // doc / weight arrays of different lengths.
  EXPECT_ERROR(bm25_error(table_of(4, {{11, 0, 2}}), 1, 2, doc, {1.f}), "inconsistent index");
  // More terms claimed than slots.
  EXPECT_ERROR(bm25_error(table_of(4, {{11, 0, 2}}), 9, 2, doc, w), "inconsistent index");
  // Posting range past the postings array.
  EXPECT_ERROR(bm25_error(table_of(4, {{11, 1, 2}}), 1, 2, doc, w), "posting range out of bounds");
  EXPECT_ERROR(bm25_error(table_of(4, {{11, 0xFFFFFFFFu, 0xFFFFFFFFu}}), 1, 2, doc, w), "posting range out of bounds");
  // Term count disagreeing with the occupied slots.
  EXPECT_ERROR(bm25_error(table_of(4, {{11, 0, 2}, {12, 0, 1}}), 1, 2, doc, w), "bad term table");
  // A completely full table: a lookup miss would probe forever.
  EXPECT_ERROR(bm25_error(table_of(2, {{11, 0, 1}, {12, 1, 1}}), 2, 2, doc, w), "bad term table");
  // Doc id beyond the doc count.
  EXPECT_ERROR(bm25_error(table_of(4, {{11, 0, 2}}), 1, 1, doc, w), "doc id out of range");
}

// -------------------------- IVF index, crafted streams ------------------------
constexpr std::size_t kSmallD = 64;

struct IvfBytes {
  std::uint32_t nlist = 2, covered = 4;
  std::uint8_t flags = 1;
  std::vector<float> centroids = std::vector<float>(2 * kSmallD, 0.f);
  std::vector<std::uint32_t> offsets{0, 2, 4}, ids{0, 1, 2, 3};
  std::size_t f32_rows = 4, i8_rows = 0;
};

std::string ivf_error(const IvfBytes& x, std::uint32_t max_rows = 4) {
  const auto p = temp_index("ivf");
  {
    io::Writer w(p);
    w.pod(x.nlist);
    w.pod(x.covered);
    w.pod(x.flags);
    if (x.nlist) {
      w.bytes(x.centroids.data(), x.centroids.size() * sizeof(float));
      w.vec(x.offsets);
      w.vec(x.ids);
      const std::vector<float> f(x.f32_rows * kSmallD, 0.5f);
      w.bytes(f.data(), f.size() * sizeof(float));
      const std::vector<std::int8_t> q(x.i8_rows * kSmallD, 1);
      w.bytes(q.data(), q.size());
      const std::vector<float> sc(x.i8_rows, 1.f);
      w.bytes(sc.data(), sc.size() * sizeof(float));
    }
    w.commit();
  }
  return error_of([&] {
    io::Reader r(p);
    IvfIndex<kSmallD> y;
    y.load(r, max_rows);
    r.finish();
  });
}

TEST(ForgeryIvf, EveryStructuralCheck) {
  EXPECT_EQ(ivf_error(IvfBytes{}), "no exception");  // valid fp32 baseline
  { IvfBytes x; x.flags = 2; x.f32_rows = 0; x.i8_rows = 4; EXPECT_EQ(ivf_error(x), "no exception"); }  // valid int8
  { IvfBytes x; x.flags = 3; x.i8_rows = 4; EXPECT_EQ(ivf_error(x), "no exception"); }                  // valid both
  { IvfBytes x; x.nlist = 0; x.covered = 0; x.flags = 0; EXPECT_EQ(ivf_error(x), "no exception"); }     // valid empty

  { IvfBytes x; x.nlist = 0; x.covered = 3; x.flags = 0; EXPECT_ERROR(ivf_error(x), "inconsistent empty index"); }
  { IvfBytes x; x.nlist = 0; x.covered = 0; x.flags = 1; EXPECT_ERROR(ivf_error(x), "inconsistent empty index"); }
  EXPECT_ERROR(ivf_error(IvfBytes{}, /*max_rows=*/3), "bad header");  // more rows than the store holds
  { IvfBytes x; x.nlist = 5; EXPECT_ERROR(ivf_error(x), "bad header"); }   // more lists than rows
  { IvfBytes x; x.flags = 0; EXPECT_ERROR(ivf_error(x), "bad header"); }   // no storage at all
  { IvfBytes x; x.flags = 4; EXPECT_ERROR(ivf_error(x), "bad header"); }   // unknown storage bit
  { IvfBytes x; x.offsets = {0, 4}; EXPECT_ERROR(ivf_error(x), "bad list layout"); }        // wrong length
  { IvfBytes x; x.offsets = {1, 2, 4}; EXPECT_ERROR(ivf_error(x), "bad list layout"); }     // doesn't start at 0
  { IvfBytes x; x.offsets = {0, 2, 3}; EXPECT_ERROR(ivf_error(x), "bad list layout"); }     // doesn't end at covered
  { IvfBytes x; x.offsets = {0, 5, 4}; EXPECT_ERROR(ivf_error(x), "bad list layout"); }     // not monotonic
  { IvfBytes x; x.ids = {0, 1, 2}; EXPECT_ERROR(ivf_error(x), "bad list layout"); }         // ids != covered
  { IvfBytes x; x.ids = {0, 1, 2, 9}; EXPECT_ERROR(ivf_error(x), "row id out of range"); }
  { IvfBytes x; x.f32_rows = 3; EXPECT_ERROR(ivf_error(x), "truncated"); }                  // short vector data
  { IvfBytes x; x.flags = 2; x.f32_rows = 0; x.i8_rows = 0; EXPECT_ERROR(ivf_error(x), "truncated"); }
}

}  // namespace
