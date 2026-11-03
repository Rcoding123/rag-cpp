// pdf_tests.cpp — PDF extraction and page-cited retrieval, on PDFs generated
// by tests/pdf_writer.hpp (no binary fixtures in the repository).
#include <gtest/gtest.h>
#include <unistd.h>

#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "pdf_writer.hpp"
#include "rag/pdf.hpp"

using namespace rag;
using rag::testutil::PdfPage;
using rag::testutil::write_pdf;
using Engine = RagEngine<HashingEmbedder<384>>;

namespace {

class PdfTest : public ::testing::Test {
 protected:
  void SetUp() override {
    dir_ = std::filesystem::temp_directory_path() / ("rag-pdf-tests-" + std::to_string(::getpid()) + "-" +
                                                     ::testing::UnitTest::GetInstance()->current_test_info()->name());
    std::filesystem::create_directories(dir_);
  }
  void TearDown() override { std::filesystem::remove_all(dir_); }
  std::string path(const std::string& name) const { return (dir_ / name).string(); }
  std::filesystem::path dir_;
};

const std::string kHeader = "Northwind Capital - Internal Research";

// A realistic 3-page report: running header, page-number footer, prose.
void write_report(const std::string& p) {
  write_pdf(p,
            {
                PdfPage{{kHeader, "Execution Costs", "Implementation shortfall measures the gap between the",
                         "decision price and the final execution price of an order.", "Page 1 of 3"}},
                PdfPage{{kHeader, "Exchange fees are charged per share. Maker rebates",
                         "reward liquidity providers while takers pay the fee.", "Page 2 of 3"}},
                PdfPage{{kHeader, "Iceberg orders display only a small visible quantity",
                         "and replenish it from a hidden reserve.", "Page 3 of 3"}},
            },
            "Execution Research Notes");
}

}  // namespace

TEST_F(PdfTest, ExtractsPagesTitleAndStripsRunningHeadersAndFooters) {
  write_report(path("report.pdf"));
  const auto d = pdf::extract(path("report.pdf"));
  ASSERT_EQ(d.pages.size(), 3u);
  EXPECT_EQ(d.title, "Execution Research Notes");
  EXPECT_EQ(d.empty_pages, 0u);
  for (const auto& p : d.pages) {
    EXPECT_EQ(p.find("Northwind"), std::string::npos) << p;  // header gone
    EXPECT_EQ(p.find("Page "), std::string::npos) << p;      // footer gone
  }
  EXPECT_EQ(d.pages[0],
            "Execution Costs Implementation shortfall measures the gap between the decision price and the final "
            "execution price of an order.");
  EXPECT_NE(d.pages[1].find("Maker rebates reward liquidity providers"), std::string::npos);
  // With cleanup off, the boilerplate is still there: the cleaner removed it.
  text::CleanOptions raw;
  raw.strip_repeated_lines = false;
  EXPECT_NE(pdf::extract(path("report.pdf"), raw).pages[1].find("Page 2 of 3"), std::string::npos);
}

TEST_F(PdfTest, TwoColumnPagesComeOutInReadingOrder) {
  write_pdf(path("cols.pdf"), {PdfPage{{"Left column opens the argument", "and the left column ends here."},
                                       {"Right column continues it", "and the right column closes."}}});
  const auto d = pdf::extract(path("cols.pdf"));
  ASSERT_EQ(d.pages.size(), 1u);
  const auto& t = d.pages[0];
  const auto l1 = t.find("Left column opens"), l2 = t.find("left column ends"), r1 = t.find("Right column continues");
  ASSERT_NE(l1, std::string::npos);
  ASSERT_NE(r1, std::string::npos);
  EXPECT_LT(l1, l2);
  EXPECT_LT(l2, r1);  // the whole left column before the right one, not interleaved line by line
}

TEST_F(PdfTest, LatinOneTextAndEscapedCharacters) {
  write_pdf(path("latin.pdf"), {PdfPage{{"Caf\xC3\xA9 cr\xC3\xA8me (na\xC3\xAFve) \\ test"}}});
  const auto d = pdf::extract(path("latin.pdf"));
  ASSERT_EQ(d.pages.size(), 1u);
  EXPECT_EQ(d.pages[0], "Caf\xC3\xA9 cr\xC3\xA8me (na\xC3\xAFve) \\ test");
}

TEST_F(PdfTest, EmptyPagesAreCountedAndKeepTheirPageNumbers) {
  // Page 2 has no text layer at all (like a scanned image page).
  write_pdf(path("gap.pdf"), {PdfPage{{"Opening page about auctions."}}, PdfPage{},
                              PdfPage{{"Closing page about dark pools."}}});
  const auto d = pdf::extract(path("gap.pdf"));
  ASSERT_EQ(d.pages.size(), 3u);
  EXPECT_EQ(d.empty_pages, 1u);
  EXPECT_TRUE(d.pages[1].empty());
  Engine e(EngineConfig{.max_chunks = 64});
  const auto r = pdf::ingest(e, d, 0, "gap.pdf");
  EXPECT_EQ(r.pages, 3u);
  EXPECT_EQ(r.empty_pages, 1u);
  e.commit();
  auto ctx = e.make_context();
  const auto q = e.query("dark pools", 1, ctx);
  ASSERT_FALSE(q.hits.empty());
  // One chunk holds pages 1 and 3; the citation spans them (the empty page 2 lies between).
  EXPECT_EQ(e.chunk(q.hits[0].id).page_first, 1u);
  EXPECT_EQ(e.chunk(q.hits[0].id).page_last, 3u);
}

TEST_F(PdfTest, RejectsMissingNonPdfAndDamagedFiles) {
  EXPECT_THROW((void)pdf::extract(path("nope.pdf")), pdf::PdfError);
  EXPECT_THROW((void)pdf::extract(dir_), pdf::PdfError);  // a directory
  std::ofstream(path("fake.pdf")) << "this is plain text, not a PDF";
  EXPECT_THROW((void)pdf::extract(path("fake.pdf")), pdf::PdfError);
  std::ofstream(path("empty.pdf")).close();
  EXPECT_THROW((void)pdf::extract(path("empty.pdf")), pdf::PdfError);
  // Truncated real PDFs: Poppler may repair them or give up. Either way the
  // call must not crash (ASan/UBSan in CI) and must not return garbage pages.
  write_report(path("full.pdf"));
  std::ifstream in(path("full.pdf"), std::ios::binary);
  const std::string bytes{std::istreambuf_iterator<char>(in), {}};
  for (std::size_t cut : {std::size_t{20}, bytes.size() / 3, bytes.size() / 2, bytes.size() - 30}) {
    std::ofstream(path("cut.pdf"), std::ios::binary) << bytes.substr(0, cut);
    try {
      const auto d = pdf::extract(path("cut.pdf"));
      EXPECT_LE(d.pages.size(), 3u) << "cut at " << cut;
    } catch (const pdf::PdfError&) {
    }
  }
}

TEST_F(PdfTest, ZeroPageDocument) {
  write_pdf(path("zero.pdf"), {});
  const auto d = pdf::extract(path("zero.pdf"));
  EXPECT_TRUE(d.pages.empty());
  Engine e(EngineConfig{.max_chunks = 8});
  EXPECT_EQ(pdf::ingest(e, d, 0, "zero.pdf").chunks, 0u);
}

TEST_F(PdfTest, AnswersCiteTheRightFileAndPage) {
  write_report(path("report.pdf"));
  write_pdf(path("glossary.pdf"),
            {PdfPage{{"Glossary", "Spoofing: placing orders one intends to cancel before execution."}},
             PdfPage{{"VWAP: the volume weighted average price over a trading session."}}});
  Engine e(EngineConfig{.max_chunks = 256, .chunker = {24, 4}});
  const auto a = pdf::ingest_pdf(e, path("report.pdf"), 0);
  const auto b = pdf::ingest_pdf(e, path("glossary.pdf"), 1);
  EXPECT_EQ(a.pages, 3u);
  EXPECT_EQ(b.pages, 2u);
  EXPECT_GT(a.chunks, 0u);
  e.commit();
  auto ctx = e.make_context();
  struct Case {
    const char* q;
    const char* source;
    std::uint32_t page;
  };
  for (const Case& c : {Case{"maker rebates exchange fees", "report.pdf", 2}, Case{"iceberg hidden reserve", "report.pdf", 3},
                        Case{"implementation shortfall decision price", "report.pdf", 1},
                        Case{"volume weighted average price", "glossary.pdf", 2}, Case{"spoofing cancel", "glossary.pdf", 1}}) {
    const auto r = e.query(c.q, 3, ctx);
    ASSERT_FALSE(r.hits.empty()) << c.q;
    const auto top = e.chunk(r.hits[0].id);
    EXPECT_EQ(top.source, c.source) << c.q;
    EXPECT_LE(top.page_first, c.page) << c.q;
    EXPECT_GE(top.page_last, c.page) << c.q;
    EXPECT_NE(r.prompt.prompt.find(std::string("(") + c.source + ", p"), std::string_view::npos) << r.prompt.prompt;
  }
}

TEST_F(PdfTest, ChunkSpanningAPageBreakCitesBothPages) {
  write_pdf(path("span.pdf"), {PdfPage{{"alpha bravo charlie delta echo"}}, PdfPage{{"foxtrot golf hotel india juliet"}}});
  Engine e(EngineConfig{.max_chunks = 16, .chunker = {6, 0}});
  pdf::ingest_pdf(e, path("span.pdf"), 0);
  e.commit();
  ASSERT_EQ(e.size(), 2u);  // words 1-6 and 7-10
  EXPECT_EQ(e.chunk(0).page_first, 1u);
  EXPECT_EQ(e.chunk(0).page_last, 2u);  // "alpha ... echo foxtrot"
  EXPECT_EQ(e.chunk(1).page_first, 2u);
  EXPECT_EQ(e.chunk(1).page_last, 2u);
  auto ctx = e.make_context();
  EXPECT_NE(e.query("echo foxtrot", 1, ctx).prompt.prompt.find("(span.pdf, pp. 1-2)"), std::string_view::npos);
}

TEST_F(PdfTest, IndexOfPdfsSurvivesSaveAndLoad) {
  write_report(path("report.pdf"));
  Engine e(EngineConfig{.max_chunks = 64, .chunker = {16, 4}});
  pdf::ingest_pdf(e, path("report.pdf"), 0);
  e.commit();
  e.save(path("pdfs.ragidx"));
  auto l = Engine::load(path("pdfs.ragidx"), e.config());
  auto ca = e.make_context(), cb = l->make_context();
  const auto ra = e.query("maker rebates", 3, ca), rb = l->query("maker rebates", 3, cb);
  EXPECT_EQ(ra.prompt.prompt, rb.prompt.prompt);  // same passages, same citations
  ASSERT_FALSE(rb.hits.empty());
  const auto top = l->chunk(rb.hits[0].id);
  EXPECT_EQ(top.source, "report.pdf");
  EXPECT_LE(top.page_first, 2u);  // the maker-rebates text is on page 2 (the chunk may start on page 1)
  EXPECT_GE(top.page_last, 2u);
}
