// text_clean_tests.cpp — cleanup of extracted page text (no PDF dependency).
#include <gtest/gtest.h>

#include <random>
#include <string>
#include <vector>

#include "rag.hpp"
#include "rag/text_clean.hpp"

using namespace rag;
using namespace rag::text;

TEST(TextClean, LigaturesBecomeLettersSoQueriesMatch) {
  // (Literals are split after each escape: "\x80ect" would parse as one hex escape.)
  EXPECT_EQ(normalize_chars("\xEF\xAC\x81" "nance \xEF\xAC\x82" "ow e\xEF\xAC\x80" "ect \xEF\xAC\x83" "x \xEF\xAC\x84"
                            "e \xEF\xAC\x85" "op"),
            "finance flow effect ffix ffle stop");
  // The point: after cleanup the tokenizer sees the same term as the query.
  std::vector<Token> t;
  const std::string cleaned = normalize_chars("\xEF\xAC\x81" "nance");
  for_each_token(cleaned, [&](const Token& x) { t.push_back(x); });
  ASSERT_EQ(t.size(), 1u);
  EXPECT_EQ(t[0].hash, hash_term("finance"));
}

TEST(TextClean, InvisibleAndUnusualSpaceCharacters) {
  EXPECT_EQ(normalize_chars("co\xC2\xADoperate"), "cooperate");           // soft hyphen
  EXPECT_EQ(normalize_chars("a\xC2\xA0" "b"), "a b");                     // NBSP
  EXPECT_EQ(normalize_chars("a\xE2\x80\x83" "b\xE2\x80\xAF" "c"), "a b c");  // em space, narrow NBSP
  EXPECT_EQ(normalize_chars("zero\xE2\x80\x8Bwidth"), "zerowidth");       // ZWSP
  EXPECT_EQ(normalize_chars("\xEF\xBB\xBFstart"), "start");               // BOM
  EXPECT_EQ(normalize_chars("pre\xE2\x80\x91" "trade"), "pre-trade");     // non-breaking hyphen
  EXPECT_EQ(normalize_chars("a\r\nb\rc\fd\ve"), "a\nb\nc\nd\ne");
  EXPECT_EQ(normalize_chars(std::string("x\0y\x1bz\tq", 8)), "xyz q");    // controls dropped, tab -> space
  // Everything else passes through, including invalid UTF-8 and truncated sequences.
  EXPECT_EQ(normalize_chars("caf\xC3\xA9 \xFF\xFE \xEF\xAC"), "caf\xC3\xA9 \xFF\xFE \xEF\xAC");
  // Malformed sequences whose prefix looks like a ligature or a Unicode space
  // (found by the random test: the ligature table index used to underflow).
  EXPECT_EQ(normalize_chars("\xEF\xAC" "A"), "\xEF\xAC" "A");
  EXPECT_EQ(normalize_chars("\xE2\x80" "A"), "\xE2\x80" "A");
}

TEST(TextClean, ReflowJoinsLinesAndDehyphenates) {
  CleanOptions o;
  EXPECT_EQ(reflow("Market micro-\nstructure studies\n  how   orders\tbecome prices.", o),
            "Market microstructure studies how orders become prices.");
  EXPECT_EQ(reflow("First para line one\nline two\n\n\nSecond para", o), "First para line one line two\n\nSecond para");
  // Not a line-break hyphenation: next line starts uppercase or with a digit, or no letter before '-'.
  EXPECT_EQ(reflow("Anglo-\nSaxon", o), "Anglo- Saxon");
  EXPECT_EQ(reflow("pages 10-\n12", o), "pages 10- 12");
  EXPECT_EQ(reflow("x -\nfoo", o), "x - foo");
  CleanOptions keep = o;
  keep.dehyphenate = false;
  EXPECT_EQ(reflow("micro-\nstructure", keep), "micro- structure");
  keep.join_lines = false;
  EXPECT_EQ(reflow("a\nb\n\nc", keep), "a\nb\n\nc");
  EXPECT_EQ(reflow("\n\n   \n", o), "");
}

TEST(TextClean, LineKeyIgnoresCaseDigitsAndSpacing) {
  EXPECT_EQ(line_key("  Page 3   of 10 "), "page # of #");
  EXPECT_EQ(line_key("PAGE 12 OF 10"), "page # of #");
  EXPECT_EQ(line_key("- 7 -"), "- # -");
  EXPECT_EQ(line_key("2024-01-31"), "#-#-#");
  // Long lines keep their digits: they must repeat exactly to count as running.
  EXPECT_EQ(line_key("Q3 revenue grew 12% year on year"), "q3 revenue grew 12% year on year");
}

TEST(TextClean, BodyLinesThatDifferOnlyInNumbersAreNotHeaders) {
  // Short pages (slides): every line is an "edge" line. Lines that differ only
  // in their numbers must survive, while the exact repeat is removed.
  std::vector<std::string> slides;
  for (int q = 1; q <= 4; ++q)
    slides.push_back("Acme Investor Update\nQ" + std::to_string(q) + " revenue grew " + std::to_string(10 + q) +
                     "% year on year in every region");
  const auto out = clean_pages(slides);
  for (int q = 1; q <= 4; ++q) {
    EXPECT_EQ(out[q - 1], "Q" + std::to_string(q) + " revenue grew " + std::to_string(10 + q) +
                              "% year on year in every region");
  }
}

TEST(TextClean, StripsRunningHeadersFootersAndPageNumbers) {
  std::vector<std::string> pages;
  for (int i = 1; i <= 6; ++i)
    pages.push_back("ACME Research | Q3 Report\nBody text of page " + std::to_string(i) +
                    " mentions ACME Research | Q3 Report in the middle.\nMore body.\n- " + std::to_string(i) + " -");
  const auto out = clean_pages(pages);
  for (int i = 0; i < 6; ++i) {
    EXPECT_EQ(out[i].find("- " + std::to_string(i + 1) + " -"), std::string::npos) << out[i];  // page number gone
    EXPECT_NE(out[i].rfind("Body text of page", 0), std::string::npos) << out[i];  // header gone: body comes first
    EXPECT_NE(out[i].find("mentions ACME Research | Q3 Report in the middle"), std::string::npos);  // body copy kept
  }
}

TEST(TextClean, KeepsRepeatsInShortDocumentsAndRareLines) {
  // One page: nothing can repeat.
  EXPECT_EQ(clean_pages({"Report Title\nalpha\nPage 1"})[0], "Report Title alpha Page 1");
  // Two pages: a header/footer must be on both to count (found on real 2-page
  // handouts, where it used to survive).
  const auto two = clean_pages({"Risk Controls - Acme\nalpha\nPage 1", "Risk Controls - Acme\nbeta\nPage 2"});
  EXPECT_EQ(two[0], "alpha");
  EXPECT_EQ(two[1], "beta");
  const auto one_sided = clean_pages({"Risk Controls - Acme\nalpha", "Appendix\nbeta"});
  EXPECT_EQ(one_sided[0], "Risk Controls - Acme alpha");
  // A line at the top of only 1 of 5 pages is content, not a header.
  std::vector<std::string> five = {"Chapter One\nalpha", "x\nbeta", "y\ngamma", "z\ndelta", "w\nepsilon"};
  EXPECT_EQ(clean_pages(five)[0], "Chapter One alpha");
  // Threshold is configurable.
  CleanOptions strict;
  strict.repeated_line_fraction = 1.0;  // must be on every page
  std::vector<std::string> p = {"H\na", "H\nb", "c", "H\nd"};
  EXPECT_EQ(clean_pages(p, strict)[0], "H a");
  CleanOptions off;
  off.strip_repeated_lines = false;
  EXPECT_EQ(clean_pages({"H\na", "H\nb", "H\nc"}, off)[1], "H b");
}

TEST(TextClean, JoinPagesGivesStrictlyIncreasingStartsEvenWithEmptyPages) {
  const auto j = join_pages({"", "B", "", "", "CC", ""});
  ASSERT_EQ(j.page_starts.size(), 6u);
  for (std::size_t i = 1; i < 6; ++i) EXPECT_GT(j.page_starts[i], j.page_starts[i - 1]);
  EXPECT_EQ(j.text.substr(j.page_starts[1], 1), "B");
  EXPECT_EQ(j.text.substr(j.page_starts[4], 2), "CC");
  EXPECT_LE(j.page_starts.back(), j.text.size());
  EXPECT_TRUE(join_pages({}).page_starts.empty());
}

TEST(TextClean, RandomInputInvariants) {
  std::mt19937 rng(77);
  for (int it = 0; it < 500; ++it) {
    std::vector<std::string> pages(1 + rng() % 6);
    for (auto& p : pages) {
      p.resize(rng() % 120);
      for (auto& c : p) c = static_cast<char>(rng() % 4 == 0 ? '\n' : rng() % 256);
    }
    const auto out = clean_pages(pages);
    ASSERT_EQ(out.size(), pages.size());
    for (const auto& p : out) {
      for (unsigned char c : p) ASSERT_TRUE(c >= 0x20 || c == '\n') << "control char survived";
      EXPECT_EQ(p.find("\n\n\n"), std::string::npos);
      EXPECT_EQ(p.find("  "), std::string::npos);
      if (!p.empty()) {
        EXPECT_NE(p.front(), ' ');
        EXPECT_NE(p.back(), ' ');
        EXPECT_NE(p.back(), '\n');
      }
    }
  }
}

// ----------------------- engine: source and page metadata ---------------------
using Engine = RagEngine<HashingEmbedder<384>>;

TEST(PagedIngest, ChunksRecordSourceAndPageSpanAndPromptCitesThem) {
  EngineConfig cfg;
  cfg.max_chunks = 256;
  cfg.chunker = {8, 0};
  Engine e(cfg);
  // Page 1: 10 words, page 2: 6 words; window 8 => chunks [0,8) p.1, [8,16) pp.1-2.
  const std::string text = "alpha beta gamma delta epsilon zeta eta theta iota kappa\n\nlambda mu nu xi omicron pi";
  const std::uint32_t starts[] = {0, static_cast<std::uint32_t>(text.find("lambda"))};
  ASSERT_EQ(e.ingest(text, 3, IngestOptions{.source = "greek.pdf", .page_starts = starts}), 2u);
  e.ingest("no pages here at all", 4, IngestOptions{.source = "notes.txt"});
  e.ingest("plain text with nothing extra", 5);
  e.commit();
  EXPECT_EQ(e.chunk(0).source, "greek.pdf");
  EXPECT_EQ(e.chunk(0).page_first, 1u);
  EXPECT_EQ(e.chunk(0).page_last, 1u);
  EXPECT_EQ(e.chunk(1).page_first, 1u);
  EXPECT_EQ(e.chunk(1).page_last, 2u);
  EXPECT_EQ(e.chunk(0).source.data(), e.chunk(1).source.data());  // interned once
  EXPECT_EQ(e.chunk(2).page_first, 0u);

  auto ctx = e.make_context();
  EXPECT_NE(e.query("alpha gamma", 1, ctx).prompt.prompt.find("[1] (greek.pdf, p. 1)\n"), std::string_view::npos);
  EXPECT_NE(e.query("lambda omicron iota", 1, ctx).prompt.prompt.find("[1] (greek.pdf, pp. 1-2)\n"), std::string_view::npos);
  EXPECT_NE(e.query("nothing extra plain", 1, ctx).prompt.prompt.find("[1] (doc 5, chunk 0)\n"), std::string_view::npos);
  EXPECT_NE(e.query("pages here", 1, ctx).prompt.prompt.find("[1] (notes.txt, chunk 0)\n"), std::string_view::npos);
}

TEST(PagedIngest, RejectsBadPageOffsetsWithoutChangingTheEngine) {
  Engine e(EngineConfig{.max_chunks = 16});
  const std::string t = "one two three four";
  const std::uint32_t not_zero[] = {2}, not_increasing[] = {0, 5, 5}, too_far[] = {0, 999};
  EXPECT_THROW(e.ingest(t, 0, IngestOptions{.page_starts = not_zero}), std::invalid_argument);
  EXPECT_THROW(e.ingest(t, 0, IngestOptions{.page_starts = not_increasing}), std::invalid_argument);
  EXPECT_THROW(e.ingest(t, 0, IngestOptions{.page_starts = too_far}), std::invalid_argument);
  EXPECT_EQ(e.size(), 0u);
  EXPECT_EQ(e.arena_bytes(), 0u);  // nothing was copied before validation failed
}

TEST(PagedIngest, LongSourceNamesAreCutAtUtf8BoundaryInCitations) {
  Engine e(EngineConfig{.max_chunks = 16});
  std::string name;
  for (int i = 0; i < 70; ++i) name += "\xC3\xA9";  // 140 bytes of 'é'
  e.ingest("orderbook depth", 0, IngestOptions{.source = name});
  e.commit();
  auto ctx = e.make_context();
  const auto p = e.query("orderbook", 1, ctx).prompt.prompt;
  const auto open = p.find("[1] (") + 5, close = p.find(", chunk 0)");
  const std::string_view shown = p.substr(open, close - open);
  EXPECT_LE(shown.size(), 120u);
  EXPECT_EQ(shown.substr(shown.size() - 3), "...");
  EXPECT_EQ((shown.size() - 3) % 2, 0u);  // whole 2-byte characters only
  EXPECT_EQ(e.chunk(0).source, name);     // the stored name is complete
}

TEST(PagedIngest, SourcesAndPagesSurviveSaveAndLoadAndKeepInterning) {
  Engine e(EngineConfig{.max_chunks = 64});
  const std::string t = "first page words here\n\nsecond page words there";
  const std::uint32_t starts[] = {0, static_cast<std::uint32_t>(t.find("second"))};
  e.ingest(t, 1, IngestOptions{.source = "a.pdf", .page_starts = starts});
  e.ingest("unsourced text", 2);
  e.commit();
  const auto path = std::filesystem::temp_directory_path() / ("paged-" + std::to_string(::getpid()) + ".ragidx");
  e.save(path);
  const auto info = Engine::inspect(path);
  EXPECT_EQ(info.format_version, io::kFormatVersion);
  EXPECT_EQ(info.chunks, e.size());
  EXPECT_EQ(info.dim, 384u);
  auto l = Engine::load(path, EngineConfig{.max_chunks = 64});
  std::filesystem::remove(path);
  for (std::uint32_t i = 0; i < e.size(); ++i) {
    EXPECT_EQ(l->chunk(i).source, e.chunk(i).source);
    EXPECT_EQ(l->chunk(i).page_first, e.chunk(i).page_first);
    EXPECT_EQ(l->chunk(i).page_last, e.chunk(i).page_last);
  }
  l->ingest("more from the same file", 3, IngestOptions{.source = "a.pdf"});
  EXPECT_EQ(l->chunk(l->size() - 1).source.data(), l->chunk(0).source.data());  // reloaded names are interned
}
