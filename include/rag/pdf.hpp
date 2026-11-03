// rag/pdf.hpp — PDF ingestion: text extraction (Poppler), cleanup
// (rag/text_clean.hpp), and page-aware ingestion into a RagEngine.
//
// Requires libpoppler-cpp (Debian/Ubuntu: libpoppler-cpp-dev; link the
// `rag_pdf_support` CMake target). Everything else in rag-cpp stays
// dependency-free; this header is the only one that needs Poppler.
//
//   auto r = rag::pdf::ingest_pdf(engine, "report.pdf", /*doc_id=*/7);
//   // chunks now cite "(report.pdf, p. 12)" in prompts
//
// Scanned PDFs (page images with no text layer) yield empty pages; they are
// counted in the result so callers can warn or route them to OCR. Encrypted
// PDFs are rejected unless they open with an empty password.
#pragma once

#include <poppler-document.h>
#include <poppler-global.h>
#include <poppler-page.h>

#include <cstdint>
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "rag/engine.hpp"
#include "rag/text_clean.hpp"

namespace rag::pdf {

struct PdfError : std::runtime_error {
  using std::runtime_error::runtime_error;
};

struct Document {
  std::vector<std::string> pages;  // cleaned text per page, page 1 first
  std::size_t empty_pages = 0;     // pages with no text after cleanup (often scanned images)
  std::string title;               // from the document info dictionary, if any
};

namespace detail {
inline std::string utf8(const poppler::ustring& s) {
  const auto b = s.to_utf8();
  return {b.begin(), b.end()};
}
// Poppler reports recoverable syntax problems on stderr; collect instead.
inline void quiet(const std::string&, void*) {}
}  // namespace detail

// Extract and clean every page. Throws PdfError if the file is missing, not
// a PDF, or encrypted.
inline Document extract(const std::filesystem::path& path, const text::CleanOptions& clean = {}) {
  poppler::set_debug_error_function(detail::quiet, nullptr);
  if (!std::filesystem::is_regular_file(path)) throw PdfError("no such file: " + path.string());
  std::unique_ptr<poppler::document> doc(poppler::document::load_from_file(path.string()));
  if (!doc) throw PdfError("not a readable PDF: " + path.string());
  if (doc->is_locked()) throw PdfError("encrypted PDF (password required): " + path.string());

  std::vector<std::string> raw;
  const int n = doc->pages();
  raw.reserve(static_cast<std::size_t>(n > 0 ? n : 0));
  for (int i = 0; i < n; ++i) {
    std::unique_ptr<poppler::page> page(doc->create_page(i));
    // Reading order (not physical layout): columns come out one after another.
    raw.push_back(page ? detail::utf8(page->text(poppler::rectf(), poppler::page::non_raw_non_physical_layout))
                       : std::string{});
  }
  Document out;
  out.pages = text::clean_pages(std::move(raw), clean);
  for (const auto& p : out.pages) out.empty_pages += p.empty();
  out.title = detail::utf8(doc->get_title());
  return out;
}

struct IngestResult {
  std::uint32_t pages = 0;
  std::uint32_t empty_pages = 0;
  std::uint32_t chunks = 0;
  std::uint64_t text_bytes = 0;
};

// Ingest an already-extracted document under `source` (shown in citations).
template <class Engine>
IngestResult ingest(Engine& engine, const Document& doc, std::uint32_t doc_id, std::string_view source) {
  const auto joined = text::join_pages(doc.pages);
  IngestResult r;
  r.pages = static_cast<std::uint32_t>(doc.pages.size());
  r.empty_pages = static_cast<std::uint32_t>(doc.empty_pages);
  r.text_bytes = joined.text.size();
  r.chunks = engine.ingest(joined.text, doc_id, IngestOptions{.source = source, .page_starts = joined.page_starts});
  return r;
}

// Extract, clean and ingest a PDF; citations use the file name.
template <class Engine>
IngestResult ingest_pdf(Engine& engine, const std::filesystem::path& path, std::uint32_t doc_id,
                        const text::CleanOptions& clean = {}) {
  return ingest(engine, extract(path, clean), doc_id, path.filename().string());
}

}  // namespace rag::pdf
