#include <cstdio>
#include "pdf_writer.hpp"
#include "rag/pdf.hpp"
int main(int, char** argv) {
  using rag::testutil::PdfPage;
  rag::testutil::write_pdf(argv[1], {
    PdfPage{{"ACME Research - Confidential", "Market micro-", "structure studies how orders", "become prices. Café crème (test).", "", "Second paragraph here.", "Page 1 of 3"}},
    PdfPage{{"ACME Research - Confidential", "Iceberg orders hide size.", "Page 2 of 3"}},
    PdfPage{{"ACME Research - Confidential", "Left column first line", "left second line"}, {"Right column first", "right second"}},
  }, "Probe Title");
  const auto d = rag::pdf::extract(argv[1]);
  std::printf("title=[%s] pages=%zu empty=%zu\n", d.title.c_str(), d.pages.size(), d.empty_pages);
  for (std::size_t i = 0; i < d.pages.size(); ++i) std::printf("--- page %zu\n%s\n", i + 1, d.pages[i].c_str());
  rag::text::CleanOptions raw; raw.strip_repeated_lines = false; raw.join_lines = false; raw.dehyphenate = false;
  const auto r = rag::pdf::extract(argv[1], raw);
  std::printf("=== raw page 1\n%s\n", r.pages[0].c_str());
}
