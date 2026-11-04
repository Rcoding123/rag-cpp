#include <cstdio>
#include "rag/pdf.hpp"
int main(int, char** argv) {
  rag::text::CleanOptions off; off.strip_repeated_lines = false; off.join_lines = false; off.dehyphenate = false;
  // Raw Poppler text, before any cleanup at all:
  std::unique_ptr<poppler::document> d(poppler::document::load_from_file(argv[1]));
  std::unique_ptr<poppler::page> p(d->create_page(1));
  const auto b = p->text(poppler::rectf(), poppler::page::non_raw_non_physical_layout).to_utf8();
  const std::string raw(b.begin(), b.end());
  std::printf("raw has U+FB01 ligature: %s\n", raw.find("\xEF\xAC\x81") != std::string::npos ? "yes" : "no");
  const auto cleaned = rag::pdf::extract(argv[1]).pages[1];
  std::printf("cleaned has ligature: %s, has 'final': %s\n", cleaned.find("\xEF\xAC\x81") != std::string::npos ? "yes" : "no",
              cleaned.find("final net fee") != std::string::npos ? "yes" : "no");
  for (const char* f : {"handbook/01_execution.pdf", "handbook/02_risk.pdf"}) {
    const auto doc = rag::pdf::extract(std::string(argv[2]) + "/" + f);
    std::printf("%s: header left on page 1: %s | footer left: %s\n", f,
                doc.pages[0].find("Northwind") != std::string::npos ? "YES" : "no", doc.pages[0].find("Page 1") != std::string::npos ? "YES" : "no");
  }
}
