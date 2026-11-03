// Minimal PDF writer for test fixtures: multi-page documents of text lines in
// Helvetica (WinAnsiEncoding), optionally in two columns, with a valid xref
// table and an optional /Title. Deterministic and dependency-free, so tests
// never rely on binary files checked into the repository.
#pragma once

#include <cstdio>
#include <fstream>
#include <string>
#include <utility>
#include <vector>

namespace rag::testutil {

struct PdfPage {
  PdfPage() = default;
  PdfPage(std::vector<std::string> l, std::vector<std::string> right = {})
      : lines(std::move(l)), right_column(std::move(right)) {}
  std::vector<std::string> lines;        // UTF-8; Latin-1 characters are encoded as WinAnsi
  std::vector<std::string> right_column; // optional second column, set beside `lines`
};

namespace detail {
// PDF string literal body: escape ( ) \ and map Latin-1 UTF-8 sequences to one byte.
inline std::string pdf_escape(const std::string& s) {
  std::string out;
  for (std::size_t i = 0; i < s.size(); ++i) {
    const auto c = static_cast<unsigned char>(s[i]);
    if (c == '(' || c == ')' || c == '\\') { out += '\\'; out += static_cast<char>(c); continue; }
    if ((c == 0xC2 || c == 0xC3) && i + 1 < s.size()) {  // U+0080..U+00FF
      const unsigned v = ((c & 0x1Fu) << 6) | (static_cast<unsigned char>(s[++i]) & 0x3Fu);
      char oct[8];
      std::snprintf(oct, sizeof oct, "\\%03o", v);
      out += oct;
      continue;
    }
    out += static_cast<char>(c);
  }
  return out;
}

inline std::string column(const std::vector<std::string>& lines, int x) {
  std::string s = "BT /F1 10 Tf 13 TL " + std::to_string(x) + " 740 Td\n";
  for (std::size_t i = 0; i < lines.size(); ++i) s += (i ? "T* (" : "(") + pdf_escape(lines[i]) + ") Tj\n";
  return s + "ET\n";
}
}  // namespace detail

inline void write_pdf(const std::string& path, const std::vector<PdfPage>& pages, const std::string& title = "") {
  std::vector<std::string> objs;  // objs[i] is object i+1
  const std::size_t n = pages.size();
  // 1 catalog, 2 pages tree, 3 font, 4 info, then (page, contents) pairs.
  std::string kids;
  for (std::size_t i = 0; i < n; ++i) kids += std::to_string(5 + 2 * i) + " 0 R ";
  objs.push_back("<< /Type /Catalog /Pages 2 0 R >>");
  objs.push_back("<< /Type /Pages /Kids [" + kids + "] /Count " + std::to_string(n) + " >>");
  objs.push_back("<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>");
  objs.push_back("<< /Title (" + detail::pdf_escape(title) + ") /Producer (rag-cpp tests) >>");
  for (std::size_t i = 0; i < n; ++i) {
    const bool two = !pages[i].right_column.empty();
    std::string content = detail::column(pages[i].lines, 56);
    if (two) content += detail::column(pages[i].right_column, 320);
    objs.push_back("<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 3 0 R >> >> "
                   "/Contents " + std::to_string(6 + 2 * i) + " 0 R >>");
    objs.push_back("<< /Length " + std::to_string(content.size()) + " >>\nstream\n" + content + "endstream");
  }
  std::string out = "%PDF-1.4\n%\xE2\xE3\xCF\xD3\n";
  std::vector<std::size_t> offsets;
  for (std::size_t i = 0; i < objs.size(); ++i) {
    offsets.push_back(out.size());
    out += std::to_string(i + 1) + " 0 obj\n" + objs[i] + "\nendobj\n";
  }
  const std::size_t xref = out.size();
  out += "xref\n0 " + std::to_string(objs.size() + 1) + "\n0000000000 65535 f \n";
  for (auto off : offsets) {
    char line[24];
    std::snprintf(line, sizeof line, "%010zu 00000 n \n", off);
    out += line;
  }
  out += "trailer\n<< /Size " + std::to_string(objs.size() + 1) + " /Root 1 0 R /Info 4 0 R >>\nstartxref\n" +
         std::to_string(xref) + "\n%%EOF\n";
  std::ofstream(path, std::ios::binary) << out;
}

}  // namespace rag::testutil
