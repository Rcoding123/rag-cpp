// rag/text_clean.hpp — cleanup for text extracted from paginated documents
// (PDFs), before it is chunked and indexed. No dependencies; cold path.
//
// Text extracted from a PDF has artifacts that hurt retrieval:
//   * typographic ligatures: "ﬁnance" is U+FB01 + "nance", which never
//     matches the query "finance";
//   * words hyphenated across line ends ("micro-" / "structure");
//   * hard line breaks in the middle of every sentence;
//   * running headers, footers and page numbers repeated on every page,
//     which pollute every chunk and dominate BM25 statistics;
//   * soft hyphens, zero-width spaces, BOMs, non-breaking and other Unicode
//     spaces, and control characters.
// clean_pages() fixes all of these, page by page, keeping the page structure
// so that chunks can still be cited by page.
#pragma once

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <map>
#include <set>
#include <string>
#include <string_view>
#include <vector>

namespace rag::text {

struct CleanOptions {
  bool dehyphenate = true;   // "micro-\nstructure" -> "microstructure"
  bool join_lines = true;    // single line breaks inside a paragraph become spaces
  bool strip_repeated_lines = true;  // running headers, footers, page numbers
  // A line is "running" if, in at least this share of pages, a line with the
  // same shape (see line_key) is among the first or last
  // `edge_lines` non-empty lines of the page...
  double repeated_line_fraction = 0.5;
  std::size_t edge_lines = 2;
  // ...and the document has at least this many pages. A repeat always needs
  // at least two pages, so in a 2-page document the line must be on both.
  std::size_t min_pages_for_repeats = 2;
};

// Character-level normalisation of one page:
//   ligatures U+FB00..FB06 -> ff fi fl ffi ffl st st
//   U+2010 / U+2011 (hyphen, non-breaking hyphen) -> '-'
//   NBSP, U+2000..200A, U+202F, U+205F, U+3000 -> ' '
//   soft hyphen, zero-width space/joiners, BOM -> removed
//   CR LF / CR / form feed / vertical tab -> '\n'; tab -> ' '
//   other ASCII control characters -> removed
// Anything else (including invalid UTF-8) passes through unchanged.
inline std::string normalize_chars(std::string_view in) {
  std::string out;
  out.reserve(in.size());
  const auto* p = reinterpret_cast<const unsigned char*>(in.data());
  const std::size_t n = in.size();
  for (std::size_t i = 0; i < n;) {
    const unsigned char c = p[i];
    if (c < 0x80) {
      if (c == '\r') {
        out += '\n';
        i += (i + 1 < n && p[i + 1] == '\n') ? 2 : 1;
        continue;
      }
      if (c == '\f' || c == '\v') out += '\n';
      else if (c == '\t') out += ' ';
      else if (c >= 0x20 || c == '\n') out += static_cast<char>(c);
      // other control characters (NUL, ESC, ...) are dropped
      ++i;
      continue;
    }
    if (c == 0xC2 && i + 1 < n) {
      if (p[i + 1] == 0xA0) { out += ' '; i += 2; continue; }  // NBSP
      if (p[i + 1] == 0xAD) { i += 2; continue; }              // soft hyphen
    }
    if (c == 0xE2 && i + 2 < n && p[i + 1] == 0x80 && p[i + 2] >= 0x80 && p[i + 2] <= 0xBF) {
      const unsigned char d = p[i + 2];  // a real continuation byte: U+2000..U+203F
      if (d <= 0x8A || d == 0xAF) { out += ' '; i += 3; continue; }  // U+2000..200A, U+202F spaces
      if (d >= 0x8B && d <= 0x8D) { i += 3; continue; }              // zero-width space / non-joiner / joiner
      if (d == 0x90 || d == 0x91) { out += '-'; i += 3; continue; }  // hyphen, non-breaking hyphen
    }
    if (c == 0xE2 && i + 2 < n && p[i + 1] == 0x81 && p[i + 2] == 0x9F) { out += ' '; i += 3; continue; }  // U+205F
    if (c == 0xE3 && i + 2 < n && p[i + 1] == 0x80 && p[i + 2] == 0x80) { out += ' '; i += 3; continue; }  // U+3000
    if (c == 0xEF && i + 2 < n && p[i + 1] == 0xBB && p[i + 2] == 0xBF) { i += 3; continue; }  // BOM / ZWNBSP
    // Ligatures U+FB00..FB06. The third byte must be a real continuation byte
    // (>= 0x80): indexing kLig with a smaller one would underflow.
    if (c == 0xEF && i + 2 < n && p[i + 1] == 0xAC && p[i + 2] >= 0x80 && p[i + 2] <= 0x86) {
      static constexpr std::string_view kLig[] = {"ff", "fi", "fl", "ffi", "ffl", "st", "st"};
      out += kLig[p[i + 2] - 0x80];
      i += 3;
      continue;
    }
    out += static_cast<char>(c);
    ++i;
  }
  return out;
}

namespace detail {

inline bool is_hspace(char c) { return c == ' ' || c == '\t'; }

inline std::string_view trim(std::string_view s) {
  while (!s.empty() && is_hspace(s.front())) s.remove_prefix(1);
  while (!s.empty() && is_hspace(s.back())) s.remove_suffix(1);
  return s;
}

inline std::vector<std::string_view> split_lines(std::string_view s) {
  std::vector<std::string_view> lines;
  std::size_t start = 0;
  for (std::size_t i = 0; i <= s.size(); ++i)
    if (i == s.size() || s[i] == '\n') {
      lines.push_back(s.substr(start, i - start));
      start = i + 1;
    }
  return lines;
}

}  // namespace detail

// The "shape" of a line for header/footer detection: ASCII case-folded and
// whitespace collapsed. In short lines (<= kDigitBlindMaxWords words), every
// run of digits also becomes '#', so "Page 3 of 10" and "Page 4 of 10" share
// the key "page # of #". Longer lines must repeat exactly: otherwise body
// sentences that differ only in numbers ("Q3 revenue grew 12% ...",
// "Q4 revenue grew 15% ...") would look like one running header.
inline constexpr std::size_t kDigitBlindMaxWords = 6;

inline std::string line_key(std::string_view line) {
  const std::string_view t = detail::trim(line);
  std::size_t words = 0;
  for (std::size_t i = 0; i < t.size(); ++i)
    if (!detail::is_hspace(t[i]) && (i == 0 || detail::is_hspace(t[i - 1]))) ++words;
  const bool digit_blind = words <= kDigitBlindMaxWords;
  std::string k;
  bool space = false, digit = false;
  for (char ch : t) {
    const auto c = static_cast<unsigned char>(ch);
    if (detail::is_hspace(ch)) { space = true; digit = false; continue; }
    if (space && !k.empty()) k += ' ';
    space = false;
    if (digit_blind && std::isdigit(c)) {
      if (!digit) k += '#';
      digit = true;
      continue;
    }
    digit = false;
    k += (c < 0x80) ? static_cast<char>(std::tolower(c)) : ch;
  }
  return k;
}

// Remove running headers/footers: lines whose key occurs near the top or
// bottom of enough pages (see CleanOptions). Only lines in those edge
// positions are removed; the same text in the body of a page is kept.
inline void strip_repeated_lines(std::vector<std::string>& pages, const CleanOptions& o) {
  if (pages.size() < std::max<std::size_t>(o.min_pages_for_repeats, 2)) return;
  // Edge line indices of every page.
  std::vector<std::vector<std::size_t>> edges(pages.size());
  std::vector<std::vector<std::string_view>> lines(pages.size());
  std::map<std::string, std::size_t> pages_with;
  for (std::size_t pg = 0; pg < pages.size(); ++pg) {
    lines[pg] = detail::split_lines(pages[pg]);
    std::vector<std::size_t> nonempty;
    for (std::size_t i = 0; i < lines[pg].size(); ++i)
      if (!detail::trim(lines[pg][i]).empty()) nonempty.push_back(i);
    std::set<std::size_t> e;
    for (std::size_t j = 0; j < nonempty.size() && j < o.edge_lines; ++j) e.insert(nonempty[j]);
    for (std::size_t j = 0; j < nonempty.size() && j < o.edge_lines; ++j) e.insert(nonempty[nonempty.size() - 1 - j]);
    edges[pg].assign(e.begin(), e.end());
    std::set<std::string> keys;
    for (std::size_t i : edges[pg]) keys.insert(line_key(lines[pg][i]));
    for (const auto& k : keys) ++pages_with[k];
  }
  const auto needed = std::max<std::size_t>(
      2, static_cast<std::size_t>(std::ceil(o.repeated_line_fraction * static_cast<double>(pages.size()))));
  for (std::size_t pg = 0; pg < pages.size(); ++pg) {
    std::set<std::size_t> drop;
    for (std::size_t i : edges[pg])
      if (pages_with[line_key(lines[pg][i])] >= needed) drop.insert(i);
    if (drop.empty()) continue;
    std::string kept;
    for (std::size_t i = 0; i < lines[pg].size(); ++i) {
      if (drop.count(i)) continue;
      if (!kept.empty()) kept += '\n';
      kept += lines[pg][i];
    }
    pages[pg] = std::move(kept);
  }
}

// Rebuild paragraphs from hard-wrapped lines. Lines are trimmed and inner
// whitespace collapsed; blank lines separate paragraphs ("\n\n" in the
// output). With dehyphenate, a line ending in letter + '-' followed by a line
// starting with a lowercase letter is joined without the hyphen.
inline std::string reflow(std::string_view page, const CleanOptions& o) {
  std::string out;
  bool in_paragraph = false, pending_break = false;
  for (std::string_view raw : detail::split_lines(page)) {
    // Trim and collapse runs of horizontal whitespace.
    std::string line;
    bool sp = false;
    for (char ch : detail::trim(raw)) {
      if (detail::is_hspace(ch)) { sp = true; continue; }
      if (sp) line += ' ';
      sp = false;
      line += ch;
    }
    if (line.empty()) {
      if (in_paragraph) pending_break = true;
      continue;
    }
    if (!in_paragraph) {
      out += line;
      in_paragraph = true;
      continue;
    }
    if (pending_break || !o.join_lines) {
      out += pending_break ? "\n\n" : "\n";
      out += line;
      pending_break = false;
      continue;
    }
    const bool hyphen_break = o.dehyphenate && out.size() >= 2 && out.back() == '-' &&
                              std::isalpha(static_cast<unsigned char>(out[out.size() - 2])) &&
                              std::islower(static_cast<unsigned char>(line.front()));
    if (hyphen_break) out.pop_back();
    else out += ' ';
    out += line;
  }
  return out;
}

// The full pipeline for one document: normalize characters on every page,
// strip running headers/footers across pages, then reflow each page.
inline std::vector<std::string> clean_pages(std::vector<std::string> pages, const CleanOptions& o = {}) {
  for (auto& p : pages) p = normalize_chars(p);
  if (o.strip_repeated_lines) strip_repeated_lines(pages, o);
  for (auto& p : pages) p = reflow(p, o);
  return pages;
}

// Join cleaned pages into one text (pages separated by a blank line) plus the
// byte offset at which each page starts, ready for IngestOptions::page_starts.
struct PagedText {
  std::string text;
  std::vector<std::uint32_t> page_starts;
};

inline PagedText join_pages(const std::vector<std::string>& pages) {
  PagedText r;
  for (std::size_t i = 0; i < pages.size(); ++i) {
    // Always separate, even around empty pages: page starts then strictly
    // increase, and no page's bytes are ever attributed to a neighbour.
    if (i > 0) r.text += "\n\n";
    r.page_starts.push_back(static_cast<std::uint32_t>(r.text.size()));
    r.text += pages[i];
  }
  return r;
}

}  // namespace rag::text
