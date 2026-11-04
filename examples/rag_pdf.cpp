// rag_pdf — index PDFs and text/source files, then ask questions with citations.
//
//   rag_pdf index <index-file> <pdf-or-directory>... [options]
//   rag_pdf ask   <index-file> "<question>" [-k N] [--prompt] [options]
//   rag_pdf info  <index-file>
//
// Options:
//   --mode exact|int8|ivf|ivf-sq8   dense retrieval mode (default exact)
//   --window N --overlap N          chunk size in words (default 128 / 32)
//   -k N                            passages to show (default 5)
//   --prompt                        also print the prompt an LLM would receive
//
// Directories are searched recursively for PDFs and text files (.md .txt .rst
// .hpp .h .cpp .cc .c .py .rs .go .java .ts .js .json .yml .yaml .toml .csv
// .cmake CMakeLists.txt), skipping hidden directories, build*, dist and
// node_modules. PDFs are cited by page ("q3.pdf, p. 4"); text files by chunk
// ("api.md, chunk 2"). Files found in a directory are cited by their path
// relative to it; files named directly by file name. Text files are indexed
// verbatim (characters normalized, no reflow, so code keeps its lines);
// binary files (containing NUL bytes) and files over 8 MB are skipped.
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

#include "rag/pdf.hpp"

namespace fs = std::filesystem;
using Engine = rag::RagEngine<rag::HashingEmbedder<384>>;

namespace {

[[noreturn]] void usage(const char* msg = nullptr) {
  if (msg) std::fprintf(stderr, "error: %s\n\n", msg);
  std::fprintf(stderr,
               "usage:\n"
               "  rag_pdf index <index-file> <pdf-or-directory>... [--mode M] [--window N] [--overlap N]\n"
               "  rag_pdf ask   <index-file> \"<question>\" [-k N] [--prompt] [--mode M]\n"
               "  rag_pdf info  <index-file>\n"
               "modes: exact (default), int8, ivf, ivf-sq8\n");
  std::exit(2);
}

struct Args {
  std::vector<std::string> positional;
  rag::DenseMode mode = rag::DenseMode::Exact;
  std::uint32_t window = 128, overlap = 32, k = 5;
  bool prompt = false;
};

std::uint32_t parse_u32(const char* s, const char* what) {
  char* end = nullptr;
  const unsigned long v = std::strtoul(s, &end, 10);
  if (!*s || *end || v == 0 || v > 1'000'000) usage((std::string("bad value for ") + what + ": " + s).c_str());
  return static_cast<std::uint32_t>(v);
}

Args parse(int argc, char** argv, int from) {
  Args a;
  for (int i = from; i < argc; ++i) {
    const std::string s = argv[i];
    auto next = [&]() -> const char* {
      if (i + 1 >= argc) usage(("missing value after " + s).c_str());
      return argv[++i];
    };
    if (s == "--mode") {
      const std::string m = next();
      if (m == "exact") a.mode = rag::DenseMode::Exact;
      else if (m == "int8") a.mode = rag::DenseMode::Int8Rerank;
      else if (m == "ivf") a.mode = rag::DenseMode::Ivf;
      else if (m == "ivf-sq8") a.mode = rag::DenseMode::IvfInt8;
      else usage(("unknown mode " + m).c_str());
    } else if (s == "--window") a.window = parse_u32(next(), "--window");
    else if (s == "--overlap") a.overlap = static_cast<std::uint32_t>(std::strtoul(next(), nullptr, 10));
    else if (s == "-k") a.k = parse_u32(next(), "-k");
    else if (s == "--prompt") a.prompt = true;
    else if (s.size() > 1 && s[0] == '-') usage(("unknown option " + s).c_str());
    else a.positional.push_back(s);
  }
  if (a.overlap >= a.window) usage("--overlap must be smaller than --window");
  return a;
}

rag::EngineConfig config_for(const Args& a, std::uint32_t max_chunks) {
  rag::EngineConfig cfg;
  cfg.max_chunks = std::max<std::uint32_t>(max_chunks, 1);
  cfg.chunker = {a.window, a.overlap};
  cfg.dense_mode = a.mode;
  cfg.nprobe = 16;
  cfg.prompt.max_context_tokens = 2048;
  cfg.prompt_capacity = 64 * 1024;
  return cfg;
}

enum class Kind { Pdf, Text, Other };

struct Input {
  fs::path path;
  std::string cite_as;
  Kind kind;
};

Kind kind_of(const fs::path& p) {
  std::string ext = p.extension().string();
  std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  if (ext == ".pdf") return Kind::Pdf;
  static const char* text[] = {".md", ".txt", ".rst", ".hpp", ".h", ".cpp", ".cc", ".c", ".py", ".rs", ".go", ".java",
                               ".ts", ".js", ".json", ".yml", ".yaml", ".toml", ".csv", ".cmake"};
  for (const char* t : text)
    if (ext == t) return Kind::Text;
  return p.filename() == "CMakeLists.txt" ? Kind::Text : Kind::Other;
}

bool skip_dir(const fs::path& d) {
  const std::string n = d.filename().string();
  return (n.size() > 1 && n[0] == '.') || n.rfind("build", 0) == 0 || n == "dist" || n == "node_modules";
}

constexpr std::uintmax_t kMaxTextBytes = 8u << 20;

// Reads a text file; empty optional-like result (ok=false) for binary or oversized files.
bool read_text(const fs::path& p, std::string& out, std::string& why) {
  if (fs::file_size(p) > kMaxTextBytes) { why = "larger than 8 MB"; return false; }
  std::ifstream f(p, std::ios::binary);
  out.assign(std::istreambuf_iterator<char>(f), {});
  if (out.find('\0') != std::string::npos) { why = "binary (contains NUL bytes)"; return false; }
  out = rag::text::normalize_chars(out);  // CRLF, odd spaces, ligatures; lines are kept
  return true;
}

std::vector<Input> collect(const std::vector<std::string>& args) {
  std::vector<Input> out;
  for (const auto& a : args) {
    const fs::path p = a;
    if (fs::is_directory(p)) {
      std::vector<Input> found;
      for (auto it = fs::recursive_directory_iterator(p, fs::directory_options::skip_permission_denied);
           it != fs::recursive_directory_iterator(); ++it) {
        if (it->is_directory() && skip_dir(it->path())) { it.disable_recursion_pending(); continue; }
        const Kind k = it->is_regular_file() ? kind_of(it->path()) : Kind::Other;
        if (k != Kind::Other) found.push_back({it->path(), fs::relative(it->path(), p).generic_string(), k});
      }
      std::sort(found.begin(), found.end(), [](const Input& x, const Input& y) { return x.cite_as < y.cite_as; });
      out.insert(out.end(), found.begin(), found.end());
    } else if (fs::is_regular_file(p)) {
      const Kind k = kind_of(p);
      if (k == Kind::Other) std::fprintf(stderr, "warning: skipping %s (not a PDF or a known text type)\n", a.c_str());
      else out.push_back({p, p.filename().string(), k});
    } else {
      std::fprintf(stderr, "warning: skipping %s (not a file or directory)\n", a.c_str());
    }
  }
  return out;
}

// Snippet for display: whitespace collapsed, cut at a word boundary.
std::string snippet(std::string_view t, std::size_t max = 280) {
  std::string s;
  bool sp = false;
  for (char c : t) {
    if (c == ' ' || c == '\n' || c == '\t') { sp = !s.empty(); continue; }
    if (sp) s += ' ';
    sp = false;
    s += c;
  }
  if (s.size() <= max) return s;
  std::size_t cut = s.rfind(' ', max);
  if (cut == std::string::npos || cut < max / 2) cut = max;
  while (cut > 0 && (static_cast<unsigned char>(s[cut]) & 0xC0) == 0x80) --cut;  // UTF-8 boundary
  return s.substr(0, cut) + " ...";
}

std::string where(const rag::ChunkView& c) {
  std::string w = c.source.empty() ? "doc " + std::to_string(c.doc_id) : std::string(c.source);
  if (c.page_first == 0) return w + ", chunk " + std::to_string(c.ordinal);
  if (c.page_first == c.page_last) return w + ", p. " + std::to_string(c.page_first);
  return w + ", pp. " + std::to_string(c.page_first) + "-" + std::to_string(c.page_last);
}

int cmd_index(const Args& a) {
  if (a.positional.size() < 2) usage("index needs an index file and at least one file or directory");
  const fs::path index = a.positional[0];
  const auto inputs = collect({a.positional.begin() + 1, a.positional.end()});
  if (inputs.empty()) usage("no PDF or text files found");

  // Extract everything first, so the engine can be sized exactly.
  struct Extracted {
    Input in;
    rag::pdf::Document doc;  // PDFs
    std::string text;        // text files
  };
  std::vector<Extracted> docs;
  std::uint64_t words = 0;
  for (const auto& in : inputs) {
    try {
      Extracted x{in, {}, {}};
      std::string why;
      if (in.kind == Kind::Pdf) {
        x.doc = rag::pdf::extract(in.path);
        for (const auto& p : x.doc.pages) rag::for_each_token(p, [&](const rag::Token&) { ++words; });
      } else if (read_text(in.path, x.text, why)) {
        rag::for_each_token(x.text, [&](const rag::Token&) { ++words; });
      } else {
        std::fprintf(stderr, "warning: skipping %s: %s\n", in.cite_as.c_str(), why.c_str());
        continue;
      }
      docs.push_back(std::move(x));
    } catch (const std::exception& e) {
      std::fprintf(stderr, "warning: skipping %s: %s\n", in.cite_as.c_str(), e.what());
    }
  }
  if (docs.empty()) { std::fprintf(stderr, "error: no file could be read\n"); return 1; }
  // Chunks per document <= words / stride + 1.
  const std::uint64_t stride = a.window - a.overlap;
  const std::uint64_t estimate = words / stride + docs.size() + 16;
  if (estimate > 0xFFFFFFF0ull) { std::fprintf(stderr, "error: corpus too large\n"); return 1; }

  Engine e(config_for(a, static_cast<std::uint32_t>(estimate)));
  std::uint32_t total_pages = 0, total_empty = 0;
  std::size_t n_pdf = 0;
  for (std::uint32_t i = 0; i < docs.size(); ++i) {
    if (docs[i].in.kind == Kind::Text) {
      const auto chunks = e.ingest(docs[i].text, i, rag::IngestOptions{.source = docs[i].in.cite_as});
      std::printf("  %-40s   text %5u chunks\n", docs[i].in.cite_as.c_str(), chunks);
      continue;
    }
    ++n_pdf;
    const auto r = rag::pdf::ingest(e, docs[i].doc, i, docs[i].in.cite_as);
    total_pages += r.pages;
    total_empty += r.empty_pages;
    std::string note;
    if (r.empty_pages)
      note = "  (" + std::to_string(r.empty_pages) + (r.empty_pages == 1 ? " page" : " pages") + " without text: scanned?)";
    std::printf("  %-40s %4u pages %5u chunks%s\n", docs[i].in.cite_as.c_str(), r.pages, r.chunks, note.c_str());
  }
  e.commit();
  e.save(index);
  std::printf("indexed %zu %s (%zu PDF, %u pages), %u chunks -> %s (%.1f MB)\n", docs.size(),
              docs.size() == 1 ? "file" : "files", n_pdf, total_pages, e.size(),
              index.string().c_str(), static_cast<double>(fs::file_size(index)) / 1e6);
  if (total_empty)
    std::printf("note: %u %s no extractable text. Scanned pages need OCR before indexing.\n", total_empty,
                total_empty == 1 ? "page had" : "pages had");
  return 0;
}

int cmd_ask(const Args& a) {
  if (a.positional.size() != 2) usage("ask needs an index file and one question (quote it)");
  const auto info = Engine::inspect(a.positional[0]);
  auto e = Engine::load(a.positional[0], config_for(a, info.chunks + 1));
  auto ctx = e->make_context();
  const std::string& q = a.positional[1];
  const auto r = e->query(q, std::min<std::uint32_t>(a.k, static_cast<std::uint32_t>(Engine::kMaxK)), ctx);
  if (r.hits.empty()) {
    std::printf("No matching passages for: %s\n", q.c_str());
    return 0;
  }
  auto score_in = [](std::span<const rag::Hit> list, std::uint32_t id) -> std::string {
    for (std::size_t i = 0; i < list.size(); ++i)
      if (list[i].id == id) return "#" + std::to_string(i + 1);
    return "-";
  };
  if (r.hits.size() == 1) std::printf("Top passage for: %s\n\n", q.c_str());
  else std::printf("Top %zu passages for: %s\n\n", r.hits.size(), q.c_str());
  for (std::size_t i = 0; i < r.hits.size(); ++i) {
    const auto c = e->chunk(r.hits[i].id);
    std::printf("[%zu] %s   (semantic %s, keyword %s)\n    %s\n\n", i + 1, where(c).c_str(),
                score_in(r.dense, r.hits[i].id).c_str(), score_in(r.lexical, r.hits[i].id).c_str(),
                snippet(c.text).c_str());
  }
  if (a.prompt) std::printf("----- prompt for the LLM -----\n%.*s\n", static_cast<int>(r.prompt.prompt.size()), r.prompt.prompt.data());
  return 0;
}

int cmd_info(const Args& a) {
  if (a.positional.size() != 1) usage("info needs an index file");
  const auto info = Engine::inspect(a.positional[0]);
  std::printf("format v%u, %u chunks, %u-dim vectors, %.1f MB of text\n", info.format_version, info.chunks, info.dim,
              static_cast<double>(info.text_bytes) / 1e6);
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) usage();
  const std::string cmd = argv[1];
  try {
    const Args a = parse(argc, argv, 2);
    if (cmd == "index") return cmd_index(a);
    if (cmd == "ask") return cmd_ask(a);
    if (cmd == "info") return cmd_info(a);
    usage(("unknown command " + cmd).c_str());
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
  }
}
