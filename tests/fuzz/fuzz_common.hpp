// Shared setup for the libFuzzer targets.
#pragma once

#include <unistd.h>

#include <cstdlib>
#include <filesystem>
#include <string>

#include "rag.hpp"

namespace rag::fuzz {

// Small dimension keeps each execution fast; the code paths are the same.
inline constexpr std::size_t kD = 64;
using Engine = RagEngine<HashingEmbedder<kD>, 16>;

// Crash loudly: libFuzzer reports aborts with the input that caused them.
#define FUZZ_CHECK(cond)                                                                   \
  do {                                                                                     \
    if (!(cond)) {                                                                         \
      std::fprintf(stderr, "FUZZ_CHECK failed: %s at %s:%d\n", #cond, __FILE__, __LINE__); \
      std::abort();                                                                        \
    }                                                                                      \
  } while (0)

inline std::filesystem::path scratch_file(const char* name) {
  const char* base = std::getenv("RAG_FUZZ_TMP");
  const std::filesystem::path dir = base ? base : (std::filesystem::exists("/dev/shm") ? "/dev/shm" : "/tmp");
  return dir / (std::string("rag-fuzz-") + name + "-" + std::to_string(::getpid()));
}

inline EngineConfig small_config(DenseMode mode = DenseMode::Exact) {
  EngineConfig c;
  c.max_chunks = 2048;
  c.chunker = {12, 3};
  c.retrieve_depth = 8;
  c.dense_mode = mode;
  c.ivf.nlist = 4;
  c.nprobe = 2;
  c.prompt_capacity = 4096;
  c.prompt.max_context_tokens = 256;
  return c;
}

}  // namespace rag::fuzz
