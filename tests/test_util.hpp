// Helpers shared by the test translation units.
#pragma once

#include <gtest/gtest.h>
#include <unistd.h>

#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

#include "rag/io.hpp"

namespace rag::testutil {

// Heap allocations (every operator new) between begin and end; defined in
// rag_tests.cpp, which replaces the global allocation functions.
void begin_alloc_count();
std::size_t end_alloc_count();

// Every temp index file a test creates; deleted when the test process ends.
inline std::vector<std::filesystem::path>& created_files() {
  static std::vector<std::filesystem::path> v;
  return v;
}

// Unique per test *and* per process: ctest -j runs tests as parallel processes,
// so a shared name would let two tests clobber each other's files.
inline std::filesystem::path temp_index(const char* name) {
  const auto* info = ::testing::UnitTest::GetInstance()->current_test_info();
  std::string id = std::string(info ? info->test_suite_name() : "nosuite") + "." + (info ? info->name() : "notest") +
                   "." + name + "." + std::to_string(::getpid()) + ".ragidx";
  for (char& c : id) if (c == '/') c = '_';
  auto p = std::filesystem::path(::testing::TempDir()) / id;
  created_files().push_back(p);
  return p;
}

inline std::vector<char> read_file(const std::filesystem::path& p) {
  std::ifstream f(p, std::ios::binary);
  return {std::istreambuf_iterator<char>(f), {}};
}

inline void write_file(const std::filesystem::path& p, const std::vector<char>& b) {
  std::ofstream f(p, std::ios::binary | std::ios::trunc);
  f.write(b.data(), static_cast<std::streamsize>(b.size()));
}

// Rewrite the trailing checksum so a deliberately edited file passes the
// integrity check and exercises the structural validation behind it.
inline void fix_checksum(std::vector<char>& b) {
  io::Checksum c;
  c.update(b.data(), b.size() - 8);
  const std::uint64_t h = c.value();
  std::memcpy(b.data() + b.size() - 8, &h, 8);
}

template <class T>
void poke(std::vector<char>& b, std::size_t off, T v) { std::memcpy(b.data() + off, &v, sizeof v); }

template <class T>
T peek(const std::vector<char>& b, std::size_t off) {
  T v;
  std::memcpy(&v, b.data() + off, sizeof v);
  return v;
}

// The message of the exception f() throws, or "no exception".
template <class F>
std::string error_of(F&& f) {
  try {
    f();
  } catch (const std::exception& e) {
    return e.what();
  }
  return "no exception";
}

}  // namespace rag::testutil
