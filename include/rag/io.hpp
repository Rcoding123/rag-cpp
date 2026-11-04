// rag/io.hpp — binary index file I/O (cold path only).
//
// Format: little-endian, fixed-width PODs, length-prefixed arrays, and a
// trailing 64-bit FNV-1a checksum over every byte before it. The reader
// rejects bad magic, unknown versions, truncation, trailing garbage, and
// checksum mismatches. Array lengths are checked against the bytes actually
// left in the file before anything is allocated, so a corrupted length can't
// trigger a huge allocation.
//
// Saves are crash-safe: data goes to "<path>.tmp", is fsync'ed, and is then
// renamed over <path>, and the directory is fsync'ed so the rename itself is
// durable. A crash or power loss at any point leaves either the complete old
// index or the complete new one under <path>, never a partial file.
#pragma once

#include <bit>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#if defined(__unix__) || defined(__APPLE__)
#include <fcntl.h>
#include <unistd.h>
#elif defined(_WIN32)
#include <fcntl.h>
#include <io.h>  // _wopen, _commit (FlushFileBuffers), _close
#endif

namespace rag::io {

namespace detail {
// Flush a file's (or directory's) contents to stable storage.
inline void fsync_path(const std::filesystem::path& p, bool directory) {
#if defined(__unix__) || defined(__APPLE__)
  const int fd = ::open(p.c_str(), directory ? O_RDONLY | O_DIRECTORY : O_RDONLY);
  if (fd < 0) throw std::runtime_error("rag::io: cannot open " + p.string() + " to sync");
  const int rc = ::fsync(fd);
  ::close(fd);
  if (rc != 0) throw std::runtime_error("rag::io: fsync failed for " + p.string());
#elif defined(_WIN32)
  // Windows has no directory handle to flush; NTFS journals the rename itself.
  if (directory) return;
  const int fd = ::_wopen(p.c_str(), _O_RDWR | _O_BINARY);
  if (fd < 0) throw std::runtime_error("rag::io: cannot open " + p.string() + " to sync");
  const int rc = ::_commit(fd);
  ::_close(fd);
  if (rc != 0) throw std::runtime_error("rag::io: flush failed for " + p.string());
#else
  (void)p; (void)directory;  // portable fallback: rename-only (atomic, not durable)
#endif
}
}  // namespace detail

static_assert(std::endian::native == std::endian::little, "index files are little-endian; add byte swapping to port");

struct FormatError : std::runtime_error {
  using std::runtime_error::runtime_error;
};

inline constexpr char kMagic[8] = {'R', 'A', 'G', 'C', 'P', 'P', 'I', 'X'};
inline constexpr std::uint32_t kFormatVersion = 2;  // 2: chunk source + page range

// Streaming 64-bit checksum, 8 bytes per step. Each step h = mix((h ^ w) * K)
// is a bijection in h, so changing any single word always changes the
// result. Fed byte-at-a-time or in arbitrary chunks, it gives the same
// value: a partial word is buffered until it fills.
class Checksum {
 public:
  void update(const void* p, std::size_t n) noexcept {
    if (n == 0) return;  // p may be null (empty vector): memcpy(dst, nullptr, 0) is UB
    const auto* b = static_cast<const unsigned char*>(p);
    while (n && fill_) {  // top up a partial word left by the previous call
      buf_[fill_++] = *b++;
      --n;
      if (fill_ == 8) { step(load(buf_)); fill_ = 0; }
    }
    for (; n >= 8; b += 8, n -= 8) step(load(b));
    // Here either n == 0, or the buffer is empty and n < 8: the tail fits.
    std::memcpy(buf_ + fill_, b, n);
    fill_ += static_cast<std::uint32_t>(n);
  }
  [[nodiscard]] std::uint64_t value() const noexcept {
    std::uint64_t h = h_;
    if (fill_) {  // fold in the tail, tagged with its length
      unsigned char t[8] = {};
      std::memcpy(t, buf_, fill_);
      h = mix((h ^ load(t) ^ (std::uint64_t{fill_} << 56)) * kMul);
    }
    return mix(h ^ total_bits());
  }

 private:
  static constexpr std::uint64_t kMul = 0x9E3779B97F4A7C15ull;
  static std::uint64_t load(const unsigned char* b) noexcept {
    std::uint64_t w;
    std::memcpy(&w, b, 8);
    return w;
  }
  static constexpr std::uint64_t mix(std::uint64_t x) noexcept { return x ^ (x >> 29); }
  void step(std::uint64_t w) noexcept {
    h_ = mix((h_ ^ w) * kMul);
    ++words_;
  }
  [[nodiscard]] std::uint64_t total_bits() const noexcept { return (words_ * 8 + fill_) * 8; }

  std::uint64_t h_ = 0x243F6A8885A308D3ull;
  std::uint64_t words_ = 0;
  unsigned char buf_[8] = {};
  std::uint32_t fill_ = 0;
};

class Writer {
 public:
  explicit Writer(std::filesystem::path path) : path_(std::move(path)), tmp_(path_.string() + ".tmp") {
    f_.open(tmp_, std::ios::binary | std::ios::trunc);
    if (!f_) throw std::runtime_error("rag::io: cannot open " + tmp_.string() + " for writing");
  }
  Writer(const Writer&) = delete;
  Writer& operator=(const Writer&) = delete;
  ~Writer() {
    if (!committed_) {  // abandoned (exception mid-save): leave the old file alone
      f_.close();
      std::error_code ec;
      std::filesystem::remove(tmp_, ec);
    }
  }

  void bytes(const void* p, std::size_t n) {
    f_.write(static_cast<const char*>(p), static_cast<std::streamsize>(n));
    sum_.update(p, n);
  }
  template <class T>
  void pod(const T& v) {
    static_assert(std::is_trivially_copyable_v<T>);
    bytes(&v, sizeof(T));
  }
  template <class T>
  void vec(const std::vector<T>& v) {
    pod<std::uint64_t>(v.size());
    bytes(v.data(), v.size() * sizeof(T));
  }

  // Append the checksum, flush, and atomically replace the target file.
  void commit() {
    const std::uint64_t h = sum_.value();
    f_.write(reinterpret_cast<const char*>(&h), sizeof h);
    f_.flush();
    if (!f_) throw std::runtime_error("rag::io: write failed for " + tmp_.string());
    f_.close();
    detail::fsync_path(tmp_, false);  // data is on disk before it gets the real name
    std::filesystem::rename(tmp_, path_);
    committed_ = true;
    const auto dir = path_.has_parent_path() ? path_.parent_path() : std::filesystem::path(".");
    detail::fsync_path(dir, true);  // make the rename itself durable
  }

 private:
  std::filesystem::path path_, tmp_;
  std::ofstream f_;
  Checksum sum_;
  bool committed_ = false;
};

class Reader {
 public:
  explicit Reader(const std::filesystem::path& path) : f_(path, std::ios::binary) {
    if (!f_) throw std::runtime_error("rag::io: cannot open " + path.string());
    f_.seekg(0, std::ios::end);
    const auto size = static_cast<std::uint64_t>(f_.tellg());
    f_.seekg(0);
    if (size < sizeof(std::uint64_t)) throw FormatError("rag::io: file too small");
    remaining_ = size - sizeof(std::uint64_t);  // payload bytes before the checksum
  }

  void bytes(void* p, std::size_t n) {
    if (n > remaining_) throw FormatError("rag::io: truncated file");
    f_.read(static_cast<char*>(p), static_cast<std::streamsize>(n));
    if (!f_) throw FormatError("rag::io: read failed");
    remaining_ -= n;
    sum_.update(p, n);
  }
  template <class T>
  [[nodiscard]] T pod() {
    static_assert(std::is_trivially_copyable_v<T>);
    T v;
    bytes(&v, sizeof(T));
    return v;
  }
  template <class T>
  [[nodiscard]] std::vector<T> vec() {
    const auto n = pod<std::uint64_t>();
    if (n > remaining_ / sizeof(T)) throw FormatError("rag::io: array length exceeds file size");
    std::vector<T> v(static_cast<std::size_t>(n));
    bytes(v.data(), v.size() * sizeof(T));
    return v;
  }
  [[nodiscard]] std::uint64_t remaining() const noexcept { return remaining_; }

  // Verify that the payload was consumed exactly and the checksum matches.
  void finish() {
    if (remaining_ != 0) throw FormatError("rag::io: unexpected trailing data");
    std::uint64_t h;
    f_.read(reinterpret_cast<char*>(&h), sizeof h);
    if (!f_ || h != sum_.value()) throw FormatError("rag::io: checksum mismatch (corrupted file)");
  }

 private:
  std::ifstream f_;
  std::uint64_t remaining_ = 0;
  Checksum sum_;
};

}  // namespace rag::io
