// Minimal DBN (Databento Binary Encoding) v2/v3 reader for MBO records,
// streaming from .dbn.zst (or plain .dbn). Layout verified against
// databento/dbn v0.74.0 (rust/dbn/src/record.rs, decode/dbn/fsm.rs).
// Python `databento` stays the reference reader; see scripts/verify_dbn.py.
#pragma once
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#if __has_include(<zstd.h>)
#include <zstd.h>
#else
// libzstd-dev is not installed on the server; the public streaming ABI is
// stable, so declare just what we use and link libzstd.so.1 directly.
extern "C" {
typedef struct ZSTD_DCtx_s ZSTD_DCtx;
typedef struct { const void* src; size_t size; size_t pos; } ZSTD_inBuffer;
typedef struct { void* dst; size_t size; size_t pos; } ZSTD_outBuffer;
ZSTD_DCtx* ZSTD_createDCtx(void);
size_t ZSTD_freeDCtx(ZSTD_DCtx*);
size_t ZSTD_decompressStream(ZSTD_DCtx*, ZSTD_outBuffer*, ZSTD_inBuffer*);
unsigned ZSTD_isError(size_t);
const char* ZSTD_getErrorName(size_t);
size_t ZSTD_DStreamInSize(void);
}
#endif

namespace mbo {

constexpr uint8_t RTYPE_MBO = 0xA0;
constexpr int64_t UNDEF_PRICE = INT64_MAX;

namespace flag {
constexpr uint8_t LAST = 1u << 7, TOB = 1u << 6, SNAPSHOT = 1u << 5, MBP = 1u << 4,
                  BAD_TS_RECV = 1u << 3, MAYBE_BAD_BOOK = 1u << 2;
}

#pragma pack(push, 1)
struct RecordHeader {
  uint8_t length;  // record size in 4-byte units
  uint8_t rtype;
  uint16_t publisher_id;
  uint32_t instrument_id;
  uint64_t ts_event;
};
struct MboMsg {
  RecordHeader hd;
  uint64_t order_id;
  int64_t price;  // 1e-9 units
  uint32_t size;
  uint8_t flags;
  uint8_t channel_id;
  char action;  // A C M R T F N
  char side;    // A B N
  uint64_t ts_recv;
  int32_t ts_in_delta;
  uint32_t sequence;
};
#pragma pack(pop)
static_assert(sizeof(RecordHeader) == 16, "header layout");
static_assert(sizeof(MboMsg) == 56, "MboMsg layout");

struct Metadata {
  uint8_t version = 0;
  std::string dataset;
  uint16_t schema = 0xFFFF;  // 0 = mbo
  uint64_t start = 0, end = 0;
  bool ts_out = false;
};

class DbnReader {
 public:
  explicit DbnReader(const std::string& path, size_t buf_bytes = 8u << 20) : buf_(buf_bytes) {
    f_ = std::fopen(path.c_str(), "rb");
    if (!f_) throw std::runtime_error("cannot open " + path);
    unsigned char magic[4];
    if (std::fread(magic, 1, 4, f_) != 4) throw std::runtime_error("short file " + path);
    std::rewind(f_);
    zstd_ = magic[0] == 0x28 && magic[1] == 0xB5 && magic[2] == 0x2F && magic[3] == 0xFD;
    if (zstd_) {
      ctx_ = ZSTD_createDCtx();
      in_.resize(ZSTD_DStreamInSize());
    }
    read_metadata();
  }
  ~DbnReader() {
    if (ctx_) ZSTD_freeDCtx(ctx_);
    if (f_) std::fclose(f_);
  }
  DbnReader(const DbnReader&) = delete;
  DbnReader& operator=(const DbnReader&) = delete;

  const Metadata& metadata() const { return meta_; }

  // Next record (header + body), valid until the next call. nullptr at clean
  // EOF. Throws on truncation/corruption.
  const RecordHeader* next() {
    if (!ensure(1)) return nullptr;
    size_t len = size_t(buf_[rpos_]) * 4;
    if (len < sizeof(RecordHeader)) throw std::runtime_error("bad record length");
    if (!ensure(len)) throw std::runtime_error("truncated record at end of stream");
    auto* h = reinterpret_cast<const RecordHeader*>(&buf_[rpos_]);
    rpos_ += len;
    return h;
  }
  // Next MBO record, skipping any other rtype. nullptr at EOF.
  const MboMsg* next_mbo() {
    while (const RecordHeader* h = next()) {
      if (h->rtype == RTYPE_MBO) {
        if (size_t(h->length) * 4 < sizeof(MboMsg)) throw std::runtime_error("short MBO record");
        return reinterpret_cast<const MboMsg*>(h);
      }
      ++skipped_;
    }
    return nullptr;
  }
  uint64_t skipped() const { return skipped_; }

 private:
  // Ensure n unread bytes are buffered. False only if EOF hits first with
  // zero bytes pending (n bytes unavailable but some pending => caller errors).
  bool ensure(size_t n) {
    while (wpos_ - rpos_ < n) {
      if (eof_) return false;
      if (rpos_ > 0) {
        std::memmove(buf_.data(), buf_.data() + rpos_, wpos_ - rpos_);
        wpos_ -= rpos_;
        rpos_ = 0;
      }
      size_t got = fill(buf_.data() + wpos_, buf_.size() - wpos_);
      if (got == 0) { eof_ = true; return false; }
      wpos_ += got;
    }
    return true;
  }
  size_t fill(uint8_t* dst, size_t cap) {
    if (!zstd_) return std::fread(dst, 1, cap, f_);
    ZSTD_outBuffer out{dst, cap, 0};
    while (out.pos == 0) {
      if (in_pos_ == in_len_) {
        in_len_ = std::fread(in_.data(), 1, in_.size(), f_);
        in_pos_ = 0;
        if (in_len_ == 0) {
          if (frame_open_) throw std::runtime_error("truncated zstd stream");
          return 0;
        }
      }
      ZSTD_inBuffer in{in_.data(), in_len_, in_pos_};
      size_t r = ZSTD_decompressStream(ctx_, &out, &in);
      if (ZSTD_isError(r)) throw std::runtime_error(std::string("zstd: ") + ZSTD_getErrorName(r));
      in_pos_ = in.pos;
      frame_open_ = (r != 0);  // 0 => a frame ended exactly here
    }
    return out.pos;
  }
  void read_metadata() {
    if (!ensure(8)) throw std::runtime_error("short DBN header");
    if (std::memcmp(&buf_[rpos_], "DBN", 3) != 0) throw std::runtime_error("not a DBN file");
    meta_.version = buf_[rpos_ + 3];
    if (meta_.version != 2 && meta_.version != 3)
      throw std::runtime_error("unsupported DBN version " + std::to_string(meta_.version));
    uint32_t mlen;
    std::memcpy(&mlen, &buf_[rpos_ + 4], 4);
    rpos_ += 8;
    if (mlen < 60 || !ensure(mlen)) throw std::runtime_error("short DBN metadata");
    const uint8_t* m = &buf_[rpos_];
    // v2/v3: dataset[16] schema:u16 start:u64 end:u64 limit:u64 stype_in stype_out ts_out
    meta_.dataset.assign(reinterpret_cast<const char*>(m), strnlen(reinterpret_cast<const char*>(m), 16));
    std::memcpy(&meta_.schema, m + 16, 2);
    std::memcpy(&meta_.start, m + 18, 8);
    std::memcpy(&meta_.end, m + 26, 8);
    meta_.ts_out = m[44] != 0;
    rpos_ += mlen;
  }

  std::FILE* f_ = nullptr;
  bool zstd_ = false, eof_ = false, frame_open_ = false;
  ZSTD_DCtx* ctx_ = nullptr;
  std::vector<uint8_t> in_;
  size_t in_pos_ = 0, in_len_ = 0;
  std::vector<uint8_t> buf_;
  size_t rpos_ = 0, wpos_ = 0;
  uint64_t skipped_ = 0;
  Metadata meta_;
};

}  // namespace mbo
