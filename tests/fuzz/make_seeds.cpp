// make_seeds — writes valid index payloads (file minus trailing checksum)
// for every dense mode, as a starting corpus for fuzz_load.
#include <cstdio>
#include <fstream>
#include <iterator>
#include <vector>

#include "fuzz_common.hpp"

using namespace rag;
using namespace rag::fuzz;

int main(int argc, char** argv) {
  if (argc < 2) { std::fprintf(stderr, "usage: make_seeds <out-dir>\n"); return 2; }
  const std::filesystem::path out = argv[1];
  std::filesystem::create_directories(out);
  const char* docs[] = {"alpha beta gamma delta epsilon zeta eta theta iota kappa",
                        "order book imbalance predicts the next mid price move",
                        "the of and", "t0w1 t0w2 t1w3 f4 f5 f6 f7 f8 f9 f10 f11 f12 f13 f14"};
  const DenseMode modes[] = {DenseMode::Exact, DenseMode::Int8Rerank, DenseMode::Ivf, DenseMode::IvfInt8};
  int idx = 0;
  for (int m = 0; m < 4; ++m) {
    for (int ndocs : {0, 1, 4}) {
      Engine e(small_config(modes[m]));
      for (int d = 0; d < ndocs; ++d) e.ingest(docs[d], static_cast<std::uint32_t>(d));
      e.commit();
      if (ndocs == 4) e.ingest("uncommitted tail chunk", 9);
      const auto tmp = out / "tmp.idx";
      e.save(tmp);
      std::ifstream f(tmp, std::ios::binary);
      std::vector<char> b{std::istreambuf_iterator<char>(f), {}};
      std::filesystem::remove(tmp);
      b.resize(b.size() - 8);  // drop the checksum: the fuzzer appends a fresh one
      std::ofstream o(out / ("seed-" + std::to_string(idx++)), std::ios::binary);
      o.write(b.data(), static_cast<std::streamsize>(b.size()));
    }
  }
  std::printf("wrote %d seeds to %s\n", idx, out.c_str());
}
