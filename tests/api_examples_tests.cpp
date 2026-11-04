// api_examples_tests.cpp — every code example in docs/ai/API.md, compiled
// and run. If an API changes, this file stops compiling and the docs get
// fixed along with the code.
#include <gtest/gtest.h>
#include <unistd.h>

#include <filesystem>
#include <string>
#include <vector>

#include "rag.hpp"
#include "rag/agent/execution_dp.hpp"
#include "rag/agent/model_based.hpp"
#include "rag/agent/trainer.hpp"
#include "rag/text_clean.hpp"

namespace {
std::filesystem::path tmp(const char* n) {
  return std::filesystem::temp_directory_path() / (std::string(n) + "-" + std::to_string(::getpid()));
}
}  // namespace

// --- API.md §1: index and query ----------------------------------------------
TEST(ApiExamples, IndexAndQuery) {
  using Engine = rag::RagEngine<rag::HashingEmbedder<384>>;
  rag::EngineConfig cfg;
  cfg.max_chunks = 100'000;                       // capacity; memory is committed lazily
  cfg.chunker = {.window_tokens = 128, .overlap_tokens = 32};
  cfg.dense_mode = rag::DenseMode::IvfInt8;       // Exact | Int8Rerank | Ivf | IvfInt8
  Engine engine(cfg);

  engine.ingest("Iceberg orders show a small visible size and refill from a hidden reserve.", /*doc_id=*/1);
  engine.ingest("VWAP algorithms follow the intraday volume curve.", 2);
  engine.commit();  // freeze BM25 (+ IVF) into a new snapshot; safe while queries run

  auto ctx = engine.make_context();  // one per query thread; reuse it
  const auto r = engine.query("how do iceberg orders work?", /*k=*/3, ctx);
  ASSERT_FALSE(r.hits.empty());
  const rag::ChunkView top = engine.chunk(r.hits[0].id);
  EXPECT_EQ(top.doc_id, 1u);
  EXPECT_NE(r.prompt.prompt.find("### Question"), std::string_view::npos);  // ready for an LLM
}

// --- API.md §2: page-aware ingestion and citations ---------------------------
TEST(ApiExamples, PagesAndCitations) {
  rag::RagEngine<rag::HashingEmbedder<384>> engine(rag::EngineConfig{.max_chunks = 1024});
  const std::string text = "Page one talks about spreads.\n\nPage two talks about maker rebates.";
  const std::uint32_t page_starts[] = {0, static_cast<std::uint32_t>(text.find("Page two"))};
  engine.ingest(text, 7, rag::IngestOptions{.source = "fees.pdf", .page_starts = page_starts});
  engine.commit();
  auto ctx = engine.make_context();
  const auto r = engine.query("maker rebates", 1, ctx);
  EXPECT_NE(r.prompt.prompt.find("(fees.pdf, p"), std::string_view::npos);  // "(fees.pdf, p. 1)" or "pp. 1-2"
}

// --- API.md §3: save, inspect, load ------------------------------------------
TEST(ApiExamples, Persistence) {
  using Engine = rag::RagEngine<rag::HashingEmbedder<384>>;
  const rag::EngineConfig cfg{.max_chunks = 1024};
  Engine engine(cfg);
  engine.ingest("Kill switches cancel all resting orders.", 1);
  engine.commit();
  const auto path = tmp("api-index");
  engine.save(path);                                   // crash-safe: tmp file, fsync, rename
  const rag::IndexInfo info = Engine::inspect(path);   // header only
  Engine::LoadReport report;
  auto loaded = Engine::load(path, cfg, {}, &report);  // throws on damage or mismatch
  EXPECT_EQ(info.chunks, loaded->size());
  EXPECT_TRUE(report.reused_indexes);                  // same parameters: saved indexes reused
  std::filesystem::remove(path);
}

// --- API.md §4: batched queries ----------------------------------------------
TEST(ApiExamples, BatchedQueries) {
  using Engine = rag::RagEngine<rag::HashingEmbedder<384>>;
  Engine engine(rag::EngineConfig{.max_chunks = 1024});
  engine.ingest("alpha beta gamma", 1);
  engine.ingest("delta epsilon zeta", 2);
  engine.commit();
  std::vector<std::string_view> queries = {"alpha", "zeta", "gamma delta"};
  std::vector<Engine::QueryContext> ctxs;
  for (std::size_t i = 0; i < queries.size(); ++i) ctxs.push_back(engine.make_context());
  std::vector<Engine::QueryResult> out(queries.size());
  engine.query_batch(queries, /*k=*/2, ctxs, out);  // identical to query(), one memory pass per 8
  EXPECT_EQ(engine.chunk(out[1].hits[0].id).doc_id, 2u);
}

// --- API.md §5: text cleanup without Poppler ----------------------------------
TEST(ApiExamples, TextCleanup) {
  std::vector<std::string> pages = {"Header\nThe \xEF\xAC\x81rst page", "Header\nsecond page"};
  const auto cleaned = rag::text::clean_pages(pages);  // ligatures, headers, hyphenation, reflow
  EXPECT_EQ(cleaned[0], "The first page");
  const auto joined = rag::text::join_pages(cleaned);  // text + page_starts for IngestOptions
  EXPECT_EQ(joined.page_starts.size(), 2u);
}

// --- API.md §6: train an agent -----------------------------------------------
TEST(ApiExamples, TrainAnAgent) {
  using namespace rag::agent;
  ExecutionEnv env{ExecutionParams{}};
  AgentConfig<3> cfg;  // 3 = ExecutionEnv::kStateDims
  cfg.eps_decay_episodes = 2000;
  Agent<ExecutionEnv> agent(env, cfg);
  Trainer<ExecutionEnv> trainer(agent, env, {.eval_every = 500, .eval_episodes = 200});
  const TrainReport report = trainer.run(4000);
  // Deploy the champion; the learner keeps learning separately.
  const Action a = trainer.deployed_act(ExecutionObs{.t = 5, .inventory = 60, .signal = -0.1});
  EXPECT_LT(a, ExecutionEnv::kNumActions);
  // Measure against the exact optimum on unseen scenarios (paired vs TWAP).
  const ExecutionDP dp{ExecutionParams{}};
  auto twap = [](const ExecutionObs&) { return ExecutionEnv::kTwapAction; };
  const EvalStats mine = compare(env, [&](const ExecutionObs& o) { return trainer.deployed_act(o); }, twap, 500, 1);
  const EvalStats best = compare(env, [&](const ExecutionObs& o) { return dp.act(o); }, twap, 500, 1);
  EXPECT_GT(best.mean, mine.mean - 1.0);  // the DP is (nearly) unbeatable
  EXPECT_FALSE(report.evals.empty());
}

// docs/ai/API.md section 7.
TEST(ApiExamples, DeploySafelyWithABaselineGuard) {
  using namespace rag::agent;
  ExecutionEnv env{ExecutionParams{}};
  AgentConfig<3> cfg;
  cfg.eps_decay_episodes = 2000;
  Agent<ExecutionEnv> agent(env, cfg);
  auto twap = [](const ExecutionObs&) { return ExecutionEnv::kTwapAction; };
  Trainer<ExecutionEnv> guarded(agent, env, {.eval_every = 500, .eval_episodes = 300});
  guarded.set_baseline(twap);
  EXPECT_FALSE(guarded.learned_policy_deployed());  // nothing has beaten the baseline yet
  EXPECT_EQ(guarded.deployed_act(ExecutionObs{.t = 0, .inventory = 100, .signal = 0.05}), ExecutionEnv::kTwapAction);
  guarded.run(4000);
  const bool learned = guarded.learned_policy_deployed();
  const Action b = guarded.deployed_act(ExecutionObs{.t = 0, .inventory = 100, .signal = 0.05});
  EXPECT_LT(b, ExecutionEnv::kNumActions);
  if (!learned) {
    EXPECT_EQ(b, ExecutionEnv::kTwapAction);
  }
}

// docs/ai/API.md section 8.
TEST(ApiExamples, ModelBasedAgent) {
  using namespace rag::agent;
  ExecutionEnv env{ExecutionParams{}};
  ExecutionParams structure;
  ModelBasedExecutionAgent mb(structure, {.cost_model = CostModel::Shrunk, .seed = 1});
  Rng rng(42);
  train_model_based(mb, env, /*episodes=*/50, rng);
  mb.replan();
  const ModelEstimate est = mb.current_estimate();
  EXPECT_NEAR(est.alpha, 1.0, 0.2);
  EXPECT_NEAR(est.impact, 0.01, 0.003);
  const Action c = mb.act(ExecutionObs{.t = 5, .inventory = 60, .signal = -0.1}, /*explore=*/false);
  EXPECT_LT(c, ExecutionEnv::kNumActions);
}

// docs/ai/API.md section 9.
TEST(ApiExamples, CheckAModelAndWarmStartFromIt) {
  using namespace rag::agent;
  ExecutionEnv env{ExecutionParams{}};
  ModelBasedExecutionAgent mb(ExecutionParams{}, {.cost_model = CostModel::Shrunk, .seed = 1});
  Rng rng(42);
  train_model_based(mb, env, 50, rng);
  mb.replan();
  const ModelCheck check = mb.check_model(env, /*episodes=*/100, /*seed=*/6'000'000);
  const bool model_ok = check.z < 4.0;
  EXPECT_TRUE(model_ok) << check.z;
  const ExecutionDP& model = *mb.plan();
  AgentConfig<3> warm_cfg;
  warm_cfg.eps_start = 0.1;
  Agent<ExecutionEnv> warm(env, warm_cfg);
  EXPECT_TRUE(warm.distill(ExecutionEnv(model.params()),
                           [&](const ExecutionObs& o, Action a) { return model.q_value(o.t, o.inventory, o.signal, a); },
                           /*episodes=*/2000, /*seed=*/7));
  Trainer<ExecutionEnv> fine(warm, env, {.eval_every = 500, .eval_episodes = 300, .detect_drift = false});
  fine.set_baseline([&](const ExecutionObs& o) { return mb.act(o, false); });
  fine.run(2000);
  auto twap = [](const ExecutionObs&) { return ExecutionEnv::kTwapAction; };
  EXPECT_GT(compare(env, [&](const ExecutionObs& o) { return fine.deployed_act(o); }, twap, 300, 1).mean, 5.0);
}
