# API cheat sheet

Every snippet here is compiled and run by `tests/api_examples_tests.cpp`. If
you change an API, update both files.

## 1. Index and query

```cpp
#include "rag.hpp"
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
const rag::ChunkView top = engine.chunk(r.hits[0].id);   // text, doc_id, ordinal, source, page_first, page_last
std::string_view prompt = r.prompt.prompt;               // ready for an LLM
```

`QueryResult` fields:
* `hits`: fused ranking of `Hit{score, id}`, best first;
* `dense`, `lexical`: each retriever's own list;
* `prompt`: a `PromptResult` with `.prompt`, `.chunks_used`,
  `.context_tokens` and `.truncated`;
* `snapshot_version`.

Every span points into `ctx` and stays valid until the next query on that
context.

`EngineConfig` fields, in declaration order (designated initializers must
follow it):

| Field | Default | Meaning |
|---|---|---|
| `max_chunks` | 65,536 | capacity |
| `chunker` | `{128, 32}` | `window_tokens`, `overlap_tokens` |
| `bm25` | `{1.2, 0.75}` | `k1`, `b` |
| `rrf` | `{60, 1, 1}` | `k0`, `w_dense`, `w_lexical` |
| `prompt` | `{1024, system}` | `max_context_tokens`, system text |
| `dense_mode` | `Exact` | dense retrieval mode |
| `hybrid` | `true` | `false` = dense only |
| `dense_min_score` | `0` | strict cosine floor |
| `retrieve_depth` | 32 | per-retriever list depth fed to RRF |
| `rerank_candidates` | 128 | shortlist size for int8 modes |
| `nprobe` | 8 | IVF lists probed per query |
| `ivf` | `{nlist 0 = auto, train_iters 10, ...}` | IVF parameters |
| `prompt_capacity` | 32 KiB | prompt buffer size |

## 2. Page-aware ingestion and citations

```cpp
const std::string text = "Page one talks about spreads.\n\nPage two talks about maker rebates.";
const std::uint32_t page_starts[] = {0, static_cast<std::uint32_t>(text.find("Page two"))};
engine.ingest(text, 7, rag::IngestOptions{.source = "fees.pdf", .page_starts = page_starts});
// prompts cite "[n] (fees.pdf, p. 2)" or "(fees.pdf, pp. 1-2)" for chunks spanning a page break
```

For real PDFs (Poppler; link `rag_pdf_support`):

```cpp
#include "rag/pdf.hpp"
auto res = rag::pdf::ingest_pdf(engine, "report.pdf", /*doc_id=*/7);  // res.pages, res.chunks, res.empty_pages
```

CLI: `rag_pdf index out.ragidx docs/ a.pdf`, then
`rag_pdf ask out.ragidx "question" -k 5 --prompt`. It also indexes text and
source files verbatim (`.md .txt .hpp .cpp .py ...`, cited by chunk), so
`rag_pdf index project.ragidx .` makes this repository searchable.

## 3. Save, inspect, load

```cpp
engine.save(path);                                   // crash-safe: tmp file, fsync, rename
const rag::IndexInfo info = Engine::inspect(path);   // header only: version, dim, chunks, text_bytes
Engine::LoadReport report;
auto loaded = Engine::load(path, cfg, {}, &report);  // unique_ptr; throws io::FormatError / invalid_argument
// report.reused_indexes: true if the saved BM25/IVF matched cfg and were reused
```

## 4. Batched queries

```cpp
std::vector<std::string_view> queries = {"alpha", "zeta", "gamma delta"};
std::vector<Engine::QueryContext> ctxs;              // one context per query in flight
for (std::size_t i = 0; i < queries.size(); ++i) ctxs.push_back(engine.make_context());
std::vector<Engine::QueryResult> out(queries.size());
engine.query_batch(queries, /*k=*/2, ctxs, out);     // bit-identical to query(); one memory pass per 8
```

## 5. Text cleanup (no Poppler needed)

```cpp
#include "rag/text_clean.hpp"
const auto cleaned = rag::text::clean_pages(pages);  // ligatures, headers/footers, hyphenation, reflow
const auto joined = rag::text::join_pages(cleaned);  // joined.text + joined.page_starts
```

## 6. Train an agent

```cpp
#include "rag/agent/execution_dp.hpp"
#include "rag/agent/trainer.hpp"
using namespace rag::agent;

ExecutionEnv env{ExecutionParams{}};
AgentConfig<3> cfg;                 // 3 = ExecutionEnv::kStateDims
cfg.eps_decay_episodes = 2000;
Agent<ExecutionEnv> agent(env, cfg);
Trainer<ExecutionEnv> trainer(agent, env, {.eval_every = 500, .eval_episodes = 200});
const TrainReport report = trainer.run(4000);   // report.evals, .drift_alarms, .drift_rejected, .bank_recalls, .returns
const Action a = trainer.deployed_act(ExecutionObs{.t = 5, .inventory = 60, .signal = -0.1});

// Measure against the exact optimum on unseen scenarios, paired vs TWAP:
const ExecutionDP dp{ExecutionParams{}};
auto twap = [](const ExecutionObs&) { return ExecutionEnv::kTwapAction; };
EvalStats mine = compare(env, [&](const ExecutionObs& o) { return trainer.deployed_act(o); }, twap, 500, 1);
EvalStats best = compare(env, [&](const ExecutionObs& o) { return dp.act(o); }, twap, 500, 1);
// share of the optimal gain = mine.mean / best.mean
```

`AgentConfig` fields, in order:

| Field | Default | Meaning |
|---|---|---|
| `learner` | `TileQ` | or `Episodic`, `ActorCritic`, `NeuralQ` |
| `eps_start`, `eps_end`, `eps_decay_episodes` | 1.0, 0.02, 1500 | exploration schedule |
| `bins` | 0 = 8 per dimension | tile grid |
| `tilings` | 8 | number of offset grids |
| `alpha`, `alpha_min`, `alpha_decay_episodes` | 0.1, 0.003, 3000 | annealed step size |
| `gamma` | 1.0 | discount |
| `init_q` | 0 | initial Q-values |
| `double_q` | false | double Q-learning |
| `replay_capacity`, `replay_per_step` | 100,000, 4 | experience replay |
| `n_step` | 3 | n-step returns (1 = Q-learning, ≥ horizon = Monte Carlo) |
| `actor_alpha`, `critic_alpha`, `temperature` | 0.03, 0.1, 1.0 | `ActorCritic` step sizes and softmax temperature |
| `ac_decay_episodes` | 10,000 | `ActorCritic` annealing half-life (3,000 is too fast, E5) |
| `nn_hidden`, `nn_lr`, `nn_batch` | 64, 1e-3, 32 | `NeuralQ`: units per hidden layer (two layers), Adam step, minibatch |
| `nn_target_sync`, `nn_warmup` | 2,000, 2,000 | `NeuralQ`: updates between target-network copies (0 = none), transitions before learning starts |
| `ac_reexplore_temperature` | 0 | `ActorCritic` re-exploration: 0 = ε-uniform actions; > 0 = sample the policy at a raised temperature (task-dependent, E13) |
| `memory` | | `EpisodicParams` |
| `kernel_width` | 0 = 10% of each range | episodic similarity scale |
| `seed` | 1 | RNG seed |

`TrainConfig` fields, in order:

| Field | Default | Meaning |
|---|---|---|
| `eval_every`, `eval_episodes`, `eval_seed` | 250, 300, 9,000,000 | held-out evaluation |
| `train_seed` | 1 | training scenario stream |
| `keep_champion` | true | deploy the best policy seen |
| `detect_drift` | true | drift detection on/off |
| `drift` | | `DriftParams`: lag 50, allowance 0.3, threshold 16, baseline_rate 0.001, warmup 1500, two_sided false |
| `drift_after_episodes` | 0 = after the ε schedule | when drift checks start |
| `reexplore_epsilon`, `reexplore_episodes` | 0.5, 5000 | response to drift |
| `forget_on_drift` | true | clear replay on drift |
| `policy_bank`, `bank_capacity` | true, 8 | bank past champions; recall one that fits a returning regime |
| `recall_epsilon`, `recall_episodes` | 0.05, 500 | light exploration after a recall |

`ExecutionParams` (the market) fields that change what is optimal:

| Field | Default | Meaning |
|---|---|---|
| `lots`, `horizon` | 100, 20 | inventory to sell, steps to sell it in |
| `impact`, `impact_exponent`, `fixed_cost` | 0.01, 2.0, 0 | cost of selling q lots in a step: impact·q^exponent + fixed (if q > 0) |
| `terminal_impact` | 0.05 | leftover inventory is dumped at terminal_impact·rem² (+ fixed) |
| `risk_aversion` | 0 | λ: reward −= λ·noise_sd²·I² per step (mean-variance; E4) |
| `permanent_impact` | 0 | γ: each lot sold lowers the price for the rest of the episode; held inventory loses γ·q·I' (E7) |
| `alpha`, `signal_rho`, `signal_sd`, `noise_sd` | 1, 0.7, 0.1, 0.05 | price move = alpha·signal + noise; signal is AR(1) |
| `sizes` | {0, 0.5, 1, 2, 3} | action k sells sizes[k] × the TWAP rate |

`ExecutionDP{params}` is exact for every combination above;
`ExecutionDP{params, trade_costs, dump_costs}` takes arbitrary cost tables.
`dp.policy_value(policy)` is the exact expected value of any policy under
that model (backward induction with the policy's action).

## 7. Deploy safely: a baseline guard

```cpp
Trainer<ExecutionEnv> guarded(agent, env, {.eval_every = 500, .eval_episodes = 300});
guarded.set_baseline(twap);              // deploy TWAP until the champion beats it by 2 standard errors
guarded.run(4000);
const bool learned = guarded.learned_policy_deployed();   // false = the baseline is live
const Action b = guarded.deployed_act(ExecutionObs{.t = 0, .inventory = 100, .signal = 0.05});
```

## 8. Model-based agent (learn the market, then plan)

```cpp
#include "rag/agent/model_based.hpp"
ExecutionParams structure;               // only lots, horizon and sizes are read; the rest is learned
ModelBasedExecutionAgent mb(structure, {.cost_model = CostModel::Shrunk, .seed = 1});
Rng rng(42);
train_model_based(mb, env, /*episodes=*/50, rng);   // act, observe, end_episode, replan on a schedule
mb.replan();
const ModelEstimate est = mb.current_estimate();      // impact, terminal_impact, alpha, rho, signal_sd, holding
const Action c = mb.act(ExecutionObs{.t = 5, .inventory = 60, .signal = -0.1}, /*explore=*/false);
```

`ModelBasedExecutionAgent::Config` fields, in order: `explore_episodes` 10,
`eps` 0.01, `explore_lo` 0, `explore_hi` 4, `replan_every` 20, `replan_max_gap` 500, `dp_grid` 61,
`cost_model` (`Quadratic`, `PerSize`, or `Shrunk`), `shrink_steps` 30,
`learn_holding_cost` true, `learn_permanent_impact` true, `forget` 1.0,
`detect_drift` false, `drift` (lag 10, allowance 0.3, threshold 10,
baseline_rate 0.01, warmup 50), `drift_forget` 0, `seed` 1. Use `Shrunk`
when the cost curve's shape is unknown (E1c). In a market that may change,
set `detect_drift = true` (E9); `drift_alarms()` lists when it fired.

## 9. Check a model, and warm-start a learner from it

```cpp
// Is the learned model right? Paired predicted vs realized gain over TWAP.
const ModelCheck check = mb.check_model(env, /*episodes=*/100, /*seed=*/6'000'000);
const bool model_ok = check.z < 4.0;       // a wrong model is optimistic: z >> 4

// Distill the model's plan into a tile-coded learner (states visited in the
// model's own simulator: no real episodes spent), then fine-tune it on real
// rewards with the plan as the guard's baseline.
const ExecutionDP& model = *mb.plan();
AgentConfig<3> warm_cfg;
warm_cfg.eps_start = 0.1;
Agent<ExecutionEnv> warm(env, warm_cfg);
warm.distill(ExecutionEnv(model.params()),
             [&](const ExecutionObs& o, Action a) { return model.q_value(o.t, o.inventory, o.signal, a); },
             /*episodes=*/2000, /*seed=*/7);
Trainer<ExecutionEnv> fine(warm, env, {.eval_every = 500, .eval_episodes = 300, .detect_drift = false});
fine.set_baseline([&](const ExecutionObs& o) { return mb.act(o, false); });
fine.run(2000);                            // deployed: never worse than the plan
```

`evaluate()`, `compare()` and `EvalStats` are in `rag/agent/evaluate.hpp`
(included by `trainer.hpp` and `model_based.hpp`), with
`episode_returns(env, policy, n, seed)` (every episode's return, same
scenarios) and `cvar(returns, alpha)` (mean of the worst `alpha` fraction)
for tails (E14).

## 10. Off-policy evaluation from logs

```cpp
#include "rag/agent/ope.hpp"
// Logged under the behaviour policy, with its propensity b(a | obs) > 0:
std::vector<LoggedEpisode<ExecutionObs>> logs = /* {obs, action, reward, propensity} per step */ {};
auto target = [&](const ExecutionObs& o) { return dp.act(o); };
const auto values = mb.plan()->evaluate_policy(target);   // a fitted model's V/Q for the target
const OpeEstimates e = off_policy_estimates(
    logs, target,
    [&](const ExecutionObs& o, Action a) { return values.q(o.t, o.inventory, o.signal, a); },
    [&](const ExecutionObs& o) { return values.value(o.t, o.inventory, o.signal); },
    /*horizon=*/20);
// e.pdis, e.wpdis, e.dr; trust them only if e.min_ess is large and e.uncovered_steps == 0.
const double direct = values.start_value();               // the model's own estimate (DM)
```

At horizon 20 the importance-sampling estimates are unusable unless the
target is close to the behaviour (E10); rely on the model, and check it.

## 11. Another task: market making

```cpp
#include "rag/agent/market_making.hpp"
const MarketMakingParams mp;                // quote distances, fill model, risk penalty
MarketMakingEnv mm(mp);
AgentConfig<2> mm_cfg;                       // 2 = MarketMakingEnv::kStateDims: (t, inventory)
mm_cfg.learner = LearnerKind::ActorCritic;
mm_cfg.eps_decay_episodes = 5000;
Agent<MarketMakingEnv> maker(mm, mm_cfg);
Trainer<MarketMakingEnv> mm_trainer(maker, mm, {.eval_every = 1000, .eval_episodes = 500, .detect_drift = false});
mm_trainer.run(5000);
const MarketMakingDP mm_dp(mp);              // the exact optimum, for scoring
const Action quote = mm_trainer.deployed_act(MarketMakingObs{.t = 25, .q = 4});
// long 4 units: expect a wide bid and a tight ask (mm.bid_distance(quote) > mm.ask_distance(quote))
```

Any type satisfying the `Environment` concept works the same way.

## 12. Tune and analyse (optional: NLopt + Eigen)

```cpp
#include "rag/agent/tuning.hpp"    // NLopt
#include "rag/agent/analysis.hpp"  // Eigen
const std::vector<tuning::Param> space = {
    {"alpha", 0.001, 1.0, /*log_scale=*/true}, {"bins", 2, 10, false, /*integer=*/true}};
auto score = [&](const std::vector<double>& v) { /* train with v on fixed seeds, return the mean share */ return 0.0; };
const tuning::Result r = tuning::maximize_global(space, {0.1, 8}, score, /*global=*/80, /*local=*/40, /*seed=*/1);
// r.best, r.best_score, r.history (every trial). Validate r.best on held-out seeds:
const Interval ci = bootstrap_mean(held_out_scores, 10000, 1);   // evaluate.hpp, no dependency
const auto sens = analysis::quadratic_sensitivity(rows, scores);  // which parameters matter
const Eigen::VectorXd b = analysis::least_squares(X, y, weights); // rank-revealing weighted LS
```

`examples/tune_agent.cpp` (target `tune_agent`) does all of this for
TileQ on ExecutionEnv. `InnovationControlledEnv` (`variance_reduction.hpp`)
wraps ExecutionEnv with a control variate for the price noise (E17).

A new environment only needs to satisfy the `Environment` concept: an
`Observation` type, `kNumActions`, `kStateDims`, `kLo`/`kHi`, and
`reset(rng)`, `step(action, rng)` and `features(obs)`.
