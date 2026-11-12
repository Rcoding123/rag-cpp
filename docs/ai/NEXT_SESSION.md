# Continuing the work: the prompt and the Claude Code agent architecture

Two things for whoever picks this up with Claude Code: the prompt to start
a session with, and how that session should organise its own agents
(orchestrator, subagents, workflows) to get the work done safely. The
trading-side plan itself is `ORDERFLOW.md` §7.

## 1. The prompt (paste into a new Claude Code session)

> Continue the ES order-flow trading work in the repository
> Rcoding123/rag-cpp, branch `research/rl` (`main` is fast-forwarded
> only after the full build and tests pass on the user's Windows server;
> everything runs there, nothing on Linux).
>
> **Read first, in this order:** `docs/ai/README.md`, `docs/ai/ORDERFLOW.md`
> (objectives, architecture, results OF1–OF10, honest status, ranked plan),
> `docs/ai/STATUS.md`, `docs/ai/CONVENTIONS.md`, `docs/ai/LESSONS.md`
> (32–54 are the trading lessons), `docs/ai/NEXT_SESSION.md` (this file:
> how to organise agents) and `experiments/research_synthesis.md`.
>
> **State on 2026-11-12:** iterations 1–3 of price-discovery work are
> done (OF10 v3 at `e229af4`: stationary activity model fixes the warm-up;
> the v3 `discovery` preset passes (a), (b), (d), naive (e) and (f1),
> fails passive quoting, and needs a 1,050 s warm-up for its meta-orders;
> A62–A66). Every OF2–OF9 episode ran at the legacy activity floor
> (ORDERFLOW.md §5). The RL reruns on `discovery` are done (OF11,
> `1a73bd6`): no edge; the 950M+ block is unused. OF12 compared v3 with
> the user's real ES; OF13 (`31f51c4`) added simulator v4, the opt-in
> `discovery_v4` preset matched to real ES: 25/25 gate rows G1–G8 and the
> efficiency gates pass on tuning, held-out and skeptic seeds at warm-up
> 1,050 s (A67–A69), but both oracles lose after costs, so "no RL edge"
> on v4 is close to true by construction (LESSONS 48). Run the live
> viewer alongside every RL run (A70; ORDERFLOW.md §3). OF14 (RL on v4,
> `6d678c6`): no edge (all 24 seeds negative). OF15: a leak-proof
> learning pipeline (`python/ragrl` over the C++ `VecScalpEnv`, bound as
> `_ragof` with `-DRAG_PYTHON=ON`; seed ledger enforced in
> `seed_ledger.hpp`; A71–A76) ran 8 agent families on v4 and v3, one
> sealed opening each: all 16 NO_EDGE, most ensembles flat (LESSONS 50).
> On both simulators no RL agent finds an edge after costs. WMI is broken
> on this server (bypassed in `ragrl`; repairing it is the user's call),
> and the 0xc0000374 heap crash is not root-caused. Next: ORDERFLOW.md §7.
>
> **Goal of this session:** the first item of ORDERFLOW.md §7: run the
> `ragrl` zoo on the user's recorded ES ticks through `VecReplayEnv`,
> walk-forward by day (train on earlier days, score once on later days),
> with the replay data registered as a sealed block in the ledger before
> any agent is trained on it. Needs the user's tick export; without it,
> do the opt-in simulator realism items
> (v5) of §7. Done when every config is scored once on later days,
> paired against flat and the 30 s delta rule, with orders/episode and
> the mean of seeds beside the headline, and any positive result goes to
> a fresh confirmation before it is claimed. The user's 24-thread server
> runs the long jobs (`experiments/of15_regen.sh`-style scripts through
> the kit); this session runs smoke runs.
>
> **Rules:**
> - Every new simulator behaviour is opt-in, so the defaults reproduce
>   OF1–OF9.
> - Use the seed blocks in NEXT_SESSION.md §4 and confirm any edge on
>   fresh seeds.
> - Regenerate result files from committed code, in the same command.
> - Tests stay warning-free (`-Werror`, MSVC too).
> - Update ORDERFLOW.md, EXPERIMENTS.md, STATUS.md, DECISIONS.md and
>   LESSONS.md with every result, negative results included, and
>   regenerate `handoff/` with `tools/make_context_pack.py`.
> - Report red flags (NEXT_SESSION.md §5) instead of celebrating them.
>
> **Organise the work as in NEXT_SESSION.md §2–3:** you orchestrate and own
> git; subagents design, implement, review adversarially and analyse; use
> the workflow template there for each work item.

## 2. Agent architecture (Claude Code side)

```
                         ┌───────────── Orchestrator (main session) ─────────────┐
                         │ plans, owns git (commit/push/CI/main), decides,        │
                         │ keeps docs current, holds the seed ledger and budget   │
                         └───────┬─────────────┬──────────────┬──────────────┬────┘
                                 │             │              │              │
             Scout (Explore) ────┘   Designers (2, independent)   Implementer    Experiment runner
             maps code + invariants  approaches, judged           one change,    background runs,
                                                                   tests         CSVs, logs
                                 │             │              │              │
                 Research (rag_pdf on research_corpus/)  Adversarial reviewer   Skeptical analyst
                 literature for a design question        fake-edge hunt, fixes  red flags, fresh-seed
                                                         clear bugs             confirmation
                                                                │
                                                     Doc writer (results → docs, handoff/)
```

| Role | Agent type | Owns (may edit) | Never |
|---|---|---|---|
| Orchestrator | main session | git, CI, `main`, the plan, docs review | delegates the decision to merge |
| Scout | `Explore` | nothing (read-only) | edits |
| Designer ×2 | `general-purpose`, read-only | a design note returned as text | edits code |
| Implementer | `general-purpose` (worktree if parallel with another implementer) | the files named in its brief, tests | commits, pushes, docs/ |
| Adversarial reviewer | `general-purpose` | minimal fixes to clear bugs, with a test each | redesigns; trusts the implementer's summary |
| Experiment runner | `general-purpose` | scratch output, result files | long runs in the foreground; editing code |
| Skeptical analyst | `general-purpose` | nothing | accepting a result without the §5 checks |
| Research | `general-purpose` | a short brief with citations from `research_corpus/` | inventing citations |
| Doc writer | `general-purpose` | `docs/`, `handoff/`, README | changing numbers it didn't get from result files |

Rules that kept the last session out of trouble:

- **One owner per file at a time.** Parallel implementers get disjoint
  files or worktrees. Subagents never commit. The orchestrator commits
  after reading the diff.
- **Briefs are self-contained:** the goal, the files, what was already
  ruled out, the success criteria, the seed blocks, the CPU budget (the
  cloud box has 4 vCPUs; heavy runs go to the user's server), and "don't
  edit docs/".
- **Every number in the docs comes from a results file** regenerated from
  committed code. A stale binary once produced a committed table
  (LESSONS 37).
- **Push less often during CI.** Each push cancels the running CI, and
  `main` moves only on a full green run.

## 3. Workflow template (one work item)

Pass to the Workflow tool (explicit opt-in: "use a workflow"); edit the
`ITEM` text per work item. It designs two ways, judges them, implements
the winner, reviews it adversarially from two angles, runs a smoke
experiment and has a skeptic check the numbers. The orchestrator then
commits, runs long jobs (or hands the commands to the user's server) and
updates the docs.

```js
export const meta = {
  name: 'orderflow-work-item',
  description: 'Design (2 approaches, judged) -> implement -> adversarial review (2 lenses) -> smoke run -> skeptical analysis',
  phases: [{ title: 'Design' }, { title: 'Implement' }, { title: 'Review' }, { title: 'Run' }, { title: 'Analyze' }],
}
const ITEM = args.item           // e.g. "makers re-price at once when value moves (ORDERFLOW.md plan item 1)"
const RULES = `Repo /home/user/rag-cpp, branch research/rl. Read docs/ai/ORDERFLOW.md and docs/ai/NEXT_SESSION.md first.
Do not commit or push; do not edit docs/. New simulator behaviour must be opt-in (defaults reproduce OF1-OF9).
Seed blocks: NEXT_SESSION.md section 4. CPU: at most 2 processes, short runs only. Warning-free with -Werror.`
const DESIGN = { type: 'object', properties: { approach: { type: 'string' }, files: { type: 'array', items: { type: 'string' } },
  success_metrics: { type: 'string' }, risks: { type: 'string' } }, required: ['approach', 'files', 'success_metrics'] }
const VERDICT = { type: 'object', properties: { winner: { type: 'number' }, why: { type: 'string' }, graft: { type: 'string' } }, required: ['winner', 'why'] }
const DONE = { type: 'object', properties: { files_changed: { type: 'array', items: { type: 'string' } }, tests: { type: 'string' },
  smoke: { type: 'string' }, commands_for_full_run: { type: 'string' } }, required: ['files_changed', 'tests', 'smoke'] }
const REVIEW = { type: 'object', properties: { findings: { type: 'array', items: { type: 'object', properties: {
  severity: { type: 'string' }, issue: { type: 'string' }, fixed: { type: 'boolean' } }, required: ['severity', 'issue', 'fixed'] } } }, required: ['findings'] }

phase('Design')
const designs = await parallel([0, 1].map(i => () => agent(`${RULES}\nDesign approach #${i + 1} for: ${ITEM}.
${i ? 'Favour the smallest change that meets the criteria.' : 'Favour fidelity to published market microstructure (use research_corpus via build/rag_pdf ask).'}
Return files to touch and measurable success criteria.`, { label: `design ${i + 1}`, phase: 'Design', schema: DESIGN })))
const verdict = await agent(`${RULES}\nJudge these designs for: ${ITEM}.\n${JSON.stringify(designs)}\nPick one; say what to graft from the other.`,
  { label: 'judge', phase: 'Design', schema: VERDICT })
phase('Implement')
const done = await agent(`${RULES}\nImplement for: ${ITEM}\nDesign: ${JSON.stringify(designs[verdict.winner - 1] || designs[0])}\nGraft: ${verdict.graft || 'nothing'}
Add tests (tests/orderflow_tests.cpp or tests/stylized_tests.cpp), build rag_tests, run the relevant filters.`, { label: 'implement', phase: 'Implement', schema: DONE })
phase('Review')
const reviews = await parallel(['fake-edge lens: hidden state or future info reaching strategies, P&L/fee accounting, seed leakage, oracle bound',
  'reproducibility lens: defaults unchanged (rebuild and diff OF4/OF2 result files), determinism, MSVC/GCC portability, warnings'].map(lens => () =>
  agent(`${RULES}\nAdversarially review the uncommitted change (git diff, git status) for: ${ITEM}. Summary: ${JSON.stringify(done)}.
Lens: ${lens}. Fix clear bugs minimally with a test; report everything.`, { label: lens.split(':')[0], phase: 'Review', schema: REVIEW })))
phase('Run')
const run = await agent(`${RULES}\nSmoke-run the change for: ${ITEM}: of_calibrate sim facts, the OF9 imbalance/oracle diagnostics, OF2 baselines (100 episodes).
Report numbers vs the success criteria: ${JSON.stringify(designs[verdict.winner - 1]?.success_metrics)}`, { label: 'smoke run', phase: 'Run' })
phase('Analyze')
const analysis = await agent(`${RULES}\nYou are a skeptic. Check these results against NEXT_SESSION.md section 5 (red flags) and say what must be confirmed on fresh seeds
before anything is claimed:\n${run}`, { label: 'skeptic', phase: 'Analyze' })
return { verdict, done, reviews, run, analysis }
```

## 4. Seed ledger (never reuse a block for a new choice)

Since OF15 the block table is enforced in code: `include/rag/orderflow/seed_ledger.hpp`
(block bases and sizes, sealed and privileged flags, the process-role guard;
`python/ragrl` mirrors it and never hard-codes bases), and every sealed
opening is appended to `build-avx512/rl/ledger_usage.jsonl`. This table
records what each block has been used for.

| Block | Use |
|---|---|
| run seed (1, 2, ...) | training streams (`Trainer` train_seed; ragrl `EpisodeSource.train(run_seed)`, run seeds in [1, 700M)); OF15 used run seeds 1–12 per config |
| 700,000,000+ | validation: champion selection; OF15: checkpoints on 700M+[0,200) (`neuralq_ref` [40,240)), cross-agent selection on 701M+[0,400), HPO objective 702M+[0,400) (not used); the OF15 audits replayed 700M/701M only |
| 900,000,000+ | sealed scalping test (OF2, OF3, OF9) — viewed on the discovery market (OF10 v1, v2 and an iteration-3 smoke: `of2_baselines 300`, the `of_train` smoke tests' episodes): confirm discovery-market edges once on 950M+; OF11 scored all 24 v3 nets on the first 300 at warm-up 1,050 s (pre-RL smoke: the first 100); OF13 tuning's (e) ran `of2_baselines` on the first 300 on v4. ragrl's seal refuses 900M+[0,100000) (viewed by OF2–OF14). **OF15:** 901M+[0,300) (indices 1,000,000–1,000,299) pre-registered for family `of15_v4` on v4 (9 slots, 8 used: one per agent; the HPO slot is free), 901M+[300,600) for `of15_v3` on v3 (8 of 8: full); each config opened once (16 ledger lines) |
| 910,000,000+ | sealed tick-scalping test (OF4, OF5, OF7) — viewed many times, also on the discovery market (`of4_tick_baselines 300`, OF10 v1, v2 and an iteration-3 smoke): confirm once on 950M+ |
| diagnostic simulator seeds 1–3 | OF10 tuning (`of_discovery`; v1/v2 on 2 h windows, v3 on 400 episodes + 16 sessions per seed; rule episodes 1,000,000 × seed + i): look as often as needed |
| diagnostic seeds 4–6 | OF10 held-out sanity check (θ* pre-registered on 1–3) — viewed several times (v1, v2, skeptic's 600 s run): no longer held out |
| diagnostic seeds 7–12 | OF10 v2 final check (6 h, once) — **spent** |
| diagnostic seeds 13–18 | OF10 iteration 3 final check (v3 frozen, once, acceptance agent) — **spent** |
| diagnostic seeds 19–21 | OF10 iteration 3 skeptic's fresh check — viewed: no longer fresh |
| **diagnostic seeds 22–24** | **reserved: the pre-RL warm-up check (1,050 s `ScalpEnv` warm-up, A63), once** |
| diagnostic seeds 25–27 | OF13 v4 tuning (efficiency probe): look as often as needed |
| diagnostic seeds 28–33 | OF13 v4 final check (once, acceptance agent) — **spent** |
| diagnostic seeds 34–36 | OF13 v4 skeptic's fresh check — viewed: no longer fresh |
| diagnostic seeds 37+ | free |
| 920,000,000+ | stress scenarios (OF6; OF11 on discovery v3 at warm-up 1,050 s) |
| 930,000,000+ | trading days (OF8; OF11 on discovery v3 at warm-up 1,050 s) |
| 940,000,000+ | pairing probes (separate streams) |
| 960,000,000+ | visualization only (`of_trace` → `tools/of_viz.py` episode viewer): look as often as you like, never select or score on it; OF12 exported sessions 960,012,001–012 for the real-vs-sim comparison, which OF13 then used as v4's tuning sessions; OF13 final sessions 960,013,001–012 (used once, with `of2_baselines` episodes 960,013,000,000 + i) and skeptic sessions 960,014,001–012 (episodes 960,014,000,000 + i): spent for gating v4; the live training viewer (`of_train_live`, A70) probes the net on this block; OF15: the ragrl live hook's probe episodes, the skeptic's fresh-episode rerun 960,015,000,000+[0,600) (both markets, four conditions) and quirk probe 960,015,001,000+[0,400) (viewed: no longer fresh), and `tools/of_zoo_trace.py` replays on 960M+[0,5) (look only) |
| **950,000,000+** | **fresh confirmation: unused (OF15 too); use once per claimed edge, then record it here. ragrl: `confirm --i-am-the-orchestrator --claim ID` only, taking the next free index after any earlier confirmation** |

## 5. Red flags (stop and diagnose before reporting a win)

- A learned policy beats the fair-value oracle (OF9).
- An oracle bound that barely trades (under ~2 lots/episode): "nothing
  beats it" then means nothing (OF10, LESSONS 43).
- Facts checked on windows the agents never see: compare trades/s and
  1-min sd in the episode windows themselves (OF10 warm-up finding,
  LESSONS 40).
- Annualized Sharpe above ~5, or no losing day, for a 1-lot directional
  strategy (OF8: ~80 meant a too-easy simulator). In ragrl this is R1:
  hard only when the Sharpe's lower 2-SE bound is above 5 (A77; about
  t > 2.9 at 300 episodes), a warning (R1v) below 2 orders/episode (A74)
  or when only the point estimate is above 5 (LESSONS 51).
- An ensemble headline that places ~0 orders: its "no edge" is vacuous;
  read the mean of seeds and orders/episode (OF15, LESSONS 50).
- A naive rule profits a lot after costs (naive quoting winning every
  episode exposed the queue model, OF4).
- A validation lead that shrinks on sealed seeds (OF7: +1.0 → +0.1).
- A feature that correlates strongly with hidden state
  (`fair_value()`); check with a probe like the OF9 imbalance diagnostic.
- Results that change between compilers or with the build (argument
  evaluation order, `-march=native` / FMA).
- Memory growing with episodes: check RSS on long runs (the 3.8 GB tile
  tables).

## 6. Where the work runs

- **Cloud Claude Code session:** code, tests, CI, smoke runs, docs.
- **The user's Windows server** (Ryzen 9 9900X, 24 threads, AVX-512,
  `C:\rag-cpp`): long training and calibration (`of_train`,
  `of_tick_train`, `of_calibrate`, `of_backtest`); see `docs/WINDOWS.md`.
  After `git pull`, rebuild with
  `cmake --build --preset windows-avx512-release --parallel`.
- **GPUs:** not useful at the current network size (2 × 64). They are
  worth renting only with much larger models (e.g. sequence models over
  raw book data), which would need a GPU framework fed by this simulator.
  OF15 measured it for the SB3 zoo: PPO-family training is 61–80% C++
  simulator on v4, the DQN family update-bound on 64×64 MLPs (per-update
  overhead, not arithmetic): no GPU now (A75).
