# Continuing the work: the prompt and the Claude Code agent architecture

Two things for whoever picks this up with Claude Code: the prompt to start
a session with, and how that session should organise its own agents
(orchestrator, subagents, workflows) to get the work done safely. The
trading-side plan itself is `ORDERFLOW.md` §7.

## 1. The prompt (paste into a new Claude Code session)

> Continue the ES order-flow trading work in the repository
> Rcoding123/rag-cpp, branch `research/rl` (`main` is fast-forwarded
> only after all 9 CI jobs pass).
>
> **Read first, in this order:** `docs/ai/README.md`, `docs/ai/ORDERFLOW.md`
> (objectives, architecture, results OF1–OF9, honest status, ranked plan),
> `docs/ai/STATUS.md`, `docs/ai/CONVENTIONS.md`, `docs/ai/LESSONS.md`
> (32–39 are the trading lessons), `docs/ai/NEXT_SESSION.md` (this file:
> how to organise agents) and `experiments/research_synthesis.md`.
>
> **Goal of this session:** plan item 1 in ORDERFLOW.md §7: fix slow price
> discovery in `include/rag/orderflow/simulator.hpp`. Makers must re-price
> at once when value moves (quote jumps, not rate tilts), as an opt-in
> setting so earlier results reproduce. Done when, on 3 seeds × 2 h:
> - the hidden value gap no longer predicts the next 30 s return (it does
>   at 0.58–0.65 today);
> - book imbalance predicts only the next tick or two;
> - the markout curve is flat after a few seconds;
> - the stylized facts (`of_calibrate sim`) stay ES-like;
> - naive rules are near breakeven after costs;
> - no learned policy beats the fair-value oracle.
>
> Then re-run the scalping baselines and RL (OF2, OF3, OF6, OF8) on the
> fixed market. Use `of_train` for training: the user's 24-thread server
> runs the long jobs, and this session runs only smoke runs. If the user
> provides an ES tick CSV, do plan items 2–3 (calibration, backtest)
> next.
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

| Block | Use |
|---|---|
| run seed (1, 2, ...) | training streams (`Trainer` train_seed) |
| 700,000,000+ | validation: champion selection |
| 900,000,000+ | sealed scalping test (OF2, OF3, OF9) |
| 910,000,000+ | sealed tick-scalping test (OF4, OF5, OF7) — viewed many times: confirm elsewhere |
| 920,000,000+ | stress scenarios (OF6) |
| 930,000,000+ | trading days (OF8) |
| 940,000,000+ | pairing probes (separate streams) |
| **950,000,000+** | **fresh confirmation: unused; use once per claimed edge, then record it here** |

## 5. Red flags (stop and diagnose before reporting a win)

- A learned policy beats the fair-value oracle (OF9).
- Annualized Sharpe above ~5, or no losing day, for a 1-lot directional
  strategy (OF8: ~80 meant a too-easy simulator).
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
