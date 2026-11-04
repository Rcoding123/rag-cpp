# Hand-off pack for an AI assistant

This folder describes rag-cpp for a language model (or a person) taking over
the project. It's written to be pasted into a model's context, either as
separate files or as the single bundle produced by:

```bash
python3 tools/make_context_pack.py                 # docs only  -> dist/context_pack_docs.md
python3 tools/make_context_pack.py --with-source   # docs + all headers, tests, examples -> dist/context_pack_full.md
```

## Reading order

1. `OBJECTIVES.md`: what the project is for, success criteria, non-goals,
   and the requests it grew from.
2. `PROJECT_MAP.md`: what exists, where, and how it's built.
3. `SPECS.md`: toolchain, libraries (with versions), build options, the
   measurement machine, the CI matrix.
4. `API.md`: how to use it. Every snippet compiles (`tests/api_examples_tests.cpp`).
5. `RAG_ENGINE.md`: engine internals, invariants, file format.
6. `../agent.md`: the self-learning agent's architecture.
7. `TASK_SPECS.md`: exact task dynamics, rewards, optima, learner update
   rules and metrics.
8. `LESSONS.md`: the experiments' cross-cutting lessons, one paragraph each.
9. `EXPERIMENTS.md`: what was tried with the agent and what happened.
10. `DECISIONS.md`: why things are the way they are, with evidence.
11. `CONVENTIONS.md`: rules to follow when changing code.
12. `STATUS.md`: limitations and next steps.

## Ground rules for an AI working on this code

* **Measure, don't assume.** Every number in these docs came from a run.
  If you change code, re-measure before quoting.
* **Run the checks.** Run the full test matrix (CONVENTIONS.md) before
  claiming something works. CI runs 8 jobs.
* **Keep the guarantees.** Hot paths must not allocate. Batched queries are
  bit-identical to single ones. Loaders validate every field. Tests enforce
  all three.
* **Document negative results.** Record failed experiments in
  EXPERIMENTS.md; they're as useful as the successes.
