# Hand-off pack for an AI assistant

This folder describes rag-cpp for a language model (or a person) taking over
the project. It's written to be pasted into a model's context, either as
separate files or as the single bundle produced by:

```bash
python3 tools/make_context_pack.py                 # docs only  -> dist/context_pack_docs.md
python3 tools/make_context_pack.py --with-source   # docs + all headers, tests, examples -> dist/context_pack_full.md
```

## Reading order

1. `PROJECT_MAP.md`: what exists, where, and how it's built.
2. `API.md`: how to use it. Every snippet compiles (`tests/api_examples_tests.cpp`).
3. `RAG_ENGINE.md`: engine internals, invariants, file format.
4. `../agent.md`: the self-learning agent's architecture.
5. `EXPERIMENTS.md`: what was tried with the agent and what happened.
6. `DECISIONS.md`: why things are the way they are, with evidence.
7. `CONVENTIONS.md`: rules to follow when changing code.
8. `STATUS.md`: limitations and next steps.

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
