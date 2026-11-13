# Hand-off bundles

Single files to paste into another AI's context, generated from the
repository by `tools/make_context_pack.py`:

| File | Contents | Size |
|---|---|---|
| `context_pack_docs.md` | every document in `docs/ai/`, `docs/agent.md`, the README, and the list of repository files | ~147k tokens |
| `context_pack_full.md` | the same, plus every header, test, example, benchmark and build file, and the Python pipeline (`python/`) | ~567k tokens |

Start with the docs bundle; it contains the reading order (`docs/ai/README.md`)
and is enough to understand, use and extend the project. Use the full bundle
when the other AI needs to change code.

The bundles are snapshots of the commit they were generated at. After
changing the code or docs, regenerate them:

```bash
python3 tools/make_context_pack.py --out handoff/context_pack_docs.md
python3 tools/make_context_pack.py --with-source --out handoff/context_pack_full.md
```
