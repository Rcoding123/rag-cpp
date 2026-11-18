# mbo-engine

ES futures analysis engine and RL agents trained on Databento market-by-order
(MBO) data, developed on a headless Linux server. It is separate from the rest
of rag-cpp, which is a Windows/MSVC project.

| File | What it is |
|---|---|
| `PLAN.md` | The plan: principles, architecture, phases and acceptance checks |
| `HANDOFF.md` | Current state: server, data inventory, scripts, open issues |
| `CLAUDE.md` | Working rules for AI sessions in this folder |
| `scripts/` | Databento quote/order/download and data-check scripts |
| `server/` | Server setup scripts (research box and separate build box) |

To continue on the server:

```bash
cd /srv/quant/code && git clone https://github.com/Rcoding123/rag-cpp.git && cd rag-cpp/mbo-engine
```

Then start your AI session in `rag-cpp/mbo-engine` so it picks up `CLAUDE.md`.
