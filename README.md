# graphrag-bench

**Does GraphRAG beat vector RAG on multi-hop questions, and is it worth the cost?**
This repo answers that with a controlled benchmark: the same corpus, the same generator model,
the same prompt and the same context token budget across four pipelines. Only retrieval changes.
**It runs for $0**: local open models via Ollama (Qwen 2.5 7B, Llama 3.1 8B), local embeddings, free hosting.

| Pipeline | Retrieval |
|---|---|
| `closed_book` | none (lower bound) |
| `vector_rag` | top-k paragraphs by dense similarity (bge-base-en-v1.5) |
| `graph_rag` | question entities → knowledge-graph nodes → 2-hop expansion → facts + source paragraphs |
| `hybrid` | vector top-k plus graph neighbours of the entities they mention |

Measured: Exact Match, F1, LLM-judge accuracy, supporting-paragraph recall, accuracy by hop count,
p50/p95 latency, tokens per query and one-time indexing time, all with bootstrap 95% confidence intervals.

> **Results:** _coming after milestone M3. See [docs/PLAN.md](docs/PLAN.md)._

Architecture diagrams: [docs/architecture.md](docs/architecture.md).

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[all]"

# 1. Offline smoke test: synthetic fixture, fake LLM, no models needed (numbers are meaningless, it checks plumbing)
make smoke

# 2. Real run with free local models (install Ollama from https://ollama.com first)
ollama pull qwen2.5:7b-instruct && ollama pull llama3.1:8b
make prepare index run      # uses configs/default.yaml (100 questions, laptop-sized)
make dashboard              # results + demo
```

Bigger run on a free Colab/Kaggle GPU, timings and trade-offs: [docs/FREE_RUN.md](docs/FREE_RUN.md).

Ad-hoc question across all pipelines:

```bash
grb ask "Which mountain range is the source of the river that flows through Ostrand?" --config configs/offline.yaml
```

Explore the extracted graph in Neo4j: `docker compose up -d && grb export-neo4j`.

## Configuration

Everything lives in [`configs/default.yaml`](configs/default.yaml): sample size, distractors per question, `top_k`,
graph hops, the shared context budget, and the model for each role. The prefix picks the backend:
`ollama:<model>` (free, default), `claude-<id>` (paid, optional: `pip install -e ".[claude]"`), or `fake` (tests).

## Layout

```
src/grb/  data/ index/ graph/ pipelines/ eval/ llm.py embed.py cli.py
app/dashboard.py     docs/  configs/  tests/
```

## Roadmap

- [x] M1 scaffold: loaders, four pipelines, metrics, runner, dashboard, offline CI
- [ ] M1 first local run: 25 HotpotQA questions, closed book + vector RAG
- [ ] M2 graph extraction + GraphRAG + hybrid at 100 questions
- [ ] M3 MuSiQue (3-4 hops), 300-question free-GPU run, judge validation, error analysis
- [ ] M4 hosted dashboard (replay mode), blog post, final numbers in this README
