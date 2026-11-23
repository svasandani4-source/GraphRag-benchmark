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

## First results (pilot run, 24 questions)

A small pilot on a free Colab T4 GPU: 24 HotpotQA questions (12 bridge, 12 comparison), 240 paragraphs in the
shared corpus, Qwen 2.5 7B for extraction and answering, Llama 3.1 8B as judge, all through Ollama.

| Pipeline | EM | F1 | Judge acc | Support recall | All gold retrieved | p50 latency (s) | p95 latency (s) | Tokens/query |
|---|---|---|---|---|---|---|---|---|
| closed_book | 37.5 | 46.1 | 58.3 | n/a | n/a | 6.5 | 7.3 | 71 |
| vector_rag | 50.0 | 62.7 | **95.8** | 91.7 | 83.3 | **7.1** | **10.8** | **648** |
| graph_rag | 50.0 | 60.4 | 83.3 | 72.9 | 54.2 | 22.8 | 39.2 | 1,158 |
| hybrid | **54.2** | **64.9** | 87.5 | **93.8** | **87.5** | 22.2 | 28.0 | 1,136 |

Indexing: 240 paragraphs → graph of 2,218 entities, 1,736 relations, **647 disconnected components**;
extraction took 56 minutes and 240,884 tokens. Cost: $0.

**What this shows (and doesn't)**

- **Retrieval helps a lot.** Every retrieval pipeline beats closed book by about 15 F1 points.
- **Hybrid retrieves best.** It found every gold paragraph for 87.5% of questions, against 83.3% for vector and
  54.2% for graph alone, and has the best EM and F1.
- **GraphRAG alone retrieves worst.** The graph is very fragmented (647 components for 2,218 nodes, fewer edges than
  nodes), so 2-hop expansion often can't reach the second paragraph. A 7B extractor lists many entities but links few of them.
  Hybrid works around this by seeding the graph from the vector hits.
- **The graph pipelines cost about 3× the latency and 1.8× the tokens** of vector RAG, mostly from the extra
  question-entity LLM call and the longer context.
- **Too small to call a winner.** With 24 questions, one question is 4.2 points, so the differences between vector,
  graph and hybrid are within noise. The full run (300 questions) will give real confidence intervals.
- **The judge looks lenient.** 95.8% judge accuracy vs 50% EM for vector RAG needs checking against hand labels before
  the judge numbers are trusted.
- Latencies are on a shared T4 with 4 parallel requests, so compare them across pipelines, not as absolute numbers.

**Next:** reduce graph fragmentation, then the full 300-question run.


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
- [x] M1 pilot run: 24 HotpotQA questions, all four pipelines (results above)
- [ ] M2 graph extraction + GraphRAG + hybrid at 100 questions
- [ ] M3 MuSiQue (3-4 hops), 300-question free-GPU run, judge validation, error analysis
- [ ] M4 hosted dashboard (replay mode), blog post, final numbers in this README
