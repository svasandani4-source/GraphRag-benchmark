# Plan: GraphRAG vs. Vector RAG, an honest benchmark

**One-line pitch:** Same corpus, same LLM, same questions, **$0 to run**. Four retrieval strategies compared on
accuracy, multi-hop reasoning, compute cost (tokens, indexing time) and latency, with a live demo and a write-up.

## 1. Research questions

1. Does graph retrieval beat dense vector retrieval on **multi-hop** questions, and by how much per hop count?
2. What does the graph cost to build (one-time indexing tokens and hours) and to query (tokens/query, p50/p95 latency)?
3. Where does each approach fail? (entity resolution errors, missing edges, retrieval misses, generation errors)
4. Is a **hybrid** (vector seeds + graph expansion) the best of both?

## 2. Systems under test

| Pipeline | Retrieval | Why it is in the benchmark |
|---|---|---|
| `closed_book` | none | Lower bound: what the LLM already knows |
| `vector_rag` | top-k dense paragraphs (bge-base-en-v1.5, cosine) | The standard baseline |
| `graph_rag` | question entities → entity nodes → 2-hop expansion → triples + source paragraphs | The system being tested |
| `hybrid` | vector top-k ∪ graph neighbours of entities found in those paragraphs | The practical answer most teams ship |

Every pipeline uses the **same generator model, prompt template and context token budget**, so the only
variable is retrieval.

## 3. Tech stack (all free)

| Layer | Choice | Notes |
|---|---|---|
| Language | Python 3.11+ | |
| LLM: extraction + generation | `qwen2.5:7b-instruct` via **Ollama** (local, free) | Good at JSON/structured output for its size; Ollama enforces a JSON schema |
| LLM: judge | `llama3.1:8b` via Ollama | Different model family from the generator, to reduce self-grading bias |
| Optional paid backend | Claude via `anthropic` (`pip install .[claude]`) | Off by default. One config line if you ever want a frontier-model comparison |
| Embeddings | `BAAI/bge-base-en-v1.5` via `sentence-transformers` (local) | Same model as your Ex7 notebook |
| Vector index | NumPy exact cosine | Small corpus, so exact search is fast and removes ANN noise |
| Graph store | NetworkX in-process + optional Neo4j export (Docker or free AuraDB) | In-process for both keeps latency comparisons fair |
| Compute | Your laptop (16 GB RAM) or a free **Colab T4** / **Kaggle** GPU session | See `docs/FREE_RUN.md` |
| Dashboard + demo | Streamlit, hosted free on **Streamlit Community Cloud** or **Hugging Face Spaces** | Hosted demo uses replay mode (saved answers); live questions run locally |
| Tests / CI | pytest with a fake LLM + hashing embedder; GitHub Actions is free for public repos | |

## 4. Corpus and ground truth

- **HotpotQA (distractor, dev split)**: 2-hop questions with gold supporting paragraphs; types `bridge` and `comparison`.
- **MuSiQue-Answerable (dev)**: 2/3/4-hop questions built to resist shortcuts. Added in milestone 3.
- Build **one shared corpus** from the union of all paragraphs (gold + distractors) for the sampled questions, so
  every retriever searches the same ~5k paragraphs instead of a per-question context.
- Sample (sized for free local models): **100 HotpotQA questions with 4 distractors each (~600 paragraphs)** on a laptop,
  or **300 questions with all 8 distractors (up to ~3k paragraphs)** on a free GPU session. MuSiQue adds ~200 questions in M3.
  Stratified by type / hop count, fixed seed, sample IDs committed.

## 5. Metrics

| Category | Metric |
|---|---|
| Answer accuracy | Exact Match and token F1 (official HotpotQA normalisation) as headline numbers; LLM-judge correctness as a secondary check, spot-validated by hand on 50 answers |
| Retrieval | Supporting-paragraph Recall@k and "all gold paragraphs retrieved" rate |
| Multi-hop | All of the above sliced by hop count (2/3/4) and question type (bridge vs comparison) |
| Cost | Input/output tokens per query; one-time indexing tokens and wall-clock ($ too, if the optional Claude backend is used) |
| Latency | p50 / p95 per stage (retrieve, generate) and end to end |
| Robustness | Bootstrap 95% confidence intervals on every headline number |

## 6. Repo structure

```
graphrag-bench/
├── configs/default.yaml        # models, k, hops, sample sizes, budgets
├── src/grb/
│   ├── types.py                # Paragraph, Question, PipelineResult
│   ├── llm.py                  # Ollama (default), optional Claude, FakeLLM for tests
│   ├── embed.py                # bge embedder; hashing embedder for offline
│   ├── data/                   # HotpotQA / MuSiQue loaders → shared corpus
│   ├── index/vector_store.py
│   ├── graph/extract.py        # LLM triple extraction + entity resolution
│   ├── graph/store.py          # NetworkX graph, k-hop retrieval, Neo4j export
│   ├── pipelines/              # closed_book, vector_rag, graph_rag, hybrid
│   ├── eval/                   # metrics, LLM judge, runner, report
│   └── cli.py                  # grb prepare | index | run | report | ask
├── app/dashboard.py            # results dashboard + live side-by-side demo
├── docs/architecture.md        # Mermaid diagrams
├── docs/blog_outline.md
├── tests/
└── results/                    # run outputs (JSONL + summary.json)
```

## 7. Milestones

| # | Scope | Done when |
|---|---|---|
| M1 | Ollama set up, first real run of closed book + vector RAG on 25 questions | `grb run` produces a summary table on your machine |
| M2 | Graph extraction + graph RAG + hybrid on 100 HotpotQA questions | Graph vs vector numbers with timings |
| M3 | MuSiQue (3-4 hops), GPU-session run at 300 questions, judge validation, error analysis | Final results JSON committed |
| M4 | Dashboard + live demo deployed, architecture diagram, README with real numbers, blog post | Public links in README and on your resume |

## 8. Cost: $0

Everything runs on free software and free compute. What you pay instead is **time**. These are rough estimates
for a 7B model at 4-bit quantisation; the runner logs real timings, so the README will report measured numbers.

| Setup | Graph extraction | Benchmark run (4 pipelines + judge) |
|---|---|---|
| Laptop CPU, 100 questions / ~600 paragraphs | ~5-10 h (resumable, run overnight) | ~3-5 h |
| Apple Silicon (M1+), same size | ~1-3 h | ~1-2 h |
| Free Colab T4 or Kaggle GPU, 300 questions / ~3k paragraphs | ~1-2 h | ~1 h |

Extraction is cached per paragraph, so a closed laptop or a dropped Colab session resumes where it stopped.

### Trade-offs of going free

1. **Lower absolute accuracy.** A 7B model answers and extracts worse than a frontier model, so every pipeline scores
   lower. The comparison between pipelines is still fair because they all share the same model.
2. **Noisier graphs.** Small models miss relations and are worse at resolving "he", "the company" and aliases.
   GraphRAG is the pipeline most hurt by this, so the result is really "GraphRAG *with a 7B extractor*" and the write-up must say so.
   That is also a genuine finding worth reporting.
3. **Weaker judge.** An 8B judge is less reliable than a frontier judge. EM/F1 stay the headline metrics and the judge is
   checked against 50 hand-labelled answers.
4. **Smaller sample, wider error bars.** 100-300 questions instead of 600, so confidence intervals are wider and only
   large differences will be significant.
5. **Latency numbers depend on your hardware.** Report them as relative (graph vs vector on the same machine), not absolute.
6. **The hosted demo replays saved answers.** Free hosts cannot run a 7B model, so "ask your own question" only works locally.
   Record a short GIF of live mode for the README.

## 9. Resume bullet (fill in after M3)

> Built an open-source benchmark comparing GraphRAG, vector RAG and a hybrid on N multi-hop questions
> (HotpotQA, MuSiQue) using fully local open models (Qwen 2.5, Llama 3.1); GraphRAG changed 3-4 hop accuracy by X pts F1
> at Y× query latency and Z h one-time indexing; shipped a dashboard and demo.
