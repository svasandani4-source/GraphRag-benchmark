# Architecture

## Indexing (offline, once per corpus)

```mermaid
flowchart LR
    DS[(HotpotQA / MuSiQue sample)] --> P[grb prepare<br/>shared corpus + questions]
    P --> E[bge-base-en-v1.5<br/>embeddings]
    E --> V[(Vector index<br/>NumPy, exact cosine)]
    P --> X[Local LLM (Ollama, JSON schema)<br/>entities + relations per paragraph]
    X --> R[Entity resolution<br/>name normalisation]
    R --> G[(Knowledge graph<br/>NetworkX, optional Neo4j export)]
    X -. tokens, time .-> S[index_stats.json]
```

## Query time (same generator, prompt and token budget for every pipeline)

```mermaid
flowchart TB
    Q[Question] --> CB[closed_book<br/>no context]
    Q --> VR[vector_rag<br/>top-k paragraphs]
    Q --> QE[LLM: question entities] --> L[Entity linking<br/>exact + embedding match] --> GX[k-hop BFS expansion]
    GX --> GR[graph_rag<br/>triples + best-supported paragraphs]
    VR --> HY[hybrid<br/>vector top-k ∪ graph neighbours]
    GX --> HY
    CB & GR & HY & VR --> GEN[Shared generator<br/>Qwen 2.5 7B via Ollama]
    GEN --> EV[Scoring: EM, F1, LLM judge,<br/>support recall, latency, tokens]
    EV --> OUT[(results.jsonl + summary.json)] --> DASH[Streamlit dashboard + live demo]
```
