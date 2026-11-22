# Blog post outline: "Does GraphRAG actually beat vector RAG? I measured it."

1. **Hook**: one multi-hop question vector RAG gets wrong and GraphRAG gets right (pull from the dashboard's "Where they differ" tab). Then the counter-example.
2. **Why this benchmark**: most GraphRAG posts are demos. Fixed corpus, fixed generator, fixed token budget, 600 questions, confidence intervals.
3. **Setup**: architecture diagram, datasets, how the shared corpus is built, the four pipelines.
4. **Results**: headline table; accuracy by hop count chart; accuracy vs cost scatter; latency.
5. **What the graph costs**: indexing $ and time, tokens per query, and the break-even query volume.
6. **Error analysis**: 20 hand-labelled failures per pipeline: entity resolution misses, missing edges, retrieval misses, generator errors.
7. **When to use which**: a decision table.
8. **Limitations**: Wikipedia-style corpora, one generator family, LLM-judge bias, approx token budget.
9. **Reproduce it**: repo link, `make smoke`, `make run`, the live demo.
