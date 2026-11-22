# Running the benchmark for $0

Everything here is free: open models through [Ollama](https://ollama.com), local embeddings, and free hosting.

## Option A: your laptop (16 GB RAM recommended)

```bash
# 1. Install Ollama from https://ollama.com/download, then pull the two models (~9 GB total)
ollama pull qwen2.5:7b-instruct
ollama pull llama3.1:8b

# 2. Install the project (no paid SDKs)
pip install -e ".[embed,data,app,dev]"

# 3. Run. Extraction is cached per paragraph, so you can stop and restart any time.
grb prepare && grb index && grb run && grb report
streamlit run app/dashboard.py
```

Start with `n_questions: 25` in `configs/default.yaml` to see results within an hour or two, then scale up.

## Option B: free GPU (Colab T4 or Kaggle), for the larger run

In a notebook cell (Colab: Runtime → Change runtime type → T4 GPU):

```bash
!curl -fsSL https://ollama.com/install.sh | sh
!nohup env OLLAMA_NUM_PARALLEL=4 ollama serve > ollama.log 2>&1 &
!sleep 5 && ollama pull qwen2.5:7b-instruct && ollama pull llama3.1:8b
!git clone https://github.com/svasandani4-source/GraphRag-benchmark && cd GraphRag-benchmark && pip install -q -e ".[embed,data]"
!cd GraphRag-benchmark && grb --config configs/gpu_free.yaml prepare && grb --config configs/gpu_free.yaml index
!cd GraphRag-benchmark && grb --config configs/gpu_free.yaml run
```

Free sessions can disconnect. Copy `artifacts/` to Google Drive (Colab) or save it as an output (Kaggle)
between steps so a new session resumes instead of starting over. Kaggle gives a weekly GPU quota, Colab's free GPU
availability varies.

## Free hosting for the dashboard

Commit the `results/<run>/` folder, then deploy `app/dashboard.py` on
[Streamlit Community Cloud](https://streamlit.io/cloud) or a Hugging Face Space (Streamlit SDK).
Free hosts can't run a 7B model, so the hosted demo uses **replay mode** (saved answers for every benchmark question).
"Ask your own question" works when you run the dashboard locally with Ollama running.

## Optional: free Neo4j for exploring the graph

`docker compose up -d` locally, or a free AuraDB instance, then `grb export-neo4j`.
