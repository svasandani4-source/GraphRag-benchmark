CFG ?= configs/default.yaml

install:
	pip install -e ".[all]"

models:           ## pull the free local models (needs Ollama installed)
	ollama pull qwen2.5:7b-instruct && ollama pull llama3.1:8b

test:
	pytest -q

smoke:            ## end-to-end on the bundled fixture, no API key needed
	grb --config configs/offline.yaml prepare && \
	grb --config configs/offline.yaml index && \
	grb --config configs/offline.yaml run && \
	grb --config configs/offline.yaml report

prepare:
	grb --config $(CFG) prepare
index:
	grb --config $(CFG) index
run:
	grb --config $(CFG) run
report:
	grb --config $(CFG) report
dashboard:
	streamlit run app/dashboard.py
