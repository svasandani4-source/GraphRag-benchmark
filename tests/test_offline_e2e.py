"""End-to-end on the synthetic fixture with the fake LLM: no API key, no downloads."""
import json

import yaml

from grb.cli import main


def test_offline_pipeline(tmp_path):
    cfg = yaml.safe_load(open("configs/offline.yaml"))
    cfg["artifacts_dir"] = str(tmp_path / "art")
    cfg["results_dir"] = str(tmp_path / "res")
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(cfg))
    for cmd in ("prepare", "index", "run"):
        main(["--config", str(path), cmd])
    summary = json.loads((tmp_path / "res" / "summary.json").read_text())
    assert set(summary["pipelines"]) == {"closed_book", "vector_rag", "graph_rag", "hybrid"}
    for name in ("vector_rag", "graph_rag", "hybrid"):
        assert summary["pipelines"][name]["overall"]["support_recall"]["mean"] > 0
    rows = [json.loads(l) for l in open(tmp_path / "res" / "results.jsonl")]
    assert len(rows) == 8 * 4
