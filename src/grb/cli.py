"""grb: command line for the benchmark.

  grb prepare   sample questions and build the shared corpus
  grb index     embed paragraphs and extract the knowledge graph (records indexing cost)
  grb run       run all pipelines on all questions and score them
  grb report    print the summary table
  grb ask "..." run every pipeline on one ad-hoc question
  grb export-neo4j   push the graph into Neo4j for Cypher exploration
"""
from __future__ import annotations

import argparse
import json
import os
import time

from .config import Config, load_config
from .data import load_dataset_sample, read_corpus, read_questions, write_jsonl
from .embed import make_embedder
from .eval.runner import run_benchmark, summarize, summary_markdown
from .graph.extract import extract_corpus
from .graph.store import KnowledgeGraph
from .index.vector_store import VectorIndex
from .llm import make_llm
from .pipelines import PIPELINES, Resources
from .types import Question


def _load_env() -> None:
    if os.path.exists(".env"):
        for line in open(".env"):
            if "=" in line and not line.lstrip().startswith("#"):
                k, v = line.strip().split("=", 1)
                if v:
                    os.environ.setdefault(k, v)


def cmd_prepare(cfg: Config) -> None:
    d = cfg.data
    qs, corpus = load_dataset_sample(d["dataset"], d.get("split", "validation"), d["n_questions"], d["seed"],
                                     d.get("distractors_per_question"))
    write_jsonl(cfg.artifacts / "questions.jsonl", qs)
    write_jsonl(cfg.artifacts / "corpus.jsonl", corpus)
    print(f"prepared {len(qs)} questions, {len(corpus)} paragraphs -> {cfg.artifacts}")


def cmd_index(cfg: Config) -> None:
    corpus = read_corpus(cfg.artifacts / "corpus.jsonl")
    embedder = make_embedder(cfg.embedding["model"])
    t0 = time.perf_counter()
    VectorIndex.build(corpus, embedder).save(cfg.artifacts)
    t1 = time.perf_counter()
    llm = make_llm(cfg.models["extraction"])
    extractions, usage = extract_corpus(llm, corpus, cfg.artifacts / "extractions.jsonl",
                                        effort=cfg.models["effort"]["extraction"],
                                        workers=cfg.get("workers", 4))
    t2 = time.perf_counter()
    kg = KnowledgeGraph.build(extractions)
    kg.save(cfg.artifacts / "graph.json")
    stats = {"paragraphs": len(corpus), "graph": kg.stats(), "embed_seconds": t1 - t0,
             "extract_seconds": t2 - t1, "extraction_calls_this_run": usage.calls,
             "extraction_input_tokens": usage.input_tokens, "extraction_output_tokens": usage.output_tokens,
             "extraction_cost_usd": usage.cost_usd, "extraction_model": cfg.models["extraction"]}
    (cfg.artifacts / "index_stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))


def build_resources(cfg: Config) -> Resources:
    corpus = {p.pid: p for p in read_corpus(cfg.artifacts / "corpus.jsonl")}
    embedder = make_embedder(cfg.embedding["model"])
    kg = KnowledgeGraph.load(cfg.artifacts / "graph.json")
    kg.index_names(embedder)
    r = cfg.retrieval
    return Resources(corpus=corpus, embedder=embedder, vector_index=VectorIndex.load(cfg.artifacts), graph=kg,
                     llm=make_llm(cfg.models["generation"]),
                     cfg={**r, "effort_generation": cfg.models["effort"]["generation"]})


def cmd_run(cfg: Config) -> None:
    res = build_resources(cfg)
    pipes = {name: PIPELINES[name](res) for name in cfg["pipelines"]}
    judge_llm = make_llm(cfg.models["judge"]) if cfg.get("use_judge") else None
    qs = read_questions(cfg.artifacts / "questions.jsonl")
    rows = run_benchmark(qs, pipes, judge_llm, cfg.models["effort"]["judge"], cfg.results / "results.jsonl",
                         workers=cfg.get("workers", 4))
    stats_path = cfg.artifacts / "index_stats.json"
    summary = summarize(rows, json.loads(stats_path.read_text()) if stats_path.exists() else None)
    summary["config"] = dict(cfg)
    (cfg.results / "summary.json").write_text(json.dumps(summary, indent=2))
    (cfg.results / "summary.md").write_text(summary_markdown(summary))
    print(summary_markdown(summary))


def cmd_report(cfg: Config) -> None:
    print((cfg.results / "summary.md").read_text())


def cmd_ask(cfg: Config, question: str) -> None:
    res = build_resources(cfg)
    q = Question(qid="adhoc", question=question, answer="")
    for name in cfg["pipelines"]:
        r = PIPELINES[name](res).run(q)
        print(f"{name:12s} {r.answer!r:40s} {r.latency_ms['total']:7.0f} ms  ${r.usage.cost_usd:.4f}  {r.debug}")


def cmd_export_neo4j(cfg: Config) -> None:
    from neo4j import GraphDatabase

    kg = KnowledgeGraph.load(cfg.artifacts / "graph.json")
    drv = GraphDatabase.driver(os.environ["NEO4J_URI"], auth=(os.environ.get("NEO4J_USERNAME", "neo4j"),
                                                               os.environ["NEO4J_PASSWORD"]))
    with drv.session() as s:
        s.run("CREATE CONSTRAINT entity_key IF NOT EXISTS FOR (e:Entity) REQUIRE e.key IS UNIQUE")
        nodes = [{"key": k, **{a: d[a] for a in ("name", "type", "pids")}} for k, d in kg.g.nodes(data=True)]
        s.run("UNWIND $rows AS r MERGE (e:Entity {key: r.key}) SET e.name=r.name, e.type=r.type, e.pids=r.pids",
              rows=nodes)
        edges = [{"s": u, "t": v, **d} for u, v, d in kg.g.edges(data=True)]
        s.run("UNWIND $rows AS r MATCH (a:Entity {key: r.s}), (b:Entity {key: r.t}) "
              "CREATE (a)-[:REL {type: r.relation, evidence: r.evidence, pid: r.pid}]->(b)", rows=edges)
    print(f"exported {len(nodes)} nodes, {len(edges)} edges")


def main(argv: list[str] | None = None) -> None:
    _load_env()
    ap = argparse.ArgumentParser(prog="grb")
    ap.add_argument("--config", default="configs/default.yaml")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for c in ("prepare", "index", "run", "report", "export-neo4j"):
        sub.add_parser(c)
    ask = sub.add_parser("ask")
    ask.add_argument("question")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    {"prepare": cmd_prepare, "index": cmd_index, "run": cmd_run, "report": cmd_report,
     "export-neo4j": cmd_export_neo4j}.get(args.cmd, lambda c: cmd_ask(c, args.question))(cfg)


if __name__ == "__main__":
    main()
