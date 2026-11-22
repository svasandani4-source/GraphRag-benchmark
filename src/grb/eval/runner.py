"""Runs every pipeline on every question, scores the answers, and writes results + summary."""
from __future__ import annotations

import json
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from tqdm import tqdm

from ..types import Question
from .judge import judge
from .metrics import all_support, bootstrap_ci, exact_match, f1_score, support_recall


def score_one(q: Question, res, judge_llm, effort: str) -> dict:
    row = res.to_dict()
    row.update(
        question=q.question, gold=q.answer, qtype=q.qtype, hops=q.hops,
        em=exact_match(res.answer, q.answer), f1=f1_score(res.answer, q.answer),
        support_recall=support_recall(res.retrieved_pids, q.supporting_pids) if res.pipeline != "closed_book" else None,
        all_support=all_support(res.retrieved_pids, q.supporting_pids) if res.pipeline != "closed_book" else None,
    )
    if judge_llm is not None:
        ok, u = judge(judge_llm, q.question, q.answer, res.answer, effort)
        row["judge_correct"] = None if ok is None else float(ok)
        row["judge_cost_usd"] = u.cost_usd
    return row


def run_benchmark(questions: list[Question], pipelines: dict, judge_llm, judge_effort: str,
                  out: Path, workers: int = 4) -> list[dict]:
    rows: list[dict] = []
    jobs = [(q, p) for q in questions for p in pipelines.values()]

    def work(job):
        q, p = job
        return score_one(q, p.run(q), judge_llm, judge_effort)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for row in tqdm(pool.map(work, jobs), total=len(jobs), desc="run"):
            rows.append(row)
    with open(out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return rows


def _agg(rows: list[dict]) -> dict:
    def col(k):
        return [r[k] for r in rows if r.get(k) is not None]

    out = {"n": len(rows)}
    for k in ("em", "f1", "judge_correct", "support_recall", "all_support"):
        vals = col(k)
        if vals:
            lo, hi = bootstrap_ci(vals)
            out[k] = {"mean": float(np.mean(vals)), "ci95": [lo, hi]}
    lat = col("latency_ms")
    for stage in ("retrieve", "generate", "total"):
        v = [l[stage] for l in lat]
        out[f"latency_{stage}_ms"] = {"p50": float(np.percentile(v, 50)), "p95": float(np.percentile(v, 95))}
    out["cost_per_query_usd"] = float(np.mean([r["usage"]["cost_usd"] for r in rows]))
    out["tokens_in_per_query"] = float(np.mean([r["usage"]["input_tokens"] for r in rows]))
    out["tokens_out_per_query"] = float(np.mean([r["usage"]["output_tokens"] for r in rows]))
    return out


def summarize(rows: list[dict], index_stats: dict | None = None) -> dict:
    by_pipe = defaultdict(list)
    for r in rows:
        by_pipe[r["pipeline"]].append(r)
    summary: dict = {"index": index_stats or {}, "pipelines": {}}
    for name, rs in by_pipe.items():
        entry = {"overall": _agg(rs), "by_hops": {}, "by_type": {}}
        for key, field in (("by_hops", "hops"), ("by_type", "qtype")):
            groups = defaultdict(list)
            for r in rs:
                groups[str(r[field])].append(r)
            entry[key] = {g: _agg(v) for g, v in sorted(groups.items())}
        summary["pipelines"][name] = entry
    return summary


def summary_markdown(summary: dict) -> str:
    def m(d, k):
        return f"{d[k]['mean']*100:.1f}" if k in d else "n/a"

    lines = ["| Pipeline | EM | F1 | Judge acc | Support recall | All gold retrieved | p50 latency (ms) | p95 latency (ms) | Tokens/query | $/query |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for name, e in summary["pipelines"].items():
        o = e["overall"]
        lines.append(f"| {name} | {m(o,'em')} | {m(o,'f1')} | {m(o,'judge_correct')} | {m(o,'support_recall')} | "
                     f"{m(o,'all_support')} | {o['latency_total_ms']['p50']:.0f} | {o['latency_total_ms']['p95']:.0f} | "
                     f"{o['tokens_in_per_query'] + o['tokens_out_per_query']:.0f} | {o['cost_per_query_usd']:.4f} |")
    hops = sorted({h for e in summary["pipelines"].values() for h in e["by_hops"]})
    if len(hops) > 1:
        lines += ["", "**F1 by hop count**", "", "| Pipeline | " + " | ".join(f"{h}-hop" for h in hops) + " |",
                  "|---|" + "---|" * len(hops)]
        for name, e in summary["pipelines"].items():
            lines.append(f"| {name} | " + " | ".join(m(e["by_hops"][h], "f1") if h in e["by_hops"] else "n/a" for h in hops) + " |")
    idx = summary.get("index") or {}
    if idx:
        lines += ["", f"Indexing: {idx.get('paragraphs')} paragraphs, graph {idx.get('graph')}, "
                      f"extraction {idx.get('extraction_input_tokens', 0) + idx.get('extraction_output_tokens', 0):,} tokens "
                      f"(${idx.get('extraction_cost_usd', 0):.2f}), "
                      f"embed {idx.get('embed_seconds', 0):.1f}s, extract {idx.get('extract_seconds', 0):.1f}s"]
    return "\n".join(lines)
