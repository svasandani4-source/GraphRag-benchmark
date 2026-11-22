"""Streamlit dashboard: benchmark results + a live side-by-side demo.

    streamlit run app/dashboard.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# Fixed colour per pipeline (validated categorical palette, slots 1-4). Colour follows the
# pipeline everywhere, never its rank.
COLORS = {"closed_book": "#eda100", "vector_rag": "#2a78d6", "graph_rag": "#eb6834", "hybrid": "#1baf7a"}
ORDER = list(COLORS)

st.set_page_config(page_title="GraphRAG vs Vector RAG", layout="wide")
st.title("GraphRAG vs Vector RAG")
st.caption("Same corpus, same generator, same token budget. Only retrieval changes.")

runs = sorted(p.parent for p in (ROOT / "results").glob("*/summary.json"))
if not runs:
    st.info("No results yet. Run `make smoke` (offline) or `grb run` first.")
    st.stop()
run = st.selectbox("Run", runs, format_func=lambda p: p.name, index=len(runs) - 1)
summary = json.loads((run / "summary.json").read_text())
rows = pd.DataFrame([json.loads(l) for l in open(run / "results.jsonl")])
pipes = [p for p in ORDER if p in summary["pipelines"]]

metric = st.radio("Accuracy metric", ["f1", "em", "judge_correct"], horizontal=True,
                  format_func={"f1": "Token F1", "em": "Exact match", "judge_correct": "LLM judge"}.get)

tab_overview, tab_hops, tab_cost, tab_errors, tab_demo = st.tabs(
    ["Overview", "Multi-hop", "Cost and latency", "Where they differ", "Live demo"])

with tab_overview:
    o = {p: summary["pipelines"][p]["overall"] for p in pipes}
    cols = st.columns(len(pipes))
    for c, p in zip(cols, pipes):
        m = o[p].get(metric)
        c.metric(p, f"{m['mean']*100:.1f}" if m else "n/a",
                 help=f"95% CI {m['ci95'][0]*100:.1f} to {m['ci95'][1]*100:.1f}" if m else None)
    df = pd.DataFrame([{"pipeline": p, "score": o[p][metric]["mean"] * 100,
                        "lo": o[p][metric]["ci95"][0] * 100, "hi": o[p][metric]["ci95"][1] * 100}
                       for p in pipes if metric in o[p]])
    if not df.empty:
        fig = px.bar(df, x="pipeline", y="score", color="pipeline", color_discrete_map=COLORS,
                     error_y=df["hi"] - df["score"], error_y_minus=df["score"] - df["lo"],
                     category_orders={"pipeline": pipes}, labels={"score": f"{metric} (%)"})
        fig.update_layout(showlegend=False, bargap=0.5)
        st.plotly_chart(fig, width="stretch")
    st.markdown((run / "summary.md").read_text())

with tab_hops:
    recs = [{"pipeline": p, "hops": int(h), "score": v[metric]["mean"] * 100, "n": v["n"]}
            for p in pipes for h, v in summary["pipelines"][p]["by_hops"].items() if metric in v]
    if recs:
        fig = px.line(pd.DataFrame(recs), x="hops", y="score", color="pipeline", markers=True,
                      color_discrete_map=COLORS, category_orders={"pipeline": pipes}, hover_data=["n"],
                      labels={"score": f"{metric} (%)", "hops": "Reasoning hops"})
        fig.update_xaxes(dtick=1)
        st.plotly_chart(fig, width="stretch")
    rec = [{"pipeline": p, "support recall": v["overall"].get("support_recall", {}).get("mean", 0) * 100,
            "all gold retrieved": v["overall"].get("all_support", {}).get("mean", 0) * 100}
           for p, v in summary["pipelines"].items() if p != "closed_book"]
    st.subheader("Retrieval quality")
    st.dataframe(pd.DataFrame(rec).set_index("pipeline").round(1))

with tab_cost:
    paid = any(o[p]["cost_per_query_usd"] > 0 for p in pipes)
    xcol = "cost per 1k queries ($)" if paid else "tokens per query"
    recs = [{"pipeline": p, "cost per 1k queries ($)": o[p]["cost_per_query_usd"] * 1000,
             "tokens per query": o[p]["tokens_in_per_query"] + o[p]["tokens_out_per_query"],
             "score": o[p][metric]["mean"] * 100, "p50 ms": o[p]["latency_total_ms"]["p50"],
             "p95 ms": o[p]["latency_total_ms"]["p95"]} for p in pipes if metric in o[p]]
    c1, c2 = st.columns(2)
    if recs:
        d = pd.DataFrame(recs)
        f1 = px.scatter(d, x=xcol, y="score", color="pipeline", text="pipeline",
                        color_discrete_map=COLORS, labels={"score": f"{metric} (%)"})
        f1.update_traces(marker_size=12, textposition="top center")
        f1.update_layout(showlegend=False, title=f"Accuracy vs {xcol}")
        c1.plotly_chart(f1, width="stretch")
        f2 = px.bar(d.melt(id_vars="pipeline", value_vars=["p50 ms", "p95 ms"]), x="pipeline", y="value",
                    color="pipeline", pattern_shape="variable", barmode="group", color_discrete_map=COLORS,
                    labels={"value": "End-to-end latency (ms)"}, title="Latency")
        c2.plotly_chart(f2, width="stretch")
    idx = summary.get("index", {})
    if idx:
        st.caption(f"One-time indexing: {idx.get('paragraphs')} paragraphs, graph {idx.get('graph')}, "
                   f"extraction {idx.get('extract_seconds', 0) / 3600:.2f} h and "
                   f"{idx.get('extraction_input_tokens', 0) + idx.get('extraction_output_tokens', 0):,} tokens "
                   f"with {idx.get('extraction_model')}.")

with tab_errors:
    a, b = st.columns(2)
    left = a.selectbox("Correct with", pipes, index=pipes.index("graph_rag") if "graph_rag" in pipes else 0)
    right = b.selectbox("Wrong with", pipes, index=pipes.index("vector_rag") if "vector_rag" in pipes else 0)
    key = "judge_correct" if "judge_correct" in rows and rows["judge_correct"].notna().any() else "em"
    wide = rows.pivot_table(index=["qid", "question", "gold", "hops"], columns="pipeline", values=key).reset_index()
    ans = rows.pivot(index="qid", columns="pipeline", values="answer")
    diff = wide[(wide[left] == 1) & (wide[right] == 0)]
    st.write(f"{len(diff)} questions answered correctly by **{left}** but not **{right}** (by {key}).")
    st.dataframe(diff[["qid", "question", "gold", "hops"]].assign(
        **{f"{left} answer": diff["qid"].map(ans[left]), f"{right} answer": diff["qid"].map(ans[right])}),
        width="stretch", hide_index=True)

with tab_demo:
    mode = st.radio("Mode", ["Replay benchmark questions", "Ask your own question"], horizontal=True,
                    help="Replay works anywhere (including free hosting) because it shows saved answers. "
                         "Asking your own question needs the run's models available, e.g. a local Ollama server.")
    if mode == "Replay benchmark questions":
        qs = rows.drop_duplicates("qid").set_index("qid")
        qid = st.selectbox("Question", qs.index, format_func=lambda i: f"[{qs.loc[i, 'hops']}-hop] {qs.loc[i, 'question']}")
        st.markdown(f"**Gold answer:** {qs.loc[qid, 'gold']}")
        cols = st.columns(len(pipes))
        for c, p in zip(cols, pipes):
            r = rows[(rows.qid == qid) & (rows.pipeline == p)].iloc[0]
            ok = r.get("judge_correct")
            verdict = "correct" if ok == 1 else "wrong" if ok == 0 else "not judged"
            c.markdown(f"**{p}**")
            c.markdown(f"### {r['answer'] or '(no answer)'}")
            c.caption(f"{verdict} · {r['latency_ms']['total']:.0f} ms")
            with c.expander("Retrieved"):
                st.write(r["retrieved_pids"])
                st.json(r["debug"])
    else:
        q = st.text_input("Question", placeholder="e.g. a 2-hop question about entities in the corpus")
        if q and st.button("Run all pipelines"):
            from grb.cli import build_resources
            from grb.config import Config
            from grb.pipelines import PIPELINES
            from grb.types import Question

            cfg = Config(summary["config"])

            @st.cache_resource
            def _res(path: str):
                return build_resources(cfg)

            try:
                res = _res(cfg["artifacts_dir"])
                cols = st.columns(len(pipes))
                for c, p in zip(cols, pipes):
                    r = PIPELINES[p](res).run(Question(qid="demo", question=q, answer=""))
                    c.markdown(f"**{p}**")
                    c.markdown(f"### {r.answer or '(no answer)'}")
                    c.caption(f"{r.latency_ms['total']:.0f} ms")
                    with c.expander("Retrieved"):
                        st.write([res.corpus[pid].title for pid in r.retrieved_pids])
                        st.json(r.debug)
            except Exception as e:  # e.g. no Ollama server on a hosted deployment
                st.warning(f"Live mode needs this run's models and index available locally ({type(e).__name__}). "
                           "Use replay mode on hosted deployments.")
