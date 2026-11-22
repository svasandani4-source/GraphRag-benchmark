"""Shared pipeline machinery: every pipeline uses the same generator, prompt and token budget,
so the only thing that differs between them is retrieval."""
from __future__ import annotations

import time
from dataclasses import dataclass

from ..types import Paragraph, PipelineResult, Question, Retrieval, Usage

ANSWER_SYSTEM = """You answer questions using the provided context.
Reply with the short answer only: an entity name, date, number, or yes/no. No explanation, no full sentence.
If the context does not contain the answer, give your best guess."""

CLOSED_BOOK_SYSTEM = """Answer the question from your own knowledge.
Reply with the short answer only: an entity name, date, number, or yes/no. No explanation, no full sentence."""


def approx_tokens(text: str) -> int:
    return len(text) // 4 + 1


@dataclass
class Resources:
    corpus: dict[str, Paragraph]
    embedder: object
    vector_index: object | None
    graph: object | None
    llm: object                    # generator
    cfg: dict                      # retrieval + effort settings


class Pipeline:
    name = "base"
    system = ANSWER_SYSTEM

    def __init__(self, res: Resources):
        self.res = res
        self.cfg = res.cfg

    def retrieve(self, q: Question) -> tuple[Retrieval, Usage]:
        raise NotImplementedError

    def paragraphs_block(self, pids: list[str], budget: int) -> tuple[str, list[str]]:
        parts, used, kept = [], 0, []
        for pid in pids:
            p = self.res.corpus.get(pid)
            if p is None:
                continue
            chunk = f"[{p.title}] {p.text}"
            t = approx_tokens(chunk)
            if used + t > budget:
                break
            parts.append(chunk)
            kept.append(pid)
            used += t
        return "\n\n".join(parts), kept

    def run(self, q: Question) -> PipelineResult:
        t0 = time.perf_counter()
        retrieval, usage = self.retrieve(q)
        t1 = time.perf_counter()
        prompt = (f"Context:\n{retrieval.context}\n\n" if retrieval.context else "") + f"Question: {q.question}"
        answer, gen_usage = self.res.llm.complete(self.system, prompt, max_tokens=256,
                                                  effort=self.cfg["effort_generation"])
        t2 = time.perf_counter()
        usage += gen_usage
        return PipelineResult(
            qid=q.qid, pipeline=self.name, answer=answer.strip().rstrip("."),
            retrieved_pids=retrieval.pids, usage=usage,
            latency_ms={"retrieve": (t1 - t0) * 1000, "generate": (t2 - t1) * 1000, "total": (t2 - t0) * 1000},
            debug=retrieval.debug,
        )
