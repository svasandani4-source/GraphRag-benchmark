"""LLM-as-judge for semantic correctness (catches 'Barack Obama' vs 'Obama', '1891' vs 'in 1891')."""
from __future__ import annotations

from pydantic import BaseModel

from ..types import Usage

JUDGE_SYSTEM = """You grade answers to trivia questions.
Mark the predicted answer correct if it refers to the same entity, value or yes/no as the gold answer,
allowing for aliases, extra qualifiers, formatting and unit differences. Mark it incorrect if it is
a different answer, is vaguer than the gold answer in a way that changes meaning, or hedges between options."""


class JudgeVerdict(BaseModel):
    rationale: str
    correct: bool


def judge(llm, question: str, gold: str, pred: str, effort: str = "low") -> tuple[bool | None, Usage]:
    prompt = f"Question: {question}\nGold answer: {gold}\nPredicted answer: {pred}"
    verdict, usage = llm.parse(JUDGE_SYSTEM, prompt, JudgeVerdict, max_tokens=1024, effort=effort)
    return (verdict.correct if verdict else None), usage
