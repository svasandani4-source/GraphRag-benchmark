"""LLM-based knowledge-graph extraction: one structured-output call per paragraph."""
from __future__ import annotations

import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from pydantic import BaseModel, Field
from tqdm import tqdm

from ..types import Paragraph, Usage


class Entity(BaseModel):
    name: str = Field(description="Canonical name as written, e.g. 'Barack Obama', not 'Obama' or 'he'")
    type: str = Field(description="One of: Person, Organization, Location, Work, Event, Date, Concept, Thing")
    description: str = Field(description="One short sentence about the entity, from this paragraph only")


class Relation(BaseModel):
    source: str = Field(description="Entity name (must appear in entities)")
    relation: str = Field(description="snake_case predicate, e.g. born_in, founded_by, capital_of")
    target: str = Field(description="Entity name (must appear in entities)")
    evidence: str = Field(description="The sentence from the paragraph that states this fact")


class ExtractionResult(BaseModel):
    entities: list[Entity]
    relations: list[Relation]


class QuestionEntities(BaseModel):
    entities: list[str] = Field(description="Named entities and key noun phrases mentioned in the question")


EXTRACTION_SYSTEM = """You build a knowledge graph from encyclopedia paragraphs.
Extract every named entity and every factual relation stated in the paragraph.
- Resolve pronouns and short forms to the full entity name (the paragraph title is usually the main entity).
- Include dates, numbers and places as entities when they take part in a relation (e.g. founded_in -> 1891).
- Only extract facts stated in the paragraph. Do not add outside knowledge."""

QUESTION_ENTITIES_SYSTEM = """List the entities a search engine would need to look up to answer the question:
named entities, titles of works, and distinctive noun phrases. Use full names. Do not answer the question."""


def extract_paragraph(llm, p: Paragraph, effort: str) -> tuple[ExtractionResult, Usage]:
    result, usage = llm.parse(EXTRACTION_SYSTEM, f"Paragraph:\n{p.full_text}", ExtractionResult,
                              max_tokens=4096, effort=effort)
    return result or ExtractionResult(entities=[], relations=[]), usage


def extract_corpus(llm, corpus: list[Paragraph], cache: Path, effort: str = "low",
                   workers: int = 8) -> tuple[dict[str, ExtractionResult], Usage]:
    """Extract all paragraphs, resuming from `cache` (JSONL) so an interrupted run is not re-billed."""
    done: dict[str, ExtractionResult] = {}
    if cache.exists():
        for line in open(cache):
            row = json.loads(line)
            done[row["pid"]] = ExtractionResult.model_validate(row["result"])
    todo = [p for p in corpus if p.pid not in done]
    total, lock = Usage(), threading.Lock()
    with open(cache, "a") as out, ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(extract_paragraph, llm, p, effort): p for p in todo}
        for fut in tqdm(as_completed(futs), total=len(futs), desc="extract", disable=not todo):
            p = futs[fut]
            res, usage = fut.result()
            with lock:
                total += usage
                done[p.pid] = res
                out.write(json.dumps({"pid": p.pid, "result": res.model_dump()}) + "\n")
    return done, total
