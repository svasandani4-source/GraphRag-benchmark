"""Dataset loaders. Each returns (questions, corpus) where the corpus is the de-duplicated
union of every paragraph (gold + distractor) across the sampled questions. All retrievers
search this one shared corpus."""
from __future__ import annotations

import json
import random
import re
from dataclasses import asdict
from importlib import resources
from pathlib import Path

from ..types import Paragraph, Question


def _pid(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", title.lower()).strip("_")[:80]


def load_fixture() -> tuple[list[Question], list[Paragraph]]:
    raw = json.loads(resources.files("grb.data.fixtures").joinpath("synthetic_multihop.json").read_text())
    corpus = [Paragraph(pid=_pid(p["title"]), title=p["title"], text=p["text"]) for p in raw["paragraphs"]]
    qs = [Question(qid=q["id"], question=q["question"], answer=q["answer"], qtype=q["type"],
                   hops=q["hops"], supporting_pids=[_pid(t) for t in q["supporting_titles"]])
          for q in raw["questions"]]
    return qs, corpus


def load_hotpotqa(split: str, n: int, seed: int,
                  distractors: int | None = None) -> tuple[list[Question], list[Paragraph]]:
    """`distractors` caps non-gold paragraphs per question (HotpotQA has 8) to shrink the corpus
    so local-model graph extraction stays feasible. Retrieval is still over the whole shared pool."""
    from datasets import load_dataset

    ds = load_dataset("hotpotqa/hotpot_qa", "distractor", split=split)
    # Stratify: half bridge, half comparison, fixed seed.
    rng = random.Random(seed)
    by_type: dict[str, list[int]] = {}
    for i, t in enumerate(ds["type"]):
        by_type.setdefault(t, []).append(i)
    idx: list[int] = []
    for t, ids in sorted(by_type.items()):
        idx += rng.sample(ids, min(len(ids), n // len(by_type)))
    rng.shuffle(idx)

    questions, corpus = [], {}
    for row in ds.select(idx):
        gold_titles = set(row["supporting_facts"]["title"])
        kept_distractors = 0
        for title, sents in zip(row["context"]["title"], row["context"]["sentences"]):
            if title not in gold_titles:
                if distractors is not None and kept_distractors >= distractors:
                    continue
                kept_distractors += 1
            pid = _pid(title)
            corpus.setdefault(pid, Paragraph(pid=pid, title=title, text="".join(sents).strip()))
        gold = sorted({_pid(t) for t in gold_titles})
        questions.append(Question(qid=row["id"], question=row["question"], answer=row["answer"],
                                  qtype=row["type"], hops=2, supporting_pids=gold))
    return questions, list(corpus.values())


def load_dataset_sample(name: str, split: str = "validation", n: int = 50, seed: int = 13,
                        distractors: int | None = None):
    if name == "fixture":
        qs, corpus = load_fixture()
        return qs[:n], corpus
    if name == "hotpotqa":
        return load_hotpotqa(split, n, seed, distractors)
    if name == "musique":
        raise NotImplementedError("MuSiQue loader is milestone M3 (see docs/PLAN.md)")
    raise ValueError(f"unknown dataset {name!r}")


def write_jsonl(path: Path, items) -> None:
    with open(path, "w") as f:
        for it in items:
            f.write(json.dumps(asdict(it) if hasattr(it, "__dataclass_fields__") else it) + "\n")


def read_corpus(path: Path) -> list[Paragraph]:
    return [Paragraph(**json.loads(l)) for l in open(path)]


def read_questions(path: Path) -> list[Question]:
    return [Question(**json.loads(l)) for l in open(path)]
