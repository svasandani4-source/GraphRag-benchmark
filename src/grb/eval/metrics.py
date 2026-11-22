"""Answer and retrieval metrics. EM/F1 follow the official HotpotQA evaluation script."""
from __future__ import annotations

import re
import string
from collections import Counter

import numpy as np


def normalize_answer(s: str) -> str:
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def exact_match(pred: str, gold: str) -> float:
    return float(normalize_answer(pred) == normalize_answer(gold))


def f1_score(pred: str, gold: str) -> float:
    p, g = normalize_answer(pred), normalize_answer(gold)
    if p in {"yes", "no", "noanswer"} or g in {"yes", "no", "noanswer"}:
        return float(p == g)
    pt, gt = p.split(), g.split()
    common = Counter(pt) & Counter(gt)
    same = sum(common.values())
    if same == 0:
        return 0.0
    precision, recall = same / len(pt), same / len(gt)
    return 2 * precision * recall / (precision + recall)


def support_recall(retrieved: list[str], gold: list[str]) -> float:
    return len(set(retrieved) & set(gold)) / len(gold) if gold else float("nan")


def all_support(retrieved: list[str], gold: list[str]) -> float:
    return float(bool(gold) and set(gold) <= set(retrieved))


def bootstrap_ci(values, n: int = 2000, alpha: float = 0.05, seed: int = 0) -> tuple[float, float]:
    v = np.asarray([x for x in values if x == x], dtype=float)   # drop NaN
    if len(v) == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    means = rng.choice(v, size=(n, len(v)), replace=True).mean(axis=1)
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))
