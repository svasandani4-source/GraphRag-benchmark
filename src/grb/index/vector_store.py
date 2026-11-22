"""Exact cosine-similarity index over paragraph embeddings (vectors are L2-normalised)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..types import Paragraph


class VectorIndex:
    def __init__(self, pids: list[str], matrix: np.ndarray):
        self.pids = pids
        self.matrix = matrix

    @classmethod
    def build(cls, corpus: list[Paragraph], embedder) -> "VectorIndex":
        return cls([p.pid for p in corpus], embedder.embed_documents([p.full_text for p in corpus]))

    def search(self, qvec: np.ndarray, k: int) -> list[tuple[str, float]]:
        scores = self.matrix @ qvec
        top = np.argsort(-scores)[:k]
        return [(self.pids[i], float(scores[i])) for i in top]

    def save(self, d: Path) -> None:
        np.save(d / "vectors.npy", self.matrix)
        (d / "vector_pids.json").write_text(json.dumps(self.pids))

    @classmethod
    def load(cls, d: Path) -> "VectorIndex":
        return cls(json.loads((d / "vector_pids.json").read_text()), np.load(d / "vectors.npy"))
