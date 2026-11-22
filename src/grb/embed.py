"""Text embedders. `BgeEmbedder` for real runs, `HashingEmbedder` for offline tests."""
from __future__ import annotations

import hashlib
import re

import numpy as np

BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class HashingEmbedder:
    """Feature-hashed bag of words + bigrams, L2-normalised. Deterministic and dependency-free."""

    def __init__(self, dim: int = 1024):
        self.dim = dim

    def _vec(self, text: str) -> np.ndarray:
        toks = re.findall(r"[a-z0-9]+", text.lower())
        grams = toks + [a + "_" + b for a, b in zip(toks, toks[1:])]
        v = np.zeros(self.dim, dtype=np.float32)
        for g in grams:
            h = int(hashlib.md5(g.encode()).hexdigest(), 16)
            v[h % self.dim] += 1.0 if (h >> 64) & 1 else -1.0
        n = np.linalg.norm(v)
        return v / n if n else v

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return np.stack([self._vec(t) for t in texts]) if texts else np.zeros((0, self.dim), np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        return self._vec(text)


class BgeEmbedder:
    def __init__(self, model: str = "BAAI/bge-base-en-v1.5"):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model)

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return self.model.encode(texts, batch_size=64, normalize_embeddings=True,
                                 show_progress_bar=len(texts) > 500).astype(np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        return self.model.encode(BGE_QUERY_PREFIX + text, normalize_embeddings=True).astype(np.float32)


def make_embedder(name: str):
    return HashingEmbedder() if name == "hashing" else BgeEmbedder(name)
