from .base import Pipeline, Resources
from .closed_book import ClosedBook
from .graph_rag import GraphRAG
from .hybrid import HybridRAG
from .vector_rag import VectorRAG

PIPELINES: dict[str, type[Pipeline]] = {
    p.name: p for p in (ClosedBook, VectorRAG, GraphRAG, HybridRAG)
}

__all__ = ["PIPELINES", "Pipeline", "Resources"]
