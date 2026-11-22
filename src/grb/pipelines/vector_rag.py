from ..types import Retrieval, Usage
from .base import Pipeline


class VectorRAG(Pipeline):
    """Standard dense retrieval: top-k paragraphs by cosine similarity."""
    name = "vector_rag"

    def search(self, question: str, k: int) -> list[tuple[str, float]]:
        return self.res.vector_index.search(self.res.embedder.embed_query(question), k)

    def retrieve(self, q):
        hits = self.search(q.question, self.cfg["top_k"])
        context, kept = self.paragraphs_block([pid for pid, _ in hits], self.cfg["max_context_tokens"])
        return Retrieval(pids=kept, context=context, debug={"scores": [round(s, 4) for _, s in hits]}), Usage()
