from ..types import Retrieval, Usage
from .base import CLOSED_BOOK_SYSTEM, Pipeline


class ClosedBook(Pipeline):
    """Lower bound: no retrieval at all."""
    name = "closed_book"
    system = CLOSED_BOOK_SYSTEM

    def retrieve(self, q):
        return Retrieval(pids=[], context=""), Usage()
