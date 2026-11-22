from ..types import Retrieval
from .base import approx_tokens
from .graph_rag import GraphRAG


class HybridRAG(GraphRAG):
    """Vector top-k for recall, graph expansion from question entities *and* from entities found in
    the vector hits to pull in the bridge paragraphs that pure similarity misses."""
    name = "hybrid"

    def retrieve(self, q):
        k = self.cfg["top_k"]
        vec_hits = [pid for pid, _ in self.res.vector_index.search(self.res.embedder.embed_query(q.question), k)]
        mentions, seeds, usage = self.seeds(q.question)
        seeds = list(dict.fromkeys(seeds + self.res.graph.entities_in(set(vec_hits[:2]))))[: 2 * self.cfg["graph_seed_entities"]]
        triples, pid_scores = self.res.graph.expand(seeds, hops=1)
        graph_pids = [p for p in sorted(pid_scores, key=pid_scores.get, reverse=True) if p not in vec_hits]
        n_vec = (k + 1) // 2
        merged = vec_hits[:n_vec] + graph_pids[: k - n_vec]
        merged += [p for p in vec_hits[n_vec:] if len(merged) < k]
        budget = self.cfg["max_context_tokens"]
        facts = self.facts_block(triples, budget // 4)
        paras, kept = self.paragraphs_block(merged, budget - approx_tokens(facts))
        context = (f"Knowledge graph facts:\n{facts}\n\nSource passages:\n{paras}" if facts else paras)
        return Retrieval(pids=kept, context=context,
                         debug={"mentions": mentions, "seeds": seeds, "n_triples": len(triples)}), usage
