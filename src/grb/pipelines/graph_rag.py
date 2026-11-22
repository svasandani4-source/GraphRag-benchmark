from ..graph.extract import QUESTION_ENTITIES_SYSTEM, QuestionEntities
from ..types import Retrieval, Usage
from .base import Pipeline, approx_tokens


class GraphRAG(Pipeline):
    """Entity-centric retrieval: link question entities to graph nodes, expand k hops,
    and give the generator the facts (triples) plus the source paragraphs of the best-supported facts."""
    name = "graph_rag"

    def seeds(self, question: str) -> tuple[list[str], list[str], Usage]:
        ents, usage = self.res.llm.parse(QUESTION_ENTITIES_SYSTEM, f"Question: {question}", QuestionEntities,
                                         max_tokens=512, effort=self.cfg["effort_generation"])
        mentions = ents.entities if ents else []
        qvec = self.res.embedder.embed_query(question)
        seeds = self.res.graph.link(mentions, self.res.embedder, qvec, n_seeds=self.cfg["graph_seed_entities"])
        return mentions, seeds, usage

    def facts_block(self, triples, budget: int) -> str:
        lines, used = [], 0
        for _, s, rel, t, _pid in triples:
            line = f"- {s} -[{rel}]-> {t}"
            used += approx_tokens(line)
            if used > budget:
                break
            lines.append(line)
        return "\n".join(lines)

    def retrieve(self, q):
        mentions, seeds, usage = self.seeds(q.question)
        triples, pid_scores = self.res.graph.expand(seeds, hops=self.cfg["graph_hops"])
        ranked = sorted(pid_scores, key=pid_scores.get, reverse=True)[: self.cfg["top_k"]]
        budget = self.cfg["max_context_tokens"]
        facts = self.facts_block(triples, budget // 4)
        paras, kept = self.paragraphs_block(ranked, budget - approx_tokens(facts))
        context = (f"Knowledge graph facts:\n{facts}\n\nSource passages:\n{paras}" if facts else paras)
        return Retrieval(pids=kept, context=context,
                         debug={"mentions": mentions, "seeds": seeds, "n_triples": len(triples)}), usage
