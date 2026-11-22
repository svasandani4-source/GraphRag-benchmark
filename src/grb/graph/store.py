"""In-process knowledge graph (NetworkX) with entity linking and k-hop retrieval."""
from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

import networkx as nx
import numpy as np

from .extract import ExtractionResult


def norm_entity(name: str) -> str:
    s = re.sub(r"[^a-z0-9 ]+", " ", name.lower())
    s = re.sub(r"^(the|a|an) ", "", s.strip())
    return re.sub(r"\s+", " ", s).strip()


class KnowledgeGraph:
    def __init__(self, g: nx.MultiDiGraph | None = None):
        self.g = g if g is not None else nx.MultiDiGraph()
        self._keys: list[str] = []
        self._name_vecs: np.ndarray | None = None

    # ------------------------------------------------------------------ build
    @classmethod
    def build(cls, extractions: dict[str, ExtractionResult]) -> "KnowledgeGraph":
        kg = cls()
        g = kg.g
        for pid, res in extractions.items():
            for e in res.entities:
                k = norm_entity(e.name)
                if not k:
                    continue
                if k not in g:
                    g.add_node(k, name=e.name, type=e.type, descriptions=[], pids=[])
                node = g.nodes[k]
                if pid not in node["pids"]:
                    node["pids"].append(pid)
                if e.description and len(node["descriptions"]) < 5:
                    node["descriptions"].append(e.description)
            for r in res.relations:
                s, t = norm_entity(r.source), norm_entity(r.target)
                if not s or not t or s == t:
                    continue
                for k, name in ((s, r.source), (t, r.target)):
                    if k not in g:
                        g.add_node(k, name=name, type="Thing", descriptions=[], pids=[pid])
                g.add_edge(s, t, relation=r.relation, evidence=r.evidence, pid=pid)
        return kg

    def stats(self) -> dict:
        return {"nodes": self.g.number_of_nodes(), "edges": self.g.number_of_edges(),
                "components": nx.number_weakly_connected_components(self.g) if len(self.g) else 0}

    # ------------------------------------------------------------------ entity linking
    def index_names(self, embedder) -> None:
        self._keys = list(self.g.nodes)
        names = [self.g.nodes[k]["name"] for k in self._keys]
        self._name_vecs = embedder.embed_documents(names)

    def link(self, mentions: list[str], embedder, qvec: np.ndarray | None = None,
             n_seeds: int = 5, min_sim: float = 0.75) -> list[str]:
        """Map question mentions to graph nodes: exact normalised match first, then nearest name."""
        seeds: list[str] = []
        for m in mentions:
            k = norm_entity(m)
            if k in self.g:
                seeds.append(k)
            elif self._name_vecs is not None and len(self._keys):
                sims = self._name_vecs @ embedder.embed_query(m)
                i = int(np.argmax(sims))
                if sims[i] >= min_sim:
                    seeds.append(self._keys[i])
        # Fall back to question-embedding neighbours only when linking found too little,
        # so a confident link is not diluted by loosely related seeds.
        if qvec is not None and self._name_vecs is not None and len(seeds) < 2:
            for i in np.argsort(-(self._name_vecs @ qvec))[: 2 - len(seeds)]:
                seeds.append(self._keys[i])
        return list(dict.fromkeys(seeds))[:n_seeds]

    # ------------------------------------------------------------------ retrieval
    def expand(self, seeds: list[str], hops: int = 2, max_triples: int = 40):
        """BFS over the undirected view. Returns (triples, pid_scores) with closer facts first."""
        und = self.g.to_undirected(as_view=True)
        dist: dict[str, int] = {}
        frontier = [s for s in seeds if s in self.g]
        for s in frontier:
            dist[s] = 0
        for h in range(1, hops + 1):
            nxt = []
            for u in frontier:
                for v in und.neighbors(u):
                    if v not in dist:
                        dist[v] = h
                        nxt.append(v)
            frontier = nxt

        triples, pid_scores = [], defaultdict(float)
        for u, v, data in self.g.edges(data=True):
            if u in dist and v in dist:
                d = max(dist[u], dist[v])
                if d > hops:
                    continue
                triples.append((d, self.g.nodes[u]["name"], data["relation"], self.g.nodes[v]["name"], data["pid"]))
                pid_scores[data["pid"]] += 1.0 / (1 + d)
        for k, d in dist.items():
            for pid in self.g.nodes[k]["pids"]:
                pid_scores[pid] += 0.5 / (1 + d)
        triples.sort(key=lambda t: t[0])
        return triples[:max_triples], dict(pid_scores)

    def entities_in(self, pids: set[str]) -> list[str]:
        return [k for k, d in self.g.nodes(data=True) if pids & set(d["pids"])]

    # ------------------------------------------------------------------ io
    def save(self, path: Path) -> None:
        path.write_text(json.dumps(nx.node_link_data(self.g, edges="links")))

    @classmethod
    def load(cls, path: Path) -> "KnowledgeGraph":
        return cls(nx.node_link_graph(json.loads(path.read_text()), directed=True, multigraph=True, edges="links"))
