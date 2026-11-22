from grb.graph.extract import Entity, ExtractionResult, Relation
from grb.graph.store import KnowledgeGraph, norm_entity


def _kg():
    ex = {
        "p1": ExtractionResult(entities=[Entity(name="Velmora Institute", type="Organization", description=""),
                                         Entity(name="Idris Kael", type="Person", description="")],
                               relations=[Relation(source="Velmora Institute", relation="founded_by",
                                                   target="Idris Kael", evidence="")]),
        "p2": ExtractionResult(entities=[Entity(name="Idris Kael", type="Person", description=""),
                                         Entity(name="Brennock", type="Location", description="")],
                               relations=[Relation(source="Idris Kael", relation="born_in",
                                                   target="Brennock", evidence="")]),
    }
    return KnowledgeGraph.build(ex)


def test_entity_resolution_merges_across_paragraphs():
    kg = _kg()
    assert kg.stats()["nodes"] == 3
    assert set(kg.g.nodes[norm_entity("Idris Kael")]["pids"]) == {"p1", "p2"}


def test_two_hop_expansion_reaches_bridge_paragraph():
    triples, pid_scores = _kg().expand([norm_entity("Velmora Institute")], hops=2)
    assert {"p1", "p2"} <= set(pid_scores)
    assert any(t[2] == "born_in" for t in triples)


def test_roundtrip(tmp_path):
    kg = _kg()
    kg.save(tmp_path / "g.json")
    assert KnowledgeGraph.load(tmp_path / "g.json").stats() == kg.stats()
