"""Retrieval layer. Marked integration: needs live Pinecone and Neo4j, skipped without credentials."""

import pytest

from expert_finder.config import get_settings
from expert_finder.retrieval import DEFAULT_NAMESPACES, EvidenceBundle, GraphRetriever, VectorRetriever, _license_phrase

pytestmark = pytest.mark.integration

QUERY = "A sprinkler head is geysering in my Tarrytown yard, who do neighbors recommend?"


@pytest.fixture(scope="module")
def settings():
    s = get_settings()
    if not (s.pinecone_api_key and s.openai_api_key):
        pytest.skip("needs PINECONE_API_KEY and OPENAI_API_KEY")
    return s


@pytest.fixture(scope="module")
def index(settings):
    from expert_finder.vector_store import VectorIndex

    vi = VectorIndex(settings)
    if not vi.exists():
        pytest.skip("Pinecone index not built; run `expert-finder index`")
    return vi


@pytest.fixture(scope="module")
def driver(settings):
    from expert_finder.graph_db import neo4j_driver

    try:
        d = neo4j_driver(settings)
        d.verify_connectivity()
    except Exception as e:
        pytest.skip(f"Neo4j unavailable: {type(e).__name__}")
    yield d
    d.close()


@pytest.fixture(scope="module")
def vector_bundle(index):
    return VectorRetriever(index, k=4).retrieve(QUERY, namespaces=("social", "reviews"))


@pytest.fixture(scope="module")
def graph_bundle(index, driver, settings):
    return GraphRetriever(index, driver, settings.neo4j_database, k=4).retrieve(
        QUERY, namespaces=("social", "reviews"))


def test_license_phrase_distinguishes_trades_without_a_license_requirement():
    assert "none required" in _license_phrase("none", required=False)
    assert _license_phrase("none", required=True) == "NO LICENSE on file"
    assert _license_phrase("expired", required=True) == "EXPIRED license"


def test_vector_bundle_has_unique_refs_and_real_record_ids(vector_bundle):
    assert vector_bundle.mode == "vector" and vector_bundle.items
    refs = [i.ref for i in vector_bundle.items]
    assert len(refs) == len(set(refs))
    for i in vector_bundle.items:
        assert i.record_id and i.score is not None and i.via == "vector"
        assert i.ref[0] in "RPB"


def test_vector_retrieval_rejects_unknown_namespace(index):
    with pytest.raises(ValueError, match="unknown namespace"):
        VectorRetriever(index).retrieve(QUERY, namespaces=("nope",))


def test_pinecone_filters_are_applied(index):
    bundle = VectorRetriever(index, k=5).retrieve("AC stopped cooling", namespaces=("reviews",),
                                                  filters={"reviews": {"category": "hvac"}})
    assert bundle.items and all(i.source["category"] == "hvac" for i in bundle.items)


def test_graph_mode_is_a_superset_of_vector_mode(vector_bundle, graph_bundle):
    """Graph retrieval keeps every vector hit and adds to it; that is what makes the comparison fair."""
    assert graph_bundle.mode == "graph"
    vector_ids = {i.record_id for i in vector_bundle.items}
    graph_vector_ids = {i.record_id for i in graph_bundle.items if i.via == "vector"}
    assert vector_ids == graph_vector_ids
    assert len(graph_bundle.items) > len(vector_bundle.items)
    assert graph_bundle.stats["graph_items"] > 0


def test_graph_mode_resolves_posts_to_identifiable_businesses(graph_bundle):
    facts = [i.text for i in graph_bundle.items if i.source.get("relation") == "post_mentions"]
    assert facts, "expected at least one post resolved to a business"
    assert any("phone" in f for f in facts)


def test_graph_mode_pulls_in_the_rest_of_the_thread(graph_bundle):
    threads = graph_bundle.by_kind("thread")
    assert threads
    retrieved_ids = {i.record_id for i in graph_bundle.items if i.via == "vector"}
    assert all(t.record_id not in retrieved_ids for t in threads)
    assert all(t.source["thread_id"] for t in threads)


def test_prompt_includes_every_ref_and_groups_by_kind(graph_bundle):
    prompt = graph_bundle.to_prompt()
    for item in graph_bundle.items:
        assert f"[{item.ref}]" in prompt
    assert "## Customer reviews" in prompt or "## Social posts" in prompt


def test_empty_bundle_renders_without_error():
    assert EvidenceBundle(query="x", mode="vector").to_prompt() == "No evidence found."


def test_default_namespaces_are_known(index):
    bundle = VectorRetriever(index, k=1).retrieve("new roof after hail", namespaces=DEFAULT_NAMESPACES)
    assert set(bundle.stats["namespaces"]) == set(DEFAULT_NAMESPACES)
