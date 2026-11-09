"""Retrieval logic that needs no live stores: score fusion, profile chunking, query expansion.

The stub index below returns fixed hits with fixed scores, which is what makes these assertions
possible: with a real embedding model the scores move, so the only things that could be checked
are shapes. Live retrieval is covered by test_retrieval.py.
"""

from __future__ import annotations

import pytest
from langchain_core.documents import Document

from expert_finder.retrieval import (
    CHUNKED_NAMESPACE,
    LLMQueryExpander,
    VectorRetriever,
    _chunk_label,
    _license_phrase,
)


class StubIndex:
    """Stands in for Pinecone: hits and scores are given, not computed."""

    def __init__(self, hits: dict[str, list[tuple[Document, float]]],
                 parents: dict[str, Document] | None = None):
        self.hits = hits
        self.parents = parents or {}
        self.searches: list[tuple[str, str, int, dict | None]] = []

    def search(self, query, namespace, k=6, filter=None):
        self.searches.append((query, namespace, k, filter))
        return self.hits.get(query, [])[:k]

    def fetch(self, ids, namespace):
        return {i: self.parents[i] for i in ids if i in self.parents}


class StubExpander:
    def __init__(self, queries: list[str]):
        self.queries = queries

    def expand(self, query):
        return self.queries


def review(doc_id: str) -> Document:
    return Document(id=doc_id, page_content=f"review {doc_id}",
                    metadata={"doc_type": "review", "category": "hvac", "rating": 4})


def chunk(business_id: str, section: str, body: str, key: str | None = None) -> Document:
    return Document(id=f"{business_id}#{section}:{key or section}",
                    page_content=f"Acme — HVAC in Austin, TX.\nHeading:\n{body}",
                    metadata={"doc_type": "business_chunk", "section": section,
                              "business_id": business_id, "business_name": f"Name {business_id}",
                              "category": "hvac"})


def test_license_phrase_distinguishes_trades_without_a_license_requirement():
    assert "none required" in _license_phrase("none", required=False)
    assert _license_phrase("none", required=True) == "NO LICENSE on file"
    assert _license_phrase("expired", required=True) == "EXPIRED license"


# --- score fusion across query variants ---------------------------------------------------------


def test_a_document_found_by_two_phrasings_keeps_its_best_score_once():
    a, b = review("rev_1"), review("rev_2")
    index = StubIndex({"q1": [(a, 0.50), (b, 0.40)], "q2": [(a, 0.91)]})
    bundle = VectorRetriever(index, k=5, expander=StubExpander(["q1", "q2"])).retrieve(
        "original", namespaces=("reviews",))

    assert [i.record_id for i in bundle.items] == ["rev_1", "rev_2"]
    assert bundle.items[0].score == 0.91
    assert bundle.stats["queries"] == 2


def test_expansion_changes_which_k_come_back_not_how_many():
    """Prompt size has to stay stable, or the vector-vs-graph comparison moves for two reasons."""
    index = StubIndex({"q1": [(review(f"a{i}"), 0.5) for i in range(5)],
                       "q2": [(review(f"b{i}"), 0.9) for i in range(5)]})
    bundle = VectorRetriever(index, k=3, expander=StubExpander(["q1", "q2"])).retrieve(
        "original", namespaces=("reviews",))

    assert len(bundle.items) == 3
    assert all(i.record_id.startswith("b") for i in bundle.items), "the better-scoring query wins"


def test_without_an_expander_only_the_question_is_searched():
    index = StubIndex({"original": [(review("rev_1"), 0.5)]})
    bundle = VectorRetriever(index, k=3).retrieve("original", namespaces=("reviews",))

    assert [q for q, *_ in index.searches] == ["original"]
    assert bundle.stats["queries"] == 1 and "rewrites" not in bundle.stats


def test_unknown_namespace_is_rejected():
    with pytest.raises(ValueError, match="unknown namespace"):
        VectorRetriever(StubIndex({})).retrieve("q", namespaces=("nope",))


# --- business profiles: chunks match, the whole profile comes back ------------------------------


@pytest.fixture
def profile_index():
    hits = [
        (chunk("biz_0001", "service", "Tesla charger installation: $1,200-$1,800 per job.",
               key="ele_ev_charger"), 0.88),
        (chunk("biz_0001", "team", "Ana Ruiz is the lead electrician with 12 years."), 0.71),
        (chunk("biz_0002", "service", "Panel upgrade: $2,000 per job."), 0.62),
    ]
    parents = {bid: Document(id=bid, page_content=f"# Profile of {bid}\nEverything about {bid}.",
                             metadata={"doc_type": "business_profile", "business_id": bid,
                                       "business_name": f"Name {bid}", "category": "electrical"})
               for bid in ("biz_0001", "biz_0002")}
    return StubIndex({"q": hits}, parents)


def test_several_matching_chunks_collapse_to_one_profile_per_business(profile_index):
    bundle = VectorRetriever(profile_index, k=5).retrieve("q", namespaces=(CHUNKED_NAMESPACE,))

    assert [i.record_id for i in bundle.items] == ["biz_0001", "biz_0002"]
    assert [i.ref for i in bundle.items] == ["B1", "B2"]
    assert bundle.items[0].score == 0.88, "the business keeps its best chunk's score"


def test_the_model_reads_the_whole_profile_not_the_chunk(profile_index):
    bundle = VectorRetriever(profile_index, k=5).retrieve("q", namespaces=(CHUNKED_NAMESPACE,))
    text = bundle.items[0].text

    assert "Everything about biz_0001." in text, "the parent profile is what comes back"
    assert "Matched on: Tesla charger installation, Ana Ruiz." in text, "and why it matched"
    assert bundle.items[0].source["matched_sections"] == ["Tesla charger installation", "Ana Ruiz"]


def test_only_chunks_are_searched_and_overfetched(profile_index):
    VectorRetriever(profile_index, k=3, chunk_overfetch=4).retrieve(
        "q", namespaces=(CHUNKED_NAMESPACE,), filters={CHUNKED_NAMESPACE: {"category": "electrical"}})

    (_, namespace, k, filter) = profile_index.searches[0]
    assert namespace == CHUNKED_NAMESPACE
    assert k == 12, "k businesses need more than k chunks, because one company can match several"
    assert filter == {"category": "electrical", "doc_type": "business_chunk"}


def test_a_missing_parent_does_not_lose_the_hit():
    """An index built before profiles were chunked has no parent to fetch."""
    index = StubIndex({"q": [(chunk("biz_0001", "service", "Panel upgrade: $2,000 per job."), 0.8)]})
    bundle = VectorRetriever(index, k=3).retrieve("q", namespaces=(CHUNKED_NAMESPACE,))

    assert len(bundle.items) == 1
    assert bundle.items[0].record_id == "biz_0001"
    assert bundle.items[0].source["business_name"] == "Name biz_0001"


@pytest.mark.parametrize("section, body, expected", [
    ("service", "Drain cleaning: $140-$355 per job (typical $225).", "Drain cleaning"),
    ("team", "Ricardo Hernandez is the owner with 26 years in the trade.", "Ricardo Hernandez"),
    ("credentials", "Licenses: Master Plumber, active.", "credentials"),
])
def test_chunk_label_names_what_matched(section, body, expected):
    assert _chunk_label(chunk("biz_0001", section, body)) == expected


# --- query expansion ----------------------------------------------------------------------------


class FakeModel:
    """Minimal stand-in for a chat model bound to structured output."""

    def __init__(self, queries=None, error=None):
        self.queries = queries or []
        self.error = error
        self.calls = 0

    def with_structured_output(self, _schema):
        return self

    def invoke(self, _prompt):
        self.calls += 1
        if self.error:
            raise self.error
        return type("R", (), {"queries": list(self.queries)})()


def test_expansion_puts_the_original_first_and_caps_the_rewrites():
    model = FakeModel(["AC not cooling", "refrigerant recharge", "condenser fan replacement"])
    expander = LLMQueryExpander(model, n=2)

    assert expander.expand("my AC blows warm air") == [
        "my AC blows warm air", "AC not cooling", "refrigerant recharge"]


def test_a_rewrite_identical_to_the_question_is_not_searched_twice():
    expander = LLMQueryExpander(FakeModel(["same question", "something else"]), n=2)
    assert expander.expand("same question") == ["same question", "something else"]


def test_rewrites_are_cached_because_several_tools_retrieve_on_one_question():
    model = FakeModel(["other phrasing"])
    expander = LLMQueryExpander(model, n=1)
    expander.expand("q")
    expander.expand("q")
    assert model.calls == 1


def test_a_failed_rewrite_still_searches_the_original_question():
    expander = LLMQueryExpander(FakeModel(error=RuntimeError("rate limited")), n=2)
    assert expander.expand("leaking water heater") == ["leaking water heater"]


def test_expansion_can_be_switched_off_for_a_comparison_run():
    model = FakeModel(["unused"])
    assert LLMQueryExpander(model, n=0).expand("q") == ["q"]
    assert model.calls == 0
