"""The LangSmith payload processors.

These need tests precisely because langsmith fails closed: a processor that raises loses its
payload and logs a warning, so a bug here shows up as traces that are quietly missing their
inputs or outputs rather than as anything breaking. The two things that can go wrong are leaving
an unserialisable object in (the Pinecone client, the Neo4j driver) and letting a payload grow to
a second copy of the evidence, so both are checked directly.
"""

from __future__ import annotations

import json

import pytest
from langchain_core.documents import Document

from expert_finder import tracing
from expert_finder.config import get_settings
from expert_finder.retrieval import Evidence, EvidenceBundle


class Unserialisable:
    """Stands in for a Pinecone client or a Neo4j driver: holds a socket, cannot be JSON."""

    def __init__(self):
        self.socket = object()


def serialisable(payload: dict) -> bool:
    try:
        json.dumps(payload)
    except (TypeError, ValueError):
        return False
    return True


@pytest.fixture
def bundle():
    return EvidenceBundle(
        query="tankless water heater", mode="graph",
        items=[
            Evidence(ref="B1", kind="business", text="x" * 1200, score=0.51, via="vector",
                     source={"id": "biz_0009", "matched_sections": ["Tankless water heater installation"]}),
            Evidence(ref="G1", kind="graph", text="Review rev_1 is about Acme.", via="graph",
                     source={"id": "rev_1", "relation": "review_context"}),
        ],
        stats={"items": 2, "queries": 3, "graph_items": 1},
        notes=["Independent sources naming the same business: Acme in 2 retrieved sources"])


# --- the failure mode that matters: unserialisable plumbing ------------------------------------


def test_the_retriever_itself_is_not_sent_to_langsmith(bundle):
    processed = tracing.retriever_inputs(
        {"self": Unserialisable(), "query": "tankless", "namespaces": ("businesses",),
         "k": 6, "filters": None, "refs": Unserialisable()})

    assert processed == {"query": "tankless", "namespaces": ("businesses",), "k": 6, "filters": None}
    assert serialisable(processed)


def test_the_neo4j_driver_is_not_sent_to_langsmith():
    processed = tracing.query_inputs(
        {"driver": Unserialisable(), "database": "neo4j", "review_ids": ["rev_1", "rev_2"]})

    assert processed == {"review_ids": ["rev_1", "rev_2"]}
    assert serialisable(processed)


def test_the_pinecone_index_is_not_sent_to_langsmith():
    processed = tracing.index_inputs(
        {"self": Unserialisable(), "query": "tankless", "namespace": "businesses", "k": 6})

    assert "self" not in processed
    assert serialisable(processed)


@pytest.mark.parametrize("name", ["bundle_outputs", "hits_outputs", "fetched_outputs",
                                  "rows_outputs", "expansion_outputs"])
def test_every_output_processor_produces_json(name, bundle):
    payloads = {
        "bundle_outputs": bundle,
        "hits_outputs": [(Document(id="biz_0009#service:x", page_content="a price line",
                                   metadata={"doc_type": "business_chunk", "section": "service",
                                             "business_id": "biz_0009"}), 0.512345)],
        "fetched_outputs": {"biz_0009": Document(id="biz_0009", page_content="the whole profile")},
        "rows_outputs": [{"business_id": "biz_0009", "avg_rating": 4.7}],
        "expansion_outputs": ["original", "rewrite one"],
    }
    assert serialisable(getattr(tracing, name)(payloads[name]))


# --- payload size ------------------------------------------------------------------------------


def test_document_text_is_truncated_rather_than_duplicated(bundle):
    """The tool's own output already carries the full text; the span only has to identify it."""
    documents = tracing.bundle_outputs(bundle)["documents"]
    first = documents[0]["page_content"]

    assert len(first) < 1200
    assert first.startswith("x" * tracing.DOC_TEXT_LIMIT)
    assert "+700 chars" in first, "the reader is told what was cut"


def test_short_text_is_left_alone(bundle):
    assert tracing.bundle_outputs(bundle)["documents"][1]["page_content"] == "Review rev_1 is about Acme."


# --- what the spans should actually say ---------------------------------------------------------


def test_a_retrieval_span_carries_the_comparison_stats(bundle):
    """stats is the part a vector-vs-graph comparison is read from."""
    processed = tracing.bundle_outputs(bundle)
    assert processed["stats"] == {"items": 2, "queries": 3, "graph_items": 1}
    assert processed["notes"] == bundle.notes


def test_a_retrieval_span_says_how_each_item_was_found(bundle):
    metadata = [d["metadata"] for d in tracing.bundle_outputs(bundle)["documents"]]
    assert [m["via"] for m in metadata] == ["vector", "graph"]
    assert [m["ref"] for m in metadata] == ["B1", "G1"]
    assert metadata[0]["matched_sections"] == ["Tankless water heater installation"]
    assert metadata[1]["relation"] == "review_context"


def test_a_pinecone_span_keeps_the_score_and_which_chunk_matched():
    doc = Document(id="biz_0009#service:plb_tankless", page_content="Tankless: $1,200",
                   metadata={"doc_type": "business_chunk", "section": "service",
                             "business_id": "biz_0009"})
    processed = tracing.hits_outputs([(doc, 0.5123456)])

    assert processed["hits"] == 1
    assert processed["documents"][0]["metadata"] == {
        "id": "biz_0009#service:plb_tankless", "score": 0.5123, "doc_type": "business_chunk",
        "section": "service", "business_id": "biz_0009"}


def test_a_fetched_profile_has_no_score_because_it_was_not_matched():
    processed = tracing.fetched_outputs({"biz_0009": Document(id="biz_0009", page_content="profile")})
    assert processed == {"documents": [{"page_content": "profile", "metadata": {"id": "biz_0009"}}],
                         "fetched": 1}


def test_cypher_results_keep_their_rows_whether_a_list_or_a_dict():
    assert tracing.rows_outputs([{"a": 1}, {"a": 2}]) == {"rows": 2, "result": [{"a": 1}, {"a": 2}]}
    # review_authenticity returns a dict rather than rows.
    assert tracing.rows_outputs({"business_id": "biz_0009"}) == {"result": {"business_id": "biz_0009"}}


def test_expansion_reports_the_phrasings_that_were_searched():
    assert tracing.expansion_outputs(["original", "a", "b"]) == {
        "queries": ["original", "a", "b"], "rewrites": 2}


# --- the root run -------------------------------------------------------------------------------


def test_the_root_run_is_named_and_tagged_by_mode():
    """Otherwise every trace in the project is called "LangGraph" and cannot be filtered."""
    config = tracing.agent_config("graph", "cli", get_settings())

    assert config["run_name"] == "expert-finder-graph"
    assert "mode:graph" in config["tags"]
    assert config["metadata"]["mode"] == "graph"
    assert serialisable({"tags": config["tags"], "metadata": config["metadata"]})


def test_the_thread_id_still_reaches_the_checkpointer():
    """agent_config replaces the invoke config, so losing this would silently break memory."""
    config = tracing.agent_config("vector", "session-42", get_settings())
    assert config["configurable"]["thread_id"] == "session-42"
