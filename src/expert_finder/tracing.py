"""Explicit LangSmith tracing for the parts LangChain cannot see by itself.

LangChain traces its own objects for free: the LangGraph nodes, the model calls and the @tool
functions. It cannot trace a Pinecone query or a Cypher statement, because neither is a Runnable.
The cost of that gap is measurable - on a recorded run of one profile search, 3.4 of the 4.1
seconds inside the tool span were unaccounted for, being three Pinecone queries and a parent
profile fetch. Retrieval is exactly what this project compares, so it is what gets instrumented
here; the SQL tools are left alone, since they are already visible as tool spans.

Everything in this module is a payload processor, and they exist for two reasons.

* Correctness: a traced call's inputs include `self` and the Neo4j driver, which hold a Pinecone
  client and a socket pool. Those cannot be serialised, so they are stripped rather than left to
  fail at send time.
* Size: the default for a pydantic return value is `model_dump()`, so an untouched retrieval span
  would carry every evidence item in full - text the tool's own output already sends. These
  summarise instead, and truncate document text, because the span is there to show what was
  retrieved and how it scored, not to be a second copy of it.

A processor that raises loses its payload but not the run (langsmith fails closed), and when
tracing is off `@traceable` does nothing, so none of this runs in tests or offline.
"""

from __future__ import annotations

from typing import Any

# Enough of a document to recognise it in a trace, without duplicating the tool output.
DOC_TEXT_LIMIT = 500


def _drop(inputs: dict, *names: str) -> dict:
    """Inputs without the unserialisable plumbing (`self`, the driver) and internal bookkeeping."""
    return {k: v for k, v in inputs.items() if k not in names}


def _clip(text: str) -> str:
    return text if len(text) <= DOC_TEXT_LIMIT else text[:DOC_TEXT_LIMIT] + f"... (+{len(text) - DOC_TEXT_LIMIT} chars)"


# --- retrieval ----------------------------------------------------------------------------------


def retriever_inputs(inputs: dict) -> dict:
    """What was asked for. `refs` is a counter shared with the caller, not an input worth tracing."""
    return _drop(inputs, "self", "refs")


def bundle_outputs(bundle: Any) -> dict:
    """An EvidenceBundle as a retriever span: what came back, how, and how well it scored.

    `stats` is the useful part for the vector-vs-graph comparison - items per namespace, how many
    came from the graph, how many businesses the graph added, and the rewrites that were searched.
    """
    return {
        "documents": [{"page_content": _clip(item.text),
                       "metadata": {"ref": item.ref, "id": item.record_id, "kind": item.kind,
                                    "via": item.via, "score": item.score,
                                    "relation": item.source.get("relation"),
                                    "matched_sections": item.source.get("matched_sections")}}
                      for item in bundle.items],
        "stats": bundle.stats,
        "notes": bundle.notes,
    }


def expansion_outputs(queries: list[str]) -> dict:
    """The phrasings that were actually searched, which is the thing worth reading back later."""
    return {"queries": queries, "rewrites": len(queries) - 1}


# --- the vector index ---------------------------------------------------------------------------


def index_inputs(inputs: dict) -> dict:
    return _drop(inputs, "self")


def hits_outputs(hits: list[tuple[Any, float]]) -> dict:
    """similarity_search_with_score returns (document, score) pairs."""
    return {
        "documents": [{"page_content": _clip(doc.page_content),
                       "metadata": {"id": doc.id, "score": round(score, 4),
                                    "doc_type": doc.metadata.get("doc_type"),
                                    "section": doc.metadata.get("section"),
                                    "business_id": doc.metadata.get("business_id")}}
                      for doc, score in hits],
        "hits": len(hits),
    }


def fetched_outputs(documents: dict[str, Any]) -> dict:
    """fetch() returns documents by id, with no scores: these were looked up, not matched."""
    return {"documents": [{"page_content": _clip(doc.page_content), "metadata": {"id": doc_id}}
                          for doc_id, doc in documents.items()],
            "fetched": len(documents)}


# --- graph queries ------------------------------------------------------------------------------


def query_inputs(inputs: dict) -> dict:
    """The parameters the Cypher was run with. The driver and database name are not data."""
    return _drop(inputs, "driver", "database")


def rows_outputs(result: Any) -> dict:
    """Cypher results, summarised. Rows are facts rather than documents, so they stay as rows.

    Bounded by LIMIT in the query itself, so these are small enough to keep whole.
    """
    if isinstance(result, dict):
        return {"result": result}
    return {"rows": len(result), "result": result}


# --- the agent run ------------------------------------------------------------------------------


def agent_config(mode: str, thread_id: str, settings: Any) -> dict:
    """Name and label the root run.

    Without this every trace in the project is called "LangGraph", so a run from the chat UI, a
    CLI question and an evaluation question are indistinguishable. The tag is what makes
    "all graph-mode runs" a filter in LangSmith rather than a manual search.
    """
    return {
        "configurable": {"thread_id": thread_id},
        "run_name": f"expert-finder-{mode}",
        "tags": [f"mode:{mode}", f"model:{settings.chat_model}"],
        "metadata": {"mode": mode, "thread_id": thread_id, "top_k": settings.top_k,
                     "multi_query_rewrites": settings.multi_query_rewrites,
                     "chat_model": settings.chat_model},
    }
