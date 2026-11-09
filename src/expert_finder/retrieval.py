"""Evidence retrieval in two modes.

VectorRetriever: semantic search over Pinecone namespaces. Two refinements sit here rather than
                 in either mode, so both modes get them and a vector-vs-graph comparison still
                 measures only the graph: the question is expanded into several phrasings whose
                 hits are fused (LLMQueryExpander), and business profiles are matched as small
                 section chunks but returned whole (parent-document retrieval).
GraphRetriever:  the same vector hits, then expanded through Neo4j to pull in the connected
                 businesses, people, quotes and whole discussion threads.

Both return an EvidenceBundle with the same shape, so the answer prompt is identical and the
only variable in a vector-vs-graph comparison is what retrieval put in the bundle. Every item
carries a short ref (R1, P2, B1, T1, G3) that maps back to a real record id, which lets the
evaluation verify citations and the UI link to sources.
"""

from __future__ import annotations

import time
from typing import Any, Protocol

from langsmith import traceable
from neo4j import Driver
from pydantic import BaseModel, Field

from expert_finder import graph_queries as gq
from expert_finder import tracing
from expert_finder.documents import NAMESPACES
from expert_finder.formatting import usd
from expert_finder.vector_store import VectorIndex

# Which namespaces a mode searches when the caller does not say.
DEFAULT_NAMESPACES = ("businesses", "reviews", "social")
REF_PREFIX = {"review": "R", "post": "P", "business": "B", "thread": "T", "graph": "G"}


class Evidence(BaseModel):
    ref: str
    kind: str                                   # review | post | business | thread | graph
    text: str                                   # what the model reads
    source: dict[str, Any] = Field(default_factory=dict)  # structured metadata for the UI and eval
    score: float | None = None                  # similarity score, for vector hits
    via: str = "vector"                         # how this item was found: vector | graph

    @property
    def record_id(self) -> str | None:
        return self.source.get("id")


class EvidenceBundle(BaseModel):
    query: str
    mode: str                                   # vector | graph
    items: list[Evidence] = Field(default_factory=list)
    stats: dict[str, Any] = Field(default_factory=dict)
    notes: list[str] = Field(default_factory=list)

    def refs(self) -> set[str]:
        return {i.ref for i in self.items}

    def by_kind(self, kind: str) -> list[Evidence]:
        return [i for i in self.items if i.kind == kind]

    def to_prompt(self) -> str:
        """Render the bundle for the answer prompt, grouped by evidence type."""
        if not self.items:
            return "No evidence found."
        headings = {"business": "Businesses", "review": "Customer reviews", "post": "Social posts",
                    "thread": "Rest of the discussion thread", "graph": "Facts from the knowledge graph"}
        out = []
        for note in self.notes:
            out.append(f"Note: {note}")
        for kind, heading in headings.items():
            group = self.by_kind(kind)
            if group:
                out.append(f"\n## {heading}")
                out += [f"[{i.ref}] {i.text}" for i in group]
        return "\n".join(out)


class _RefMaker:
    """Hands out sequential refs per kind: R1, R2, P1, ..."""

    def __init__(self) -> None:
        self._counts: dict[str, int] = {}

    def next(self, kind: str) -> str:
        self._counts[kind] = self._counts.get(kind, 0) + 1
        return f"{REF_PREFIX[kind]}{self._counts[kind]}"


def _license_phrase(status: str, required: bool) -> str:
    """Roofing and landscaping need no Texas license, so absence is only notable where one is required."""
    if status == "expired":
        return "EXPIRED license"
    if status == "active":
        return "active license"
    return "no license on file (none required for this trade)" if not required else "NO LICENSE on file"


# --- Query expansion ----------------------------------------------------------------------------


class QueryExpander(Protocol):
    def expand(self, query: str) -> list[str]:
        """The original query first, then alternative phrasings to search as well."""
        ...


class _Rewrites(BaseModel):
    """Structured output, so the rewrites never have to be parsed out of prose."""

    queries: list[str] = Field(description="Alternative search phrasings, one per line, no numbering.")


REWRITE_PROMPT = """A homeowner asked: "{query}"

Write {n} different search queries that would find relevant customer reviews, neighbourhood posts and business profiles for this need. Vary the wording: use the trade's own terms for the symptom, name the likely repair or service, and keep one phrasing close to how a customer would say it. Each query stands alone and mentions no business by name."""


class LLMQueryExpander:
    """Rewrite a question into several phrasings before searching.

    Vague questions ("my AC is blowing warm air") and the text that answers them ("recharged the
    refrigerant", "condenser fan motor replaced") often share few words, so one embedding of the
    question misses them. Rewrites are cached per query because several tools retrieve against the
    same question within one turn, and a rewrite is worth paying for once.
    """

    def __init__(self, model: Any, n: int = 2):
        self.model = model
        self.n = n
        self._cache: dict[str, list[str]] = {}

    @traceable(name="query_expansion", process_inputs=tracing.index_inputs,
               process_outputs=tracing.expansion_outputs)
    def expand(self, query: str) -> list[str]:
        if self.n < 1:
            return [query]
        if query not in self._cache:
            try:
                # Bound here rather than in __init__ so that a model which cannot produce
                # structured output costs nothing until a query is actually expanded.
                result = self.model.with_structured_output(_Rewrites).invoke(
                    REWRITE_PROMPT.format(query=query, n=self.n))
                extra = [q.strip() for q in result.queries if q.strip() and q.strip() != query]
            except Exception:
                # Retrieval still works on the original query, so a failed rewrite must not take
                # the whole answer down with it; bundle.stats["queries"] stays 1 when this happens.
                extra = []
            self._cache[query] = [query, *extra[:self.n]]
        return self._cache[query]


# --- Vector retrieval ---------------------------------------------------------------------------

KIND_BY_NAMESPACE = {"reviews": "review", "social": "post", "businesses": "business"}

# Metadata worth carrying into an evidence item: what the UI shows and the evaluation filters on.
_KEEP_METADATA = ("platform", "rating", "business_id", "business_name", "category", "created_date",
                  "community", "post_type", "city", "neighborhood", "review_type", "cities",
                  "license_status", "emergency_service")

# Business profiles are indexed as small section chunks plus the whole profile (see documents.py).
# Chunks are what we search; the parent profile is what the model reads.
CHUNKED_NAMESPACE = "businesses"


def _chunk_label(doc) -> str:
    """What part of a profile matched, in words, for the evidence card and the prompt."""
    section = doc.metadata.get("section", "profile")
    body = doc.page_content.splitlines()[2:]
    first = body[0] if body else ""
    if section == "service":
        return first.split(":")[0]
    if section == "team":
        return first.split(" is ")[0]
    return section


class VectorRetriever:
    mode = "vector"

    def __init__(self, index: VectorIndex, k: int = 6, expander: QueryExpander | None = None,
                 chunk_overfetch: int = 4):
        self.index = index
        self.k = k
        self.expander = expander
        self.chunk_overfetch = chunk_overfetch

    @traceable(run_type="retriever", name="vector_retrieval", process_inputs=tracing.retriever_inputs,
               process_outputs=tracing.bundle_outputs)
    def retrieve(self, query: str, namespaces: tuple[str, ...] = DEFAULT_NAMESPACES, k: int | None = None,
                 filters: dict[str, dict] | None = None, refs: _RefMaker | None = None) -> EvidenceBundle:
        """Search each namespace and turn the hits into evidence.

        `filters` is keyed by namespace, because a rating filter is meaningless for business
        profiles. Filtering inside Pinecone (rather than after) means k relevant hits, not k hits
        that we then discard.
        """
        refs = refs or _RefMaker()
        started = time.perf_counter()
        bundle = EvidenceBundle(query=query, mode=self.mode)
        queries = self.expander.expand(query) if self.expander else [query]
        per_namespace: dict[str, int] = {}
        for ns in namespaces:
            if ns not in NAMESPACES:
                raise ValueError(f"unknown namespace {ns!r}")
            top_k = k or self.k
            if ns == CHUNKED_NAMESPACE:
                items = self._business_evidence(queries, top_k, (filters or {}).get(ns), refs)
            else:
                items = [self._to_evidence(ns, doc, score, refs)
                         for doc, score in self._search(ns, queries, top_k, (filters or {}).get(ns))]
            per_namespace[ns] = len(items)
            bundle.items += items
        bundle.stats = {"namespaces": per_namespace, "items": len(bundle.items),
                        "queries": len(queries),
                        "retrieval_ms": round((time.perf_counter() - started) * 1000)}
        if len(queries) > 1:
            bundle.stats["rewrites"] = queries[1:]
        return bundle

    def _search(self, namespace: str, queries: list[str], k: int,
                filter: dict | None) -> list[tuple[Any, float]]:
        """Run every query variant against one namespace and fuse the hits.

        Scores from a single embedding model and namespace are directly comparable, so a document
        found by two phrasings keeps its best score rather than being counted twice. The fused list
        is cut back to k, so expansion changes which k documents are returned, not how many.
        """
        best: dict[str, tuple[Any, float]] = {}
        for q in queries:
            for doc, score in self.index.search(q, namespace, k=k, filter=filter):
                current = best.get(doc.id)
                if current is None or score > current[1]:
                    best[doc.id] = (doc, score)
        return sorted(best.values(), key=lambda pair: -pair[1])[:k]

    def _business_evidence(self, queries: list[str], k: int, filter: dict | None,
                           refs: _RefMaker) -> list[Evidence]:
        """Match profile chunks, then return one whole profile per business.

        A hit on a single priced service is the precise signal; the full profile is what the model
        needs to answer with (licences, insurance, the rest of the prices, who works there). So we
        overfetch chunks, collapse them to the best k distinct businesses, and fetch those parents.
        """
        chunk_filter = {**(filter or {}), "doc_type": "business_chunk"}
        hits = self._search(CHUNKED_NAMESPACE, queries, k * self.chunk_overfetch, chunk_filter)
        matched: dict[str, dict[str, Any]] = {}
        for doc, score in hits:
            entry = matched.setdefault(doc.metadata["business_id"],
                                       {"score": score, "sections": [], "metadata": doc.metadata})
            entry["score"] = max(entry["score"], score)
            label = _chunk_label(doc)
            if label not in entry["sections"]:
                entry["sections"].append(label)
        chosen = sorted(matched.items(), key=lambda kv: -kv[1]["score"])[:k]
        parents = self.index.fetch([bid for bid, _ in chosen], CHUNKED_NAMESPACE)

        items = []
        for business_id, entry in chosen:
            parent = parents.get(business_id)
            # Without the parent (an index built before profiles were chunked) the chunk metadata
            # still identifies the business, so fall back to it rather than dropping the hit.
            md = parent.metadata if parent else entry["metadata"]
            text = parent.page_content.strip() if parent else ""
            matched_on = ", ".join(entry["sections"][:4])
            items.append(Evidence(
                ref=refs.next("business"), kind="business", via="vector",
                text=f"Matched on: {matched_on}.\n{text}" if text else f"Matched on: {matched_on}.",
                source={"id": business_id, "namespace": CHUNKED_NAMESPACE,
                        "matched_sections": entry["sections"],
                        **{key: md[key] for key in _KEEP_METADATA if key in md}},
                score=round(entry["score"], 3)))
        return items

    @staticmethod
    def _to_evidence(namespace: str, doc, score: float, refs: _RefMaker) -> Evidence:
        kind = KIND_BY_NAMESPACE[namespace]
        md = doc.metadata
        source = {"id": doc.id, "namespace": namespace, **{k: md[k] for k in _KEEP_METADATA if k in md}}
        return Evidence(ref=refs.next(kind), kind=kind, text=doc.page_content.strip(), source=source,
                        score=round(score, 3), via="vector")


# --- Graph retrieval ----------------------------------------------------------------------------


class GraphRetriever:
    """Vector hits plus graph expansion around them."""

    mode = "graph"

    def __init__(self, index: VectorIndex, driver: Driver, database: str, k: int = 6,
                 thread_limit: int = 10, expander: QueryExpander | None = None,
                 chunk_overfetch: int = 4):
        # Graph mode wraps the same vector stage, so query expansion and profile chunking apply
        # identically in both modes and the comparison still isolates the graph.
        self.vector = VectorRetriever(index, k=k, expander=expander, chunk_overfetch=chunk_overfetch)
        self.driver = driver
        self.database = database
        self.thread_limit = thread_limit

    @traceable(run_type="retriever", name="graph_retrieval", process_inputs=tracing.retriever_inputs,
               process_outputs=tracing.bundle_outputs)
    def retrieve(self, query: str, namespaces: tuple[str, ...] = DEFAULT_NAMESPACES, k: int | None = None,
                 filters: dict[str, dict] | None = None, service_ids: list[str] | None = None) -> EvidenceBundle:
        refs = _RefMaker()
        bundle = self.vector.retrieve(query, namespaces, k, filters, refs=refs)
        bundle.mode = self.mode
        started = time.perf_counter()

        review_ids = [i.record_id for i in bundle.by_kind("review")]
        post_ids = [i.record_id for i in bundle.by_kind("post")]
        seen_businesses = {i.record_id for i in bundle.by_kind("business")}

        # 1. Annotate each retrieved review with its business, the tech named, and quote vs invoice.
        mentioned: dict[str, int] = {}
        for row in gq.expand_reviews(self.driver, self.database, review_ids):
            mentioned[row["business_id"]] = mentioned.get(row["business_id"], 0) + 1
            bundle.items.append(Evidence(
                ref=refs.next("graph"), kind="graph", via="graph",
                text=self._review_fact(row),
                source={"id": row["review_id"], "business_id": row["business_id"], "relation": "review_context"}))

        # 2. Resolve the businesses that posts refer to; these are candidates vector search did not return.
        post_businesses: dict[str, dict] = {}
        for row in gq.expand_posts(self.driver, self.database, post_ids):
            for b in row["businesses"]:
                mentioned[b["business_id"]] = mentioned.get(b["business_id"], 0) + 1
                post_businesses.setdefault(b["business_id"], b)
            if row["businesses"] or row["experts"]:
                bundle.items.append(Evidence(
                    ref=refs.next("graph"), kind="graph", via="graph",
                    text=self._post_fact(row),
                    source={"id": row["post_id"], "relation": "post_mentions"}))

        # 3. A hit on one reply should bring the question and the other replies with it.
        for row in gq.expand_threads(self.driver, self.database, post_ids, limit=self.thread_limit):
            bundle.items.append(Evidence(
                ref=refs.next("thread"), kind="thread", via="graph",
                text=f"{row['post_type'].replace('_', ' ')} by {row['author']} on "
                     f"{row['created_at'][:10]}: \"{row['text']}\""
                     + (f" [mentions {', '.join(row['mentioned'])}]" if row["mentioned"] else ""),
                source={"id": row["post_id"], "thread_id": row["thread_id"], "relation": "same_thread"}))

        # 4. Full context for businesses the graph surfaced but the vector search missed.
        new_business_ids = [b for b in post_businesses if b not in seen_businesses]
        for row in gq.business_context(self.driver, self.database, new_business_ids, service_ids):
            bundle.items.append(Evidence(
                ref=refs.next("business"), kind="business", via="graph",
                text=self._business_fact(row, post_businesses.get(row["business_id"], {})),
                source={"id": row["business_id"], "business_name": row["business_name"],
                        "category": row["category"], "relation": "mentioned_in_post"}))

        # 5. Corroboration: the same business named by several independent retrieved sources.
        corroborated = {b: n for b, n in mentioned.items() if n > 1}
        if corroborated:
            names = {r["business_id"]: r["business_name"]
                     for r in gq.business_context(self.driver, self.database, list(corroborated), None)}
            bundle.notes.append("Independent sources naming the same business: "
                                + "; ".join(f"{names.get(b, b)} in {n} retrieved sources"
                                            for b, n in sorted(corroborated.items(), key=lambda x: -x[1])))

        bundle.stats["graph_ms"] = round((time.perf_counter() - started) * 1000)
        bundle.stats["items"] = len(bundle.items)
        bundle.stats["graph_items"] = sum(1 for i in bundle.items if i.via == "graph")
        bundle.stats["businesses_added_by_graph"] = len(new_business_ids)
        return bundle

    @staticmethod
    def _review_fact(row: dict) -> str:
        parts = [f"Review {row['review_id']} is about {row['business_name']} ({row['category']}, "
                 f"phone {row['phone']}, {_license_phrase(row['license_status'], row['license_required'])})."]
        if row["service"]:
            parts.append(f"Service: {row['service']}.")
        if row["experts"]:
            parts.append(f"Names {', '.join(row['experts'])}.")
        parts.append(f"That business averages {row['business_avg_rating']} over "
                     f"{row['business_rated_reviews']} ratings with {row['business_complaints']} complaints.")
        if row["quoted_usd"]:
            parts.append(f"The job behind this review was quoted {usd(row['quoted_usd'])} and invoiced "
                         f"{usd(row['final_invoice_usd'])}.")
        return " ".join(parts)

    @staticmethod
    def _post_fact(row: dict) -> str:
        parts = [f"Post {row['post_id']}"
                 + (f" in {row['community']}" if row["community"] else "") + " refers to "
                 + "; ".join(f"{b['business_name']} ({b['category']}, phone {b['phone']}) "
                             f"— written as \"{b['matched_text']}\"" for b in row["businesses"])
                 if row["businesses"] else f"Post {row['post_id']} names no identifiable business."]
        if row["experts"]:
            parts.append(f"It names {', '.join(row['experts'])}.")
        return " ".join(parts)

    @staticmethod
    def _business_fact(row: dict, mention: dict) -> str:
        parts = [f"{row['business_name']} ({row['category']}, based in {row['based_in']}, "
                 f"phone {row['phone']}, {_license_phrase(row['license_status'], row['license_required'])},"
                 f"{' emergency service,' if row['emergency_service'] else ''} "
                 f"hourly rate {usd(row['hourly_rate_usd'])})."]
        if row["avg_rating"] is not None:
            parts.append(f"Averages {row['avg_rating']} over {row['rated_reviews']} ratings "
                         f"with {row['complaints']} complaints.")
        for p in row["prices"]:
            parts.append(f"{p['service']}: {usd(p['price_low_usd'])}-{usd(p['price_high_usd'])} "
                         f"per {p['unit']} (typical {usd(p['typical_price_usd'])}).")
        if mention.get("matched_text"):
            parts.append(f"Surfaced from a social post that called it \"{mention['matched_text']}\".")
        return " ".join(parts)
