"""Evidence retrieval in two modes.

VectorRetriever: semantic search over Pinecone namespaces.
GraphRetriever:  the same vector hits, then expanded through Neo4j to pull in the connected
                 businesses, people, quotes and whole discussion threads.

Both return an EvidenceBundle with the same shape, so the answer prompt is identical and the
only variable in a vector-vs-graph comparison is what retrieval put in the bundle. Every item
carries a short ref (R1, P2, B1, T1, G3) that maps back to a real record id, which lets the
evaluation verify citations and the UI link to sources.
"""

from __future__ import annotations

import time
from typing import Any

from neo4j import Driver
from pydantic import BaseModel, Field

from expert_finder import graph_queries as gq
from expert_finder.documents import NAMESPACES
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


def _usd(x: float | None) -> str:
    return f"${x:,.0f}" if x is not None else "n/a"


def _license_phrase(status: str, required: bool) -> str:
    """Roofing and landscaping need no Texas license, so absence is only notable where one is required."""
    if status == "expired":
        return "EXPIRED license"
    if status == "active":
        return "active license"
    return "no license on file (none required for this trade)" if not required else "NO LICENSE on file"


# --- Vector retrieval ---------------------------------------------------------------------------

KIND_BY_NAMESPACE = {"reviews": "review", "social": "post", "businesses": "business"}


class VectorRetriever:
    mode = "vector"

    def __init__(self, index: VectorIndex, k: int = 6):
        self.index = index
        self.k = k

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
        per_namespace = {}
        for ns in namespaces:
            if ns not in NAMESPACES:
                raise ValueError(f"unknown namespace {ns!r}")
            hits = self.index.search(query, ns, k=k or self.k, filter=(filters or {}).get(ns))
            per_namespace[ns] = len(hits)
            for doc, score in hits:
                bundle.items.append(self._to_evidence(ns, doc, score, refs))
        bundle.stats = {"namespaces": per_namespace, "items": len(bundle.items),
                        "retrieval_ms": round((time.perf_counter() - started) * 1000)}
        return bundle

    @staticmethod
    def _to_evidence(namespace: str, doc, score: float, refs: _RefMaker) -> Evidence:
        kind = KIND_BY_NAMESPACE[namespace]
        md = doc.metadata
        keep = ("platform", "rating", "business_id", "business_name", "category", "created_date",
                "community", "post_type", "city", "neighborhood", "review_type", "cities",
                "license_status", "emergency_service")
        source = {"id": doc.id, "namespace": namespace, **{k: md[k] for k in keep if k in md}}
        return Evidence(ref=refs.next(kind), kind=kind, text=doc.page_content.strip(), source=source,
                        score=round(score, 3), via="vector")


# --- Graph retrieval ----------------------------------------------------------------------------


class GraphRetriever:
    """Vector hits plus graph expansion around them."""

    mode = "graph"

    def __init__(self, index: VectorIndex, driver: Driver, database: str, k: int = 6,
                 thread_limit: int = 10):
        self.vector = VectorRetriever(index, k=k)
        self.driver = driver
        self.database = database
        self.thread_limit = thread_limit

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
            parts.append(f"The job behind this review was quoted {_usd(row['quoted_usd'])} and invoiced "
                         f"{_usd(row['final_invoice_usd'])}.")
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
                 f"hourly rate {_usd(row['hourly_rate_usd'])})."]
        if row["avg_rating"] is not None:
            parts.append(f"Averages {row['avg_rating']} over {row['rated_reviews']} ratings "
                         f"with {row['complaints']} complaints.")
        for p in row["prices"]:
            parts.append(f"{p['service']}: {_usd(p['price_low_usd'])}-{_usd(p['price_high_usd'])} "
                         f"per {p['unit']} (typical {_usd(p['typical_price_usd'])}).")
        if mention.get("matched_text"):
            parts.append(f"Surfaced from a social post that called it \"{mention['matched_text']}\".")
        return " ".join(parts)
