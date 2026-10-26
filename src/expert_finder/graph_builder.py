"""Build the Neo4j knowledge graph from the raw dataset.

Node ids match Pinecone and SQLite ids (rev_000004, post_00012, biz_0001, ...) so graph RAG can
start from Pinecone hits and expand. MENTIONS links come from our own entity resolution, never
from ground truth.
"""

from __future__ import annotations


from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime

from neo4j import Driver

from expert_finder.dataset import Dataset
from expert_finder.entity_resolution import EntityResolver

LABELS = ["Category", "Service", "City", "Neighborhood", "Business", "Expert", "Customer", "Review", "Post", "Estimate"]
BATCH = 1000


@dataclass
class GraphData:
    nodes: dict[str, list[dict]] = field(default_factory=lambda: defaultdict(list))  # label -> [{id, ...props}]
    rels: dict[tuple[str, str, str], list[dict]] = field(default_factory=lambda: defaultdict(list))
    # (type, from_label, to_label) -> [{s, t, props}]

    def node(self, label: str, id: str, **props) -> None:
        self.nodes[label].append({"id": id, "props": {k: v for k, v in props.items() if v is not None}})

    def rel(self, type_: str, from_label: str, s: str, to_label: str, t: str, **props) -> None:
        self.rels[(type_, from_label, to_label)].append(
            {"s": s, "t": t, "props": {k: v for k, v in props.items() if v is not None}})


def _ts(iso: str | None) -> int | None:
    return int(datetime.fromisoformat(iso).timestamp()) if iso else None


def _nbhd_id(city: str, name: str) -> str:
    return f"{city}/{name}"


def build_graph_data(ds: Dataset, resolver: EntityResolver) -> GraphData:
    g = GraphData()

    # Taxonomy and geography.
    for c in ds.categories:
        if c["in_dataset"]:
            g.node("Category", c["category_id"], name=c["name"])
    for s in ds.services:
        g.node("Service", s["service_id"], name=s["name"], common_name=s["common_name"], category=s["category"],
               pricing_model=s["pricing_model"], unit=s["unit"])
        g.rel("IN_CATEGORY", "Service", s["service_id"], "Category", s["category"])
    places = {(c["city"], c["neighborhood"], c["zip"]) for c in ds.customers}
    for city in sorted({p[0] for p in places} | {b["address"]["city"] for b in ds.businesses}):
        g.node("City", city, name=city)
    for city, name, zip_ in sorted(places):
        g.node("Neighborhood", _nbhd_id(city, name), name=name, city=city, zip=zip_)
        g.rel("IN_CITY", "Neighborhood", _nbhd_id(city, name), "City", city)

    # Businesses, their offerings and their people.
    for b in ds.businesses:
        statuses = {lic["status"] for lic in b["licenses"]}
        g.node("Business", b["business_id"], name=b["name"], category=b["primary_category"],
               founded_year=b["founded_year"], business_size=b["business_size"],
               emergency_service=b["emergency_service"]["available"],
               license_status="expired" if "expired" in statuses else "active" if statuses else "none",
               hourly_rate_usd=b["pricing_policy"]["hourly_rate_usd"], based_in=b["address"]["city"],
               phone=b["contact"]["phone"])
        g.rel("IN_CATEGORY", "Business", b["business_id"], "Category", b["primary_category"], primary=True)
        for c in b["secondary_categories"]:
            g.rel("IN_CATEGORY", "Business", b["business_id"], "Category", c, primary=False)
        g.rel("BASED_IN", "Business", b["business_id"], "City", b["address"]["city"])
        for city in b["service_area"]["cities"]:
            g.rel("SERVES", "Business", b["business_id"], "City", city)
    for o in ds.offerings:
        g.rel("OFFERS", "Business", o["business_id"], "Service", o["service_id"], price_low_usd=o["price_low_usd"],
              typical_price_usd=o["typical_price_usd"], price_high_usd=o["price_high_usd"], unit=o["unit"],
              pricing_model=o["pricing_model"], lead_time_days=o["lead_time_days"])
    for e in ds.experts:
        g.node("Expert", e["expert_id"], name=f"{e['first_name']} {e['last_name']}", first_name=e["first_name"],
               last_name=e["last_name"], goes_by=e["goes_by"], role=e["role"],
               years_experience=e["years_experience"], is_owner=e["is_owner"])
        g.rel("WORKS_AT", "Expert", e["expert_id"], "Business", e["business_id"], role=e["role"], is_owner=e["is_owner"])
        for s in e["specialties"]:
            g.rel("SPECIALIZES_IN", "Expert", e["expert_id"], "Service", s)

    # Customers.
    for c in ds.customers:
        g.node("Customer", c["customer_id"], name=f"{c['first_name']} {c['last_name'][0]}.", city=c["city"],
               neighborhood=c["neighborhood"], home_type=c["home"]["type"], home_year_built=c["home"]["year_built"])
        g.rel("LIVES_IN", "Customer", c["customer_id"], "Neighborhood", _nbhd_id(c["city"], c["neighborhood"]))

    # Estimates, and competing estimates for the same job.
    by_project: dict[str, list[str]] = defaultdict(list)
    for e in ds.estimates:
        job = e["job"] or {}
        g.node("Estimate", e["estimate_id"], status=e["status"], total_usd=e["total_usd"],
               subtotal_usd=e["subtotal_usd"], final_invoice_usd=job.get("final_invoice_usd"),
               quantity=e["quantity"], unit=e["unit"], issued_at=e["issued_at"], issued_ts=_ts(e["issued_at"]),
               decline_reason=e["decline_reason"], city=e["job_address"]["city"], project_id=e["customer_project_id"],
               job_description=e["job_description"])
        g.rel("QUOTED_BY", "Estimate", e["estimate_id"], "Business", e["business_id"])
        g.rel("FOR_SERVICE", "Estimate", e["estimate_id"], "Service", e["service_id"])
        g.rel("REQUESTED", "Customer", e["customer_id"], "Estimate", e["estimate_id"])
        if e["expert_id"]:
            g.rel("PREPARED_BY", "Estimate", e["estimate_id"], "Expert", e["expert_id"])
        if e["customer_project_id"]:
            by_project[e["customer_project_id"]].append(e["estimate_id"])
    for ids in by_project.values():
        for i, a in enumerate(ids):
            for b_ in ids[i + 1:]:
                g.rel("COMPETED_WITH", "Estimate", a, "Estimate", b_)

    # Reviews, with expert mentions resolved from the text.
    for r in ds.reviews:
        resp = r["owner_response"]
        g.node("Review", r["review_id"], platform=r["platform"], review_type=r["review_type"], rating=r["rating"],
               title=r["title"], text=r["text"], created_at=r["created_at"], created_ts=_ts(r["created_at"]),
               complaint_status=r["complaint_status"], verified_hire=r["verified_hire"],
               reviewer_total_reviews=r["reviewer_total_reviews"], owner_response=resp["text"] if resp else None,
               cost_reported_usd=r["cost_reported_usd"])
        g.rel("ABOUT", "Review", r["review_id"], "Business", r["business_id"])
        if r["customer_id"]:
            g.rel("WROTE", "Customer", r["customer_id"], "Review", r["review_id"])
        if r["service_id"]:
            g.rel("FOR_SERVICE", "Review", r["review_id"], "Service", r["service_id"])
        if r["estimate_id"]:
            g.rel("ABOUT_JOB", "Review", r["review_id"], "Estimate", r["estimate_id"])
        for m in resolver.resolve_review_experts(r):
            g.rel("MENTIONS", "Review", r["review_id"], "Expert", m.target_id, surface_form=m.surface_form,
                  method=m.method)

    # Social posts, threads and resolved mentions.
    for p in ds.posts:
        text = " ".join([p["text"], *(f"[video] {m['transcript']}" for m in p["media"] if m.get("transcript"))])
        g.node("Post", p["post_id"], platform=p["platform"], post_type=p["post_type"], community=p["community"],
               text=text, created_at=p["created_at"], created_ts=_ts(p["created_at"]),
               author_handle=p["author"]["handle"], author_type=p["author"]["account_type"],
               likes=p["engagement"].get("likes"), views=p["engagement"].get("views"), thread_id=p["thread_id"])
        if p["customer_id"]:
            g.rel("POSTED", "Customer", p["customer_id"], "Post", p["post_id"])
        if p["parent_post_id"]:
            g.rel("REPLY_TO", "Post", p["post_id"], "Post", p["parent_post_id"])
        if p["platform"] == "nextdoor" and p["community"]:
            name, _, city = p["community"].rpartition(", ")
            g.rel("IN_NEIGHBORHOOD", "Post", p["post_id"], "Neighborhood", _nbhd_id(city, name))
        for m in resolver.resolve_post(p):
            target = "Business" if m.kind == "business" else "Expert"
            g.rel("MENTIONS", "Post", p["post_id"], target, m.target_id, surface_form=m.surface_form, method=m.method)
    return g


# --- Writing to Neo4j -----------------------------------------------------------------------------


def _batches(rows: list[dict]):
    for i in range(0, len(rows), BATCH):
        yield rows[i:i + BATCH]


def clear(driver: Driver, database: str) -> None:
    """Delete only nodes with this project's labels, in batches (keeps transactions small)."""
    for label in LABELS:
        while True:
            records, _, _ = driver.execute_query(
                f"MATCH (n:{label}) WITH n LIMIT 5000 DETACH DELETE n RETURN count(*) AS deleted", database_=database)
            if records[0]["deleted"] == 0:
                break


def write(driver: Driver, database: str, g: GraphData) -> dict[str, int]:
    for label in LABELS:
        driver.execute_query(f"CREATE CONSTRAINT {label.lower()}_id IF NOT EXISTS "
                             f"FOR (n:{label}) REQUIRE n.id IS UNIQUE", database_=database)
    # Labels and relationship types can't be query parameters in Cypher; they come from our own
    # constants above, never from user input.
    for label, rows in g.nodes.items():
        for batch in _batches(rows):
            driver.execute_query(f"UNWIND $rows AS row MERGE (n:{label} {{id: row.id}}) SET n += row.props",
                                 rows=batch, database_=database)
    for (type_, la, lb), rows in g.rels.items():
        for batch in _batches(rows):
            driver.execute_query(
                f"UNWIND $rows AS row MATCH (a:{la} {{id: row.s}}) MATCH (b:{lb} {{id: row.t}}) "
                f"MERGE (a)-[r:{type_}]->(b) SET r += row.props", rows=batch, database_=database)
    return stats(driver, database)


def stats(driver: Driver, database: str) -> dict[str, int]:
    out = {}
    for label in LABELS:
        records, _, _ = driver.execute_query(f"MATCH (n:{label}) RETURN count(n) AS c", database_=database)
        out[label] = records[0]["c"]
    records, _, _ = driver.execute_query(
        "MATCH ()-[r]->() RETURN type(r) AS t, count(r) AS c ORDER BY c DESC", database_=database)
    out["relationships"] = {r["t"]: r["c"] for r in records}
    return out
