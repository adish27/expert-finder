"""SQLite store for structured facts and numbers (prices, quotes, ratings, licenses).

Built from the raw dataset into one file; rebuilt from scratch whenever needed. Both the
vector-RAG and graph-RAG modes query it through the functions in sql_queries.py.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from expert_finder.dataset import Dataset

SCHEMA = """
CREATE TABLE services (
    service_id TEXT PRIMARY KEY, category TEXT NOT NULL, name TEXT NOT NULL, common_name TEXT NOT NULL,
    aliases TEXT, pricing_model TEXT NOT NULL, unit TEXT NOT NULL,
    market_low_usd REAL, market_high_usd REAL, permit_typically_required INTEGER
);
CREATE TABLE businesses (
    business_id TEXT PRIMARY KEY, name TEXT NOT NULL, legal_name TEXT, primary_category TEXT NOT NULL,
    description TEXT, founded_year INTEGER, business_size TEXT, employee_count INTEGER, owner_expert_id TEXT,
    phone TEXT, email TEXT, website TEXT, street TEXT, city TEXT, zip TEXT, radius_miles INTEGER,
    hours_mon_fri TEXT, hours_sat TEXT, hours_sun TEXT,
    emergency_available INTEGER, emergency_hours TEXT, after_hours_surcharge_pct INTEGER,
    general_liability INTEGER, liability_coverage_usd INTEGER, workers_comp INTEGER, bonded INTEGER,
    financing_available INTEGER, warranty_months INTEGER,
    hourly_rate_usd REAL, service_call_fee_usd REAL, service_call_fee_waived INTEGER, free_estimates INTEGER,
    minimum_charge_usd REAL, typical_reply_hours INTEGER, typical_days_to_schedule INTEGER
);
CREATE TABLE business_categories (
    business_id TEXT NOT NULL REFERENCES businesses, category TEXT NOT NULL, is_primary INTEGER NOT NULL,
    PRIMARY KEY (business_id, category)
);
CREATE TABLE business_service_areas (
    business_id TEXT NOT NULL REFERENCES businesses, city TEXT NOT NULL, PRIMARY KEY (business_id, city)
);
CREATE TABLE business_licenses (
    business_id TEXT NOT NULL REFERENCES businesses, type TEXT, issuer TEXT, number TEXT, status TEXT, expires TEXT
);
CREATE TABLE business_attributes (
    business_id TEXT NOT NULL REFERENCES businesses, kind TEXT NOT NULL, value TEXT NOT NULL
);
CREATE TABLE experts (
    expert_id TEXT PRIMARY KEY, business_id TEXT NOT NULL REFERENCES businesses, first_name TEXT, last_name TEXT,
    goes_by TEXT, role TEXT, is_owner INTEGER, field_role INTEGER, years_experience INTEGER,
    licenses TEXT, certifications TEXT, languages TEXT, bio TEXT
);
CREATE TABLE expert_specialties (
    expert_id TEXT NOT NULL REFERENCES experts, service_id TEXT NOT NULL REFERENCES services,
    PRIMARY KEY (expert_id, service_id)
);
CREATE TABLE offerings (
    offering_id TEXT PRIMARY KEY, business_id TEXT NOT NULL REFERENCES businesses,
    service_id TEXT NOT NULL REFERENCES services, is_secondary_category INTEGER, pricing_model TEXT, unit TEXT,
    price_low_usd REAL, price_high_usd REAL, typical_price_usd REAL, price_is_starting_at INTEGER,
    includes TEXT, excludes TEXT, permit_required INTEGER, lead_time_days INTEGER, warranty_months INTEGER,
    available_for_emergency INTEGER, notes TEXT
);
CREATE TABLE customers (
    customer_id TEXT PRIMARY KEY, first_name TEXT, last_name TEXT, city TEXT, neighborhood TEXT, zip TEXT,
    home_type TEXT, year_built INTEGER, sq_ft INTEGER, stories INTEGER, has_pool INTEGER
);
CREATE TABLE estimates (
    estimate_id TEXT PRIMARY KEY, customer_project_id TEXT, business_id TEXT NOT NULL REFERENCES businesses,
    expert_id TEXT REFERENCES experts, customer_id TEXT REFERENCES customers,
    service_id TEXT NOT NULL REFERENCES services, category TEXT, request_channel TEXT,
    requested_at TEXT, site_visit_at TEXT, issued_at TEXT, valid_until TEXT, city TEXT, neighborhood TEXT, zip TEXT,
    job_description TEXT, quantity REAL, unit TEXT, subtotal_usd REAL, sales_tax_usd REAL, total_usd REAL,
    financing_offered INTEGER, status TEXT, decline_reason TEXT,
    job_id TEXT, completed_at TEXT, final_invoice_usd REAL, change_order_total_usd REAL, duration_days INTEGER
);
CREATE TABLE estimate_line_items (
    estimate_id TEXT NOT NULL REFERENCES estimates, line_no INTEGER NOT NULL, description TEXT, category TEXT,
    quantity REAL, unit TEXT, unit_price_usd REAL, total_usd REAL, PRIMARY KEY (estimate_id, line_no)
);
CREATE TABLE reviews (
    review_id TEXT PRIMARY KEY, platform TEXT NOT NULL, business_id TEXT NOT NULL REFERENCES businesses,
    customer_id TEXT REFERENCES customers, reviewer_display_name TEXT, reviewer_total_reviews INTEGER,
    review_type TEXT, rating INTEGER, title TEXT, text TEXT, created_at TEXT, service_id TEXT REFERENCES services,
    job_date TEXT, estimate_id TEXT REFERENCES estimates, cost_reported_usd REAL, verified_hire INTEGER,
    photos_count INTEGER, helpful_count INTEGER, complaint_status TEXT,
    owner_response_text TEXT, owner_response_at TEXT,
    sub_quality INTEGER, sub_value INTEGER, sub_punctuality INTEGER, sub_professionalism INTEGER,
    sub_responsiveness INTEGER
);
CREATE TABLE social_posts (
    post_id TEXT PRIMARY KEY, platform TEXT, post_type TEXT, thread_id TEXT, parent_post_id TEXT, community TEXT,
    author_handle TEXT, author_type TEXT, customer_id TEXT REFERENCES customers, created_at TEXT, text TEXT,
    likes INTEGER, comments INTEGER, shares INTEGER, views INTEGER
);

CREATE INDEX ix_areas_city ON business_service_areas (city);
CREATE INDEX ix_cats_category ON business_categories (category);
CREATE INDEX ix_attr ON business_attributes (kind, value);
CREATE INDEX ix_offer_service ON offerings (service_id);
CREATE INDEX ix_offer_business ON offerings (business_id);
CREATE INDEX ix_est_business ON estimates (business_id);
CREATE INDEX ix_est_service ON estimates (service_id, city);
CREATE INDEX ix_est_project ON estimates (customer_project_id);
CREATE INDEX ix_rev_business ON reviews (business_id, created_at);
CREATE INDEX ix_posts_thread ON social_posts (thread_id);
"""


def _b(x) -> int | None:
    """SQLite has no boolean type; store 0/1."""
    return None if x is None else int(bool(x))


def _j(items) -> str | None:
    """Short lists that are only displayed, never filtered, are stored as '; '-joined text."""
    return "; ".join(items) if items else None


def build(ds: Dataset, path: Path) -> dict[str, int]:
    """Create the database file from scratch and return row counts per table."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.unlink(missing_ok=True)
    con = sqlite3.connect(tmp)
    con.execute("PRAGMA foreign_keys = ON")
    con.executescript(SCHEMA)
    with con:  # one transaction
        _load(con, ds)
    counts = {t: con.execute(f"SELECT count(*) FROM {t}").fetchone()[0]
              for (t,) in con.execute("SELECT name FROM sqlite_master WHERE type = 'table' ORDER BY name")}
    con.close()
    tmp.replace(path)  # swap in only after a complete, successful build
    return counts


def _load(con: sqlite3.Connection, ds: Dataset) -> None:
    def insert(table: str, rows: list[dict]) -> None:
        if rows:
            cols = list(rows[0])
            con.executemany(f"INSERT INTO {table} ({', '.join(cols)}) VALUES ({', '.join('?' * len(cols))})",
                            [tuple(r[c] for c in cols) for r in rows])

    insert("services", [{
        "service_id": s["service_id"], "category": s["category"], "name": s["name"],
        "common_name": s["common_name"], "aliases": _j(s["aliases"]), "pricing_model": s["pricing_model"],
        "unit": s["unit"],
        "market_low_usd": s["market_price_low_usd"], "market_high_usd": s["market_price_high_usd"],
        "permit_typically_required": _b(s["permit_typically_required"])} for s in ds.services])

    biz, cats, areas, lics, attrs = [], [], [], [], []
    for b in ds.businesses:
        pp, em, ins = b["pricing_policy"], b["emergency_service"], b["insurance"]
        biz.append({
            "business_id": b["business_id"], "name": b["name"], "legal_name": b["legal_name"],
            "primary_category": b["primary_category"], "description": b["description"],
            "founded_year": b["founded_year"], "business_size": b["business_size"],
            "employee_count": b["employee_count"], "owner_expert_id": b["owner_expert_id"],
            "phone": b["contact"]["phone"], "email": b["contact"]["email"], "website": b["contact"]["website"],
            "street": b["address"]["street"], "city": b["address"]["city"], "zip": b["address"]["zip"],
            "radius_miles": b["service_area"]["radius_miles"],
            "hours_mon_fri": b["hours"]["mon_fri"], "hours_sat": b["hours"]["sat"], "hours_sun": b["hours"]["sun"],
            "emergency_available": _b(em["available"]), "emergency_hours": em["hours"],
            "after_hours_surcharge_pct": em["after_hours_surcharge_pct"],
            "general_liability": _b(ins["general_liability"]), "liability_coverage_usd": ins["liability_coverage_usd"],
            "workers_comp": _b(ins["workers_comp"]), "bonded": _b(ins["bonded"]),
            "financing_available": _b(b["financing_available"]),
            "warranty_months": b["warranty"]["workmanship_months"],
            "hourly_rate_usd": pp["hourly_rate_usd"], "service_call_fee_usd": pp["service_call_fee_usd"],
            "service_call_fee_waived": _b(pp["service_call_fee_waived_with_repair"]),
            "free_estimates": _b(pp["free_estimates"]), "minimum_charge_usd": pp["minimum_charge_usd"],
            "typical_reply_hours": b["response_time"]["typical_reply_hours"],
            "typical_days_to_schedule": b["response_time"]["typical_days_to_schedule"]})
        cats.append({"business_id": b["business_id"], "category": b["primary_category"], "is_primary": 1})
        cats += [{"business_id": b["business_id"], "category": c, "is_primary": 0} for c in b["secondary_categories"]]
        areas += [{"business_id": b["business_id"], "city": c} for c in b["service_area"]["cities"]]
        lics += [{"business_id": b["business_id"], **{k: lic[k] for k in ("type", "issuer", "number", "status", "expires")}}
                 for lic in b["licenses"]]
        for kind, values in (("certification", b["certifications"]), ("brand", b["brands"]),
                             ("language", b["languages"]), ("payment_method", b["payment_methods"]),
                             ("discount", pp["discounts"])):
            attrs += [{"business_id": b["business_id"], "kind": kind, "value": v} for v in values]
        attrs += [{"business_id": b["business_id"], "kind": f"social:{k}", "value": v}
                  for k, v in b["social_accounts"].items()]
    insert("businesses", biz)
    insert("business_categories", cats)
    insert("business_service_areas", areas)
    insert("business_licenses", lics)
    insert("business_attributes", attrs)

    insert("experts", [{
        "expert_id": e["expert_id"], "business_id": e["business_id"], "first_name": e["first_name"],
        "last_name": e["last_name"], "goes_by": e["goes_by"], "role": e["role"], "is_owner": _b(e["is_owner"]),
        "field_role": _b(e["field_role"]), "years_experience": e["years_experience"],
        "licenses": _j([f"{lic['type']} #{lic['number']}" for lic in e["licenses"]]),
        "certifications": _j(e["certifications"]), "languages": _j(e["languages"]), "bio": e["bio"]}
        for e in ds.experts])
    insert("expert_specialties", [{"expert_id": e["expert_id"], "service_id": s}
                                  for e in ds.experts for s in e["specialties"]])

    insert("offerings", [{
        "offering_id": o["offering_id"], "business_id": o["business_id"], "service_id": o["service_id"],
        "is_secondary_category": _b(o["is_secondary_category"]), "pricing_model": o["pricing_model"],
        "unit": o["unit"], "price_low_usd": o["price_low_usd"], "price_high_usd": o["price_high_usd"],
        "typical_price_usd": o["typical_price_usd"], "price_is_starting_at": _b(o["price_is_starting_at"]),
        "includes": _j(o["includes"]), "excludes": _j(o["excludes"]), "permit_required": _b(o["permit_required"]),
        "lead_time_days": o["lead_time_days"], "warranty_months": o["warranty_months"],
        "available_for_emergency": _b(o["available_for_emergency"]), "notes": o["notes"]} for o in ds.offerings])

    insert("customers", [{
        "customer_id": c["customer_id"], "first_name": c["first_name"], "last_name": c["last_name"],
        "city": c["city"], "neighborhood": c["neighborhood"], "zip": c["zip"], "home_type": c["home"]["type"],
        "year_built": c["home"]["year_built"], "sq_ft": c["home"]["sq_ft"], "stories": c["home"]["stories"],
        "has_pool": _b(c["home"]["has_pool"])} for c in ds.customers])

    ests, lines = [], []
    for e in ds.estimates:
        job = e["job"] or {}
        ests.append({
            "estimate_id": e["estimate_id"], "customer_project_id": e["customer_project_id"],
            "business_id": e["business_id"], "expert_id": e["expert_id"], "customer_id": e["customer_id"],
            "service_id": e["service_id"], "category": e["category"], "request_channel": e["request_channel"],
            "requested_at": e["requested_at"], "site_visit_at": e["site_visit_at"], "issued_at": e["issued_at"],
            "valid_until": e["valid_until"], "city": e["job_address"]["city"],
            "neighborhood": e["job_address"]["neighborhood"], "zip": e["job_address"]["zip"],
            "job_description": e["job_description"], "quantity": e["quantity"], "unit": e["unit"],
            "subtotal_usd": e["subtotal_usd"], "sales_tax_usd": e["sales_tax_usd"], "total_usd": e["total_usd"],
            "financing_offered": _b(e["financing_offered"]), "status": e["status"],
            "decline_reason": e["decline_reason"], "job_id": job.get("job_id"),
            "completed_at": job.get("completed_at"), "final_invoice_usd": job.get("final_invoice_usd"),
            "change_order_total_usd": sum(c["amount_usd"] for c in job.get("change_orders", [])) if job else None,
            "duration_days": job.get("duration_days")})
        lines += [{"estimate_id": e["estimate_id"], "line_no": i, "description": li["description"],
                   "category": li["category"], "quantity": li["quantity"], "unit": li["unit"],
                   "unit_price_usd": li["unit_price_usd"], "total_usd": li["total_usd"]}
                  for i, li in enumerate(e["line_items"], 1)]
    insert("estimates", ests)
    insert("estimate_line_items", lines)

    insert("reviews", [{
        "review_id": r["review_id"], "platform": r["platform"], "business_id": r["business_id"],
        "customer_id": r["customer_id"], "reviewer_display_name": r["reviewer_display_name"],
        "reviewer_total_reviews": r["reviewer_total_reviews"], "review_type": r["review_type"],
        "rating": r["rating"], "title": r["title"], "text": r["text"], "created_at": r["created_at"],
        "service_id": r["service_id"], "job_date": r["job_date"], "estimate_id": r["estimate_id"],
        "cost_reported_usd": r["cost_reported_usd"], "verified_hire": _b(r["verified_hire"]),
        "photos_count": r["photos_count"], "helpful_count": r["helpful_count"],
        "complaint_status": r["complaint_status"],
        "owner_response_text": (r["owner_response"] or {}).get("text"),
        "owner_response_at": (r["owner_response"] or {}).get("created_at"),
        **{f"sub_{k}": (r["sub_ratings"] or {}).get(k)
           for k in ("quality", "value", "punctuality", "professionalism", "responsiveness")}}
        for r in ds.reviews])

    insert("social_posts", [{
        "post_id": p["post_id"], "platform": p["platform"], "post_type": p["post_type"],
        "thread_id": p["thread_id"], "parent_post_id": p["parent_post_id"], "community": p["community"],
        "author_handle": p["author"]["handle"], "author_type": p["author"]["account_type"],
        "customer_id": p["customer_id"], "created_at": p["created_at"], "text": p["text"],
        **{k: p["engagement"].get(k) for k in ("likes", "comments", "shares", "views")}} for p in ds.posts])


def connect(path: Path) -> sqlite3.Connection:
    """Read-only connection with dict-like rows, for query functions and agent tools."""
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run `expert-finder build-sql` first")
    con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    return con
