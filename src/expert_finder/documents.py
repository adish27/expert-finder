"""Turn raw dataset records into documents for semantic search.

One record = one document (reviews and posts are short; business profiles fit in one
embedding). Each document carries filterable metadata. Only raw data is used: nothing
from ground_truth/, and social posts carry no business ids, because resolving informal
mentions ("Mike at the plumbing place") is part of what retrieval has to do.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from langchain_core.documents import Document

from expert_finder.dataset import Dataset

NAMESPACES = {
    "reviews": "Customer reviews and BBB complaints from Google, Yelp, Angi, Thumbtack, BBB and Houzz.",
    "social": "Social posts and threads from Nextdoor, Facebook groups, Reddit, TikTok and Instagram.",
    "businesses": "Business profiles: services and price ranges, team, licenses, insurance, hours, policies.",
}

PLATFORM_NAMES = {"google": "Google", "yelp": "Yelp", "angi": "Angi", "thumbtack": "Thumbtack",
                  "bbb": "BBB", "houzz": "Houzz", "nextdoor": "Nextdoor", "facebook": "Facebook",
                  "reddit": "Reddit", "tiktok": "TikTok", "instagram": "Instagram"}

UNIT_PHRASE = {"flat": "per job", "hourly": "per hour", "per_unit": "per {unit}", "per_sqft": "per sq ft",
               "per_linear_ft": "per linear ft", "per_square": "per roofing square (100 sq ft)",
               "per_zone": "per zone", "per_visit": "per visit", "per_tree": "per tree", "per_year": "per year"}


def _epoch(iso: str) -> int:
    return int(datetime.fromisoformat(iso).timestamp())


def _clean(meta: dict[str, Any]) -> dict[str, Any]:
    """Pinecone metadata accepts str, number, bool and list[str]; drop nulls and empties."""
    return {k: v for k, v in meta.items() if v is not None and v != [] and v != ""}


def _usd(x: float) -> str:
    return f"${x:,.0f}" if x >= 10 else f"${x:,.2f}"


def _customer_place(ds: Dataset, customer_id: str | None) -> dict[str, str]:
    c = ds.customer_by_id.get(customer_id) if customer_id else None
    return {"city": c["city"], "neighborhood": c["neighborhood"]} if c else {}


# --- Reviews -----------------------------------------------------------------


def review_documents(ds: Dataset) -> list[Document]:
    docs = []
    for r in ds.reviews:
        if not r["text"].strip():
            continue  # rating-only reviews carry nothing to search semantically
        b = ds.business_by_id[r["business_id"]]
        svc = ds.service_by_id.get(r["service_id"]) if r["service_id"] else None
        platform = PLATFORM_NAMES[r["platform"]]
        kind = "complaint" if r["review_type"] == "complaint" else "review"
        lines = [f"{platform} {kind} of {b['name']} ({ds.category_name[b['primary_category']]})."]
        if r["rating"] is not None:
            lines.append(f"Rating: {r['rating']}/5.")
        if r["complaint_status"]:
            lines.append(f"Complaint status: {r['complaint_status']}.")
        if svc:
            lines.append(f"Service: {svc['name']}.")
        if r["title"]:
            lines.append(f"Title: {r['title']}.")
        lines.append(r["text"])
        if r["owner_response"]:
            lines.append(f"Owner response: {r['owner_response']['text']}")
        docs.append(Document(id=r["review_id"], page_content="\n".join(lines), metadata=_clean({
            "doc_type": "review", "review_id": r["review_id"], "business_id": b["business_id"],
            "business_name": b["name"], "category": b["primary_category"], "platform": r["platform"],
            "review_type": r["review_type"], "rating": r["rating"], "service_id": r["service_id"],
            "created_at": _epoch(r["created_at"]), "created_date": r["created_at"][:10],
            "customer_id": r["customer_id"], **_customer_place(ds, r["customer_id"]),
            "reviewer_total_reviews": r["reviewer_total_reviews"], "verified_hire": r["verified_hire"],
            "cost_reported_usd": r["cost_reported_usd"], "estimate_id": r["estimate_id"],
            "has_owner_response": r["owner_response"] is not None,
        })))
    return docs


# --- Social posts --------------------------------------------------------------


def social_documents(ds: Dataset) -> list[Document]:
    docs = []
    for p in ds.posts:
        platform = PLATFORM_NAMES[p["platform"]]
        where = f" in {p['community']}" if p["community"] else ""
        lines = [f"{platform} post{where} by {p['author']['handle']} ({p['author']['account_type']}), "
                 f"{p['post_type'].replace('_', ' ')}."]
        if p["parent_post_id"]:
            # A reply like "We used Mike at Ace for ours" only makes sense next to the question.
            root = ds.post_by_id[p["parent_post_id"]]
            lines.append(f"In reply to: \"{root['text']}\"")
        lines.append(p["text"])
        for m in p["media"]:
            if m.get("transcript"):
                lines.append(f"Video transcript: {m['transcript']}")
        if p["hashtags"]:
            lines.append(" ".join(p["hashtags"]))
        place = _customer_place(ds, p["customer_id"])
        if not place and p["platform"] == "nextdoor" and p["community"]:
            nbhd, _, city = p["community"].rpartition(", ")
            place = {"city": city, "neighborhood": nbhd}
        eng = p["engagement"]
        docs.append(Document(id=p["post_id"], page_content="\n".join(lines), metadata=_clean({
            "doc_type": "social_post", "post_id": p["post_id"], "platform": p["platform"],
            "post_type": p["post_type"], "thread_id": p["thread_id"], "parent_post_id": p["parent_post_id"],
            "community": p["community"], "author_handle": p["author"]["handle"],
            "author_type": p["author"]["account_type"], "customer_id": p["customer_id"], **place,
            "created_at": _epoch(p["created_at"]), "created_date": p["created_at"][:10],
            "likes": eng.get("likes"), "views": eng.get("views"),
        })))
    return docs


# --- Business profiles ---------------------------------------------------------


def _price_phrase(o: dict) -> str:
    unit = UNIT_PHRASE.get(o["pricing_model"], "per {unit}").format(unit=o["unit"])
    if o["price_high_usd"] == 0:
        return "free"
    low = "free" if o["price_low_usd"] == 0 else _usd(o["price_low_usd"])
    return f"{low}-{_usd(o['price_high_usd'])} {unit} (typical {_usd(o['typical_price_usd'])})"


def business_profile_text(ds: Dataset, b: dict) -> str:
    cat = ds.category_name
    pp, em, ins = b["pricing_policy"], b["emergency_service"], b["insurance"]
    lines = [f"# {b['name']}", b["description"],
             f"Category: {cat[b['primary_category']]}"
             + (f"; also {', '.join(cat[c] for c in b['secondary_categories'])}" if b["secondary_categories"] else "")
             + ".",
             f"Based in {b['address']['city']}, TX. Serves {', '.join(b['service_area']['cities'])} "
             f"(about {b['service_area']['radius_miles']} miles).",
             f"Size: {b['business_size'].replace('_', ' ')}, {b['employee_count']} employees, founded {b['founded_year']}.",
             f"Hours: Mon-Fri {b['hours']['mon_fri']}, Sat {b['hours']['sat'] or 'closed'}, Sun {b['hours']['sun'] or 'closed'}.",
             (f"Emergency service: {em['hours']}, {em['after_hours_surcharge_pct']}% after-hours surcharge."
              if em["available"] else "No emergency service.")]
    if b["licenses"]:
        lines.append("Licenses: " + "; ".join(f"{lic['type']} ({lic['issuer']}) #{lic['number']}, {lic['status']}, "
                                              f"expires {lic['expires']}" for lic in b["licenses"]) + ".")
    else:
        lines.append("Licenses: none listed.")
    lines.append(f"Insurance: general liability {'yes' if ins['general_liability'] else 'no'} "
                 f"({_usd(ins['liability_coverage_usd'])} coverage), workers comp "
                 f"{'yes' if ins['workers_comp'] else 'no'}, bonded {'yes' if ins['bonded'] else 'no'}.")
    if b["certifications"]:
        lines.append(f"Certifications: {', '.join(b['certifications'])}.")
    lines.append(f"Brands: {', '.join(b['brands'])}. Languages: {', '.join(b['languages'])}.")
    lines.append(f"Payment: {', '.join(b['payment_methods'])}. Financing: {'available' if b['financing_available'] else 'not offered'}.")
    lines.append(f"Warranty: {b['warranty']['workmanship_months']} months workmanship. {b['warranty']['notes']}")
    fee = (f"service call fee {_usd(pp['service_call_fee_usd'])}"
           + (" (waived with repair)" if pp["service_call_fee_waived_with_repair"] else "")
           if pp["service_call_fee_usd"] else "no service call fee")
    lines.append(f"Pricing: hourly rate {_usd(pp['hourly_rate_usd'])}, {fee}, minimum charge "
                 f"{_usd(pp['minimum_charge_usd'])}, {'free estimates' if pp['free_estimates'] else 'estimates may carry a fee'}"
                 + (f", discounts: {', '.join(pp['discounts'])}" if pp["discounts"] else "") + ".")
    rt = b["response_time"]
    lines.append(f"Responsiveness: usually replies within {rt['typical_reply_hours']} hours; "
                 f"typically schedules within {rt['typical_days_to_schedule']} days.")
    lines.append("\n## Services and prices")
    for o in sorted(ds.offerings_of(b["business_id"]), key=lambda o: o["service_id"]):
        svc = ds.service_by_id[o["service_id"]]
        extras = []
        if o["includes"]:
            extras.append(f"includes {', '.join(o['includes'])}")
        if o["excludes"]:
            extras.append(f"excludes {', '.join(o['excludes'])}")
        extras.append(f"lead time ~{o['lead_time_days']} days")
        lines.append(f"- {svc['name']}: {_price_phrase(o)}; {'; '.join(extras)}.")
    lines.append("\n## Team")
    for e in ds.experts_of(b["business_id"]):
        lines.append(f"- {e['bio']}" + (f" Goes by {e['goes_by']}." if e["goes_by"] != e["first_name"] else ""))
    if b["social_accounts"]:
        lines.append("\nOnline: " + ", ".join(f"{k} {v}" for k, v in b["social_accounts"].items()) + ".")
    return "\n".join(lines)


def business_documents(ds: Dataset) -> list[Document]:
    docs = []
    for b in ds.businesses:
        statuses = {lic["status"] for lic in b["licenses"]}
        docs.append(Document(id=b["business_id"], page_content=business_profile_text(ds, b), metadata=_clean({
            "doc_type": "business_profile", "business_id": b["business_id"], "business_name": b["name"],
            "category": b["primary_category"], "secondary_categories": b["secondary_categories"],
            "cities": b["service_area"]["cities"], "home_city": b["address"]["city"],
            "emergency_service": b["emergency_service"]["available"],
            "license_status": "expired" if "expired" in statuses else "active" if statuses else "none",
            "founded_year": b["founded_year"], "business_size": b["business_size"],
            "financing_available": b["financing_available"], "languages": b["languages"],
            "services": [o["service_id"] for o in ds.offerings_of(b["business_id"])],
        })))
    return docs


def build_documents(ds: Dataset) -> dict[str, list[Document]]:
    return {"reviews": review_documents(ds), "social": social_documents(ds), "businesses": business_documents(ds)}
