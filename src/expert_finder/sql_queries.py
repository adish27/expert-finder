"""Fixed, parameterized queries over the SQLite store; the agent's structured tools wrap these."""

from __future__ import annotations

import difflib
import re
import sqlite3
from datetime import date
from statistics import mean, median, quantiles

NEW_ACCOUNT_MAX_REVIEWS = 2  # reviewers with this many reviews or fewer count as brand-new accounts
OVER_QUOTE_THRESHOLD = 0.05  # a final bill more than 5% above the quote counts as "over quote"


def _rows(con: sqlite3.Connection, sql: str, params=()) -> list[dict]:
    return [dict(r) for r in con.execute(sql, params)]


def _short(name: str) -> str:
    return re.sub(r"\s+(LLC|Inc\.?|Co\.)$", "", name).strip()


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9& ]+", " ", s.lower()).strip()


def _quartiles(values: list[float]) -> dict | None:
    if not values:
        return None
    q = quantiles(values, n=4) if len(values) >= 2 else [values[0]] * 3
    return {"n": len(values), "min": round(min(values), 2), "p25": round(q[0], 2), "median": round(median(values), 2),
            "p75": round(q[2], 2), "max": round(max(values), 2)}


# --- Resolution: free text -> ids --------------------------------------------------------------


def resolve_business(con: sqlite3.Connection, name: str, limit: int = 3) -> list[dict]:
    """Best matches for a business name, most likely first. Exact/prefix matches beat fuzzy ones."""
    q = _norm(name)
    candidates = _rows(con, "SELECT business_id, name, primary_category, city FROM businesses")
    scored = []
    for c in candidates:
        full, short = _norm(c["name"]), _norm(_short(c["name"]))
        if q in (full, short):
            score = 1.0
        elif full.startswith(q) or short.startswith(q) or (len(q) >= 4 and q in full):
            score = 0.9
        else:
            score = max(difflib.SequenceMatcher(None, q, full).ratio(), difflib.SequenceMatcher(None, q, short).ratio())
        scored.append((score, c))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [{**c, "match_score": round(s, 2)} for s, c in scored[:limit] if s >= 0.6]


def resolve_service(con: sqlite3.Connection, text: str, category: str | None = None, limit: int = 3) -> list[dict]:
    """Best matching services for free text like 'tankless' or 'can lights'."""
    q = _norm(text)
    rows = _rows(con, "SELECT service_id, category, name, common_name, aliases FROM services"
                      + (" WHERE category = ?" if category else ""), (category,) if category else ())
    scored = []
    for r in rows:
        forms = [_norm(r["name"]), _norm(r["common_name"])] + [_norm(a) for a in (r["aliases"] or "").split("; ") if a]
        if q in forms:
            score = 1.0
        elif any(f and (f in q or q in f) for f in forms):
            # Prefer the longest overlapping form: "tankless water heater" over "water heater".
            score = 0.8 + 0.1 * max(len(f) for f in forms if f and (f in q or q in f)) / max(len(q), 1)
        else:
            score = max(difflib.SequenceMatcher(None, q, f).ratio() for f in forms)
        scored.append((min(score, 0.99) if score < 1 else score, r))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [{k: r[k] for k in ("service_id", "category", "name")} | {"match_score": round(s, 2)}
            for s, r in scored[:limit] if s >= 0.55]


# --- Business search ---------------------------------------------------------------------------


def find_businesses(con: sqlite3.Connection, category: str | None = None, city: str | None = None,
                    service_id: str | None = None, emergency: bool | None = None, language: str | None = None,
                    min_rating: float | None = None, limit: int = 10) -> list[dict]:
    """Businesses matching hard filters, with rating stats and (if a service is given) its advertised price."""
    where, params = [], []
    if category:
        where.append("b.business_id IN (SELECT business_id FROM business_categories WHERE category = ?)")
        params.append(category)
    if city:
        where.append("b.business_id IN (SELECT business_id FROM business_service_areas WHERE city = ?)")
        params.append(city)
    if service_id:
        where.append("b.business_id IN (SELECT business_id FROM offerings WHERE service_id = ?)")
        params.append(service_id)
    if emergency is not None:
        where.append("b.emergency_available = ?")
        params.append(int(emergency))
    if language:
        where.append("b.business_id IN (SELECT business_id FROM business_attributes "
                     "WHERE kind = 'language' AND value = ?)")
        params.append(language)
    sql = f"""
        SELECT b.business_id, b.name, b.primary_category, b.city AS based_in, b.emergency_available,
               b.typical_reply_hours, b.typical_days_to_schedule, b.hourly_rate_usd,
               (SELECT group_concat(status) FROM business_licenses l WHERE l.business_id = b.business_id) AS license_status,
               round(avg(r.rating), 2) AS avg_rating, count(r.rating) AS rated_reviews,
               sum(r.review_type = 'complaint') AS complaints,
               o.price_low_usd, o.typical_price_usd, o.price_high_usd, o.unit AS price_unit
        FROM businesses b
        LEFT JOIN reviews r ON r.business_id = b.business_id
        LEFT JOIN offerings o ON o.business_id = b.business_id AND o.service_id = ?
        {"WHERE " + " AND ".join(where) if where else ""}
        GROUP BY b.business_id
        {"HAVING avg(r.rating) >= ?" if min_rating is not None else ""}
        ORDER BY avg_rating DESC, rated_reviews DESC
        LIMIT ?"""
    params = [service_id, *params] + ([min_rating] if min_rating is not None else []) + [limit]
    rows = _rows(con, sql, params)
    for r in rows:
        r["emergency_available"] = bool(r["emergency_available"])
        r["license_status"] = r["license_status"] or "none listed"
    return rows


def business_facts(con: sqlite3.Connection, business_id: str) -> dict | None:
    rows = _rows(con, "SELECT * FROM businesses WHERE business_id = ?", (business_id,))
    if not rows:
        return None
    b = rows[0]
    b["categories"] = [r["category"] for r in _rows(
        con, "SELECT category FROM business_categories WHERE business_id = ? ORDER BY is_primary DESC", (business_id,))]
    b["cities_served"] = [r["city"] for r in _rows(
        con, "SELECT city FROM business_service_areas WHERE business_id = ?", (business_id,))]
    b["licenses"] = _rows(con, "SELECT type, issuer, number, status, expires FROM business_licenses "
                               "WHERE business_id = ?", (business_id,))
    attrs: dict[str, list[str]] = {}
    for r in _rows(con, "SELECT kind, value FROM business_attributes WHERE business_id = ?", (business_id,)):
        attrs.setdefault(r["kind"], []).append(r["value"])
    b["attributes"] = attrs
    b["team"] = _rows(con, "SELECT expert_id, first_name, last_name, goes_by, role, years_experience, licenses "
                           "FROM experts WHERE business_id = ? ORDER BY is_owner DESC", (business_id,))
    return b


# --- Prices and quotes -------------------------------------------------------------------------


def price_stats(con: sqlite3.Connection, service_id: str, city: str | None = None) -> dict:
    """Advertised price ranges and real quotes/invoices for a service, optionally within one city."""
    svc = _rows(con, "SELECT service_id, name, pricing_model, unit FROM services WHERE service_id = ?", (service_id,))
    if not svc:
        raise ValueError(f"unknown service {service_id!r}")
    advertised = _rows(con, f"""
        SELECT o.price_low_usd, o.typical_price_usd, o.price_high_usd FROM offerings o
        WHERE o.service_id = ? {"AND o.business_id IN (SELECT business_id FROM business_service_areas WHERE city = ?)"
                                 if city else ""}""", (service_id, city) if city else (service_id,))
    quotes = _rows(con, f"""
        SELECT total_usd, quantity, status, final_invoice_usd FROM estimates
        WHERE service_id = ? {"AND city = ?" if city else ""}""", (service_id, city) if city else (service_id,))
    per_unit = svc[0]["pricing_model"] not in ("flat", "per_visit", "per_year")
    return {
        "service": svc[0], "city": city or "all cities",
        "advertised": {
            "businesses": len(advertised),
            "typical_price_per_unit": _quartiles([a["typical_price_usd"] for a in advertised]),
            "lowest_advertised": min((a["price_low_usd"] for a in advertised), default=None),
            "highest_advertised": max((a["price_high_usd"] for a in advertised), default=None),
        },
        "quotes": _quartiles([q["total_usd"] for q in quotes]),
        "quotes_per_unit": _quartiles([q["total_usd"] / q["quantity"] for q in quotes if q["quantity"]]) if per_unit else None,
        "final_invoices": _quartiles([q["final_invoice_usd"] for q in quotes if q["final_invoice_usd"]]),
    }


def compare_quote(con: sqlite3.Connection, service_id: str, amount: float, city: str | None = None,
                  quantity: float | None = None) -> dict:
    """Where a quote falls among real quotes for the same service (per unit when quantity is given)."""
    sql = "SELECT total_usd, quantity FROM estimates WHERE service_id = ?" + (" AND city = ?" if city else "")
    rows = _rows(con, sql, (service_id, city) if city else (service_id,))
    if len(rows) < 5 and city:  # too few local quotes: compare against the whole metro instead
        return {**compare_quote(con, service_id, amount, None, quantity), "note": f"few quotes in {city}; used all cities"}
    values = [r["total_usd"] / r["quantity"] for r in rows if r["quantity"]] if quantity else [r["total_usd"] for r in rows]
    target = amount / quantity if quantity else amount
    if not values:
        return {"service_id": service_id, "amount": amount, "comparable_quotes": 0, "verdict": "no data"}
    pct = round(100 * sum(v <= target for v in values) / len(values))
    verdict = "low" if pct < 25 else "typical" if pct <= 75 else "high" if pct <= 90 else "very high"
    return {"service_id": service_id, "amount": amount, "per_unit": quantity is not None, "compared_value": round(target, 2),
            "comparable_quotes": len(values), "percentile": pct, "verdict": verdict, "distribution": _quartiles(values)}


def quote_accuracy(con: sqlite3.Connection, business_id: str) -> dict:
    """How the final bill compares to the written quote on this business's completed jobs."""
    rows = _rows(con, "SELECT total_usd, subtotal_usd, final_invoice_usd FROM estimates "
                      "WHERE business_id = ? AND status = 'accepted' AND final_invoice_usd IS NOT NULL", (business_id,))
    # Final invoices are pre-tax, so compare them with the pre-tax subtotal.
    overruns = [(r["final_invoice_usd"] - r["subtotal_usd"]) / r["subtotal_usd"] for r in rows if r["subtotal_usd"]]
    over = [o for o in overruns if o > OVER_QUOTE_THRESHOLD]
    return {"business_id": business_id, "completed_jobs_with_quote": len(overruns),
            "jobs_over_quote": len(over),
            "share_over_quote": round(len(over) / len(overruns), 2) if overruns else None,
            "median_overrun_when_over_pct": round(100 * median(over), 1) if over else 0.0,
            "worst_overrun_pct": round(100 * max(overruns), 1) if overruns else None}


# --- Ratings and trust signals -----------------------------------------------------------------


def rating_summary(con: sqlite3.Connection, business_id: str, split_date: str | None = None) -> dict:
    """Ratings overall, by platform and by half-year, plus complaint and review-authenticity signals."""
    base = "FROM reviews WHERE business_id = ?"
    overall = _rows(con, f"SELECT round(avg(rating), 2) AS avg_rating, count(rating) AS rated, "
                         f"sum(review_type = 'complaint') AS complaints, "
                         f"sum(review_type = 'complaint' AND complaint_status = 'unresolved') AS unresolved_complaints, "
                         f"round(avg(owner_response_text IS NOT NULL), 2) AS owner_response_rate {base}", (business_id,))[0]
    by_platform = _rows(con, f"SELECT platform, round(avg(rating), 2) AS avg_rating, count(*) AS reviews {base} "
                             f"GROUP BY platform ORDER BY reviews DESC", (business_id,))
    by_half_year = _rows(con, f"""
        SELECT substr(created_at, 1, 4) || CASE WHEN CAST(substr(created_at, 6, 2) AS INTEGER) <= 6
               THEN '-H1' ELSE '-H2' END AS period,
               round(avg(rating), 2) AS avg_rating, count(rating) AS rated {base}
        GROUP BY period ORDER BY period""", (business_id,))
    # Public signals only (what a review platform shows): reviewer account size and timing.
    new_rows = _rows(con, f"SELECT rating, substr(created_at, 1, 10) AS day {base} "
                          f"AND reviewer_total_reviews <= ? AND rating IS NOT NULL ORDER BY day",
                     (business_id, NEW_ACCOUNT_MAX_REVIEWS))
    days = [date.fromisoformat(r["day"]) for r in new_rows]
    burst = max((sum(1 for d2 in days if 0 <= (d2 - d1).days < 30) for d1 in days), default=0)
    new_accounts = {"reviews": len(new_rows),
                    "avg_rating": round(mean(r["rating"] for r in new_rows), 2) if new_rows else None,
                    "share_of_rated": round(len(new_rows) / overall["rated"], 2) if overall["rated"] else None,
                    "max_in_30_days": burst}
    established = _rows(con, f"SELECT round(avg(rating), 2) AS avg_rating, count(rating) AS rated {base} "
                             f"AND reviewer_total_reviews > ?", (business_id, NEW_ACCOUNT_MAX_REVIEWS))[0]
    out = {"business_id": business_id, **overall, "by_platform": by_platform, "by_half_year": by_half_year,
           "new_account_reviews": new_accounts, "rating_from_established_accounts": established}
    if split_date:
        out["before_split"], out["after_split"] = (
            _rows(con, f"SELECT round(avg(rating), 2) AS avg_rating, count(rating) AS rated {base} AND created_at {op} ?",
                  (business_id, split_date))[0] for op in ("<", ">="))
    return out
