"""Cypher queries over the knowledge graph.

Two groups:
- expand_*: given ids from a vector search, pull in the connected entities and facts.
- graph-native queries (neighbor_recommendations, praised_experts, ...) that start in the graph
  and answer questions vector search cannot express.

All queries are parameterized and bounded by LIMIT; labels and relationship types are literals
from our own schema, never from user input.
"""

from __future__ import annotations

from typing import Any

from neo4j import Driver


def _run(driver: Driver, database: str, cypher: str, **params) -> list[dict[str, Any]]:
    records, _, _ = driver.execute_query(cypher, params, database_=database)
    return [dict(r) for r in records]


# --- Expansion from vector hits ---------------------------------------------------------------

EXPAND_REVIEWS = """
MATCH (r:Review)-[:ABOUT]->(b:Business)
WHERE r.id IN $ids
OPTIONAL MATCH (r)-[:MENTIONS]->(e:Expert)
OPTIONAL MATCH (r)-[:ABOUT_JOB]->(est:Estimate)
OPTIONAL MATCH (r)-[:FOR_SERVICE]->(s:Service)
WITH r, b, est, s, collect(DISTINCT e.goes_by + ' ' + e.last_name + ' (' + e.role + ')') AS experts
CALL (b) {
    MATCH (b)<-[:ABOUT]-(all:Review)
    RETURN round(avg(all.rating), 2) AS business_avg_rating, count(all.rating) AS business_rated_reviews,
           sum(CASE WHEN all.review_type = 'complaint' THEN 1 ELSE 0 END) AS business_complaints
}
RETURN r.id AS review_id, b.id AS business_id, b.name AS business_name, b.category AS category,
       b.phone AS phone, b.license_status AS license_status, b.license_required AS license_required,
       b.emergency_service AS emergency_service, s.name AS service, experts, business_avg_rating, business_rated_reviews, business_complaints,
       est.total_usd AS quoted_usd, est.final_invoice_usd AS final_invoice_usd
"""

EXPAND_POSTS = """
MATCH (p:Post) WHERE p.id IN $ids
OPTIONAL MATCH (p)-[m:MENTIONS]->(b:Business)
OPTIONAL MATCH (p)-[:MENTIONS]->(e:Expert)
WITH p, collect(DISTINCT {business_id: b.id, business_name: b.name, category: b.category,
                          phone: b.phone, matched_text: m.surface_form}) AS businesses,
     collect(DISTINCT e.goes_by + ' ' + e.last_name) AS experts
RETURN p.id AS post_id, p.community AS community,
       [x IN businesses WHERE x.business_id IS NOT NULL] AS businesses, experts
"""

# A hit on one reply should bring its question and the sibling replies: that is the conversation.
EXPAND_THREADS = """
MATCH (p:Post) WHERE p.id IN $ids AND p.thread_id IS NOT NULL
MATCH (sibling:Post {thread_id: p.thread_id})
WHERE NOT sibling.id IN $ids
OPTIONAL MATCH (sibling)-[m:MENTIONS]->(b:Business)
WITH p.thread_id AS thread_id, sibling, collect(DISTINCT b.name) AS mentioned
RETURN thread_id, sibling.id AS post_id, sibling.post_type AS post_type, sibling.text AS text,
       sibling.author_handle AS author, sibling.created_at AS created_at, mentioned
ORDER BY thread_id, sibling.created_at
LIMIT $limit
"""

BUSINESS_CONTEXT = """
MATCH (b:Business) WHERE b.id IN $ids
OPTIONAL MATCH (b)<-[:ABOUT]-(r:Review)
OPTIONAL MATCH (b)-[o:OFFERS]->(s:Service) WHERE s.id IN $service_ids
WITH b, round(avg(r.rating), 2) AS avg_rating, count(r.rating) AS rated_reviews,
     sum(CASE WHEN r.review_type = 'complaint' THEN 1 ELSE 0 END) AS complaints,
     collect(DISTINCT {service: s.name, typical_price_usd: o.typical_price_usd,
                       price_low_usd: o.price_low_usd, price_high_usd: o.price_high_usd, unit: o.unit}) AS prices
RETURN b.id AS business_id, b.name AS business_name, b.category AS category, b.phone AS phone,
       b.license_status AS license_status, b.license_required AS license_required,
       b.emergency_service AS emergency_service, b.based_in AS based_in, b.hourly_rate_usd AS hourly_rate_usd,
       avg_rating, rated_reviews, complaints,
       [x IN prices WHERE x.service IS NOT NULL] AS prices
"""


def expand_reviews(driver: Driver, database: str, review_ids: list[str]) -> list[dict]:
    return _run(driver, database, EXPAND_REVIEWS, ids=review_ids) if review_ids else []


def expand_posts(driver: Driver, database: str, post_ids: list[str]) -> list[dict]:
    return _run(driver, database, EXPAND_POSTS, ids=post_ids) if post_ids else []


def expand_threads(driver: Driver, database: str, post_ids: list[str], limit: int = 12) -> list[dict]:
    return _run(driver, database, EXPAND_THREADS, ids=post_ids, limit=limit) if post_ids else []


def business_context(driver: Driver, database: str, business_ids: list[str],
                     service_ids: list[str] | None = None) -> list[dict]:
    return _run(driver, database, BUSINESS_CONTEXT, ids=business_ids,
                service_ids=service_ids or []) if business_ids else []


# --- Graph-native queries ---------------------------------------------------------------------


def neighbor_recommendations(driver: Driver, database: str, neighborhood: str | None = None,
                             city: str | None = None, category: str | None = None,
                             limit: int = 5) -> list[dict]:
    """Businesses recommended by people who live in a neighborhood (or city).

    Path: Neighborhood <- LIVES_IN - Customer - POSTED -> Post - MENTIONS -> Business.
    """
    return _run(driver, database, """
        MATCH (n:Neighborhood)<-[:LIVES_IN]-(c:Customer)-[:POSTED]->(p:Post)-[:MENTIONS]->(b:Business)
        WHERE ($neighborhood IS NULL OR n.name = $neighborhood)
          AND ($city IS NULL OR n.city = $city)
          AND ($category IS NULL OR b.category = $category)
          AND p.post_type IN ['reply_recommendation', 'homeowner_recommendation', 'price_reveal',
                              'reply_price_comparison', 'reply_warning', 'storytime_complaint', 'warning']
        // Aggregate the posts before touching reviews: joining both at once multiplies rows.
        WITH b,
             collect(DISTINCT CASE WHEN p.post_type IN ['reply_warning', 'storytime_complaint', 'warning']
                                   THEN p.id END) AS negative_ids,
             collect(DISTINCT CASE WHEN NOT p.post_type IN ['reply_warning', 'storytime_complaint', 'warning']
                                   THEN p.id END) AS positive_ids,
             collect(DISTINCT n.name) AS neighborhoods,
             collect(DISTINCT {post_id: p.id, post_type: p.post_type, text: p.text}) AS examples
        WITH b, neighborhoods, examples,
             size([x IN positive_ids WHERE x IS NOT NULL]) AS positive_posts,
             size([x IN negative_ids WHERE x IS NOT NULL]) AS negative_posts
        CALL (b) {
            MATCH (b)<-[:ABOUT]-(r:Review)
            RETURN round(avg(r.rating), 2) AS avg_rating, count(r.rating) AS reviews
        }
        RETURN b.id AS business_id, b.name AS business_name, b.category AS category, b.phone AS phone,
               positive_posts, negative_posts, positive_posts - negative_posts AS net_sentiment,
               neighborhoods[0..3] AS neighborhoods, avg_rating, reviews, examples[0..3] AS examples
        ORDER BY net_sentiment DESC, positive_posts DESC
        LIMIT $limit""", neighborhood=neighborhood, city=city, category=category, limit=limit)


def praised_experts(driver: Driver, database: str, business_id: str, min_rating: int = 4,
                    limit: int = 5) -> list[dict]:
    """Which named people at a business appear in good (or bad) reviews.

    Path: Review - ABOUT -> Business, Review - MENTIONS -> Expert.
    """
    return _run(driver, database, """
        MATCH (r:Review)-[:ABOUT]->(b:Business {id: $business_id}), (r)-[:MENTIONS]->(e:Expert)
        WITH e, b,
             sum(CASE WHEN r.rating >= $min_rating THEN 1 ELSE 0 END) AS good_reviews,
             sum(CASE WHEN r.rating < $min_rating THEN 1 ELSE 0 END) AS poor_reviews,
             round(avg(r.rating), 2) AS avg_rating,
             collect(CASE WHEN r.rating >= $min_rating THEN {review_id: r.id, text: r.text} END) AS quotes
        OPTIONAL MATCH (e)-[:SPECIALIZES_IN]->(s:Service)
        RETURN e.id AS expert_id, e.goes_by + ' ' + e.last_name AS name, e.role AS role,
               e.years_experience AS years_experience, good_reviews, poor_reviews, avg_rating,
               collect(DISTINCT s.name)[0..4] AS specialties,
               [q IN quotes WHERE q IS NOT NULL][0..2] AS example_reviews
        ORDER BY good_reviews DESC, avg_rating DESC
        LIMIT $limit""", business_id=business_id, min_rating=min_rating, limit=limit)


def review_authenticity(driver: Driver, database: str, business_id: str) -> dict:
    """Split a business's ratings by whether the reviewer is traceable to a real customer.

    A review written by a Customer who also REQUESTED an estimate is corroborated by a
    transaction; a brand-new account with no customer record is not. Graph-native: it joins
    reviews to the estimate history through the customer.
    """
    rows = _run(driver, database, """
        MATCH (b:Business {id: $business_id})<-[:ABOUT]-(r:Review)
        WHERE r.rating IS NOT NULL
        OPTIONAL MATCH (c:Customer)-[:WROTE]->(r)
        OPTIONAL MATCH (c)-[:REQUESTED]->(e:Estimate)-[:QUOTED_BY]->(b)
        WITH r, c, count(e) AS estimates_with_business
        WITH CASE WHEN estimates_with_business > 0 THEN 'hired_and_reviewed'
                  WHEN c IS NOT NULL THEN 'known_customer_no_estimate'
                  ELSE 'untraceable_account' END AS reviewer_type,
             r
        RETURN reviewer_type, count(r) AS reviews, round(avg(r.rating), 2) AS avg_rating,
               min(substring(r.created_at, 0, 10)) AS earliest,
               max(substring(r.created_at, 0, 10)) AS latest,
               round(avg(r.reviewer_total_reviews), 1) AS avg_reviewer_history
        ORDER BY reviews DESC""", business_id=business_id)
    overall = _run(driver, database, """
        MATCH (:Business {id: $business_id})<-[:ABOUT]-(r:Review) WHERE r.rating IS NOT NULL
        RETURN round(avg(r.rating), 2) AS avg_rating, count(r) AS reviews""", business_id=business_id)
    return {"business_id": business_id, "overall": overall[0] if overall else {}, "by_reviewer_type": rows}


def competing_quotes(driver: Driver, database: str, project_id: str | None = None,
                     estimate_id: str | None = None) -> list[dict]:
    """Every quote for one job, including the ones that lost and why.

    Path: Estimate - COMPETED_WITH -> Estimate, each - QUOTED_BY -> Business.
    """
    return _run(driver, database, """
        MATCH (e:Estimate)-[:QUOTED_BY]->(b:Business), (e)-[:FOR_SERVICE]->(s:Service)
        WHERE ($project_id IS NOT NULL AND e.project_id = $project_id)
           OR ($estimate_id IS NOT NULL AND (e.id = $estimate_id
               OR EXISTS { (e)-[:COMPETED_WITH]-(:Estimate {id: $estimate_id}) }))
        OPTIONAL MATCH (b)<-[:ABOUT]-(r:Review)
        RETURN e.id AS estimate_id, e.project_id AS project_id, b.id AS business_id,
               b.name AS business_name, s.name AS service, e.quantity AS quantity, e.unit AS unit,
               e.total_usd AS quoted_usd, e.final_invoice_usd AS final_invoice_usd, e.status AS status,
               e.decline_reason AS decline_reason, round(avg(r.rating), 2) AS business_avg_rating
        ORDER BY quoted_usd""", project_id=project_id, estimate_id=estimate_id)
