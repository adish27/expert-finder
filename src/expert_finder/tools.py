"""Agent tools over the three stores.

Tools are fixed and typed rather than letting the model write SQL or Cypher: answers stay
reproducible and cheap to evaluate. Each docstring is the model's instruction for when to reach
for that tool, so they are written as carefully as the code.

Every call is captured by a ToolRecorder: the model reads formatted text, while the UI and the
evaluation get the structured EvidenceBundle behind it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from langchain_core.tools import BaseTool, tool

from expert_finder import graph_queries as gq
from expert_finder import sql_queries as sq
from expert_finder.app import ExpertFinder, Mode
from expert_finder.retrieval import EvidenceBundle

# Tools available in both modes; the rest are graph-only.
SHARED_TOOLS = ["search_reviews_and_posts", "search_business_profiles", "find_businesses",
                "get_price_range", "check_quote", "get_business_report"]
GRAPH_TOOLS = ["neighbor_recommendations", "who_to_ask_for", "check_review_authenticity"]


@dataclass
class ToolCall:
    name: str
    args: dict[str, Any]
    output: str
    evidence: EvidenceBundle | None = None


@dataclass
class ToolRecorder:
    """Collects what the tools did during one agent run."""

    calls: list[ToolCall] = field(default_factory=list)

    def record(self, name: str, args: dict, output: str, evidence: EvidenceBundle | None = None) -> str:
        self.calls.append(ToolCall(name=name, args=args, output=output, evidence=evidence))
        return output

    @property
    def bundles(self) -> list[EvidenceBundle]:
        return [c.evidence for c in self.calls if c.evidence is not None]

    def evidence_items(self) -> list:
        return [item for b in self.bundles for item in b.items]

    def reset(self) -> None:
        self.calls.clear()


def _usd(x: float | None) -> str:
    return f"${x:,.0f}" if x is not None else "n/a"


def _quartile_line(label: str, q: dict | None) -> str:
    if not q:
        return f"{label}: no data"
    return (f"{label}: {q['n']} data points, median {_usd(q['median'])} "
            f"(25th-75th percentile {_usd(q['p25'])}-{_usd(q['p75'])}, full range {_usd(q['min'])}-{_usd(q['max'])})")


def build_tools(app: ExpertFinder, mode: Mode, recorder: ToolRecorder) -> list[BaseTool]:
    """All tools for the given mode. Graph mode adds the three graph-native tools."""
    retriever = app.retriever(mode)

    # app.sql is thread-local (LangGraph runs tools in a thread pool), so fetch it per call
    # instead of capturing one connection when the tools are built.
    def resolve_one_business(name: str) -> dict | None:
        matches = sq.resolve_business(app.sql, name)
        return matches[0] if matches else None

    # --- retrieval (the tool where the two modes differ) ----------------------

    @tool
    def search_reviews_and_posts(query: str, city: str | None = None, category: str | None = None,
                                 max_rating: int | None = None) -> str:
        """Search what customers and neighbors have written: reviews from Google, Yelp, Angi,
        Thumbtack, BBB and Houzz, plus posts from Nextdoor, Facebook, Reddit, TikTok and Instagram.

        Use this for experience and reputation questions: what people say about a company, who
        neighbors recommend, complaints about pricing or no-shows, whether work held up.
        Set max_rating=2 to look specifically for bad experiences.
        category must be one of: plumbing, electrical, hvac, roofing, landscaping.
        """
        filters: dict[str, dict] = {}
        review_filter: dict[str, Any] = {}
        if category:
            review_filter["category"] = category
            filters["social"] = {}
        if max_rating is not None:
            review_filter["rating"] = {"$lte": max_rating}
        if city:
            filters["social"] = {"city": city}
            review_filter["city"] = city
        if review_filter:
            filters["reviews"] = review_filter
        bundle = retriever.retrieve(query, namespaces=("reviews", "social"), filters=filters or None)
        text = bundle.to_prompt() if bundle.items else "No matching reviews or posts."
        return recorder.record("search_reviews_and_posts",
                               {"query": query, "city": city, "category": category, "max_rating": max_rating},
                               text, bundle)

    @tool
    def search_business_profiles(query: str, city: str | None = None, category: str | None = None) -> str:
        """Search business profiles by what they do: services offered, price ranges, brands,
        certifications, languages spoken, hours and policies.

        Use this when the need is descriptive ("someone who installs tankless heaters and speaks
        Spanish") rather than a filter on structured fields; use find_businesses for hard filters.
        """
        f: dict[str, Any] = {}
        if city:
            f["cities"] = {"$in": [city]}
        if category:
            f["category"] = category
        bundle = retriever.retrieve(query, namespaces=("businesses",), filters={"businesses": f} if f else None)
        text = bundle.to_prompt() if bundle.items else "No matching business profiles."
        return recorder.record("search_business_profiles", {"query": query, "city": city, "category": category},
                               text, bundle)

    # --- structured facts (shared) --------------------------------------------

    @tool
    def find_businesses(category: str | None = None, city: str | None = None, service: str | None = None,
                        emergency_only: bool = False, language: str | None = None,
                        min_rating: float | None = None) -> str:
        """Find businesses by hard filters, ranked by average rating.

        Use this when the request has concrete constraints: a trade, a city, a specific service,
        "someone who can come out tonight" (emergency_only), a language, or a rating floor.
        category must be one of: plumbing, electrical, hvac, roofing, landscaping.
        service is free text such as "tankless water heater" or "can lights"; it is matched to the
        service catalog. Returns business ids you can pass to get_business_report.
        """
        service_id = None
        notes = []
        if service:
            matches = sq.resolve_service(app.sql, service, category)
            if matches:
                service_id = matches[0]["service_id"]
                notes.append(f"Interpreted service {service!r} as {matches[0]['name']}.")
            else:
                notes.append(f"No service in the catalog matches {service!r}; ignoring it.")
        rows = sq.find_businesses(app.sql, category=category, city=city, service_id=service_id,
                                  emergency=True if emergency_only else None, language=language,
                                  min_rating=min_rating, limit=8)
        if emergency_only:
            # For an urgent job, how fast they answer matters more than the star rating.
            rows.sort(key=lambda r: (r["typical_reply_hours"], -(r["avg_rating"] or 0)))
            notes.append("Sorted by how fast they respond, because this is an urgent request.")
        if not rows:
            text = "No businesses match those filters. Try relaxing the city or the rating floor."
        else:
            lines = notes + [f"{len(rows)} matches, best rated first:"]
            for r in rows:
                price = (f", {service} typically {_usd(r['typical_price_usd'])} per {r['price_unit']}"
                         if r.get("typical_price_usd") else "")
                trade = ("" if r["is_primary_trade"] else
                         f", {category} is a SIDELINE for them, their main trade is {r['primary_category']}")
                lines.append(
                    f"- [{r['business_id']}] {r['name']} ({r['primary_category']}, based in {r['based_in']}{trade}): "
                    f"rated {r['avg_rating']} from {r['rated_reviews']} reviews, {r['complaints']} complaints; "
                    f"license {r['license_status']}; "
                    f"{'takes 24/7 emergency calls' if r['emergency_available'] else 'no emergency service'}; "
                    f"replies in ~{r['typical_reply_hours']}h; books routine work ~"
                    f"{r['typical_days_to_schedule']} days out; "
                    f"hourly {_usd(r['hourly_rate_usd'])}{price}; phone {r['phone']}")
            text = "\n".join(lines)
        return recorder.record("find_businesses", {"category": category, "city": city, "service": service,
                                                   "emergency_only": emergency_only, "language": language,
                                                   "min_rating": min_rating}, text)

    @tool
    def get_price_range(service: str, city: str | None = None) -> str:
        """What a service actually costs: advertised price ranges plus real quotes and final invoices.

        Use this for "what should this cost" questions, and before judging whether a quote the user
        received is reasonable. service is free text such as "new roof" or "sprinkler repair".
        """
        matches = sq.resolve_service(app.sql, service)
        if not matches:
            text = f"No service in the catalog matches {service!r}."
        else:
            svc = matches[0]
            stats = sq.price_stats(app.sql, svc["service_id"], city)
            adv = stats["advertised"]
            lines = [f"{svc['name']} in {stats['city']} (priced per {stats['service']['unit']}):",
                     f"- {adv['businesses']} businesses offer it; advertised from "
                     f"{_usd(adv['lowest_advertised'])} to {_usd(adv['highest_advertised'])}",
                     "- " + _quartile_line("Real quotes issued", stats["quotes"]),
                     "- " + _quartile_line("Final invoices on completed jobs", stats["final_invoices"])]
            if stats["quotes_per_unit"]:
                lines.append("- " + _quartile_line(f"Quotes per {stats['service']['unit']}", stats["quotes_per_unit"]))
            text = "\n".join(lines)
        return recorder.record("get_price_range", {"service": service, "city": city}, text)

    @tool
    def check_quote(service: str, amount_usd: float, city: str | None = None,
                    quantity: float | None = None) -> str:
        """Judge whether a quote the user received is low, typical, high or very high, by comparing
        it with real quotes for the same service.

        Pass quantity when the job is priced per unit (roofing squares, sq ft, zones, trees) so the
        comparison is per unit rather than per job.
        """
        matches = sq.resolve_service(app.sql, service)
        if not matches:
            text = f"No service in the catalog matches {service!r}."
        else:
            svc = matches[0]
            r = sq.compare_quote(app.sql, svc["service_id"], amount_usd, city, quantity)
            if r.get("comparable_quotes", 0) == 0:
                text = f"No comparable quotes for {svc['name']}."
            else:
                text = (f"{_usd(amount_usd)} for {svc['name']} is {r['verdict'].upper()}: higher than "
                        f"{r['percentile']}% of {r['comparable_quotes']} comparable quotes. "
                        + _quartile_line("Comparable quotes", r["distribution"])
                        + (f" ({r['note']})" if r.get("note") else ""))
        return recorder.record("check_quote", {"service": service, "amount_usd": amount_usd, "city": city,
                                               "quantity": quantity}, text)

    @tool
    def get_business_report(business_name: str) -> str:
        """Full trust report on one business: rating trend over time, how often the final bill
        exceeded the quote, licenses and insurance, complaints, and who works there.

        Use this before recommending a company, and whenever the user asks whether a company is
        any good, trustworthy, legitimate, or has got worse.
        """
        b = resolve_one_business(business_name)
        if not b:
            text = f"No business found matching {business_name!r}."
        else:
            facts = sq.business_facts(app.sql, b["business_id"])
            ratings = sq.rating_summary(app.sql, b["business_id"])
            quotes = sq.quote_accuracy(app.sql, b["business_id"])
            if facts["licenses"]:
                lic = ", ".join(f"{x['type']} #{x['number']} ({x['status']}, expires {x['expires']})"
                                for x in facts["licenses"])
            elif facts["license_required"]:
                lic = f"NONE ON FILE, and {facts['primary_category']} work requires a Texas license"
            else:
                lic = f"none on file; {facts['primary_category']} work does not require a Texas license"
            trend = ", ".join(f"{p['period']}: {p['avg_rating']} ({p['rated']} reviews)"
                              for p in ratings["by_half_year"] if p["rated"])
            platforms = ", ".join(f"{p['platform']} {p['avg_rating']} ({p['reviews']})"
                                  for p in ratings["by_platform"])
            new = ratings["new_account_reviews"]
            lines = [
                f"[{b['business_id']}] {facts['name']} — {facts['primary_category']}, "
                f"{facts['business_size'].replace('_', ' ')}, founded {facts['founded_year']}, "
                f"based in {facts['city']}, serves {', '.join(facts['cities_served'])}.",
                f"Contact: {facts['phone']}. Languages: {', '.join(facts['attributes'].get('language', []))}.",
                f"Licenses: {lic}.",
                f"Insurance: general liability {'yes' if facts['general_liability'] else 'NO'} "
                f"({_usd(facts['liability_coverage_usd'])}), workers comp "
                f"{'yes' if facts['workers_comp'] else 'no'}, bonded {'yes' if facts['bonded'] else 'no'}.",
                f"Warranty: {facts['warranty_months']} months workmanship. Emergency service: "
                f"{'yes' if facts['emergency_available'] else 'no'}.",
                f"Rating: {ratings['avg_rating']} from {ratings['rated']} ratings. "
                f"Complaints: {ratings['complaints']} ({ratings['unresolved_complaints']} unresolved). "
                f"Owner replies to {ratings['owner_response_rate']:.0%} of reviews.",
                f"By platform: {platforms}.",
                f"Rating over time — {trend}.",
                f"Quote accuracy: of {quotes['completed_jobs_with_quote']} completed jobs with a written "
                f"quote, {quotes['jobs_over_quote']} were billed more than 5% over"
                + (f" (median overrun {quotes['median_overrun_when_over_pct']}%, worst "
                   f"{quotes['worst_overrun_pct']}%)" if quotes["jobs_over_quote"] else "") + ".",
                f"Reviews from brand-new accounts (2 or fewer reviews): {new['reviews']} "
                f"({(new['share_of_rated'] or 0):.0%} of ratings), averaging {new['avg_rating']}; "
                f"most in any 30-day window: {new['max_in_30_days']}. "
                f"Excluding them, the rating is {ratings['rating_from_established_accounts']['avg_rating']} "
                f"from {ratings['rating_from_established_accounts']['rated']} reviews.",
                "Team: " + "; ".join(f"{t['goes_by']} {t['last_name']} ({t['role']}, "
                                     f"{t['years_experience']}y)" for t in facts["team"]) + ".",
            ]
            text = "\n".join(lines)
        return recorder.record("get_business_report", {"business_name": business_name}, text)

    # --- graph-only tools -----------------------------------------------------

    @tool
    def neighbor_recommendations(neighborhood: str | None = None, city: str | None = None,
                                 category: str | None = None) -> str:
        """Who people in a specific neighborhood or city actually recommend, from their own posts.

        Use this for "who do my neighbors use", "anyone in <neighborhood> have someone for X", or
        when local word of mouth matters more than overall star ratings. Counts recommendations and
        warnings separately, so a company with both shows up as contested.
        """
        rows = gq.neighbor_recommendations(app.driver, app.database, neighborhood=neighborhood, city=city,
                                           category=category, limit=6)
        if not rows:
            text = f"No neighborhood posts found for {neighborhood or city or 'that area'}."
        else:
            lines = [f"Word of mouth in {neighborhood or city}"
                     + (f" for {category}" if category else "") + ":"]
            for r in rows:
                lines.append(
                    f"- [{r['business_id']}] {r['business_name']}: {r['positive_posts']} recommendation(s), "
                    f"{r['negative_posts']} warning(s) from residents of "
                    f"{', '.join(r['neighborhoods'])}; overall rating {r['avg_rating']} from "
                    f"{r['reviews']} reviews; phone {r['phone']}")
                for ex in r["examples"][:2]:
                    lines.append(f"    [{ex['post_id']}] {ex['post_type'].replace('_', ' ')}: \"{ex['text']}\"")
            text = "\n".join(lines)
        return recorder.record("neighbor_recommendations",
                               {"neighborhood": neighborhood, "city": city, "category": category}, text)

    @tool
    def who_to_ask_for(business_name: str) -> str:
        """Which named person at a business customers single out, good or bad.

        Use this when the user wants a specific technician to request, or asks who to avoid at a
        company. Works by matching the people named in reviews to that company's staff.
        """
        b = resolve_one_business(business_name)
        if not b:
            text = f"No business found matching {business_name!r}."
        else:
            rows = gq.praised_experts(app.driver, app.database, b["business_id"], limit=5)
            if not rows:
                text = f"No individual staff member is named in reviews of {b['name']}."
            else:
                lines = [f"People named in reviews of {b['name']}:"]
                for r in rows:
                    lines.append(f"- [{r['expert_id']}] {r['name']} ({r['role']}, {r['years_experience']}y): "
                                 f"named in {r['good_reviews']} good and {r['poor_reviews']} poor reviews, "
                                 f"average {r['avg_rating']}"
                                 + (f"; works on {', '.join(r['specialties'][:3])}" if r["specialties"] else ""))
                    for ex in r["example_reviews"][:1]:
                        lines.append(f"    [{ex['review_id']}] \"{ex['text'][:200]}\"")
                text = "\n".join(lines)
        return recorder.record("who_to_ask_for", {"business_name": business_name}, text)

    @tool
    def check_review_authenticity(business_name: str) -> str:
        """Check whether a business's reviews come from real, traceable customers.

        Use this when a rating looks too good, when the user is suspicious of the reviews, or before
        recommending a company whose rating rests on a burst of recent five-star reviews. It splits
        ratings by whether the reviewer can be traced to a customer who actually requested a quote
        from that business, which is far stronger evidence than the review text alone.
        """
        b = resolve_one_business(business_name)
        if not b:
            text = f"No business found matching {business_name!r}."
        else:
            r = gq.review_authenticity(app.driver, app.database, b["business_id"])
            labels = {"hired_and_reviewed": "reviewers who requested a quote from this business",
                      "known_customer_no_estimate": "reviewers who are known customers but have no quote from them",
                      "untraceable_account": "reviewers with no customer record at all"}
            lines = [f"{b['name']} shows an overall rating of {r['overall'].get('avg_rating')} "
                     f"from {r['overall'].get('reviews')} ratings. Broken down by how traceable the reviewer is:"]
            for row in r["by_reviewer_type"]:
                lines.append(f"- {labels.get(row['reviewer_type'], row['reviewer_type'])}: {row['reviews']} "
                             f"reviews averaging {row['avg_rating']}, posted between {row['earliest']} and "
                             f"{row['latest']}, with an average of {row['avg_reviewer_history']} reviews per account")
            text = "\n".join(lines)
        return recorder.record("check_review_authenticity", {"business_name": business_name}, text)

    tools = [search_reviews_and_posts, search_business_profiles, find_businesses, get_price_range,
             check_quote, get_business_report]
    if mode == "graph":
        tools += [neighbor_recommendations, who_to_ask_for, check_review_authenticity]
    return tools
