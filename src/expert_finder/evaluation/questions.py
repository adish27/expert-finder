"""The evaluation question set, built from ground truth.

The questions are generated rather than hand-written, so they cannot drift out of step with the
data: the company that actually has a burst of fake reviews is the one the trust question asks
about. Generation is deterministic - everything is selected in sorted order, nothing is sampled -
so two runs score the same questions and experiments stay comparable.

Each question carries the reference answer the evaluators score against. `kind` decides which
evaluators apply, because "did it avoid the bad roofer" is meaningless for a pricing question.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from expert_finder.dataset import Dataset
from expert_finder.evaluation.ground_truth import GroundTruth

# How many of each kind. Small on purpose: every question is a full agent run in each mode.
PER_KIND = {"trust": 2, "declining": 2, "recommend": 5, "neighborhood": 2, "expert": 2, "quote": 2}

# Above compare_quote's own five-quote floor, so the expected verdict comes from the city's
# own quotes and not from its fallback to the whole metro.
MIN_QUOTES_FOR_A_PRICE_QUESTION = 10


def _trade(ds: Dataset, category: str) -> str:
    """The trade in plain words: the catalog name carries a parenthetical gloss a person would not say."""
    return ds.category_name[category].split(" (")[0].lower()


@dataclass(frozen=True)
class EvalQuestion:
    id: str
    kind: str
    question: str
    reference: dict[str, Any] = field(default_factory=dict)

    def as_example(self) -> tuple[dict, dict]:
        return {"question": self.question, "kind": self.kind, "id": self.id}, self.reference


def _percentile_of(amount: float, values: list[float]) -> int:
    """The same calculation sql_queries.compare_quote makes, so the expected verdict matches."""
    return round(100 * sum(v <= amount for v in values) / len(values))


def _verdict_for(amount: float, values: list[float]) -> str:
    pct = _percentile_of(amount, values)
    return "low" if pct < 25 else "typical" if pct <= 75 else "high" if pct <= 90 else "very high"


def _trust_questions(gt: GroundTruth) -> list[EvalQuestion]:
    """Companies whose good rating is not real. The answer must say so without being asked."""
    out = []
    for truth in sorted(gt.with_archetype("suspicious_reviews"), key=lambda t: t.business_id)[:PER_KIND["trust"]]:
        out.append(EvalQuestion(
            id=f"trust-{truth.business_id}", kind="trust",
            question=f"Is {truth.name} legit? I'm seeing a lot of 5-star reviews.",
            reference={"business_id": truth.business_id, "verdict": truth.verdict,
                       "problems": list(truth.problems)}))
    return out


def _declining_questions(gt: GroundTruth) -> list[EvalQuestion]:
    """Companies that were good and got worse: the rating still looks fine overall."""
    out = []
    for truth in sorted(gt.with_archetype("declining"), key=lambda t: t.business_id)[:PER_KIND["declining"]]:
        out.append(EvalQuestion(
            id=f"declining-{truth.business_id}", kind="trust",
            question=f"Has {truth.name} got worse? A friend had a bad experience with them recently.",
            reference={"business_id": truth.business_id, "verdict": truth.verdict,
                       "problems": list(truth.problems)}))
    return out


def _recommend_questions(ds: Dataset, gt: GroundTruth) -> list[EvalQuestion]:
    """Open recommendations, in a trade and city where there is both a right and a wrong answer."""
    out = []
    for category in sorted(ds.category_name):
        cities = sorted({city for t in gt.businesses.values() if t.category == category for city in t.cities})
        for city in cities:
            good = gt.by_verdict("good", category, city)
            poor = gt.by_verdict("poor", category, city)
            if len(good) >= 2 and poor:
                out.append(EvalQuestion(
                    id=f"recommend-{category}-{city.lower().replace(' ', '-')}", kind="recommend",
                    question=f"Who should I call for {_trade(ds, category)} work in {city}?",
                    # No list of right answers: whether the companies it names are any good
                    # is looked up in ground truth when the answer is scored, which also covers
                    # a company it names from outside this city.
                    reference={"category": category, "city": city}))
                break
    return out[:PER_KIND["recommend"]]


def _neighborhood_questions(ds: Dataset, gt: GroundTruth) -> list[EvalQuestion]:
    """Local word of mouth: the answer lives in neighbours' posts, not in star ratings."""
    mentions = gt.positive_mentions_by_neighborhood()
    ranked = sorted(mentions.items(), key=lambda kv: (-sum(kv[1].values()), kv[0]))
    out = []
    for (neighborhood, category), counts in ranked:
        recommended = sorted(b for b, n in counts.items() if n >= 2)
        if len(recommended) < 2:
            continue
        out.append(EvalQuestion(
            id=f"neighborhood-{neighborhood.lower().replace(' ', '-')}-{category}", kind="neighborhood",
            question=f"What do people in {neighborhood} say about {_trade(ds, category)} "
                     f"companies? Who do they actually use?",
            reference={"neighborhood": neighborhood, "category": category,
                       "expected": recommended}))
        if len(out) == PER_KIND["neighborhood"]:
            break
    return out


def _expert_questions(ds: Dataset, gt: GroundTruth) -> list[EvalQuestion]:
    """Who to ask for by name: the answer is a person, which only the mentions reveal."""
    praised = gt.praised_experts()
    out = []
    for business_id, expert_ids in sorted(praised.items()):
        truth = gt.businesses[business_id]
        if truth.verdict != "good" or len(expert_ids) != 1:
            continue  # one clear name, at a company worth recommending in the first place
        expert = ds.expert_by_id[expert_ids[0]]
        out.append(EvalQuestion(
            id=f"expert-{business_id}", kind="expert",
            question=f"Is there a specific person I should ask for at {truth.name}?",
            reference={"business_id": business_id, "experts": expert_ids,
                       "expert_names": [f"{expert['goes_by']} {expert['last_name']}"]}))
        if len(out) == PER_KIND["expert"]:
            break
    return out


def _quote_questions(ds: Dataset) -> list[EvalQuestion]:
    """Is this quote fair? The expected verdict is computed from the real quote distribution."""
    by_service_city: dict[tuple[str, str], list[float]] = {}
    for e in ds.estimates:
        by_service_city.setdefault((e["service_id"], e["job_address"]["city"]), []).append(e["total_usd"])

    candidates = sorted((key for key, values in by_service_city.items()
                         if len(values) >= MIN_QUOTES_FOR_A_PRICE_QUESTION),
                        key=lambda key: (-len(by_service_city[key]), key))
    out = []
    # One quote that should read as normal and one that should read as a rip-off, so the metric
    # cannot be won by always saying "that sounds about right".
    used: set[tuple[str, str]] = set()
    for target_share, label in ((0.5, "typical"), (0.93, "very high")):
        for service_id, city in candidates:
            if (service_id, city) in used:
                continue
            values = sorted(by_service_city[(service_id, city)])
            amount = round(values[min(int(len(values) * target_share), len(values) - 1)], -1)
            if _verdict_for(amount, values) != label:
                continue
            used.add((service_id, city))
            service = ds.service_by_id[service_id]
            out.append(EvalQuestion(
                id=f"quote-{service_id}-{label.replace(' ', '-')}", kind="quote",
                question=f"I was quoted ${amount:,.0f} for {service['name'].lower()} in {city}. "
                         f"Is that a fair price?",
                reference={"service_id": service_id, "city": city, "amount": amount,
                           "verdict": label, "percentile": _percentile_of(amount, values)}))
            break
    return out[:PER_KIND["quote"]]


def build_questions(ds: Dataset, gt: GroundTruth) -> list[EvalQuestion]:
    """Every question, in a stable order."""
    return [*_trust_questions(gt), *_declining_questions(gt), *_recommend_questions(ds, gt),
            *_neighborhood_questions(ds, gt), *_expert_questions(ds, gt), *_quote_questions(ds)]
