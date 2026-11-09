"""The hidden labels an evaluation scores against.

This is the only module that loads the ground-truth corpus, and nothing here may be reachable from
the agent, the tools or the three stores: if the assistant could see these labels it would be
marking its own homework. (The one other place that opens a ground-truth file is
entity_resolution.evaluate_mentions, which is handed the path and is only called from scoring.)

Each business gets two independent judgements, and keeping them apart is deliberate:

* `verdict` comes from the archetype the generator assigned ("reliable_veteran", "declining", ...).
  That is the intent behind the data.
* `outcome_score` comes only from what happened on that company's jobs - quality, punctuality,
  communication and how often the final bill beat the quote. That is the realised evidence.

Because neither is derived from the other, they can be compared, which is a check on the labels
themselves rather than an assumption about them (see tests/test_evaluation.py).
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from expert_finder.dataset import Dataset, read_jsonl

GOOD_ARCHETYPES = frozenset({"reliable_veteran", "premium_craftsman", "budget_value"})
POOR_ARCHETYPES = frozenset({"inconsistent", "upsell_heavy", "declining", "suspicious_reviews"})
# "new_and_eager" is deliberately in neither: too few jobs behind it to call either way, and an
# assistant that refuses to recommend every young company is not doing homeowners a favour.

# A bill is "over the quote" past 5%, the same threshold sql_queries.quote_accuracy reports on, so
# the expected answer and what the agent can see agree on what counts as an overrun.
OVER_QUOTE_RATIO = 1.05
OVER_QUOTE_SHARE_FOR_A_PROBLEM = 0.25
SUSPICIOUS_BURST_FLAGS_FOR_A_PROBLEM = 5

# Problems an answer ought to raise, keyed for the code and worded for the LLM judge.
PROBLEM_TEXT = {
    "fake_reviews": "the rating is propped up by reviews from brand-new or untraceable accounts",
    "declining": "the quality of their work has got noticeably worse recently",
    "cost_overruns": "final bills regularly come in well above the written quote",
    "upselling": "they push extra work the customer did not ask for",
    "expired_license": "their trade licence has expired",
    "no_insurance": "they carry no general liability insurance",
    "unresolved_complaints": "they have complaints that were never resolved",
}


@dataclass(frozen=True)
class BusinessTruth:
    business_id: str
    name: str
    category: str
    cities: tuple[str, ...]
    archetype: str
    verdict: str                      # good | mixed | poor, from the archetype
    outcome_score: float              # 0..1, from the jobs alone
    jobs: int
    problems: tuple[str, ...]         # keys of PROBLEM_TEXT


def _verdict(archetype: str) -> str:
    if archetype in GOOD_ARCHETYPES:
        return "good"
    if archetype in POOR_ARCHETYPES:
        return "poor"
    return "mixed"


class GroundTruth:
    """Labels for one dataset, loaded once and queried by the question builder and evaluators."""

    def __init__(self, ds: Dataset, gt_dir: Path):
        self.ds = ds
        self._profiles = {g["business_id"]: g for g in read_jsonl(gt_dir / "business_profiles.jsonl")}
        self._jobs = read_jsonl(gt_dir / "jobs.jsonl")
        self._review_flags = read_jsonl(gt_dir / "review_flags.jsonl")
        self._social_mentions = read_jsonl(gt_dir / "social_mentions.jsonl")
        self._review_experts = read_jsonl(gt_dir / "review_expert_mentions.jsonl")
        self.businesses = self._build_business_truth()

    @classmethod
    def load(cls, ds: Dataset, data_dir: Path) -> GroundTruth:
        return cls(ds, data_dir / "ground_truth")

    # --- per business ---------------------------------------------------------

    def _build_business_truth(self) -> dict[str, BusinessTruth]:
        jobs_by_business: dict[str, list[dict]] = defaultdict(list)
        for job in self._jobs:
            jobs_by_business[job["business_id"]].append(job)
        bursts = Counter(self.ds.review_by_id[f["review_id"]]["business_id"]
                         for f in self._review_flags if f["flag"] == "suspicious_burst")

        out = {}
        for business_id, profile in sorted(self._profiles.items()):
            b = self.ds.business_by_id[business_id]
            jobs = jobs_by_business.get(business_id, [])
            out[business_id] = BusinessTruth(
                business_id=business_id, name=profile["name"],
                category=b["primary_category"], cities=tuple(b["service_area"]["cities"]),
                archetype=profile["archetype"], verdict=_verdict(profile["archetype"]),
                outcome_score=self._outcome_score(jobs), jobs=len(jobs),
                problems=self._problems(b, profile, jobs, bursts[business_id]))
        return out

    @staticmethod
    def _outcome_score(jobs: list[dict]) -> float:
        """How the work actually went, on 0..1.

        Weighted towards whether the job was done right, because that is what a homeowner is
        buying; punctuality, being kept informed, and billing the quoted amount split the rest.
        """
        if not jobs:
            return 0.0
        n = len(jobs)
        shares = {
            "quality": sum(1 for j in jobs if j["outcome"]["quality_ok"]) / n,
            "on_time": sum(1 for j in jobs if j["outcome"]["on_time"]) / n,
            "communicated": sum(1 for j in jobs if j["outcome"]["communicated"]) / n,
            "billed_as_quoted": 1 - GroundTruth._over_quote_share(jobs),
        }
        weights = {"quality": 0.4, "on_time": 0.2, "communicated": 0.2, "billed_as_quoted": 0.2}
        return round(sum(shares[k] * w for k, w in weights.items()), 3)

    @staticmethod
    def _over_quote_share(jobs: list[dict]) -> float:
        quoted = [j for j in jobs if j["quoted_usd"] and j["final_usd"]]
        if not quoted:
            return 0.0
        over = sum(1 for j in quoted if j["final_usd"] > j["quoted_usd"] * OVER_QUOTE_RATIO)
        return over / len(quoted)

    def _problems(self, b: dict, profile: dict, jobs: list[dict], burst_flags: int) -> tuple[str, ...]:
        """What an honest answer about this company should mention, unprompted."""
        problems = []
        if profile["archetype"] == "suspicious_reviews" or burst_flags >= SUSPICIOUS_BURST_FLAGS_FOR_A_PROBLEM:
            problems.append("fake_reviews")
        if profile["decline_after"]:
            problems.append("declining")
        if self._over_quote_share(jobs) > OVER_QUOTE_SHARE_FOR_A_PROBLEM:
            problems.append("cost_overruns")
        if profile["archetype"] == "upsell_heavy":
            problems.append("upselling")
        # The rest are visible in the raw data too; they are listed here because the expected
        # answer is "all of it", whether or not the agent had to infer it.
        if any(lic["status"] == "expired" for lic in b["licenses"]):
            problems.append("expired_license")
        if not b["insurance"]["general_liability"]:
            problems.append("no_insurance")
        if any(r["business_id"] == b["business_id"] and r["complaint_status"] == "unresolved"
               for r in self.ds.reviews):
            problems.append("unresolved_complaints")
        return tuple(problems)

    # --- selections the question builder needs --------------------------------

    def by_verdict(self, verdict: str, category: str | None = None,
                   city: str | None = None) -> list[BusinessTruth]:
        """Businesses with this verdict, in id order so a question set never shuffles."""
        return [t for t in self.businesses.values()
                if t.verdict == verdict
                and (category is None or t.category == category)
                and (city is None or city in t.cities)]

    def with_archetype(self, archetype: str) -> list[BusinessTruth]:
        return [t for t in self.businesses.values() if t.archetype == archetype]

    def positive_mentions_by_neighborhood(self) -> dict[tuple[str, str], Counter]:
        """(neighborhood, category) -> how often each business was recommended in a post."""
        out: dict[tuple[str, str], Counter] = defaultdict(Counter)
        for m in self._social_mentions:
            if m["sentiment"] != "positive":
                continue
            post = self.ds.post_by_id.get(m["post_id"])
            truth = self.businesses.get(m["business_id"])
            if not post or not truth:
                continue
            where = self._neighborhood_of(post)
            if where:
                out[(where, truth.category)][m["business_id"]] += 1
        return out

    def _neighborhood_of(self, post: dict) -> str | None:
        customer = self.ds.customer_by_id.get(post["customer_id"]) if post["customer_id"] else None
        if customer:
            return customer["neighborhood"]
        if post["platform"] == "nextdoor" and post["community"]:
            return post["community"].rpartition(", ")[0] or None
        return None

    def praised_experts(self, min_good: int = 3) -> dict[str, list[str]]:
        """business_id -> experts named in good reviews far more often than in bad ones."""
        good: Counter = Counter()
        poor: Counter = Counter()
        for m in self._review_experts:
            review = self.ds.review_by_id.get(m["review_id"])
            if not review or review["rating"] is None:
                continue
            if review["rating"] >= 4:
                good[m["expert_id"]] += 1
            elif review["rating"] <= 2:
                poor[m["expert_id"]] += 1
        out: dict[str, list[str]] = defaultdict(list)
        for expert_id, count in sorted(good.items()):
            if count >= min_good and count > poor[expert_id] * 2:
                out[self.ds.expert_by_id[expert_id]["business_id"]].append(expert_id)
        return dict(out)
