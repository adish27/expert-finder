"""Link free-text mentions in posts and reviews to businesses and experts.

Rule-based and deliberately conservative: an ambiguous mention is skipped rather than guessed.
The links feed the Neo4j MENTIONS relationships; their accuracy is measured against
ground truth by evaluate_mentions() (the only code here that reads ground_truth/).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from expert_finder.dataset import Dataset
from expert_finder.formatting import short_name

# Words people use for each trade in "<Name> the <trade>" mentions.
TRADE_WORDS = {
    "plumbing": ["plumber"],
    "electrical": ["electrician"],
    "hvac": ["hvac tech", "hvac guy", "ac guy", "ac tech"],
    "roofing": ["roofer"],
    "landscaping": ["landscaper", "yard guy"],
}


@dataclass(frozen=True)
class Mention:
    target_id: str        # business_id or expert_id
    kind: str             # "business" | "expert"
    surface_form: str     # the text that matched
    method: str           # "name" | "handle" | "name_the_trade" | "expert_name"


def _alternation(forms) -> str:
    return "|".join(re.escape(f) for f in sorted(set(forms), key=len, reverse=True))


class EntityResolver:
    def __init__(self, ds: Dataset):
        self.ds = ds
        # Surface form (lowercase) -> businesses it can refer to. Two businesses may share a form
        # ("Hayes Roofing" vs "Hayes Roofing Co."), so values are sets.
        self._form_to_businesses: dict[str, set[str]] = {}
        self._form_method: dict[str, str] = {}
        self._handle_to_business: dict[str, str] = {}
        for b in ds.businesses:
            for form in (b["name"], short_name(b["name"])):
                self._form_to_businesses.setdefault(form.lower(), set()).add(b["business_id"])
                self._form_method[form.lower()] = "name"
            for handle in b["social_accounts"].values():
                if handle.startswith("@"):
                    self._form_to_businesses.setdefault(handle.lower(), set()).add(b["business_id"])
                    self._form_method[handle.lower()] = "handle"
                    self._handle_to_business[handle.lower()] = b["business_id"]
        # One alternation, longest form first: at each position the longest name wins, and
        # finditer never returns overlapping matches.
        self._business_re = re.compile(rf"(?<![\w@])(?:{_alternation(self._form_to_businesses)})(?![\w])",
                                       re.IGNORECASE)

        self._experts_by_business: dict[str, list[dict]] = {}
        for e in ds.experts:
            self._experts_by_business.setdefault(e["business_id"], []).append(e)

        # "<Name> the <trade>" mentions.
        names = {n for e in ds.experts for n in (e["first_name"], e["goes_by"])}
        trades = {w for ws in TRADE_WORDS.values() for w in ws}
        self._name_trade_re = re.compile(rf"\b({_alternation(names)}) the ({_alternation(trades)})\b", re.IGNORECASE)
        self._trade_category = {w: c for c, ws in TRADE_WORDS.items() for w in ws}

        # Service phrases ("gas line work", "new sod"), used to disambiguate between businesses.
        self._service_forms: dict[str, str] = {}
        for svc in ds.services:
            for form in (svc["name"], svc["common_name"], *svc["aliases"]):
                if len(form) > 3:
                    self._service_forms[form.lower()] = svc["service_id"]
        self._service_re = re.compile(rf"\b(?:{_alternation(self._service_forms)})\b", re.IGNORECASE)
        self._offers = {b["business_id"]: {o["service_id"] for o in ds.offerings_of(b["business_id"])}
                        for b in ds.businesses}

    def services_mentioned(self, text: str) -> set[str]:
        return {self._service_forms[m.group(0).lower()] for m in self._service_re.finditer(text)}

    def _disambiguate(self, candidates: set[str], text: str, city: str | None) -> str | None:
        """Narrow candidates by the author's city, then by services named in the text; None if still ambiguous."""
        if len(candidates) > 1 and city:
            candidates = {b for b in candidates if city in self.ds.business_by_id[b]["service_area"]["cities"]}
        if len(candidates) > 1:
            services = self.services_mentioned(text)
            if services:
                candidates = {b for b in candidates if self._offers[b] & services} or candidates
        return next(iter(candidates)) if len(candidates) == 1 else None

    def resolve_businesses(self, text: str, city: str | None = None) -> list[Mention]:
        """Businesses mentioned in text. `city` (where the author lives) helps with ambiguous mentions."""
        found: dict[str, Mention] = {}
        for m in self._business_re.finditer(text):
            form = m.group(0).lower()
            bid = self._disambiguate(self._form_to_businesses[form], text, city)
            if bid and bid not in found:
                found[bid] = Mention(bid, "business", m.group(0), self._form_method[form])
        for m in self._name_trade_re.finditer(text):
            name, category = m.group(1).lower(), self._trade_category[m.group(2).lower()]
            candidates = {
                e["business_id"] for e in self.ds.experts
                if name in (e["first_name"].lower(), e["goes_by"].lower())
                and category in [self.ds.business_by_id[e["business_id"]]["primary_category"],
                                 *self.ds.business_by_id[e["business_id"]]["secondary_categories"]]}
            bid = self._disambiguate(candidates, text, city)
            if bid and bid not in found:
                found[bid] = Mention(bid, "business", m.group(0), "name_the_trade")
        return list(found.values())

    def resolve_experts(self, text: str, business_id: str) -> list[Mention]:
        """Experts of a known business whose first name or nickname appears in the text.

        Two people at one business can share a first name. Reviews and posts describe whoever did
        the work, so a field role wins over office staff; a tie between field roles is left unlinked.
        """
        by_form: dict[str, list[dict]] = {}
        for e in self._experts_by_business.get(business_id, []):
            for form in dict.fromkeys([e["goes_by"], e["first_name"]]):
                if re.search(rf"\b{re.escape(form)}\b", text):
                    by_form.setdefault(form, []).append(e)
                    break
        found = []
        for form, experts in by_form.items():
            if len(experts) > 1:
                experts = [e for e in experts if e["field_role"]]
            if len(experts) == 1:
                found.append(Mention(experts[0]["expert_id"], "expert", form, "expert_name"))
        return found

    def _mask(self, text: str, mentions: list[Mention]) -> str:
        """Blank out matched business names so "Michael's Lawn Care" doesn't count as naming Michael."""
        for m in mentions:
            if m.method in ("name", "handle"):
                text = re.sub(re.escape(m.surface_form), " ", text, flags=re.IGNORECASE)
        return text

    def resolve_review_experts(self, review: dict) -> list[Mention]:
        """Experts named in a review. The business is already known, so only names need matching.

        The owner's response is excluded: owners sign their replies, which says nothing about who
        did the work, and the owner is already linked to the business by WORKS_AT.
        """
        bid = review["business_id"]
        text = f"{review['title'] or ''} {review['text']}"
        name = self.ds.business_by_id[bid]["name"]
        masked = self._mask(text, [Mention(bid, "business", f, "name") for f in (name, short_name(name))])
        return self.resolve_experts(masked, bid)

    def resolve_post(self, post: dict) -> list[Mention]:
        """All business and expert mentions in a social post (text plus any video transcript)."""
        text = " ".join([post["text"], *(m.get("transcript", "") for m in post["media"])])
        cust = self.ds.customer_by_id.get(post["customer_id"]) if post["customer_id"] else None
        city = cust["city"] if cust else None
        if not city and post["platform"] == "nextdoor" and post["community"]:
            city = post["community"].rpartition(", ")[2]
        businesses = self.resolve_businesses(text, city)
        # A business posting on its own account is talking about itself.
        if post["author"]["account_type"] == "business":
            bid = self._handle_to_business.get(post["author"]["handle"].lower())
            if bid and all(b.target_id != bid for b in businesses):
                businesses.append(Mention(bid, "business", post["author"]["handle"], "author"))
        masked = self._mask(text, businesses)
        experts = [x for b in businesses for x in self.resolve_experts(masked, b.target_id)]
        return businesses + experts


def evaluate_mentions(ds: Dataset, resolver: EntityResolver, gold_path: Path) -> dict:
    """Precision/recall of post -> business and post -> expert links against ground truth."""
    gold_biz, gold_exp = set(), set()
    for line in gold_path.read_text().splitlines():
        g = json.loads(line)
        gold_biz.add((g["post_id"], g["business_id"]))
        if g["expert_id"]:
            gold_exp.add((g["post_id"], g["expert_id"]))
    pred_biz, pred_exp = set(), set()
    by_method: dict[str, list[bool]] = {}
    for p in ds.posts:
        for m in resolver.resolve_post(p):
            pair = (p["post_id"], m.target_id)
            (pred_biz if m.kind == "business" else pred_exp).add(pair)
            by_method.setdefault(m.method, []).append(pair in (gold_biz if m.kind == "business" else gold_exp))

    def score(pred: set, gold: set) -> dict:
        tp = len(pred & gold)
        return {"predicted": len(pred), "gold": len(gold), "correct": tp,
                "precision": round(tp / len(pred), 3) if pred else None,
                "recall": round(tp / len(gold), 3) if gold else None}

    return {"business_mentions": score(pred_biz, gold_biz), "expert_mentions": score(pred_exp, gold_exp),
            "precision_by_method": {k: round(sum(v) / len(v), 3) for k, v in by_method.items()}}


def evaluate_review_mentions(ds: Dataset, resolver: EntityResolver, gold_path: Path) -> dict:
    """Precision/recall of review -> expert links against ground truth."""
    gold = {(json.loads(line)["review_id"], json.loads(line)["expert_id"]) for line in gold_path.read_text().splitlines()}
    pred = {(r["review_id"], m.target_id) for r in ds.reviews for m in resolver.resolve_review_experts(r)}
    tp = len(pred & gold)
    return {"predicted": len(pred), "gold": len(gold), "correct": tp,
            "precision": round(tp / len(pred), 3) if pred else None,
            "recall": round(tp / len(gold), 3) if gold else None}
