"""Generate the synthetic home-services dataset into data/home_services/.

Deterministic (fixed seed). Every business has a hidden behavioral profile (archetype +
traits) that drives its pricing, job outcomes, review ratings and text, owner responses
and social chatter, so all sources tell a consistent story. The profiles are written to
ground_truth/ for evaluation and are never referenced by the raw files.

Usage: uv run python scripts/generate_home_services_data.py
"""

from __future__ import annotations

import json
import math
import random
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).parent))
import home_services_catalog as cat  # noqa: E402

SEED = 20261025
TODAY = date(2026, 10, 25)
START = date(2021, 1, 1)
TZ = ZoneInfo("America/Chicago")
OUT = Path(__file__).resolve().parents[1] / "data" / "home_services"
CATEGORY_IDS = ["plumbing", "electrical", "hvac", "roofing", "landscaping"]
BUSINESSES_PER_CATEGORY = 20
N_CUSTOMERS = 1400

rng = random.Random(SEED)

# --- Archetypes (hidden behavioral profiles) ---------------------------------

ARCHETYPES = {
    "reliable_veteran": dict(n=5, quality=.93, punctuality=.88, communication=.88, upsell=.08, tiers=["mid", "premium"],
                             sizes=["established", "regional"], founded=(1992, 2014), reviews=(18, 32)),
    "premium_craftsman": dict(n=2, quality=.97, punctuality=.86, communication=.93, upsell=.10, tiers=["premium"],
                              sizes=["small_team", "established"], founded=(2000, 2016), reviews=(12, 22)),
    "budget_value": dict(n=4, quality=.75, punctuality=.62, communication=.62, upsell=.10, tiers=["budget"],
                         sizes=["solo_operator", "small_team"], founded=(2010, 2021), reviews=(8, 18)),
    "new_and_eager": dict(n=2, quality=.87, punctuality=.92, communication=.92, upsell=.04, tiers=["budget", "mid"],
                          sizes=["solo_operator", "small_team"], founded=(2023, 2025), reviews=(3, 8)),
    "inconsistent": dict(n=3, quality=.60, punctuality=.50, communication=.45, upsell=.30, tiers=["mid"],
                         sizes=["small_team", "established"], founded=(2008, 2020), reviews=(12, 24)),
    "declining": dict(n=2, quality=.92, punctuality=.86, communication=.86, upsell=.12, tiers=["mid"],
                      sizes=["established", "regional"], founded=(2004, 2014), reviews=(22, 32),
                      decline_after=date(2025, 5, 1), decline=dict(quality=.45, punctuality=.40, communication=.35)),
    "upsell_heavy": dict(n=1, quality=.80, punctuality=.82, communication=.55, upsell=.80, tiers=["premium"],
                         sizes=["regional"], founded=(2005, 2015), reviews=(20, 30)),
    "suspicious_reviews": dict(n=1, quality=.50, punctuality=.45, communication=.40, upsell=.35, tiers=["mid"],
                               sizes=["small_team"], founded=(2018, 2022), reviews=(6, 10), fake_burst=(14, 20)),
}
TIER_MULT = {"budget": 0.82, "mid": 1.0, "premium": 1.28}
SIZE_EMPLOYEES = {"solo_operator": (1, 1), "small_team": (2, 6), "established": (7, 25), "regional": (26, 85)}
SIZE_EXPERTS = {"solo_operator": (1, 1), "small_team": (2, 3), "established": (3, 4), "regional": (4, 6)}
BASE_HOURLY = {"plumbing": 125, "electrical": 120, "hvac": 125, "roofing": 85, "landscaping": 65}
SERVICE_CALL = {"plumbing": (79, 129), "electrical": (79, 119), "hvac": (79, 149), "roofing": (0, 0), "landscaping": (0, 0)}
SECONDARY = {"plumbing": ["hvac"], "hvac": ["plumbing", "electrical"], "electrical": ["hvac"], "roofing": [], "landscaping": []}
TRADE = {c["category_id"]: c["trade_noun"] for c in cat.CATEGORIES}
SYMPTOM = {"plumbing": ["leak", "clog", "drip"], "electrical": ["breaker tripping", "flickering"],
           "hvac": ["warm air", "same noise"], "roofing": ["leak", "drip in the ceiling"],
           "landscaping": ["dead patches", "broken sprinkler"]}
UPSELL_ITEM = {"plumbing": ["whole-house water treatment system", "full repipe", "new water heater"],
               "electrical": ["full panel replacement", "whole-home surge package", "rewire of the whole house"],
               "hvac": ["brand new system", "UV air purifier", "premium maintenance membership"],
               "roofing": ["full roof replacement", "upgraded designer shingles", "new gutters"],
               "landscaping": ["monthly maintenance contract", "full irrigation overhaul", "premium sod"]}
CAT_HASHTAG = {"plumbing": "plumbing", "electrical": "electrician", "hvac": "hvac", "roofing": "roofing",
               "landscaping": "landscaping"}


# --- Helpers -----------------------------------------------------------------


def money(x: float) -> float:
    return round(x + 1e-9, 2)


def round_price(x: float) -> float:
    if x >= 1000:
        return float(round(x / 50) * 50)
    if x >= 100:
        return float(round(x / 5) * 5)
    if x >= 10:
        return float(round(x))
    return round(x, 2)


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def ts(d: date, hour_range=(7, 21)) -> str:
    dt = datetime(d.year, d.month, d.day, rng.randint(*hour_range), rng.randint(0, 59), tzinfo=TZ)
    return dt.isoformat()


def rand_date(lo: date, hi: date) -> date:
    if hi <= lo:
        return lo
    return lo + timedelta(days=rng.randint(0, (hi - lo).days))


def seasonal_date(lo: date, hi: date, season: dict[int, float] | None) -> date:
    """Sample a date in [lo, hi] weighted by month seasonality."""
    if not season:
        return rand_date(lo, hi)
    for _ in range(50):
        d = rand_date(lo, hi)
        if rng.random() < season.get(d.month, 0.7) / 2.6:
            return d
    return rand_date(lo, hi)


def wchoice(items, weights):
    return rng.choices(items, weights=weights, k=1)[0]


def slugify(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", s.lower().replace("&", "and"))


def cap(s: str) -> str:
    return s[:1].upper() + s[1:]


THIRD_PERSON = [(r"\bwe're\b", "they're"), (r"\bWe're\b", "They're"), (r"\bwe\b", "they"), (r"\bWe\b", "They"),
                (r"\bour\b", "their"), (r"\bOur\b", "Their"), (r"\bus\b", "them")]


def third_person(text: str) -> str:
    """Customer phrasing ("our AC died") -> phrasing a business would use ("their AC died")."""
    for pat, rep in THIRD_PERSON:
        text = re.sub(pat, rep, text)
    return text


def problem_and_work(svc: cat.Service) -> tuple[str, str]:
    """A problem and the matching fix (lists are written index-aligned)."""
    i = rng.randrange(len(svc.problems))
    return svc.problems[i], svc.work[min(i, len(svc.work) - 1)]


COMPANY_NOUN = {"plumbing": "plumbing", "electrical": "electrical", "hvac": "heating and air",
                "roofing": "roofing", "landscaping": "landscaping"}


def months_between(a: date, b: date) -> int:
    return (b.year - a.year) * 12 + b.month - a.month


def ago_phrase(job: date, now: date) -> str:
    m = months_between(job, now)
    if m <= 0:
        return "a couple of weeks ago"
    if m == 1:
        return "last month"
    if m < 12:
        return f"about {m} months ago"
    if m < 24:
        return "last year"
    return f"a few years back ({job.year})"


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n")


# --- People ------------------------------------------------------------------

_used_names: set[tuple[str, str]] = set()


def new_person_name() -> tuple[str, str]:
    while True:
        n = (rng.choice(cat.FIRST_NAMES), rng.choice(cat.LAST_NAMES))
        if n not in _used_names:
            _used_names.add(n)
            return n


def goes_by(first: str) -> str:
    nick = cat.NICKNAMES.get(first)
    return nick if nick and rng.random() < 0.7 else first


# --- Geography ---------------------------------------------------------------

CITY_WEIGHTS = {a["city"]: a["weight"] for a in cat.SERVICE_AREAS}
NEIGHBORHOODS = {a["city"]: a["neighborhoods"] for a in cat.SERVICE_AREAS}
STREETS = ["Burnet Rd", "Research Blvd", "S Congress Ave", "N Lamar Blvd", "Commerce Dr", "Industrial Blvd",
           "Ranch Rd 620", "W Anderson Ln", "E Whitestone Blvd", "Sam Bass Rd", "Main St", "Ben White Blvd",
           "Pecan St", "FM 1626", "Hwy 290 W", "Williams Dr", "Bell Blvd", "Center St", "Kenney Fort Blvd"]


def pick_city() -> str:
    return wchoice(list(CITY_WEIGHTS), list(CITY_WEIGHTS.values()))


def home_desc(home: dict, nbhd: str) -> str:
    y = home["year_built"]
    if home["type"] == "condo":
        kind = "condo"
    elif home["type"] == "townhome":
        kind = "townhome"
    elif y < 1970:
        kind = rng.choice(["bungalow", "ranch house", "old house"])
    else:
        kind = rng.choice(["house", "two-story" if home["stories"] == 2 else "one-story", "home"])
    era = f"{(y // 10) * 10}s" if y < 2000 else str(y)
    return f"{era} {kind} in {nbhd}"


# --- Data containers ---------------------------------------------------------


@dataclass
class Biz:
    record: dict
    archetype: str
    traits: dict
    tier: str
    mult: float
    category: str
    owner_name: tuple[str, str]
    experts: list[dict] = field(default_factory=list)
    offerings: dict[str, dict] = field(default_factory=dict)  # service_id -> offering

    @property
    def id(self) -> str:
        return self.record["business_id"]

    @property
    def short(self) -> str:
        return re.sub(r"\s+(LLC|Inc\.?|Co\.)$", "", self.record["name"])

    @property
    def owner(self) -> dict:
        return self.experts[0]

    def traits_at(self, d: date) -> dict:
        a = ARCHETYPES[self.archetype]
        if "decline_after" in a and d >= a["decline_after"]:
            return {**self.traits, **a["decline"]}
        return self.traits


# --- Businesses --------------------------------------------------------------

_used_biz_names: set[str] = set()


def make_business(idx: int, category: str, archetype: str) -> Biz:
    a = ARCHETYPES[archetype]
    tier = rng.choice(a["tiers"])
    size = rng.choice(a["sizes"])
    first, last = new_person_name()
    while True:
        pattern = rng.choice(cat.NAME_PATTERNS[category])
        name = pattern.format(last=last, first=first, place=rng.choice(cat.PLACE_WORDS))
        if name not in _used_biz_names:
            _used_biz_names.add(name)
            break
    founded = rng.randint(*a["founded"])
    home_city = pick_city()
    nearby = cat.NEARBY[home_city]
    n_extra = {"solo_operator": (1, 2), "small_team": (2, 4), "established": (3, 6), "regional": (5, 7)}[size]
    cities = [home_city] + rng.sample(nearby, k=min(len(nearby), rng.randint(*n_extra)))
    slug = slugify(re.sub(r"\s+(LLC|Inc\.?|Co\.)$", "", name))
    traits = {k: round(clamp(a[k] + rng.gauss(0, 0.04), 0.05, 0.99), 3)
              for k in ("quality", "punctuality", "communication", "upsell")}
    mult = TIER_MULT[tier] * clamp(rng.gauss(1, 0.05), 0.9, 1.1)
    secondary = [c for c in SECONDARY[category] if rng.random() < 0.18][:1]
    emp = rng.randint(*SIZE_EMPLOYEES[size])

    licenses = []
    for lic_name, issuer, prefix in cat.LICENSES[category]:
        if category == "landscaping" and rng.random() < 0.45:
            continue  # not every landscaper is a licensed irrigator
        licenses.append({"type": lic_name, "issuer": issuer, "number": f"{prefix}-SYN{rng.randint(10000, 99999)}",
                         "status": "active", "expires": f"{rng.randint(2026, 2028)}-{rng.randint(1, 12):02d}-28"})
        if category != "electrical":
            break
    if archetype == "suspicious_reviews" and licenses:
        licenses[0]["status"] = "expired"
        licenses[0]["expires"] = "2025-08-31"

    big_ticket = category in ("hvac", "roofing", "landscaping")
    emergency = category in ("plumbing", "hvac", "roofing", "electrical") and (size != "solo_operator" or rng.random() < 0.3)
    hourly = round_price(BASE_HOURLY[category] * mult)
    call_lo, call_hi = SERVICE_CALL[category]
    service_call_fee = round_price(rng.uniform(call_lo, call_hi) * (0.9 if tier == "budget" else 1.0)) if call_hi else 0.0
    labor_warranty = {"budget": rng.choice([3, 6]), "mid": 12, "premium": rng.choice([24, 36, 60])}[tier]
    if category == "roofing":
        labor_warranty = {"budget": 24, "mid": rng.choice([60, 120]), "premium": rng.choice([120, 300])}[tier]

    languages = ["English"] + (["Spanish"] if rng.random() < 0.55 else []) + (
        [rng.choice(cat.LANGUAGES[2:])] if rng.random() < 0.1 else [])
    payments = ["credit card", "debit card"] + [p for p in ("check", "ACH transfer", "Zelle", "cash", "Venmo")
                                                if rng.random() < 0.5]
    financing = (big_ticket or category == "electrical") and size != "solo_operator" and rng.random() < 0.7
    if financing:
        payments.append("financing")
    socials = {}
    if rng.random() < 0.85:
        socials["facebook"] = f"facebook.com/{slug}"
    if rng.random() < (0.75 if category in ("landscaping", "roofing") else 0.5):
        socials["instagram"] = f"@{slug}"
    if rng.random() < (0.5 if category in ("landscaping", "roofing", "hvac") else 0.3) or archetype == "new_and_eager":
        socials["tiktok"] = f"@{slug[:20]}"
    if rng.random() < 0.45:
        socials["nextdoor"] = f"nextdoor.com/pages/{slug}"

    response_hours = {"new_and_eager": (1, 3), "premium_craftsman": (6, 24), "inconsistent": (8, 72),
                      "suspicious_reviews": (12, 96), "budget_value": (2, 24)}.get(archetype, (1, 8))
    schedule_days = (7, 21) if archetype == "premium_craftsman" else (1, 3) if archetype == "new_and_eager" else (1, 6)
    if big_ticket:
        schedule_days = (schedule_days[0] + 3, schedule_days[1] + 14)

    size_phrase = {"solo_operator": "owner-operated", "small_team": "small, family-run",
                   "established": "established", "regional": "multi-crew"}[size]
    city_phrase = ", ".join(cities[:-1]) + f" and {cities[-1]}" if len(cities) > 1 else cities[0]
    article = "an" if size_phrase[0] in "aeiou" else "a"
    description = (f"{name} is {article} {size_phrase} {COMPANY_NOUN[category]} company founded in {founded}, "
                   f"serving {city_phrase}.")

    record = {
        "business_id": f"biz_{idx:04d}",
        "name": name,
        "legal_name": f"{re.sub(r' Co\.$', '', name)} {'LLC' if rng.random() < 0.7 else 'Inc.'}",
        "primary_category": category,
        "secondary_categories": secondary,
        "description": description,
        "founded_year": founded,
        "business_size": size,
        "employee_count": emp,
        "owner_expert_id": None,  # filled once experts exist
        "contact": {"phone": f"({rng.choice(['512', '737'])}) 555-01{idx - 1:02d}",
                    "email": f"office@{slug}.example", "website": f"https://www.{slug}.example"},
        "address": {"street": f"{rng.randint(100, 19999)} {rng.choice(STREETS)}", "city": home_city,
                    "state": "TX", "zip": rng.choice(NEIGHBORHOODS[home_city])["zip"]},
        "service_area": {"cities": cities, "radius_miles": {"solo_operator": 15, "small_team": 20,
                                                             "established": 30, "regional": 45}[size]},
        "hours": {"mon_fri": rng.choice(["07:00-17:00", "07:00-18:00", "08:00-17:00", "07:30-17:30"]),
                  "sat": rng.choice([None, "08:00-12:00", "08:00-14:00"]) if size != "regional" else "08:00-16:00",
                  "sun": None},
        "emergency_service": {"available": emergency, "hours": "24/7" if emergency else None,
                              "after_hours_surcharge_pct": rng.choice([25, 50, 75, 100]) if emergency else None},
        "licenses": licenses,
        "insurance": {"general_liability": archetype != "suspicious_reviews" or rng.random() < 0.5,
                      "liability_coverage_usd": rng.choice([500000, 1000000, 2000000]),
                      "workers_comp": size in ("established", "regional"), "bonded": rng.random() < 0.55},
        "certifications": rng.sample(cat.CERTIFICATIONS[category], k=rng.randint(0, 2 if size == "solo_operator" else 3)),
        "brands": rng.sample(cat.BRANDS[category], k=rng.randint(2, 4)),
        "languages": languages,
        "payment_methods": payments,
        "financing_available": financing,
        "warranty": {"workmanship_months": labor_warranty,
                     "notes": "Manufacturer warranties on parts and equipment are passed through to the customer."},
        "pricing_policy": {
            "hourly_rate_usd": hourly,
            "service_call_fee_usd": service_call_fee,
            "service_call_fee_waived_with_repair": bool(service_call_fee) and rng.random() < 0.6,
            "free_estimates": category in ("roofing", "landscaping") or rng.random() < 0.35,
            "minimum_charge_usd": round_price(hourly * rng.choice([1, 1.5, 2])),
            "discounts": rng.sample(["military", "senior", "first responder", "teacher", "maintenance plan member",
                                     "repeat customer"], k=rng.randint(0, 3)),
        },
        "response_time": {"typical_reply_hours": rng.randint(*response_hours),
                          "typical_days_to_schedule": rng.randint(*schedule_days)},
        "social_accounts": socials,
        "listed_on": [],  # filled from reviews
    }
    return Biz(record=record, archetype=archetype, traits=traits, tier=tier, mult=mult, category=category,
               owner_name=(first, last))


def make_experts(b: Biz, next_id) -> None:
    size = b.record["business_size"]
    n = rng.randint(*SIZE_EXPERTS[size])
    category = b.category
    roles = cat.EXPERT_ROLES[category]
    founded = b.record["founded_year"]
    # The owner's name is the one used in the business name (when the pattern uses it).
    owner_first, owner_last = b.owner_name
    plan = ["owner"] + rng.choices(["senior", "mid", "mid", "junior", "office"], k=n - 1)
    for i, level in enumerate(plan):
        first, last = (owner_first, owner_last) if i == 0 else new_person_name()
        if i == 0:
            years = TODAY.year - founded + rng.randint(4, 12)
            role = "Owner" if size in ("solo_operator", "small_team") else rng.choice(["Owner", "Owner & General Manager", "Founder"])
            lic_level = "senior"
        else:
            years = {"senior": rng.randint(10, 28), "mid": rng.randint(4, 12), "junior": rng.randint(0, 3),
                     "office": rng.randint(3, 15)}[level]
            role = rng.choice(roles[level])
            lic_level = level
        lic = cat.EXPERT_LICENSES[category].get(lic_level) if lic_level != "office" else None
        licenses = []
        if lic and (i > 0 or category not in ("roofing",)):
            licenses.append({"type": lic[0], "issuer": lic[1], "number": f"{lic[2]}-SYN{rng.randint(10000, 99999)}"})
        certs = rng.sample(cat.CERTIFICATIONS[category], k=rng.randint(0, 2)) if level in ("owner", "senior", "mid") else []
        langs = ["English"] + (["Spanish"] if rng.random() < 0.45 else [])
        b.experts.append({
            "expert_id": next_id(), "business_id": b.id, "first_name": first, "last_name": last,
            "goes_by": goes_by(first), "role": role, "is_owner": i == 0, "field_role": level != "office",
            "years_experience": years, "licenses": licenses, "certifications": certs, "languages": langs,
            "specialties": [], "bio": "",
        })
    b.record["owner_expert_id"] = b.experts[0]["expert_id"]


def make_offerings(b: Biz, next_id) -> None:
    def offer(svc: cat.Service, secondary: bool) -> None:
        mult = b.mult * clamp(rng.gauss(1, 0.06), 0.85, 1.15)
        low = round_price(svc.market_low * mult)
        high = round_price(svc.market_high * mult)
        typical = round_price(low + (high - low) * rng.uniform(0.35, 0.6))
        includes, excludes = [], []
        if svc.permit:
            (includes if rng.random() < 0.7 else excludes).append("permit and city inspection")
        if any(w in svc.name.lower() for w in ("replacement", "removal", "installation", "makeover", "repipe")):
            includes.append("haul-away and disposal of old materials")
        if svc.pricing_model in ("per_sqft", "per_square", "per_linear_ft"):
            excludes.append("structural repairs discovered during work (priced separately)")
        if svc.category == "hvac" and "replacement" in svc.name.lower():
            includes.append("load calculation (Manual J)" if b.tier != "budget" else "standard sizing")
        notes = None
        if svc.market_low == 0 and low == 0:
            notes = "Free inspection; written report with photos."
        elif svc.pricing_model == "hourly":
            notes = f"Billed in half-hour increments after the first hour; minimum {b.record['pricing_policy']['minimum_charge_usd']:.0f} USD."
        b.offerings[svc.service_id] = {
            "offering_id": next_id(), "business_id": b.id, "service_id": svc.service_id, "category": svc.category,
            "is_secondary_category": secondary, "pricing_model": svc.pricing_model, "unit": svc.unit,
            "price_low_usd": low, "price_high_usd": high, "typical_price_usd": typical,
            "price_is_starting_at": svc.pricing_model in ("flat",) and rng.random() < 0.2,
            "includes": includes, "excludes": excludes,
            "permit_required": svc.permit,
            "lead_time_days": max(0, int(rng.gauss(2 if svc.market_high < 2000 else 10, 3))),
            "warranty_months": b.record["warranty"]["workmanship_months"] if svc.market_high >= 500 else min(
                b.record["warranty"]["workmanship_months"], 12),
            "available_for_emergency": b.record["emergency_service"]["available"] and svc.market_high < 2500,
            "notes": notes,
        }

    for svc in [s for s in cat.SERVICES if s.category == b.category]:
        p = 0.92 if svc.core else (0.62 if b.tier == "premium" or b.record["business_size"] == "regional" else 0.38)
        if b.record["business_size"] == "solo_operator" and svc.market_high > 9000:
            p *= 0.3
        if rng.random() < p:
            offer(svc, False)
    for sec in b.record["secondary_categories"]:
        pool = [s for s in cat.SERVICES if s.category == sec and s.core]
        for svc in rng.sample(pool, k=min(len(pool), rng.randint(2, 4))):
            offer(svc, True)
    # Specialties: what each field expert is known for.
    field_services = list(b.offerings)
    for e in b.experts:
        if e["field_role"]:
            e["specialties"] = rng.sample(field_services, k=min(len(field_services), rng.randint(2, 4)))
        names = [cat.SERVICE_BY_ID[s].name.lower() for s in e["specialties"]]
        spec = f" Specializes in {', '.join(names[:-1])} and {names[-1]}." if len(names) > 1 else ""
        lic = f" Holds a {e['licenses'][0]['type']} license." if e["licenses"] else ""
        e["bio"] = (f"{e['goes_by']} {e['last_name']} is the {e['role'].lower()} at {b.record['name']} with "
                    f"{e['years_experience']} years in the trade.{lic}{spec}")
        if e["years_experience"] == 0:
            e["bio"] = f"{e['goes_by']} {e['last_name']} recently joined {b.record['name']} as a {e['role'].lower()}.{spec}"


# --- Customers ---------------------------------------------------------------


def make_customers() -> list[dict]:
    rows = []
    for i in range(1, N_CUSTOMERS + 1):
        city = pick_city()
        n = rng.choice(NEIGHBORHOODS[city])
        first, last = rng.choice(cat.FIRST_NAMES), rng.choice(cat.LAST_NAMES)
        year = rng.randint(*n["home_era"])
        htype = wchoice(["single_family", "townhome", "condo"], [0.84, 0.1, 0.06] if year > 1990 else [0.95, 0.02, 0.03])
        stories = 1 if htype == "condo" else wchoice([1, 2], [0.6, 0.4] if year < 1990 else [0.4, 0.6])
        sqft = int(rng.gauss(2200 if htype == "single_family" else 1300, 450))
        handle_word = rng.choice(["atx", "texas", "hillcountry", "bbq", "taco", "bat", "longhorn", "porch", "oak"])
        rows.append({
            "customer_id": f"cust_{i:05d}", "first_name": first, "last_name": last, "city": city,
            "neighborhood": n["name"], "zip": n["zip"],
            "home": {"type": htype, "year_built": year, "sq_ft": max(650, sqft), "stories": stories,
                     "has_pool": htype == "single_family" and rng.random() < 0.18},
            "platform_handles": {
                "google": rng.choice([f"{first} {last}", f"{first} {last[0]}."]),
                "yelp": f"{first} {last[0]}.", "angi": f"{first} {last[0]}.", "thumbtack": f"{first} {last[0]}.",
                "houzz": f"{first.lower()}{last.lower()[:3]}{rng.randint(1, 99)}",
                "bbb": f"{first} {last[0]}.",
                "nextdoor": f"{first} {last}", "facebook": f"{first} {last}",
                "reddit": f"u/{handle_word}_{first.lower()}{rng.randint(10, 999)}",
                "tiktok": f"@{first.lower()}.{last.lower()}{rng.randint(1, 99)}" if rng.random() < 0.3 else None,
                "instagram": f"@{first.lower()}_{handle_word}{rng.randint(1, 99)}" if rng.random() < 0.4 else None,
            },
        })
    return rows


# --- Jobs, estimates ---------------------------------------------------------


@dataclass
class Job:
    job_id: str
    biz: Biz
    customer: dict
    service: cat.Service
    offering: dict
    expert: dict | None
    date: date
    quantity: float
    quote: float
    final: float
    outcome: dict
    estimate_id: str | None = None


def job_outcome(b: Biz, d: date) -> dict:
    t = b.traits_at(d)
    quality_ok = rng.random() < t["quality"]
    on_time = rng.random() < t["punctuality"]
    communicated = rng.random() < t["communication"]
    upsold = rng.random() < t["upsell"]
    overrun = 0.0
    if upsold and not communicated:
        overrun = rng.uniform(0.15, 0.45)
    elif not communicated and rng.random() < 0.4:
        overrun = rng.uniform(0.08, 0.25)
    score = (2.2 * quality_ok + 0.9 * on_time + 0.9 * communicated - 0.7 * (upsold and not communicated)
             - 0.6 * (overrun > 0.1) + (0.3 if b.tier == "budget" else 0) + rng.gauss(0, 0.45))
    rating = 5 if score >= 3.35 else 4 if score >= 2.6 else 3 if score >= 1.9 else 2 if score >= 1.0 else 1
    return {"quality_ok": quality_ok, "on_time": on_time, "communicated": communicated, "upsold": upsold,
            "overrun_pct": round(overrun, 3), "rating": rating, "clean": rng.random() < (0.5 + 0.45 * t["quality"])}


def job_quantity(svc: cat.Service) -> float:
    lo, hi = svc.qty
    if lo == hi:
        return lo
    q = rng.uniform(lo, hi)
    return float(round(q)) if hi > 5 else float(max(1, round(q)))


def line_items(svc: cat.Service, total: float, qty: float, b: Biz) -> tuple[list[dict], float]:
    """Split a target total into realistic line items. Returns (items, sales_tax)."""
    items = []
    remaining = total
    if svc.permit and rng.random() < 0.85:
        permit = round_price(rng.uniform(75, 450))
        items.append({"description": "City permit and inspection", "category": "permit", "quantity": 1,
                      "unit": "each", "unit_price_usd": permit, "total_usd": permit})
        remaining -= permit
    if any(w in svc.name.lower() for w in ("replacement", "removal", "repipe", "makeover")):
        disp = round_price(rng.uniform(75, 450) * (3 if "roof" in svc.service_id else 1))
        items.append({"description": "Haul-away and disposal", "category": "disposal", "quantity": 1,
                      "unit": "each", "unit_price_usd": disp, "total_usd": disp})
        remaining -= disp
    install = svc.market_high >= 900
    mat_share = rng.uniform(0.35, 0.6) if install else rng.uniform(0.0, 0.2)
    materials = money(remaining * mat_share)
    labor = money(remaining - materials)
    if svc.pricing_model in ("per_unit", "per_sqft", "per_linear_ft", "per_square", "per_zone", "per_tree", "hourly"):
        items.insert(0, {"description": f"{svc.name} ({svc.unit} rate, labor)", "category": "labor",
                         "quantity": qty, "unit": svc.unit, "unit_price_usd": money(labor / qty), "total_usd": labor})
    else:
        items.insert(0, {"description": f"{svc.name} - labor", "category": "labor", "quantity": 1, "unit": "job",
                         "unit_price_usd": labor, "total_usd": labor})
    if materials > 0:
        brand = rng.choice(b.record["brands"])
        items.insert(1, {"description": f"Materials and parts ({brand})" if install else "Parts and supplies",
                         "category": "materials", "quantity": 1, "unit": "lot", "unit_price_usd": materials,
                         "total_usd": materials})
    if rng.random() < 0.15 and b.record["pricing_policy"]["discounts"]:
        disc = -money(total * rng.choice([0.05, 0.1]))
        items.append({"description": f"{cap(rng.choice(b.record['pricing_policy']['discounts']))} discount",
                      "category": "discount", "quantity": 1, "unit": "each", "unit_price_usd": disc, "total_usd": disc})
    tax = money(materials * 0.0825)
    return items, tax


def make_estimate(est_id: str, b: Biz, customer: dict, svc: cat.Service, qty: float, total: float,
                  issued: date, expert: dict | None, status: str, decline_reason: str | None = None,
                  project_id: str | None = None, job: dict | None = None) -> dict:
    items, tax = line_items(svc, total, qty, b)
    subtotal = money(sum(i["total_usd"] for i in items))
    problem, work = problem_and_work(svc)
    requested = issued - timedelta(days=rng.randint(0, 4))
    visited = issued - timedelta(days=rng.randint(0, min(2, (issued - requested).days))) if svc.market_high > 600 else None
    return {
        "estimate_id": est_id, "customer_project_id": project_id, "business_id": b.id,
        "expert_id": expert["expert_id"] if expert else None, "customer_id": customer["customer_id"],
        "service_id": svc.service_id, "category": svc.category,
        "request_channel": wchoice(["phone", "website form", "angi", "thumbtack", "referral", "nextdoor", "google"],
                                   [0.3, 0.2, 0.12, 0.08, 0.15, 0.07, 0.08]),
        "requested_at": ts(requested), "site_visit_at": ts(visited, (8, 17)) if visited else None,
        "issued_at": ts(issued), "valid_until": (issued + timedelta(days=30)).isoformat(),
        "job_address": {"city": customer["city"], "neighborhood": customer["neighborhood"], "zip": customer["zip"]},
        "job_description": f"Customer reports {third_person(problem)}. Scope: {third_person(work)}.",
        "quantity": qty, "unit": svc.unit,
        "line_items": items, "subtotal_usd": subtotal, "sales_tax_usd": tax, "total_usd": money(subtotal + tax),
        "financing_offered": b.record["financing_available"] and total > 2500,
        "status": status, "decline_reason": decline_reason, "job": job,
    }


# --- Review text -------------------------------------------------------------


def fmt_usd(x: float) -> str:
    return f"${x:,.0f}"


def tech_ref(job: Job) -> str | None:
    return job.expert["goes_by"] if job.expert else None


def review_sentences(job: Job, platform: str) -> dict[str, list[str]]:
    """Candidate sentences per component, conditioned on the job outcome."""
    o, b, svc = job.outcome, job.biz, job.service
    tech = tech_ref(job)
    home = home_desc(job.customer["home"], job.customer["neighborhood"])
    problem, work = problem_and_work(svc)
    name = rng.choice([b.short, b.short, b.record["name"]])
    good = o["rating"] >= 4
    s: dict[str, list[str]] = defaultdict(list)

    # Context hooks tied to real seasonal events.
    hook = ""
    if svc.category == "plumbing" and job.date.year == 2021 and job.date.month in (2, 3):
        hook = "During the February freeze "
    elif svc.category == "hvac" and job.date.month in (7, 8):
        hook = rng.choice(["In the middle of a July heat wave, ", "With triple-digit heat outside, ", ""])
    elif svc.service_id in ("roof_repair", "roof_shingle_replacement", "roof_storm_claim", "roof_emergency_tarp",
                            "roof_inspection") and job.date.month in (3, 4, 5, 6) and rng.random() < 0.6:
        hook = "After the spring hailstorm, "
    opener = f"{hook}{problem}" if hook else cap(problem)
    where = "" if re.search(r"\b(house|home|condo)\b", problem) else f" in our {home}"
    s["open"] = [f"{opener}{where}, so we called {name}.",
                 f"{opener}. {'A neighbor recommended' if rng.random() < 0.5 else 'We found'} {name}.",
                 f"We hired {name} because {problem}."]
    if good and rng.random() < 0.2:
        s["open"].append(f"Second time using {name}. This time {problem}.")

    s["response"] = ([f"They called back within the hour and had someone out {rng.choice(['the same day', 'the next morning', 'that afternoon'])}.",
                      "Scheduling was easy and they gave us a two-hour window."]
                     if o["communicated"] else
                     ["It took three calls and two days just to get a callback.",
                      "The office never confirmed the appointment, we had to chase them."])
    who = tech or "The tech"
    if svc.category in ("roofing", "landscaping"):
        who = f"{tech} and the crew" if tech else "The crew"
    s["arrival"] = ([f"{who} showed up right on time.", f"{who} texted when they were on the way and arrived early."]
                    if o["on_time"] else
                    [f"{who} showed up {rng.choice(['two', 'three', 'four'])} hours late without a call.",
                     "Nobody showed up the first day and we had to reschedule.",
                     f"{who} arrived at the very end of the window."])
    s["work"] = [f"{tech or 'They'} {work}."]
    s["explain"] = ([f"{tech or 'The tech'} explained everything and showed us photos before doing any work.",
                     "They walked us through a few options with no pressure.",
                     "Clear written quote, no surprises."]
                    if o["communicated"] else
                    ["Nobody explained what was being done or why.",
                     "We got one story from the office and a different one from the tech."])

    if o["upsold"]:
        item = rng.choice(UPSELL_ITEM[svc.category])
        s["upsell"] = [f"They pushed hard for a {item} we didn't need.",
                       f"The tech spent 20 minutes trying to sell us a {item}.",
                       f"Felt like a sales pitch for a {item} more than a service call."]
    if job.quote:
        if o["overrun_pct"] > 0.05:
            s["price"] = [f"We were quoted {fmt_usd(job.quote)} but the final bill came to {fmt_usd(job.final)}.",
                          f"Quote was {fmt_usd(job.quote)}, invoice was {fmt_usd(job.final)}, and nobody told us why until we asked."]
        else:
            s["price"] = [f"The quote was {fmt_usd(job.quote)} and the final bill matched.",
                          f"Paid {fmt_usd(job.final)}, which felt fair."]
            if b.tier == "budget":
                s["price"].append(f"They were the cheapest of the quotes we got, at {fmt_usd(job.final)}.")
            if b.tier == "premium":
                s["price"].append(f"Not the cheapest at {fmt_usd(job.final)}, but worth every penny.")
    if o["quality_ok"]:
        later = rng.randint(2, 14)
        s["quality"] = ["Everything has worked perfectly since.", f"{later} months later and no problems.",
                        "Work looks great and it fixed the problem."]
        if svc.permit:
            s["quality"].append("Passed the city inspection on the first try.")
    else:
        symptom = rng.choice(SYMPTOM[svc.category])
        s["quality"] = [f"Two weeks later the {symptom} was back.",
                        "The problem wasn't actually fixed and they were slow to come back out.",
                        "We ended up paying another company to redo it."]
        if svc.permit:
            s["quality"].append("The work failed the city inspection.")
    s["clean"] = (["They cleaned up after themselves.", "Left the place cleaner than they found it."]
                  if o["clean"] else ["They left a mess for us to clean up."])
    if o["rating"] >= 5:
        s["close"] = ["Highly recommend.", f"{b.short} is our go-to now.", "Will absolutely use them again.",
                      f"Ask for {tech}!" if tech else "Can't recommend them enough."]
    elif o["rating"] == 4:
        s["close"] = ["Would use again.", "Good experience overall.", "Solid work, minor hiccups."]
    elif o["rating"] == 3:
        s["close"] = ["Mixed feelings.", "Decent work, but set your expectations on timing.", "Not sure we'd call them again."]
    else:
        s["close"] = ["Would not recommend.", "Save yourself the headache.", "Very disappointed."]
    return s


PLATFORM_SHAPE = {
    "google": (["open", "work", "quality", "close"], (1, 4)),
    "yelp": (["open", "response", "arrival", "work", "explain", "upsell", "price", "quality", "clean", "close"], (5, 8)),
    "angi": (["open", "arrival", "work", "price", "quality", "close"], (3, 5)),
    "thumbtack": (["work", "arrival", "quality", "close"], (1, 3)),
    "houzz": (["open", "explain", "work", "quality", "clean", "price", "close"], (4, 6)),
    "bbb": (["open", "explain", "upsell", "price", "quality", "close"], (4, 6)),
}


def compose_review(job: Job, platform: str) -> str:
    s = review_sentences(job, platform)
    order, (lo, hi) = PLATFORM_SHAPE[platform]
    o = job.outcome
    # A low rating always states what went wrong; a high one usually says what went right.
    failing = [k for k, bad in (("quality", not o["quality_ok"]), ("arrival", not o["on_time"]),
                                ("explain", not o["communicated"]), ("price", o["overrun_pct"] > 0.05),
                                ("upsell", o["upsold"])) if bad and k in s]
    must = ["open"] + (failing[:2] if o["rating"] <= 3 else failing[:1])
    optional = [k for k in order if k not in must and k in s]
    k = max(0, rng.randint(lo, hi) - len(must))
    chosen = set(must) | set(rng.sample(optional, k=min(k, len(optional))))
    if "close" in s and rng.random() < 0.8:
        chosen.add("close")
    parts = [rng.choice(s[key]) for key in order if key in chosen and s.get(key)]
    return " ".join(parts)


def review_title(job: Job) -> str:
    r = job.outcome["rating"]
    svc = cat.casual_name(job.service.service_id)
    if r >= 5:
        return rng.choice([f"Great {svc}", "Professional and fair", "Highly recommend", "Fast and honest"])
    if r == 4:
        return rng.choice(["Good work, a few hiccups", f"Solid {svc}", "Would hire again"])
    if r == 3:
        return rng.choice(["Mixed experience", "OK but slow", "Work fine, communication not"])
    return rng.choice(["Disappointed", "Would not hire again", "Problem not fixed", "Overcharged"])


def bbb_complaint(job: Job) -> str:
    o = job.outcome
    issues = []
    if not o["quality_ok"]:
        issues.append("the work was not completed correctly and the problem returned")
    if o["overrun_pct"] > 0.05:
        issues.append(f"I was charged {fmt_usd(job.final)} against a written estimate of {fmt_usd(job.quote)}")
    if o["upsold"]:
        issues.append("the technician pressured me into additional work")
    if not o["on_time"]:
        issues.append("they missed scheduled appointments")
    if not issues:
        issues.append("the company stopped responding to my calls")
    amt = fmt_usd(job.final) if job.final else "the full amount"
    ask = (f"I am requesting a refund of {fmt_usd(job.final * rng.choice([0.25, 0.5, 1.0]))}"
           if job.final else "I am requesting that they return to complete the work")
    return (f"On {job.date.strftime('%B %d, %Y')} I hired {job.biz.record['name']} for "
            f"{job.service.name.lower()} and paid {amt}. " + cap("; ".join(issues)) +
            f". I contacted the company multiple times without resolution. {ask}.")


def owner_response(job: Job, rating: int, reviewer: str) -> str | None:
    b = job.biz
    t = b.traits_at(job.date)
    owner = b.owner["goes_by"]
    first = reviewer.split()[0]
    if rating >= 4:
        if rng.random() > t["communication"] * 0.6:
            return None
        return rng.choice([f"Thank you {first}! We appreciate your business. - {owner}",
                           f"Thanks for the kind words, {first}. We'll pass this along to the team!",
                           f"Glad we could help, {first}. See you at the next tune-up!" if b.category == "hvac"
                           else f"Glad we could help, {first}!"])
    if rng.random() > 0.35 + 0.5 * t["communication"]:
        return None
    if t["communication"] >= 0.6:
        return (f"{first}, I'm sorry this wasn't the experience we aim for. I'd like to make it right - please "
                f"call me directly at the office and ask for {owner}.")
    return rng.choice([
        f"Our records show the work was completed as quoted and signed off by the customer.",
        f"{first}, we attempted to contact you several times. The additional charges were explained on site.",
        "This review does not reflect what happened. We stand by our work.",
    ])


# --- Main generation ---------------------------------------------------------


def counter(prefix: str, width: int):
    n = 0

    def nxt() -> str:
        nonlocal n
        n += 1
        return f"{prefix}_{n:0{width}d}"

    return nxt


def main() -> None:
    next_expert, next_offering = counter("exp", 4), counter("off", 5)
    next_est, next_review, next_job = counter("est", 5), counter("rev", 6), counter("job", 5)
    next_post, next_thread, next_proj = counter("post", 5), counter("thr", 4), counter("proj", 5)

    # Businesses, experts, offerings.
    businesses: list[Biz] = []
    idx = 1
    for category in CATEGORY_IDS:
        plan = [name for name, a in ARCHETYPES.items() for _ in range(a["n"])]
        rng.shuffle(plan)
        for archetype in plan:
            b = make_business(idx, category, archetype)
            make_experts(b, next_expert)
            make_offerings(b, next_offering)
            businesses.append(b)
            idx += 1

    customers = make_customers()
    customers_by_city: dict[str, list[dict]] = defaultdict(list)
    for c in customers:
        customers_by_city[c["city"]].append(c)

    # Jobs (each reviewed at least once) and their estimates.
    jobs: list[Job] = []
    estimates: list[dict] = []
    by_cat: dict[str, list[Biz]] = defaultdict(list)
    for b in businesses:
        by_cat[b.category].append(b)

    for b in businesses:
        a = ARCHETYPES[b.archetype]
        n_jobs = rng.randint(*a["reviews"])
        active_from = max(START, date(b.record["founded_year"], rng.randint(1, 12), 1))
        offered = list(b.offerings.values())
        weights = [3.0 if cat.SERVICE_BY_ID[o["service_id"]].core else 1.0 for o in offered]
        for _ in range(n_jobs):
            off = wchoice(offered, weights)
            svc = cat.SERVICE_BY_ID[off["service_id"]]
            d = seasonal_date(active_from, TODAY - timedelta(days=3), svc.season)
            if svc.category == "plumbing" and rng.random() < 0.08 and active_from <= date(2021, 2, 14):
                svc = cat.SERVICE_BY_ID["plb_emergency"] if "plb_emergency" in b.offerings else svc
                off = b.offerings.get(svc.service_id, off)
                d = rand_date(date(2021, 2, 15), date(2021, 3, 10))
            city = rng.choice(b.record["service_area"]["cities"])
            customer = rng.choice(customers_by_city[city] or customers)
            field_experts = [e for e in b.experts if e["field_role"]]
            expert = rng.choice(field_experts) if field_experts and rng.random() < 0.75 else None
            qty = job_quantity(svc)
            unit_price = off["typical_price_usd"] * rng.uniform(0.8, 1.25)
            quote = round_price(clamp(unit_price, off["price_low_usd"], off["price_high_usd"]) * qty)
            outcome = job_outcome(b, d)
            final = round_price(quote * (1 + outcome["overrun_pct"]))
            job = Job(next_job(), b, customer, svc, off, expert, d, qty, quote, final, outcome)
            jobs.append(job)

            big = quote >= 1500
            if rng.random() < (0.65 if big else 0.2):
                issued = d - timedelta(days=rng.randint(1, 21 if big else 3))
                project_id = next_proj()
                job.estimate_id = next_est()
                estimates.append(make_estimate(
                    job.estimate_id, b, customer, svc, qty, quote, issued, expert, "accepted", project_id=project_id,
                    job={"job_id": job.job_id, "completed_at": ts(d), "final_invoice_usd": final,
                         "change_orders": ([{"description": rng.choice(["Additional parts found during work",
                                                                         "Scope change requested on site",
                                                                         "Unforeseen damage repaired"]),
                                             "amount_usd": money(final - quote)}] if final > quote else []),
                         "duration_days": max(1, int(rng.gauss(1 if svc.market_high < 3000 else 4, 1)))}))
                # Comparison shopping: competing estimates for the same project.
                if big:
                    rivals = [r for r in by_cat[b.category] if r is not b and svc.service_id in r.offerings
                              and city in r.record["service_area"]["cities"]]
                    for r in rng.sample(rivals, k=min(len(rivals), rng.randint(0, 2))):
                        roff = r.offerings[svc.service_id]
                        rquote = round_price(clamp(roff["typical_price_usd"] * rng.uniform(0.85, 1.25),
                                                   roff["price_low_usd"], roff["price_high_usd"]) * qty)
                        rexperts = [e for e in r.experts if e["field_role"]] or r.experts
                        reason = ("price too high" if rquote > quote else
                                  rng.choice(["chose a company with better reviews", "could not schedule soon enough",
                                              "did not hear back after the estimate", "went with a referral"]))
                        estimates.append(make_estimate(next_est(), r, customer, svc, qty, rquote,
                                                       issued + timedelta(days=rng.randint(-5, 5)),
                                                       rng.choice(rexperts), "declined", reason, project_id))

    # Open estimates that never became jobs.
    for _ in range(160):
        b = rng.choice(businesses)
        off = rng.choice(list(b.offerings.values()))
        svc = cat.SERVICE_BY_ID[off["service_id"]]
        city = rng.choice(b.record["service_area"]["cities"])
        customer = rng.choice(customers_by_city[city] or customers)
        qty = job_quantity(svc)
        total = round_price(off["typical_price_usd"] * rng.uniform(0.85, 1.2) * qty)
        issued = rand_date(date(2023, 1, 1), TODAY - timedelta(days=1))
        status = ("pending" if (TODAY - issued).days <= 30 else
                  wchoice(["expired", "declined"], [0.6, 0.4]))
        estimates.append(make_estimate(next_est(), b, customer, svc, qty, total, issued,
                                       rng.choice(b.experts), status,
                                       rng.choice(["price too high", "decided to wait", "DIY instead"]) if status == "declined" else None,
                                       next_proj()))

    # Reviews.
    reviews: list[dict] = []
    review_flags: list[dict] = []
    review_mentions: list[dict] = []
    platform_ids = ["google", "yelp", "angi", "thumbtack", "bbb", "houzz"]

    def platform_for(b: Biz, rating: int) -> str:
        w = {"google": 0.44, "yelp": 0.18, "angi": 0.13, "thumbtack": 0.10, "bbb": 0.04,
             "houzz": 0.11 if b.category in ("landscaping", "roofing") else 0.02}
        if rating <= 2:
            w["bbb"] *= 4
            w["yelp"] *= 1.4
        return wchoice(list(w), list(w.values()))

    def make_review(job: Job, platform: str, rating: int, created: date) -> dict:
        c = job.customer
        display = c["platform_handles"][platform]
        is_complaint = platform == "bbb" and rating <= 2
        text = bbb_complaint(job) if is_complaint else compose_review(job, platform)
        if platform == "google" and rating >= 4 and rng.random() < 0.08:
            text = ""  # rating-only review
        resp = owner_response(job, rating, display)
        rid = next_review()
        rev = {
            "review_id": rid, "platform": platform, "business_id": job.biz.id, "customer_id": c["customer_id"],
            "reviewer_display_name": display,
            "reviewer_total_reviews": rng.randint(1, 12) if rng.random() < 0.6 else rng.randint(13, 240),
            "review_type": "complaint" if is_complaint else "review",
            "rating": None if is_complaint else rating,
            "title": review_title(job) if platform in ("angi", "houzz") else None,
            "text": text, "created_at": ts(created), "service_id": job.service.service_id,
            "job_date": job.date.isoformat(),
            "estimate_id": job.estimate_id,
            "cost_reported_usd": job.final if platform in ("angi", "thumbtack", "houzz") and rng.random() < 0.7 else None,
            "verified_hire": platform in ("angi", "thumbtack") and rng.random() < 0.85,
            "sub_ratings": None, "photos_count": rng.choice([0, 0, 0, 1, 2, 4]) if rating >= 4 or rng.random() < 0.3 else 0,
            "helpful_count": int(rng.expovariate(1 / (6 if platform == "yelp" else 2))),
            "complaint_status": (wchoice(["resolved", "unresolved", "answered"], [0.4, 0.35, 0.25]) if is_complaint else None),
            "owner_response": ({"text": resp, "created_at": ts(min(TODAY, created + timedelta(days=rng.randint(0, 10))))}
                               if resp else None),
            "language": "en",
        }
        if platform == "angi":
            o = job.outcome
            def sub(ok: bool) -> int:
                return int(clamp(rating + (0 if ok else -rng.randint(1, 2)) + rng.choice([0, 0, 1]), 1, 5))
            rev["sub_ratings"] = {"quality": sub(o["quality_ok"]), "value": sub(o["overrun_pct"] < 0.05),
                                  "punctuality": sub(o["on_time"]), "professionalism": sub(not o["upsold"]),
                                  "responsiveness": sub(o["communicated"])}
        if job.expert and job.expert["goes_by"] in text:
            review_mentions.append({"review_id": rid, "expert_id": job.expert["expert_id"],
                                    "surface_form": job.expert["goes_by"]})
        return rev

    for job in jobs:
        platform = platform_for(job.biz, job.outcome["rating"])
        created = min(TODAY, job.date + timedelta(days=int(rng.expovariate(1 / 6))))
        rev = make_review(job, platform, job.outcome["rating"], created)
        reviews.append(rev)
        job_review = rev
        # Same customer posting about the same job on a second platform.
        if rng.random() < 0.07:
            other = rng.choice([p for p in platform_ids if p not in (platform, "bbb")])
            r2 = int(clamp(job.outcome["rating"] + rng.choice([0, 0, 0, -1, 1]), 1, 5))
            dup = make_review(job, other, r2, min(TODAY, created + timedelta(days=rng.randint(0, 12))))
            reviews.append(dup)
            review_flags.append({"review_id": dup["review_id"], "flag": "cross_platform_duplicate",
                                 "related_review_id": job_review["review_id"],
                                 "detail": "Same customer and job reviewed on two platforms."})

    # Suspicious burst of generic 5-star reviews for one business per category.
    generic = ["Best {trade} in Austin!!!", "Amazing service, very professional. 5 stars!",
               "Great company highly recommend", "Fast and affordable. Will use again!",
               "Five stars. Super professional team.", "Excellent work, great price, friendly staff.",
               "Highly recommend {name}!! Best in town", "Very professional and honest. Thank you!"]
    for b in businesses:
        a = ARCHETYPES[b.archetype]
        if "fake_burst" not in a:
            continue
        start = rand_date(date(2025, 9, 1), date(2026, 6, 1))
        for _ in range(rng.randint(*a["fake_burst"])):
            first, last = rng.choice(cat.FIRST_NAMES), rng.choice(cat.LAST_NAMES)
            rid = next_review()
            platform = wchoice(["google", "yelp"], [0.85, 0.15])
            reviews.append({
                "review_id": rid, "platform": platform, "business_id": b.id, "customer_id": None,
                "reviewer_display_name": f"{first} {last[0]}." if rng.random() < 0.5 else f"{first}{rng.randint(1, 999)}",
                "reviewer_total_reviews": rng.randint(1, 2), "review_type": "review", "rating": 5, "title": None,
                "text": rng.choice(generic).format(trade=TRADE[b.category], name=b.short),
                "created_at": ts(start + timedelta(days=rng.randint(0, 20))), "service_id": None, "job_date": None,
                "estimate_id": None, "cost_reported_usd": None, "verified_hire": False, "sub_ratings": None,
                "photos_count": 0, "helpful_count": 0, "complaint_status": None, "owner_response": None,
                "language": "en"})
            review_flags.append({"review_id": rid, "flag": "suspicious_burst", "related_review_id": None,
                                 "detail": "Part of a burst of short generic 5-star reviews from new accounts."})

    for b in businesses:
        b.record["listed_on"] = sorted({r["platform"] for r in reviews if r["business_id"] == b.id})

    # --- Social posts --------------------------------------------------------
    posts: list[dict] = []
    social_mentions: list[dict] = []
    experiences: dict[tuple[str, str], list[Job]] = defaultdict(list)
    for j in jobs:
        experiences[(j.customer["city"], j.biz.category)].append(j)
    jobs_by_biz: dict[str, list[Job]] = defaultdict(list)
    for j in jobs:
        jobs_by_biz[j.biz.id].append(j)

    def surface(b: Biz, platform: str) -> str:
        owner = b.owner["goes_by"]
        forms = [b.record["name"], b.short, b.short, b.short.lower(), f"{owner} at {b.short}",
                 f"{owner} with {b.short}", f"{owner} the {TRADE[b.category]}"]
        if platform in ("tiktok", "instagram") and b.record["social_accounts"].get(platform):
            forms += [b.record["social_accounts"][platform]] * 3
        return rng.choice(forms)

    def engagement(platform: str, boost: float = 1.0) -> dict:
        if platform == "tiktok":
            views = int(rng.lognormvariate(8.2, 1.4) * boost)
            return {"views": views, "likes": int(views * rng.uniform(0.02, 0.09)),
                    "comments": int(views * rng.uniform(0.001, 0.01)), "shares": int(views * rng.uniform(0.001, 0.02))}
        if platform == "instagram":
            likes = int(rng.lognormvariate(3.5, 1.0) * boost)
            return {"views": None, "likes": likes, "comments": int(likes * rng.uniform(0.02, 0.15)), "shares": None}
        likes = int(rng.expovariate(1 / 4))
        return {"views": None, "likes": likes, "comments": None, "shares": None}

    def add_post(platform: str, post_type: str, author: dict, created: date, text: str, *, thread_id=None,
                 parent=None, community=None, customer_id=None, media=None, hashtags=None,
                 mentions: list[tuple[Biz, str, dict | None, str]] = (), boost=1.0, job: Job | None = None) -> dict:
        pid = next_post()
        post = {"post_id": pid, "platform": platform, "post_type": post_type, "thread_id": thread_id,
                "parent_post_id": parent, "community": community, "author": author, "customer_id": customer_id,
                "created_at": ts(created), "text": text, "media": media or [], "hashtags": hashtags or [],
                "engagement": engagement(platform, boost)}
        posts.append(post)
        for b, form, expert, sentiment in mentions:
            social_mentions.append({"post_id": pid, "business_id": b.id, "surface_form": form,
                                    "expert_id": expert["expert_id"] if expert else None,
                                    "expert_surface_form": expert["goes_by"] if expert else None,
                                    "sentiment": sentiment, "source_job_id": job.job_id if job else None})
        return post

    def customer_author(c: dict, platform: str) -> dict:
        handle = c["platform_handles"].get(platform) or f"{c['first_name']} {c['last_name'][0]}."
        return {"handle": handle, "display_name": handle if platform in ("nextdoor", "facebook") else None,
                "account_type": "homeowner"}

    def experience_reply(j: Job, platform: str, post_date: date) -> tuple[str, str, Biz, str, dict | None]:
        b, o = j.biz, j.outcome
        form = surface(b, platform)
        tech = j.expert if j.expert and rng.random() < 0.5 else None
        when = ago_phrase(j.date, post_date)
        svc = cat.casual_name(j.service.service_id)
        if o["rating"] >= 4:
            sentiment = "positive"
            bits = [f"We used {form} for our {svc} {when}.",
                    rng.choice([f"{tech['goes_by']} was great." if tech else "They were great.",
                                "Showed up on time and explained everything." if o["on_time"] else "Took a bit to schedule but worth it.",
                                "Honest and fair pricing.", "Would definitely use again."])]
            if rng.random() < 0.4:
                bits.append(f"Paid about {fmt_usd(j.final)}.")
        elif o["rating"] == 3:
            sentiment = "mixed"
            bits = [f"We had {form} out {when} for our {svc}.", "Work was okay but communication was rough."]
        else:
            sentiment = "negative"
            complaint = ("they never really fixed the problem" if not o["quality_ok"] else
                         f"the bill came in way over the {fmt_usd(j.quote)} quote" if o["overrun_pct"] > 0.05 else
                         "they kept trying to upsell us" if o["upsold"] else "they were impossible to get ahold of")
            bits = [rng.choice([f"I'd steer clear of {form}.", f"Not {form}, ", f"Avoid {form}."]).rstrip(", ") +
                    f" We used them {when} and {complaint}."]
        return " ".join(bits), sentiment, b, form, tech

    def thread(platform: str, city: str, community: str, start_bounds=(date(2022, 1, 1), TODAY - timedelta(days=2))):
        category = rng.choice(CATEGORY_IDS)
        pool = [j for j in experiences[(city, category)]]
        author_c = rng.choice(customers_by_city[city])
        d = rand_date(*start_bounds)
        svc = rng.choice([s for s in cat.SERVICES if s.category == category])
        trade = TRADE[category]
        where = author_c["neighborhood"] if platform == "nextdoor" else city
        ask = rng.choice([
            f"Can anyone recommend a good {trade} in {where}? {cap(rng.choice(svc.problems))}.",
            f"Looking for a reliable {trade}. {cap(rng.choice(svc.problems))} and I don't want to get ripped off.",
            f"Who do you all use for {cat.casual_name(svc.service_id)}? Need someone licensed and honest in {where}.",
            f"{cap(rng.choice(svc.problems))}. Any {trade} recommendations? Bonus if they can come out this week.",
        ])
        tid = next_thread()
        root = add_post(platform, "recommendation_request", customer_author(author_c, platform), d, ask,
                        thread_id=tid, community=community, customer_id=author_c["customer_id"])
        prior = [j for j in pool if j.date < d and j.customer is not author_c]
        rng.shuffle(prior)
        for j in prior[: rng.randint(1, 5)]:
            rd = min(TODAY, d + timedelta(days=rng.choice([0, 0, 1, 2])))
            text, sentiment, b, form, tech = experience_reply(j, platform, rd)
            mentions = [(b, form, tech, sentiment)]
            add_post(platform, "reply_recommendation" if sentiment != "negative" else "reply_warning",
                     customer_author(j.customer, platform), rd, text, thread_id=tid, parent=root["post_id"],
                     community=community, customer_id=j.customer["customer_id"], mentions=mentions, job=j)
        if rng.random() < 0.45:
            other = rng.choice(customers_by_city[city])
            add_post(platform, "reply_other", customer_author(other, platform), min(TODAY, d + timedelta(days=1)),
                     rng.choice(["Following!", "Also need one, following.",
                                 f"Make sure they're licensed - you can look them up on the state website.",
                                 "Get at least three quotes. Prices are all over the place.",
                                 "Whatever you do, don't pay the full amount upfront."]),
                     thread_id=tid, parent=root["post_id"], community=community, customer_id=other["customer_id"])

    cities = list(CITY_WEIGHTS)
    city_w = list(CITY_WEIGHTS.values())
    for _ in range(70):
        city = wchoice(cities, city_w)
        nbhd = rng.choice(NEIGHBORHOODS[city])["name"]
        thread("nextdoor", city, f"{nbhd}, {city}")
    for _ in range(35):
        city = wchoice(cities, city_w)
        thread("facebook", city, rng.choice(cat.FACEBOOK_GROUPS[city]))
    for _ in range(14):
        city = wchoice(cities, city_w)
        thread("reddit", city, cat.SUBREDDITS[city])

    # Facebook price checks built from real estimates.
    big_estimates = [e for e in estimates if e["total_usd"] >= 2000 and e["status"] in ("accepted", "declined")]
    biz_by_id = {b.id: b for b in businesses}
    cust_by_id = {c["customer_id"]: c for c in customers}
    for e in rng.sample(big_estimates, k=min(30, len(big_estimates))):
        b, c = biz_by_id[e["business_id"]], cust_by_id[e["customer_id"]]
        svc = cat.SERVICE_BY_ID[e["service_id"]]
        d = date.fromisoformat(e["issued_at"][:10]) + timedelta(days=rng.randint(0, 3))
        community = rng.choice(cat.FACEBOOK_GROUPS[c["city"]])
        form = surface(b, "facebook")
        tid = next_thread()
        root = add_post("facebook", "price_check", customer_author(c, "facebook"), d,
                        f"Got a quote for {fmt_usd(e['total_usd'])} from {form} for our {cat.casual_name(svc.service_id)}"
                        + (f" ({e['quantity']:g} {svc.unit})" if svc.unit not in ("job", "system", "visit") else "")
                        + f". Is that reasonable for {c['city']}?",
                        thread_id=tid, community=community, customer_id=c["customer_id"],
                        mentions=[(b, form, None, "neutral")])
        comparable = [j for j in jobs if j.service is svc and j.date < d and j.biz is not b]
        for j in rng.sample(comparable, k=min(len(comparable), rng.randint(1, 3))):
            form2 = surface(j.biz, "facebook")
            sentiment = "positive" if j.outcome["rating"] >= 4 else "negative"
            text = (f"We paid {fmt_usd(j.final)} with {form2} {ago_phrase(j.date, d)}. "
                    + ("Happy with them." if sentiment == "positive" else "Wouldn't use them again though."))
            add_post("facebook", "reply_price_comparison", customer_author(j.customer, "facebook"),
                     min(TODAY, d + timedelta(days=rng.randint(0, 2))), text, thread_id=tid, parent=root["post_id"],
                     community=community, customer_id=j.customer["customer_id"],
                     mentions=[(j.biz, form2, None, sentiment)], job=j)

    # Reddit PSAs about badly behaving businesses.
    bad = [b for b in businesses if b.archetype in ("inconsistent", "suspicious_reviews", "upsell_heavy", "declining")]
    for b in rng.sample(bad, k=10):
        neg = [j for j in jobs_by_biz[b.id] if j.outcome["rating"] <= 2]
        if not neg:
            continue
        j = rng.choice(neg)
        d = min(TODAY, j.date + timedelta(days=rng.randint(5, 40)))
        city = j.customer["city"]
        form = surface(b, "reddit")
        issue = ("quoted one price and billed another" if j.outcome["overrun_pct"] > 0.05 else
                 "tried to sell me a whole new system" if j.outcome["upsold"] else
                 "did shoddy work and ghosted me")
        tid = next_thread()
        root = add_post("reddit", "warning", customer_author(j.customer, "reddit"), d,
                        f"PSA: be careful with {form}. They {issue}. {cap(rng.choice(['Get multiple quotes.', 'Check their license.', 'Read the 1-star reviews first.']))}",
                        thread_id=tid, community=cat.SUBREDDITS[city], customer_id=j.customer["customer_id"],
                        mentions=[(b, form, None, "negative")], job=j)
        for j2 in [x for x in jobs_by_biz[b.id] if x is not j and x.date < d][:rng.randint(1, 3)]:
            text, sentiment, _, form2, tech = experience_reply(j2, "reddit", d)
            add_post("reddit", "reply_recommendation" if sentiment == "positive" else "reply_warning",
                     customer_author(j2.customer, "reddit"), min(TODAY, d + timedelta(days=1)), text, thread_id=tid,
                     parent=root["post_id"], community=cat.SUBREDDITS[city], customer_id=j2.customer["customer_id"],
                     mentions=[(b, form2, tech, sentiment)], job=j2)

    # TikTok and Instagram: business showcases, homeowner storytimes, price reveals.
    for b in businesses:
        for platform in ("tiktok", "instagram"):
            handle = b.record["social_accounts"].get(platform)
            if not handle:
                continue
            for _ in range(rng.randint(1, 3)):
                j = rng.choice(jobs_by_biz[b.id])
                if j.outcome["rating"] <= 2:
                    continue
                svc = j.service
                tag = slugify(svc.name.split(" (")[0])[:24]
                d = min(TODAY, j.date + timedelta(days=rng.randint(1, 14)))
                hashtags = ["#austin", "#atx", f"#{CAT_HASHTAG[b.category]}", f"#{tag}",
                            rng.choice(["#homeowner", "#beforeandafter", "#satisfying", "#texas", "#smallbusiness"])]
                if platform == "tiktok":
                    problem, work = problem_and_work(svc)
                    hook = rng.choice([f"POV: the customer says {third_person(problem)} in {j.customer['city']}",
                                       f"Watch us fix this: {third_person(problem)}",
                                       f"What a {cat.casual_name(svc.service_id)} looks like start to finish"])
                    tech = j.expert or b.owner
                    transcript = (f"Hey y'all, {tech['goes_by']} here with {b.short}. Today we're in "
                                  f"{j.customer['neighborhood']} because {third_person(problem)}. "
                                  f"So what we did was we {third_person(work)}. "
                                  f"{rng.choice(['If you are in the Austin area, give us a call.', 'Follow for more.', 'Questions? Drop them below.'])}")
                    media = [{"type": "video", "duration_s": rng.randint(12, 95), "transcript": transcript}]
                    add_post("tiktok", "project_showcase", {"handle": handle, "display_name": b.short, "account_type": "business"},
                             d, hook, media=media, hashtags=hashtags, mentions=[(b, handle, tech, "positive")])
                else:
                    media = [{"type": "photo", "alt_text": f"Before and after: {svc.name.lower()}"}] * rng.randint(1, 2)
                    add_post("instagram", "project_showcase", {"handle": handle, "display_name": b.short, "account_type": "business"},
                             d, f"Before ➡️ after. {cap(third_person(rng.choice(svc.work)))} in {j.customer['neighborhood']}. "
                                f"Thanks for trusting us! {rng.choice(['DM for a free estimate.', 'Link in bio.', ''])}".strip(),
                             media=media, hashtags=hashtags, mentions=[(b, handle, None, "positive")])

    homeowner_jobs = [j for j in jobs if j.customer["platform_handles"].get("tiktok") or j.customer["platform_handles"].get("instagram")]
    for j in rng.sample(homeowner_jobs, k=min(70, len(homeowner_jobs))):
        c, b, o = j.customer, j.biz, j.outcome
        platform = "tiktok" if c["platform_handles"].get("tiktok") and rng.random() < 0.7 else "instagram"
        if not c["platform_handles"].get(platform):
            platform = "tiktok" if c["platform_handles"].get("tiktok") else "instagram"
        d = min(TODAY, j.date + timedelta(days=rng.randint(1, 30)))
        form = surface(b, platform)
        svc = cat.casual_name(j.service.service_id)
        if o["rating"] <= 2:
            kind, sentiment = "storytime_complaint", "negative"
            caption = rng.choice([f"Storytime: the {TRADE[b.category]} who {('billed me ' + fmt_usd(j.final) + ' on a ' + fmt_usd(j.quote) + ' quote') if o['overrun_pct'] > 0.05 else 'never actually fixed it'} 😤",
                                  f"Don't make my mistake with your {svc}"])
            transcript = (f"So we hired {form} for {svc}. "
                          + ("They quoted us " + fmt_usd(j.quote) + " and the final bill was " + fmt_usd(j.final) + ". " if o["overrun_pct"] > 0.05 else "")
                          + ("And two weeks later it was broken again. " if not o["quality_ok"] else "")
                          + ("The tech tried to sell us a whole bunch of stuff we didn't need. " if o["upsold"] else "")
                          + "Get multiple quotes, y'all.")
            boost = 3.0
        elif rng.random() < 0.5:
            kind, sentiment = "price_reveal", "positive" if o["rating"] >= 4 else "mixed"
            caption = f"How much our {svc} cost in {c['city']} ({datetime(j.date.year, 1, 1).year}): {fmt_usd(j.final)} 💸"
            transcript = (f"Here's exactly what we paid for {svc}: {fmt_usd(j.final)} with {form}. "
                          f"We got {rng.randint(2, 4)} quotes. " + ("Super happy with how it turned out." if o["rating"] >= 4 else "It's fine, communication could have been better."))
            boost = 1.5
        else:
            kind, sentiment = "homeowner_recommendation", "positive" if o["rating"] >= 4 else "mixed"
            caption = rng.choice([f"Shoutout to {form} 🙌", f"Our {svc} is DONE", f"Finally fixed our {svc.split()[0]} situation"])
            transcript = f"Huge shoutout to {form}" + (f" and {j.expert['goes_by']}" if j.expert else "") + f" for our {svc}. " + (
                "Showed up on time, cleaned up, and the price was exactly the quote." if o["on_time"] and o["overrun_pct"] == 0 else "Took a while but it looks great.")
            boost = 1.0
        media = ([{"type": "video", "duration_s": rng.randint(15, 180), "transcript": transcript}] if platform == "tiktok"
                 else [{"type": "photo", "alt_text": f"Photo of completed {svc}"}])
        text = caption if platform == "tiktok" else f"{caption.rstrip('.!')}. {transcript}"
        add_post(platform, kind, {"handle": c["platform_handles"][platform], "display_name": None, "account_type": "homeowner"},
                 d, text, customer_id=c["customer_id"], media=media,
                 hashtags=["#austin", f"#{CAT_HASHTAG[b.category]}", rng.choice(["#homeownerproblems", "#firsttimehomebuyer", "#diyfail", "#homeimprovement"])],
                 mentions=[(b, form, j.expert if j.expert and j.expert["goes_by"] in transcript else None, sentiment)],
                 boost=boost, job=j)

    # --- Write ---------------------------------------------------------------
    experts = [e for b in businesses for e in b.experts]
    offerings = [o for b in businesses for o in b.offerings.values()]
    estimates.sort(key=lambda e: e["estimate_id"])
    posts.sort(key=lambda p: p["post_id"])

    ref = OUT / "reference"
    write_json(ref / "categories.json", cat.CATEGORIES)
    write_json(ref / "services.json", [{
        "service_id": s.service_id, "category": s.category, "name": s.name,
        "common_name": cat.casual_name(s.service_id), "pricing_model": s.pricing_model,
        "unit": s.unit, "market_price_low_usd": s.market_low, "market_price_high_usd": s.market_high,
        "typical_duration": s.duration, "permit_typically_required": s.permit, "aliases": s.aliases,
        "common_problems": s.problems, "typical_work": s.work,
        "peak_months": sorted(m for m, w in (s.season or {}).items() if w >= 1.8)} for s in cat.SERVICES])
    write_json(ref / "service_areas.json", [{"city": a["city"], "county": a["county"], "state": "TX",
                                              "neighborhoods": [{"name": n["name"], "zip": n["zip"],
                                                                 "typical_home_years": list(n["home_era"])}
                                                                for n in a["neighborhoods"]]}
                                             for a in cat.SERVICE_AREAS])
    write_json(ref / "platforms.json", {"review_platforms": cat.REVIEW_PLATFORMS, "social_platforms": cat.SOCIAL_PLATFORMS})

    write_jsonl(OUT / "businesses.jsonl", [b.record for b in businesses])
    write_jsonl(OUT / "experts.jsonl", experts)
    write_jsonl(OUT / "service_offerings.jsonl", offerings)
    write_jsonl(OUT / "customers.jsonl", customers)
    write_jsonl(OUT / "estimates.jsonl", estimates)
    write_jsonl(OUT / "reviews.jsonl", reviews)
    write_jsonl(OUT / "social_posts.jsonl", posts)

    gt = OUT / "ground_truth"
    write_jsonl(gt / "business_profiles.jsonl", [{
        "business_id": b.id, "name": b.record["name"], "archetype": b.archetype, "price_tier": b.tier,
        "price_multiplier": round(b.mult, 3), "traits": b.traits,
        "decline_after": ARCHETYPES[b.archetype].get("decline_after", None) and ARCHETYPES[b.archetype]["decline_after"].isoformat(),
        "declined_traits": ARCHETYPES[b.archetype].get("decline")} for b in businesses])
    write_jsonl(gt / "jobs.jsonl", [{
        "job_id": j.job_id, "business_id": j.biz.id, "customer_id": j.customer["customer_id"],
        "expert_id": j.expert["expert_id"] if j.expert else None, "service_id": j.service.service_id,
        "date": j.date.isoformat(), "quantity": j.quantity, "quoted_usd": j.quote, "final_usd": j.final,
        "estimate_id": j.estimate_id, "outcome": j.outcome} for j in jobs])
    write_jsonl(gt / "review_flags.jsonl", review_flags)
    write_jsonl(gt / "review_expert_mentions.jsonl", review_mentions)
    write_jsonl(gt / "social_mentions.jsonl", social_mentions)

    stats = validate_and_summarize(businesses, experts, offerings, customers, estimates, reviews, posts,
                                   social_mentions, review_flags)
    write_json(OUT / "manifest.json", {
        "name": "home_services_synthetic", "version": "1.0.0", "generated_on": TODAY.isoformat(), "seed": SEED,
        "synthetic": True,
        "disclaimer": ("All businesses, people, reviews and posts are fictional and generated. Cities, neighborhoods, "
                       "platform names, product brands and licensing bodies are real and used only for realism. "
                       "License numbers contain 'SYN', phone numbers use the reserved 555-01xx range and "
                       "websites use the reserved .example domain."),
        "timezone": "America/Chicago", "currency": "USD", "region": "Austin, TX metro",
        "date_range": [START.isoformat(), TODAY.isoformat()],
        "files": stats["files"], "summary": stats["summary"]})
    print(json.dumps(stats, indent=2))


def validate_and_summarize(businesses, experts, offerings, customers, estimates, reviews, posts, social_mentions, flags):
    biz_ids = {b.id for b in businesses}
    exp_ids = {e["expert_id"] for e in experts}
    cust_ids = {c["customer_id"] for c in customers}
    est_ids = {e["estimate_id"] for e in estimates}
    svc_ids = set(cat.SERVICE_BY_ID)
    post_ids = {p["post_id"] for p in posts}
    errors = []

    def check(cond, msg):
        if not cond:
            errors.append(msg)

    for b in businesses:
        check(b.record["owner_expert_id"] in exp_ids, f"{b.id} owner")
        check(any(o["business_id"] == b.id for o in offerings), f"{b.id} has no offerings")
    for e in experts:
        check(e["business_id"] in biz_ids, f"{e['expert_id']} business")
    for o in offerings:
        check(o["service_id"] in svc_ids and o["business_id"] in biz_ids, f"{o['offering_id']} refs")
        check(o["price_low_usd"] <= o["typical_price_usd"] <= o["price_high_usd"], f"{o['offering_id']} price order")
    for e in estimates:
        check(e["business_id"] in biz_ids and e["customer_id"] in cust_ids and e["service_id"] in svc_ids,
              f"{e['estimate_id']} refs")
        check(e["expert_id"] is None or e["expert_id"] in exp_ids, f"{e['estimate_id']} expert")
        check(abs(sum(i["total_usd"] for i in e["line_items"]) - e["subtotal_usd"]) < 0.05, f"{e['estimate_id']} sum")
        if e["job"]:
            check(e["job"]["completed_at"] >= e["issued_at"], f"{e['estimate_id']} dates")
    founded = {b.id: b.record["founded_year"] for b in businesses}
    for r in reviews:
        check(r["business_id"] in biz_ids, f"{r['review_id']} business")
        check(r["customer_id"] is None or r["customer_id"] in cust_ids, f"{r['review_id']} customer")
        check(r["estimate_id"] is None or r["estimate_id"] in est_ids, f"{r['review_id']} estimate")
        check(int(r["created_at"][:4]) >= founded[r["business_id"]], f"{r['review_id']} before founding")
        check(r["created_at"][:10] <= TODAY.isoformat(), f"{r['review_id']} in future")
    for p in posts:
        check(p["parent_post_id"] is None or p["parent_post_id"] in post_ids, f"{p['post_id']} parent")
        check(p["customer_id"] is None or p["customer_id"] in cust_ids, f"{p['post_id']} customer")
    for m in social_mentions:
        check(m["post_id"] in post_ids and m["business_id"] in biz_ids, f"mention {m['post_id']}")
    if errors:
        raise SystemExit("Validation failed:\n" + "\n".join(errors[:30]))

    rated = [r for r in reviews if r["rating"]]
    by_cat = Counter(b.category for b in businesses)
    return {
        "files": {"businesses.jsonl": len(businesses), "experts.jsonl": len(experts),
                  "service_offerings.jsonl": len(offerings), "customers.jsonl": len(customers),
                  "estimates.jsonl": len(estimates), "reviews.jsonl": len(reviews), "social_posts.jsonl": len(posts),
                  "ground_truth/social_mentions.jsonl": len(social_mentions),
                  "ground_truth/review_flags.jsonl": len(flags)},
        "summary": {
            "businesses_per_category": dict(by_cat),
            "services_in_catalog": len(cat.SERVICES),
            "reviews_per_platform": dict(Counter(r["platform"] for r in reviews)),
            "rating_distribution": dict(sorted(Counter(r["rating"] for r in rated).items())),
            "mean_rating": round(sum(r["rating"] for r in rated) / len(rated), 2),
            "bbb_complaints": sum(r["review_type"] == "complaint" for r in reviews),
            "estimates_by_status": dict(Counter(e["status"] for e in estimates)),
            "posts_per_platform": dict(Counter(p["platform"] for p in posts)),
            "post_types": dict(Counter(p["post_type"] for p in posts)),
        },
    }


if __name__ == "__main__":
    main()
