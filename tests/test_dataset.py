"""Integrity checks for the synthetic home-services dataset."""

import json

import pytest

from expert_finder.config import get_settings

DATA = get_settings().data_dir


def load(name: str) -> list[dict]:
    with (DATA / name).open() as f:
        return [json.loads(line) for line in f]


@pytest.fixture(scope="module")
def ds() -> dict[str, list[dict]]:
    names = ["businesses", "experts", "service_offerings", "customers", "estimates", "reviews", "social_posts"]
    data = {n: load(f"{n}.jsonl") for n in names}
    data["services"] = json.loads((DATA / "reference" / "services.json").read_text())
    return data


def ids(rows: list[dict], key: str) -> set[str]:
    return {r[key] for r in rows}


def test_counts_match_manifest(ds):
    manifest = json.loads((DATA / "manifest.json").read_text())
    for name, n in manifest["files"].items():
        if "/" not in name:
            assert len(ds[name.removesuffix(".jsonl")]) == n, name


def test_five_categories_with_twenty_businesses_each(ds):
    counts: dict[str, int] = {}
    for b in ds["businesses"]:
        counts[b["primary_category"]] = counts.get(b["primary_category"], 0) + 1
    assert counts == {c: 20 for c in ("plumbing", "electrical", "hvac", "roofing", "landscaping")}


def test_references_resolve(ds):
    biz, exp, cust = ids(ds["businesses"], "business_id"), ids(ds["experts"], "expert_id"), ids(ds["customers"], "customer_id")
    svc, est, posts = ids(ds["services"], "service_id"), ids(ds["estimates"], "estimate_id"), ids(ds["social_posts"], "post_id")
    assert all(b["owner_expert_id"] in exp for b in ds["businesses"])
    assert all(e["business_id"] in biz for e in ds["experts"])
    assert all(o["business_id"] in biz and o["service_id"] in svc for o in ds["service_offerings"])
    for e in ds["estimates"]:
        assert e["business_id"] in biz and e["customer_id"] in cust and e["service_id"] in svc
    for r in ds["reviews"]:
        assert r["business_id"] in biz and r["customer_id"] in cust | {None} and r["estimate_id"] in est | {None}
    assert all(p["parent_post_id"] in posts | {None} for p in ds["social_posts"])


def test_prices_are_consistent(ds):
    for o in ds["service_offerings"]:
        assert o["price_low_usd"] <= o["typical_price_usd"] <= o["price_high_usd"]
    for e in ds["estimates"]:
        assert abs(sum(i["total_usd"] for i in e["line_items"]) - e["subtotal_usd"]) < 0.05
        assert e["total_usd"] >= e["subtotal_usd"]
