"""Entity resolution accuracy and graph construction (offline: no Neo4j connection needed)."""

import pytest

from expert_finder.config import get_settings
from expert_finder.dataset import Dataset
from expert_finder.entity_resolution import EntityResolver, evaluate_mentions, evaluate_review_mentions
from expert_finder.graph_builder import LABELS, build_graph_data


@pytest.fixture(scope="module")
def ds():
    return Dataset.load(get_settings().data_dir)


@pytest.fixture(scope="module")
def resolver(ds):
    return EntityResolver(ds)


@pytest.fixture(scope="module")
def graph(ds, resolver):
    return build_graph_data(ds, resolver)


# --- Entity resolution -------------------------------------------------------


def test_resolves_exact_lowercase_and_handle_forms(ds, resolver):
    b = ds.business_by_id["biz_0001"]
    for text in (b["name"], b["name"].lower(), f"We used {b['name']} last year"):
        assert [m.target_id for m in resolver.resolve_businesses(text)] == ["biz_0001"]


def test_longer_business_name_wins_over_shorter_one(ds, resolver):
    pairs = [(a, b) for a in ds.businesses for b in ds.businesses
             if a is not b and b["name"].startswith(a["name"])]
    if not pairs:
        pytest.skip("no overlapping business names in the dataset")
    short, long = pairs[0]
    assert [m.target_id for m in resolver.resolve_businesses(f"We hired {long['name']}")] == [long["business_id"]]


def test_business_name_containing_a_first_name_is_not_an_expert_mention(ds, resolver):
    # "Michael's Lawn Care" names the company, not necessarily the person Michael.
    biz = next((b for b in ds.businesses if "'s" in b["name"]), None)
    if biz is None:
        pytest.skip("no possessive business names")
    post = {"text": f"We used {biz['name']} for our lawn care", "media": [], "customer_id": None,
            "platform": "facebook", "community": None, "author": {"handle": "x", "account_type": "homeowner"}}
    mentions = resolver.resolve_post(post)
    assert [m.target_id for m in mentions] == [biz["business_id"]]


def test_duplicate_first_name_prefers_the_field_worker(ds, resolver):
    for bid, experts in [(b["business_id"], ds.experts_of(b["business_id"])) for b in ds.businesses]:
        names = [e["goes_by"] for e in experts]
        dupes = [n for n in names if names.count(n) > 1]
        if not dupes:
            continue
        name = dupes[0]
        same = [e for e in experts if e["goes_by"] == name]
        if sum(e["field_role"] for e in same) != 1:
            continue
        expected = next(e["expert_id"] for e in same if e["field_role"])
        found = resolver.resolve_experts(f"{name} did the work", bid)
        assert [m.target_id for m in found] == [expected]
        return
    pytest.skip("no business has one field and one office worker sharing a name")


def test_review_mentions_ignore_the_owner_response(ds, resolver):
    review = next(r for r in ds.reviews if r["owner_response"] and r["text"].strip())
    signed = dict(review, text="They did the work.", title=None)
    assert resolver.resolve_review_experts(signed) == []


def test_resolution_accuracy_meets_thresholds(ds, resolver):
    gt = get_settings().data_dir / "ground_truth"
    posts = evaluate_mentions(ds, resolver, gt / "social_mentions.jsonl")
    reviews = evaluate_review_mentions(ds, resolver, gt / "review_expert_mentions.jsonl")
    assert posts["business_mentions"]["precision"] >= 0.98
    assert posts["business_mentions"]["recall"] >= 0.90
    assert posts["expert_mentions"]["precision"] >= 0.98
    assert reviews["precision"] >= 0.98 and reviews["recall"] >= 0.90


# --- Graph construction ------------------------------------------------------


def test_every_node_label_is_declared_and_ids_are_unique(graph):
    assert set(graph.nodes) <= set(LABELS)
    for label, rows in graph.nodes.items():
        ids = [r["id"] for r in rows]
        assert len(ids) == len(set(ids)), label


def test_relationship_endpoints_exist(graph):
    known = {label: {r["id"] for r in rows} for label, rows in graph.nodes.items()}
    for (rel_type, from_label, to_label), rows in graph.rels.items():
        for row in rows:
            assert row["s"] in known[from_label], (rel_type, row["s"])
            assert row["t"] in known[to_label], (rel_type, row["t"])


def test_node_ids_match_the_other_stores(ds, graph):
    assert {r["id"] for r in graph.nodes["Review"]} == {r["review_id"] for r in ds.reviews}
    assert {r["id"] for r in graph.nodes["Business"]} == {b["business_id"] for b in ds.businesses}
    assert {r["id"] for r in graph.nodes["Post"]} == {p["post_id"] for p in ds.posts}


def test_no_null_properties_are_written(graph):
    for rows in graph.nodes.values():
        assert all(v is not None for r in rows for v in r["props"].values())
    for rows in graph.rels.values():
        assert all(v is not None for r in rows for v in r["props"].values())


def test_offers_carries_prices_and_competed_with_is_pairwise(ds, graph):
    offers = graph.rels[("OFFERS", "Business", "Service")]
    assert len(offers) == len(ds.offerings)
    assert all("typical_price_usd" in o["props"] for o in offers)
    competed = graph.rels[("COMPETED_WITH", "Estimate", "Estimate")]
    assert competed and all(c["s"] != c["t"] for c in competed)
