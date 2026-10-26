"""SQLite store and query functions, checked against direct recomputation from the raw dataset."""

import pytest

from expert_finder import sql_queries as q
from expert_finder import sql_store
from expert_finder.config import get_settings
from expert_finder.dataset import Dataset


@pytest.fixture(scope="module")
def ds():
    return Dataset.load(get_settings().data_dir)


@pytest.fixture(scope="module")
def con(ds, tmp_path_factory):
    path = tmp_path_factory.mktemp("sql") / "test.db"
    sql_store.build(ds, path)
    c = sql_store.connect(path)
    yield c
    c.close()


def test_build_row_counts_and_foreign_keys(ds, con):
    assert con.execute("SELECT count(*) FROM businesses").fetchone()[0] == len(ds.businesses)
    assert con.execute("SELECT count(*) FROM reviews").fetchone()[0] == len(ds.reviews)
    assert con.execute("SELECT count(*) FROM estimate_line_items").fetchone()[0] == sum(
        len(e["line_items"]) for e in ds.estimates)
    assert con.execute("PRAGMA foreign_key_check").fetchall() == []


def test_connection_is_read_only(con):
    with pytest.raises(Exception, match="readonly"):
        con.execute("DELETE FROM businesses")


def test_resolve_business_exact_and_fuzzy(con):
    assert q.resolve_business(con, "Bee Cave Roofing")[0]["business_id"] == "biz_0080"
    assert q.resolve_business(con, "hernandez")[0]["business_id"] == "biz_0001"
    assert q.resolve_business(con, "Hernandes and Sons Plumbng")[0]["business_id"] == "biz_0001"


@pytest.mark.parametrize("text,expected", [("tankless", "plb_tankless_install"), ("can lights", "ele_recessed_lighting"),
                                           ("new roof", "roof_shingle_replacement"), ("French drain", "land_drainage")])
def test_resolve_service_uses_names_and_aliases(con, text, expected):
    assert q.resolve_service(con, text)[0]["service_id"] == expected


def test_find_businesses_filters_are_exact(ds, con):
    rows = q.find_businesses(con, category="hvac", city="Round Rock", emergency=True, language="Spanish", limit=100)
    assert rows
    for r in rows:
        b = ds.business_by_id[r["business_id"]]
        assert "hvac" in [b["primary_category"], *b["secondary_categories"]]
        assert "Round Rock" in b["service_area"]["cities"]
        assert b["emergency_service"]["available"] and "Spanish" in b["languages"]


def test_find_businesses_includes_requested_service_price(ds, con):
    rows = q.find_businesses(con, service_id="plb_tankless_install", limit=100)
    offering = {o["business_id"]: o for o in ds.offerings if o["service_id"] == "plb_tankless_install"}
    assert {r["business_id"] for r in rows} == set(offering)
    assert all(r["typical_price_usd"] == offering[r["business_id"]]["typical_price_usd"] for r in rows)


def test_price_stats_matches_raw_quotes(ds, con):
    stats = q.price_stats(con, "hvac_ac_replacement", "Round Rock")
    raw = [e["total_usd"] for e in ds.estimates
           if e["service_id"] == "hvac_ac_replacement" and e["job_address"]["city"] == "Round Rock"]
    assert stats["quotes"]["n"] == len(raw)
    assert stats["quotes"]["min"] == round(min(raw), 2) and stats["quotes"]["max"] == round(max(raw), 2)


def test_compare_quote_verdicts_follow_percentiles(con):
    dist = q.price_stats(con, "hvac_ac_replacement")["quotes"]
    assert q.compare_quote(con, "hvac_ac_replacement", dist["min"] - 1)["verdict"] == "low"
    assert q.compare_quote(con, "hvac_ac_replacement", dist["median"])["verdict"] == "typical"
    assert q.compare_quote(con, "hvac_ac_replacement", dist["max"] + 1)["verdict"] == "very high"


def test_quote_accuracy_matches_raw_invoices(ds, con):
    for bid in ("biz_0001", "biz_0080", "biz_0058"):
        jobs = [e for e in ds.estimates if e["business_id"] == bid and e["status"] == "accepted" and e["job"]]
        over = [e for e in jobs if (e["job"]["final_invoice_usd"] - e["subtotal_usd"]) / e["subtotal_usd"] > 0.05]
        result = q.quote_accuracy(con, bid)
        assert result["completed_jobs_with_quote"] == len(jobs) and result["jobs_over_quote"] == len(over)


def test_rating_summary_split_partitions_reviews(ds, con):
    r = q.rating_summary(con, "biz_0046", split_date="2025-05-01")
    rated = [x for x in ds.reviews if x["business_id"] == "biz_0046" and x["rating"] is not None]
    assert r["rated"] == len(rated) == r["before_split"]["rated"] + r["after_split"]["rated"]
    new = r["new_account_reviews"]["reviews"]
    assert new + r["rating_from_established_accounts"]["rated"] == len(rated)
