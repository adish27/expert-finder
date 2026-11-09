"""Offline checks for the documents sent to Pinecone."""

import pytest

from expert_finder.config import get_settings
from expert_finder.dataset import Dataset
from expert_finder.documents import NAMESPACES, build_documents, business_chunk_documents

ALLOWED = (str, int, float, bool, list)


@pytest.fixture(scope="module")
def ds():
    return Dataset.load(get_settings().data_dir)


@pytest.fixture(scope="module")
def docs(ds):
    return build_documents(ds)


@pytest.fixture(scope="module")
def parents(docs):
    return [d for d in docs["businesses"] if d.metadata["doc_type"] == "business_profile"]


@pytest.fixture(scope="module")
def chunks(docs):
    return [d for d in docs["businesses"] if d.metadata["doc_type"] == "business_chunk"]


def test_one_document_per_searchable_record(ds, docs, parents):
    assert set(docs) == set(NAMESPACES)
    assert len(docs["reviews"]) == sum(1 for r in ds.reviews if r["text"].strip())
    assert len(docs["social"]) == len(ds.posts)
    assert len(parents) == len(ds.businesses)
    ids = [d.id for group in docs.values() for d in group]
    assert len(ids) == len(set(ids))


def test_every_service_and_team_member_is_separately_searchable(ds, chunks):
    """The point of chunking: one priced service is not buried in a profile covering ten."""
    sections = [c.metadata["section"] for c in chunks]
    assert sections.count("service") == len(ds.offerings)
    assert sections.count("team") == len(ds.experts)
    for fixed in ("overview", "credentials", "pricing"):
        assert sections.count(fixed) == len(ds.businesses)


def test_each_chunk_stands_alone_as_an_embedding(ds, chunks):
    """"$1,200 per job" is useless as a vector without knowing whose price it is."""
    for c in chunks:
        b = ds.business_by_id[c.metadata["business_id"]]
        header = c.page_content.splitlines()[0]
        assert b["name"] in header
        assert ds.category_name[b["primary_category"]] in header
        assert b["address"]["city"] in header


def test_chunks_carry_the_metadata_the_tool_filters_use(ds, chunks):
    # search_business_profiles filters on city and category, so a chunk must be filterable too.
    for c in chunks:
        b = ds.business_by_id[c.metadata["business_id"]]
        assert c.metadata["category"] == b["primary_category"]
        assert c.metadata["cities"] == b["service_area"]["cities"]


def test_a_chunk_belongs_to_its_parent_profile(ds, parents):
    """Chunk ids must resolve to a parent id, or parent-document retrieval cannot fetch it."""
    parent_ids = {p.id for p in parents}
    for b in ds.businesses[:5]:
        for c in business_chunk_documents(ds, b):
            assert c.id.split("#")[0] in parent_ids
            assert c.metadata["business_id"] == b["business_id"]


def test_metadata_is_pinecone_compatible(docs):
    for group in docs.values():
        for d in group:
            assert d.page_content.strip()
            for k, v in d.metadata.items():
                assert v is not None and isinstance(v, ALLOWED), (d.id, k, v)
                if isinstance(v, list):
                    assert all(isinstance(x, str) for x in v), (d.id, k)


def test_social_posts_do_not_leak_mention_labels(docs):
    # Which business a post refers to is ground truth; retrieval has to work it out.
    assert all("business_id" not in d.metadata for d in docs["social"])


def test_replies_carry_their_thread_question(ds, docs):
    reply = next(d for d in docs["social"] if d.metadata.get("parent_post_id"))
    parent = ds.post_by_id[reply.metadata["parent_post_id"]]
    assert parent["text"] in reply.page_content


def test_business_profiles_include_prices_and_team(ds, parents):
    profile = next(d for d in parents if d.id == "biz_0001").page_content
    assert "## Services and prices" in profile and "## Team" in profile
    owner = next(e for e in ds.experts if e["expert_id"] == ds.business_by_id["biz_0001"]["owner_expert_id"])
    assert owner["last_name"] in profile
