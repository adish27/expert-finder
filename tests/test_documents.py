"""Offline checks for the documents sent to Pinecone."""

import pytest

from expert_finder.config import get_settings
from expert_finder.dataset import Dataset
from expert_finder.documents import NAMESPACES, build_documents

ALLOWED = (str, int, float, bool, list)


@pytest.fixture(scope="module")
def ds():
    return Dataset.load(get_settings().data_dir)


@pytest.fixture(scope="module")
def docs(ds):
    return build_documents(ds)


def test_one_document_per_searchable_record(ds, docs):
    assert set(docs) == set(NAMESPACES)
    assert len(docs["reviews"]) == sum(1 for r in ds.reviews if r["text"].strip())
    assert len(docs["social"]) == len(ds.posts)
    assert len(docs["businesses"]) == len(ds.businesses)
    ids = [d.id for group in docs.values() for d in group]
    assert len(ids) == len(set(ids))


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


def test_business_profiles_include_prices_and_team(ds, docs):
    profile = next(d for d in docs["businesses"] if d.id == "biz_0001").page_content
    assert "## Services and prices" in profile and "## Team" in profile
    owner = next(e for e in ds.experts if e["expert_id"] == ds.business_by_id["biz_0001"]["owner_expert_id"])
    assert owner["last_name"] in profile
