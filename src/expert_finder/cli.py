"""Command line entry point: `uv run expert-finder <command>`."""

from __future__ import annotations

import argparse
import json
import os

from expert_finder.config import get_settings
from expert_finder.dataset import Dataset
from expert_finder.documents import NAMESPACES, build_documents


def check() -> None:
    """Confirm every external service is reachable with the configured credentials."""
    s = get_settings()
    results = {}
    try:
        from expert_finder.embeddings import get_embeddings

        results["openai"] = f"ok ({len(get_embeddings(s).embed_query('ping'))} dims)"
    except Exception as e:
        results["openai"] = f"FAILED: {type(e).__name__}: {e}"
    try:
        from pinecone import Pinecone

        names = [i["name"] for i in Pinecone(api_key=s.pinecone_api_key).list_indexes()]
        results["pinecone"] = f"ok (indexes: {names})"
    except Exception as e:
        results["pinecone"] = f"FAILED: {type(e).__name__}: {e}"
    try:
        from expert_finder.graph_db import neo4j_driver

        with neo4j_driver(s) as d:
            d.verify_connectivity()
        results["neo4j"] = f"ok ({s.neo4j_uri})"
    except Exception as e:
        results["neo4j"] = f"FAILED: {type(e).__name__}: {e}"
    try:
        from langsmith import Client

        next(iter(Client().list_projects(limit=1)), None)
        results["langsmith"] = f"ok (project: {os.getenv('LANGSMITH_PROJECT')}, tracing: {os.getenv('LANGSMITH_TRACING')})"
    except Exception as e:
        results["langsmith"] = f"FAILED: {type(e).__name__}: {e}"
    for k, v in results.items():
        print(f"{k:10} {v}")


def index(namespaces: list[str], reset: bool) -> None:
    from expert_finder.vector_store import VectorIndex

    s = get_settings()
    docs = build_documents(Dataset.load(s.data_dir))
    vi = VectorIndex(s)
    vi.ensure_index()
    for ns in namespaces:
        if reset:
            vi.clear_namespace(ns)
        n = vi.upsert(ns, docs[ns])
        print(f"{ns:11} upserted {n} documents")
    print("index stats:", json.dumps(vi.stats()))


def build_sql() -> None:
    from expert_finder import sql_store

    s = get_settings()
    path = s.storage_dir / "home_services.db"
    counts = sql_store.build(Dataset.load(s.data_dir), path)
    print(f"built {path}")
    for table, n in counts.items():
        print(f"  {table:24} {n:6}")


def search(query: str, namespace: str, k: int, filter_json: str | None) -> None:
    from expert_finder.vector_store import VectorIndex

    hits = VectorIndex(get_settings()).search(query, namespace, k=k,
                                              filter=json.loads(filter_json) if filter_json else None)
    for doc, score in hits:
        md = doc.metadata
        label = md.get("business_name") or md.get("community") or ""
        print(f"[{doc.id}] score={score:.3f} {md.get('platform', '')} {label} {md.get('created_date', '')}")
        print("   " + doc.page_content[:300].replace("\n", " | "))


def main() -> None:
    parser = argparse.ArgumentParser(prog="expert-finder", description="Home services expert finder")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("check", help="test connections to OpenAI, Pinecone, Neo4j and LangSmith")
    p = sub.add_parser("index", help="embed documents and upsert them into Pinecone")
    p.add_argument("--namespace", choices=sorted(NAMESPACES), action="append",
                   help="limit to one namespace (repeatable); default: all")
    p.add_argument("--reset", action="store_true", help="delete each namespace's vectors before upserting")
    sub.add_parser("build-sql", help="rebuild the SQLite store from the dataset")
    p = sub.add_parser("search", help="semantic search in one namespace")
    p.add_argument("query")
    p.add_argument("--namespace", choices=sorted(NAMESPACES), default="reviews")
    p.add_argument("-k", type=int, default=5)
    p.add_argument("--filter", help='Pinecone metadata filter as JSON, e.g. \'{"category": "hvac"}\'')
    args = parser.parse_args()

    if args.cmd == "check":
        check()
    elif args.cmd == "index":
        index(args.namespace or list(NAMESPACES), args.reset)
    elif args.cmd == "build-sql":
        build_sql()
    elif args.cmd == "search":
        search(args.query, args.namespace, args.k, args.filter)


if __name__ == "__main__":
    main()
