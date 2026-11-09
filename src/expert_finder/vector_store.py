"""Pinecone index: one serverless index, one namespace per corpus."""

from __future__ import annotations

import time

from langchain_core.documents import Document
from langchain_pinecone import PineconeVectorStore
from langsmith import traceable
from pinecone import Pinecone, ServerlessSpec

from expert_finder import tracing
from expert_finder.config import Settings
from expert_finder.documents import NAMESPACES
from expert_finder.embeddings import get_embeddings


class VectorIndex:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.pc = Pinecone(api_key=settings.pinecone_api_key)
        self.embeddings = get_embeddings(settings)
        self._store: PineconeVectorStore | None = None

    def exists(self) -> bool:
        return self.pc.has_index(self.settings.pinecone_index)

    def ensure_index(self) -> None:
        s = self.settings
        if not self.exists():
            self.pc.create_index(name=s.pinecone_index, dimension=s.embedding_dimensions, metric="cosine",
                                 spec=ServerlessSpec(cloud=s.pinecone_cloud, region=s.pinecone_region),
                                 tags={"project": "home-services-expert-finder"})
        while not self.pc.describe_index(s.pinecone_index).status["ready"]:
            time.sleep(1)

    @property
    def store(self) -> PineconeVectorStore:
        if self._store is None:
            self._store = PineconeVectorStore(index=self.pc.Index(self.settings.pinecone_index),
                                              embedding=self.embeddings)
        return self._store

    def clear_namespace(self, namespace: str) -> None:
        index = self.pc.Index(self.settings.pinecone_index)
        if namespace in (index.describe_index_stats().get("namespaces") or {}):
            index.delete(delete_all=True, namespace=namespace)

    def upsert(self, namespace: str, docs: list[Document], batch_size: int = 100) -> int:
        """Embed and upsert; ids are record ids, so re-running overwrites instead of duplicating."""
        for i in range(0, len(docs), batch_size):
            batch = docs[i:i + batch_size]
            self.store.add_documents(batch, ids=[d.id for d in batch], namespace=namespace)
        return len(docs)

    @traceable(run_type="retriever", name="pinecone_fetch", process_inputs=tracing.index_inputs,
               process_outputs=tracing.fetched_outputs)
    def fetch(self, ids: list[str], namespace: str) -> dict[str, Document]:
        """Documents by id, without embedding a query.

        Used to pull a parent business profile once a chunk of it has matched: the text is already
        in Pinecone's metadata (PineconeVectorStore stores page_content under "text"), so this
        costs one lookup and no embedding call.
        """
        if not ids:
            return {}
        res = self.pc.Index(self.settings.pinecone_index).fetch(ids=ids, namespace=namespace)
        out = {}
        for vector_id, vector in (res.vectors or {}).items():
            md = dict(vector.metadata or {})
            out[vector_id] = Document(id=vector_id, page_content=md.pop("text", ""), metadata=md)
        return out

    @traceable(run_type="retriever", name="pinecone_search", process_inputs=tracing.index_inputs,
               process_outputs=tracing.hits_outputs)
    def search(self, query: str, namespace: str, k: int = 6, filter: dict | None = None) -> list[tuple[Document, float]]:
        if namespace not in NAMESPACES:
            raise ValueError(f"unknown namespace {namespace!r}; expected one of {sorted(NAMESPACES)}")
        return self.store.similarity_search_with_score(query, k=k, filter=filter, namespace=namespace)

    def stats(self) -> dict[str, int]:
        ns = self.pc.Index(self.settings.pinecone_index).describe_index_stats().get("namespaces") or {}
        return {name: info["vector_count"] for name, info in ns.items()}
