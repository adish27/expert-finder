"""Application container: holds the stores and builds retrievers, tools and the agent.

Connections are opened lazily and reused, so a Streamlit rerun or a batch of evaluation
questions does not reconnect to Pinecone, Neo4j and SQLite each time.
"""

from __future__ import annotations

import threading
from typing import Literal

from expert_finder.config import Settings, get_settings

Mode = Literal["vector", "graph"]


class ExpertFinder:
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self._index = None
        self._driver = None
        # LangGraph runs tools in a thread pool and SQLite connections belong to one thread, so
        # each thread gets its own read-only connection. The Neo4j driver and Pinecone client are
        # thread-safe and stay shared.
        self._local = threading.local()

    # --- stores ---------------------------------------------------------------

    @property
    def index(self):
        if self._index is None:
            from expert_finder.vector_store import VectorIndex

            self._index = VectorIndex(self.settings)
        return self._index

    @property
    def driver(self):
        if self._driver is None:
            from expert_finder.graph_db import neo4j_driver

            self._driver = neo4j_driver(self.settings)
        return self._driver

    @property
    def sql(self):
        con = getattr(self._local, "sql", None)
        if con is None:
            from expert_finder import sql_store

            con = self._local.sql = sql_store.connect(self.settings.storage_dir / "home_services.db")
        return con

    @property
    def database(self) -> str:
        return self.settings.neo4j_database

    # --- components -----------------------------------------------------------

    def retriever(self, mode: Mode):
        from expert_finder.retrieval import GraphRetriever, VectorRetriever

        if mode == "graph":
            return GraphRetriever(self.index, self.driver, self.database, k=self.settings.top_k)
        return VectorRetriever(self.index, k=self.settings.top_k)

    def chat_model(self, temperature: float = 0.0):
        from langchain_openai import ChatOpenAI

        if not self.settings.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY is not set")
        return ChatOpenAI(model=self.settings.chat_model, temperature=temperature,
                          api_key=self.settings.openai_api_key)

    def close(self) -> None:
        if self._driver is not None:
            self._driver.close()
            self._driver = None
        con = getattr(self._local, "sql", None)
        if con is not None:
            con.close()
            self._local.sql = None
