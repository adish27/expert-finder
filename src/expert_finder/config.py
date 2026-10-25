"""Runtime configuration loaded from environment variables / .env."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=PROJECT_ROOT / ".env", extra="ignore")

    openai_api_key: str | None = None
    chat_model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"

    # "auto" picks OpenAI when an API key is present, otherwise a local hashing embedder.
    embeddings_provider: Literal["auto", "openai", "local"] = "auto"

    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "password"
    neo4j_database: str = "neo4j"

    data_dir: Path = PROJECT_ROOT / "data" / "home_services"
    storage_dir: Path = PROJECT_ROOT / ".storage"

    chunk_size: int = 700
    chunk_overlap: int = 100
    top_k: int = 6
    tool_loadout_k: int = 3

    def resolved_embeddings_provider(self) -> Literal["openai", "local"]:
        if self.embeddings_provider == "auto":
            return "openai" if self.openai_api_key else "local"
        return self.embeddings_provider

    @property
    def chroma_dir(self) -> Path:
        # Separate directories per provider: embedding dimensions differ.
        return self.storage_dir / f"chroma-{self.resolved_embeddings_provider()}"


@lru_cache
def get_settings() -> Settings:
    return Settings()
