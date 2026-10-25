"""Runtime configuration loaded from environment variables / .env."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# LangSmith and the provider SDKs read os.environ directly, so export .env there too.
load_dotenv(PROJECT_ROOT / ".env")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=PROJECT_ROOT / ".env", extra="ignore")

    openai_api_key: str | None = None
    chat_model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"
    embedding_dimensions: int = 1536

    pinecone_api_key: str | None = None
    pinecone_index: str = "home-services"
    pinecone_cloud: str = "aws"
    pinecone_region: str = "us-east-1"

    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_username: str = "neo4j"
    neo4j_password: str = "password"
    neo4j_database: str = "neo4j"

    data_dir: Path = PROJECT_ROOT / "data" / "home_services"
    storage_dir: Path = PROJECT_ROOT / ".storage"

    top_k: int = 6
    tool_loadout_k: int = 3


@lru_cache
def get_settings() -> Settings:
    return Settings()
