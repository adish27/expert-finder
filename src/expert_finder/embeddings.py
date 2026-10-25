"""Embedding model."""

from __future__ import annotations

from langchain_openai import OpenAIEmbeddings

from expert_finder.config import Settings


def get_embeddings(settings: Settings) -> OpenAIEmbeddings:
    return OpenAIEmbeddings(model=settings.embedding_model, dimensions=settings.embedding_dimensions,
                            api_key=settings.openai_api_key)
