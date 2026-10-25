"""Embedding model selection."""

from __future__ import annotations

from langchain_core.embeddings import Embeddings
from sklearn.feature_extraction.text import HashingVectorizer

from expert_finder.config import Settings


class LocalHashEmbeddings(Embeddings):
    """Offline lexical embeddings (hashed unigrams + bigrams, L2-normalized).

    Not semantic, so it is only a stand-in for running the pipeline and tests
    without an OpenAI key. Use text-embedding-3-small for real retrieval quality.
    """

    def __init__(self, n_features: int = 2048):
        self._vectorizer = HashingVectorizer(n_features=n_features, ngram_range=(1, 2), stop_words="english",
                                             alternate_sign=False, norm="l2")

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._vectorizer.transform(texts).toarray().tolist()

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


def get_embeddings(settings: Settings) -> Embeddings:
    if settings.resolved_embeddings_provider() == "openai":
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(model=settings.embedding_model, api_key=settings.openai_api_key)
    return LocalHashEmbeddings()
