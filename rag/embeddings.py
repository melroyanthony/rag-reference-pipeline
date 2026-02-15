"""Multi-provider embedding factory."""

from __future__ import annotations

from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings

from rag.config import EmbeddingProvider, settings


def get_embeddings(provider: EmbeddingProvider | None = None) -> Embeddings:
    """Return an embedding model instance for the requested *provider*.

    Supported providers:
    * ``openai``  -- OpenAI ``text-embedding-3-small`` (default)
    * ``voyage``  -- Voyage AI embeddings via ``langchain-voyageai``

    The provider defaults to ``settings.embedding_provider`` when not
    supplied explicitly.
    """
    provider = provider or settings.embedding_provider

    if provider == EmbeddingProvider.OPENAI:
        return OpenAIEmbeddings(
            model=settings.openai_embedding_model,
            openai_api_key=settings.openai_api_key,  # type: ignore[arg-type]
        )

    if provider == EmbeddingProvider.VOYAGE:
        # Voyage AI embeddings use the Anthropic API key.
        try:
            from langchain_voyageai import VoyageAIEmbeddings  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ImportError(
                "Install langchain-voyageai to use Voyage embeddings: "
                "pip install langchain-voyageai"
            ) from exc

        return VoyageAIEmbeddings(
            model=settings.voyage_embedding_model,
            voyage_api_key=settings.anthropic_api_key,  # type: ignore[arg-type]
        )

    msg = f"Unsupported embedding provider: {provider}"
    raise ValueError(msg)
