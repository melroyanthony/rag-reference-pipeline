"""ChromaDB vector-store wrapper."""

from __future__ import annotations

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from rag.config import settings
from rag.embeddings import get_embeddings


def _get_chroma(
    embeddings: Embeddings | None = None,
    collection_name: str | None = None,
) -> Chroma:
    """Return a ``Chroma`` instance backed by the configured persist directory."""
    return Chroma(
        collection_name=collection_name or settings.chroma_collection,
        embedding_function=embeddings or get_embeddings(),
        persist_directory=str(settings.chroma_persist_dir),
    )


def ingest(
    documents: list[Document],
    embeddings: Embeddings | None = None,
    collection_name: str | None = None,
) -> Chroma:
    """Embed and store *documents* in ChromaDB.

    Returns the ``Chroma`` instance for further querying.
    """
    store = _get_chroma(embeddings=embeddings, collection_name=collection_name)
    store.add_documents(documents)
    return store


def semantic_search(
    query: str,
    k: int | None = None,
    embeddings: Embeddings | None = None,
    collection_name: str | None = None,
) -> list[Document]:
    """Run a similarity search against ChromaDB and return the top-*k* results."""
    store = _get_chroma(embeddings=embeddings, collection_name=collection_name)
    return store.similarity_search(query, k=k or settings.semantic_k)


def delete_collection(collection_name: str | None = None) -> None:
    """Delete the named (or default) collection from ChromaDB."""
    store = _get_chroma(collection_name=collection_name)
    store.delete_collection()
