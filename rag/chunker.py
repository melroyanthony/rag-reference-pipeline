"""Document chunking with configurable strategies."""

from __future__ import annotations

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from rag.config import settings


def build_splitter(
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> RecursiveCharacterTextSplitter:
    """Return a configured ``RecursiveCharacterTextSplitter``."""
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size or settings.chunk_size,
        chunk_overlap=chunk_overlap or settings.chunk_overlap,
        length_function=len,
        is_separator_regex=False,
    )


def chunk_documents(
    documents: list[Document],
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[Document]:
    """Split *documents* into smaller chunks while preserving metadata.

    Each resulting chunk inherits the metadata of its parent document and
    receives an additional ``chunk_index`` field.
    """
    splitter = build_splitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks: list[Document] = []

    for doc in documents:
        splits = splitter.split_text(doc.page_content)
        for idx, text in enumerate(splits):
            meta = {**doc.metadata, "chunk_index": idx}
            chunks.append(Document(page_content=text, metadata=meta))

    return chunks
