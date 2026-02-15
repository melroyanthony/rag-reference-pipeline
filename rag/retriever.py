"""Hybrid retriever -- semantic search + BM25 with Reciprocal Rank Fusion."""

from __future__ import annotations

from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from rag.config import settings
from rag.vectorstore import semantic_search


def _tokenize(text: str) -> list[str]:
    """Whitespace tokenizer used by BM25."""
    return text.lower().split()


def _bm25_search(
    query: str,
    corpus: list[Document],
    k: int | None = None,
) -> list[Document]:
    """Rank *corpus* documents against *query* using BM25 and return top-*k*."""
    k = k or settings.bm25_k
    if not corpus:
        return []

    tokenized_corpus = [_tokenize(doc.page_content) for doc in corpus]
    bm25 = BM25Okapi(tokenized_corpus)
    scores = bm25.get_scores(_tokenize(query))

    ranked = sorted(zip(scores, corpus, strict=False), key=lambda x: x[0], reverse=True)
    return [doc for _, doc in ranked[:k]]


def reciprocal_rank_fusion(
    result_lists: list[list[Document]],
    k: int | None = None,
) -> list[Document]:
    """Merge multiple ranked lists using Reciprocal Rank Fusion (RRF).

    ``k`` is the RRF constant (default 60) that dampens the contribution of
    low-ranked results.
    """
    k = k or settings.rrf_k
    scores: dict[str, float] = {}
    doc_map: dict[str, Document] = {}

    for result_list in result_lists:
        for rank, doc in enumerate(result_list, start=1):
            key = doc.page_content
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
            doc_map[key] = doc

    sorted_keys = sorted(scores, key=lambda x: scores[x], reverse=True)
    return [doc_map[key] for key in sorted_keys]


def hybrid_retrieve(
    query: str,
    corpus: list[Document] | None = None,
    semantic_k: int | None = None,
    bm25_k: int | None = None,
    final_top_k: int | None = None,
) -> list[Document]:
    """Run hybrid retrieval and return the top-*final_top_k* documents.

    1. Semantic search via ChromaDB.
    2. BM25 sparse retrieval over the supplied *corpus* (or the semantic
       results when no explicit corpus is given).
    3. RRF merge of both ranked lists.
    """
    sem_results = semantic_search(query, k=semantic_k)

    bm25_corpus = corpus if corpus is not None else sem_results
    bm25_results = _bm25_search(query, bm25_corpus, k=bm25_k)

    fused = reciprocal_rank_fusion([sem_results, bm25_results])
    top_k = final_top_k or settings.final_top_k
    return fused[:top_k]
