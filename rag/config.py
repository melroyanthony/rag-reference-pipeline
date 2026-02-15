"""Centralised configuration loaded from environment variables."""

from __future__ import annotations

import os
from enum import StrEnum
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


class LLMProvider(StrEnum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"


class EmbeddingProvider(StrEnum):
    OPENAI = "openai"
    VOYAGE = "voyage"


class Settings:
    """Application settings sourced from environment variables."""

    # --- API keys ---
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    anthropic_api_key: str = os.getenv("ANTHROPIC_API_KEY", "")

    # --- Provider selection ---
    llm_provider: LLMProvider = LLMProvider(os.getenv("LLM_PROVIDER", "openai"))
    embedding_provider: EmbeddingProvider = EmbeddingProvider(
        os.getenv("EMBEDDING_PROVIDER", "openai")
    )

    # --- Model names ---
    openai_llm_model: str = os.getenv("OPENAI_LLM_MODEL", "gpt-4o")
    anthropic_llm_model: str = os.getenv("ANTHROPIC_LLM_MODEL", "claude-sonnet-4-20250514")
    openai_embedding_model: str = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
    voyage_embedding_model: str = os.getenv("VOYAGE_EMBEDDING_MODEL", "voyage-3")

    # --- Chunking ---
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "1000"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "200"))

    # --- ChromaDB ---
    chroma_persist_dir: Path = Path(os.getenv("CHROMA_PERSIST_DIR", "./chroma_data"))
    chroma_collection: str = os.getenv("CHROMA_COLLECTION", "documents")

    # --- Retrieval ---
    semantic_k: int = int(os.getenv("SEMANTIC_K", "10"))
    bm25_k: int = int(os.getenv("BM25_K", "10"))
    rrf_k: int = int(os.getenv("RRF_K", "60"))
    final_top_k: int = int(os.getenv("FINAL_TOP_K", "5"))


settings = Settings()
