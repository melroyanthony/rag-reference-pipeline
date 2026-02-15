"""Main RAG pipeline -- retrieve context, generate answer."""

from __future__ import annotations

from dataclasses import dataclass, field

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from rag.config import LLMProvider, settings
from rag.retriever import hybrid_retrieve


_SYSTEM_PROMPT = (
    "You are a helpful assistant. Use ONLY the context below to answer the "
    "user's question. If the context does not contain enough information, "
    "say so -- do not fabricate an answer.\n\n"
    "Context:\n{context}"
)


@dataclass
class RAGResult:
    """Container for a RAG pipeline response."""

    answer: str
    sources: list[Document] = field(default_factory=list)


def _build_llm(provider: LLMProvider | None = None) -> BaseChatModel:
    """Instantiate the configured LLM."""
    provider = provider or settings.llm_provider

    if provider == LLMProvider.OPENAI:
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=settings.openai_llm_model,
            api_key=settings.openai_api_key,  # type: ignore[arg-type]
        )

    if provider == LLMProvider.ANTHROPIC:
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(
            model=settings.anthropic_llm_model,  # type: ignore[arg-type]
            api_key=settings.anthropic_api_key,  # type: ignore[arg-type]
        )

    msg = f"Unsupported LLM provider: {provider}"
    raise ValueError(msg)


def query(
    question: str,
    llm_provider: LLMProvider | None = None,
    top_k: int | None = None,
) -> RAGResult:
    """Run the full RAG pipeline: retrieve, augment, generate.

    Parameters
    ----------
    question:
        The user's natural-language question.
    llm_provider:
        Override the default LLM provider for this call.
    top_k:
        Number of documents to feed as context.

    Returns
    -------
    RAGResult
        The generated answer together with source documents.
    """
    docs = hybrid_retrieve(question, final_top_k=top_k)
    context = "\n\n---\n\n".join(doc.page_content for doc in docs)

    llm = _build_llm(provider=llm_provider)
    messages = [
        SystemMessage(content=_SYSTEM_PROMPT.format(context=context)),
        HumanMessage(content=question),
    ]

    response = llm.invoke(messages)
    return RAGResult(answer=str(response.content), sources=docs)
