"""FastAPI service exposing the RAG pipeline over HTTP."""

from __future__ import annotations

from fastapi import FastAPI, HTTPException
from langchain_core.documents import Document
from pydantic import BaseModel, Field

from rag.chunker import chunk_documents
from rag.pipeline import query as rag_query
from rag.vectorstore import ingest as vectorstore_ingest

app = FastAPI(
    title="RAG Reference Pipeline",
    version="0.1.0",
    description="Production-ready Retrieval-Augmented Generation with hybrid search.",
)


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class DocumentIn(BaseModel):
    """A single document to ingest."""

    content: str = Field(..., description="Plain-text content of the document.")
    metadata: dict[str, str] = Field(
        default_factory=dict, description="Arbitrary key-value metadata."
    )


class IngestRequest(BaseModel):
    """Batch of documents to ingest."""

    documents: list[DocumentIn] = Field(..., min_length=1)


class IngestResponse(BaseModel):
    """Acknowledgement after ingestion."""

    ingested: int
    chunks: int


class QueryRequest(BaseModel):
    """A natural-language question for the RAG pipeline."""

    question: str = Field(..., min_length=1)
    top_k: int = Field(default=5, ge=1, le=50)


class SourceOut(BaseModel):
    """A source document returned alongside the answer."""

    content: str
    metadata: dict[str, str]


class QueryResponse(BaseModel):
    """Generated answer plus supporting sources."""

    answer: str
    sources: list[SourceOut]


class HealthResponse(BaseModel):
    """Health-check payload."""

    status: str


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse, tags=["ops"])
async def health() -> HealthResponse:
    """Liveness / readiness probe."""
    return HealthResponse(status="ok")


@app.post("/ingest", response_model=IngestResponse, tags=["rag"])
async def ingest(request: IngestRequest) -> IngestResponse:
    """Chunk and ingest documents into the vector store."""
    try:
        docs = [
            Document(page_content=d.content, metadata=d.metadata)
            for d in request.documents
        ]
        chunks = chunk_documents(docs)
        vectorstore_ingest(chunks)
        return IngestResponse(ingested=len(docs), chunks=len(chunks))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/query", response_model=QueryResponse, tags=["rag"])
async def query(request: QueryRequest) -> QueryResponse:
    """Query the RAG pipeline and return a generated answer with sources."""
    try:
        result = rag_query(request.question, top_k=request.top_k)
        sources = [
            SourceOut(content=doc.page_content, metadata=doc.metadata)
            for doc in result.sources
        ]
        return QueryResponse(answer=result.answer, sources=sources)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
