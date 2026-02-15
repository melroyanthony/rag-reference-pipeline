FROM python:3.12-slim AS base

# Install uv for fast dependency resolution
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Copy dependency manifest first for layer caching
COPY pyproject.toml ./

# Install production dependencies
RUN uv pip install --system --no-cache .

# Copy application code
COPY rag/ rag/
COPY api/ api/

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
