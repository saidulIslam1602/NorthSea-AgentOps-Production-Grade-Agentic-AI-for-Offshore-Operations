FROM python:3.11-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq-dev gcc curl && \
    rm -rf /var/lib/apt/lists/*

# ─── Production dependencies only ─────────────────────────────────────────────
# Install runtime deps without the [dev] extras (pytest, ruff, mypy, etc.)
# so the production image stays lean and has no test toolchain attack surface.
FROM base AS deps

COPY pyproject.toml ./
RUN pip install --upgrade pip && \
    pip install -e ".[prod]"

# ─── Dev dependencies (used by CI / local development only) ───────────────────
FROM deps AS dev-deps

RUN pip install -e ".[dev]"

# ─── Production image ─────────────────────────────────────────────────────────
FROM base AS production

COPY --from=deps /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=deps /usr/local/bin /usr/local/bin

COPY src/ ./src/
COPY eval/ ./eval/
COPY alembic/ ./alembic/
COPY alembic.ini ./

RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
