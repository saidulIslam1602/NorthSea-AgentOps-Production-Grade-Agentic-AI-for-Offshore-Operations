"""Application configuration via Pydantic settings."""

from functools import lru_cache

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ─── LLM ──────────────────────────────────────────────────────────────────
    openai_api_key: SecretStr = Field(..., description="OpenAI API key")
    openai_model: str = Field("gpt-4o", description="Primary LLM model")
    openai_mini_model: str = Field("gpt-4o-mini", description="Fast/cheap LLM model")
    openai_embedding_model: str = Field("text-embedding-3-small", description="Embedding model")

    # ─── Database ─────────────────────────────────────────────────────────────
    database_url: str = Field("postgresql+psycopg://northsea:northsea_dev@localhost:5432/northsea_agentops")
    database_pool_size: int = Field(10)
    database_max_overflow: int = Field(20)

    # ─── Kafka ────────────────────────────────────────────────────────────────
    kafka_bootstrap_servers: str = Field("localhost:9092")
    kafka_topic_telemetry: str = Field("well.telemetry")
    kafka_topic_anomalies: str = Field("well.anomalies")
    kafka_topic_escalations: str = Field("agent.escalations")
    kafka_consumer_group: str = Field("northsea-agentops")

    # ─── Agent Configuration ──────────────────────────────────────────────────
    agent_confidence_threshold: float = Field(0.75)
    agent_evidence_coverage_threshold: float = Field(0.60)
    agent_max_iterations: int = Field(10)
    agent_timeout_seconds: int = Field(120)

    # ─── RAG Configuration ────────────────────────────────────────────────────
    rag_chunk_size: int = Field(640, description="Target chunk length in words (ingestion)")
    rag_chunk_overlap: int = Field(112, description="Overlap in words between consecutive chunks")
    rag_top_k: int = Field(14, description="Number of fused chunks passed to generation / citations")
    rag_similarity_threshold: float = Field(
        0.52,
        description="Minimum cosine similarity (0–1) for semantic_search when filtering by similarity",
    )
    rag_coverage_weak_threshold: float = Field(
        0.38,
        description="Coverage below this (stopword‑aware query term overlap) marks weak_evidence",
    )
    rag_evidence_relief_min_coverage: float = Field(
        0.24,
        description="Min coverage alongside strong fused score → skip weak_evidence pessimism gate",
    )
    rag_evidence_relief_min_relevance: float = Field(
        0.18,
        description="Minimum top‑chunk fused relevance_score to qualify for relief (0–1 RRF‑scaled)",
    )
    rag_field_query_bm25_alpha: float = Field(
        0.42,
        description="When field‑aggregate heuristic fires, tilt hybrid fusion toward lexical (overview hits)",
    )
    rag_lexical_overlap_rerank_weight: float = Field(
        0.28,
        description="Additive weight when re‑ordering pooled chunks by substantive token overlap with query",
    )
    rag_field_overview_inject_chunks: int = Field(
        14,
        description="Chunks prioritized from VOLVE_Field_Overview for field-scope hybrid questions",
    )
    rag_completion_model: str | None = Field(
        None,
        description="Override OpenAI model for RAG answer generation (CLI/RAG eval); unset uses openai_model",
    )
    rag_min_fused_score: float = Field(
        0.04,
        description=(
            "Drop fused RRF rows below this *before* top‑k truncation; cosine threshold must not be reused "
            "here (different scale)"
        ),
    )
    rag_hybrid_alpha: float = Field(0.55, description="Weight for semantic vs BM25 (1.0=semantic only)")

    # ─── Observability ────────────────────────────────────────────────────────
    otel_service_name: str = Field("northsea-agentops")
    otel_exporter_otlp_endpoint: str = Field("http://localhost:4318")
    otel_exporter_otlp_enabled: bool = Field(
        True,
        description="When False, OTLP exporters are not registered (avoids errors without a local collector)",
    )

    # ─── MLflow ───────────────────────────────────────────────────────────────
    mlflow_tracking_uri: str = Field("http://localhost:5001")
    mlflow_experiment_name: str = Field("northsea-agentops-eval")

    # ─── Application ──────────────────────────────────────────────────────────
    app_env: str = Field("development")
    app_log_level: str = Field("INFO")
    app_secret_key: str = Field("change-me-in-production")
    api_host: str = Field("0.0.0.0")
    api_port: int = Field(8000)


@lru_cache
def get_settings() -> Settings:
    return Settings()  # type: ignore[call-arg]
