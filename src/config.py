"""Application configuration via Pydantic settings."""

from functools import lru_cache

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
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
    rag_chunk_size: int = Field(512)
    rag_chunk_overlap: int = Field(64)
    rag_top_k: int = Field(8)
    rag_similarity_threshold: float = Field(0.65)
    rag_hybrid_alpha: float = Field(0.5, description="Weight for semantic vs BM25 (1.0=semantic)")

    # ─── Observability ────────────────────────────────────────────────────────
    otel_service_name: str = Field("northsea-agentops")
    otel_exporter_otlp_endpoint: str = Field("http://localhost:4318")

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
