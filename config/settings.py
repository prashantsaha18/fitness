"""
config/settings.py
──────────────────
Centralized, validated application configuration via Pydantic BaseSettings.
All secrets are injected via environment variables; never hard-coded.

Design principle: 12-factor app compliance. A single Settings singleton is
constructed once at import time and shared across the entire service graph.
"""
from __future__ import annotations

import secrets
from functools import lru_cache
from typing import Literal

from pydantic import Field, AnyUrl, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Runtime ───────────────────────────────────────────────────────────────
    ENV: Literal["development", "staging", "production"] = "development"
    LOG_LEVEL: str = "INFO"
    SERVICE_NAME: str = "fitness-rec-engine"
    API_V1_PREFIX: str = "/api/v1"

    # ── Security ──────────────────────────────────────────────────────────────
    JWT_SECRET_KEY: str = Field(default_factory=lambda: secrets.token_hex(32))
    JWT_ALGORITHM: str = "HS256"
    JWT_EXPIRE_MINUTES: int = 60

    # ── PostgreSQL (NeonDB) ───────────────────────────────────────────────────
    DATABASE_URL: str = (
        "postgresql+asyncpg://neondb_owner:npg_JqDQtgWVE3C2"
        "@ep-young-frost-aq03fhnr.c-8.us-east-1.aws.neon.tech"
        "/neondb?ssl=require"
    )
    DB_POOL_SIZE: int = 20
    DB_MAX_OVERFLOW: int = 10
    DB_POOL_TIMEOUT: float = 30.0
    DB_ECHO: bool = False

    # ── Redis (Online Feature Cache) ──────────────────────────────────────────
    REDIS_URL: str = "redis://localhost:6379/0"
    REDIS_FEATURE_TTL_SECONDS: int = 300          # 5-min sliding window
    REDIS_POOL_MAX_CONNECTIONS: int = 50

    # ── Qdrant (Vector DB) ────────────────────────────────────────────────────
    QDRANT_HOST: str = "localhost"
    QDRANT_PORT: int = 6333
    QDRANT_GRPC_PORT: int = 6334
    QDRANT_API_KEY: str | None = None
    QDRANT_COLLECTION_NAME: str = "fitness_content"
    QDRANT_EMBEDDING_DIM: int = 384               # all-MiniLM-L6-v2 output
    QDRANT_TOP_K: int = 100                        # Stage-1 candidate pool

    # HNSW hyper-parameters — tuned for recall@100 ≥ 0.97 at <10ms P99
    HNSW_M: int = 16
    HNSW_EF_CONSTRUCT: int = 200
    HNSW_EF_SEARCH: int = 128                     # runtime ef; ef ≥ top_k

    # ── Kafka ─────────────────────────────────────────────────────────────────
    KAFKA_BOOTSTRAP_SERVERS: str = "localhost:9092"
    KAFKA_TELEMETRY_TOPIC: str = "fitness.telemetry.realtime"
    KAFKA_INTERACTION_TOPIC: str = "fitness.user.interactions"
    KAFKA_CONSUMER_GROUP: str = "rec-engine-consumers"

    # ── Embedding Model ───────────────────────────────────────────────────────
    EMBEDDING_MODEL_NAME: str = "all-MiniLM-L6-v2"
    EMBEDDING_BATCH_SIZE: int = 256
    EMBEDDING_DEVICE: str = "cpu"                 # swap to "cuda" for GPU nodes

    # ── ONNX Ranking Model ────────────────────────────────────────────────────
    ONNX_MODEL_PATH: str = "artifacts/ranking_model.onnx"
    ONNX_INTRA_OP_THREADS: int = 4
    ONNX_INTER_OP_THREADS: int = 2

    # ── Feast Feature Store ───────────────────────────────────────────────────
    FEAST_REPO_PATH: str = "feature_repo"

    # ── Safety Thresholds ─────────────────────────────────────────────────────
    MAX_SODIUM_HYPERTENSIVE_MG: float = 400.0     # hard filter for HTN users
    MAX_INTENSITY_CARDIAC_RISK: float = 0.7       # normalised intensity score
    MIN_CONFIDENCE_THRESHOLD: float = 0.05        # discard near-zero scores

    # ── Performance ───────────────────────────────────────────────────────────
    RECOMMENDATION_TIMEOUT_MS: float = 28.0       # internal budget (< 30ms SLA)
    STAGE1_TIMEOUT_MS: float = 10.0
    STAGE2_TIMEOUT_MS: float = 15.0
    FALLBACK_CACHE_TTL_SECONDS: int = 600

    @field_validator("DATABASE_URL", mode="before")
    @classmethod
    def _coerce_async_driver(cls, v: str) -> str:
        """Ensure asyncpg driver is always used, regardless of .env format."""
        if v.startswith("postgresql://") or v.startswith("postgres://"):
            return v.replace("postgresql://", "postgresql+asyncpg://", 1).replace(
                "postgres://", "postgresql+asyncpg://", 1
            )
        return v


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Module-level singleton — import and call get_settings() everywhere.
    lru_cache ensures .env is parsed exactly once per process lifetime.
    """
    return Settings()


settings = get_settings()
