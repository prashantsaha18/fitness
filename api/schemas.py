"""
api/schemas.py
───────────────
Pydantic v2 request/response schemas for the recommendation API.

Design principles:
  • Strict mode on all request models — reject extra fields to prevent
    parameter injection attacks.
  • Response models are permissive (extra="ignore") to allow schema evolution
    without breaking existing clients.
  • All optional realtime context fields have safe defaults documented in
    field descriptions — the API is fully functional with user_id alone.
"""
from __future__ import annotations

from typing import Optional
from uuid import UUID

from pydantic import BaseModel, Field, field_validator, model_validator


# ── Request Schemas ───────────────────────────────────────────────────────────

class RealtimeContextOverride(BaseModel):
    """
    Optional client-side biometric state override.
    When provided, these values take precedence over the Kafka stream cache.
    Useful for client-computed signals (wearable SDK direct integration).
    """
    model_config = {"extra": "forbid", "str_strip_whitespace": True}

    heart_rate_bpm: Optional[int] = Field(
        default=None, ge=30, le=250,
        description="Current instantaneous heart rate from wearable device."
    )
    fatigue_level: Optional[float] = Field(
        default=None, ge=0.0, le=1.0,
        description="Client-computed fatigue index (0=fresh, 1=exhausted)."
    )
    active_calories_kcal: Optional[float] = Field(
        default=None, ge=0.0, le=5000.0,
        description="Cumulative active calories burned in current session."
    )
    session_id: Optional[str] = Field(
        default=None, max_length=64,
        description="Active workout session ID for interaction attribution."
    )

    @field_validator("heart_rate_bpm", "fatigue_level", "active_calories_kcal", mode="before")
    @classmethod
    def _coerce_none_strings(cls, v):
        return None if v == "" else v


class RecommendationRequest(BaseModel):
    """
    Core recommendation request payload.

    Minimal valid request: {"user_id": "<uuid>"}
    All other fields are optional overrides for context-aware ranking.
    """
    model_config = {"extra": "forbid"}

    user_id: UUID = Field(
        description="Platform user UUID — must match an active user record."
    )
    content_types: Optional[list[str]] = Field(
        default=None,
        description="Filter to specific content types: video, workout_routine, meal_recipe."
    )
    top_n: int = Field(
        default=10, ge=1, le=50,
        description="Number of ranked recommendations to return."
    )
    realtime_context: Optional[RealtimeContextOverride] = Field(
        default=None,
        description="Optional real-time biometric overrides from the client device."
    )
    diversity_factor: float = Field(
        default=0.1, ge=0.0, le=1.0,
        description="MMR diversity injection factor (0=pure relevance, 1=max diversity)."
    )
    exclude_ids: Optional[list[UUID]] = Field(
        default=None, max_length=200,
        description="Content IDs to exclude from results (recently consumed items)."
    )

    @field_validator("content_types")
    @classmethod
    def _validate_content_types(cls, v):
        allowed = {"video", "workout_routine", "meal_recipe"}
        if v is not None:
            invalid = set(v) - allowed
            if invalid:
                raise ValueError(f"Invalid content types: {invalid}. Allowed: {allowed}")
        return v


# ── Response Schemas ──────────────────────────────────────────────────────────

class RecommendedItemMetadata(BaseModel):
    """Rich metadata for a single recommended content item."""
    model_config = {"extra": "ignore"}

    content_id: str
    title: str
    content_type: str
    thumbnail_url: Optional[str] = None

    # Workout-specific
    workout_type: Optional[str] = None
    duration_minutes: Optional[int] = None
    intensity_score: Optional[float] = None
    calories_burned_estimate: Optional[float] = None

    # Recipe-specific
    sodium_mg: Optional[float] = None
    calories_kcal: Optional[float] = None
    protein_g: Optional[float] = None
    dietary_tags: Optional[list[str]] = None


class RankedRecommendation(BaseModel):
    """A single ranked recommendation with inference metadata."""
    model_config = {"extra": "ignore"}

    rank: int = Field(description="1-based rank position.")
    content_id: str
    inference_score: float = Field(
        description="Model-predicted engagement probability [0, 1]."
    )
    retrieval_score: float = Field(
        description="Stage-1 ANN cosine similarity score."
    )
    metadata: RecommendedItemMetadata
    safety_flags: list[str] = Field(
        default_factory=list,
        description="Applied safety filters (e.g., 'low_sodium_enforced')."
    )


class RecommendationResponse(BaseModel):
    """
    Top-level API response envelope.
    Includes pipeline diagnostics for observability and A/B testing.
    """
    model_config = {"extra": "ignore"}

    request_id: str
    user_id: str
    recommendations: list[RankedRecommendation]

    # Pipeline diagnostics
    pipeline_metadata: "PipelineMetadata"


class PipelineMetadata(BaseModel):
    model_config = {"extra": "ignore"}

    stage1_candidates: int = Field(description="Number of ANN candidates retrieved.")
    stage2_ranked: int = Field(description="Number of items scored by ranking model.")
    safety_filtered: int = Field(description="Number of items removed by safety rules.")
    total_latency_ms: float
    stage1_latency_ms: float
    stage2_latency_ms: float
    model_version: str
    feature_source: str = Field(
        description="'online' | 'fallback_cache' | 'cold_start'"
    )
    is_fallback: bool = Field(
        default=False,
        description="True when vector store was unavailable and fallback was served."
    )


# ── Auth Schemas ──────────────────────────────────────────────────────────────

class UserCreate(BaseModel):
    model_config = {"extra": "forbid"}

    username: str = Field(min_length=3, max_length=64)
    email: str = Field(pattern=r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
    password: str = Field(min_length=8)


class UserResponse(BaseModel):
    model_config = {"extra": "ignore"}

    id: str
    username: str
    email: str
    fitness_goal: Optional[str] = None
    is_active: bool


class HealthResponse(BaseModel):
    status: str
    version: str
    components: dict[str, str]
