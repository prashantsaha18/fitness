"""
data_pipeline/schemas.py
─────────────────────────
Production-grade SQLAlchemy 2.0 async ORM definitions.

Schema design decisions:
  • Users        — normalised profile + health markers (used for safety filters)
  • ContentItems — multi-modal catalogue with pre-computed metadata vectors
  • Interactions — append-only event log; never UPDATE, only INSERT (audit trail)
  • UserEmbedding — materialised user vector refreshed by offline batch job

Indexing strategy:
  • B-Tree on all foreign keys and high-cardinality filter columns
  • GIN index on JSONB payload columns (Postgres-specific)
  • Composite index on (user_id, created_at DESC) for interaction time-series scans
"""
from __future__ import annotations

import enum
import uuid
from datetime import datetime
from typing import Optional

from sqlalchemy import (
    BigInteger,
    Boolean,
    Column,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Index,
    Integer,
    SmallInteger,
    String,
    Text,
    UniqueConstraint,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID, ARRAY
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


# ── Base ──────────────────────────────────────────────────────────────────────

class Base(DeclarativeBase):
    """Shared declarative base with server-side timestamp defaults."""
    pass


# ── Enumerations ──────────────────────────────────────────────────────────────

class ContentType(str, enum.Enum):
    VIDEO = "video"
    WORKOUT_ROUTINE = "workout_routine"
    MEAL_RECIPE = "meal_recipe"


class FitnessGoal(str, enum.Enum):
    WEIGHT_LOSS = "weight_loss"
    MUSCLE_GAIN = "muscle_gain"
    ENDURANCE = "endurance"
    FLEXIBILITY = "flexibility"
    MAINTENANCE = "maintenance"


class WorkoutType(str, enum.Enum):
    HIIT = "hiit"
    STRENGTH = "strength"
    YOGA = "yoga"
    CARDIO = "cardio"
    PILATES = "pilates"
    RECOVERY = "recovery"


class InteractionType(str, enum.Enum):
    CLICK = "click"
    COMPLETE = "complete"
    SKIP = "skip"
    SAVE = "save"
    SHARE = "share"
    ABANDON = "abandon"


# ── Models ────────────────────────────────────────────────────────────────────

class User(Base):
    """
    Core user profile with health markers.
    Health markers drive Stage-2 safety filters — never expose raw values via API.
    """
    __tablename__ = "users"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    username: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    email: Mapped[str] = mapped_column(String(255), unique=True, nullable=False)
    hashed_password: Mapped[str] = mapped_column(String(255), nullable=False)

    # ── Demographic ───────────────────────────────────────────────────────────
    age: Mapped[Optional[int]] = mapped_column(SmallInteger)
    weight_kg: Mapped[Optional[float]] = mapped_column(Float)
    height_cm: Mapped[Optional[float]] = mapped_column(Float)
    fitness_goal: Mapped[Optional[str]] = mapped_column(
        Enum(FitnessGoal, name="fitness_goal_enum")
    )
    preferred_workout_types: Mapped[Optional[list]] = mapped_column(
        ARRAY(String), nullable=True
    )

    # ── Health Markers (used for safety hard-filters) ─────────────────────────
    is_hypertensive: Mapped[bool] = mapped_column(Boolean, default=False)
    has_cardiac_risk: Mapped[bool] = mapped_column(Boolean, default=False)
    has_diabetes: Mapped[bool] = mapped_column(Boolean, default=False)
    dietary_restrictions: Mapped[Optional[dict]] = mapped_column(JSONB, default=dict)

    # ── Timestamps ────────────────────────────────────────────────────────────
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )
    last_active_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)

    # ── Relationships ─────────────────────────────────────────────────────────
    interactions: Mapped[list["UserInteraction"]] = relationship(
        back_populates="user", lazy="noload"
    )
    embedding: Mapped[Optional["UserEmbedding"]] = relationship(
        back_populates="user", uselist=False, lazy="noload"
    )

    __table_args__ = (
        Index("ix_users_email", "email"),
        Index("ix_users_fitness_goal", "fitness_goal"),
        Index("ix_users_health_markers", "is_hypertensive", "has_cardiac_risk"),
    )


class ContentItem(Base):
    """
    Multi-modal content catalogue.
    embedding_vector stored in Qdrant; this table holds scalar metadata only.
    """
    __tablename__ = "content_items"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    title: Mapped[str] = mapped_column(String(512), nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=False)
    content_type: Mapped[str] = mapped_column(
        Enum(ContentType, name="content_type_enum"), nullable=False
    )

    # ── Workout-specific metadata ─────────────────────────────────────────────
    workout_type: Mapped[Optional[str]] = mapped_column(
        Enum(WorkoutType, name="workout_type_enum")
    )
    duration_minutes: Mapped[Optional[int]] = mapped_column(SmallInteger)
    intensity_score: Mapped[Optional[float]] = mapped_column(Float)   # 0.0–1.0
    calories_burned_estimate: Mapped[Optional[float]] = mapped_column(Float)
    target_muscle_groups: Mapped[Optional[list]] = mapped_column(ARRAY(String))
    required_equipment: Mapped[Optional[list]] = mapped_column(ARRAY(String))

    # ── Recipe-specific metadata ──────────────────────────────────────────────
    sodium_mg: Mapped[Optional[float]] = mapped_column(Float)
    calories_kcal: Mapped[Optional[float]] = mapped_column(Float)
    protein_g: Mapped[Optional[float]] = mapped_column(Float)
    carbs_g: Mapped[Optional[float]] = mapped_column(Float)
    fat_g: Mapped[Optional[float]] = mapped_column(Float)
    dietary_tags: Mapped[Optional[list]] = mapped_column(ARRAY(String))

    # ── Engagement signals (updated by async aggregation job) ─────────────────
    global_ctr: Mapped[float] = mapped_column(Float, default=0.0)
    global_completion_rate: Mapped[float] = mapped_column(Float, default=0.0)
    total_interactions: Mapped[int] = mapped_column(BigInteger, default=0)

    # ── Extended attributes ───────────────────────────────────────────────────
    attributes: Mapped[Optional[dict]] = mapped_column(JSONB, default=dict)
    thumbnail_url: Mapped[Optional[str]] = mapped_column(String(1024))
    media_url: Mapped[Optional[str]] = mapped_column(String(1024))

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )
    is_published: Mapped[bool] = mapped_column(Boolean, default=True)

    __table_args__ = (
        Index("ix_content_type", "content_type"),
        Index("ix_content_workout_type", "workout_type"),
        Index("ix_content_published", "is_published"),
        Index("ix_content_intensity", "intensity_score"),
        Index("ix_content_attrs_gin", "attributes", postgresql_using="gin"),
    )


class UserInteraction(Base):
    """
    Append-only interaction event log.
    This is the ground truth training signal for the ranking model.
    Never mutate rows — only INSERT new events.
    """
    __tablename__ = "user_interactions"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    content_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("content_items.id", ondelete="CASCADE"), nullable=False
    )
    interaction_type: Mapped[str] = mapped_column(
        Enum(InteractionType, name="interaction_type_enum"), nullable=False
    )

    # ── Contextual signals at event time ──────────────────────────────────────
    session_id: Mapped[Optional[str]] = mapped_column(String(64))
    rank_position: Mapped[Optional[int]] = mapped_column(SmallInteger)
    dwell_time_seconds: Mapped[Optional[float]] = mapped_column(Float)
    completion_pct: Mapped[Optional[float]] = mapped_column(Float)   # 0.0–1.0

    # ── Real-time biometric snapshot at event time ────────────────────────────
    heart_rate_bpm: Mapped[Optional[int]] = mapped_column(SmallInteger)
    fatigue_level: Mapped[Optional[float]] = mapped_column(Float)   # 0.0–1.0
    active_calories: Mapped[Optional[float]] = mapped_column(Float)

    # ── Recommendation metadata ───────────────────────────────────────────────
    model_version: Mapped[Optional[str]] = mapped_column(String(32))
    inference_score: Mapped[Optional[float]] = mapped_column(Float)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    # ── Relationships ─────────────────────────────────────────────────────────
    user: Mapped["User"] = relationship(back_populates="interactions", lazy="noload")
    content: Mapped["ContentItem"] = relationship(lazy="noload")

    __table_args__ = (
        Index("ix_interactions_user_time", "user_id", "created_at"),
        Index("ix_interactions_content", "content_id"),
        Index("ix_interactions_session", "session_id"),
        Index("ix_interactions_type", "interaction_type"),
    )


class UserEmbedding(Base):
    """
    Materialised user embedding refreshed by the offline batch pipeline.
    Stored here for audit/reproducibility; hot copy lives in Redis.
    """
    __tablename__ = "user_embeddings"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        unique=True,
    )
    embedding: Mapped[list] = mapped_column(ARRAY(Float), nullable=False)
    model_version: Mapped[str] = mapped_column(String(32), nullable=False)
    computed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    user: Mapped["User"] = relationship(back_populates="embedding", lazy="noload")

    __table_args__ = (
        Index("ix_user_embeddings_user_id", "user_id"),
        Index("ix_user_embeddings_version", "model_version"),
    )
