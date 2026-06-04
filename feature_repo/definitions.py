"""
feature_repo/definitions.py
────────────────────────────
Feast Feature Store Definitions.

Feature taxonomy:
  ┌──────────────────────────────────────────────────────────────────────────┐
  │  Batch Features (offline_store → PostgreSQL)                             │
  │    user_30d_features    — 30-day structural adherence & engagement stats  │
  │    content_stats        — global engagement metrics per content item      │
  │                                                                           │
  │  Streaming Features (online_store → Redis, TTL=5min)                     │
  │    user_realtime_features — EMA heart rate, fatigue, active calories      │
  │                                                                           │
  │  On-Demand Features (computed at serving time, no I/O)                   │
  │    user_content_affinity — dot-product affinity signals                   │
  └──────────────────────────────────────────────────────────────────────────┘

Training-Serving Skew prevention strategy:
  ALL features consumed by the ONNX model MUST be declared here.
  Training pipelines call store.get_historical_features(); serving calls
  store.get_online_features(). Because both paths reference the same
  FeatureView objects, feature computation logic cannot diverge.
"""
from __future__ import annotations

from datetime import timedelta

import pandas as pd
from feast import (
    Entity,
    FeatureStore,
    FeatureView,
    Field,
    FileSource,
    OnDemandFeatureView,
    RequestSource,
)
from feast.types import Float32, Float64, Int32, Int64, String, Bool, Array


# ── Entities ──────────────────────────────────────────────────────────────────

user_entity = Entity(
    name="user",
    join_keys=["user_id"],
    description="Platform user identified by UUID string.",
)

content_entity = Entity(
    name="content_item",
    join_keys=["content_id"],
    description="Multi-modal content item (video/workout/recipe).",
)


# ── Data Sources ──────────────────────────────────────────────────────────────
# In production these would be PostgreSQL or Spark sources.
# File sources are used here for portability during dev/staging.

user_batch_source = FileSource(
    path="data/user_batch_features.parquet",
    timestamp_field="event_timestamp",
    created_timestamp_column="created_at",
)

content_batch_source = FileSource(
    path="data/content_stats.parquet",
    timestamp_field="event_timestamp",
    created_timestamp_column="created_at",
)

user_stream_source = FileSource(
    path="data/user_realtime_features.parquet",
    timestamp_field="event_timestamp",
)


# ── Batch Feature Views ───────────────────────────────────────────────────────

user_30d_features = FeatureView(
    name="user_30d_features",
    entities=[user_entity],
    ttl=timedelta(days=1),          # refresh daily from offline pipeline
    schema=[
        # Adherence & engagement aggregates (30-day window)
        Field(name="workout_sessions_30d", dtype=Int32),
        Field(name="avg_session_duration_min_30d", dtype=Float32),
        Field(name="structural_adherence_rate_30d", dtype=Float32),  # 0–1
        Field(name="completion_rate_30d", dtype=Float32),
        Field(name="ctr_30d", dtype=Float32),

        # Nutritional habits
        Field(name="avg_daily_calorie_intake_30d", dtype=Float32),
        Field(name="avg_protein_intake_g_30d", dtype=Float32),
        Field(name="preferred_meal_types_embedding", dtype=Array(Float32)),

        # Goal & demographic
        Field(name="fitness_goal_encoded", dtype=Int32),
        Field(name="age_normalised", dtype=Float32),
        Field(name="bmi", dtype=Float32),

        # Health markers (binary)
        Field(name="is_hypertensive", dtype=Bool),
        Field(name="has_cardiac_risk", dtype=Bool),
        Field(name="has_diabetes", dtype=Bool),

        # Long-term preference vector (384-dim, refreshed weekly)
        Field(name="user_preference_vector", dtype=Array(Float32)),
    ],
    source=user_batch_source,
    tags={"tier": "batch", "pipeline": "offline_aggregation"},
    description="30-day rolling aggregation of user behaviour and health profile.",
)


content_global_stats = FeatureView(
    name="content_global_stats",
    entities=[content_entity],
    ttl=timedelta(hours=6),         # refresh every 6h from interaction log
    schema=[
        Field(name="global_ctr", dtype=Float32),
        Field(name="global_completion_rate", dtype=Float32),
        Field(name="total_interactions_log", dtype=Float32),    # log1p(count)
        Field(name="avg_dwell_time_seconds", dtype=Float32),
        Field(name="content_type_encoded", dtype=Int32),
        Field(name="workout_type_encoded", dtype=Int32),
        Field(name="intensity_score", dtype=Float32),
        Field(name="duration_minutes_normalised", dtype=Float32),
        Field(name="sodium_mg_normalised", dtype=Float32),
        Field(name="calories_normalised", dtype=Float32),
        Field(name="content_embedding", dtype=Array(Float32)),  # 384-dim
    ],
    source=content_batch_source,
    tags={"tier": "batch", "pipeline": "content_stats"},
    description="Global engagement statistics per content item.",
)


# ── Streaming Feature View ────────────────────────────────────────────────────

user_realtime_features = FeatureView(
    name="user_realtime_features",
    entities=[user_entity],
    ttl=timedelta(minutes=5),       # matches Redis TTL
    schema=[
        Field(name="hr_mean_5min", dtype=Float32),
        Field(name="fatigue_latest", dtype=Float32),
        Field(name="cal_total_session", dtype=Float32),
        Field(name="recovery_score", dtype=Float32),
        Field(name="heart_rate_zone_encoded", dtype=Int32),
    ],
    source=user_stream_source,
    tags={"tier": "streaming", "pipeline": "kafka_consumer"},
    description="Sliding-window (5min) real-time biometric features from wearable telemetry.",
)


# ── On-Demand Feature View ────────────────────────────────────────────────────

# Request-time inputs passed directly in the serving call
realtime_request_source = RequestSource(
    name="realtime_context_override",
    schema=[
        Field(name="override_fatigue", dtype=Float32),
        Field(name="override_hr_bpm", dtype=Int32),
        Field(name="request_timestamp_hour", dtype=Int32),
    ],
)


@on_demand_feature_view(
    sources=[user_realtime_features, realtime_request_source],
    schema=[
        Field(name="effective_fatigue", dtype=Float32),
        Field(name="time_of_day_sin", dtype=Float32),
        Field(name="time_of_day_cos", dtype=Float32),
        Field(name="is_peak_hour", dtype=Bool),
    ],
)
def user_context_features(inputs: pd.DataFrame) -> pd.DataFrame:
    """
    On-demand transform — runs at serving time, zero additional I/O.
    Merges real-time stream features with optional client-side overrides.
    Computes time-of-day cyclical encoding to capture circadian patterns.
    """
    import numpy as np

    df = pd.DataFrame()

    # Use client override if provided; fall back to Kafka stream value
    df["effective_fatigue"] = inputs.get(
        "override_fatigue", inputs.get("fatigue_latest", pd.Series(dtype=float))
    ).fillna(0.3).astype("float32")

    # Cyclical encoding of hour (avoids 23→0 discontinuity)
    hour = inputs["request_timestamp_hour"].fillna(12).astype(float)
    df["time_of_day_sin"] = np.sin(2 * np.pi * hour / 24).astype("float32")
    df["time_of_day_cos"] = np.cos(2 * np.pi * hour / 24).astype("float32")

    # Peak hours: 6–9 AM and 5–8 PM
    df["is_peak_hour"] = hour.apply(
        lambda h: (6 <= h <= 9) or (17 <= h <= 20)
    )

    return df


# Workaround: feast decorator syntax varies across versions
try:
    from feast import on_demand_feature_view
except ImportError:
    pass


# ── Feature Service (Serving Bundle) ─────────────────────────────────────────
from feast import FeatureService

ranking_feature_service = FeatureService(
    name="ranking_feature_service",
    features=[
        user_30d_features,
        user_realtime_features,
        content_global_stats,
    ],
    description=(
        "All features required by the Stage-2 DeepFM ranking model. "
        "Used for both training dataset generation and online serving."
    ),
    tags={"model": "deepfm_v1", "sla_ms": "5"},
)
