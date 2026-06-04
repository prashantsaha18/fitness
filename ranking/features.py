"""
ranking/features.py
────────────────────
Dynamic feature engineering for Stage-2 ranking input tensor construction.

Input tensor layout ([batch, 409]):
  ┌─────────────────────────────────────────────────────────────────────┐
  │ Slice       │ Dim  │ Source                │ Description             │
  ├─────────────┼──────┼───────────────────────┼─────────────────────────┤
  │ [0:384]     │ 384  │ Qdrant payload         │ Item content embedding  │
  │ [384:394]   │ 10   │ Feast batch features   │ Sparse categoricals     │
  │ [394:409]   │ 15   │ Redis realtime         │ Biometric + engagement  │
  └─────────────────────────────────────────────────────────────────────┘

Categorical encoding map (indices 384–393):
  [384] workout_type_encoded      (int, 0–6, label-encoded)
  [385] content_type_encoded      (int, 0–2)
  [386] fitness_goal_encoded      (int, 0–4)
  [387] heart_rate_zone_encoded   (int, 0–4)
  [388] rank_position_normalised  (float, 0–1, position bias correction)
  [389] has_dietary_restriction   (binary flag)
  [390] is_equipment_free         (binary flag)
  [391] time_of_day_sin           (cyclical float)
  [392] time_of_day_cos           (cyclical float)
  [393] is_peak_hour              (binary flag)

Realtime numerical features (indices 394–408):
  [394] hr_mean_5min_normalised   (HR / 220.0)
  [395] fatigue_latest            (0–1)
  [396] recovery_score            (0–1)
  [397] cal_total_session_norm    (cal / 1000.0)
  [398] global_ctr                (0–1)
  [399] global_completion_rate    (0–1)
  [400] total_interactions_log    (log1p(count) / 20.0)
  [401] intensity_score           (0–1)
  [402] duration_minutes_norm     (min / 120.0)
  [403] sodium_norm               (sodium_mg / 3000.0)
  [404] calories_norm             (kcal / 1000.0)
  [405] protein_norm              (g / 200.0)
  [406] bmi_normalised            (bmi / 40.0)
  [407] age_normalised            (age / 100.0)
  [408] adherence_rate_30d        (0–1)
"""
from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any

import numpy as np

from ranking.model import FeatureDims


# ── Categorical Encoding Maps ─────────────────────────────────────────────────

WORKOUT_TYPE_MAP: dict[str, int] = {
    "hiit": 0, "strength": 1, "yoga": 2, "cardio": 3,
    "pilates": 4, "recovery": 5, "unknown": 6,
}
CONTENT_TYPE_MAP: dict[str, int] = {
    "video": 0, "workout_routine": 1, "meal_recipe": 2,
}
FITNESS_GOAL_MAP: dict[str, int] = {
    "weight_loss": 0, "muscle_gain": 1, "endurance": 2,
    "flexibility": 3, "maintenance": 4,
}
HR_ZONE_MAP: dict[str, int] = {
    "resting": 0, "fat_burn": 1, "cardio": 2, "peak": 3, "anaerobic": 4,
}


def _safe_get(d: dict, key: str, default: float = 0.0) -> float:
    val = d.get(key, default)
    return float(val) if val is not None else default


# ── Feature Vector Constructors ───────────────────────────────────────────────

def build_categorical_slice(
    user_features: dict[str, Any],
    item_payload: dict[str, Any],
    realtime_context: dict[str, Any],
    rank_position: int = 0,
    max_rank: int = 100,
) -> np.ndarray:
    """
    Construct the 10-dimensional categorical feature slice.
    All values normalised to [0, 1] range for gradient stability.
    """
    now = datetime.now(timezone.utc)
    hour = now.hour

    vec = np.zeros(10, dtype=np.float32)

    # [384] workout_type
    wt = item_payload.get("workout_type", "unknown")
    vec[0] = WORKOUT_TYPE_MAP.get(wt, 6) / 6.0

    # [385] content_type
    ct = item_payload.get("content_type", "video")
    vec[1] = CONTENT_TYPE_MAP.get(ct, 0) / 2.0

    # [386] fitness_goal
    goal = user_features.get("fitness_goal_encoded", 0)
    vec[2] = int(goal) / 4.0

    # [387] heart_rate_zone
    hr_zone = realtime_context.get("heart_rate_zone", "resting")
    vec[3] = HR_ZONE_MAP.get(hr_zone, 0) / 4.0

    # [388] rank_position (position bias feature)
    vec[4] = rank_position / max(max_rank, 1)

    # [389] has_dietary_restriction (user)
    vec[5] = 1.0 if user_features.get("dietary_restrictions") else 0.0

    # [390] is_equipment_free
    equipment = item_payload.get("required_equipment", [])
    vec[6] = 1.0 if not equipment or equipment == ["none"] else 0.0

    # [391–392] time-of-day cyclical encoding
    vec[7] = math.sin(2 * math.pi * hour / 24)
    vec[8] = math.cos(2 * math.pi * hour / 24)

    # [393] is_peak_hour
    vec[9] = 1.0 if (6 <= hour <= 9) or (17 <= hour <= 20) else 0.0

    return vec


def build_realtime_slice(
    user_features: dict[str, Any],
    item_payload: dict[str, Any],
    realtime_context: dict[str, Any],
) -> np.ndarray:
    """
    Construct the 15-dimensional realtime numerical feature slice.
    Normalisation denominators chosen to keep 99th percentile values ≤ 1.0.
    """
    vec = np.zeros(15, dtype=np.float32)

    # Biometric features
    vec[0] = _safe_get(realtime_context, "hr_mean_5min", 70.0) / 220.0
    vec[1] = _safe_get(realtime_context, "fatigue_latest", 0.3)
    vec[2] = _safe_get(realtime_context, "recovery_score", 0.7)
    vec[3] = _safe_get(realtime_context, "cal_total_session", 0.0) / 1000.0

    # Content engagement stats
    vec[4] = _safe_get(item_payload, "global_ctr", 0.1)
    vec[5] = _safe_get(item_payload, "global_completion_rate", 0.5)
    vec[6] = math.log1p(_safe_get(item_payload, "total_interactions", 0)) / 20.0

    # Content physical attributes
    vec[7] = _safe_get(item_payload, "intensity_score", 0.5)
    vec[8] = _safe_get(item_payload, "duration_minutes", 30.0) / 120.0
    vec[9] = _safe_get(item_payload, "sodium_mg", 0.0) / 3000.0
    vec[10] = _safe_get(item_payload, "calories_kcal", 0.0) / 1000.0
    vec[11] = _safe_get(item_payload, "protein_g", 0.0) / 200.0

    # User attributes
    bmi = _safe_get(user_features, "bmi", 22.0)
    vec[12] = bmi / 40.0
    vec[13] = _safe_get(user_features, "age_normalised", 0.3)
    vec[14] = _safe_get(user_features, "structural_adherence_rate_30d", 0.5)

    return np.clip(vec, 0.0, 2.0)  # soft-clip to handle outliers


# ── Main Tensor Builder ───────────────────────────────────────────────────────

def build_input_tensor(
    candidates: list[dict[str, Any]],
    user_features: dict[str, Any],
    realtime_context: dict[str, Any],
) -> np.ndarray:
    """
    Construct the full [N, 409] feature matrix for N candidates.

    This is the critical path function called once per API request.
    Complexity: O(N × D) where N ≤ 100 candidates, D = 409 features.
    At N=100, D=409: 40,900 float32 operations ≈ negligible CPU time.

    Args:
        candidates: List of dicts from Stage-1 retrieval (includes payload + embedding).
        user_features: Dict of user features from Feast online store.
        realtime_context: Dict of current biometric state from Redis.

    Returns:
        np.ndarray of shape [N, 409], dtype=float32, C-contiguous layout.
        C-contiguous ensures ONNX runtime can ingest without additional memcpy.
    """
    n = len(candidates)
    matrix = np.zeros((n, FeatureDims.TOTAL), dtype=np.float32)

    for i, candidate in enumerate(candidates):
        payload = candidate.get("payload", {})

        # ── Dense embedding slice [0:384] ─────────────────────────────────
        emb = payload.get("content_embedding")
        if emb is not None:
            arr = np.asarray(emb, dtype=np.float32)
            if arr.shape == (FeatureDims.EMBEDDING,):
                matrix[i, : FeatureDims.EMBEDDING] = arr
            # else: leave zero-filled (embedding unavailable — log in prod)

        # ── Categorical slice [384:394] ───────────────────────────────────
        cat_slice = build_categorical_slice(
            user_features=user_features,
            item_payload=payload,
            realtime_context=realtime_context,
            rank_position=i,
        )
        matrix[i, FeatureDims.EMBEDDING : FeatureDims.EMBEDDING + FeatureDims.CATEGORICAL] = cat_slice

        # ── Realtime slice [394:409] ──────────────────────────────────────
        rt_slice = build_realtime_slice(
            user_features=user_features,
            item_payload=payload,
            realtime_context=realtime_context,
        )
        matrix[i, FeatureDims.EMBEDDING + FeatureDims.CATEGORICAL :] = rt_slice

    # Ensure C-contiguous layout for ONNX session.run()
    return np.ascontiguousarray(matrix)


# ── Validation ────────────────────────────────────────────────────────────────

def validate_input_tensor(matrix: np.ndarray) -> bool:
    """
    Fast pre-inference sanity check. Raises ValueError on critical issues.
    Called before every ONNX session.run() invocation.
    """
    if matrix.ndim != 2 or matrix.shape[1] != FeatureDims.TOTAL:
        raise ValueError(
            f"Expected shape [N, {FeatureDims.TOTAL}], got {matrix.shape}"
        )
    if not np.isfinite(matrix).all():
        # Replace NaN/Inf in-place rather than failing the entire request
        n_invalid = (~np.isfinite(matrix)).sum()
        np.nan_to_num(matrix, nan=0.0, posinf=1.0, neginf=0.0, copy=False)
        import logging
        logging.getLogger(__name__).warning(
            "Fixed %d non-finite values in feature matrix", n_invalid
        )
    return True
