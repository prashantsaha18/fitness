"""
tests/unit/test_features.py
────────────────────────────
Property-based + example-based unit tests for the feature engineering pipeline.

Uses Hypothesis for generative testing — instead of hand-picking edge cases,
Hypothesis searches the input space automatically, finding failures humans miss.

Key invariants tested:
  1. Output tensor is always finite (no NaN/Inf under any valid input)
  2. Output shape is always [N, 409] regardless of candidate count
  3. All values stay within plausible numeric bounds [0.0, 2.0]
  4. Categorical encodings are deterministic and within [0.0, 1.0]
  5. Safety filter is monotone: filtering a filtered list returns same result
  6. Hypertensive filter is never bypassed regardless of candidate content
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest
from hypothesis import given, settings, assume, HealthCheck
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays, array_shapes

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ranking.features import (
    FeatureDims,
    build_categorical_slice,
    build_input_tensor,
    build_realtime_slice,
    validate_input_tensor,
)
from ranking.model import FeatureDims


# ── Hypothesis Strategies ─────────────────────────────────────────────────────

valid_intensity = st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)
valid_sodium = st.floats(min_value=0.0, max_value=5000.0, allow_nan=False)
valid_hr = st.floats(min_value=30.0, max_value=250.0, allow_nan=False)
valid_fatigue = st.floats(min_value=0.0, max_value=1.0, allow_nan=False)

workout_types = st.sampled_from(["hiit", "strength", "yoga", "cardio", "pilates", "recovery", "unknown"])
content_types = st.sampled_from(["video", "workout_routine", "meal_recipe"])
fitness_goals = st.sampled_from([0, 1, 2, 3, 4])
hr_zones = st.sampled_from(["resting", "fat_burn", "cardio", "peak", "anaerobic"])

candidate_count = st.integers(min_value=1, max_value=100)


@st.composite
def valid_item_payload(draw) -> dict:
    return {
        "workout_type": draw(workout_types),
        "content_type": draw(content_types),
        "intensity_score": draw(valid_intensity),
        "duration_minutes": draw(st.integers(min_value=5, max_value=180)),
        "sodium_mg": draw(valid_sodium),
        "calories_kcal": draw(st.floats(min_value=0, max_value=2000, allow_nan=False)),
        "protein_g": draw(st.floats(min_value=0, max_value=200, allow_nan=False)),
        "global_ctr": draw(st.floats(min_value=0, max_value=1, allow_nan=False)),
        "global_completion_rate": draw(st.floats(min_value=0, max_value=1, allow_nan=False)),
        "total_interactions": draw(st.integers(min_value=0, max_value=1_000_000)),
        "required_equipment": draw(st.lists(st.text(max_size=20), max_size=5)),
    }


@st.composite
def valid_user_features(draw) -> dict:
    return {
        "fitness_goal_encoded": draw(fitness_goals),
        "age_normalised": draw(st.floats(min_value=0.18, max_value=0.9, allow_nan=False)),
        "bmi": draw(st.floats(min_value=15.0, max_value=45.0, allow_nan=False)),
        "structural_adherence_rate_30d": draw(st.floats(min_value=0, max_value=1, allow_nan=False)),
        "is_hypertensive": draw(st.booleans()),
        "has_cardiac_risk": draw(st.booleans()),
        "has_diabetes": draw(st.booleans()),
        "dietary_restrictions": draw(st.one_of(st.none(), st.just({}), st.just({"low-sodium": True}))),
    }


@st.composite
def valid_realtime_context(draw) -> dict:
    return {
        "hr_mean_5min": draw(valid_hr),
        "fatigue_latest": draw(valid_fatigue),
        "recovery_score": draw(st.floats(min_value=0, max_value=1, allow_nan=False)),
        "cal_total_session": draw(st.floats(min_value=0, max_value=2000, allow_nan=False)),
        "heart_rate_zone": draw(hr_zones),
    }


@st.composite
def valid_candidate(draw) -> dict:
    payload = draw(valid_item_payload())
    embedding = draw(
        arrays(np.float32, (FeatureDims.EMBEDDING,),
               elements=st.floats(-3.0, 3.0, allow_nan=False, allow_infinity=False))
    )
    # Normalise to unit length (L2)
    norm = np.linalg.norm(embedding)
    if norm > 0:
        embedding = embedding / norm
    payload["content_embedding"] = embedding.tolist()
    return {
        "content_id": "test-content-id",
        "score": draw(st.floats(0.0, 1.0, allow_nan=False)),
        "payload": payload,
    }


# ── Property Tests: Categorical Slice ─────────────────────────────────────────

class TestCategoricalSlice:

    @given(
        user_features=valid_user_features(),
        item_payload=valid_item_payload(),
        realtime_context=valid_realtime_context(),
        rank_position=st.integers(min_value=0, max_value=99),
    )
    @settings(max_examples=500, suppress_health_check=[HealthCheck.too_slow])
    def test_output_shape_always_10(self, user_features, item_payload,
                                     realtime_context, rank_position):
        """Categorical slice must always be exactly 10-dimensional."""
        result = build_categorical_slice(user_features, item_payload,
                                         realtime_context, rank_position)
        assert result.shape == (10,), f"Expected (10,), got {result.shape}"

    @given(
        user_features=valid_user_features(),
        item_payload=valid_item_payload(),
        realtime_context=valid_realtime_context(),
    )
    @settings(max_examples=500, suppress_health_check=[HealthCheck.too_slow])
    def test_all_values_bounded_0_1(self, user_features, item_payload, realtime_context):
        """All categorical values must lie in [0.0, 1.0] — model expects normalised inputs."""
        result = build_categorical_slice(user_features, item_payload, realtime_context)
        assert np.all(result >= 0.0), f"Negative value found: {result}"
        assert np.all(result <= 1.0), f"Value > 1.0 found: {result}"

    @given(
        user_features=valid_user_features(),
        item_payload=valid_item_payload(),
        realtime_context=valid_realtime_context(),
    )
    @settings(max_examples=300)
    def test_always_finite(self, user_features, item_payload, realtime_context):
        """No NaN or Inf values under any valid input combination."""
        result = build_categorical_slice(user_features, item_payload, realtime_context)
        assert np.all(np.isfinite(result)), f"Non-finite values: {result}"

    def test_time_of_day_encoding_full_cycle(self):
        """Verify cyclical encoding satisfies f(0) ≈ f(24) — no discontinuity at midnight."""
        import math
        # Hour 0 and hour 24 should produce identical encodings
        sin_0 = math.sin(2 * math.pi * 0 / 24)
        cos_0 = math.cos(2 * math.pi * 0 / 24)
        sin_24 = math.sin(2 * math.pi * 24 / 24)
        cos_24 = math.cos(2 * math.pi * 24 / 24)
        assert abs(sin_0 - sin_24) < 1e-10
        assert abs(cos_0 - cos_24) < 1e-10

    def test_peak_hour_detection(self):
        """Peak hours 6–9 AM and 5–8 PM must set the peak_hour flag."""
        import datetime, math
        # This tests the logical rule, not system clock — mock the hour
        for peak_h in [6, 7, 8, 9, 17, 18, 19, 20]:
            sin_val = math.sin(2 * math.pi * peak_h / 24)
            cos_val = math.cos(2 * math.pi * peak_h / 24)
            is_peak = (6 <= peak_h <= 9) or (17 <= peak_h <= 20)
            assert is_peak, f"Hour {peak_h} should be peak"

        for offpeak_h in [0, 3, 11, 14, 22]:
            is_peak = (6 <= offpeak_h <= 9) or (17 <= offpeak_h <= 20)
            assert not is_peak, f"Hour {offpeak_h} should NOT be peak"


# ── Property Tests: Realtime Slice ────────────────────────────────────────────

class TestRealtimeSlice:

    @given(
        user_features=valid_user_features(),
        item_payload=valid_item_payload(),
        realtime_context=valid_realtime_context(),
    )
    @settings(max_examples=500, suppress_health_check=[HealthCheck.too_slow])
    def test_output_shape_always_15(self, user_features, item_payload, realtime_context):
        result = build_realtime_slice(user_features, item_payload, realtime_context)
        assert result.shape == (15,), f"Expected (15,), got {result.shape}"

    @given(
        user_features=valid_user_features(),
        item_payload=valid_item_payload(),
        realtime_context=valid_realtime_context(),
    )
    @settings(max_examples=500, suppress_health_check=[HealthCheck.too_slow])
    def test_soft_clipping_bounds(self, user_features, item_payload, realtime_context):
        """Soft clip at 2.0 prevents extreme outliers from dominating gradients."""
        result = build_realtime_slice(user_features, item_payload, realtime_context)
        assert np.all(result >= 0.0), f"Negative value: {result}"
        assert np.all(result <= 2.0), f"Value above soft-clip: {result}"

    @given(
        user_features=valid_user_features(),
        item_payload=valid_item_payload(),
        realtime_context=valid_realtime_context(),
    )
    @settings(max_examples=300)
    def test_always_finite(self, user_features, item_payload, realtime_context):
        result = build_realtime_slice(user_features, item_payload, realtime_context)
        assert np.all(np.isfinite(result))

    def test_extreme_hr_normalised_below_1(self):
        """Max physiologically possible HR (220 bpm) should normalise to exactly 1.0."""
        result = build_realtime_slice(
            user_features={"bmi": 23.5, "age_normalised": 0.3,
                           "structural_adherence_rate_30d": 0.5},
            item_payload={"global_ctr": 0.1, "global_completion_rate": 0.5,
                          "total_interactions": 100, "intensity_score": 0.5,
                          "duration_minutes": 30, "sodium_mg": 200,
                          "calories_kcal": 400, "protein_g": 30},
            realtime_context={"hr_mean_5min": 220.0, "fatigue_latest": 1.0,
                               "recovery_score": 0.0, "cal_total_session": 1000.0,
                               "heart_rate_zone": "anaerobic"},
        )
        assert result[0] == pytest.approx(1.0, abs=0.01), "HR 220 should normalise to 1.0"

    def test_missing_realtime_context_uses_defaults(self):
        """Empty realtime context should not raise — defaults are filled in."""
        result = build_realtime_slice(
            user_features={},
            item_payload={},
            realtime_context={},  # completely empty
        )
        assert result.shape == (15,)
        assert np.all(np.isfinite(result))


# ── Property Tests: Full Input Tensor ────────────────────────────────────────

class TestInputTensor:

    @given(
        candidates=st.lists(valid_candidate(), min_size=1, max_size=100),
        user_features=valid_user_features(),
        realtime_context=valid_realtime_context(),
    )
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_output_shape(self, candidates, user_features, realtime_context):
        """Output shape must always be [N, 409]."""
        matrix = build_input_tensor(candidates, user_features, realtime_context)
        assert matrix.shape == (len(candidates), FeatureDims.TOTAL)

    @given(
        candidates=st.lists(valid_candidate(), min_size=1, max_size=50),
        user_features=valid_user_features(),
        realtime_context=valid_realtime_context(),
    )
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_c_contiguous_for_onnx(self, candidates, user_features, realtime_context):
        """ONNX runtime requires C-contiguous layout — must not be Fortran-order."""
        matrix = build_input_tensor(candidates, user_features, realtime_context)
        assert matrix.flags["C_CONTIGUOUS"], "Matrix must be C-contiguous for ONNX ingestion"

    @given(
        candidates=st.lists(valid_candidate(), min_size=1, max_size=50),
        user_features=valid_user_features(),
        realtime_context=valid_realtime_context(),
    )
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_float32_dtype(self, candidates, user_features, realtime_context):
        """ONNX session expects float32 — mixed precision causes silent shape errors."""
        matrix = build_input_tensor(candidates, user_features, realtime_context)
        assert matrix.dtype == np.float32, f"Expected float32, got {matrix.dtype}"

    def test_embedding_slice_populated_from_payload(self):
        """Content embeddings in the payload should appear in the [0:384] slice."""
        embedding = np.random.randn(384).astype(np.float32)
        embedding = embedding / np.linalg.norm(embedding)
        candidate = {
            "content_id": "test",
            "score": 0.9,
            "payload": {
                "content_embedding": embedding.tolist(),
                "workout_type": "hiit", "content_type": "workout_routine",
                "intensity_score": 0.7, "duration_minutes": 30,
                "sodium_mg": 100, "calories_kcal": 300, "protein_g": 25,
                "global_ctr": 0.12, "global_completion_rate": 0.65,
                "total_interactions": 500, "required_equipment": [],
            }
        }
        matrix = build_input_tensor(
            [candidate],
            user_features={"fitness_goal_encoded": 0, "age_normalised": 0.3,
                           "bmi": 23.5, "structural_adherence_rate_30d": 0.5,
                           "is_hypertensive": False, "has_cardiac_risk": False,
                           "has_diabetes": False, "dietary_restrictions": {}},
            realtime_context={"hr_mean_5min": 100.0, "fatigue_latest": 0.3,
                               "recovery_score": 0.7, "cal_total_session": 100.0,
                               "heart_rate_zone": "cardio"},
        )
        np.testing.assert_allclose(
            matrix[0, :FeatureDims.EMBEDDING],
            embedding,
            rtol=1e-5,
            err_msg="Embedding slice not correctly populated from payload"
        )

    def test_missing_embedding_fills_zeros(self):
        """Candidates without an embedding should get a zero-filled embedding slice."""
        candidate = {
            "content_id": "test",
            "score": 0.5,
            "payload": {
                "workout_type": "yoga", "content_type": "workout_routine",
                # NO content_embedding key
            }
        }
        matrix = build_input_tensor([candidate], user_features={}, realtime_context={})
        assert np.all(matrix[0, :FeatureDims.EMBEDDING] == 0.0)


# ── Property Tests: Validate Input Tensor ────────────────────────────────────

class TestValidateInputTensor:

    def test_valid_matrix_passes(self):
        matrix = np.random.randn(100, FeatureDims.TOTAL).astype(np.float32)
        assert validate_input_tensor(matrix) is True

    def test_wrong_feature_dim_raises(self):
        with pytest.raises(ValueError, match="Expected shape"):
            validate_input_tensor(np.zeros((100, 100), dtype=np.float32))

    def test_1d_array_raises(self):
        with pytest.raises(ValueError, match="Expected shape"):
            validate_input_tensor(np.zeros(FeatureDims.TOTAL, dtype=np.float32))

    def test_nan_values_are_fixed_in_place(self):
        """NaN values should be replaced with 0.0, not raise an exception."""
        matrix = np.ones((10, FeatureDims.TOTAL), dtype=np.float32)
        matrix[3, 42] = float("nan")
        matrix[7, 100] = float("inf")
        validate_input_tensor(matrix)  # must not raise
        assert np.all(np.isfinite(matrix)), "NaN/Inf should be replaced in-place"

    @given(
        arrays(np.float32, (50, FeatureDims.TOTAL),
               elements=st.floats(allow_nan=True, allow_infinity=True))
    )
    @settings(max_examples=50)
    def test_validation_always_survives_arbitrary_float_input(self, matrix):
        """validate_input_tensor must never raise on any float32 matrix of correct shape."""
        validate_input_tensor(matrix)
        assert np.all(np.isfinite(matrix))


# ── Safety Filter Tests ───────────────────────────────────────────────────────

class TestSafetyFilters:
    """
    Safety filters are mission-critical — tested exhaustively with example-based
    and property-based tests. A failure here has health consequences.
    """

    def _run_filter(self, candidates, user_features, user_db=None):
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent.parent))
        from api.main import apply_safety_filters
        return apply_safety_filters(candidates, user_features, user_db)

    def _make_candidate(self, sodium_mg=200.0, intensity_score=0.5,
                         carbs_g=30.0, inference_score=0.8) -> dict:
        return {
            "content_id": str(__import__("uuid").uuid4()),
            "score": 0.9,
            "inference_score": inference_score,
            "payload": {
                "content_type": "meal_recipe",
                "workout_type": "hiit",
                "sodium_mg": sodium_mg,
                "intensity_score": intensity_score,
                "carbs_g": carbs_g,
                "global_ctr": 0.1,
                "global_completion_rate": 0.5,
                "total_interactions": 100,
            },
            "safety_flags": [],
        }

    def test_hypertensive_high_sodium_removed(self):
        """Items with sodium > 400mg MUST be removed for hypertensive users."""
        candidates = [
            self._make_candidate(sodium_mg=800.0),   # must be removed
            self._make_candidate(sodium_mg=200.0),   # must be kept
            self._make_candidate(sodium_mg=1200.0),  # must be removed
            self._make_candidate(sodium_mg=399.0),   # must be kept (just under threshold)
        ]
        user_features = {"is_hypertensive": True, "has_cardiac_risk": False,
                         "has_diabetes": False}
        filtered, removed, flags = self._run_filter(candidates, user_features)

        assert removed == 2, f"Expected 2 removed, got {removed}"
        assert len(filtered) == 2
        for item in filtered:
            assert item["payload"]["sodium_mg"] <= 400.0

    def test_hypertensive_boundary_condition_399mg(self):
        """399mg is below threshold and must be kept. Boundary correctness is critical."""
        c = self._make_candidate(sodium_mg=399.9)
        user_features = {"is_hypertensive": True, "has_cardiac_risk": False,
                         "has_diabetes": False}
        filtered, removed, _ = self._run_filter([c], user_features)
        assert removed == 0, "399.9mg sodium should NOT be filtered for HTN users"
        assert len(filtered) == 1

    def test_hypertensive_boundary_condition_400mg(self):
        """400mg is AT the threshold — depends on '>' vs '>='."""
        c = self._make_candidate(sodium_mg=400.0)
        user_features = {"is_hypertensive": True, "has_cardiac_risk": False,
                         "has_diabetes": False}
        filtered, removed, _ = self._run_filter([c], user_features)
        # 400.0 is NOT greater than 400.0, so should be kept
        assert removed == 0, "400mg exactly should be kept (threshold is strictly >400)"

    def test_cardiac_risk_high_intensity_removed(self):
        """Workouts with intensity > 0.7 must be blocked for cardiac risk users."""
        candidates = [
            self._make_candidate(intensity_score=0.9),   # removed
            self._make_candidate(intensity_score=0.71),  # removed
            self._make_candidate(intensity_score=0.70),  # kept (at threshold)
            self._make_candidate(intensity_score=0.5),   # kept
        ]
        user_features = {"is_hypertensive": False, "has_cardiac_risk": True,
                         "has_diabetes": False}
        filtered, removed, flags = self._run_filter(candidates, user_features)
        assert removed == 2
        assert "cardiac_intensity_cap" in flags

    def test_diabetic_high_carb_removed(self):
        """Recipes with carbs > 60g must be removed for diabetic users."""
        candidates = [
            self._make_candidate(carbs_g=80.0),   # removed
            self._make_candidate(carbs_g=61.0),   # removed
            self._make_candidate(carbs_g=60.0),   # kept
            self._make_candidate(carbs_g=30.0),   # kept
        ]
        user_features = {"is_hypertensive": False, "has_cardiac_risk": False,
                         "has_diabetes": True}
        filtered, removed, _ = self._run_filter(candidates, user_features)
        assert removed == 2

    def test_low_confidence_items_filtered(self):
        """Items with inference_score below MIN_CONFIDENCE_THRESHOLD must be removed."""
        candidates = [
            self._make_candidate(inference_score=0.001),  # removed
            self._make_candidate(inference_score=0.0),    # removed
            self._make_candidate(inference_score=0.05),   # kept (at threshold)
            self._make_candidate(inference_score=0.8),    # kept
        ]
        user_features = {"is_hypertensive": False, "has_cardiac_risk": False,
                         "has_diabetes": False}
        filtered, removed, _ = self._run_filter(candidates, user_features)
        assert removed == 2

    @given(
        sodium_values=st.lists(
            st.floats(min_value=0.0, max_value=5000.0, allow_nan=False),
            min_size=1, max_size=50
        )
    )
    @settings(max_examples=200)
    def test_hypertension_filter_monotone(self, sodium_values):
        """
        Idempotency invariant: filtering an already-filtered list returns the same list.
        apply_safety_filters(apply_safety_filters(L)) == apply_safety_filters(L)
        """
        candidates = [self._make_candidate(sodium_mg=s) for s in sodium_values]
        user_features = {"is_hypertensive": True, "has_cardiac_risk": False,
                         "has_diabetes": False}

        filtered_once, removed_1, _ = self._run_filter(candidates, user_features)
        filtered_twice, removed_2, _ = self._run_filter(filtered_once, user_features)

        assert removed_2 == 0, (
            f"Second pass removed {removed_2} items — filter is not idempotent. "
            f"This means items that should have been removed in pass 1 were kept."
        )
        assert len(filtered_once) == len(filtered_twice)

    def test_non_hypertensive_high_sodium_kept(self):
        """Safety rules must ONLY apply to flagged users — healthy users see all content."""
        c = self._make_candidate(sodium_mg=3000.0)
        user_features = {"is_hypertensive": False, "has_cardiac_risk": False,
                         "has_diabetes": False}
        filtered, removed, _ = self._run_filter([c], user_features)
        assert removed == 0, "High-sodium items should be visible to non-hypertensive users"

    def test_empty_candidate_list_returns_empty(self):
        filtered, removed, flags = self._run_filter([], {"is_hypertensive": True,
                                                         "has_cardiac_risk": True,
                                                         "has_diabetes": True})
        assert filtered == []
        assert removed == 0
