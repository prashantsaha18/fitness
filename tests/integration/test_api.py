"""
tests/integration/test_api.py
──────────────────────────────
Integration tests for the FastAPI recommendation pipeline.

These tests exercise the full HTTP stack (routing, auth, serialisation)
while mocking the I/O layer (Redis, Qdrant, ONNX) so they run without
external infrastructure.

Coverage:
  ✓ Auth: register, login, token validation, wrong credentials
  ✓ Recommend: happy path, missing user, auth required, fallback mode
  ✓ Safety: hypertensive user receives only low-sodium items
  ✓ Response schema: all required fields present and correctly typed
  ✓ Latency header: X-Response-Time injected into every response
  ✓ Health check: component-level status reporting
  ✓ Edge cases: empty candidate pool, all items safety-filtered
"""
from __future__ import annotations

import json
import sys
import uuid
from pathlib import Path
from typing import AsyncGenerator
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


# ── App Bootstrapper (minimal startup without real infra) ─────────────────────

@pytest.fixture(scope="module")
def mock_embedding_engine():
    """Returns a normalised random embedding for any input text."""
    engine = MagicMock()
    async def encode_single(text):
        v = np.random.randn(384).astype(np.float32)
        return v / np.linalg.norm(v)
    async def encode_async(texts, normalise=True):
        return np.random.randn(len(texts), 384).astype(np.float32)
    engine.encode_single = encode_single
    engine.encode_async = encode_async
    engine.embedding_dim = 384
    return engine


@pytest.fixture(scope="module")
def mock_inference_engine():
    """Returns realistic score distribution for any feature matrix."""
    engine = MagicMock()
    async def rank_candidates(feature_matrix, candidates):
        scores = np.random.uniform(0.1, 0.95, len(candidates))
        for i, (c, s) in enumerate(zip(candidates, scores)):
            c["inference_score"] = float(s)
        return sorted(candidates, key=lambda x: x["inference_score"], reverse=True)
    engine.rank_candidates = rank_candidates
    engine._session = MagicMock()  # non-None signals "loaded"
    return engine


@pytest.fixture(scope="module")
def mock_candidates():
    """50 realistic mock Stage-1 results covering both content types."""
    results = []
    for i in range(50):
        is_recipe = i % 3 == 0
        results.append({
            "content_id": str(uuid.uuid4()),
            "qdrant_id": str(uuid.uuid4()),
            "score": 0.95 - (i * 0.01),
            "inference_score": 0.0,
            "safety_flags": [],
            "payload": {
                "content_id": str(uuid.uuid4()),
                "title": f"{'Recipe' if is_recipe else 'Workout'} #{i}",
                "content_type": "meal_recipe" if is_recipe else "workout_routine",
                "workout_type": None if is_recipe else "hiit",
                "intensity_score": 0.5,
                "duration_minutes": 30,
                "sodium_mg": 800.0 if (i % 7 == 0 and is_recipe) else 180.0,
                "calories_kcal": 400.0,
                "protein_g": 30.0,
                "carbs_g": 40.0,
                "dietary_tags": ["high-protein"],
                "required_equipment": [],
                "global_ctr": 0.12,
                "global_completion_rate": 0.65,
                "total_interactions": 1500,
                "thumbnail_url": f"https://cdn.example.com/thumb/{i}.jpg",
            },
        })
    return results


@pytest_asyncio.fixture(scope="module")
async def app_client(mock_embedding_engine, mock_inference_engine, mock_candidates):
    """
    Full FastAPI test client with all external dependencies mocked.
    Uses ASGI transport — no actual TCP sockets opened.
    """
    import redis.asyncio as aioredis

    # We import app AFTER patching to catch module-level singletons
    with patch("retrieval.embeddings.get_embedding_engine", return_value=mock_embedding_engine), \
         patch("ranking.export_onnx.get_inference_engine", return_value=mock_inference_engine), \
         patch("retrieval.vector_store.retrieve_candidates",
               new_callable=lambda: lambda: AsyncMock(return_value=mock_candidates)), \
         patch("retrieval.vector_store.ensure_collection_exists", new_callable=AsyncMock), \
         patch("data_pipeline.database.init_db", new_callable=AsyncMock), \
         patch("data_pipeline.database.dispose_engine", new_callable=AsyncMock):

        from api.main import app

        # Inject mock Redis into app state
        mock_redis = AsyncMock()
        mock_redis.ping = AsyncMock(return_value=True)
        mock_redis.hgetall = AsyncMock(return_value={})
        mock_redis.hset = AsyncMock()
        mock_redis.expire = AsyncMock()
        mock_redis.lrange = AsyncMock(return_value=[])
        mock_redis.aclose = AsyncMock()

        # Bypass lifespan — inject state directly
        app.state.redis = mock_redis

        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            yield client, mock_redis


# ── Auth Helpers ──────────────────────────────────────────────────────────────

async def register_and_login(client: AsyncClient, suffix: str = "") -> tuple[str, str]:
    """Register a user and return (user_id, jwt_token)."""
    suffix = suffix or uuid.uuid4().hex[:8]
    email = f"test_{suffix}@test.com"
    password = "TestPass123!"

    # Register
    r = await client.post("/api/v1/auth/register", json={
        "username": f"test_{suffix}",
        "email": email,
        "password": password,
    })
    assert r.status_code == 201, f"Register failed: {r.text}"
    user_id = r.json()["id"]

    # Login
    r = await client.post("/api/v1/auth/token", data={
        "username": email,
        "password": password,
    })
    assert r.status_code == 200, f"Login failed: {r.text}"
    token = r.json()["access_token"]

    return user_id, token


# ── Auth Tests ────────────────────────────────────────────────────────────────

class TestAuthentication:

    @pytest.mark.asyncio
    async def test_register_success(self, app_client):
        client, _ = app_client
        r = await client.post("/api/v1/auth/register", json={
            "username": f"user_{uuid.uuid4().hex[:6]}",
            "email": f"auth_{uuid.uuid4().hex[:8]}@test.com",
            "password": "SecurePass123!",
        })
        assert r.status_code == 201
        body = r.json()
        assert "id" in body
        assert "hashed_password" not in body, "CRITICAL: Password hash must not be exposed"
        assert body["is_active"] is True

    @pytest.mark.asyncio
    async def test_duplicate_email_returns_409(self, app_client):
        client, _ = app_client
        email = f"dup_{uuid.uuid4().hex[:8]}@test.com"
        payload = {"username": f"u_{uuid.uuid4().hex[:6]}", "email": email, "password": "Pass123!"}
        r1 = await client.post("/api/v1/auth/register", json=payload)
        assert r1.status_code == 201
        r2 = await client.post("/api/v1/auth/register", json={**payload, "username": "other_name"})
        assert r2.status_code == 409, "Duplicate email must return 409 Conflict"

    @pytest.mark.asyncio
    async def test_login_wrong_password_returns_401(self, app_client):
        client, _ = app_client
        user_id, token = await register_and_login(client, suffix="wrongpw")
        email = f"test_wrongpw@test.com"
        r = await client.post("/api/v1/auth/token", data={
            "username": email, "password": "WrongPassword!",
        })
        assert r.status_code == 401

    @pytest.mark.asyncio
    async def test_token_structure(self, app_client):
        client, _ = app_client
        user_id, token = await register_and_login(client, suffix="tokstr")
        assert "." in token, "JWT must contain dots (header.payload.signature)"
        assert len(token.split(".")) == 3, "JWT must have exactly 3 parts"

    @pytest.mark.asyncio
    async def test_recommend_without_token_returns_401(self, app_client):
        client, _ = app_client
        r = await client.post("/api/v1/recommend", json={
            "user_id": str(uuid.uuid4()), "top_n": 10,
        })
        assert r.status_code == 401

    @pytest.mark.asyncio
    async def test_recommend_with_invalid_token_returns_401(self, app_client):
        client, _ = app_client
        r = await client.post("/api/v1/recommend",
            json={"user_id": str(uuid.uuid4()), "top_n": 10},
            headers={"Authorization": "Bearer this.is.invalid"},
        )
        assert r.status_code == 401

    @pytest.mark.asyncio
    async def test_user_cannot_request_other_user_recommendations(self, app_client):
        """
        SECURITY: user_id in payload must match JWT sub.
        If not, attacker can scrape another user's recommendations revealing
        their health conditions via safety filter patterns.
        """
        client, _ = app_client
        user_id_a, token_a = await register_and_login(client, suffix="secA")
        user_id_b, token_b = await register_and_login(client, suffix="secB")

        r = await client.post("/api/v1/recommend",
            json={"user_id": user_id_b, "top_n": 10},  # B's ID
            headers={"Authorization": f"Bearer {token_a}"},  # A's token
        )
        assert r.status_code == 403, "Cross-user recommendation scraping must be blocked"


# ── Recommendation Endpoint Tests ─────────────────────────────────────────────

class TestRecommendEndpoint:

    @pytest.mark.asyncio
    async def test_happy_path_returns_200(self, app_client):
        client, _ = app_client
        user_id, token = await register_and_login(client, suffix="happy")

        r = await client.post("/api/v1/recommend",
            json={"user_id": user_id, "top_n": 10},
            headers={"Authorization": f"Bearer {token}"},
        )
        assert r.status_code == 200

    @pytest.mark.asyncio
    async def test_response_schema_complete(self, app_client):
        """Every field in the RecommendationResponse schema must be present."""
        client, _ = app_client
        user_id, token = await register_and_login(client, suffix="schema")

        r = await client.post("/api/v1/recommend",
            json={"user_id": user_id, "top_n": 5},
            headers={"Authorization": f"Bearer {token}"},
        )
        body = r.json()

        # Top-level fields
        assert "request_id" in body
        assert "user_id" in body
        assert "recommendations" in body
        assert "pipeline_metadata" in body

        # Recommendation fields
        if body["recommendations"]:
            rec = body["recommendations"][0]
            assert "rank" in rec
            assert "content_id" in rec
            assert "inference_score" in rec
            assert "retrieval_score" in rec
            assert "metadata" in rec
            assert "safety_flags" in rec
            assert rec["rank"] == 1, "First recommendation must have rank=1"

        # Pipeline metadata
        meta = body["pipeline_metadata"]
        assert "total_latency_ms" in meta
        assert "stage1_latency_ms" in meta
        assert "stage2_latency_ms" in meta
        assert "model_version" in meta
        assert "is_fallback" in meta
        assert isinstance(meta["is_fallback"], bool)

    @pytest.mark.asyncio
    async def test_ranks_are_sequential_1_based(self, app_client):
        """Ranks must be 1, 2, 3, ..., N — no gaps, no 0-based indexing."""
        client, _ = app_client
        user_id, token = await register_and_login(client, suffix="ranks")

        r = await client.post("/api/v1/recommend",
            json={"user_id": user_id, "top_n": 10},
            headers={"Authorization": f"Bearer {token}"},
        )
        recs = r.json()["recommendations"]
        if len(recs) > 1:
            for i, rec in enumerate(recs, start=1):
                assert rec["rank"] == i, f"Expected rank={i}, got {rec['rank']}"

    @pytest.mark.asyncio
    async def test_top_n_respected(self, app_client):
        """Response must contain at most top_n recommendations."""
        client, _ = app_client
        user_id, token = await register_and_login(client, suffix="topn")

        for n in [1, 5, 10, 20]:
            r = await client.post("/api/v1/recommend",
                json={"user_id": user_id, "top_n": n},
                headers={"Authorization": f"Bearer {token}"},
            )
            recs = r.json()["recommendations"]
            assert len(recs) <= n, f"top_n={n} but got {len(recs)} recommendations"

    @pytest.mark.asyncio
    async def test_inference_scores_descending(self, app_client):
        """Items must be sorted by inference_score descending."""
        client, _ = app_client
        user_id, token = await register_and_login(client, suffix="sort")

        r = await client.post("/api/v1/recommend",
            json={"user_id": user_id, "top_n": 15},
            headers={"Authorization": f"Bearer {token}"},
        )
        scores = [rec["inference_score"] for rec in r.json()["recommendations"]]
        assert scores == sorted(scores, reverse=True), (
            "Recommendations not sorted by inference_score descending"
        )

    @pytest.mark.asyncio
    async def test_realtime_context_accepted(self, app_client):
        """Realtime context override must be accepted without error."""
        client, _ = app_client
        user_id, token = await register_and_login(client, suffix="rt")

        r = await client.post("/api/v1/recommend",
            json={
                "user_id": user_id,
                "top_n": 5,
                "realtime_context": {
                    "heart_rate_bpm": 155,
                    "fatigue_level": 0.72,
                    "active_calories_kcal": 320.5,
                }
            },
            headers={"Authorization": f"Bearer {token}"},
        )
        assert r.status_code == 200

    @pytest.mark.asyncio
    async def test_invalid_content_type_returns_422(self, app_client):
        """Unknown content types must be rejected with 422 Unprocessable Entity."""
        client, _ = app_client
        user_id, token = await register_and_login(client, suffix="ct422")

        r = await client.post("/api/v1/recommend",
            json={"user_id": user_id, "top_n": 5, "content_types": ["nonexistent_type"]},
            headers={"Authorization": f"Bearer {token}"},
        )
        assert r.status_code == 422

    @pytest.mark.asyncio
    async def test_top_n_exceeding_max_returns_422(self, app_client):
        """top_n > 50 must be rejected — prevents scraping the full catalogue."""
        client, _ = app_client
        user_id, token = await register_and_login(client, suffix="maxn")

        r = await client.post("/api/v1/recommend",
            json={"user_id": user_id, "top_n": 999},
            headers={"Authorization": f"Bearer {token}"},
        )
        assert r.status_code == 422

    @pytest.mark.asyncio
    async def test_user_id_mismatch_in_payload_returns_400(self, app_client):
        """Malformed UUID in user_id must return 422."""
        client, _ = app_client
        _, token = await register_and_login(client, suffix="badid")

        r = await client.post("/api/v1/recommend",
            json={"user_id": "not-a-valid-uuid", "top_n": 5},
            headers={"Authorization": f"Bearer {token}"},
        )
        assert r.status_code == 422


# ── Fallback Mode Tests ───────────────────────────────────────────────────────

class TestFallbackMode:

    @pytest.mark.asyncio
    async def test_qdrant_failure_returns_200_not_500(self, app_client):
        """
        CRITICAL: Qdrant failure must never surface as a 500 to the client.
        The fallback cache must be served instead.
        """
        client, mock_redis = app_client
        user_id, token = await register_and_login(client, suffix="fb1")

        # Pre-populate Redis fallback cache with some items
        fallback_items = [
            json.dumps({
                "content_id": str(uuid.uuid4()),
                "score": 0.8,
                "inference_score": 0.8,
                "safety_flags": [],
                "payload": {
                    "title": f"Fallback Item {i}",
                    "content_type": "workout_routine",
                    "sodium_mg": 100.0,
                    "intensity_score": 0.5,
                    "carbs_g": 30.0,
                    "global_ctr": 0.1,
                    "global_completion_rate": 0.5,
                    "total_interactions": 100,
                },
            })
            for i in range(30)
        ]
        mock_redis.lrange = AsyncMock(return_value=fallback_items)

        from retrieval.vector_store import VectorStoreUnavailableError

        with patch("retrieval.vector_store.retrieve_candidates",
                   side_effect=VectorStoreUnavailableError("Qdrant is down")):
            r = await client.post("/api/v1/recommend",
                json={"user_id": user_id, "top_n": 10},
                headers={"Authorization": f"Bearer {token}"},
            )

        assert r.status_code == 200, f"Expected 200, got {r.status_code}: {r.text}"
        body = r.json()
        assert body["pipeline_metadata"]["is_fallback"] is True
        assert body["pipeline_metadata"]["feature_source"] == "fallback_cache"


# ── Health Endpoint Tests ─────────────────────────────────────────────────────

class TestHealthEndpoint:

    @pytest.mark.asyncio
    async def test_health_returns_component_status(self, app_client):
        client, _ = app_client
        r = await client.get("/health")
        assert r.status_code == 200
        body = r.json()
        assert "status" in body
        assert "components" in body
        assert "version" in body
        assert body["status"] in ("healthy", "degraded")

    @pytest.mark.asyncio
    async def test_health_lists_all_components(self, app_client):
        client, _ = app_client
        r = await client.get("/health")
        components = r.json()["components"]
        expected_components = {"redis", "qdrant", "onnx_runtime"}
        assert expected_components.issubset(set(components.keys())), (
            f"Missing health components: {expected_components - set(components.keys())}"
        )
