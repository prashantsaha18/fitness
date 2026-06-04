"""
tests/conftest.py
──────────────────
Pytest fixtures shared across the test suite.

Test philosophy:
  - Unit tests: pure function testing, no I/O. Mock all external dependencies.
  - Integration tests: real Redis + Qdrant via Docker fixtures.
  - All tests must complete in < 10 seconds each.
  - Coverage target: ≥ 85% on ranking/, retrieval/, api/ modules.

Fixture hierarchy:
  event_loop          (session-scoped asyncio loop)
  ├── redis_client    (module-scoped mock Redis)
  ├── qdrant_client   (module-scoped mock Qdrant)
  ├── db_session      (function-scoped in-memory SQLite)
  └── test_app        (function-scoped FastAPI TestClient)
"""
from __future__ import annotations

import asyncio
import uuid
from typing import AsyncGenerator
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
import pytest_asyncio
from httpx import AsyncClient

# ── Event Loop ────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def event_loop():
    """Session-scoped event loop — prevents loop creation overhead per test."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


# ── Sample Data Factories ─────────────────────────────────────────────────────

@pytest.fixture
def sample_user_id() -> str:
    return str(uuid.uuid4())


@pytest.fixture
def sample_content_id() -> str:
    return str(uuid.uuid4())


@pytest.fixture
def sample_embedding() -> np.ndarray:
    """L2-normalised 384-dim unit vector."""
    v = np.random.randn(384).astype(np.float32)
    return v / np.linalg.norm(v)


@pytest.fixture
def sample_embeddings_batch(sample_embedding) -> list[np.ndarray]:
    """Batch of 100 normalised embeddings simulating Stage-1 output."""
    return [
        np.random.randn(384).astype(np.float32) / np.linalg.norm(np.random.randn(384))
        for _ in range(100)
    ]


@pytest.fixture
def sample_candidates(sample_content_id) -> list[dict]:
    """100 mock Stage-1 ANN results with realistic payload structure."""
    return [
        {
            "content_id": str(uuid.uuid4()),
            "qdrant_id": str(uuid.uuid4()),
            "score": float(np.random.uniform(0.5, 0.99)),
            "payload": {
                "content_id": str(uuid.uuid4()),
                "title": f"Workout #{i}",
                "content_type": "workout_routine" if i % 3 != 0 else "meal_recipe",
                "workout_type": "hiit" if i % 3 == 0 else "strength",
                "intensity_score": float(np.random.uniform(0.2, 0.9)),
                "duration_minutes": 30,
                "sodium_mg": 200.0 if i % 5 != 0 else 800.0,   # some high-sodium items
                "calories_kcal": 400.0,
                "protein_g": 30.0,
                "global_ctr": 0.12,
                "global_completion_rate": 0.65,
                "total_interactions": 1500,
                "dietary_tags": ["high-protein"],
                "required_equipment": [],
            },
        }
        for i in range(100)
    ]


@pytest.fixture
def sample_user_features() -> dict:
    return {
        "fitness_goal_encoded": 1,
        "age_normalised": 0.28,
        "bmi": 23.5,
        "structural_adherence_rate_30d": 0.72,
        "completion_rate_30d": 0.68,
        "is_hypertensive": False,
        "has_cardiac_risk": False,
        "has_diabetes": False,
        "dietary_restrictions": {},
    }


@pytest.fixture
def sample_user_features_hypertensive() -> dict:
    return {
        "fitness_goal_encoded": 0,
        "age_normalised": 0.52,
        "bmi": 28.1,
        "structural_adherence_rate_30d": 0.45,
        "completion_rate_30d": 0.50,
        "is_hypertensive": True,
        "has_cardiac_risk": False,
        "has_diabetes": False,
        "dietary_restrictions": {"low-sodium": True},
    }


@pytest.fixture
def sample_realtime_context() -> dict:
    return {
        "hr_mean_5min": 142.0,
        "fatigue_latest": 0.55,
        "recovery_score": 0.62,
        "cal_total_session": 280.0,
        "heart_rate_zone": "cardio",
    }


# ── Mock External Services ────────────────────────────────────────────────────

@pytest.fixture
def mock_redis():
    """In-memory dict-backed mock Redis client."""
    store = {}
    lists = {}

    redis = AsyncMock()

    async def hget(key, field):
        return store.get(key, {}).get(field)

    async def hgetall(key):
        return store.get(key, {})

    async def hset(key, mapping=None, **kwargs):
        store.setdefault(key, {}).update(mapping or kwargs)

    async def expire(key, ttl):
        pass

    async def ping():
        return True

    async def lrange(key, start, end):
        return lists.get(key, [])[start:end + 1 if end >= 0 else None]

    async def rpush(key, *values):
        lists.setdefault(key, []).extend(values)

    async def delete(*keys):
        for k in keys:
            store.pop(k, None)
            lists.pop(k, None)

    async def rename(src, dst):
        if src in lists:
            lists[dst] = lists.pop(src)

    redis.hget = hget
    redis.hgetall = hgetall
    redis.hset = hset
    redis.expire = expire
    redis.ping = ping
    redis.lrange = lrange
    redis.rpush = rpush
    redis.delete = delete
    redis.rename = rename
    redis.pipeline = MagicMock(return_value=AsyncMock())

    return redis


@pytest.fixture
def mock_qdrant_client(sample_candidates):
    """Mock Qdrant client returning controlled search results."""
    client = AsyncMock()

    async def search(*args, **kwargs):
        results = []
        for c in sample_candidates[:kwargs.get("limit", 100)]:
            hit = MagicMock()
            hit.id = c["qdrant_id"]
            hit.score = c["score"]
            hit.payload = c["payload"]
            results.append(hit)
        return results

    async def get_collections():
        coll = MagicMock()
        coll.name = "fitness_content"
        result = MagicMock()
        result.collections = [coll]
        return result

    client.search = search
    client.get_collections = get_collections
    client.upsert = AsyncMock()
    client.create_collection = AsyncMock()
    client.close = AsyncMock()

    return client
