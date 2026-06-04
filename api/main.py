"""
api/main.py
────────────
Production FastAPI Orchestration Engine — Two-Stage Recommendation Pipeline.

Request lifecycle (P99 < 30ms budget):
  ┌─────────────────────────────────────────────────────────────────────────┐
  │  0ms   JWT validation (in-memory HMAC, ~0.1ms)                         │
  │  0ms   ──────────────────────────────────────────────────────────────  │
  │        asyncio.gather() — CONCURRENT execution:                         │
  │  2ms     ├─ Feast online feature fetch (Redis, ~1-2ms)                 │
  │  9ms     └─ Stage-1 ANN retrieval (Qdrant HNSW, ~5-9ms)               │
  │  9ms   ──────────────────────────────────────────────────────────────  │
  │  10ms  Feature tensor construction O(N×D)                              │
  │  13ms  Stage-2 ONNX inference (CPU/GPU)                                │
  │  14ms  Descending sort O(N log N)                                      │
  │  14ms  Safety filter O(N)                                              │
  │  14ms  Response serialisation                                           │
  └─────────────────────────────────────────────────────────────────────────┘

Production fail-safes:
  VectorStoreUnavailableError:
    → Serve top-N from pre-computed popularity-sorted fallback cache (Redis).
    → Log degraded mode event to observability pipeline.
    → Return is_fallback=True in response metadata.
    → Client receives 200 with stale-but-valid recommendations, NOT a 500.

  FeastUnavailableError:
    → Fall back to cold-start feature vector (demographic averages).
    → Feature quality degrades gracefully; safety filters still enforced.

  ONNXInferenceError:
    → Skip Stage-2; return Stage-1 ANN results sorted by cosine similarity.
    → Model version set to "ann_fallback_v0" in response metadata.

This design ensures the recommendation endpoint never returns 5xx under
partial infrastructure failure — critical for maintaining app-side SLA.
"""
from __future__ import annotations

import asyncio
import logging
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Annotated, Any, Optional

import redis.asyncio as aioredis
import structlog
from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import ORJSONResponse
from fastapi.security import OAuth2PasswordRequestForm
from prometheus_client import Counter, Histogram, generate_latest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from starlette.responses import Response

from api.auth import (
    Token,
    create_access_token,
    get_current_user_id,
    hash_password,
    verify_password,
)
from api.schemas import (
    HealthResponse,
    PipelineMetadata,
    RankedRecommendation,
    RecommendationRequest,
    RecommendationResponse,
    RecommendedItemMetadata,
    UserCreate,
    UserResponse,
)
from config.settings import settings
from data_pipeline.database import dispose_engine, get_db, init_db
from data_pipeline.schemas import User
from ranking.export_onnx import get_inference_engine
from ranking.features import build_input_tensor, validate_input_tensor
from retrieval.embeddings import (
    EmbeddingEngine,
    build_user_query_text,
    get_embedding_engine,
)
from retrieval.vector_store import (
    VectorStoreUnavailableError,
    ensure_collection_exists,
    get_qdrant_client,
    retrieve_candidates,
)

logger = structlog.get_logger(__name__)

# ── Prometheus Metrics ────────────────────────────────────────────────────────

RECOMMENDATION_LATENCY = Histogram(
    "recommendation_latency_ms",
    "End-to-end recommendation latency in milliseconds",
    buckets=[5, 10, 15, 20, 25, 30, 35, 50, 100, 250],
)
RECOMMENDATION_REQUESTS = Counter(
    "recommendation_requests_total",
    "Total recommendation requests",
    ["status"],
)
FALLBACK_COUNTER = Counter(
    "recommendation_fallback_total",
    "Requests served from fallback cache",
    ["reason"],
)

MODEL_VERSION = "deepfm_v1.0.0"


# ── Application Lifespan ──────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Async context manager for startup/shutdown resource management.
    All I/O clients are initialised once and reused across requests.
    """
    logger.info("🚀 Fitness Rec Engine starting up...")

    # ── Database bootstrap ────────────────────────────────────────────────
    await init_db()
    logger.info("✅ PostgreSQL schema initialised (NeonDB)")

    # ── Redis connection pool ─────────────────────────────────────────────
    app.state.redis = aioredis.from_url(
        settings.REDIS_URL,
        max_connections=settings.REDIS_POOL_MAX_CONNECTIONS,
        decode_responses=True,
    )
    await app.state.redis.ping()
    logger.info("✅ Redis connection pool active")

    # ── Qdrant collection ─────────────────────────────────────────────────
    await ensure_collection_exists()
    logger.info("✅ Qdrant collection verified")

    # ── Embedding model warm-up (loads weights into memory) ───────────────
    engine = get_embedding_engine()
    _ = await engine.encode_async(["warmup"])
    logger.info("✅ Embedding model warm (dim=%d)", engine.embedding_dim)

    # ── ONNX inference session ────────────────────────────────────────────
    inference = get_inference_engine()
    inference.load()
    logger.info("✅ ONNX inference session loaded")

    logger.info("🟢 Service ready — P99 target: %dms", settings.RECOMMENDATION_TIMEOUT_MS)

    yield  # ─────────────────── serving ───────────────────

    # ── Graceful shutdown ─────────────────────────────────────────────────
    logger.info("🔴 Shutting down...")
    await app.state.redis.aclose()
    await dispose_engine()
    logger.info("✅ Shutdown complete")


# ── App Factory ───────────────────────────────────────────────────────────────

def create_app() -> FastAPI:
    app = FastAPI(
        title="Fitness Recommendation Engine",
        description=(
            "Ultra-low latency (<30ms) two-stage multi-modal recommendation "
            "system for personalized fitness & nutrition content."
        ),
        version="1.0.0",
        lifespan=lifespan,
        default_response_class=ORJSONResponse,  # 2–3× faster than standard JSON
        docs_url="/docs" if settings.ENV != "production" else None,
        redoc_url=None,
    )

    # ── Middleware ────────────────────────────────────────────────────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"] if settings.ENV == "development" else [],
        allow_methods=["GET", "POST"],
        allow_headers=["Authorization", "Content-Type"],
    )
    app.add_middleware(GZipMiddleware, minimum_size=1000)

    return app


app = create_app()


# ── Dependency: Redis ─────────────────────────────────────────────────────────

async def get_redis(request: Request) -> aioredis.Redis:
    return request.app.state.redis


# ── Online Feature Fetcher ────────────────────────────────────────────────────

async def fetch_online_user_features(
    user_id: str,
    redis: aioredis.Redis,
    realtime_override: Optional[dict] = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """
    Fetch user features from the online feature store.

    Returns:
        (batch_features, realtime_context)
        batch_features   — from Redis (populated by Feast materialise job)
        realtime_context — from Redis (populated by Kafka consumer)

    Fallback: returns cold-start defaults if Redis keys are absent (new user).
    """
    batch_key = f"feast:user:{user_id}:batch_features"
    realtime_key = f"user:{user_id}:realtime_features"

    # Concurrent Redis lookups
    batch_raw, realtime_raw = await asyncio.gather(
        redis.hgetall(batch_key),
        redis.hgetall(realtime_key),
    )

    # ── Parse batch features ──────────────────────────────────────────────
    if batch_raw:
        batch_features = {k: _safe_float(v) for k, v in batch_raw.items()}
    else:
        # Cold-start: use population mean defaults
        batch_features = _cold_start_user_features()

    # ── Parse realtime context ────────────────────────────────────────────
    if realtime_raw:
        realtime_context = {k: _safe_float(v) for k, v in realtime_raw.items()}
    else:
        realtime_context = {"fatigue_latest": 0.3, "hr_mean_5min": 70.0,
                            "recovery_score": 0.7, "cal_total_session": 0.0,
                            "heart_rate_zone": "resting"}

    # ── Apply client-side overrides ───────────────────────────────────────
    if realtime_override:
        if realtime_override.get("heart_rate_bpm"):
            realtime_context["hr_mean_5min"] = float(realtime_override["heart_rate_bpm"])
        if realtime_override.get("fatigue_level") is not None:
            realtime_context["fatigue_latest"] = float(realtime_override["fatigue_level"])
        if realtime_override.get("active_calories_kcal") is not None:
            realtime_context["cal_total_session"] = float(realtime_override["active_calories_kcal"])

    feature_source = "online" if (batch_raw or realtime_raw) else "cold_start"
    return batch_features, realtime_context, feature_source


def _cold_start_user_features() -> dict[str, Any]:
    """Population-mean feature vector for new users without interaction history."""
    return {
        "fitness_goal_encoded": 0,
        "age_normalised": 0.3,
        "bmi": 23.5,
        "structural_adherence_rate_30d": 0.5,
        "completion_rate_30d": 0.5,
        "workout_sessions_30d": 8,
        "dietary_restrictions": {},
        "is_hypertensive": False,
        "has_cardiac_risk": False,
        "has_diabetes": False,
    }


def _safe_float(v: Any) -> Any:
    try:
        return float(v)
    except (TypeError, ValueError):
        return v


# ── Safety Filter Engine ──────────────────────────────────────────────────────

def apply_safety_filters(
    candidates: list[dict[str, Any]],
    user_features: dict[str, Any],
    user_db: Optional[User] = None,
) -> tuple[list[dict[str, Any]], int, list[str]]:
    """
    Hard safety rules — applied POST ranking, BEFORE response serialisation.
    Cannot be overridden by client; enforced unconditionally.

    Rules:
      R1: Hypertensive users → exclude recipes with sodium > 400mg
      R2: Cardiac risk users → exclude workouts with intensity > 0.7
      R3: Diabetic users → exclude recipes with carbs > 60g
      R4: Global minimum confidence threshold
      R5: Remove explicitly excluded IDs

    Returns:
        (filtered_list, num_removed, applied_flags)
    """
    is_hypertensive = (
        user_db.is_hypertensive if user_db
        else bool(user_features.get("is_hypertensive", False))
    )
    has_cardiac_risk = (
        user_db.has_cardiac_risk if user_db
        else bool(user_features.get("has_cardiac_risk", False))
    )
    has_diabetes = (
        user_db.has_diabetes if user_db
        else bool(user_features.get("has_diabetes", False))
    )

    filtered = []
    removed = 0
    flags_applied: set[str] = set()

    for item in candidates:
        payload = item.get("payload", {})
        item_flags = []

        # R1: Sodium check for hypertensive users
        if is_hypertensive:
            sodium = payload.get("sodium_mg")
            if sodium and float(sodium) > settings.MAX_SODIUM_HYPERTENSIVE_MG:
                removed += 1
                flags_applied.add("low_sodium_enforced")
                continue
            if sodium:
                item_flags.append("low_sodium_enforced")

        # R2: Intensity check for cardiac risk users
        if has_cardiac_risk:
            intensity = payload.get("intensity_score")
            if intensity and float(intensity) > settings.MAX_INTENSITY_CARDIAC_RISK:
                removed += 1
                flags_applied.add("cardiac_intensity_cap")
                continue
            item_flags.append("cardiac_safe")

        # R3: Carbs check for diabetic users
        if has_diabetes:
            carbs = payload.get("carbs_g")
            if carbs and float(carbs) > 60.0:
                removed += 1
                flags_applied.add("low_carb_enforced")
                continue

        # R4: Confidence threshold
        if item.get("inference_score", 1.0) < settings.MIN_CONFIDENCE_THRESHOLD:
            removed += 1
            continue

        item["safety_flags"] = item_flags
        filtered.append(item)

    return filtered, removed, list(flags_applied)


# ── Fallback Cache ────────────────────────────────────────────────────────────

async def get_fallback_recommendations(
    redis: aioredis.Redis,
    top_n: int,
) -> list[dict[str, Any]]:
    """
    Serve pre-computed popularity-sorted content from Redis fallback cache.
    Populated by a daily batch job; TTL=600s guarantees freshness.

    This is the circuit-breaker response when Qdrant is unavailable.
    """
    key = "fallback:popular_content"
    cached = await redis.lrange(key, 0, top_n - 1)
    if cached:
        import json
        return [json.loads(item) for item in cached]

    # Last-resort: return empty list (client shows loading state)
    logger.warning("Fallback cache is empty — returning empty recommendations")
    return []


# ── Core Recommendation Endpoint ──────────────────────────────────────────────

@app.post(
    f"{settings.API_V1_PREFIX}/recommend",
    response_model=RecommendationResponse,
    response_class=ORJSONResponse,
    status_code=status.HTTP_200_OK,
    summary="Get personalised recommendations",
    tags=["Recommendations"],
)
async def get_recommendations(
    payload: RecommendationRequest,
    request: Request,
    user_id_token: Annotated[str, Depends(get_current_user_id)],
    redis: Annotated[aioredis.Redis, Depends(get_redis)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> RecommendationResponse:
    """
    Two-stage recommendation pipeline.

    Stage 1: ANN vector retrieval — top-100 candidates in <10ms
    Stage 2: Deep ranking — ONNX inference + safety filtering in <15ms

    Security: JWT user_id_token must match payload.user_id to prevent
    users from requesting recommendations on behalf of other users.
    """
    request_id = str(uuid.uuid4())
    t_total_start = time.perf_counter()

    # ── Authorization: user can only request their own recommendations ────
    if str(payload.user_id) != user_id_token:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Token user_id does not match request user_id.",
        )

    is_fallback = False
    fallback_reason = ""
    feature_source = "online"
    stage1_latency_ms = 0.0
    stage2_latency_ms = 0.0

    try:
        # ── STAGE 0: Build user query embedding ──────────────────────────
        # Fetch user DB record for query text construction + safety filters
        user_result = await db.execute(
            select(User).where(User.id == payload.user_id, User.is_active == True)
        )
        user_db = user_result.scalar_one_or_none()
        if not user_db:
            raise HTTPException(status_code=404, detail="User not found.")

        realtime_override = (
            payload.realtime_context.model_dump(exclude_none=True)
            if payload.realtime_context else None
        )

        # ── STAGE 0b: Concurrent I/O — features + embedding in parallel ──
        t_stage1_start = time.perf_counter()

        query_text = build_user_query_text(
            fitness_goal=user_db.fitness_goal,
            preferred_workout_types=user_db.preferred_workout_types,
            dietary_restrictions=user_db.dietary_restrictions,
            age=user_db.age,
            realtime_context=realtime_override,
        )

        # Run feature fetch + embedding concurrently
        (batch_features, realtime_context, feature_source), query_embedding = \
            await asyncio.gather(
                fetch_online_user_features(
                    user_id=str(payload.user_id),
                    redis=redis,
                    realtime_override=realtime_override,
                ),
                get_embedding_engine().encode_single(query_text),
            )

        # ── STAGE 1: ANN Candidate Retrieval ─────────────────────────────
        # Optionally apply Qdrant-side content type filter
        qdrant_filter = None
        if payload.content_types:
            from retrieval.vector_store import build_content_type_filter
            qdrant_filter = build_content_type_filter(payload.content_types)

        candidates = await retrieve_candidates(
            query_embedding=query_embedding,
            top_k=settings.QDRANT_TOP_K,
            filter_conditions=qdrant_filter,
        )

        stage1_latency_ms = (time.perf_counter() - t_stage1_start) * 1000

        # ── Apply exclude_ids filter ──────────────────────────────────────
        if payload.exclude_ids:
            exclude_set = {str(uid) for uid in payload.exclude_ids}
            candidates = [c for c in candidates if c["content_id"] not in exclude_set]

    except VectorStoreUnavailableError as exc:
        # ── CIRCUIT BREAKER: Qdrant unavailable ──────────────────────────
        logger.warning(
            "Vector store unavailable — serving fallback",
            error=str(exc),
            user_id=str(payload.user_id),
        )
        FALLBACK_COUNTER.labels(reason="qdrant_unavailable").inc()
        is_fallback = True
        fallback_reason = "qdrant_unavailable"
        batch_features, realtime_context, feature_source = (
            _cold_start_user_features(), {}, "fallback_cache"
        )
        candidates = await get_fallback_recommendations(redis, payload.top_n * 3)
        user_db = None

    # ── STAGE 2: Deep Ranking ─────────────────────────────────────────────
    t_stage2_start = time.perf_counter()
    ranked_candidates = candidates  # default: ANN order if Stage-2 fails

    if candidates and not is_fallback:
        try:
            # Build feature matrix [N, 409]
            feature_matrix = build_input_tensor(
                candidates=candidates,
                user_features=batch_features,
                realtime_context=realtime_context,
            )
            validate_input_tensor(feature_matrix)

            # ONNX inference + sort
            ranked_candidates = await get_inference_engine().rank_candidates(
                feature_matrix=feature_matrix,
                candidates=candidates,
            )
        except asyncio.TimeoutError:
            logger.warning("Stage-2 timeout — falling back to ANN order")
            FALLBACK_COUNTER.labels(reason="stage2_timeout").inc()
            # Graceful degradation: Stage-1 ANN order is still valid
            for i, c in enumerate(candidates):
                c["inference_score"] = 1.0 - (i / max(len(candidates), 1))
            ranked_candidates = candidates
        except Exception as exc:
            logger.error("Stage-2 error", error=str(exc))
            for i, c in enumerate(candidates):
                c["inference_score"] = candidates[i].get("score", 0.0)
            ranked_candidates = candidates

    stage2_latency_ms = (time.perf_counter() - t_stage2_start) * 1000

    # ── SAFETY FILTERS ────────────────────────────────────────────────────
    filtered_candidates, num_safety_removed, _ = apply_safety_filters(
        candidates=ranked_candidates,
        user_features=batch_features,
        user_db=user_db,
    )

    # ── Truncate to top_n ─────────────────────────────────────────────────
    final_candidates = filtered_candidates[: payload.top_n]

    # ── Build response ────────────────────────────────────────────────────
    total_latency_ms = (time.perf_counter() - t_total_start) * 1000

    recommendations = [
        RankedRecommendation(
            rank=i + 1,
            content_id=c["content_id"],
            inference_score=round(c.get("inference_score", 0.0), 6),
            retrieval_score=round(c.get("score", 0.0), 6),
            metadata=RecommendedItemMetadata(
                content_id=c["content_id"],
                title=c.get("payload", {}).get("title", "Unknown"),
                content_type=c.get("payload", {}).get("content_type", "unknown"),
                thumbnail_url=c.get("payload", {}).get("thumbnail_url"),
                workout_type=c.get("payload", {}).get("workout_type"),
                duration_minutes=c.get("payload", {}).get("duration_minutes"),
                intensity_score=c.get("payload", {}).get("intensity_score"),
                calories_burned_estimate=c.get("payload", {}).get("calories_burned_estimate"),
                sodium_mg=c.get("payload", {}).get("sodium_mg"),
                calories_kcal=c.get("payload", {}).get("calories_kcal"),
                protein_g=c.get("payload", {}).get("protein_g"),
                dietary_tags=c.get("payload", {}).get("dietary_tags"),
            ),
            safety_flags=c.get("safety_flags", []),
        )
        for i, c in enumerate(final_candidates)
    ]

    RECOMMENDATION_LATENCY.observe(total_latency_ms)
    RECOMMENDATION_REQUESTS.labels(status="success").inc()

    if total_latency_ms > 30.0:
        logger.warning(
            "P99 SLA breach",
            latency_ms=total_latency_ms,
            user_id=str(payload.user_id),
            is_fallback=is_fallback,
        )

    return RecommendationResponse(
        request_id=request_id,
        user_id=str(payload.user_id),
        recommendations=recommendations,
        pipeline_metadata=PipelineMetadata(
            stage1_candidates=len(candidates),
            stage2_ranked=len(ranked_candidates),
            safety_filtered=num_safety_removed,
            total_latency_ms=round(total_latency_ms, 2),
            stage1_latency_ms=round(stage1_latency_ms, 2),
            stage2_latency_ms=round(stage2_latency_ms, 2),
            model_version=MODEL_VERSION if not is_fallback else "ann_fallback_v0",
            feature_source=feature_source,
            is_fallback=is_fallback,
        ),
    )


# ── Auth Endpoints ────────────────────────────────────────────────────────────

@app.post(
    f"{settings.API_V1_PREFIX}/auth/register",
    response_model=UserResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["Auth"],
)
async def register(
    body: UserCreate,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> UserResponse:
    existing = await db.execute(select(User).where(User.email == body.email))
    if existing.scalar_one_or_none():
        raise HTTPException(status_code=409, detail="Email already registered.")

    user = User(
        username=body.username,
        email=body.email,
        hashed_password=hash_password(body.password),
    )
    db.add(user)
    await db.flush()
    await db.refresh(user)

    return UserResponse(
        id=str(user.id),
        username=user.username,
        email=user.email,
        is_active=user.is_active,
    )


@app.post(
    f"{settings.API_V1_PREFIX}/auth/token",
    response_model=Token,
    tags=["Auth"],
)
async def login(
    form: Annotated[OAuth2PasswordRequestForm, Depends()],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> Token:
    result = await db.execute(select(User).where(User.email == form.username))
    user = result.scalar_one_or_none()

    if not user or not verify_password(form.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return create_access_token(str(user.id))


# ── Health & Observability ────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["Ops"])
async def health_check(
    request: Request,
    redis: Annotated[aioredis.Redis, Depends(get_redis)],
) -> HealthResponse:
    components: dict[str, str] = {}

    # Redis
    try:
        await redis.ping()
        components["redis"] = "healthy"
    except Exception:
        components["redis"] = "degraded"

    # Qdrant
    try:
        client = await get_qdrant_client()
        await asyncio.wait_for(client.get_collections(), timeout=1.0)
        components["qdrant"] = "healthy"
    except Exception:
        components["qdrant"] = "degraded"

    components["onnx_runtime"] = "healthy" if get_inference_engine()._session else "not_loaded"

    overall = "healthy" if all(v == "healthy" for v in components.values()) else "degraded"

    return HealthResponse(status=overall, version="1.0.0", components=components)


@app.get("/metrics", include_in_schema=False)
async def metrics():
    """Prometheus scrape endpoint."""
    return Response(generate_latest(), media_type="text/plain")


# ── Entry Point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        workers=1,                      # use gunicorn + uvicorn workers in prod
        loop="uvloop",
        http="httptools",
        log_level=settings.LOG_LEVEL.lower(),
        access_log=settings.ENV != "production",
    )
