"""
scripts/populate_fallback_cache.py
────────────────────────────────────
Fallback Cache Population Job.

Runs as a scheduled cron job (every 5 minutes) to pre-compute the
popularity-sorted content list that serves as the circuit-breaker response
when Qdrant is unavailable.

Fallback scoring formula:
  score = (0.5 × global_completion_rate)
        + (0.3 × global_ctr)
        + (0.2 × log1p(total_interactions) / 20)

  Completion rate is weighted highest — it correlates most strongly with
  long-term user satisfaction and reduces clickbait optimisation pressure.

Cache structure:
  Redis List key: "fallback:popular_content"
  Each element: JSON-encoded content payload dict
  TTL: 600 seconds (refreshed every 5 min → max staleness = 10 min)
  Size: top-500 items (covers any top_n up to 50 with safety filter headroom)

Content type stratification:
  To avoid the fallback serving only one content type, we stratify:
    40% workout_routine (200 items)
    35% video (175 items)
    25% meal_recipe (125 items)
  This mirrors the catalogue distribution and ensures diverse fallback sets.

Operational guarantees:
  - Idempotent: re-running always produces a consistent cache state
  - Zero-downtime: writes to a temp key, then atomic RENAME
  - Atomic TTL refresh: EXPIRE set after RENAME to avoid window of no cache
"""
from __future__ import annotations

import asyncio
import json
import logging
import math
import sys
from pathlib import Path
_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import redis.asyncio as aioredis
from sqlalchemy import select, func

from config.settings import settings
from data_pipeline.database import AsyncSessionLocal, init_db
from data_pipeline.schemas import ContentItem

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


FALLBACK_CACHE_KEY = "fallback:popular_content"
FALLBACK_CACHE_TEMP_KEY = "fallback:popular_content:tmp"
FALLBACK_CACHE_SIZE = 500
FALLBACK_CACHE_TTL = 600


def _compute_popularity_score(item: ContentItem) -> float:
    """
    Composite popularity score blending engagement quality signals.
    Used to rank the fallback list; computed offline on content metadata.
    """
    completion_component = (item.global_completion_rate or 0.0) * 0.5
    ctr_component = (item.global_ctr or 0.0) * 0.3
    volume_component = math.log1p(item.total_interactions or 0) / 20.0 * 0.2
    return completion_component + ctr_component + volume_component


def _item_to_payload(item: ContentItem, score: float) -> dict:
    """Build the lean payload stored in the fallback cache list."""
    return {
        "content_id": str(item.id),
        "title": item.title,
        "content_type": item.content_type,
        "thumbnail_url": item.thumbnail_url,
        "workout_type": item.workout_type,
        "duration_minutes": item.duration_minutes,
        "intensity_score": item.intensity_score,
        "calories_burned_estimate": item.calories_burned_estimate,
        "sodium_mg": item.sodium_mg,
        "calories_kcal": item.calories_kcal,
        "protein_g": item.protein_g,
        "dietary_tags": item.dietary_tags or [],
        "required_equipment": item.required_equipment or [],
        "global_ctr": item.global_ctr or 0.0,
        "global_completion_rate": item.global_completion_rate or 0.0,
        "total_interactions": item.total_interactions or 0,
        # Inference fields for compatibility with RankedRecommendation schema
        "inference_score": round(score, 6),
        "score": round(score, 6),
        "payload": {
            "content_id": str(item.id),
            "title": item.title,
            "content_type": item.content_type,
            "workout_type": item.workout_type,
            "duration_minutes": item.duration_minutes,
            "intensity_score": item.intensity_score,
            "sodium_mg": item.sodium_mg,
            "calories_kcal": item.calories_kcal,
            "protein_g": item.protein_g,
            "dietary_tags": item.dietary_tags or [],
            "thumbnail_url": item.thumbnail_url,
            "global_ctr": item.global_ctr or 0.0,
            "global_completion_rate": item.global_completion_rate or 0.0,
            "total_interactions": item.total_interactions or 0,
        },
    }


# ── Content Type Stratified Fetcher ──────────────────────────────────────────

async def fetch_top_by_type(
    session,
    content_type: str,
    limit: int,
) -> list[tuple[ContentItem, float]]:
    """Fetch top-N items of a specific content type by popularity score."""
    stmt = (
        select(ContentItem)
        .where(ContentItem.content_type == content_type)
        .where(ContentItem.is_published == True)
        .order_by(
            (
                ContentItem.global_completion_rate * 0.5
                + ContentItem.global_ctr * 0.3
                + func.log(ContentItem.total_interactions + 1) / 20.0 * 0.2
            ).desc()
        )
        .limit(limit)
    )
    result = await session.execute(stmt)
    items = result.scalars().all()
    return [(item, _compute_popularity_score(item)) for item in items]


async def build_stratified_fallback_list() -> list[dict]:
    """
    Build the stratified fallback list from NeonDB.

    Stratification targets:
      40% workout_routine, 35% video, 25% meal_recipe
    """
    n_workout = int(FALLBACK_CACHE_SIZE * 0.40)  # 200
    n_video = int(FALLBACK_CACHE_SIZE * 0.35)    # 175
    n_recipe = FALLBACK_CACHE_SIZE - n_workout - n_video  # 125

    async with AsyncSessionLocal() as session:
        workouts, videos, recipes = await asyncio.gather(
            fetch_top_by_type(session, "workout_routine", n_workout),
            fetch_top_by_type(session, "video", n_video),
            fetch_top_by_type(session, "meal_recipe", n_recipe),
        )

    logger.info(
        "Fetched: %d workouts, %d videos, %d recipes",
        len(workouts), len(videos), len(recipes),
    )

    # Interleave by content type for diversity (round-robin merge)
    merged = []
    max_len = max(len(workouts), len(videos), len(recipes))
    for i in range(max_len):
        for pool in (workouts, videos, recipes):
            if i < len(pool):
                item, score = pool[i]
                merged.append(_item_to_payload(item, score))

    return merged[:FALLBACK_CACHE_SIZE]


# ── Cache Writer ──────────────────────────────────────────────────────────────

async def populate_fallback_cache(redis: aioredis.Redis) -> dict:
    """
    Atomically replace the fallback cache with fresh content.

    Atomicity mechanism:
      1. Write all items to temp key (FALLBACK_CACHE_TEMP_KEY)
      2. RENAME temp key to production key (atomic in Redis)
      3. EXPIRE on production key

    This ensures readers always see either the old complete list
    or the new complete list — never a partial write.
    """
    t0 = time.perf_counter()
    logger.info("Building fallback content list...")

    fallback_items = await build_stratified_fallback_list()

    if not fallback_items:
        logger.warning("No published content found — fallback cache not updated.")
        return {"items": 0, "elapsed_ms": 0}

    # Delete old temp key if it exists from a previous failed run
    await redis.delete(FALLBACK_CACHE_TEMP_KEY)

    # Batch write to temp key
    pipeline = redis.pipeline()
    for item in fallback_items:
        pipeline.rpush(FALLBACK_CACHE_TEMP_KEY, json.dumps(item))
    await pipeline.execute()

    # Atomic rename + TTL
    await redis.rename(FALLBACK_CACHE_TEMP_KEY, FALLBACK_CACHE_KEY)
    await redis.expire(FALLBACK_CACHE_KEY, FALLBACK_CACHE_TTL)

    elapsed_ms = (time.perf_counter() - t0) * 1000

    logger.info(
        "✅ Fallback cache updated: %d items | Elapsed: %.1f ms | TTL: %ds",
        len(fallback_items), elapsed_ms, FALLBACK_CACHE_TTL,
    )

    return {"items": len(fallback_items), "elapsed_ms": round(elapsed_ms, 1)}


# ── Feast Feature Materialisation ─────────────────────────────────────────────

async def materialise_batch_features_to_redis(redis: aioredis.Redis) -> None:
    """
    Write batch user features (from offline Feast store) to Redis.

    In production this is handled by `feast materialize-incremental` running
    as a scheduled job. This function is a lightweight alternative for
    environments without a full Feast installation.

    Writes keys matching the pattern Feast uses:
      feast:user:{user_id}:batch_features → Redis Hash

    Populated from: UserEmbedding table + User health markers
    """
    from data_pipeline.schemas import User, UserEmbedding
    from sqlalchemy import select

    logger.info("Materialising user batch features to Redis...")
    t0 = time.perf_counter()
    count = 0

    async with AsyncSessionLocal() as session:
        PAGE = 1000
        offset = 0
        while True:
            stmt = (
                select(User, UserEmbedding)
                .outerjoin(UserEmbedding, User.id == UserEmbedding.user_id)
                .where(User.is_active == True)
                .offset(offset)
                .limit(PAGE)
            )
            result = await session.execute(stmt)
            rows = result.all()
            if not rows:
                break

            pipeline = redis.pipeline()
            for user, embedding in rows:
                key = f"feast:user:{user.id}:batch_features"
                import math as _math
                bmi = (
                    (user.weight_kg / ((user.height_cm / 100) ** 2))
                    if user.weight_kg and user.height_cm
                    else 23.5
                )
                features = {
                    "fitness_goal_encoded": str(
                        {"weight_loss": 0, "muscle_gain": 1, "endurance": 2,
                         "flexibility": 3, "maintenance": 4}.get(user.fitness_goal or "", 0)
                    ),
                    "age_normalised": str(round((user.age or 30) / 100.0, 4)),
                    "bmi": str(round(bmi, 2)),
                    "structural_adherence_rate_30d": "0.50",  # default; real value from offline job
                    "completion_rate_30d": "0.50",
                    "workout_sessions_30d": "8",
                    "is_hypertensive": str(int(user.is_hypertensive)),
                    "has_cardiac_risk": str(int(user.has_cardiac_risk)),
                    "has_diabetes": str(int(user.has_diabetes)),
                    "dietary_restrictions": json.dumps(user.dietary_restrictions or {}),
                }
                pipeline.hset(key, mapping=features)
                pipeline.expire(key, 86400)  # 24h TTL; refreshed daily by offline job
                count += 1

            await pipeline.execute()
            offset += PAGE
            logger.debug("Materialised %d user feature records", count)

    elapsed = (time.perf_counter() - t0) * 1000
    logger.info(
        "✅ Materialised %d user feature records in %.0f ms", count, elapsed
    )


# ── Main ──────────────────────────────────────────────────────────────────────

async def main(also_materialise: bool = False) -> None:
    await init_db()

    redis = aioredis.from_url(
        settings.REDIS_URL,
        max_connections=10,
        decode_responses=True,
    )

    try:
        await redis.ping()
        logger.info("Redis connection: OK")

        results = await populate_fallback_cache(redis)
        logger.info("Fallback cache: %s", results)

        if also_materialise:
            await materialise_batch_features_to_redis(redis)

    finally:
        await redis.aclose()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Populate Redis fallback cache")
    parser.add_argument(
        "--materialise", action="store_true",
        help="Also materialise user batch features to Redis"
    )
    args = parser.parse_args()

    asyncio.run(main(also_materialise=args.materialise))
