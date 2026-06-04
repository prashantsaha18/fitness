"""
scripts/index_content.py
─────────────────────────
Content embedding & Qdrant indexing pipeline.

This is the offline batch job that:
  1. Reads ContentItems from NeonDB (paginated to bound memory)
  2. Constructs weighted text representations per item
  3. Encodes with all-MiniLM-L6-v2 in async batches
  4. Upserts dense vectors + scalar payloads into the Qdrant HNSW index

The payload stored alongside each vector includes all fields needed by the
Stage-2 ranking pipeline, eliminating the need for a secondary database
lookup during inference. This denormalisation is intentional — the 5x
storage overhead is acceptable to maintain the <10ms Stage-1 latency SLA.

Operational notes:
  • Run after seed_data.py completes
  • Safe to re-run: Qdrant upsert is idempotent (overwrites by point_id)
  • Incremental mode: --since=YYYY-MM-DD re-indexes only updated items
  • Throughput target: 1,000,000 items in < 45 minutes on a 4-vCPU node

Complexity analysis:
  • Embedding: O(N × T / B) — N items, T avg tokens, B batch_size=256
  • Qdrant upsert: O(N × M × log(N)) amortised HNSW insertion
  • Total wall-clock: dominated by embedding (~30ms/256-item batch on CPU)
"""
from __future__ import annotations

import asyncio
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from config.settings import settings
from data_pipeline.database import AsyncSessionLocal, init_db
from data_pipeline.schemas import ContentItem
from retrieval.embeddings import build_content_text, get_embedding_engine
from retrieval.vector_store import (
    batch_upsert_embeddings,
    ensure_collection_exists,
)

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


# ── Payload Extractor ─────────────────────────────────────────────────────────

def extract_qdrant_payload(item: ContentItem) -> dict:
    """
    Build the denormalised Qdrant payload for a ContentItem.

    This payload is returned directly in Stage-1 search results without
    any secondary database lookup, enabling the <10ms retrieval SLA.

    Fields included: all Stage-2 feature engineering inputs + display metadata.
    Fields excluded: user PII, internal IDs, audit timestamps.
    """
    return {
        # ── Identity & display ────────────────────────────────────────────
        "content_id": str(item.id),
        "title": item.title,
        "content_type": item.content_type,
        "thumbnail_url": item.thumbnail_url,
        "media_url": item.media_url,

        # ── Workout attributes ────────────────────────────────────────────
        "workout_type": item.workout_type,
        "duration_minutes": item.duration_minutes,
        "intensity_score": item.intensity_score,
        "calories_burned_estimate": item.calories_burned_estimate,
        "target_muscle_groups": item.target_muscle_groups or [],
        "required_equipment": item.required_equipment or [],

        # ── Nutritional attributes (for recipes) ──────────────────────────
        "sodium_mg": item.sodium_mg,
        "calories_kcal": item.calories_kcal,
        "protein_g": item.protein_g,
        "carbs_g": item.carbs_g,
        "fat_g": item.fat_g,
        "dietary_tags": item.dietary_tags or [],

        # ── Engagement signals (used by Stage-2 feature engineering) ──────
        "global_ctr": item.global_ctr or 0.0,
        "global_completion_rate": item.global_completion_rate or 0.0,
        "total_interactions": item.total_interactions or 0,
    }


# ── Paginated DB Reader ───────────────────────────────────────────────────────

async def fetch_content_page(
    session: AsyncSession,
    offset: int,
    limit: int,
    since: Optional[datetime] = None,
) -> list[ContentItem]:
    """
    Paginated content fetch. Using OFFSET is acceptable here because:
      a) This is a background batch job, not a user-facing endpoint.
      b) Table size is bounded (100K–5M rows typical).
    For tables > 10M rows, switch to keyset pagination (WHERE id > last_id).
    """
    query = (
        select(ContentItem)
        .where(ContentItem.is_published == True)
        .order_by(ContentItem.id)
        .offset(offset)
        .limit(limit)
    )
    if since:
        query = query.where(ContentItem.updated_at >= since)

    result = await session.execute(query)
    return list(result.scalars().all())


# ── Main Indexing Pipeline ────────────────────────────────────────────────────

async def index_catalogue(
    page_size: int = 2_000,
    since: Optional[datetime] = None,
    dry_run: bool = False,
) -> dict:
    """
    Full catalogue embedding and Qdrant indexing pipeline.

    Pipeline stages:
      1. Fetch page of ContentItems from NeonDB
      2. Build text representations
      3. Batch-encode with sentence-transformer
      4. Async-upsert to Qdrant with payload
      5. Repeat until all items processed

    Args:
        page_size: Items per DB fetch + embedding batch
        since: Incremental mode — only re-index items updated after this date
        dry_run: Skip Qdrant writes; useful for embedding throughput benchmarking

    Returns:
        Pipeline statistics dict
    """
    logger.info("=" * 65)
    logger.info("FITNESS REC ENGINE — CONTENT INDEXING PIPELINE")
    logger.info("  Qdrant:     %s:%d", settings.QDRANT_HOST, settings.QDRANT_PORT)
    logger.info("  Collection: %s", settings.QDRANT_COLLECTION_NAME)
    logger.info("  Mode:       %s", "incremental" if since else "full reindex")
    logger.info("  Dry run:    %s", dry_run)
    logger.info("=" * 65)

    # Initialise DB and Qdrant collection
    await init_db()
    if not dry_run:
        await ensure_collection_exists()

    engine = get_embedding_engine()
    logger.info("Embedding engine ready (dim=%d)", engine.embedding_dim)

    t_pipeline_start = time.perf_counter()
    total_items = 0
    total_upserted = 0
    offset = 0
    page_num = 0

    async with AsyncSessionLocal() as session:
        while True:
            page_num += 1
            t_page_start = time.perf_counter()

            # 1. Fetch page
            items = await fetch_content_page(session, offset=offset, limit=page_size, since=since)
            if not items:
                logger.info("No more items — indexing complete.")
                break

            offset += len(items)
            total_items += len(items)

            # 2. Build text representations
            texts = [build_content_text(item) for item in items]
            content_ids = [str(item.id) for item in items]

            # 3. Encode embeddings
            t_enc = time.perf_counter()
            embeddings = await engine.encode_async(texts, normalise=True)
            enc_ms = (time.perf_counter() - t_enc) * 1000

            # 4. Extract payloads (denormalised for Qdrant)
            payloads_map = {str(item.id): extract_qdrant_payload(item) for item in items}
            embeddings_map = dict(zip(content_ids, embeddings))

            # 5. Upsert to Qdrant
            if not dry_run:
                t_upsert = time.perf_counter()
                await batch_upsert_embeddings(
                    embeddings_map=embeddings_map,
                    payloads_map=payloads_map,
                    batch_size=500,
                )
                upsert_ms = (time.perf_counter() - t_upsert) * 1000
                total_upserted += len(items)
            else:
                upsert_ms = 0.0

            page_ms = (time.perf_counter() - t_page_start) * 1000
            throughput = len(items) / (page_ms / 1000)

            logger.info(
                "Page %d | Items: %d | Encode: %.0fms | Upsert: %.0fms | "
                "Throughput: %.0f items/sec | Total: %d",
                page_num, len(items), enc_ms, upsert_ms, throughput, total_items,
            )

    total_elapsed = time.perf_counter() - t_pipeline_start

    stats = {
        "total_items_processed": total_items,
        "total_items_upserted": total_upserted,
        "pages_processed": page_num,
        "elapsed_seconds": round(total_elapsed, 2),
        "avg_throughput_items_per_sec": round(total_items / total_elapsed, 1),
        "collection": settings.QDRANT_COLLECTION_NAME,
        "embedding_dim": engine.embedding_dim,
        "dry_run": dry_run,
    }

    logger.info("=" * 65)
    logger.info("✅ INDEXING COMPLETE")
    for k, v in stats.items():
        logger.info("  %-40s %s", k + ":", v)
    logger.info("=" * 65)

    return stats


# ── Entry Point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Embed and index content catalogue into Qdrant")
    parser.add_argument(
        "--page-size", type=int, default=2_000,
        help="Items per batch (default: 2000)"
    )
    parser.add_argument(
        "--since", type=str, default=None,
        help="Incremental mode: only index items updated since YYYY-MM-DD"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Run embeddings but skip Qdrant upserts"
    )
    args = parser.parse_args()

    since_dt = None
    if args.since:
        since_dt = datetime.fromisoformat(args.since).replace(tzinfo=timezone.utc)

    asyncio.run(
        index_catalogue(
            page_size=args.page_size,
            since=since_dt,
            dry_run=args.dry_run,
        )
    )
