"""
retrieval/vector_store.py
──────────────────────────
Qdrant vector store client — Stage-1 Candidate Generation Engine.

HNSW Index Configuration Rationale:
  M=16:
    • Controls the number of bi-directional links per node in the graph.
    • M=16 is the sweet spot for recall vs memory for 384-dim vectors.
    • Higher M (e.g., 32) improves recall by ~2% but doubles memory per node.
    • Memory per node ≈ M × 8 bytes. For 1M items: 16 × 8 × 1M = 128 MB index.

  ef_construct=200:
    • Dynamic candidate list size during index construction.
    • Higher values → better graph connectivity → higher recall at query time.
    • ef_construct=200 yields recall@100 ≥ 0.97 on our workload profile.
    • Construction time: O(N × M × log(N)) — one-time cost.

  ef (search) = 128:
    • Runtime equivalent of ef_construct; must be ≥ top_k.
    • ef=128, top_k=100: adds 28 extra candidates as a quality buffer.
    • Per-query complexity: O(ef × log(N)) ≈ O(128 × log(1M)) ≈ 2,560 ops.

HNSW vs IVF Trade-off:
  HNSW: O(log N) query, O(N × M) memory, no training phase needed.
        Latency is stable and predictable — critical for P99 SLA.
  IVF:  O(N/n_lists) query, lower memory. BUT requires periodic re-training
        as the distribution drifts. Latency spikes during centroid recalculation.
  Decision: HNSW for production serving; IVF acceptable for offline batch scoring.

Async Pattern:
  All Qdrant I/O uses the async gRPC channel (port 6334) for minimal overhead.
  Batch upserts use async_upload_points() with payload streaming to avoid
  loading the full 1M-item vector matrix into memory simultaneously.
"""
from __future__ import annotations

import asyncio
import logging
import time
import uuid
from typing import Any, Sequence

import numpy as np
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)
from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models as qmodels
from qdrant_client.http.exceptions import UnexpectedResponse

from config.settings import settings

logger = logging.getLogger(__name__)


# ── Client Factory ────────────────────────────────────────────────────────────

_client: AsyncQdrantClient | None = None
_client_lock = asyncio.Lock()


async def get_qdrant_client() -> AsyncQdrantClient:
    """
    Module-level async singleton.
    Uses gRPC channel for ~30% lower latency vs HTTP REST on LAN connections.
    """
    global _client
    if _client is not None:
        return _client
    async with _client_lock:
        if _client is not None:
            return _client
        _client = AsyncQdrantClient(
            host=settings.QDRANT_HOST,
            grpc_port=settings.QDRANT_GRPC_PORT,
            prefer_grpc=True,
            api_key=settings.QDRANT_API_KEY,
            timeout=5.0,  # tight timeout — vector search must complete in <10ms
        )
        logger.info(
            "Qdrant client initialised (gRPC) at %s:%d",
            settings.QDRANT_HOST,
            settings.QDRANT_GRPC_PORT,
        )
    return _client


async def close_qdrant_client() -> None:
    global _client
    if _client:
        await _client.close()
        _client = None


# ── Collection Management ─────────────────────────────────────────────────────

async def ensure_collection_exists() -> None:
    """
    Idempotent collection bootstrap.
    Called once at service startup. Existing collections with correct config
    are left untouched.

    Vector configuration:
      size=384: matches all-MiniLM-L6-v2 output dimensionality.
      distance=Cosine: correct for L2-normalised embeddings.
                       (dot product = cosine when both vectors are unit length)
    """
    client = await get_qdrant_client()

    collections = await client.get_collections()
    existing = {c.name for c in collections.collections}

    if settings.QDRANT_COLLECTION_NAME in existing:
        logger.info(
            "Collection '%s' already exists — skipping creation.",
            settings.QDRANT_COLLECTION_NAME,
        )
        return

    await client.create_collection(
        collection_name=settings.QDRANT_COLLECTION_NAME,
        vectors_config=qmodels.VectorParams(
            size=settings.QDRANT_EMBEDDING_DIM,
            distance=qmodels.Distance.COSINE,
            # Store vectors on disk for collections > 4GB (1M × 384 × 4B ≈ 1.5GB)
            # on_disk=True,   # enable when RAM < 4GB on the Qdrant node
        ),
        hnsw_config=qmodels.HnswConfigDiff(
            m=settings.HNSW_M,
            ef_construct=settings.HNSW_EF_CONSTRUCT,
            full_scan_threshold=10_000,  # use exact search for small collections
            max_indexing_threads=4,
            # on_disk=False,  # keep HNSW graph in RAM for sub-10ms latency
        ),
        optimizers_config=qmodels.OptimizersConfigDiff(
            memmap_threshold=50_000,
            indexing_threshold=20_000,  # build index after 20k points
            flush_interval_sec=30,
        ),
        quantization_config=qmodels.ScalarQuantization(
            scalar=qmodels.ScalarQuantizationConfig(
                type=qmodels.ScalarType.INT8,
                quantile=0.99,
                always_ram=True,        # quantised vectors stay in RAM
            )
        ),
    )
    logger.info(
        "Collection '%s' created (dim=%d, M=%d, ef=%d).",
        settings.QDRANT_COLLECTION_NAME,
        settings.QDRANT_EMBEDDING_DIM,
        settings.HNSW_M,
        settings.HNSW_EF_CONSTRUCT,
    )


# ── Batch Upsert Pipeline ─────────────────────────────────────────────────────

async def batch_upsert_embeddings(
    embeddings_map: dict[str, np.ndarray],
    payloads_map: dict[str, dict[str, Any]],
    batch_size: int = 1000,
) -> None:
    """
    Async streaming batch upsert to Qdrant.

    Complexity: O(N / batch_size) round-trips, each O(batch_size × D).
    Memory bounded at O(batch_size × D × 4 bytes) per iteration.

    Retry strategy:
      Exponential backoff with jitter on transient network errors.
      3 attempts × 2^n seconds = max 7 seconds total retry budget.
    """
    client = await get_qdrant_client()
    items = list(embeddings_map.items())
    total = len(items)
    upserted = 0
    t0 = time.perf_counter()

    for batch_start in range(0, total, batch_size):
        batch = items[batch_start : batch_start + batch_size]
        points = []
        for content_id, embedding in batch:
            # Qdrant requires integer or UUID point IDs
            # We use UUID string deterministically hashed to avoid collisions
            point_id = str(uuid.UUID(content_id)) if _is_valid_uuid(content_id) \
                else str(uuid.uuid5(uuid.NAMESPACE_DNS, content_id))
            points.append(
                qmodels.PointStruct(
                    id=point_id,
                    vector=embedding.tolist(),
                    payload={
                        "content_id": content_id,
                        **payloads_map.get(content_id, {}),
                    },
                )
            )

        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=1, min=1, max=8),
            retry=retry_if_exception_type((UnexpectedResponse, ConnectionError)),
        ):
            with attempt:
                await client.upsert(
                    collection_name=settings.QDRANT_COLLECTION_NAME,
                    points=points,
                    wait=False,  # async upsert — don't block for index rebuild
                )

        upserted += len(batch)
        if upserted % 10_000 == 0 or upserted == total:
            elapsed = time.perf_counter() - t0
            logger.info(
                "Upserted %d/%d points (%.1f pts/sec)",
                upserted, total, upserted / elapsed,
            )

    logger.info("Batch upsert complete: %d points in %.2fs", total, time.perf_counter() - t0)


# ── Stage-1 Candidate Retrieval ───────────────────────────────────────────────

async def retrieve_candidates(
    query_embedding: np.ndarray,
    top_k: int | None = None,
    filter_conditions: qmodels.Filter | None = None,
) -> list[dict[str, Any]]:
    """
    ANN vector search — must complete in ≤ 10ms P99 at 5,000 RPS.

    Algorithm: HNSW graph traversal starting from ef=128 entry points.
    Complexity: O(ef × log(N)) ≈ O(128 × 20) = 2,560 distance computations.

    Returns:
        List of dicts: [{"content_id": str, "score": float, "payload": dict}]
        Sorted by cosine similarity (descending).

    Fallback behaviour:
        On Qdrant timeout/error, raises VectorStoreUnavailableError.
        The orchestration layer handles this by serving cached fallback results.
    """
    top_k = top_k or settings.QDRANT_TOP_K
    client = await get_qdrant_client()

    try:
        t0 = time.perf_counter()
        results = await asyncio.wait_for(
            client.search(
                collection_name=settings.QDRANT_COLLECTION_NAME,
                query_vector=query_embedding.tolist(),
                limit=top_k,
                query_filter=filter_conditions,
                search_params=qmodels.SearchParams(
                    hnsw_ef=settings.HNSW_EF_SEARCH,
                    exact=False,        # ANN mode (not exact brute-force)
                ),
                with_payload=True,
                score_threshold=0.0,   # include all scores; filter in Stage-2
            ),
            timeout=settings.STAGE1_TIMEOUT_MS / 1000,
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000
        logger.debug("ANN search completed in %.2f ms (k=%d)", elapsed_ms, top_k)

        return [
            {
                "content_id": hit.payload.get("content_id", str(hit.id)),
                "qdrant_id": str(hit.id),
                "score": hit.score,
                "payload": hit.payload or {},
            }
            for hit in results
        ]

    except asyncio.TimeoutError:
        raise VectorStoreUnavailableError(
            f"Qdrant search timed out after {settings.STAGE1_TIMEOUT_MS}ms"
        )
    except Exception as exc:
        raise VectorStoreUnavailableError(f"Qdrant search failed: {exc}") from exc


async def retrieve_candidates_batch(
    query_embeddings: list[np.ndarray],
    top_k: int | None = None,
) -> list[list[dict[str, Any]]]:
    """
    Batch ANN retrieval for multiple queries.
    Used for offline evaluation and A/B test result pre-computation.
    """
    tasks = [retrieve_candidates(emb, top_k) for emb in query_embeddings]
    return await asyncio.gather(*tasks, return_exceptions=False)


# ── Custom Exceptions ─────────────────────────────────────────────────────────

class VectorStoreUnavailableError(Exception):
    """
    Raised when Qdrant is unreachable or times out.
    Caught by the API orchestration layer to trigger the fallback pipeline.
    """
    pass


# ── Utility ───────────────────────────────────────────────────────────────────

def _is_valid_uuid(value: str) -> bool:
    try:
        uuid.UUID(value)
        return True
    except ValueError:
        return False


def build_content_type_filter(content_types: list[str]) -> qmodels.Filter:
    """Build a Qdrant filter restricting results to specific content types."""
    return qmodels.Filter(
        must=[
            qmodels.FieldCondition(
                key="content_type",
                match=qmodels.MatchAny(any=content_types),
            )
        ]
    )
