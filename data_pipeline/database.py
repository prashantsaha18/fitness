"""
data_pipeline/database.py
──────────────────────────
Async SQLAlchemy engine wired to NeonDB (Postgres).

Connection pool sizing rationale:
  • NeonDB serverless branches impose a 100-connection ceiling per project.
  • pool_size=20 + max_overflow=10 → max 30 active connections per replica.
  • With 2 API replicas this leaves a 40-connection buffer for migrations/admin.

pool_pre_ping=True:
  • NeonDB scales to zero; stale sockets silently timeout after ~5 min idle.
  • pre_ping issues a SELECT 1 before handing a connection to a coroutine,
    adding ~0.2ms overhead but eliminating OperationalError on cold wakeup.
"""
from __future__ import annotations

import contextlib
import logging
from collections.abc import AsyncGenerator

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy import event, text

from config.settings import settings
from data_pipeline.schemas import Base

logger = logging.getLogger(__name__)

# ── Engine ────────────────────────────────────────────────────────────────────

def _build_engine() -> AsyncEngine:
    engine = create_async_engine(
        settings.DATABASE_URL,
        pool_size=settings.DB_POOL_SIZE,
        max_overflow=settings.DB_MAX_OVERFLOW,
        pool_timeout=settings.DB_POOL_TIMEOUT,
        pool_pre_ping=True,
        pool_recycle=1800,          # recycle connections every 30 min
        echo=settings.DB_ECHO,
        connect_args={
            "ssl": "require",
            "server_settings": {
                "application_name": settings.SERVICE_NAME,
                "jit": "off",       # JIT warm-up latency unacceptable at P99
            },
        },
    )
    return engine


engine: AsyncEngine = _build_engine()

AsyncSessionLocal: async_sessionmaker[AsyncSession] = async_sessionmaker(
    bind=engine,
    expire_on_commit=False,
    autoflush=False,
    autocommit=False,
    class_=AsyncSession,
)


# ── Session Dependency (FastAPI DI) ──────────────────────────────────────────

@contextlib.asynccontextmanager
async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """
    Context-manager session factory.
    Always use this instead of raw AsyncSessionLocal() to guarantee
    that the session is correctly closed on exception paths.
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency injection target."""
    async with get_db_session() as session:
        yield session


# ── Schema Bootstrap ──────────────────────────────────────────────────────────

async def init_db() -> None:
    """
    Idempotent schema bootstrap — safe to call on every startup.
    In production, prefer Alembic migrations; this is a convenience helper
    for CI/dev environments.
    """
    async with engine.begin() as conn:
        # Install pgcrypto for UUID generation on the server side
        await conn.execute(text('CREATE EXTENSION IF NOT EXISTS "pgcrypto"'))
        await conn.run_sync(Base.metadata.create_all)
        logger.info("Database schema initialised successfully.")


async def dispose_engine() -> None:
    """Graceful shutdown — drain connection pool without killing in-flight queries."""
    await engine.dispose()
    logger.info("Database connection pool disposed.")
