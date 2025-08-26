
from __future__ import annotations

import os
import logging
import re
from contextlib import asynccontextmanager, contextmanager
from typing import Iterator, AsyncIterator

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

# ----------------------------------------------------------------------------
# Logging
# ----------------------------------------------------------------------------
logger = logging.getLogger(__name__)
if not logger.handlers:
    # Basic configuration if the application hasn't configured logging yet
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


def _bool_env(name: str, default: bool = False) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return v.lower() in {"1", "true", "yes", "on"}


def _sanitize_url(url: str) -> str:
    """Redact password for logging."""
    return re.sub(r":([^:@/]+)@", ":*****@", url)


DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+psycopg://postgres:root@localhost:5432/voice_face_rag",
)
ECHO = _bool_env("DB_ECHO", False)
POOL_SIZE = int(os.getenv("DB_POOL_SIZE", "5"))
POOL_TIMEOUT = int(os.getenv("DB_POOL_TIMEOUT", "30"))
POOL_RECYCLE = int(os.getenv("DB_POOL_RECYCLE", "1800"))

IS_ASYNC = DATABASE_URL.startswith("postgresql+asyncpg")

logger.info("DB url detected (async=%s): %s", IS_ASYNC, _sanitize_url(DATABASE_URL))

# ----------------------------------------------------------------------------
# Engines & Session factories
# ----------------------------------------------------------------------------
if IS_ASYNC:
    async_engine = create_async_engine(
        DATABASE_URL,
        echo=ECHO,
        pool_pre_ping=True,
        pool_recycle=POOL_RECYCLE,
    )
    AsyncSessionLocal = async_sessionmaker(
        bind=async_engine,
        expire_on_commit=False,
        autoflush=False,
    )

    # For typing-friendly import
    engine = None  # type: ignore
    SessionLocal = None  # type: ignore
else:
    engine = create_engine(
        DATABASE_URL,
        echo=ECHO,
        pool_pre_ping=True,
        pool_size=POOL_SIZE,
        pool_timeout=POOL_TIMEOUT,
        pool_recycle=POOL_RECYCLE,
    )
    SessionLocal = sessionmaker(bind=engine, expire_on_commit=False, autoflush=False)

    async_engine = None  # type: ignore
    AsyncSessionLocal = None  # type: ignore


# ----------------------------------------------------------------------------
# Extension bootstrap helpers (optional but handy in dev/test)
# ----------------------------------------------------------------------------
EXTENSION_SQL = """
CREATE EXTENSION IF NOT EXISTS pgcrypto;
CREATE EXTENSION IF NOT EXISTS vector;
"""


def ensure_extensions_sync() -> None:
    """Ensure pgcrypto/pgvector exist (sync engine only)."""
    if engine is None:
        return
    logger.info("Ensuring PostgreSQL extensions (sync): pgcrypto, vector")
    with engine.begin() as conn:
        conn.exec_driver_sql(EXTENSION_SQL)


async def ensure_extensions_async() -> None:
    """Ensure pgcrypto/pgvector exist (async engine only)."""
    if async_engine is None:
        return
    logger.info("Ensuring PostgreSQL extensions (async): pgcrypto, vector")
    async with async_engine.begin() as conn:
        await conn.exec_driver_sql(EXTENSION_SQL)


# ----------------------------------------------------------------------------
# FastAPI-friendly dependency providers
# ----------------------------------------------------------------------------
@contextmanager
def get_db() -> Iterator[Session]:
    """Yield a sync Session and close it after use."""
    if SessionLocal is None:
        raise RuntimeError("get_db() called but DATABASE_URL is async; use get_async_db() instead.")
    db = SessionLocal()
    logger.debug("Opened DB session (sync)")
    try:
        yield db
    finally:
        db.close()
        logger.debug("Closed DB session (sync)")


@asynccontextmanager
async def get_async_db() -> AsyncIterator[AsyncSession]:
    """Yield an AsyncSession and close it after use."""
    if AsyncSessionLocal is None:
        raise RuntimeError("get_async_db() called but DATABASE_URL is sync; use get_db() instead.")
    db = AsyncSessionLocal()
    logger.debug("Opened DB session (async)")
    try:
        yield db
    finally:
        await db.close()
        logger.debug("Closed DB session (async)")


# ----------------------------------------------------------------------------
# App lifecycle helpers
# ----------------------------------------------------------------------------

def init_db() -> None:
    """Optional: call at startup for sync apps."""
    if engine is None:
        return
    ensure_extensions_sync()
    logger.info("DB init complete (sync)")


async def init_db_async() -> None:
    """Optional: call at startup for async apps."""
    if async_engine is None:
        return
    await ensure_extensions_async()
    logger.info("DB init complete (async)")


__all__ = [
    "engine",
    "async_engine",
    "SessionLocal",
    "AsyncSessionLocal",
    "get_db",
    "get_async_db",
    "init_db",
    "init_db_async",
]
