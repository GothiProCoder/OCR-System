"""
Database Connection Module
==========================
Provides both synchronous and asynchronous database connections.
Uses SQLAlchemy 2.0+ patterns with proper session management.

Usage:
    # Sync (for simple operations)
    from database.connection import get_db
    with get_db() as db:
        db.query(Document).all()
    
    # Async (for FastAPI endpoints)
    from database.connection import get_async_db
    async with get_async_db() as db:
        await db.execute(select(Document))
"""

from sqlalchemy import create_engine, event
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import sessionmaker, Session, declarative_base
from sqlalchemy.pool import QueuePool
from contextlib import contextmanager, asynccontextmanager
from typing import Generator, AsyncGenerator
import logging

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import settings

# Configure logging
logger = logging.getLogger(__name__)

# Base class for all ORM models
Base = declarative_base()

# =============================================================================
# SYNCHRONOUS ENGINE & SESSION (For scripts, migrations, simple operations)
# =============================================================================

sync_engine = create_engine(
    settings.database_url,
    poolclass=QueuePool,
    pool_size=5,
    max_overflow=10,
    pool_pre_ping=True,  # Verify connections before use
    pool_recycle=3600,   # Recycle connections after 1 hour
    echo=settings.DEBUG and settings.is_development,  # SQL logging in dev
)

SyncSessionLocal = sessionmaker(
    bind=sync_engine,
    autocommit=False,
    autoflush=False,
    expire_on_commit=False,
)


@contextmanager
def get_db() -> Generator[Session, None, None]:
    """
    Synchronous database session context manager.
    
    Usage:
        with get_db() as db:
            documents = db.query(Document).all()
    """
    session = SyncSessionLocal()
    try:
        yield session
        session.commit()
    except Exception as e:
        session.rollback()
        logger.error(f"Database error: {e}")
        raise
    finally:
        session.close()


def get_sync_session() -> Session:
    """
    Get a synchronous session for dependency injection.
    Caller is responsible for closing.
    """
    return SyncSessionLocal()


# =============================================================================
# ASYNCHRONOUS ENGINE & SESSION (For FastAPI async endpoints)
# =============================================================================

async_engine = create_async_engine(
    settings.async_database_url,
    pool_size=5,
    max_overflow=10,
    pool_pre_ping=True,
    pool_recycle=3600,
    echo=settings.DEBUG and settings.is_development,
)

AsyncSessionLocal = async_sessionmaker(
    bind=async_engine,
    class_=AsyncSession,
    autocommit=False,
    autoflush=False,
    expire_on_commit=False,
)


@asynccontextmanager
async def get_async_db() -> AsyncGenerator[AsyncSession, None]:
    """
    Asynchronous database session context manager.
    
    Usage:
        async with get_async_db() as db:
            result = await db.execute(select(Document))
            documents = result.scalars().all()
    """
    session = AsyncSessionLocal()
    try:
        yield session
        await session.commit()
    except Exception as e:
        await session.rollback()
        logger.error(f"Database error: {e}")
        raise
    finally:
        await session.close()


async def get_async_session() -> AsyncGenerator[AsyncSession, None]:
    """
    FastAPI dependency for async database sessions.
    
    Usage in FastAPI:
        @app.get("/documents")
        async def get_documents(db: AsyncSession = Depends(get_async_session)):
            ...
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


# =============================================================================
# DATABASE LIFECYCLE FUNCTIONS
# =============================================================================

def create_all_tables():
    """
    Create all tables defined in models.
    Use for initial setup or testing.
    """
    from database.models import Base  # Import to register all models
    Base.metadata.create_all(bind=sync_engine)
    logger.info("All database tables created successfully")


async def async_create_all_tables():
    """
    Async version of create_all_tables.
    """
    from database.models import Base
    async with async_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("All database tables created successfully (async)")


def drop_all_tables():
    """
    Drop all tables. USE WITH CAUTION!
    """
    from database.models import Base
    Base.metadata.drop_all(bind=sync_engine)
    logger.warning("All database tables dropped!")


async def check_database_connection() -> bool:
    """
    Verify database connectivity.
    Returns True if connection successful.
    """
    try:
        async with async_engine.connect() as conn:
            await conn.execute("SELECT 1")
        logger.info("Database connection verified")
        return True
    except Exception as e:
        logger.error(f"Database connection failed: {e}")
        return False


def dispose_engines():
    """
    Dispose of all engine connections.
    Call during application shutdown.
    """
    sync_engine.dispose()
    logger.info("Sync engine disposed")


async def async_dispose_engines():
    """
    Async version of dispose_engines.
    """
    await async_engine.dispose()
    logger.info("Async engine disposed")
