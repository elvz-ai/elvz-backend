"""
LangGraph Checkpointer for conversation state persistence.
Uses PostgreSQL for durable state storage across sessions.
"""

from typing import Optional

import structlog
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

from app.core.config import settings

logger = structlog.get_logger(__name__)

# Global checkpointer instance
_checkpointer: Optional[AsyncPostgresSaver] = None


def _get_sync_connection_string() -> str:
    """Get PostgreSQL connection string (without async driver)."""
    # LangGraph checkpointer uses psycopg, not asyncpg
    return (
        f"postgresql://{settings.postgres_user}:{settings.postgres_password}"
        f"@{settings.postgres_host}:{settings.postgres_port}/{settings.postgres_database}"
    )


async def get_checkpointer() -> AsyncPostgresSaver:
    """
    Get or create the LangGraph PostgreSQL checkpointer.

    The checkpointer persists conversation state to PostgreSQL,
    enabling conversation resumption across sessions.

    Returns:
        AsyncPostgresSaver: Configured checkpointer instance
    """
    global _checkpointer

    if _checkpointer is None:
        connection_string = _get_sync_connection_string()

        try:
            _checkpointer = AsyncPostgresSaver.from_conn_string(connection_string)
            await _checkpointer.setup()

            logger.info(
                "LangGraph checkpointer initialized",
                host=settings.postgres_host,
                database=settings.postgres_database,
            )
        except Exception as e:
            logger.error(
                "Failed to initialize checkpointer",
                error=str(e),
                host=settings.postgres_host,
            )
            raise

    return _checkpointer


async def close_checkpointer() -> None:
    """Close the checkpointer connection."""
    global _checkpointer

    if _checkpointer is not None:
        # AsyncPostgresSaver handles connection cleanup internally
        _checkpointer = None
        logger.info("LangGraph checkpointer closed")


class CheckpointerManager:
    """
    Context manager for checkpointer lifecycle.
    Use for standalone scripts or testing.
    """

    def __init__(self):
        self.checkpointer: Optional[AsyncPostgresSaver] = None

    async def __aenter__(self) -> AsyncPostgresSaver:
        self.checkpointer = await get_checkpointer()
        return self.checkpointer

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await close_checkpointer()


# Convenience functions for thread management

async def get_thread_state(thread_id: str) -> Optional[dict]:
    """
    Get the current state for a thread.

    Args:
        thread_id: LangGraph thread identifier

    Returns:
        Current state dict or None if not found
    """
    checkpointer = await get_checkpointer()

    try:
        config = {"configurable": {"thread_id": thread_id}}
        checkpoint = await checkpointer.aget(config)

        if checkpoint:
            return checkpoint.get("channel_values", {})
        return None

    except Exception as e:
        logger.error("Failed to get thread state", thread_id=thread_id, error=str(e))
        return None


async def list_thread_history(thread_id: str, limit: int = 10) -> list[dict]:
    """
    List checkpoint history for a thread.

    Args:
        thread_id: LangGraph thread identifier
        limit: Maximum number of checkpoints to return

    Returns:
        List of checkpoint metadata
    """
    checkpointer = await get_checkpointer()

    try:
        config = {"configurable": {"thread_id": thread_id}}
        history = []

        async for checkpoint in checkpointer.alist(config, limit=limit):
            history.append({
                "checkpoint_id": checkpoint.config.get("configurable", {}).get("checkpoint_id"),
                "thread_id": thread_id,
                "timestamp": checkpoint.metadata.get("created_at") if checkpoint.metadata else None,
            })

        return history

    except Exception as e:
        logger.error("Failed to list thread history", thread_id=thread_id, error=str(e))
        return []
