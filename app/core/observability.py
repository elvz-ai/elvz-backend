"""
Observability and monitoring integrations.
LangFuse for LLM tracing, Sentry for error tracking.
"""

from contextlib import contextmanager
from functools import wraps
from typing import Any, Callable, Optional

import structlog

from app.core.config import settings

logger = structlog.get_logger(__name__)

# Lazy imports for optional dependencies
_langfuse_client = None
_sentry_initialized = False


def get_langfuse():
    """Get or create LangFuse client."""
    global _langfuse_client

    if _langfuse_client is None and settings.langfuse_public_key:
        try:
            from langfuse import Langfuse

            _langfuse_client = Langfuse(
                public_key=settings.langfuse_public_key,
                secret_key=settings.langfuse_secret_key,
                host=settings.langfuse_host,
            )
            logger.info("LangFuse client initialized")
        except ImportError:
            logger.warning("LangFuse not installed, skipping initialization")
        except Exception as e:
            logger.error("Failed to initialize LangFuse", error=str(e))

    return _langfuse_client


def get_langfuse_handler(
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
    trace_name: Optional[str] = None,
    metadata: Optional[dict] = None,
):
    """
    Get LangFuse callback handler for LangChain/LangGraph tracing.

    Args:
        user_id: User identifier for the trace
        session_id: Session/conversation identifier
        trace_name: Name for the trace
        metadata: Additional metadata to attach

    Returns:
        CallbackHandler or None if LangFuse not configured
    """
    if not settings.langfuse_public_key:
        return None

    try:
        from langfuse.callback import CallbackHandler

        return CallbackHandler(
            public_key=settings.langfuse_public_key,
            secret_key=settings.langfuse_secret_key,
            host=settings.langfuse_host,
            user_id=user_id,
            session_id=session_id,
            trace_name=trace_name,
            metadata=metadata or {},
        )
    except ImportError:
        logger.warning("LangFuse not installed")
        return None
    except Exception as e:
        logger.error("Failed to create LangFuse handler", error=str(e))
        return None


def init_sentry() -> None:
    """Initialize Sentry error tracking."""
    global _sentry_initialized

    if _sentry_initialized or not settings.sentry_dsn:
        return

    try:
        import sentry_sdk
        from sentry_sdk.integrations.fastapi import FastApiIntegration
        from sentry_sdk.integrations.sqlalchemy import SqlalchemyIntegration
        from sentry_sdk.integrations.redis import RedisIntegration

        sentry_sdk.init(
            dsn=settings.sentry_dsn,
            environment=settings.sentry_environment,
            traces_sample_rate=settings.sentry_traces_sample_rate,
            integrations=[
                FastApiIntegration(transaction_style="endpoint"),
                SqlalchemyIntegration(),
                RedisIntegration(),
            ],
            # Don't send PII
            send_default_pii=False,
        )

        _sentry_initialized = True
        logger.info(
            "Sentry initialized",
            environment=settings.sentry_environment,
            sample_rate=settings.sentry_traces_sample_rate,
        )
    except ImportError:
        logger.warning("Sentry SDK not installed, skipping initialization")
    except Exception as e:
        logger.error("Failed to initialize Sentry", error=str(e))


def capture_exception(error: Exception, context: Optional[dict] = None) -> None:
    """
    Capture an exception to Sentry with additional context.

    Args:
        error: The exception to capture
        context: Additional context to attach
    """
    if not settings.sentry_dsn:
        return

    try:
        import sentry_sdk

        if context:
            sentry_sdk.set_context("additional", context)

        sentry_sdk.capture_exception(error)
    except ImportError:
        pass
    except Exception as e:
        logger.error("Failed to capture exception to Sentry", error=str(e))


def set_user_context(user_id: str, email: Optional[str] = None) -> None:
    """
    Set user context for Sentry.

    Args:
        user_id: User identifier
        email: User email (optional)
    """
    if not settings.sentry_dsn:
        return

    try:
        import sentry_sdk

        sentry_sdk.set_user({"id": user_id, "email": email})
    except ImportError:
        pass


def set_conversation_context(
    conversation_id: str,
    thread_id: str,
    intent: Optional[str] = None,
    platforms: Optional[list] = None,
) -> None:
    """
    Set conversation context for Sentry.

    Args:
        conversation_id: Conversation identifier
        thread_id: LangGraph thread identifier
        intent: Current intent type
        platforms: Platforms involved
    """
    if not settings.sentry_dsn:
        return

    try:
        import sentry_sdk

        sentry_sdk.set_context("conversation", {
            "conversation_id": conversation_id,
            "thread_id": thread_id,
            "intent": intent,
            "platforms": platforms,
        })
    except ImportError:
        pass


@contextmanager
def trace_span(name: str, metadata: Optional[dict] = None):
    """
    Context manager for creating trace spans.
    Works with both Sentry and LangFuse.

    Args:
        name: Span name
        metadata: Additional metadata
    """
    langfuse = get_langfuse()

    # Start LangFuse span
    lf_span = None
    if langfuse:
        try:
            lf_span = langfuse.span(name=name, metadata=metadata or {})
        except Exception:
            pass

    # Start Sentry span
    sentry_span = None
    if settings.sentry_dsn:
        try:
            import sentry_sdk
            sentry_span = sentry_sdk.start_span(op=name)
            if metadata:
                for key, value in metadata.items():
                    sentry_span.set_data(key, value)
        except ImportError:
            pass

    try:
        yield
    except Exception as e:
        if lf_span:
            try:
                lf_span.update(level="ERROR", status_message=str(e))
            except Exception:
                pass
        raise
    finally:
        if lf_span:
            try:
                lf_span.end()
            except Exception:
                pass
        if sentry_span:
            try:
                sentry_span.finish()
            except Exception:
                pass


def observe_llm_call(
    name: str = "llm_call",
    model: Optional[str] = None,
    capture_input: bool = True,
    capture_output: bool = True,
):
    """
    Decorator for observing LLM calls.

    Args:
        name: Name for the observation
        model: Model name
        capture_input: Whether to capture input
        capture_output: Whether to capture output
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def async_wrapper(*args, **kwargs) -> Any:
            langfuse = get_langfuse()

            generation = None
            if langfuse:
                try:
                    input_data = {"args": str(args)[:500], "kwargs": str(kwargs)[:500]} if capture_input else {}
                    generation = langfuse.generation(
                        name=name,
                        model=model,
                        input=input_data,
                    )
                except Exception:
                    pass

            try:
                result = await func(*args, **kwargs)

                if generation and capture_output:
                    try:
                        output_data = str(result)[:1000] if result else None
                        generation.end(output=output_data)
                    except Exception:
                        pass

                return result

            except Exception as e:
                if generation:
                    try:
                        generation.end(level="ERROR", status_message=str(e))
                    except Exception:
                        pass
                raise

        @wraps(func)
        def sync_wrapper(*args, **kwargs) -> Any:
            return func(*args, **kwargs)

        # Return appropriate wrapper based on function type
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper

    return decorator


def flush_observability() -> None:
    """Flush all pending observability data."""
    langfuse = get_langfuse()
    if langfuse:
        try:
            langfuse.flush()
        except Exception as e:
            logger.error("Failed to flush LangFuse", error=str(e))
