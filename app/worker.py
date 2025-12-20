"""
Celery worker configuration for background tasks.
"""

from celery import Celery

from app.core.config import settings

# Create Celery app
celery_app = Celery(
    "elvz_worker",
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend,
)

# Celery configuration
celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_time_limit=300,  # 5 minutes max per task
    task_soft_time_limit=240,  # Soft limit for graceful shutdown
    worker_prefetch_multiplier=1,
    worker_concurrency=4,
)

# Task routing
celery_app.conf.task_routes = {
    "app.tasks.analytics.*": {"queue": "analytics"},
    "app.tasks.content.*": {"queue": "content"},
    "app.tasks.cleanup.*": {"queue": "cleanup"},
}

# Scheduled tasks (Celery Beat)
celery_app.conf.beat_schedule = {
    "cleanup-expired-sessions": {
        "task": "app.tasks.cleanup.cleanup_expired_sessions",
        "schedule": 3600.0,  # Every hour
    },
    "aggregate-daily-usage": {
        "task": "app.tasks.analytics.aggregate_daily_usage",
        "schedule": 86400.0,  # Daily
    },
    "warm-cache": {
        "task": "app.tasks.cache.warm_cache",
        "schedule": 1800.0,  # Every 30 minutes
    },
}


@celery_app.task
def example_task(x: int, y: int) -> int:
    """Example task for testing."""
    return x + y


# Import task modules to register them
# from app.tasks import analytics, content, cleanup

