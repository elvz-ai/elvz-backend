"""
FastAPI application main entry point.
"""

from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import (
    assistant_router,
    chat_router,
    copywriter_router,
    seo_router,
    social_media_router,
)
from app.api.websocket import websocket_endpoint
from app.core.config import settings
from app.core.cache import cache
from app.core.database import init_db, close_db
from app.core.vector_store import vector_store

logger = structlog.get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.
    Handles startup and shutdown events.
    """
    # Startup
    logger.info("Starting Elvz.ai Backend", environment=settings.environment)
    
    # Connect to Redis
    try:
        await cache.connect()
        logger.info("Redis connected")
    except Exception as e:
        logger.warning("Redis connection failed", error=str(e))
    
    # Initialize database
    try:
        await init_db()
        logger.info("Database initialized")
    except Exception as e:
        logger.warning("Database initialization failed", error=str(e))
    
    # Pre-connect Pinecone (saves ~3 sec on first request)
    try:
        await vector_store.connect()
        logger.info("Pinecone connected")
    except Exception as e:
        logger.warning("Pinecone connection failed (will retry on first request)", error=str(e))
    
    # Register Elves with orchestrator
    await register_elves()
    
    yield
    
    # Shutdown
    logger.info("Shutting down Elvz.ai Backend")
    
    await cache.disconnect()
    await close_db()


async def register_elves():
    """Register all Elf agents with the platform orchestrator."""
    from app.agents.platform_orchestrator import orchestrator
    from app.agents.platform_orchestrator.intent_classifier import ElfType
    from app.agents.elves import (
        SocialMediaManagerElf,
        SEOOptimizerElf,
        CopywriterElf,
        AIAssistantElf,
    )
    
    orchestrator.register_elf(ElfType.SOCIAL_MEDIA, SocialMediaManagerElf())
    orchestrator.register_elf(ElfType.SEO, SEOOptimizerElf())
    orchestrator.register_elf(ElfType.COPYWRITER, CopywriterElf())
    orchestrator.register_elf(ElfType.ASSISTANT, AIAssistantElf())
    
    logger.info("All Elves registered")


# Create FastAPI application
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="""
    Elvz.ai Multi-Agent AI Platform Backend
    
    A platform providing specialized AI assistants (Elves) for:
    - Social Media Management
    - SEO Optimization
    - Content Writing
    - Personal Assistance
    
    Each Elf orchestrates multiple mini-agents to deliver comprehensive results.
    """,
    docs_url="/docs" if settings.environment != "production" else None,
    redoc_url="/redoc" if settings.environment != "production" else None,
    lifespan=lifespan,
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(chat_router, prefix=settings.api_v1_prefix)
app.include_router(social_media_router, prefix=settings.api_v1_prefix)
app.include_router(seo_router, prefix=settings.api_v1_prefix)
app.include_router(copywriter_router, prefix=settings.api_v1_prefix)
app.include_router(assistant_router, prefix=settings.api_v1_prefix)


# WebSocket endpoint
@app.websocket("/ws/stream/{client_id}")
async def ws_stream(websocket: WebSocket, client_id: str):
    """WebSocket endpoint for real-time agent updates."""
    await websocket_endpoint(websocket, client_id)


# Health check endpoint
@app.get("/health")
async def health_check() -> dict:
    """Health check endpoint."""
    return {
        "status": "healthy",
        "version": settings.app_version,
        "environment": settings.environment,
    }


# Root endpoint
@app.get("/")
async def root() -> dict:
    """Root endpoint with API information."""
    return {
        "name": settings.app_name,
        "version": settings.app_version,
        "docs": f"{settings.api_v1_prefix}/docs",
        "endpoints": {
            "chat": f"{settings.api_v1_prefix}/chat",
            "social_media": f"{settings.api_v1_prefix}/elves/social-media",
            "seo": f"{settings.api_v1_prefix}/elves/seo",
            "copywriter": f"{settings.api_v1_prefix}/elves/copywriter",
            "assistant": f"{settings.api_v1_prefix}/elves/assistant",
            "websocket": "/ws/stream/{{client_id}}",
        },
    }

