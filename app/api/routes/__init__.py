"""API Route modules"""

from app.api.routes.chat import router as chat_router
from app.api.routes.social_media import router as social_media_router
from app.api.routes.seo import router as seo_router
from app.api.routes.copywriter import router as copywriter_router
from app.api.routes.assistant import router as assistant_router
from app.api.routes.embeddings import router as embeddings_router

__all__ = [
    "chat_router",
    "social_media_router",
    "seo_router",
    "copywriter_router",
    "assistant_router",
    "embeddings_router",
]

