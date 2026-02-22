"""
Application configuration using Pydantic Settings.
Loads from environment variables with sensible defaults.
"""

from functools import lru_cache
from typing import Literal

from pydantic import Field, RedisDsn
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )
    
    # Application
    app_name: str = "Elvz.ai Backend"
    app_version: str = "1.0.0"
    debug: bool = False
    environment: Literal["development", "staging", "production"] = "development"
    
    # API
    api_v1_prefix: str = "/api/v1"
    cors_origins: list[str] = ["http://localhost:3000", "http://localhost:8000"]
    
    # Database - Individual settings (recommended)
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_database: str = "elvz"
    postgres_user: str = "postgres"
    postgres_password: str = "postgres"
    
    # Database pool settings
    database_pool_size: int = 20
    database_max_overflow: int = 10
    
    @property
    def database_url(self) -> str:
        """Build database URL from individual components."""
        return f"postgresql+asyncpg://{self.postgres_user}:{self.postgres_password}@{self.postgres_host}:{self.postgres_port}/{self.postgres_database}"
    
    # Redis
    redis_url: RedisDsn = Field(default="redis://localhost:6379/0")
    redis_session_ttl: int = 3600  # 1 hour
    redis_cache_ttl: int = 21600  # 6 hours
    
    # Celery
    celery_broker_url: str = "redis://localhost:6379/1"
    celery_result_backend: str = "redis://localhost:6379/1"
    
    # LLM Providers
    openai_api_key: str = ""
    anthropic_api_key: str = ""
    google_api_key: str = ""  # Gemini API key
    
    # Default LLM Provider: "openai", "anthropic", or "google"
    default_llm_provider: str = "google"
    
    # LLM Model Configuration
    openai_model_primary: str = "gpt-4o"
    openai_model_fast: str = "gpt-4o-mini"
    anthropic_model_primary: str = "claude-3-5-sonnet-20241022"
    google_model_primary: str = "gemini-1.5-pro-latest"
    google_model_fast: str = "gemini-1.5-flash-latest"
    
    # LLM Settings
    llm_temperature: float = 0.7
    llm_max_tokens: int = 4096
    llm_timeout: int = 60
    llm_max_retries: int = 3
    
    # Output Settings
    include_reasoning: bool = False  # If True, agents include detailed rationale/reasoning
    
    # Vertex AI (Google Cloud)
    google_cloud_project: str = ""  # GCP project ID
    google_cloud_location: str = "us-central1"  # GCP region
    google_application_credentials: str = "credentials/google-service-account.json"  # Path to service account JSON
    vertex_embedding_model: str = "multimodalembedding@001"  # Multimodal embedding model
    vertex_vector_index_endpoint: str = ""  # Vector Search index endpoint (full resource name)
    vertex_deployed_index_id: str = ""  # Deployed index ID
    
    # Vector Database - Qdrant (Primary)
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str = ""
    qdrant_collection_name: str = "elvz_memory"
    qdrant_social_collection: str = "social_memory_test"
    qdrant_vector_size: int = 3072  # Gemini gemini-embedding-001 dimension
    qdrant_distance: str = "Cosine"  # Cosine | Euclid | Dot

    # Vector Database - Pinecone (Legacy)
    pinecone_api_key: str = ""
    pinecone_environment: str = "us-east-1"
    pinecone_index_name: str = "elvz-knowledge"

    # Vector Database - Weaviate (Alternative)
    weaviate_url: str = "http://localhost:8080"
    weaviate_api_key: str = ""
    
    # Authentication
    jwt_secret_key: str = "your-secret-key-change-in-production"
    jwt_algorithm: str = "HS256"
    jwt_access_token_expire_minutes: int = 30
    jwt_refresh_token_expire_days: int = 7
    
    # Rate Limiting
    rate_limit_requests_per_minute: int = 60
    rate_limit_tokens_per_minute: int = 100000
    
    # External APIs (placeholders)
    hashtag_api_key: str = ""
    serp_api_key: str = ""
    google_search_console_key: str = ""
    
    # xAI / Grok
    xai_api_key: str = ""
    xai_model: str = "grok-beta"
    xai_base_url: str = "https://api.x.ai/v1"

    # OpenRouter (Unified LLM Access)
    openrouter_api_key: str = ""
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    openrouter_site_url: str = "https://elvz.ai"
    openrouter_site_name: str = "Elvz.ai"

    # S3 Storage (images → elvz-images, videos → elvz-videos)
    s3_region: str = "us-west-2"
    s3_access_key_id: str = ""
    s3_secret_access_key: str = ""

    # Logging
    log_level: str = "INFO"
    log_format: Literal["json", "console"] = "console"

    # AWS Bedrock (Guardrails)
    aws_region: str = "us-east-1"
    aws_access_key_id: str = ""
    aws_secret_access_key: str = ""
    bedrock_guardrail_id: str = ""
    bedrock_guardrail_version: str = "DRAFT"

    # LangFuse (LLM Observability)
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host: str = "https://cloud.langfuse.com"

    # Sentry (Error Tracking)
    sentry_dsn: str = ""
    sentry_environment: str = "development"
    sentry_traces_sample_rate: float = 0.1

    # Conversation Settings
    conversation_max_messages: int = 100
    conversation_context_window: int = 10  # Recent messages to include in context
    hitl_request_timeout_seconds: int = 300  # 5 minutes

    # Memory Settings
    memory_working_ttl: int = 3600  # 1 hour for working memory
    memory_short_term_limit: int = 50  # Max messages in short-term memory
    memory_rag_top_k: int = 5  # Top K for RAG retrieval
    memory_token_budget: int = 6000  # Max tokens for context

    # Artifact History Settings
    artifact_history_max_size: int = 10                   # Max artifacts in history ring buffer
    modification_confidence_threshold: float = 0.7        # Below this, ask follow-up to disambiguate

    # RAG Relevance Score Thresholds (0.0 - 1.0 cosine similarity)
    # Results below these scores are filtered out as irrelevant noise
    rag_min_score_knowledge: float = 0.60    # General best practices - lower bar acceptable
    rag_min_score_user_history: float = 0.65  # User's own generated content
    rag_min_score_conversation: float = 0.68  # Past conversation turns - must be on-topic

    # Minimum content length to save to Qdrant (chars)
    # Prevents saving noise like "ok", "yes" but allows short questions
    rag_min_save_length_query: int = 20   # User queries (short questions are still meaningful)
    rag_min_save_length_response: int = 50  # Assistant responses (avoid saving one-liners)

    # Query Decomposition
    max_platforms_per_query: int = 3
    enable_parallel_generation: bool = True

    # Elvz Next.js App Integration
    # Key used to verify INCOMING webhook requests from Next.js → Python backend.
    # Must match ELVZ_PYTHON_ENGINE_API_KEY set on the Next.js side.
    elvz_api_key: str = ""
    # Key used to authenticate OUTGOING requests from Python backend → Next.js internal APIs.
    # Must match the internal API key configured on the Next.js side.
    elvz_next_api_key: str = ""
    # Base URL of the Next.js app (used to call internal APIs after receiving a webhook)
    elvz_nextjs_base_url: str = "http://localhost:3000"

    # Dev override — when skip_user_id is true, use dev_user_id for all requests
    skip_user_id: bool = False
    dev_user_id: str = ""

    # Content generation feature flags
    enable_visual_generation: bool = False  # Set to true to enable image generation
    enable_video_generation: bool = False   # Set to true to enable video generation


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


settings = get_settings()

