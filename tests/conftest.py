"""
Pytest configuration and fixtures.
"""

import asyncio
from typing import AsyncGenerator, Generator

import pytest
import pytest_asyncio
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker

from app.api.main import app
from app.core.database import Base


# Test database URL (SQLite for testing)
TEST_DATABASE_URL = "sqlite+aiosqlite:///./test.db"


@pytest.fixture(scope="session")
def event_loop() -> Generator:
    """Create event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest_asyncio.fixture
async def test_db() -> AsyncGenerator[AsyncSession, None]:
    """Create test database session."""
    engine = create_async_engine(TEST_DATABASE_URL, echo=False)
    
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    
    async_session = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    
    async with async_session() as session:
        yield session
    
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    
    await engine.dispose()


@pytest_asyncio.fixture
async def client() -> AsyncGenerator[AsyncClient, None]:
    """Create test HTTP client."""
    async with AsyncClient(app=app, base_url="http://test") as ac:
        yield ac


@pytest.fixture
def mock_user_id() -> str:
    """Return mock user ID for testing."""
    return "test-user-001"


@pytest.fixture
def sample_post_request() -> dict:
    """Return sample create post request."""
    return {
        "platform": "linkedin",
        "topic": "The future of AI in business",
        "content_type": "thought_leadership",
        "goals": ["engagement", "awareness"],
    }


@pytest.fixture
def sample_seo_request() -> dict:
    """Return sample SEO audit request."""
    return {
        "website_url": "https://example.com",
        "include_competitors": False,
    }


@pytest.fixture
def sample_blog_request() -> dict:
    """Return sample blog writing request."""
    return {
        "topic": "10 Tips for Remote Work Productivity",
        "target_keywords": ["remote work", "productivity"],
        "word_count": 1000,
        "tone": "professional",
    }

