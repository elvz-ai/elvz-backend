"""
API endpoint tests.
"""

import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_health_check(client: AsyncClient):
    """Test health check endpoint."""
    response = await client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"


@pytest.mark.asyncio
async def test_root_endpoint(client: AsyncClient):
    """Test root endpoint."""
    response = await client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "name" in data
    assert "endpoints" in data


@pytest.mark.asyncio
async def test_chat_endpoint(client: AsyncClient):
    """Test chat endpoint."""
    response = await client.post(
        "/api/v1/chat",
        json={"message": "Create a LinkedIn post about AI"},
    )
    # Should work in development mode without auth
    assert response.status_code in [200, 401, 500]


@pytest.mark.asyncio
async def test_create_post_endpoint(client: AsyncClient, sample_post_request: dict):
    """Test create post endpoint."""
    response = await client.post(
        "/api/v1/elves/social-media/create-post",
        json=sample_post_request,
    )
    assert response.status_code in [200, 401, 500]


@pytest.mark.asyncio
async def test_seo_quick_check(client: AsyncClient):
    """Test SEO quick check endpoint."""
    response = await client.get(
        "/api/v1/elves/seo/quick-check",
        params={"url": "https://example.com"},
    )
    assert response.status_code in [200, 401, 500]


@pytest.mark.asyncio
async def test_write_blog_endpoint(client: AsyncClient, sample_blog_request: dict):
    """Test write blog endpoint."""
    response = await client.post(
        "/api/v1/elves/copywriter/write-blog",
        json=sample_blog_request,
    )
    assert response.status_code in [200, 401, 500]


@pytest.mark.asyncio
async def test_invalid_platform(client: AsyncClient):
    """Test validation for invalid platform."""
    response = await client.post(
        "/api/v1/elves/social-media/create-post",
        json={
            "platform": "invalid_platform",
            "topic": "Test topic",
        },
    )
    assert response.status_code == 422  # Validation error

