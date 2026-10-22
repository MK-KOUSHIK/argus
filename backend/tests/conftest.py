"""Pytest fixtures for Argus backend testing."""

import pytest
from httpx import ASGITransport, AsyncClient
from app.main import app


@pytest.fixture
async def async_client():
    """Async HTTP client fixture configured for FastAPI testing."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        yield client
