"""Tests for Health API endpoints."""

import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_health_check_endpoint(async_client: AsyncClient):
    """Verify that GET /api/health returns 200 OK and valid schema."""
    response = await async_client.get("/api/health")
    assert response.status_code == 200

    data = response.json()
    assert data["status"] == "ok"
    assert data["app_name"] == "Argus"
    assert "version" in data
    assert "environment" in data
    assert "timestamp" in data
    assert "operational" in data["message"].lower()


@pytest.mark.asyncio
async def test_not_found_error_handler(async_client: AsyncClient):
    """Verify that non-existent routes return structured JSON error."""
    response = await async_client.get("/api/non-existent-route")
    assert response.status_code == 404

    data = response.json()
    assert "error" in data
    assert data["error"]["code"] == "HTTP_404"
    assert data["error"]["path"] == "/api/non-existent-route"
