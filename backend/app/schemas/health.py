"""Health check response schema."""

from typing import Optional
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Structured response schema for service health status."""

    status: str = Field(default="ok", description="Operational status of the service")
    app_name: str = Field(description="Name of the application")
    version: str = Field(description="Semantic version of the application")
    environment: str = Field(description="Runtime environment (e.g., development, production)")
    timestamp: str = Field(description="ISO-8601 UTC timestamp of the health check")
    message: Optional[str] = Field(default=None, description="Optional diagnostic message")

    model_config = {
        "json_schema_extra": {
            "example": {
                "status": "ok",
                "app_name": "Argus",
                "version": "0.1.0",
                "environment": "development",
                "timestamp": "2026-10-16T14:34:00Z",
                "message": "Argus API is operational",
            }
        }
    }
