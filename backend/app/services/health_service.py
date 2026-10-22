"""Health check service implementation."""

from datetime import datetime, timezone
from app.core.config import settings
from app.schemas.health import HealthResponse


class HealthService:
    """Encapsulates health inspection and diagnostic status reporting."""

    @staticmethod
    def get_health() -> HealthResponse:
        """Inspect and return current operational health status."""
        return HealthResponse(
            status="ok",
            app_name=settings.APP_NAME,
            version=settings.VERSION,
            environment=settings.ENVIRONMENT,
            timestamp=datetime.now(timezone.utc).isoformat(),
            message="Argus API is operational",
        )
