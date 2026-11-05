"""Model Routing Package for Argus (Phase 5)."""

from app.services.routing.models import (
    ComplexityAssessment,
    ComplexityTelemetry,
    DiscoveredModel,
    ModelCatalogItem,
    ModelCatalogResponse,
    ModelTier,
    ResourceAssessment,
    ResourceTelemetry,
    RouteRequest,
    RoutingDecision,
    RoutingDecisionTelemetry,
    RoutingStatus,
)
from app.services.routing.registry import LocalModelRegistry
from app.services.routing.router import (
    ModelNotInstalledError,
    ModelResourceConstraintError,
    ModelRouter,
    ModelRouterError,
    NoSafeFallbackError,
)

__all__ = [
    "ComplexityAssessment",
    "ComplexityTelemetry",
    "DiscoveredModel",
    "LocalModelRegistry",
    "ModelCatalogItem",
    "ModelCatalogResponse",
    "ModelNotInstalledError",
    "ModelResourceConstraintError",
    "ModelRouter",
    "ModelRouterError",
    "ModelTier",
    "NoSafeFallbackError",
    "ResourceAssessment",
    "ResourceTelemetry",
    "RouteRequest",
    "RoutingDecision",
    "RoutingDecisionTelemetry",
    "RoutingStatus",
]

