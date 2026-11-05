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
from app.services.routing.router import ModelNotInstalledError, ModelRouter, ModelRouterError

__all__ = [
    "ComplexityAssessment",
    "ComplexityTelemetry",
    "DiscoveredModel",
    "LocalModelRegistry",
    "ModelCatalogItem",
    "ModelCatalogResponse",
    "ModelNotInstalledError",
    "ModelRouter",
    "ModelRouterError",
    "ModelTier",
    "ResourceAssessment",
    "ResourceTelemetry",
    "RouteRequest",
    "RoutingDecision",
    "RoutingDecisionTelemetry",
    "RoutingStatus",
]

