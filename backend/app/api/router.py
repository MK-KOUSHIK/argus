"""Centralized API Router for Argus."""

from fastapi import APIRouter
from app.api.health import router as health_router
from app.api.optimizer import router as optimizer_router
from app.api.routing import router as routing_router
from app.api.comparison import router as comparison_router

api_router = APIRouter()

# Include health router
api_router.include_router(health_router)

# Include optimizer router
api_router.include_router(optimizer_router)

# Include model routing router (Phase 5)
api_router.include_router(routing_router)

# Include output comparison router
api_router.include_router(comparison_router)
