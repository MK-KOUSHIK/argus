"""Centralized API Router for Argus."""

from fastapi import APIRouter
from app.api.health import router as health_router
from app.api.optimizer import router as optimizer_router
from app.api.main_llm import router as main_llm_router

api_router = APIRouter()

# Include health router
api_router.include_router(health_router)

# Include optimizer router
api_router.include_router(optimizer_router)

# Include main LLM and benchmark router
api_router.include_router(main_llm_router)
