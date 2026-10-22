"""Argus FastAPI Main Application Entrypoint."""

from contextlib import asynccontextmanager
from typing import AsyncGenerator
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from fastapi.encoders import jsonable_encoder
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.core.config import settings
from app.api.router import api_router


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Lifespan events for application startup and shutdown."""
    # Startup actions (if any)
    yield
    # Shutdown cleanup (if any)


def create_app() -> FastAPI:
    """FastAPI Application factory with middleware, routers, and exception handlers."""
    app = FastAPI(
        title=settings.APP_NAME,
        version=settings.VERSION,
        description="Argus: Intelligent prompt optimization middleware",
        docs_url=f"{settings.API_PREFIX}/docs" if settings.DEBUG else None,
        redoc_url=f"{settings.API_PREFIX}/redoc" if settings.DEBUG else None,
        openapi_url=f"{settings.API_PREFIX}/openapi.json" if settings.DEBUG else None,
        lifespan=lifespan,
    )

    # CORS Middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Centralized Error Handlers
    @app.exception_handler(StarletteHTTPException)
    async def http_exception_handler(request: Request, exc: StarletteHTTPException):
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": {
                    "code": f"HTTP_{exc.status_code}",
                    "message": exc.detail,
                    "path": str(request.url.path),
                }
            },
        )

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={
                "error": {
                    "code": "VALIDATION_ERROR",
                    "message": "Invalid request parameters",
                    "details": jsonable_encoder(exc.errors()),
                    "path": str(request.url.path),
                }
            },
        )

    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception):
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": {
                    "code": "INTERNAL_SERVER_ERROR",
                    "message": "An unexpected server error occurred",
                    "path": str(request.url.path),
                }
            },
        )

    # Include API Routers
    app.include_router(api_router, prefix=settings.API_PREFIX)

    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
    )
