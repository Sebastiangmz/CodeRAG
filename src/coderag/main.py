"""CodeRAG main application entry point."""

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from coderag.config import get_settings
from coderag.logging import setup_logging, get_logger

# Initialize settings and logging
settings = get_settings()
setup_logging(level=settings.server.log_level.upper())
logger = get_logger(__name__)


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description="RAG-based Q&A system for code repositories with verifiable citations",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Health check endpoint
    @app.get("/health")
    async def health_check() -> dict:
        """Health check endpoint."""
        return {
            "status": "healthy",
            "app": settings.app_name,
            "version": settings.app_version,
        }

    # Register API routes
    from coderag.api.routes import router as api_router

    app.include_router(api_router, prefix="/api/v1")

    # Mount Gradio UI
    try:
        from coderag.ui.app import create_gradio_app

        gradio_app = create_gradio_app()
        app = gradio_app.mount_gradio_app(app, gradio_app, path="/")
        logger.info("Gradio UI mounted at /")
    except ImportError as e:
        logger.warning("Gradio UI not available", error=str(e))

    @app.on_event("startup")
    async def startup_event() -> None:
        """Application startup handler."""
        logger.info(
            "Starting CodeRAG",
            app_name=settings.app_name,
            version=settings.app_version,
            debug=settings.debug,
        )

    @app.on_event("shutdown")
    async def shutdown_event() -> None:
        """Application shutdown handler."""
        logger.info("Shutting down CodeRAG")

    return app


def main() -> None:
    """Run the application."""
    app = create_app()

    logger.info(
        "Starting server",
        host=settings.server.host,
        port=settings.server.port,
    )

    uvicorn.run(
        app,
        host=settings.server.host,
        port=settings.server.port,
        reload=settings.server.reload,
        workers=settings.server.workers,
        log_level=settings.server.log_level,
    )


if __name__ == "__main__":
    main()
