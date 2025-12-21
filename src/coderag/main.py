"""CodeRAG main application entry point."""

from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from coderag.config import get_settings
from coderag.logging import setup_logging, get_logger

# Initialize settings and logging
settings = get_settings()
setup_logging(level=settings.server.log_level.upper())
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    logger.info(
        "Starting CodeRAG",
        app_name=settings.app_name,
        version=settings.app_version,
        debug=settings.debug,
    )
    yield
    logger.info("Shutting down CodeRAG")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description="RAG-based Q&A system for code repositories with verifiable citations",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
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

    # Mount MCP server
    try:
        from coderag.mcp.server import create_mcp_server

        mcp_server = create_mcp_server()
        mcp_app = mcp_server.streamable_http_app()
        app.mount("/mcp", mcp_app)
        logger.info("MCP server mounted at /mcp")
    except ImportError as e:
        logger.warning("MCP server not available", error=str(e))
    except Exception as e:
        logger.error("Failed to mount MCP server", error=str(e))

    # Mount Gradio UI
    try:
        from coderag.ui.app import create_gradio_app
        import gradio as gr

        gradio_app = create_gradio_app()
        app = gr.mount_gradio_app(app, gradio_app, path="/")
        logger.info("Gradio UI mounted at /")
    except ImportError as e:
        logger.warning("Gradio UI not available", error=str(e))
    except Exception as e:
        logger.error("Failed to mount Gradio UI", error=str(e))

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
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Application interrupted by user")
    except Exception as e:
        logger.error("Application crashed", error=str(e), exc_info=True)
        import traceback
        print("\n" + "="*80)
        print("FATAL ERROR:")
        print("="*80)
        traceback.print_exc()
        print("="*80)
        input("Press Enter to close...")  # Keep terminal open
