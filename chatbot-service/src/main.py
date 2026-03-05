
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.configurations.settings import settings
from app.configurations.logging_config import setup_logging
from app.middleware import RequestIDMiddleware, register_exception_handlers
from app.api import chatbot_router
from app.infrastructure.connectors.postgres.database import db_session
from app.infrastructure.connectors.postgres.schema import Base

# Setup centralized logging
setup_logging()
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown events."""
    # Startup: Create tables if they don't exist
    logger.info("Creating database tables if they don't exist...")
    async with db_session.engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("Database tables created successfully")
    yield
    # Shutdown: cleanup if needed
    logger.info("Shutting down...")


# Initialize FastAPI app
app = FastAPI(
    title="Chatbot Service API",
    description="Chatbot service for RAG-based conversations",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# Add request ID middleware (must be added before exception handlers)
app.add_middleware(RequestIDMiddleware)

# Register global exception handlers for consistent error responses
register_exception_handlers(app)

# Register API routers
app.include_router(chatbot_router)


@app.get("/health", tags=["health"])
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}


logger.info(f"Chatbot Service initialized on port {settings.PORT}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=True  # Remove reload=True in production
    )
