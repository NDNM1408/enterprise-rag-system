"""
Data-API main application entry point.

Configures FastAPI application with:
- Unified response format (success + error)
- Request ID tracking
- Global exception handlers
- API routers
"""
import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.configurations.configurations import settings
from app.configurations.logging_config import setup_logging
from app.middleware import RequestIDMiddleware, register_exception_handlers
from app.application.controllers.knowledge_base_controller import router as knowledge_base_api_router
from app.application.controllers.document_controller import router as document_api_router
from app.application.controllers.health_check_controller import router as health_check_api_router
from app.application.controllers.query_controller import router as query_api_router
from app.application.controllers.chatbot_controller import router as chatbot_api_router

# Setup centralized logging
setup_logging()
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Data Hub API",
    description="Document processing and RAG system API",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add request ID middleware (must be added before exception handlers)
app.add_middleware(RequestIDMiddleware)

# Register global exception handlers for consistent error responses
register_exception_handlers(app)

# Register API routers
app.include_router(knowledge_base_api_router)
app.include_router(document_api_router)
app.include_router(query_api_router)
app.include_router(health_check_api_router)
app.include_router(chatbot_api_router)

logger.info("Data-API application initialized")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=settings.PORT,
        reload=settings.is_dev
    )
