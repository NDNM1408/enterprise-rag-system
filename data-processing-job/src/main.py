import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from app.configurations.configurations import settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager to handle startup and shutdown events.
    Note: Celery workers are now run separately via celery_worker.py
    This service only provides health check endpoint.
    """
    logger.info("Data-processing-job service started (Celery mode)")
    logger.info("Run Celery worker separately: celery -A celery_worker worker --loglevel=info")

    yield  # Yield control back to FastAPI

    logger.info("Data-processing-job service shutting down")

# Initialize FastAPI app with lifespan
app = FastAPI(lifespan=lifespan)

@app.get("/health")
async def health_check():
    """
    Health check endpoint to verify the application is running.
    """
    return {"status": "OK"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=settings.PORT)