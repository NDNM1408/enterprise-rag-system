#!/usr/bin/env python
"""
Celery worker entry point for data-processing-job.

Usage:
    celery -A celery_worker worker --loglevel=info --concurrency=4
    celery -A celery_worker flower --port=5555

Environment variables:
    See .env.sample for required configuration.
"""
import logging

from celery.signals import worker_process_init

from app.configurations.logging_config import setup_logging
from app.celery_app import celery_app

# Import tasks to register them with Celery
from app.celery_app.tasks import preprocess_tasks  # noqa: F401
from app.celery_app.tasks import upsert_tasks      # noqa: F401
from app.celery_app.tasks import dlq_tasks         # noqa: F401
from app.celery_app.tasks import llm_wiki_tasks    # noqa: F401

setup_logging()
logger = logging.getLogger(__name__)


@worker_process_init.connect
def init_worker_process(**kwargs):
    """
    Initialize per-process resources after Celery's prefork fork().

    Called once per worker process (not per task).  Creating the DB engine,
    S3 client, tokenizer, and embedding service here — after the fork — means
    child processes never inherit connections tied to the parent's event loop.
    """
    from app.container import container
    container.init()
    logger.info("Worker process ready")


logger.info("Celery worker module loaded")

if __name__ == "__main__":
    celery_app.start()
