import asyncio
import logging
from datetime import datetime

from app.celery_app.config import celery_app
from app.infrastructure.connectors.postgres.postgres_service import PostgresService

logger = logging.getLogger(__name__)


@celery_app.task(name="process_dlq_message")
def process_dlq_message(task_id: str, error_msg: str, retry_count: int):
    """
    Process messages that ended up in the Dead Letter Queue.

    Logs to the dlq_log table for audit purposes.  Never re-raises so that
    DLQ processing itself does not create further DLQ entries.

    Args:
        task_id:     Celery task ID of the failed task.
        error_msg:   Error message from the failed task.
        retry_count: Number of retries attempted before giving up.
    """
    logger.error(
        "DLQ: task=%s failed after %d retries: %s", task_id, retry_count, error_msg
    )

    async def _log():
        postgres_service = PostgresService()
        try:
            await postgres_service.execute_raw_query(
                """
                INSERT INTO public.dlq_log
                    (task_id, error_message, retry_count, created_at)
                VALUES
                    (:task_id, :error_message, :retry_count, :created_at)
                """,
                {
                    "task_id": task_id,
                    "error_message": error_msg,
                    "retry_count": retry_count,
                    "created_at": datetime.now(),
                },
            )
            logger.info("DLQ: logged task=%s to dlq_log", task_id)
        except Exception as e:
            logger.error("DLQ: failed to write to dlq_log: %s", e)
            # Do not re-raise — DLQ handler must not fail.

    asyncio.run(_log())
    return {"task_id": task_id, "logged": True}
