"""
Integration tests for Celery tasks.

These tests verify task structure, retry behavior, and DLQ routing.
Run with: pytest tests/test_celery_tasks.py -v

For full integration testing with docker-compose:
1. Start services: docker-compose up -d
2. Run migrations: cd data-processing-job && alembic upgrade head
3. Start worker: celery -A celery_worker worker --loglevel=info
4. Run tests with --integration flag
"""
import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from uuid import uuid4


class TestPreprocessTask:
    """Tests for preprocess_document task."""

    @patch('app.celery_app.tasks.preprocess_tasks.DocumentPreprocessService')
    @patch('app.celery_app.tasks.preprocess_tasks.ChunkRepository')
    @patch('app.celery_app.tasks.preprocess_tasks.DocumentRepository')
    @patch('app.celery_app.tasks.preprocess_tasks.container')
    def test_preprocess_task_idempotency(
        self, mock_container, mock_doc_repo_cls, mock_chunk_repo_cls, mock_svc_cls
    ):
        """AlreadyProcessedError causes the task to return chunk_count=0."""
        from app.application.services.document_preprocess_service import AlreadyProcessedError

        mock_container.session_factory = MagicMock()
        mock_svc = MagicMock()
        mock_svc.preprocess = AsyncMock(side_effect=AlreadyProcessedError("already done"))
        mock_svc_cls.return_value = mock_svc

        from app.celery_app.tasks.preprocess_tasks import preprocess_document

        result = preprocess_document(
            document_id=str(uuid4()),
            knowledge_base_id=str(uuid4()),
            name="test.html",
            embedding_model_id="embed-1",
            bucket="test-bucket",
            correlation_id="test-123",
        )
        assert result['chunk_count'] == 0

    def test_preprocess_task_retry_configuration(self):
        """Test that preprocess task has correct retry settings."""
        from app.celery_app.tasks.preprocess_tasks import preprocess_document
        assert preprocess_document.max_retries == 3
        assert preprocess_document.acks_late is True

    @patch('app.celery_app.tasks.preprocess_tasks.DocumentPreprocessService')
    @patch('app.celery_app.tasks.preprocess_tasks.ChunkRepository')
    @patch('app.celery_app.tasks.preprocess_tasks.DocumentRepository')
    @patch('app.celery_app.tasks.preprocess_tasks.container')
    def test_preprocess_task_error_handling(
        self, mock_container, mock_doc_repo_cls, mock_chunk_repo_cls, mock_svc_cls
    ):
        """Unexpected errors are re-raised so Celery can retry."""
        mock_container.session_factory = MagicMock()
        mock_svc = MagicMock()
        mock_svc.preprocess = AsyncMock(side_effect=RuntimeError("S3 Error"))
        mock_svc_cls.return_value = mock_svc

        # Doc repo for the error handler inside the task
        mock_doc_repo = MagicMock()
        mock_doc_repo.set_status = AsyncMock()
        mock_doc_repo_cls.return_value = mock_doc_repo

        from app.celery_app.tasks.preprocess_tasks import preprocess_document

        with pytest.raises(Exception):
            preprocess_document(
                document_id=str(uuid4()),
                knowledge_base_id=str(uuid4()),
                name="test.html",
                embedding_model_id="embed-1",
                bucket="test-bucket",
            )


class TestUpsertTask:
    """Tests for upsert_chunk task."""

    @patch('app.celery_app.tasks.upsert_tasks.ChunkRepository')
    @patch('app.celery_app.tasks.upsert_tasks.container')
    def test_upsert_task_idempotency(self, mock_container, mock_chunk_repo_cls):
        """Chunk already Succeed → returns skipped without touching embedding API."""
        mock_container.session_factory = MagicMock()
        mock_chunk_repo = MagicMock()
        mock_chunk_repo.get_status = AsyncMock(return_value='Succeed')
        mock_chunk_repo_cls.return_value = mock_chunk_repo

        from app.celery_app.tasks.upsert_tasks import upsert_chunk

        result = upsert_chunk(
            chunk_id=str(uuid4()),
            s3_path="test/chunk.txt",
            knowledge_base_id=str(uuid4()),
            document_id=str(uuid4()),
            metadata={"type": "text"},
            is_last_chunk=False,
        )
        assert result['status'] == 'skipped'
        mock_container.embedding_service.get_embedding.assert_not_called()

    def test_upsert_task_retry_configuration(self):
        """Test that upsert task has correct retry settings."""
        from app.celery_app.tasks.upsert_tasks import upsert_chunk
        assert upsert_chunk.max_retries == 3
        assert upsert_chunk.acks_late is True

    @patch('app.celery_app.tasks.upsert_tasks.ChunkRepository')
    @patch('app.celery_app.tasks.upsert_tasks.container')
    def test_upsert_is_last_chunk_param_accepted(self, mock_container, mock_chunk_repo_cls):
        """is_last_chunk=True is accepted for backward compat (chord handles finalize)."""
        mock_container.session_factory = MagicMock()
        mock_chunk_repo = MagicMock()
        mock_chunk_repo.get_status = AsyncMock(return_value='Succeed')
        mock_chunk_repo_cls.return_value = mock_chunk_repo

        from app.celery_app.tasks.upsert_tasks import upsert_chunk

        # Must not raise
        result = upsert_chunk(
            chunk_id=str(uuid4()),
            s3_path="test/chunk.txt",
            knowledge_base_id=str(uuid4()),
            document_id=str(uuid4()),
            metadata={"type": "text"},
            is_last_chunk=True,
        )
        assert result['status'] == 'skipped'


class TestDLQTask:
    """Tests for DLQ handling task."""

    @patch('app.celery_app.tasks.dlq_tasks.PostgresService')
    def test_dlq_task_logs_failed_message(self, mock_pg):
        """Test that DLQ task logs failed messages to database."""
        from app.celery_app.tasks.dlq_tasks import process_dlq_message

        mock_pg_instance = MagicMock()
        mock_pg_instance.execute_raw_query = AsyncMock()
        mock_pg.return_value = mock_pg_instance

        result = process_dlq_message(
            task_id="task-123",
            error_msg="Task failed: Connection error",
            retry_count=3
        )

        # Verify logged
        assert result['logged'] is True
        assert result['task_id'] == "task-123"


class TestCeleryConfiguration:
    """Tests for Celery configuration."""

    def test_celery_app_configuration(self):
        """Test that Celery app has correct configuration."""
        from app.celery_app.config import celery_app

        # Verify retry settings
        assert celery_app.conf.task_acks_late is True
        assert celery_app.conf.task_reject_on_worker_lost is True
        assert celery_app.conf.worker_prefetch_multiplier == 1

        # Verify time limits
        assert celery_app.conf.task_soft_time_limit == 3600
        assert celery_app.conf.task_time_limit == 7200

        # Verify retry policy
        assert celery_app.conf.task_retry_backoff is True
        assert celery_app.conf.task_retry_backoff_max == 600

        # Verify serialization
        assert celery_app.conf.task_serializer == 'json'
        assert 'json' in celery_app.conf.accept_content


    def test_celery_queues_configuration(self):
        """Test that queues are configured with DLX."""
        from app.celery_app.config import celery_app

        queues = {q.name: q for q in celery_app.conf.task_queues}

        # Verify preprocess queue
        assert 'preprocess_queue' in queues
        preprocess_q = queues['preprocess_queue']
        assert preprocess_q.queue_arguments['x-dead-letter-exchange'] == 'dlx'
        assert preprocess_q.queue_arguments['x-dead-letter-routing-key'] == 'dlq.preprocess'

        # Verify upsert queue
        assert 'upsert_queue' in queues
        upsert_q = queues['upsert_queue']
        assert upsert_q.queue_arguments['x-dead-letter-exchange'] == 'dlx'

        # Verify DLQ exists
        assert 'dlq' in queues


@pytest.mark.skipif(
    True,  # Change to False when running with docker-compose
    reason="Requires docker-compose infrastructure"
)
class TestCeleryIntegration:
    """
    Full integration tests requiring:
    - PostgreSQL with pgvector
    - RabbitMQ
    - MinIO/S3
    - Celery worker running

    Run with: docker-compose up -d && pytest tests/test_celery_tasks.py::TestCeleryIntegration -v
    """

    def test_end_to_end_document_processing(self):
        """Test complete document processing workflow."""
        # This would test:
        # 1. Upload document to S3
        # 2. Trigger preprocess task
        # 3. Wait for chunks to be created
        # 4. Verify embeddings in database
        # 5. Verify document status = 'Succeed'
        pass

    def test_retry_on_transient_failure(self):
        """Test that tasks retry on transient failures."""
        # This would test:
        # 1. Mock a transient failure (e.g., timeout)
        # 2. Verify task retries with backoff
        # 3. Verify success after retry
        pass

    def test_dlq_on_permanent_failure(self):
        """Test that permanently failed tasks go to DLQ."""
        # This would test:
        # 1. Mock a permanent failure
        # 2. Verify task retries max_retries times
        # 3. Verify message ends up in DLQ
        # 4. Verify DLQ log entry created
        pass
