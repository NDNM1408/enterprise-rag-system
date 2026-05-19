"""
Celery client for data-api to send tasks to data-processing-job workers.
"""
from typing import Optional
from celery import Celery
from kombu import Queue
from app.configurations.configurations import settings

# Create Celery app instance (client mode - no worker)
celery_client = Celery(
    'data_api_client',
    broker=settings.RABBITMQ_URL,
    backend='rpc://'
)

# Configure client
celery_client.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    task_routes={
        'preprocess_document': {'queue': 'preprocess_queue', 'routing_key': 'preprocess.document'},
        'upsert_chunk': {'queue': 'upsert_queue', 'routing_key': 'upsert.chunk'},
        'graph_preprocess_document': {'queue': 'graph_ingest_queue', 'routing_key': 'graph.preprocess'},
    },
    # no_declare=True prevents PRECONDITION_FAILED when the worker has already
    # created these queues with x-dead-letter-exchange and other special args.
    task_queues=[
        Queue('preprocess_queue', no_declare=True),
        Queue('upsert_queue', no_declare=True),
        Queue('graph_ingest_queue', no_declare=True),
    ],
    task_default_queue='preprocess_queue',
    task_create_missing_queues=False,
)


def send_graph_preprocess_task(
    document_id: str,
    knowledge_base_id: str,
    name: str,
    bucket: str,
    correlation_id: Optional[str] = None,
):
    return celery_client.send_task(
        'graph_preprocess_document',
        kwargs={
            'document_id': document_id,
            'knowledge_base_id': knowledge_base_id,
            'name': name,
            'bucket': bucket,
            'correlation_id': correlation_id,
        },
        queue='graph_ingest_queue',
        routing_key='graph.preprocess',
    )


def send_preprocess_task(document_id: str, knowledge_base_id: str, name: str,
                        embedding_model_id: str, bucket: str, correlation_id: str = None,
                        parsed_markdown_s3_url: Optional[str] = None):
    """
    Send preprocess_document task to worker queue.

    Args:
        document_id: UUID of the document
        knowledge_base_id: UUID of the knowledge base
        name: Document filename
        embedding_model_id: ID of embedding model
        bucket: S3 bucket name
        correlation_id: Optional correlation ID for tracing
        parsed_markdown_s3_url: Optional pre-parsed markdown S3 URL (full s3://bucket/key).
            When provided, the worker skips its local parser and chunks the
            markdown content directly. Used after the document-parsing service
            finishes producing markdown for non-native formats (PDF, DOCX, ...).

    Returns:
        AsyncResult object
    """
    return celery_client.send_task(
        'preprocess_document',
        kwargs={
            'document_id': document_id,
            'knowledge_base_id': knowledge_base_id,
            'name': name,
            'embedding_model_id': embedding_model_id,
            'bucket': bucket,
            'correlation_id': correlation_id,
            'parsed_markdown_s3_url': parsed_markdown_s3_url,
        },
        queue='preprocess_queue',
        routing_key='preprocess.document'
    )
