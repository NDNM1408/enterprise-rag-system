from celery import Celery
from kombu import Exchange, Queue
from app.configurations.configurations import settings

celery_app = Celery('data_hub_worker', broker=settings.RABBITMQ_URL)

# Dead Letter Exchange for failed tasks
DLX_EXCHANGE = Exchange('dlx', type='topic', durable=True)

# Task queues with DLX routing
task_queues = (
    Queue('preprocess_queue',
          exchange=Exchange('data_exchange', type='topic', durable=True),
          routing_key='preprocess.*',
          queue_arguments={
              'x-dead-letter-exchange': 'dlx',
              'x-dead-letter-routing-key': 'dlq.preprocess',
              'x-message-ttl': 3600000,  # 1 hour
          }),
    Queue('upsert_queue',
          exchange=Exchange('data_exchange', type='topic', durable=True),
          routing_key='upsert.*',
          queue_arguments={
              'x-dead-letter-exchange': 'dlx',
              'x-dead-letter-routing-key': 'dlq.upsert',
          }),
    Queue('llm_wiki_queue',
          exchange=Exchange('data_exchange', type='topic', durable=True),
          routing_key='llm_wiki.*',
          queue_arguments={
              'x-dead-letter-exchange': 'dlx',
              'x-dead-letter-routing-key': 'dlq.llm_wiki',
              # Legal corpora ingestion is embedding-heavy; allow 1h soft TTL.
              'x-message-ttl': 3600000,
          }),
    Queue('dlq', exchange=DLX_EXCHANGE, routing_key='dlq.#', durable=True),
)

celery_app.conf.update(
    task_queues=task_queues,

    # Task routing
    task_routes={
        'llm_wiki_preprocess_document': {'queue': 'llm_wiki_queue', 'routing_key': 'llm_wiki.preprocess'},
    },

    # Reliability settings for long-running tasks
    task_acks_late=True,                    # ACK after completion
    task_reject_on_worker_lost=True,
    worker_prefetch_multiplier=1,            # Prefetch 1 task (prevent blocking)

    # Time limits
    task_soft_time_limit=3600,               # 1 hour soft (SIGTERM)
    task_time_limit=7200,                    # 2 hours hard (SIGKILL)

    # Retry policy (defaults, can override per-task)
    task_autoretry_for=(Exception,),
    task_retry_kwargs={'max_retries': 3},
    task_retry_backoff=True,                 # Exponential backoff
    task_retry_backoff_max=600,              # Max 10 min backoff
    task_retry_jitter=True,

    # Serialization (avoid pickle for security)
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',

    # Result backend: PostgreSQL via SQLAlchemy (required for chord support).
    # Strip +asyncpg so SQLAlchemy uses its default sync driver (psycopg2).
    result_backend='db+' + settings.DATABASE_URL.replace('+asyncpg', ''),
    result_expires=3600,  # clean up result rows after 1 hour
)

# Auto-discover tasks from tasks module
celery_app.autodiscover_tasks(['app.celery_app.tasks'])
