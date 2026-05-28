"""Celery application — shared between API (publisher) and worker (consumer).

Routing:
    queue          parse_queue
    routing_key    parse.document
    exchange       parse_exchange (direct)
    DLQ            dlq.parse (acks_late + retry guard)

Worker startup:
    worker_process_init connects to ``_warmup_parser`` so VNDocParser + ONNX
    sessions + VietOCR weights are loaded before the first task arrives.
    This eliminates the ~3-5 minute model warmup that would otherwise
    hit whatever job is unlucky enough to be first in line.
"""
from __future__ import annotations

import logging

from celery import Celery
from celery.signals import worker_process_init
from kombu import Exchange, Queue

from settings import settings

log = logging.getLogger(__name__)


def _build_celery() -> Celery:
    app = Celery(
        "document_parsing",
        broker=settings.rabbitmq_url,
        backend=None,  # we don't use celery results — DB row is the source of truth
        include=["celery_app.tasks.parse_tasks"],
    )

    parse_exchange = Exchange("parse_exchange", type="direct", durable=True)
    dlq_exchange = Exchange("dlq_exchange", type="direct", durable=True)

    app.conf.update(
        task_acks_late=True,
        task_acks_on_failure_or_timeout=True,
        task_reject_on_worker_lost=True,
        worker_prefetch_multiplier=1,        # don't grab a 2nd job while parsing
        # When a fork worker dies (OOM, SIGKILL, etc.) Celery spawns a
        # replacement via the pool-replenish path which enforces
        # ``worker_proc_alive_timeout`` (default 4.0 s). Our ``worker_process_init``
        # loads ~4 GB of ONNX + VietOCR weights and takes 2–5 minutes, so the
        # default kills every replacement → infinite fork-respawn loop.
        # Bump to 10 min to cover cold-start model load.
        worker_proc_alive_timeout=600.0,
        broker_connection_retry_on_startup=True,
        # Keep the broker connection alive over multi-hour parses; avoids
        # spurious requeue when the consumer is busy on a long task.
        broker_heartbeat=60,
        broker_pool_limit=None,
        # No soft/hard time limit — worker decides when a parse is done.
        # If you want a kill-switch, set ``task_time_limit=N`` (seconds).
        timezone="UTC",
        enable_utc=True,
        task_default_queue=settings.celery_queue,
        task_default_exchange="parse_exchange",
        task_default_routing_key=settings.celery_routing_key,
        task_queues=(
            Queue(
                settings.celery_queue,
                exchange=parse_exchange,
                routing_key=settings.celery_routing_key,
                queue_arguments={
                    "x-dead-letter-exchange": "dlq_exchange",
                    "x-dead-letter-routing-key": "dlq.parse",
                },
            ),
            Queue(
                "dlq",
                exchange=dlq_exchange,
                routing_key="dlq.parse",
            ),
        ),
        task_routes={
            "celery_app.tasks.parse_tasks.parse_document_task": {
                "queue": settings.celery_queue,
                "routing_key": settings.celery_routing_key,
            },
        },
    )
    return app


celery_app = _build_celery()


@worker_process_init.connect
def _warmup_parser(**_) -> None:
    """Eagerly instantiate every registered parser at worker process start.

    For prefork pool (default), this runs once per forked process — with
    concurrency=1 that's a single warmup. The heaviest parser by far is
    LayoutPdfParser → VNDocParser (loads layout/ocr_det/orient_cls ONNX
    sessions + VietOCR transformer + VGG19 weights). Skipping a failure
    here means the worker keeps serving non-PDF formats; PDF jobs will
    surface the same error individually.
    """
    log.info("worker_process_init: priming parser registry...")
    try:
        from core.registry import registry
        reg = registry()
        log.info("registry has %d extensions", len(reg))
    except Exception:
        log.exception("registry build failed during warmup")
        return

    from settings import settings as _s
    if _s.pdf_force_plain:
        log.info("PDF_FORCE_PLAIN=true — warming layout-aware no-OCR parser.")
        try:
            from parsers.pdf_no_ocr import _build_no_ocr_parser
            _build_no_ocr_parser()
            log.info("PdfNoOcrParser ready (layout/table_cls/table_unet/table_slanet ONNX preloaded).")
        except Exception:
            log.warning("PdfNoOcrParser warmup failed", exc_info=True)
    else:
        try:
            from parsers.pdf_layout import _get_parser
            parser_instance = _get_parser()
            log.info("VNDocParser ready (models preloaded).")
            try:
                if _s.enable_cpu_batched_pipeline:
                    from parsers.pipeline.warmup import warm_batched_inference
                    warm_batched_inference(parser_instance)
            except Exception:
                log.warning("Batched-pipeline warmup skipped", exc_info=True)
        except Exception:
            log.warning("VNDocParser warmup skipped (parsing pipeline unavailable)", exc_info=True)

    # Prewarm the boto3 client + connection pool to MinIO. Without this
    # the first job pays a 20-30 s penalty for SSL/pool setup against
    # whichever endpoint it hits. head_bucket also primes our DNS cache.
    try:
        from infrastructure import s3
        c = s3.client()
        c.head_bucket(Bucket=settings.s3_bucket)
        log.info("S3 client primed.")
    except Exception:
        log.warning("S3 prewarm skipped", exc_info=True)
