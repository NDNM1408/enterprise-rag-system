#!/usr/bin/env bash
# Worker container entrypoint: launch Celery consumer.
# Migrations are owned by the API container — worker just connects.
set -euo pipefail

cd /app/src

CONCURRENCY="${CELERY_CONCURRENCY:-1}"
QUEUE="${CELERY_QUEUE:-parse_queue}"
LOGLEVEL="${LOG_LEVEL:-info}"

echo "==> starting celery worker (queue=${QUEUE}, concurrency=${CONCURRENCY})"
exec celery -A celery_app worker \
    --loglevel="${LOGLEVEL}" \
    --concurrency="${CONCURRENCY}" \
    --queues="${QUEUE}" \
    --hostname="parse-worker@%h" \
    --without-gossip --without-mingle --without-heartbeat
