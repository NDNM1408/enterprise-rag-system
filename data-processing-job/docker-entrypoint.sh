#!/bin/bash
set -e

# Default worker command
CELERY_CMD="celery -A celery_worker worker --loglevel=info --concurrency=4 -Q preprocess_queue,upsert_queue,llm_wiki_queue,dlq"

# Check if running in dev mode
if [ "${MODE}" = "dev" ]; then
    echo "Running in DEVELOPMENT mode with auto-reload..."
    exec watchmedo auto-restart \
        --directory=/app/src \
        --pattern="*.py" \
        --recursive \
        -- \
        $CELERY_CMD
else
    echo "Running in PRODUCTION mode..."
    exec $CELERY_CMD
fi
