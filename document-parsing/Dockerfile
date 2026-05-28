# syntax=docker/dockerfile:1.6
# Worker/API container: Celery consumer + FastAPI with the vn_parser
# stack. Fully ONNX (onnxruntime) — no torch / vietocr / transformers. Runs
# CPU-only here; for GPU build with Dockerfile.gpu (see docker-compose.gpu.yaml).
# Parser ONNX weights are mounted from the host at /opt/mineru.
FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        poppler-utils \
        libgl1 \
        libglib2.0-0 \
        libsm6 \
        libxrender1 \
        libxext6 \
        curl \
    && rm -rf /var/lib/apt/lists/*

ENV PIP_DEFAULT_TIMEOUT=600 \
    PIP_RETRIES=15 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

COPY requirements.txt requirements-parser.txt ./
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip wheel "setuptools<70"

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements-parser.txt

COPY src/ ./src/
COPY alembic.ini ./alembic.ini
COPY migrations/ ./migrations/
COPY docker-entrypoint-worker.sh docker-entrypoint-api.sh ./
RUN chmod +x ./docker-entrypoint-worker.sh ./docker-entrypoint-api.sh

ENV PARSER_MODELS_DIR=/opt/mineru/models_onnx \
    PORT=8002 \
    PYTHONPATH=/app/src:/app/src/vendored

EXPOSE 8002

# Default CMD = worker. The API service in docker-compose overrides
# this with ./docker-entrypoint-api.sh.
CMD ["./docker-entrypoint-worker.sh"]
