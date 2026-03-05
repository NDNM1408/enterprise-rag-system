# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Data-hub is a document processing and RAG (Retrieval Augmented Generation) system consisting of two microservices that work together to ingest, process, and index documents for vector search.

## Architecture

### Services

**data-api**: FastAPI REST API service
- Handles document and knowledge base management
- Exposes REST endpoints for uploading documents
- Publishes events to RabbitMQ for asynchronous processing
- Entry point: `data-api/src/main.py`

**data-processing-job**: Background worker service (Celery-based)
- Celery workers process document tasks asynchronously
- Processes documents through a parser → splitter → embedding pipeline
- Stores embeddings in PostgreSQL with pgvector using direct SQL
- Health check entry point: `data-processing-job/src/main.py`
- Celery worker entry point: `data-processing-job/src/celery_worker.py`

### Layered Architecture

Both services follow a clean architecture pattern:

```
src/app/
├── application/        # Application layer
│   ├── controllers/    # API endpoints (data-api only)
│   ├── core/           # Core processing logic (data-processing-job only)
│   ├── services/       # Business logic
│   └── dtos/           # Data transfer objects
├── celery_app/         # Celery configuration and tasks (data-processing-job only)
│   ├── config.py       # Celery app configuration
│   └── tasks/          # Celery task definitions
├── infrastructure/     # Infrastructure layer
│   ├── clients/        # External service clients (S3, DocumentAI)
│   ├── repositories/   # Database repositories (data-processing-job only)
│   └── connectors/     # Database/queue connections (Postgres, RabbitMQ)
├── configurations/     # Pydantic settings loaded from .env
├── constants/          # Application constants (queue names, etc.)
└── utils/              # Utility functions
```

### Message Flow (Celery-based)

1. Client uploads documents to data-api via POST `/api/v1/{kb_id}/documents`
2. data-api publishes `preprocess_document` task to Celery via RabbitMQ (`preprocess_queue`)
3. Celery worker picks up the task and processes:
   - Fetches document from S3
   - Parses HTML and splits into chunks
   - Uploads chunks to S3
   - Inserts chunk records into database
4. For each chunk, spawns `upsert_chunk` task (`upsert_queue`)
5. Each `upsert_chunk` task:
   - Generates embeddings via API
   - Stores in pgvector using `DocumentEmbeddingsRepository`
6. After all chunks complete, `finalize_document` task updates document status

### Core Processing Pipeline (data-processing-job)

Located in `data-processing-job/src/app/application/core/`:

- **Document** (`document.py`): Simple dataclass for document chunks (no LlamaIndex)
- **Parser** (`parser.py`): Extracts text from HTML documents, handles tables, prepends titles
- **Splitter** (`splitter.py`): Chunks documents using tokenizer with configurable size

Celery tasks in `data-processing-job/src/app/celery_app/tasks/`:

- **preprocess_tasks.py**: `preprocess_document` - fetch, parse, chunk, store
- **upsert_tasks.py**: `upsert_chunk` - generate embeddings, store in pgvector
- **dlq_tasks.py**: Dead letter queue handling for failed tasks

Repository in `data-processing-job/src/app/infrastructure/repositories/`:

- **document_embeddings_repository.py**: Direct SQL queries for vector operations

## Development Setup

### Prerequisites

- Python 3.9+
- Docker and Docker Compose

### Local Development

Start infrastructure services:
```bash
docker-compose up -d
```

This starts:
- PostgreSQL (pgvector) on port 5435
- MinIO (S3-compatible storage) on ports 9000 (API) and 9001 (console)
- RabbitMQ on ports 5672 (AMQP) and 15672 (management UI)

### Running Services

**Using Docker Compose (Recommended):**
```bash
# Start all services
docker compose up -d

# View logs
docker compose logs -f data-processing-job-worker

# Stop all services
docker compose down
```

**Running locally for development:**

**data-api**:
```bash
cd data-api
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows
pip install -r requirements.txt
cp .env.sample .env  # Edit .env with your configuration
cd src
python main.py
```

**data-processing-job (Celery worker)**:
```bash
cd data-processing-job
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.sample .env  # Edit .env with your configuration
cd src
celery -A celery_worker worker --loglevel=info --concurrency=4
```

### Configuration

Both services use Pydantic Settings to load configuration from `.env` files. See `.env.sample` in each service directory for required environment variables.

Key environment variables:
- `DATABASE_URL`: PostgreSQL connection string
- `RABBITMQ_URL`: RabbitMQ connection string (format: `amqp://user:pass@host:port/`)
- `S3_ENDPOINT_URL`: MinIO/S3 endpoint
- `GOOGLE_GENAI_API_KEY`: Google GenAI API key (data-processing-job)
- `PRETRAIN_MODEL_PATH`: Hugging Face model path for tokenizer (data-processing-job)

## Key Technical Details

### Celery Task Processing

Celery is configured in `data-processing-job/src/app/celery_app/config.py`:
- RabbitMQ as broker (from `settings.RABBITMQ_URL`)
- RPC result backend for task status
- Topic exchange with routing keys
- Dead letter exchange (DLX) for failed tasks
- Reliability settings: acks_late=True, task_reject_on_worker_lost=True

Task queues:
- `preprocess_queue`: Document preprocessing tasks (routing key: `preprocess.*`)
- `upsert_queue`: Embedding generation and storage (routing key: `upsert.*`)
- `dlq`: Dead letter queue for failed tasks (routing key: `dlq.#`)

Worker command: `celery -A celery_worker worker --loglevel=info --concurrency=4`

### Vector Storage

Uses custom `DocumentEmbeddingsRepository` with direct SQL queries (NO LlamaIndex):
- pgvector extension for PostgreSQL
- HNSW indexing for efficient similarity search
- Hybrid search capability (vector + full-text search)
- 1024-dimensional embeddings
- Direct SQL control via SQLAlchemy

Table: `document_embeddings`
- Columns: chunk_id, document_id, kb_id, embedding (vector), text, metadata (jsonb)
- Supports upsert operations for idempotency
- Vector similarity search using cosine distance (`<=>` operator)
- Full-text search using PostgreSQL tsvector

### Document Processing (Celery Pipeline)

The document processing flow:
1. **preprocess_document** task:
   - Fetches document from S3
   - Parser extracts text from HTML, with special handling for tables
   - Splitter chunks text using a pretrained tokenizer (e.g., from Hugging Face)
   - Uploads chunks to S3
   - Inserts chunk records into database

2. **upsert_chunk** task (spawned for each chunk):
   - Generates embeddings using OpenAI-compatible API
   - Stores embeddings in pgvector via `DocumentEmbeddingsRepository`
   - Updates chunk status to 'Succeed'

3. **finalize_document** task:
   - Checks if all chunks succeeded
   - Updates document status accordingly

## Common Commands

**Docker Compose commands:**
```bash
# Start all services (infrastructure + applications)
docker compose up -d

# Start only infrastructure
docker compose up -d postgres rabbitmq minio litellm

# View logs for specific service
docker compose logs -f data-processing-job-worker

# Restart a service
docker compose restart data-processing-job-worker

# Stop all services
docker compose down
```

**Local development:**
```bash
# Run data-api locally
cd data-api/src && python main.py

# Run Celery worker locally
cd data-processing-job/src && celery -A celery_worker worker --loglevel=info

# Run health check service locally
cd data-processing-job/src && python main.py
```

View RabbitMQ management UI:
```bash
# Navigate to http://localhost:15672
# Default credentials: guest/guest
```

Access MinIO console:
```bash
# Navigate to http://localhost:9001
# Default credentials: minioadmin/minioadmin
```

## Notes

- Both services use FastAPI with uvicorn
- Logging is configured in each main.py with console and file handlers
- **LlamaIndex has been removed** - system now uses custom Document class and direct SQL
- Celery provides asynchronous task processing with reliability features
- Connection resilience: Celery auto-retries tasks on failure with exponential backoff
- The current branch is `chunking` (working on document chunking features)
- Main branch is `master` (use this for pull requests)

## Migration from LlamaIndex

The system has been migrated away from LlamaIndex:
- Custom `Document` dataclass replaces `llama_index.core.Document`
- `DocumentEmbeddingsRepository` replaces `PGVectorStore`
- Direct SQL queries for all vector operations
- See `MIGRATION_SUMMARY.md` for complete migration details
