# Data-Hub Refactoring Plan: Celery + Custom Vector Store + LightRAG

## Context

The current data-hub system uses RabbitMQ with custom aio_pika consumers and LlamaIndex's PGVectorStore for document processing and vector storage. This refactor aims to:

1. **Replace aio_pika consumers with Celery** for robust task processing with retry policies, dead letter queues (DLQ), and better observability
2. **Remove LlamaIndex storage dependency** and implement a custom vector store using Postgres + pgvector directly for explicit schema control
3. **Integrate LightRAG** as an optional graph-based RAG mode alongside the classic vector-based RAG

**Why this change?**
- Current system has NO DLQ (failed messages are dropped)
- LlamaIndex auto-creates schema with unpredictable table names
- Need graph-based RAG capabilities for advanced knowledge retrieval
- Celery provides industry-standard task orchestration with proven reliability

---

## 1. High-Level Architecture

### Current System
```
data-api → RabbitMQ → data-processing-job (aio_pika consumers)
                          ↓
                    LlamaIndex PGVectorStore
                    (auto-created: data_vectordb_test_v2)
```

**Issues:**
- No DLQ (failed messages lost)
- No retry policy
- Tight coupling to LlamaIndex
- Unpredictable schema management

### Target System
```
data-api → Celery/RabbitMQ → data-processing-job (Celery workers)
                                  ↓
                            Custom Vector Store
                            (explicit schema: document_embeddings)
                                  ↓
                            Optional: LightRAG mode
                            (graph tables: lightrag_entities/relations)
```

**Benefits:**
- DLQ with RabbitMQ dead-letter exchange
- Configurable retry policies (exponential backoff)
- Explicit migrations via Alembic
- Support for both classic RAG and graph-based RAG

### Message Flow

**Classic RAG Mode:**
```
POST /documents
  → data-api: upload to S3, insert DB record, send Celery task
  → Task: preprocess_document(doc_id, s3_key, kb_id)
      - Fetch from S3, parse HTML, chunk text
      - Upload chunks to S3
      - Insert chunk records (status='Processing')
      - Return list of (chunk_id, s3_key)
  → Chord: [upsert_chunk(cid, key) for each chunk in parallel]
      - Fetch chunk from S3
      - Generate embedding via OpenAI-like API
      - Upsert to document_embeddings table
      - Update chunk.status='Succeed'
  → Callback: finalize_document(doc_id)
      - Check all chunks succeeded
      - Update document.status='Succeed'
```

**LightRAG Mode:**
```
POST /documents (KB has rag_mode='lightrag')
  → Task: preprocess_document_lightrag(doc_id, s3_key, kb_id)
      - Fetch from S3, parse HTML
      - Extract entities/relations using LightRAG library
      - Store in lightrag_entities, lightrag_relations tables
      - Generate entity embeddings
      - Update document.status='Succeed'
```

---

## 2. Celery Adoption Plan

**✅ STATUS: DONE (Completed: 2026-02-18)**

**Implementation Summary:**
- ✅ Celery app configured with RabbitMQ broker, DLX for DLQ, retry policies
- ✅ Tasks created: preprocess_document, upsert_chunk, finalize_document, process_dlq_message
- ✅ Replaced aio_pika consumers with Celery tasks
- ✅ data-api now sends tasks via Celery client instead of RabbitMQ publish
- ✅ Worker entry point created: celery_worker.py
- ✅ Idempotency checks implemented in all tasks
- ✅ Retry behavior: max 3 retries, exponential backoff, DLQ routing after exhaustion

**Commands to Run:**
```bash
# Start infrastructure
docker-compose up -d

# Apply migrations
cd data-processing-job
alembic upgrade head

# Start Celery worker (from data-processing-job/src)
cd src
celery -A celery_worker worker --loglevel=info --concurrency=4

# Optional: Start Flower monitoring UI
celery -A celery_worker flower --port=5555
# Access at http://localhost:5555

# Run data-api (in separate terminal)
cd data-api/src
python main.py

# Run tests
cd data-processing-job
pytest tests/ -v
```

**New Environment Variables:**
- `RABBITMQ_URL`: Already existed, now used by Celery (format: amqp://user:pass@host:port/)
- `EMBEDDING_API_BASE`: OpenAI-like embedding API endpoint (default: https://aix-llm-gateway.cmctelecom.vn/v1)
- `EMBEDDING_MODEL_NAME`: Embedding model name (default: rag-embedding-model)
- `EMBEDDING_API_KEY`: API key for embedding service (default: fake)
- `EMBEDDING_BATCH_SIZE`: Batch size for embedding generation (default: 4)

**Files Added:**
- `data-processing-job/src/app/celery_app/config.py`
- `data-processing-job/src/app/celery_app/tasks/preprocess_tasks.py`
- `data-processing-job/src/app/celery_app/tasks/upsert_tasks.py`
- `data-processing-job/src/app/celery_app/tasks/dlq_tasks.py`
- `data-processing-job/src/celery_worker.py`
- `data-api/src/app/celery_client.py`

**Files Removed:**
- `data-processing-job/src/app/application/consumers/base_consumer.py`
- `data-processing-job/src/app/application/consumers/preprocess_document_event_consumer.py`
- `data-processing-job/src/app/application/consumers/upsert_document_event_consumer.py`

**Dependencies Updated:**
- Added: celery==5.4.0, kombu==5.4.0
- Removed: aio-pika==9.5.4

---

## 2. Celery Adoption Plan (Original Spec)

### 2.1 Directory Structure

**New files in data-processing-job:**
```
src/app/celery_app/
  __init__.py
  config.py                      # Celery app + broker/queue config
  tasks/
    preprocess_tasks.py          # preprocess_document, finalize_document
    upsert_tasks.py              # upsert_chunk
    lightrag_tasks.py            # preprocess_document_lightrag
    dlq_tasks.py                 # process_dlq_message
  monitoring.py                  # Correlation IDs, structured logging

celery_worker.py                 # Entry: celery -A app.celery_app worker
```

**New file in data-api:**
```
src/app/celery_client.py         # Client to send tasks (not a worker)
```

### 2.2 Core Configuration

**File:** `data-processing-job/src/app/celery_app/config.py`

```python
from celery import Celery
from kombu import Exchange, Queue

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
    Queue('dlq', exchange=DLX_EXCHANGE, routing_key='dlq.#', durable=True),
)

celery_app.conf.update(
    task_queues=task_queues,

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
)
```

**Key Features:**
- **DLQ via RabbitMQ DLX:** Failed tasks after max retries → dlq queue
- **acks_late=True:** Task acknowledged only after completion (worker crash → requeue)
- **prefetch_multiplier=1:** Fetch 1 task at a time (critical for long tasks)
- **Exponential backoff:** Prevents thundering herd on transient failures
- **Time limits:** Soft limit for graceful shutdown, hard limit for forced kill

### 2.3 Task Definitions

**File:** `data-processing-job/src/app/celery_app/tasks/preprocess_tasks.py`

```python
@celery_app.task(bind=True, name='preprocess_document', acks_late=True)
def preprocess_document(self, document_id: str, s3_key: str, kb_id: str,
                        correlation_id: str = None):
    """
    1. Fetch document from S3
    2. Parse HTML (via DocumentAI service)
    3. Chunk text (using existing parser/splitter)
    4. Upload chunks to S3
    5. Insert chunk records with status='Processing'
    6. Return [(chunk_id, s3_key), ...] for chord

    Idempotency: Check document.status before processing.
    """
    # Check if already processed
    if document.status in ['Processing', 'Succeed']:
        return []

    # ... (parsing, chunking, S3 upload logic)

    return [(chunk_id, s3_key) for each chunk]

@celery_app.task(bind=True, name='finalize_document', acks_late=True)
def finalize_document(self, chunk_results, document_id: str, correlation_id: str = None):
    """
    Callback after all upsert_chunk tasks complete.
    Check all chunks succeeded, update document.status='Succeed'.
    """
```

**File:** `data-processing-job/src/app/celery_app/tasks/upsert_tasks.py`

```python
@celery_app.task(bind=True, name='upsert_chunk', acks_late=True)
def upsert_chunk(self, chunk_id: str, s3_key: str, correlation_id: str = None):
    """
    1. Fetch chunk text from S3
    2. Generate embedding (via EmbeddingService)
    3. Upsert to document_embeddings table
    4. Update chunk.status='Succeed'

    Idempotency: Check chunk.status before processing.
    """
    # Check if already succeeded
    if chunk.status == 'Succeed':
        return chunk_id

    # ... (embedding generation, vector store upsert)
```

### 2.4 Workflow Orchestration

**Modified:** `data-api/src/app/application/services/document_service.py`

```python
from celery import chain, chord
from app.celery_client import celery_client

async def add_documents(self, kb_id: str, files: List[UploadFile]):
    # ... (S3 upload, DB insert)

    kb = await self.kb_repository.get(kb_id)

    if kb.rag_mode == 'classic':
        # Chain: preprocess → chord([upsert_chunk...]) → finalize
        workflow = chain(
            preprocess_document.s(doc.id, s3_key, kb_id, correlation_id),
            # Chord dynamically created from preprocess result
            chord([upsert_chunk.s(cid, ck, correlation_id)
                   for cid, ck in []],  # Filled by preprocess result
                  finalize_document.s(doc.id, correlation_id))
        )
        workflow.apply_async()

    elif kb.rag_mode == 'lightrag':
        preprocess_document_lightrag.apply_async(
            kwargs={'document_id': doc.id, 's3_key': s3_key, 'kb_id': kb_id}
        )
```

### 2.5 DLQ Operational Strategy

**RabbitMQ DLX Configuration:**
- Automatically created by Celery via `queue_arguments`
- Failed tasks (after max retries) → routed to `dlq` queue via `dlx` exchange

**DLQ Monitoring:**
```python
# File: data-processing-job/src/app/celery_app/tasks/dlq_tasks.py
@celery_app.task(name='process_dlq_message')
def process_dlq_message(task_id: str, error_msg: str, retry_count: int):
    # Log to monitoring system
    # Insert into dlq_log table for audit
    # Optionally re-enqueue with manual intervention
```

**Operational Commands:**
```bash
# View DLQ messages
curl -u guest:guest http://localhost:15672/api/queues/%2F/dlq/get

# Re-enqueue after fixing code
celery_client.send_task('preprocess_document', kwargs={...})
```

### 2.6 Observability

**Correlation IDs:**
- Generate UUID at document upload
- Pass through all tasks: `preprocess_document(..., correlation_id=uuid)`
- Include in all logs: `logger.info(f"[{correlation_id}] ...")`

**Monitoring:**
- **Celery Flower:** Real-time task monitoring dashboard
  ```bash
  celery -A app.celery_app flower --port=5555
  ```
- **Metrics to track:** Task success rate, retry rate, DLQ count, task duration
- **Structured logging:** JSON logs with task_id, correlation_id, document_id, timestamp

---

## 3. Storage Layer Refactor (No LlamaIndex)

**✅ STATUS: DONE (Completed: 2026-02-18)**

**Implementation Summary:**
- ✅ Removed LlamaIndex dependencies (llama-index, llama-index-vector-stores-postgres, llama-index-llms-google-genai)
- ✅ Created custom DocumentEmbeddingsRepository with raw SQL queries for pgvector
- ✅ Implemented upsert, delete_by_document, query_by_vector, hybrid_search methods
- ✅ Created EmbeddingService for OpenAI-like API calls
- ✅ Created VectorStoreService as business logic wrapper
- ✅ Added Alembic for migrations
- ✅ Created migrations: document_embeddings table, kb_rag_mode column, dlq_log table
- ✅ Uses cosine distance for similarity, HNSW indexing (m=16, ef_construction=64)
- ✅ Full-text search support via tsvector column

**Database Schema Changes:**
```sql
-- New table: document_embeddings
CREATE TABLE public.document_embeddings (
    id UUID PRIMARY KEY,
    chunk_id UUID NOT NULL UNIQUE,
    document_id UUID NOT NULL,
    kb_id UUID NOT NULL,
    embedding vector(1024) NOT NULL,
    text TEXT NOT NULL,
    metadata JSONB DEFAULT '{}',
    text_tsvector tsvector GENERATED,
    created_at TIMESTAMP,
    updated_at TIMESTAMP
);

-- New columns in knowledge_base
ALTER TABLE knowledge_base
ADD COLUMN rag_mode rag_mode_enum DEFAULT 'classic',
ADD COLUMN embedding_dim INTEGER DEFAULT 1024;

-- New table: dlq_log
CREATE TABLE public.dlq_log (
    id UUID PRIMARY KEY,
    task_id TEXT NOT NULL,
    error_message TEXT,
    retry_count INTEGER,
    created_at TIMESTAMP
);
```

**Migration Commands:**
```bash
cd data-processing-job

# Run migrations
alembic upgrade head

# Rollback (if needed)
alembic downgrade -1

# Check current version
alembic current

# View migration history
alembic history
```

**Vector Store Usage:**
```python
# In Celery tasks:
from app.application.services.vector_store_service import VectorStoreService
from app.application.services.embedding_service import EmbeddingService

# Generate embedding
embedding_service = EmbeddingService()
embedding = await embedding_service.get_embedding(text)

# Store in pgvector
vector_store_service = VectorStoreService(async_session)
await vector_store_service.upsert_embedding(
    chunk_id=chunk_id,
    document_id=document_id,
    kb_id=kb_id,
    embedding=embedding,
    text=text,
    metadata=metadata
)

# Query by similarity
results = await vector_store_service.search(
    kb_id=kb_id,
    query_embedding=query_embedding,
    top_k=10
)
```

**Files Added:**
- `data-processing-job/src/app/infrastructure/repositories/document_embeddings_repository.py`
- `data-processing-job/src/app/application/services/embedding_service.py`
- `data-processing-job/src/app/application/services/vector_store_service.py`
- `data-processing-job/alembic.ini`
- `data-processing-job/migrations/env.py`
- `data-processing-job/migrations/versions/001_add_document_embeddings_table.py`
- `data-processing-job/migrations/versions/002_add_kb_rag_mode.py`
- `data-processing-job/migrations/versions/003_add_dlq_log_table.py`

**Files Removed:**
- `data-processing-job/src/app/application/core/indexer.py` (used LlamaIndex)

**Dependencies Updated:**
- Added: alembic==1.13.1, psycopg2-binary==2.9.9
- Removed: llama-index==0.12.31, llama-index-vector-stores-postgres==0.4.2, llama-index-llms-google-genai==0.1.12

**Testing:**
```bash
cd data-processing-job

# Run unit tests
pytest tests/test_document_embeddings_repository.py -v

# Run all tests
pytest tests/ -v

# Test pgvector queries manually
psql -h localhost -p 5435 -U datahub -d datahub
# Then run:
SELECT chunk_id, 1 - (embedding <=> '[0.1,0.2,...]'::vector) AS similarity
FROM document_embeddings
WHERE kb_id = 'your-kb-id'
ORDER BY embedding <=> '[0.1,0.2,...]'::vector
LIMIT 10;
```

**Notes:**
- PG_VECTOR.md was used as reference for query patterns and conventions
- Cosine distance operator: `<=>` (lower is more similar, returns 0-2)
- Similarity score: `1 - (embedding <=> query)` (higher is more similar, returns -1 to 1)
- HNSW index significantly speeds up similarity search (O(log n) vs O(n))
- Metadata filters use JSONB operators: `->` for traversal, `->>` for text extraction

---

## 3. Storage Layer Refactor (Original Spec)

### 3.1 Database Schema

**Migration:** `001_add_document_embeddings_table.py`

```sql
-- Enable pgvector
CREATE EXTENSION IF NOT EXISTS vector;

-- Main embeddings table
CREATE TABLE public.document_embeddings (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    chunk_id UUID NOT NULL REFERENCES public.chunk(id) ON DELETE CASCADE,
    document_id UUID NOT NULL REFERENCES public.document(id) ON DELETE CASCADE,
    kb_id UUID NOT NULL REFERENCES public.knowledge_base(id) ON DELETE CASCADE,

    -- Vector (1024-dimensional, configurable)
    embedding vector(1024) NOT NULL,

    -- Full text for hybrid search
    text TEXT NOT NULL,

    -- Metadata (chunk position, source doc, custom fields)
    metadata JSONB NOT NULL DEFAULT '{}',

    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),

    CONSTRAINT unique_chunk_embedding UNIQUE (chunk_id)
);

-- HNSW index for fast similarity search (cosine distance)
CREATE INDEX idx_embeddings_vector_hnsw
ON public.document_embeddings
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);

-- Metadata filtering (JSONB GIN index)
CREATE INDEX idx_embeddings_metadata
ON public.document_embeddings
USING GIN (metadata jsonb_path_ops);

-- Lookups by document/KB
CREATE INDEX idx_embeddings_document_id ON public.document_embeddings (document_id);
CREATE INDEX idx_embeddings_kb_id ON public.document_embeddings (kb_id);

-- Full-text search (hybrid search support)
ALTER TABLE public.document_embeddings
ADD COLUMN text_tsvector tsvector
GENERATED ALWAYS AS (to_tsvector('english', text)) STORED;

CREATE INDEX idx_embeddings_fulltext
ON public.document_embeddings
USING GIN (text_tsvector);
```

**Migration:** `002_add_kb_rag_mode.py`

```sql
-- Add RAG mode to knowledge_base
CREATE TYPE rag_mode_enum AS ENUM ('classic', 'lightrag');

ALTER TABLE public.knowledge_base
ADD COLUMN rag_mode rag_mode_enum DEFAULT 'classic';

ALTER TABLE public.knowledge_base
ADD COLUMN embedding_dim INTEGER DEFAULT 1024;
```

**Migration:** `003_add_dlq_log_table.py`

```sql
-- Audit log for DLQ messages
CREATE TABLE public.dlq_log (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    task_id TEXT NOT NULL,
    error_message TEXT,
    retry_count INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT NOW()
);
```

### 3.2 HNSW Index Configuration

**Parameters (from existing system):**
- `m = 16`: Bi-directional links per node (memory vs recall tradeoff)
- `ef_construction = 64`: Search breadth during index build (quality vs speed)
- `ef_search = 40`: Search breadth during query (recall vs speed)
- **Distance metric:** `vector_cosine_ops` (cosine similarity)

**For production:** Use `CREATE INDEX CONCURRENTLY` for zero-downtime builds

### 3.3 Migration from LlamaIndex Table

**Existing table:** `data_vectordb_test_v2` (auto-created by LlamaIndex)

**Migration script:**
```sql
-- Migrate data to new schema
INSERT INTO public.document_embeddings (chunk_id, document_id, kb_id, embedding, text, metadata)
SELECT
    (metadata->>'chunk_id')::UUID,
    (metadata->>'document_id')::UUID,
    (metadata->>'kb_id')::UUID,
    embedding,
    text,
    metadata
FROM data_vectordb_test_v2
WHERE metadata->>'chunk_id' IS NOT NULL;

-- Verify data integrity, then drop old table
DROP TABLE data_vectordb_test_v2;
```

### 3.4 Python Repository Pattern

**File:** `data-processing-job/src/app/infrastructure/repositories/document_embeddings_repository.py`

```python
class DocumentEmbeddingsRepository:
    async def upsert(self, chunk_id, document_id, kb_id, embedding, text, metadata):
        """Upsert embedding (ON CONFLICT DO UPDATE on chunk_id)"""
        stmt = insert(document_embeddings).values(...).on_conflict_do_update(
            index_elements=['chunk_id'],
            set_=dict(embedding=embedding, text=text, updated_at=text('NOW()'))
        )

    async def delete_by_document(self, document_id):
        """Delete all embeddings for a document"""

    async def query_by_vector(self, kb_id, query_embedding, top_k=10, metadata_filter=None):
        """Vector similarity search with optional metadata filtering"""
        query = """
            SELECT chunk_id, document_id, text, metadata,
                   1 - (embedding <=> :query_embedding::vector) AS similarity
            FROM document_embeddings
            WHERE kb_id = :kb_id
            ORDER BY embedding <=> :query_embedding::vector
            LIMIT :top_k
        """

    async def hybrid_search(self, kb_id, query_embedding, query_text, top_k=10, alpha=0.5):
        """Combine vector similarity + full-text search (weighted by alpha)"""
        # Uses vector_sim * alpha + text_rank * (1-alpha)
```

**File:** `data-processing-job/src/app/application/services/vector_store_service.py`

```python
class VectorStoreService:
    """Business logic wrapper for vector operations"""
    def __init__(self):
        self.repo = DocumentEmbeddingsRepository()

    async def upsert_embedding(self, chunk_id, document_id, embedding, text, metadata):
        # Get kb_id from document, then call repo

    async def search(self, kb_id, query_embedding, top_k=10):
        return await self.repo.query_by_vector(kb_id, query_embedding, top_k)
```

**File:** `data-processing-job/src/app/application/services/embedding_service.py`

```python
class EmbeddingService:
    """Wrapper for OpenAI-like embedding API"""
    async def get_embedding(self, text: str) -> list[float]:
        # POST to embedding API, return 1024-dim vector

    async def get_embeddings_batch(self, texts: list[str]) -> list[list[float]]:
        # Batch processing (4 at a time)
```

### 3.5 Alembic Setup

```bash
cd data-processing-job
pip install alembic
alembic init migrations

# Generate migration
alembic revision --autogenerate -m "Add document_embeddings table"

# Apply
alembic upgrade head
```

---

## 4. LightRAG Integration Plan

### 4.1 Library Selection

**Use:** LightRAG Python library (https://github.com/HKUDS/LightRAG)

**Features:**
- Entity/relation extraction from text
- Knowledge graph construction
- Graph-based retrieval (node embeddings + graph traversal)
- Compatible with OpenAI-like APIs

**Wrapper:** `data-processing-job/src/app/application/core/lightrag_wrapper.py`

```python
class LightRAGWrapper:
    def __init__(self, working_dir, embedding_func, llm_func):
        self.rag = LightRAG(
            working_dir=working_dir,
            llm_model_func=llm_func,
            embedding_func=embedding_func,
        )

    async def insert_document(self, text: str):
        """Insert document and build graph"""
        await self.rag.ainsert(text)

    def get_entities(self) -> list:
        """Extract all entities from internal graph"""

    def get_relations(self) -> list:
        """Extract all relations from internal graph"""
```

### 4.2 Configuration Model

**KB-level configuration:**
```python
# knowledge_base table
rag_mode = Column(SQLAlchemyEnum('classic', 'lightrag'), default='classic')
lightrag_config = Column(JSON, default={})
# Example: {"entity_extraction_model": "gpt-4", "max_entities": 100}
```

### 4.3 Storage Design

**Choice:** Postgres tables (not external graph DB)

**Justification:**
- Simpler infrastructure (no Neo4j/ArangoDB to deploy)
- Leverage existing Postgres connection
- Use pgvector for entity embeddings
- SQL recursive CTEs for graph traversal

**Migration:** `004_add_lightrag_tables.py`

```sql
-- Entities table
CREATE TABLE public.lightrag_entities (
    id UUID PRIMARY KEY,
    kb_id UUID NOT NULL REFERENCES knowledge_base(id),
    document_id UUID REFERENCES document(id),

    entity_name TEXT NOT NULL,
    entity_type TEXT,           -- person, org, location, etc.
    description TEXT,

    embedding vector(1024),     -- Entity embedding
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_lightrag_entities_embedding_hnsw
ON public.lightrag_entities
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);

-- Relations table
CREATE TABLE public.lightrag_relations (
    id UUID PRIMARY KEY,
    kb_id UUID NOT NULL,
    document_id UUID REFERENCES document(id),

    source_entity_id UUID REFERENCES lightrag_entities(id),
    target_entity_id UUID REFERENCES lightrag_entities(id),

    relation_type TEXT NOT NULL,
    description TEXT,
    metadata JSONB DEFAULT '{}'
);
```

**Graph traversal example:**
```sql
-- Find entities within 2 hops
WITH RECURSIVE entity_graph AS (
    SELECT id, entity_name, 0 AS depth
    FROM lightrag_entities
    WHERE entity_name = 'Starting Entity'

    UNION

    SELECT e.id, e.entity_name, eg.depth + 1
    FROM entity_graph eg
    JOIN lightrag_relations r ON (eg.id = r.source_entity_id OR eg.id = r.target_entity_id)
    JOIN lightrag_entities e ON (e.id = r.source_entity_id OR e.id = r.target_entity_id)
    WHERE eg.depth < 2
)
SELECT * FROM entity_graph;
```

### 4.4 Ingestion Pipeline Differences

**Classic RAG:**
```
parse HTML → chunk text → embed chunks → store in document_embeddings
```

**LightRAG:**
```
parse HTML → extract entities/relations → build graph → embed entities → store in lightrag_entities/relations
```

**Task:** `data-processing-job/src/app/celery_app/tasks/lightrag_tasks.py`

```python
@celery_app.task(name='preprocess_document_lightrag')
def preprocess_document_lightrag(document_id, s3_key, kb_id):
    # 1. Fetch from S3, parse HTML
    # 2. Extract entities/relations using LightRAG
    # 3. Store entities in lightrag_entities table
    # 4. Store relations in lightrag_relations table
    # 5. Generate entity embeddings
    # 6. Update document.status='Succeed'
```

### 4.5 Query Interface (Future)

**Classic RAG:**
```python
# Vector similarity over chunks
POST /api/v1/{kb_id}/query
{
  "query": "What is the capital of France?",
  "top_k": 10
}
```

**LightRAG:**
```python
# Graph-based retrieval
POST /api/v1/{kb_id}/query
{
  "query": "What is the capital of France?",
  "mode": "hybrid"  # naive, local, global, hybrid
}

# Backend:
# - Extract entities from query
# - Find similar entities (embedding)
# - Traverse graph (relations)
# - Synthesize answer
```

---

## 5. Code Change Map

### 5.1 Files to Add

**data-processing-job:**
```
src/app/celery_app/__init__.py
src/app/celery_app/config.py
src/app/celery_app/tasks/preprocess_tasks.py
src/app/celery_app/tasks/upsert_tasks.py
src/app/celery_app/tasks/lightrag_tasks.py
src/app/celery_app/tasks/dlq_tasks.py
src/app/celery_app/monitoring.py

src/app/application/services/embedding_service.py
src/app/application/services/vector_store_service.py
src/app/application/core/lightrag_wrapper.py

src/app/infrastructure/repositories/document_embeddings_repository.py
src/app/infrastructure/repositories/lightrag_repository.py

migrations/versions/001_add_document_embeddings_table.py
migrations/versions/002_add_kb_rag_mode.py
migrations/versions/003_add_dlq_log_table.py
migrations/versions/004_add_lightrag_tables.py

celery_worker.py
```

**data-api:**
```
src/app/celery_client.py
```

### 5.2 Files to Modify

**data-processing-job:**
- `main.py` - Remove consumer init, keep health check
- `requirements.txt` - Add celery, alembic, lightrag; remove llama-index, aio-pika
- `configurations/configurations.py` - Add Celery settings
- `.env.sample` - Add CELERY_BROKER_URL, etc.

**data-api:**
- `application/services/document_service.py` - Replace rabbitmq.publish with celery_client.send_task
- `infrastructure/connectors/postgres/schema.py` - Add rag_mode, embedding_dim to KnowledgeBase
- `requirements.txt` - Add celery (client only)
- `.env.sample` - Add CELERY_BROKER_URL

### 5.3 Files to Remove

**data-processing-job:**
```
application/consumers/base_consumer.py
application/consumers/preprocess_document_event_consumer.py
application/consumers/upsert_document_event_consumer.py
application/core/indexer.py  (LlamaIndex usage)
```

**Dependencies to remove:**
- llama-index==0.12.31
- llama-index-vector-stores-postgres==0.4.2
- llama-index-llms-google-genai==0.1.12
- aio-pika==9.5.4

### 5.4 PR Sequence

**PR1: Infrastructure Setup (No Breaking Changes)**
- Add Celery config, migrations (Alembic)
- Add document_embeddings table migration
- Add embedding_service, vector_store_service (parallel to existing)
- Add Celery tasks (not activated yet)
- Add celery_worker.py entry point
- **Test:** Celery worker starts, connects to RabbitMQ

**PR2: Feature Flag Transition**
- Add celery_client to data-api
- Modify document_service with feature flag:
  ```python
  if settings.USE_CELERY:
      celery_client.send_task(...)
  else:
      rabbitmq_service.publish(...)  # Old path
  ```
- Deploy with USE_CELERY=False
- Test with USE_CELERY=True on staging
- **Test:** Both paths work, toggle via env var

**PR3: Remove aio-pika, Enable Celery**
- Remove feature flag, default to Celery
- Remove application/consumers/
- Remove aio-pika dependency
- Remove LlamaIndex dependencies
- **Test:** Only Celery tasks execute, DLQ captures failures

**PR4: LightRAG Integration (Optional)**
- Add LightRAG tables migration
- Add lightrag_wrapper, lightrag_tasks
- Add rag_mode column to knowledge_base
- Route by rag_mode in document_service
- **Test:** KB with rag_mode='lightrag' stores entities/relations

---

## 6. Risks & Mitigations

### At-Least-Once Delivery

**Risk:** Task may execute multiple times (worker crash + requeue)

**Mitigation:**
- Idempotency checks: Check status before processing
- UNIQUE constraints on chunk_id
- ON CONFLICT DO UPDATE for upserts
- Correlation IDs for tracing duplicates

### Poison Messages

**Risk:** Malformed message causes infinite retry loop

**Mitigation:**
- Max retries = 3
- Exponential backoff
- DLQ captures after max retries
- Monitor DLQ message count

### Performance Risks

**Embedding API Rate Limits:**
- Batch requests (4 chunks)
- Retry with backoff on 429 errors

**HNSW Index Build:**
- Use CREATE INDEX CONCURRENTLY
- Schedule during low-traffic periods

**Large Documents:**
- Chunk limit (max 5000 per doc)
- Stream processing (don't load all in memory)
- Task time limits (2 hours hard)

---

## 7. Acceptance Checklist

### Celery Adoption
- [ ] Worker starts without errors
- [ ] Task executes successfully for sample document
- [ ] Task retry works (force failure, verify exponential backoff)
- [ ] DLQ captures message after max retries
- [ ] Correlation IDs in all logs
- [ ] Idempotency: re-run task, no duplicates
- [ ] Celery Flower accessible at http://localhost:5555

**Test:**
```bash
celery -A app.celery_app worker --loglevel=info
celery -A app.celery_app flower
curl -X POST http://localhost:8000/api/v1/{kb_id}/documents -F "files=@test.html"
```

### Storage Layer
- [ ] Alembic migrations apply: `alembic upgrade head`
- [ ] document_embeddings table exists with HNSW index
- [ ] Upsert works (insert + update on conflict)
- [ ] Vector query returns results
- [ ] Hybrid search (vector + text) works
- [ ] Metadata filtering works
- [ ] Delete by document removes all embeddings
- [ ] Migration from data_vectordb_test_v2 succeeds

**Test:**
```bash
psql -h localhost -p 5435 -U datahub -c "\d document_embeddings"
psql -h localhost -p 5435 -U datahub -c "SELECT chunk_id, 1 - (embedding <=> '[...]') AS sim FROM document_embeddings ORDER BY embedding <=> '[...]' LIMIT 10"
```

### LightRAG Integration
- [ ] LightRAG tables created
- [ ] KB with rag_mode='lightrag' triggers LightRAG task
- [ ] Entities stored in lightrag_entities
- [ ] Relations stored in lightrag_relations
- [ ] Entity embeddings generated
- [ ] Graph traversal query works
- [ ] Classic mode still works (no regression)

**Test:**
```bash
curl -X POST http://localhost:8000/api/v1/knowledge_bases -d '{"name": "Test", "rag_mode": "lightrag"}'
psql -h localhost -p 5435 -U datahub -c "SELECT entity_name FROM lightrag_entities WHERE kb_id='{kb_id}'"
```

### Docker Compose
- [ ] All services start: `docker-compose up -d`
- [ ] RabbitMQ UI: http://localhost:15672
- [ ] DLX exchange created
- [ ] Queues bound correctly (preprocess_queue, upsert_queue, dlq)

---

## 8. Configuration

### Environment Variables

**data-processing-job/.env:**
```bash
CELERY_BROKER_URL=amqp://guest:guest@localhost:5672/
CELERY_RESULT_BACKEND=rpc://
EMBEDDING_API_BASE=https://aix-llm-gateway.cmctelecom.vn/v1
EMBEDDING_MODEL_NAME=rag-embedding-model
EMBEDDING_BATCH_SIZE=4
```

**data-api/.env:**
```bash
CELERY_BROKER_URL=amqp://guest:guest@localhost:5672/
USE_CELERY=True  # For PR2 feature flag
```

### Commands

```bash
# Start infrastructure
docker-compose up -d

# Run Celery worker
cd data-processing-job/src
celery -A app.celery_app worker --loglevel=info --concurrency=4

# Run Flower
celery -A app.celery_app flower --port=5555

# Apply migrations
cd data-processing-job
alembic upgrade head

# Run data-api
cd data-api/src
python main.py

# Check task status
celery -A app.celery_app inspect active
```

---

## Summary

This plan refactors data-hub with:
1. **Celery** for robust task orchestration (retry, DLQ, observability)
2. **Custom vector store** using Postgres + pgvector (explicit schema, Alembic migrations)
3. **LightRAG** as optional graph-based RAG mode
4. **Backward-compatible rollout** via feature flags (PR1 → PR2 → PR3 → PR4)

**Critical files:**
- `/home/minh/MyProject/data-hub/data-processing-job/src/app/celery_app/config.py`
- `/home/minh/MyProject/data-hub/data-processing-job/src/app/celery_app/tasks/preprocess_tasks.py`
- `/home/minh/MyProject/data-hub/data-processing-job/src/app/infrastructure/repositories/document_embeddings_repository.py`
- `/home/minh/MyProject/data-hub/data-api/src/app/application/services/document_service.py`
