"""
Worker composition root.

WorkerContainer holds per-process singletons that are safe to share across
asyncio.run() calls within the same Celery worker process.

The SQLAlchemy engine uses NullPool so connections are never cached between
asyncio.run() calls.  Each call creates fresh connections on its own event
loop and closes them on completion, eliminating the "Future attached to a
different loop" error that arises with a pooled engine after Celery's
prefork fork().

Initialization:
  - Explicit: called once per worker process via the worker_process_init
    Celery signal in celery_worker.py.
  - Lazy fallback: if a task runs without the signal (e.g. in tests or a
    development runner), the container self-initializes on first property
    access.
"""
from __future__ import annotations

import logging
import os
from typing import Any, Optional

from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
from sqlalchemy.pool import NullPool

from app.configurations.configurations import settings
from app.application.core.parser import DocumentParser
from app.application.core.markdown_splitter import MarkdownSplitter
from app.application.services.embedding_service import EmbeddingService
from app.infrastructure.clients.s3_client_service import S3ClientService
from app.infrastructure.graph.llm_client import LLMClient
from app.infrastructure.graph.entity_extractor import EntityExtractor
from app.infrastructure.graph.neo4j_store import Neo4jStore
from app.infrastructure.graph.pgvector_store import GraphVectorStore
from app.infrastructure.graph.graph_merger import GraphMerger
from app.infrastructure.graph.graph_ingestor import GraphIngestor

logger = logging.getLogger(__name__)


class WorkerContainer:
    """Per-process dependency container.

    Stores one instance of each expensive resource.  Tasks construct cheap
    per-operation objects (repositories, services) that *reference* these
    resources; they do not own them.
    """

    def __init__(self) -> None:
        self._initialized = False
        self._session_factory: Optional[Any] = None
        self._s3: Optional[Any] = None          # S3ClientService instance
        self._parser: Optional[Any] = None
        self._splitter: Optional[Any] = None
        self._embedding_service: Optional[Any] = None
        self._neo4j_store: Optional[Any] = None
        self._graph_ingestor: Optional[Any] = None

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def init(self) -> None:
        """Initialize all resources.  Called once per worker process."""
        if self._initialized:
            return

        # NullPool: the engine object is a pure factory with no cached
        # connections.  It is therefore safe to share across asyncio.run()
        # calls because no connection is ever bound to a specific event loop.
        self._session_factory = async_sessionmaker(
            create_async_engine(settings.DATABASE_URL, poolclass=NullPool),
            expire_on_commit=False,
        )

        self._s3 = S3ClientService()

        self._parser = DocumentParser()
        self._splitter = MarkdownSplitter(
            tokenizer_model=settings.TIKTOKEN_MODEL_NAME,
            retrieve_max_tokens=settings.RETRIEVE_MAX_TOKENS,
            retrieve_target_tokens=settings.RETRIEVE_TARGET_TOKENS,
        )

        self._embedding_service = EmbeddingService()

        # Graph ingestion pipeline
        self._neo4j_store = Neo4jStore(
            uri=settings.NEO4J_URI,
            username=settings.NEO4J_USERNAME,
            password=settings.NEO4J_PASSWORD,
            database=settings.NEO4J_DATABASE,
        )
        llm_client = LLMClient(
            api_base=settings.GRAPHRAG_LLM_API_BASE,
            model=settings.GRAPHRAG_LLM_MODEL,
            api_key=settings.GRAPHRAG_LLM_API_KEY,
        )
        extractor = EntityExtractor(llm_client=llm_client, max_gleaning=1)
        vector_store = GraphVectorStore(self._session_factory, self._embedding_service)
        merger = GraphMerger(self._neo4j_store, vector_store, llm_client)
        self._graph_ingestor = GraphIngestor(extractor, merger)

        self._initialized = True
        logger.info("WorkerContainer initialized (pid=%s)", os.getpid())

    def _ensure(self) -> None:
        if not self._initialized:
            self.init()

    # ------------------------------------------------------------------
    # Properties (lazy access)
    # ------------------------------------------------------------------

    @property
    def session_factory(self) -> Any:
        """async_sessionmaker backed by a NullPool engine."""
        self._ensure()
        return self._session_factory

    @property
    def s3(self) -> Any:
        """Shared S3ClientService instance."""
        self._ensure()
        return self._s3

    @property
    def parser(self) -> Any:
        """Stateless DocumentParser instance."""
        self._ensure()
        return self._parser

    @property
    def splitter(self) -> Any:
        """MarkdownSplitter (parent-child) with tiktoken encoding loaded once."""
        self._ensure()
        return self._splitter

    @property
    def embedding_service(self) -> Any:
        """EmbeddingService config wrapper (no event-loop-bound state)."""
        self._ensure()
        return self._embedding_service

    @property
    def neo4j_store(self) -> Any:
        """Shared Neo4jStore instance."""
        self._ensure()
        return self._neo4j_store

    @property
    def graph_ingestor(self) -> Any:
        """GraphIngestor wired with LLM, Neo4j, and PGVector."""
        self._ensure()
        return self._graph_ingestor


# Module-level singleton accessed by all tasks.
container = WorkerContainer()
