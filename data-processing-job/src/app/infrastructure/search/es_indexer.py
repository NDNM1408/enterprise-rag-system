"""Sync Elasticsearch indexer used by the llm-wiki Celery task.

Sync (not async) because the Celery worker runs in prefork mode — async
clients with shared event loops hit "loop closed" errors after task
completion. The official ``elasticsearch`` package's sync client is fine
for the per-document indexing volumes we deal with here.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk

from app.configurations.configurations import settings
from app.infrastructure.search.es_index_schema import (
    build_chunk_index_mapping,
    kb_index_name,
)

logger = logging.getLogger(__name__)


class ElasticsearchIndexer:
    """Sync ES client wrapper for per-KB index lifecycle + bulk indexing."""

    def __init__(self):
        self._client = Elasticsearch(
            hosts=[settings.ELASTICSEARCH_URL],
            basic_auth=(
                (settings.ELASTICSEARCH_USERNAME, settings.ELASTICSEARCH_PASSWORD)
                if settings.ELASTICSEARCH_USERNAME
                else None
            ),
            request_timeout=60,
        )
        self._prefix = settings.ELASTICSEARCH_INDEX_PREFIX

    def ensure_index(self, kb_id: str) -> str:
        """Create the per-KB index if it doesn't already exist. Returns name."""
        index = kb_index_name(kb_id, self._prefix)
        if not self._client.indices.exists(index=index):
            mapping = build_chunk_index_mapping()
            self._client.indices.create(index=index, body=mapping)
            logger.info("Created ES index %s", index)
        return index

    def delete_document_chunks(self, kb_id: str, document_id: str) -> int:
        """Wipe existing chunks for a document — used to make re-ingest idempotent."""
        index = kb_index_name(kb_id, self._prefix)
        if not self._client.indices.exists(index=index):
            return 0
        resp = self._client.delete_by_query(
            index=index,
            body={"query": {"term": {"document_id": document_id}}},
            refresh=True,
        )
        return int(resp.get("deleted", 0))

    def bulk_index(
        self,
        kb_id: str,
        document_id: str,
        doc_name: str,
        chunks: List[Dict[str, Any]],
    ) -> int:
        """Bulk-index chunks. Each ``chunks[i]`` must already carry
        ``chunk_id``, ``content``, ``embedding`` plus optional structural
        fields (``section_label``, ``heading_path``, ``ordinal``,
        ``start_line``, ``end_line``)."""
        if not chunks:
            return 0

        index = self.ensure_index(kb_id)

        def _gen():
            for ch in chunks:
                yield {
                    "_op_type": "index",
                    "_index": index,
                    "_id": ch["chunk_id"],
                    "_source": {
                        "chunk_id": ch["chunk_id"],
                        "document_id": document_id,
                        "kb_id": kb_id,
                        "doc_name": doc_name,
                        "section_label": ch.get("section_label"),
                        "heading_path": ch.get("heading_path", ""),
                        "ordinal": ch.get("ordinal", 0),
                        "start_line": ch.get("start_line"),
                        "end_line": ch.get("end_line"),
                        "content": ch["content"],
                        "embedding": ch["embedding"],
                    },
                }

        success, errors = bulk(self._client, _gen(), refresh="wait_for", raise_on_error=False)
        if errors:
            logger.warning(
                "ES bulk index for doc=%s had %d errors (succeeded=%d)",
                document_id, len(errors), success,
            )
        return success

    def close(self) -> None:
        self._client.close()
