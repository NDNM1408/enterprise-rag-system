"""
Document embeddings repository for QUERY operations only.
Queries the merged chunk table for vector similarity and hybrid search.

NOTE: Upsert/write operations belong in the worker service.
"""

import logging
from typing import List, Dict, Any, Optional
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text


logger = logging.getLogger(__name__)


class DocumentEmbeddingsRepository:
    """
    Repository for querying embeddings from the chunk table using pgvector.

    Provides:
    - Vector similarity search
    - Hybrid search (vector + full-text)
    - Metadata filtering

    Does NOT provide:
    - Upsert operations (those are in worker service)
    - Delete operations (those are in worker service)
    """

    def __init__(self, async_session: AsyncSession):
        self.session = async_session

    async def query_by_vector(
        self,
        kb_id: str,
        query_embedding: List[float],
        top_k: int = 10,
        metadata_filter: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Vector similarity search using cosine distance.

        Args:
            kb_id: ID of knowledge base to search within
            query_embedding: Query vector (1024-dim)
            top_k: Number of results to return
            metadata_filter: Optional JSONB filter (e.g., {"type": "text"})

        Returns:
            List of dicts with chunk_id, document_id, text, metadata, similarity
        """
        filter_clause = ""
        if metadata_filter:
            conditions = [
                f"metadata->>'{k}' = :{k}"
                for k in metadata_filter.keys()
            ]
            filter_clause = "AND " + " AND ".join(conditions)

        query = text(f"""
            SELECT
                id AS chunk_id,
                document_id,
                doc_name,
                heading_path,
                parent_id,
                content AS text,
                parent_text,
                chunk_type,
                table_id,
                table_dataframe,
                metadata,
                1 - (embedding <=> CAST(:query_embedding AS vector)) AS similarity
            FROM "chunk"
            WHERE kb_id = :kb_id
              AND status = 'Succeed'
              AND embedding IS NOT NULL
            {filter_clause}
            ORDER BY embedding <=> CAST(:query_embedding AS vector)
            LIMIT :top_k
        """)

        embedding_str = '[' + ','.join(str(x) for x in query_embedding) + ']'

        params = {
            'kb_id': kb_id,
            'query_embedding': embedding_str,
            'top_k': top_k
        }

        if metadata_filter:
            params.update(metadata_filter)

        try:
            result = await self.session.execute(query, params)
            rows = result.fetchall()

            results = [
                {
                    'chunk_id': str(row.chunk_id),
                    'document_id': str(row.document_id),
                    'doc_name': row.doc_name,
                    'heading_path': row.heading_path,
                    'parent_id': row.parent_id,
                    'text': row.text,
                    'parent_text': row.parent_text,
                    'chunk_type': row.chunk_type,
                    'table_id': row.table_id,
                    'table_dataframe': row.table_dataframe,
                    'metadata': row.metadata,
                    'similarity': float(row.similarity)
                }
                for row in rows
            ]

            logger.info(f"Retrieved {len(results)} results for similarity search in kb {kb_id}")
            return results
        except Exception as e:
            logger.error(f"Failed to query embeddings: {e}")
            raise

    async def hybrid_search(
        self,
        kb_id: str,
        query_embedding: List[float],
        query_text: str,
        top_k: int = 10,
        alpha: float = 0.5
    ) -> List[Dict[str, Any]]:
        """
        Hybrid search combining vector similarity and full-text search.

        Args:
            kb_id: ID of knowledge base
            query_embedding: Query vector
            query_text: Query text for full-text search
            top_k: Number of results
            alpha: Weight for vector similarity (1-alpha for text rank)

        Returns:
            List of results with combined scores
        """
        embedding_str = '[' + ','.join(str(x) for x in query_embedding) + ']'

        query = text("""
            WITH vector_results AS (
                SELECT
                    id AS chunk_id,
                    document_id,
                    doc_name,
                    heading_path,
                    parent_id,
                    content AS text,
                    parent_text,
                    metadata,
                    1 - (embedding <=> CAST(:query_embedding AS vector)) AS vector_similarity
                FROM "chunk"
                WHERE kb_id = :kb_id
                  AND status = 'Succeed'
                  AND embedding IS NOT NULL
                ORDER BY embedding <=> CAST(:query_embedding AS vector)
                LIMIT :top_k
            ),
            text_results AS (
                SELECT
                    id AS chunk_id,
                    document_id,
                    doc_name,
                    heading_path,
                    parent_id,
                    content AS text,
                    parent_text,
                    metadata,
                    ts_rank(text_tsvector, to_tsquery('english', :query_text)) AS text_rank
                FROM "chunk"
                WHERE kb_id = :kb_id
                  AND status = 'Succeed'
                  AND text_tsvector @@ to_tsquery('english', :query_text)
                ORDER BY text_rank DESC
                LIMIT :top_k
            )
            SELECT DISTINCT
                COALESCE(v.chunk_id, t.chunk_id) AS chunk_id,
                COALESCE(v.document_id, t.document_id) AS document_id,
                COALESCE(v.doc_name, t.doc_name) AS doc_name,
                COALESCE(v.heading_path, t.heading_path) AS heading_path,
                COALESCE(v.parent_id, t.parent_id) AS parent_id,
                COALESCE(v.text, t.text) AS text,
                COALESCE(v.parent_text, t.parent_text) AS parent_text,
                COALESCE(v.metadata, t.metadata) AS metadata,
                COALESCE(v.vector_similarity, 0) * :alpha + COALESCE(t.text_rank, 0) * :beta AS combined_score
            FROM vector_results v
            FULL OUTER JOIN text_results t ON v.chunk_id = t.chunk_id
            ORDER BY combined_score DESC
            LIMIT :top_k
        """)

        params = {
            'kb_id': kb_id,
            'query_embedding': embedding_str,
            'query_text': query_text.replace(' ', '|'),
            'top_k': top_k,
            'alpha': alpha,
            'beta': 1 - alpha
        }

        try:
            result = await self.session.execute(query, params)
            rows = result.fetchall()

            results = [
                {
                    'chunk_id': str(row.chunk_id),
                    'document_id': str(row.document_id),
                    'doc_name': row.doc_name,
                    'heading_path': row.heading_path,
                    'parent_id': row.parent_id,
                    'text': row.text,
                    'parent_text': row.parent_text,
                    'metadata': row.metadata,
                    'similarity': float(row.combined_score),
                }
                for row in rows
            ]

            logger.info(f"Retrieved {len(results)} hybrid search results for kb {kb_id}")
            return results
        except Exception as e:
            logger.error(f"Failed to perform hybrid search: {e}")
            raise

    async def fuzzy_search(
        self,
        kb_id: str,
        query_text: str,
        top_k: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Fuzzy/full-text search using PostgreSQL trigram similarity.

        Uses pg_trgm extension for fuzzy matching. Good for handling
        typos and partial matches.

        Args:
            kb_id: ID of knowledge base
            query_text: Query text for fuzzy matching
            top_k: Number of results

        Returns:
            List of results with similarity scores
        """
        query = text("""
            SELECT
                id AS chunk_id,
                document_id,
                doc_name,
                heading_path,
                parent_id,
                content AS text,
                parent_text,
                metadata,
                similarity(content, :query_text) AS similarity
            FROM "chunk"
            WHERE kb_id = :kb_id
              AND status = 'Succeed'
              AND content % :query_text
            ORDER BY similarity DESC
            LIMIT :top_k
        """)

        params = {
            'kb_id': kb_id,
            'query_text': query_text,
            'top_k': top_k
        }

        try:
            result = await self.session.execute(query, params)
            rows = result.fetchall()

            results = [
                {
                    'chunk_id': str(row.chunk_id),
                    'document_id': str(row.document_id),
                    'doc_name': row.doc_name,
                    'heading_path': row.heading_path,
                    'parent_id': row.parent_id,
                    'text': row.text,
                    'parent_text': row.parent_text,
                    'metadata': row.metadata,
                    'similarity': float(row.similarity),
                }
                for row in rows
            ]

            logger.info(f"Retrieved {len(results)} fuzzy search results for kb {kb_id}")
            return results
        except Exception as e:
            logger.error(f"Failed to perform fuzzy search: {e}")
            raise
