"""Elasticsearch index schema for llm-wiki KBs.

Mirror of ``data-api/src/app/infrastructure/search/es_index_schema.py`` — the
two services are deployed as separate containers, but the mapping must stay
in lockstep so the worker writes a shape the search service can query.
"""

from __future__ import annotations


EMBEDDING_DIMS = 3072


def kb_index_name(kb_id: str, prefix: str = "kb") -> str:
    """Derive the ES index name for a knowledge base."""
    return f"{prefix}-{kb_id}".lower()


def build_chunk_index_mapping() -> dict:
    return {
        "settings": {
            "number_of_shards": 1,
            "number_of_replicas": 0,
            "analysis": {
                "analyzer": {
                    "vn_text": {
                        "type": "custom",
                        "tokenizer": "standard",
                        "filter": ["lowercase", "asciifolding"],
                    }
                }
            },
        },
        "mappings": {
            "dynamic": "strict",
            "properties": {
                "chunk_id": {"type": "keyword"},
                "document_id": {"type": "keyword"},
                "kb_id": {"type": "keyword"},
                "doc_name": {"type": "keyword"},
                "section_label": {"type": "keyword"},
                "heading_path": {
                    "type": "text",
                    "analyzer": "vn_text",
                    "fields": {"kw": {"type": "keyword"}},
                },
                "ordinal": {"type": "integer"},
                "start_line": {"type": "integer"},
                "end_line": {"type": "integer"},
                "content": {
                    "type": "text",
                    "analyzer": "vn_text",
                },
                "embedding": {
                    "type": "dense_vector",
                    "dims": EMBEDDING_DIMS,
                    "similarity": "cosine",
                    "index": True,
                    "index_options": {
                        "type": "int8_hnsw",
                        "m": 16,
                        "ef_construction": 100,
                    },
                },
            },
        },
    }
