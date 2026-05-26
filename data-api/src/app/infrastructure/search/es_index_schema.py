"""Elasticsearch index schema for llm-wiki KBs.

Per-KB index ``kb-<kb_id>`` with chunked legal articles + Gemini embeddings.
"""

from __future__ import annotations


# Same dim as the pgvector ``chunk.embedding`` column (Gemini ``embedding-001``
# at the project's default output). Keep the two stores in lockstep — query
# embeddings are reused across both paths.
EMBEDDING_DIMS = 3072


def kb_index_name(kb_id: str, prefix: str = "kb") -> str:
    """Derive the ES index name for a knowledge base."""
    # ES disallows uppercase; KB ids are lowercase UUIDs anyway, but normalise
    # defensively in case the prefix or KB id ever drifts.
    return f"{prefix}-{kb_id}".lower()


def build_chunk_index_mapping() -> dict:
    """Mapping for the per-KB chunk index.

    Fields:
      - chunk_id, document_id, kb_id, doc_name: identifiers (keyword).
      - section_label: rendered legal article header (e.g. ``Điều 27``).
      - heading_path: full ``>``-joined heading stack (text + keyword sub-field).
      - ordinal: chunk order within the document.
      - start_line / end_line: provenance into the original markdown.
      - content: chunk body (text + wildcard for substring queries).
      - embedding: Gemini vector for kNN search.
    """
    return {
        "settings": {
            "number_of_shards": 1,
            "number_of_replicas": 0,
            "analysis": {
                "analyzer": {
                    # ``standard`` works reasonably for Vietnamese (Latin script).
                    # ``asciifolding`` lets queries hit accent-stripped tokens.
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
