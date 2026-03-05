"""
Neo4j graph CRUD for ingestion.

Writes to the exact same schema that LightRAG's Neo4JStorage uses:
- Node label = kb_id (workspace), property ``entity_id``
- Edge type = ``DIRECTED`` (undirected semantics via ``()-[r]-()`` match)
"""

from __future__ import annotations

import logging
from typing import Any

from neo4j import AsyncGraphDatabase

logger = logging.getLogger(__name__)


class Neo4jStore:
    """Thin async Neo4j client for graph ingestion CRUD."""

    def __init__(
        self,
        uri: str,
        username: str,
        password: str,
        database: str = "neo4j",
    ) -> None:
        self._uri = uri
        self._username = username
        self._password = password
        self._database = database
        self._driver = AsyncGraphDatabase.driver(
            uri, auth=(username, password)
        )

    async def close(self) -> None:
        await self._driver.close()

    # ------------------------------------------------------------------
    # Node operations
    # ------------------------------------------------------------------

    async def get_node(self, kb_id: str, entity_name: str) -> dict[str, Any] | None:
        """Get node properties by entity_id within the workspace label."""
        async with self._driver.session(
            database=self._database, default_access_mode="READ"
        ) as session:
            query = f"MATCH (n:`{kb_id}` {{entity_id: $entity_id}}) RETURN n"
            result = await session.run(query, entity_id=entity_name)
            try:
                records = await result.fetch(2)
                if records:
                    return dict(records[0]["n"])
                return None
            finally:
                await result.consume()

    async def upsert_node(self, kb_id: str, node_id: str, node_data: dict[str, Any]) -> None:
        """Upsert a node: MERGE on entity_id, SET all properties."""
        entity_type = node_data.get("entity_type", "UNKNOWN")

        async with self._driver.session(database=self._database) as session:

            async def _execute(tx):
                query = f"""
                MERGE (n:`{kb_id}` {{entity_id: $entity_id}})
                SET n += $properties
                SET n:`{entity_type}`
                """
                result = await tx.run(query, entity_id=node_id, properties=node_data)
                await result.consume()

            await session.execute_write(_execute)

    # ------------------------------------------------------------------
    # Edge operations
    # ------------------------------------------------------------------

    async def get_edge(self, kb_id: str, src: str, tgt: str) -> dict[str, Any] | None:
        """Get edge properties between two nodes (undirected match)."""
        async with self._driver.session(
            database=self._database, default_access_mode="READ"
        ) as session:
            query = f"""
            MATCH (a:`{kb_id}` {{entity_id: $src}})-[r]-(b:`{kb_id}` {{entity_id: $tgt}})
            RETURN properties(r) as edge_properties
            """
            result = await session.run(query, src=src, tgt=tgt)
            try:
                records = await result.fetch(2)
                if records:
                    edge = dict(records[0]["edge_properties"])
                    # Ensure required keys with defaults
                    for key, default in [("weight", 1.0), ("source_id", None), ("description", None), ("keywords", None)]:
                        if key not in edge:
                            edge[key] = default
                    return edge
                return None
            finally:
                await result.consume()

    async def upsert_edge(
        self, kb_id: str, src: str, tgt: str, edge_data: dict[str, Any]
    ) -> None:
        """Upsert an edge between two nodes using DIRECTED type (undirected semantics)."""
        async with self._driver.session(database=self._database) as session:

            async def _execute(tx):
                query = f"""
                MATCH (source:`{kb_id}` {{entity_id: $source_entity_id}})
                WITH source
                MATCH (target:`{kb_id}` {{entity_id: $target_entity_id}})
                MERGE (source)-[r:DIRECTED]-(target)
                SET r += $properties
                RETURN r
                """
                result = await tx.run(
                    query,
                    source_entity_id=src,
                    target_entity_id=tgt,
                    properties=edge_data,
                )
                try:
                    await result.fetch(2)
                finally:
                    await result.consume()

            await session.execute_write(_execute)
