"""
Neo4j graph store for graph queries and deletions.

Reads from / deletes from the exact same schema that LightRAG's Neo4JStorage /
the ingestion pipeline writes:
- Node label = kb_id (workspace), property ``entity_id``
- Edge type = ``DIRECTED`` (undirected semantics via ``()-[r]-()`` match)
"""

from __future__ import annotations

import logging
from typing import Any

from neo4j import AsyncGraphDatabase

logger = logging.getLogger(__name__)


class Neo4jStore:
    """Thin async Neo4j client for read-only graph queries."""

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
    # Batch node operations
    # ------------------------------------------------------------------

    async def get_nodes_batch(
        self, kb_id: str, node_ids: list[str]
    ) -> dict[str, dict]:
        """Retrieve multiple nodes in one query using UNWIND."""
        if not node_ids:
            return {}

        async with self._driver.session(
            database=self._database, default_access_mode="READ"
        ) as session:
            query = f"""
            UNWIND $node_ids AS id
            MATCH (n:`{kb_id}` {{entity_id: id}})
            RETURN n.entity_id AS entity_id, n
            """
            result = await session.run(query, node_ids=node_ids)
            nodes: dict[str, dict] = {}
            async for record in result:
                entity_id = record["entity_id"]
                node_dict = dict(record["n"])
                if "labels" in node_dict:
                    node_dict["labels"] = [
                        label for label in node_dict["labels"] if label != kb_id
                    ]
                nodes[entity_id] = node_dict
            await result.consume()
            return nodes

    async def node_degrees_batch(
        self, kb_id: str, node_ids: list[str]
    ) -> dict[str, int]:
        """Retrieve the degree for multiple nodes in a single query."""
        if not node_ids:
            return {}

        async with self._driver.session(
            database=self._database, default_access_mode="READ"
        ) as session:
            query = f"""
            UNWIND $node_ids AS id
            MATCH (n:`{kb_id}` {{entity_id: id}})
            RETURN n.entity_id AS entity_id, count {{ (n)--() }} AS degree;
            """
            result = await session.run(query, node_ids=node_ids)
            degrees: dict[str, int] = {}
            async for record in result:
                degrees[record["entity_id"]] = record["degree"]
            await result.consume()

            for nid in node_ids:
                if nid not in degrees:
                    degrees[nid] = 0

            return degrees

    # ------------------------------------------------------------------
    # Batch edge operations
    # ------------------------------------------------------------------

    async def get_edges_batch(
        self, kb_id: str, pairs: list[dict[str, str]]
    ) -> dict[tuple[str, str], dict]:
        """Retrieve edge properties for multiple (src, tgt) pairs in one query."""
        if not pairs:
            return {}

        async with self._driver.session(
            database=self._database, default_access_mode="READ"
        ) as session:
            query = f"""
            UNWIND $pairs AS pair
            MATCH (start:`{kb_id}` {{entity_id: pair.src}})-[r:DIRECTED]-(end:`{kb_id}` {{entity_id: pair.tgt}})
            RETURN pair.src AS src_id, pair.tgt AS tgt_id, collect(properties(r)) AS edges
            """
            result = await session.run(query, pairs=pairs)
            edges_dict: dict[tuple[str, str], dict] = {}
            async for record in result:
                src = record["src_id"]
                tgt = record["tgt_id"]
                edges = record["edges"]
                if edges and len(edges) > 0:
                    edge_props = edges[0]
                    for key, default in {
                        "weight": 1.0,
                        "source_id": None,
                        "description": None,
                        "keywords": None,
                    }.items():
                        if key not in edge_props:
                            edge_props[key] = default
                    edges_dict[(src, tgt)] = edge_props
                else:
                    edges_dict[(src, tgt)] = {
                        "weight": 1.0,
                        "source_id": None,
                        "description": None,
                        "keywords": None,
                    }
            await result.consume()
            return edges_dict

    async def get_nodes_edges_batch(
        self, kb_id: str, node_ids: list[str]
    ) -> dict[str, list[tuple[str, str]]]:
        """Batch retrieve edges for multiple nodes in one query."""
        if not node_ids:
            return {}

        async with self._driver.session(
            database=self._database, default_access_mode="READ"
        ) as session:
            query = f"""
            UNWIND $node_ids AS id
            MATCH (n:`{kb_id}` {{entity_id: id}})
            OPTIONAL MATCH (n)-[r]-(connected:`{kb_id}`)
            RETURN id AS queried_id, n.entity_id AS node_entity_id,
                   connected.entity_id AS connected_entity_id,
                   startNode(r).entity_id AS start_entity_id
            """
            result = await session.run(query, node_ids=node_ids)

            edges_dict: dict[str, list[tuple[str, str]]] = {
                nid: [] for nid in node_ids
            }

            async for record in result:
                queried_id = record["queried_id"]
                node_entity_id = record["node_entity_id"]
                connected_entity_id = record["connected_entity_id"]
                start_entity_id = record["start_entity_id"]

                if not node_entity_id or not connected_entity_id:
                    continue

                if start_entity_id == node_entity_id:
                    edges_dict[queried_id].append(
                        (node_entity_id, connected_entity_id)
                    )
                else:
                    edges_dict[queried_id].append(
                        (connected_entity_id, node_entity_id)
                    )

            await result.consume()
            return edges_dict

    # ------------------------------------------------------------------
    # Delete operations
    # ------------------------------------------------------------------

    async def delete_nodes_batch(self, kb_id: str, entity_names: list[str]) -> None:
        """Delete nodes (and all their edges) by entity_id using DETACH DELETE."""
        if not entity_names:
            return

        async with self._driver.session(database=self._database) as session:
            query = f"""
            UNWIND $entity_names AS name
            MATCH (n:`{kb_id}` {{entity_id: name}})
            DETACH DELETE n
            """

            async def _execute(tx):
                result = await tx.run(query, entity_names=entity_names)
                await result.consume()

            await session.execute_write(_execute)

    async def delete_edges_batch(
        self, kb_id: str, pairs: list[tuple[str, str]]
    ) -> None:
        """Delete edges between specific entity pairs (both directions)."""
        if not pairs:
            return

        pair_dicts = [{"src": src, "tgt": tgt} for src, tgt in pairs]

        async with self._driver.session(database=self._database) as session:
            query = f"""
            UNWIND $pairs AS pair
            MATCH (a:`{kb_id}` {{entity_id: pair.src}})-[r:DIRECTED]-(b:`{kb_id}` {{entity_id: pair.tgt}})
            DELETE r
            """

            async def _execute(tx):
                result = await tx.run(query, pairs=pair_dicts)
                await result.consume()

            await session.execute_write(_execute)

    async def edge_degrees_batch(
        self, kb_id: str, edge_pairs: list[tuple[str, str]]
    ) -> dict[tuple[str, str], int]:
        """Calculate combined degree for each edge (sum of src + tgt degrees)."""
        if not edge_pairs:
            return {}

        unique_node_ids = set()
        for src, tgt in edge_pairs:
            unique_node_ids.add(src)
            unique_node_ids.add(tgt)

        degrees = await self.node_degrees_batch(kb_id, list(unique_node_ids))

        edge_degrees: dict[tuple[str, str], int] = {}
        for src, tgt in edge_pairs:
            edge_degrees[(src, tgt)] = degrees.get(src, 0) + degrees.get(tgt, 0)
        return edge_degrees
