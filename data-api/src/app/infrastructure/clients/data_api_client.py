"""Client for querying knowledge bases (self-referencing within data-api)."""

import logging
from typing import List, Dict, Any

import httpx

from app.configurations.configurations import settings
from app.exceptions import ExternalServiceError


logger = logging.getLogger(__name__)


class DataApiClient:
    """HTTP client for calling data-api query endpoints (self-call)."""

    def __init__(self, timeout: int = 60):
        self.base_url = settings.DATA_API_URL
        self.timeout = timeout

    async def query_knowledge_base(
        self,
        kb_id: str,
        query_text: str,
        top_k: int = 5,
        search_type: str = "semantic",
        mode: str = "local",
    ) -> Dict[str, Any]:
        """
        Query a knowledge base.

        Args:
            kb_id: Knowledge base ID
            query_text: Query text
            top_k: Number of results to return
            search_type: Search type (semantic, hybrid, fuzzy)
            mode: Mode for GraphRAG (local, global, hybrid, naive, mix)

        Returns:
            Query results with retrieved context/chunks
        """
        url = f"{self.base_url}/api/v1/query/{kb_id}"
        payload = {
            "query_text": query_text,
            "top_k": top_k,
            "search_type": search_type,
            "mode": mode,
        }

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(url, json=payload)
                response.raise_for_status()
                data = response.json()
                return data.get("data", {})
        except httpx.HTTPStatusError as e:
            logger.error(f"Data API query error: {e}")
            raise ExternalServiceError(
                "Data API",
                f"Query failed with status {e.response.status_code}",
                {"kb_id": kb_id, "error": str(e)}
            )
        except Exception as e:
            logger.error(f"Data API error: {e}")
            raise ExternalServiceError("Data API", str(e))

    async def batch_query_knowledge_bases(
        self,
        kb_ids: List[str],
        query_text: str,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Query multiple knowledge bases and aggregate results.

        Args:
            kb_ids: List of knowledge base IDs
            query_text: Query text
            top_k: Number of results per KB

        Returns:
            List of query results from all KBs
        """
        results = []
        for kb_id in kb_ids:
            try:
                result = await self.query_knowledge_base(
                    kb_id=kb_id,
                    query_text=query_text,
                    top_k=top_k,
                )
                results.append({
                    "kb_id": kb_id,
                    "success": True,
                    "data": result,
                })
            except Exception as e:
                logger.warning(f"Failed to query KB {kb_id}: {e}")
                results.append({
                    "kb_id": kb_id,
                    "success": False,
                    "error": str(e),
                })
        return results
