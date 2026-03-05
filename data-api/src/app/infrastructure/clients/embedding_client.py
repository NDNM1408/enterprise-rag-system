import logging
import httpx
from typing import List
from app.configurations.configurations import settings

logger = logging.getLogger(__name__)


class EmbeddingClient:
    """
    HTTP client for the OpenAI-compatible embedding API.
    Used by QueryService to convert query text to a vector before searching.
    """

    def __init__(self):
        self.api_base = settings.EMBEDDING_API_BASE
        self.model_name = settings.EMBEDDING_MODEL_NAME
        self.api_key = settings.EMBEDDING_API_KEY
        self.timeout = 60.0

    async def get_embedding(self, text: str) -> List[float]:
        """
        Get embedding vector for a single text string.

        Args:
            text: Query text

        Returns:
            Embedding vector as list of floats

        Raises:
            httpx.HTTPStatusError: If the embedding API returns an error
        """
        url = f"{self.api_base}/embeddings"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        payload = {"model": self.model_name, "input": [text]}

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(url, json=payload, headers=headers)
            response.raise_for_status()
            data = response.json()
            embedding: List[float] = data["data"][0]["embedding"]
            logger.debug(f"Got embedding of dimension {len(embedding)}")
            return embedding
