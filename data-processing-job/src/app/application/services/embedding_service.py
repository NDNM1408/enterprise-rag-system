import logging
import httpx
from typing import List
from app.configurations.configurations import settings

logger = logging.getLogger(__name__)


class EmbeddingService:
    """
    Wrapper for an OpenAI-compatible embedding API.

    Holds only configuration (URL, model name, key, batch size).  No
    event-loop-bound resources are stored as instance state, so one instance
    can be safely shared across asyncio.run() calls within the same process.

    httpx.AsyncClient is opened and closed within each request method so it
    is always bound to the event loop that is currently running.
    """

    def __init__(self):
        self.api_base = settings.EMBEDDING_API_BASE
        self.model_name = settings.EMBEDDING_MODEL_NAME
        self.api_key = settings.EMBEDDING_API_KEY
        self.batch_size = settings.EMBEDDING_BATCH_SIZE
        self.timeout = 60.0  # seconds

    async def get_embedding(self, text: str) -> List[float]:
        """Get embedding for a single text string."""
        embeddings = await self.get_embeddings_batch([text])
        return embeddings[0]

    async def get_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for a batch of texts in a single API call."""
        url = f"{self.api_base}/embeddings"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        payload = {"model": self.model_name, "input": texts}

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(url, json=payload, headers=headers)
                response.raise_for_status()
                data = response.json()
                embeddings = [item["embedding"] for item in data["data"]]
                logger.info("Generated %d embeddings", len(embeddings))
                return embeddings

        except httpx.HTTPStatusError as e:
            logger.error(
                "HTTP error getting embeddings: %s %s",
                e.response.status_code,
                e.response.text,
            )
            raise
        except Exception as e:
            logger.error("Failed to get embeddings: %s", e)
            raise

    async def get_embeddings_batch_chunked(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for a large list by splitting into smaller batches."""
        all_embeddings = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            all_embeddings.extend(await self.get_embeddings_batch(batch))
        logger.info(
            "Generated %d embeddings in %d batches",
            len(all_embeddings),
            (len(texts) + self.batch_size - 1) // self.batch_size,
        )
        return all_embeddings
