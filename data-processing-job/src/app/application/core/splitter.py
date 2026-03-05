import logging
from typing import Any, Dict, List

import tiktoken

logger = logging.getLogger(__name__)


class DocumentSplitter:
    """Splits plain text into overlapping token-sized chunks.

    Chunking logic mirrors LightRAG's ``chunking_by_token_size``:
    - tokens are counted with tiktoken (no HuggingFace model weights required)
    - consecutive chunks share an overlap of ``chunk_overlap`` tokens
    - uses a sliding window over the token list for exact chunk boundaries
    """

    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
        chunk_size: int = 1200,
        chunk_overlap: int = 100,
    ):
        """
        Args:
            model_name: tiktoken model name used to count tokens.
            chunk_size: Maximum number of tokens per chunk.
            chunk_overlap: Number of overlapping tokens between consecutive chunks.
        """
        self.encoding = tiktoken.encoding_for_model(model_name)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split(self, text: str) -> List[Dict[str, Any]]:
        """Split text into chunks using a sliding window over the token list.

        Returns:
            List of dicts, each with:
            - ``content`` (str): chunk text
            - ``tokens`` (int): token count for the chunk
            - ``chunk_order_index`` (int): zero-based position in the document
        """
        tokens = self.encoding.encode(text)
        results = []
        stride = self.chunk_size - self.chunk_overlap

        for index, start in enumerate(range(0, len(tokens), stride)):
            chunk_tokens = tokens[start : start + self.chunk_size]
            chunk_content = self.encoding.decode(chunk_tokens)
            results.append(
                {
                    "tokens": len(chunk_tokens),
                    "content": chunk_content.strip(),
                    "chunk_order_index": index,
                }
            )

        return results
