"""
Custom Document class to replace LlamaIndex Document.
Simple data structure for holding document chunks with metadata.
"""
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field


@dataclass
class Document:
    """
    Represents a document chunk with text and metadata.

    Attributes:
        text: The text content of the document chunk
        metadata: Dictionary of metadata associated with the chunk
        excluded_embed_metadata_keys: Keys to exclude when creating embeddings
        excluded_llm_metadata_keys: Keys to exclude when sending to LLM
    """
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    excluded_embed_metadata_keys: Optional[List[str]] = None
    excluded_llm_metadata_keys: Optional[List[str]] = None

    def get_content(self) -> str:
        """Get the text content of the document."""
        return self.text

    def copy(self) -> 'Document':
        """Create a copy of the document."""
        return Document(
            text=self.text,
            metadata=self.metadata.copy(),
            excluded_embed_metadata_keys=self.excluded_embed_metadata_keys.copy() if self.excluded_embed_metadata_keys else None,
            excluded_llm_metadata_keys=self.excluded_llm_metadata_keys.copy() if self.excluded_llm_metadata_keys else None
        )
