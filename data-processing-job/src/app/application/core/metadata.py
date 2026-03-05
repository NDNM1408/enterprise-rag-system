from pydantic import BaseModel, Field
from typing import Optional, Any, Literal


class ChunkLlamaIndexMetadata(BaseModel):

    data_source_id: str = Field(description="Data source ID")
    source_doc_name: str = Field(description="Source document name")
    source_doc_id: str = Field(description="Source document ID")


class ExpandChunkLlamaIndexMetadata(ChunkLlamaIndexMetadata):
    """
    Metadata for retrievel purpose in LlamaIndex.
    """
    # Content type information
    type: Literal["text", "table"] = Field(description="Type of content: text or table")
    sub_type: Optional[str] = Field(default=None, description="Sub-type of content (para, titles, etc.)")
    
    # Table-specific fields
    is_table: Optional[bool] = Field(default=False, description="Whether the content is or contains a table")
    table_description: Optional[str] = Field(default=None, description="LLM-generated description of the table")
    table_dataframe: Optional[str] = Field(default=None, description="Base64 encoded serialized dataframe")
    
    # Text-specific fields
    text_here: Optional[str] = Field(default=None, description="The actual text content")