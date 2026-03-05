from typing import Any, Dict, Optional, List
from pydantic import BaseModel, Field

class CreateDocumentsRequest(BaseModel):
    basePath: str = Field(..., description="Base path for documents")
    cmetadata: Dict[str, Any] = Field(..., description="Custom metadata")
    
    # Để hỗ trợ dynamic fields như [key: string]: string; bạn có thể override __getitem__ nếu cần.

class DeleteDocumentsRequest(BaseModel):
    ids: List[str] = Field(..., description="List of document IDs to delete")

class UpdateDocumentsRequest(BaseModel):
    id: str = Field(..., description="Document ID")
    isActive: bool = Field(..., description="Document is active")
