from typing import Optional, List, Dict, Any, Union
from pydantic import BaseModel, Field

class BaseQueryRequest(BaseModel):
    page: int = Field(1, description="Page number")
    pageSize: int = Field(10, description="Items per page")
    filter: Optional[Dict[str, Any]] = None
    sort: Optional[Dict[str, str]] = None  # e.g., {"created_at": "desc"}