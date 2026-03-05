from typing import TypeVar, Generic, List, Optional
from pydantic import BaseModel
from dataclasses import dataclass

T = TypeVar('T')

class PaginationResponse(BaseModel):
    total: int
    page: int
    limit: int
    total_pages: Optional[int] = None

@dataclass
class PagedResponse(Generic[T]):
    data: List[T]
    pagination: PaginationResponse

class PageUtils:
    @staticmethod
    def create_paged_response(
        content: List[T],
        total: int,
        page: int,
        limit: int,
        include_total_pages: bool = False
    ) -> dict:
        """
        Create a standardized paged response.

        Args:
            content (List[T]): List of items to be included in response
            total (int): Total number of items
            page (int): Current page number
            limit (int): Number of items per page
            include_total_pages (bool): Whether to include total pages in response

        Returns:
            dict: Standardized response with data and pagination
        """
        pagination = {
            "total": total,
            "page": page,
            "limit": limit
        }

        if include_total_pages:
            pagination["total_pages"] = (total + limit - 1) // limit if limit > 0 else 0

        return {
            "data": [item.to_json() if hasattr(item, 'to_json') else item for item in content],
            "pagination": pagination
        }

    @staticmethod
    def create_empty_response(limit: int = 10) -> dict:
        """
        Create an empty paged response.

        Args:
            limit (int): Number of items per page

        Returns:
            dict: Empty standardized response
        """
        return {
            "data": [],
            "pagination": {
                "total": 0,
                "page": 1,
                "limit": limit
            }
        }