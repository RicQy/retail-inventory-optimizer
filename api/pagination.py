"""
Pagination utilities for FastAPI endpoints.

This module provides pagination functionality for API responses.
"""

from math import ceil
from typing import Generic, List, Optional, TypeVar

from fastapi import Query
from pydantic import BaseModel, Field

T = TypeVar("T")


class PaginationParams(BaseModel):
    """Pagination parameters for API requests."""

    page: int = Field(default=1, ge=1, description="Page number (starting from 1)")
    page_size: int = Field(
        default=20, ge=1, le=100, description="Number of items per page"
    )

    @property
    def offset(self) -> int:
        """Calculate offset for database queries."""
        return (self.page - 1) * self.page_size


class PaginatedResponse(BaseModel, Generic[T]):
    """Generic paginated response model."""

    items: List[T]
    total: int
    page: int
    page_size: int
    total_pages: int
    has_next: bool
    has_previous: bool

    @classmethod
    def create(
        cls, items: List[T], total: int, page: int, page_size: int
    ) -> "PaginatedResponse[T]":
        """Create a paginated response from items and pagination info."""
        total_pages = ceil(total / page_size) if total > 0 else 0

        return cls(
            items=items,
            total=total,
            page=page,
            page_size=page_size,
            total_pages=total_pages,
            has_next=page < total_pages,
            has_previous=page > 1,
        )


def pagination_params(
    page: int = Query(default=1, ge=1, description="Page number"),
    page_size: int = Query(default=20, ge=1, le=100, description="Items per page"),
) -> PaginationParams:
    """FastAPI dependency for pagination parameters."""
    return PaginationParams(page=page, page_size=page_size)


def paginate_list(items: List[T], pagination: PaginationParams) -> PaginatedResponse[T]:
    """Paginate a list of items."""
    total = len(items)
    start_index = pagination.offset
    end_index = start_index + pagination.page_size

    paginated_items = items[start_index:end_index]

    return PaginatedResponse.create(
        items=paginated_items,
        total=total,
        page=pagination.page,
        page_size=pagination.page_size,
    )


class CursorPaginationParams(BaseModel):
    """Cursor-based pagination parameters."""

    cursor: Optional[str] = Field(default=None, description="Cursor for pagination")
    limit: int = Field(
        default=20, ge=1, le=100, description="Number of items to return"
    )


class CursorPaginatedResponse(BaseModel, Generic[T]):
    """Cursor-based paginated response model."""

    items: List[T]
    next_cursor: Optional[str] = None
    has_more: bool

    @classmethod
    def create(
        cls, items: List[T], next_cursor: Optional[str] = None, has_more: bool = False
    ) -> "CursorPaginatedResponse[T]":
        """Create a cursor-based paginated response."""
        return cls(items=items, next_cursor=next_cursor, has_more=has_more)


def cursor_pagination_params(
    cursor: Optional[str] = Query(default=None, description="Cursor for pagination"),
    limit: int = Query(default=20, ge=1, le=100, description="Number of items"),
) -> CursorPaginationParams:
    """FastAPI dependency for cursor-based pagination parameters."""
    return CursorPaginationParams(cursor=cursor, limit=limit)
