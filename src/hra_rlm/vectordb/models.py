"""Pydantic models for vector database entities."""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class Chunk(BaseModel):
    """A text chunk with its embedding and metadata."""

    chunk_id: str
    text: str
    embedding: List[float]
    metadata: Dict[str, Any] = Field(default_factory=dict)


class SearchResult(BaseModel):
    """Result of a vector search query."""

    chunk_id: str
    score: float  # Cosine similarity score (0 to 1)
    text: Optional[str] = None  # Optional, can be filled later if needed