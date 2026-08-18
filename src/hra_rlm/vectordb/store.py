"""In-memory vector store using cosine similarity.

Why this module exists:
Provides a simple, testable vector store for the HRA-RLM prototype.
Uses pure NumPy for cosine similarity, making it dependency-light and
easy to replace with FAISS/HNSW later.
"""

import logging
from typing import List, Optional

import numpy as np

from src.hra_rlm.vectordb.models import Chunk, SearchResult

logger = logging.getLogger(__name__)


class InMemoryVectorStore:
    """Simple in-memory vector store with cosine similarity search."""

    def __init__(self) -> None:
        self._chunks: List[Chunk] = []
        self._embeddings: Optional[np.ndarray] = None

    @property
    def size(self) -> int:
        """Return the number of chunks in the store."""
        return len(self._chunks)

    def add(self, chunk: Chunk) -> None:
        """Add a single chunk to the store."""
        self._chunks.append(chunk)
        # Invalidate cached embedding matrix so it rebuilds on next search
        self._embeddings = None
        logger.debug(f"Added chunk {chunk.chunk_id} (store size now: {self.size})")

    def add_batch(self, chunks: List[Chunk]) -> None:
        """Add multiple chunks at once."""
        for chunk in chunks:
            self.add(chunk)

    def _build_embedding_matrix(self) -> np.ndarray:
        """Convert stored embeddings into a 2D numpy array for fast dot product."""
        if self._embeddings is None and self.size > 0:
            self._embeddings = np.array([c.embedding for c in self._chunks], dtype=np.float32)
        return self._embeddings

    def search(self, query_embedding: List[float], top_k: int = 3) -> List[SearchResult]:
        """Find top_k most similar chunks using cosine similarity.

        Args:
            query_embedding: The embedding vector of the query.
            top_k: Number of results to return.

        Returns:
            List of SearchResult objects, sorted by score descending.
        """
        if self.size == 0:
            return []

        emb_matrix = self._build_embedding_matrix()
        if emb_matrix is None:
            return []

        # Normalize query for cosine similarity
        query_vec = np.array(query_embedding, dtype=np.float32)
        query_norm = query_vec / np.linalg.norm(query_vec)

        # Normalize each stored vector (if not already normalized, we do it per row)
        norms = np.linalg.norm(emb_matrix, axis=1, keepdims=True)
        # Avoid division by zero
        norms = np.where(norms == 0, 1e-8, norms)
        normalized_embs = emb_matrix / norms

        # Dot product -> cosine similarity
        scores = np.dot(normalized_embs, query_norm)

        # Get top_k indices
        top_indices = np.argsort(scores)[-top_k:][::-1]

        results = []
        for idx in top_indices:
            if scores[idx] > 1e-6:  # ignore near-zero similarities
                results.append(
                    SearchResult(
                        chunk_id=self._chunks[idx].chunk_id,
                        score=float(scores[idx]),
                        text=self._chunks[idx].text,  # optional, useful for debugging
                    )
                )
        return results