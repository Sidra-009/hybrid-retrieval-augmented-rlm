"""Embedding provider for vector database operations.

Why this module exists:
Generates vector embeddings for text chunks and queries.
Uses local sentence-transformers models (free, no API key required) to avoid
paid embedding APIs, keeping the project cost-effective.

Supports fallback to random embeddings for testing when the model is unavailable.
"""

import logging
from typing import List, Optional

import numpy as np

from src.hra_rlm.config.settings import get_settings

logger = logging.getLogger(__name__)


class EmbeddingProvider:
    """Provides text embeddings using a local sentence-transformers model."""

    def __init__(self, model_name: Optional[str] = None):
        """Initialize the embedding provider.

        Args:
            model_name: Name of the sentence-transformers model to use.
                       Defaults to config setting.
        """
        self.model_name = model_name or get_settings().SENTENCE_TRANSFORMER_MODEL
        self._model = None
        self._embedding_dim = None

    @property
    def model(self):
        """Lazy-load the sentence-transformers model."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self.model_name)
                self._embedding_dim = self._model.get_sentence_embedding_dimension()
                logger.info(f"Loaded embedding model: {self.model_name} (dim={self._embedding_dim})")
            except ImportError:
                logger.warning("sentence-transformers not installed. Using random embeddings fallback.")
                self._model = "fallback"
                self._embedding_dim = 384
        return self._model

    @property
    def embedding_dim(self) -> int:
        """Return the dimension of embeddings produced by this provider."""
        if self._embedding_dim is None:
            _ = self.model  # trigger lazy load
        return self._embedding_dim or 384

    def embed(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts.

        Args:
            texts: List of text strings to embed.

        Returns:
            List of embedding vectors (list of floats).
        """
        if not texts:
            return []

        model = self.model

        # Fallback for testing when sentence-transformers is not available
        if model == "fallback":
            logger.debug("Using random embedding fallback")
            return [
                np.random.randn(self.embedding_dim).astype(np.float32).tolist()
                for _ in texts
            ]

        try:
            embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
            # Ensure it's a 2D array
            if len(texts) == 1:
                embeddings = embeddings.reshape(1, -1)
            return embeddings.astype(np.float32).tolist()
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            raise RuntimeError(f"Failed to generate embeddings: {e}")

    def embed_query(self, query: str) -> List[float]:
        """Generate an embedding for a single query string.

        Args:
            query: The query text.

        Returns:
            Embedding vector as a list of floats.
        """
        results = self.embed([query])
        return results[0] if results else []

    @property
    def embedding_dim(self) -> int:
        """Return the dimension of embeddings produced by this provider."""
        if self._embedding_dim is None:
            _ = self.model  # trigger lazy load
            # Use the new method name if available, fallback to old
            if hasattr(self._model, 'get_embedding_dimension'):
                self._embedding_dim = self._model.get_embedding_dimension()
            else:
                self._embedding_dim = self._model.get_sentence_embedding_dimension()
        return self._embedding_dim or 384