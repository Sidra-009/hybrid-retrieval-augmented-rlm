"""Unit tests for the HybridRLMAgent."""

from typing import Any

import pytest

from src.hra_rlm.rlm.core import MockLLMClient, RLMAgent
from src.hra_rlm.rlm.hybrid import HybridRLMAgent
from src.hra_rlm.rlm.models import ExecutionResult
from src.hra_rlm.rlm.repl import SandboxREPL
from src.hra_rlm.vectordb.embeddings import EmbeddingProvider
from src.hra_rlm.vectordb.models import Chunk
from src.hra_rlm.vectordb.store import InMemoryVectorStore


class MockEmbeddingProvider(EmbeddingProvider):
    """Mock embedding provider for testing."""

    def __init__(self, dim: int = 384):
        self.dim = dim
        self._model = "fallback"
        self._embedding_dim = dim

    def embed_query(self, query: str) -> list:
        import numpy as np
        return np.random.randn(self.dim).astype(np.float32).tolist()

    def embed(self, texts: list) -> list:
        import numpy as np
        return [np.random.randn(self.dim).astype(np.float32).tolist() for _ in texts]

    @property
    def embedding_dim(self):
        return self.dim


def test_retrieval_reduces_document_size() -> None:
    """Test that retrieval reduces the document size passed to RLM."""
    store = InMemoryVectorStore()
    for i in range(50):
        store.add(Chunk(
            chunk_id=f"chunk_{i}",
            text=f"This is chunk number {i} with some content.",
            embedding=[float(i) / 50] * 384,
        ))

    class RecordingRLMAgent(RLMAgent):
        def __init__(self):
            self.received_documents = []
            self.total_cost_usd = 0.0
            self.total_tokens = 0
            self.total_latency_ms = 0
            self.sub_call_records = []

        def run(self, query: str, document: Any) -> str:
            self.received_documents.append(document)
            return f"Answer for: {query}"

    rlm_agent = RecordingRLMAgent()
    embed_provider = MockEmbeddingProvider()

    hybrid = HybridRLMAgent(
        vector_store=store,
        rlm_agent=rlm_agent,
        embedding_provider=embed_provider,
        top_k=5,
    )

    result = hybrid.run(query="What is in the document?")

    assert len(rlm_agent.received_documents) == 1
    doc_text = rlm_agent.received_documents[0]
    assert "Chunk 1" in doc_text or "Chunk" in doc_text
    assert result["chunks_retrieved"] == 5
    assert result["full_document_size"] == 50


def test_adaptive_k_triggers_second_retrieval_on_low_confidence() -> None:
    """Test that adaptive_k triggers a second retrieval when confidence is low."""
    store = InMemoryVectorStore()
    for i in range(20):
        store.add(Chunk(
            chunk_id=f"chunk_{i}",
            text=f"Content for chunk {i}. Important information is scattered.",
            embedding=[float(i) / 20] * 384,
        ))

    # Mock RLM agent that returns low confidence on first call, high on second
    class MockRLM:
        def __init__(self):
            self.call_count = 0
            self.total_cost_usd = 0.0
            self.total_tokens = 0
            self.total_latency_ms = 0
            self.sub_call_records = []

        def run(self, query: str, document: Any) -> str:
            if self.call_count == 0:
                self.call_count += 1
                # Use "don't" to match LOW_CONFIDENCE_PHRASES
                return "I don't have enough information to answer this."
            else:
                return "Final answer with high confidence."

    rlm_agent = MockRLM()
    embed_provider = MockEmbeddingProvider()

    hybrid = HybridRLMAgent(
        vector_store=store,
        rlm_agent=rlm_agent,  # type: ignore
        embedding_provider=embed_provider,
        top_k=3,
        retrieval_strategy="adaptive_k",
        max_adaptive_attempts=3,
    )

    result = hybrid.run(query="What is the answer?")

    # Adaptive should have triggered at least 1 retry
    assert result["adaptive_attempts"] >= 1
    assert "Answer" in result["answer"] or "high confidence" in result["answer"].lower()
    assert result["total_cost"] >= 0


def test_fixed_k_does_not_retry() -> None:
    """Test that fixed_k strategy does not retry even with low confidence."""
    store = InMemoryVectorStore()
    store.add(Chunk(chunk_id="a", text="Test content", embedding=[1.0] * 384))

    class FixedRLM:
        def __init__(self):
            self.total_cost_usd = 0.0
            self.total_tokens = 0
            self.total_latency_ms = 0
            self.sub_call_records = []

        def run(self, query: str, document: Any) -> str:
            return "I am not sure"

    rlm_agent = FixedRLM()
    embed_provider = MockEmbeddingProvider()

    hybrid = HybridRLMAgent(
        vector_store=store,
        rlm_agent=rlm_agent,  # type: ignore
        embedding_provider=embed_provider,
        top_k=3,
        retrieval_strategy="fixed_k",
    )

    result = hybrid.run(query="Test query")
    assert result["adaptive_attempts"] == 0
    assert result["retrieval_strategy"] == "fixed_k"


def test_handles_empty_vector_store() -> None:
    """Test that empty vector store is handled gracefully."""
    store = InMemoryVectorStore()
    class EmptyRLM:
        def __init__(self):
            self.total_cost_usd = 0.0
            self.total_tokens = 0
            self.total_latency_ms = 0
            self.sub_call_records = []

        def run(self, query: str, document: Any) -> str:
            return "No chunks found in store."

    rlm_agent = EmptyRLM()
    embed_provider = MockEmbeddingProvider()

    hybrid = HybridRLMAgent(
        vector_store=store,
        rlm_agent=rlm_agent,  # type: ignore
        embedding_provider=embed_provider,
        top_k=3,
    )

    result = hybrid.run(query="Test")
    assert result["chunks_retrieved"] == 0
    assert result["full_document_size"] == 0