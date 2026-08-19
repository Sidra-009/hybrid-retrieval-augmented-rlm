"""Unit tests for Context Rot Detector and AutoHealer."""

from typing import Any, Dict

import pytest

from src.hra_rlm.rlm.models import SubCallRecord
from src.hra_rlm.rot_detector.detector import ContextRotDetector
from src.hra_rlm.rot_detector.healer import AutoHealer


def test_detector_high_confidence() -> None:
    """Test that high-confidence responses are not flagged as rotting."""
    detector = ContextRotDetector(threshold=0.4)

    response = "This is a clear, confident answer with specific details. The answer is 42."
    sub_calls = []

    score = detector.score_response(response, sub_calls)

    assert score.confidence > 0.7
    assert score.is_rotting is False
    assert "No rot signals" in score.reason


def test_detector_low_confidence_uncertainty() -> None:
    """Test that uncertain responses are flagged as rotting."""
    detector = ContextRotDetector(threshold=0.9)

    response = "I don't know the answer. It is unclear and I'm not sure."
    sub_calls = []

    score = detector.score_response(response, sub_calls)

    # With threshold 0.9, this should definitely be rotting
    assert score.is_rotting is True
    # Check for either uncertainty or combined confidence reason
    assert "uncertainty" in score.reason.lower() or "combined confidence" in score.reason.lower()


def test_detector_short_response() -> None:
    """Test that very short responses are flagged as rotting."""
    detector = ContextRotDetector(threshold=0.9, min_response_length=50)

    response = "No."
    sub_calls = []

    score = detector.score_response(response, sub_calls)

    assert score.is_rotting is True
    assert "short" in score.reason.lower() or "combined confidence" in score.reason.lower()


def test_detector_high_sub_call_count() -> None:
    """Test that too many sub-calls trigger rot detection."""
    detector = ContextRotDetector(threshold=0.9, max_sub_calls=2)

    response = "This is a detailed response."
    sub_calls = [
        SubCallRecord(prompt="q1", response="a1", tokens_used=10, cost_usd=0.0, latency_ms=0.0),
        SubCallRecord(prompt="q2", response="a2", tokens_used=10, cost_usd=0.0, latency_ms=0.0),
        SubCallRecord(prompt="q3", response="a3", tokens_used=10, cost_usd=0.0, latency_ms=0.0),
        SubCallRecord(prompt="q4", response="a4", tokens_used=10, cost_usd=0.0, latency_ms=0.0),
    ]

    score = detector.score_response(response, sub_calls)

    assert score.is_rotting is True
    assert "sub-call" in score.reason.lower() or "combined confidence" in score.reason.lower()


def test_detector_consistent_sub_calls() -> None:
    """Test that consistent sub-calls don't trigger rot."""
    detector = ContextRotDetector(threshold=0.4)

    response = "This is a long detailed response."
    sub_calls = [
        SubCallRecord(prompt="q1", response="Response 1", tokens_used=10, cost_usd=0.0, latency_ms=0.0),
        SubCallRecord(prompt="q2", response="Response 2", tokens_used=10, cost_usd=0.0, latency_ms=0.0),
    ]

    score = detector.score_response(response, sub_calls)

    assert score.signals["sub_call_consistency_score"] < 0.5
    assert score.is_rotting is False


def test_should_switch_strategy() -> None:
    """Test that should_switch_strategy returns correct value."""
    detector = ContextRotDetector(threshold=0.8)

    # High confidence
    score1 = detector.score_response("Clear confident answer.", [])
    assert detector.should_switch_strategy(score1) is False

    # Low confidence
    score2 = detector.score_response("I don't know. Not sure. Unclear.", [])
    assert detector.should_switch_strategy(score2) is True


def test_autohealer_no_rot() -> None:
    """Test that AutoHealer doesn't attempt healing when no rot is detected."""
    from src.hra_rlm.rlm.hybrid import HybridRLMAgent
    from src.hra_rlm.vectordb.store import InMemoryVectorStore
    from src.hra_rlm.vectordb.embeddings import EmbeddingProvider

    store = InMemoryVectorStore()

    class MockRLM:
        def __init__(self):
            self.max_recursion_depth = 3
            self.total_cost_usd = 0.0
            self.total_tokens = 0
            self.total_latency_ms = 0
            self.sub_call_records = []

        def run(self, query: str, document: Any) -> str:
            return "This is a clear, confident answer with specific details."

    mock_rlm = MockRLM()

    class RLMAgentWrapper:
        def __init__(self, mock):
            self.run = mock.run
            self.max_recursion_depth = mock.max_recursion_depth
            self.total_cost_usd = mock.total_cost_usd
            self.total_tokens = mock.total_tokens
            self.total_latency_ms = mock.total_latency_ms
            self.sub_call_records = mock.sub_call_records

    rlm_agent = RLMAgentWrapper(mock_rlm)

    embed_provider = EmbeddingProvider()
    hybrid = HybridRLMAgent(
        vector_store=store,
        rlm_agent=rlm_agent,  # type: ignore
        embedding_provider=embed_provider,
        top_k=3,
        retrieval_strategy="fixed_k",
    )

    healer = AutoHealer(hybrid_agent=hybrid, max_healing_attempts=2)

    result = healer.run(query="Test query")

    assert result["healing_attempts"] == 0


def test_autohealer_heals_rot() -> None:
    """Test that AutoHealer attempts healing when rot is detected."""
    from src.hra_rlm.rlm.hybrid import HybridRLMAgent
    from src.hra_rlm.vectordb.store import InMemoryVectorStore
    from src.hra_rlm.vectordb.embeddings import EmbeddingProvider

    store = InMemoryVectorStore()

    class MockRLM:
        def __init__(self):
            self.call_count = 0
            self.max_recursion_depth = 3
            self.total_cost_usd = 0.0
            self.total_tokens = 0
            self.total_latency_ms = 0
            self.sub_call_records = []

        def run(self, query: str, document: Any) -> str:
            self.call_count += 1
            if self.call_count == 1:
                return "I don't have enough information to answer this."
            else:
                return "This is the correct answer with high confidence."

    mock_rlm = MockRLM()

    class RLMAgentWrapper:
        def __init__(self, mock):
            self.run = mock.run
            self.max_recursion_depth = mock.max_recursion_depth
            self.total_cost_usd = mock.total_cost_usd
            self.total_tokens = mock.total_tokens
            self.total_latency_ms = mock.total_latency_ms
            self.sub_call_records = mock.sub_call_records

    rlm_agent = RLMAgentWrapper(mock_rlm)

    embed_provider = EmbeddingProvider()
    hybrid = HybridRLMAgent(
        vector_store=store,
        rlm_agent=rlm_agent,  # type: ignore
        embedding_provider=embed_provider,
        top_k=3,
        retrieval_strategy="fixed_k",
    )

    detector = ContextRotDetector(threshold=0.9)

    healer = AutoHealer(
        hybrid_agent=hybrid,
        detector=detector,
        max_healing_attempts=3,
        k_increase_factor=2.0,
        allow_depth_reduction=True,
        allow_strategy_switch=True,
    )

    result = healer.run(query="Test query")

    assert "healing_attempts" in result
    assert "rot_score" in result
    assert result["healing_attempts"] >= 1


def test_healing_history_tracking() -> None:
    """Test that healing history is tracked properly."""
    from src.hra_rlm.rlm.hybrid import HybridRLMAgent
    from src.hra_rlm.vectordb.store import InMemoryVectorStore
    from src.hra_rlm.vectordb.embeddings import EmbeddingProvider

    store = InMemoryVectorStore()
    embed_provider = EmbeddingProvider()

    class MockHybrid:
        def __init__(self):
            self.top_k = 3
            self.retrieval_strategy = "fixed_k"
            self.rlm_agent = type('obj', (object,), {'max_recursion_depth': 3})()

        def run(self, query: str) -> Dict[str, Any]:
            return {
                "answer": "I don't have enough information.",
                "total_cost": 0.0,
                "total_tokens": 0,
                "total_latency_ms": 0.0,
                "chunks_retrieved": 3,
                "adaptive_attempts": 0,
                "sub_calls": [],
                "full_document_size": 10,
                "retrieval_strategy": "fixed_k",
            }

    hybrid = MockHybrid()

    detector = ContextRotDetector(threshold=0.9)
    healer = AutoHealer(hybrid_agent=hybrid, detector=detector, max_healing_attempts=2)  # type: ignore

    result = healer.run(query="Test")

    assert "healing_history" in result
    assert "healing_attempts" in result