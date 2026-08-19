"""Hybrid Retrieval-Gated RLM agent.

Why this module exists (CORE NOVEL CONTRIBUTION):
This is the main innovation of HRA-RLM. Instead of passing the entire document
to the RLM (which is expensive and suffers from context rot), we:
1. Embed the user's query.
2. Retrieve only the top_k most relevant chunks from the vector store.
3. Pass ONLY those chunks as the `document` variable to the RLMAgent.

Tradeoffs vs raw MIT RLM:
- GAIN: ~10x cost reduction, faster inference, less context rot.
- RISK: If the retrieval misses relevant information, the answer will be
  incomplete (precision vs recall tradeoff). Adaptive_k helps mitigate this
  by increasing k when confidence is low.
"""

import logging
from typing import Any, Dict, List, Optional, Union

from src.hra_rlm.config.settings import get_settings
from src.hra_rlm.rlm.core import RLMAgent
from src.hra_rlm.rlm.models import ExecutionResult, SubCallRecord
from src.hra_rlm.vectordb.embeddings import EmbeddingProvider
from src.hra_rlm.vectordb.models import Chunk, SearchResult
from src.hra_rlm.vectordb.store import InMemoryVectorStore

logger = logging.getLogger(__name__)

# Phrases that indicate low confidence (for adaptive_k)
LOW_CONFIDENCE_PHRASES = [
    "i don't know",
    "i don't have enough information",
    "i'm not sure",
    "i cannot determine",
    "not enough context",
    "insufficient information",
    "cannot be determined",
    "unclear",
    "unknown",
]


class HybridRLMAgent:
    """Hybrid agent that gates RLM recursion behind vector retrieval."""

    def __init__(
        self,
        vector_store: InMemoryVectorStore,
        rlm_agent: RLMAgent,
        embedding_provider: Optional[EmbeddingProvider] = None,
        top_k: Optional[int] = None,
        retrieval_strategy: str = "fixed_k",
        max_adaptive_attempts: int = 2,
    ):
        self.vector_store = vector_store
        self.rlm_agent = rlm_agent
        self.embedding_provider = embedding_provider or EmbeddingProvider()
        self.top_k = top_k or get_settings().TOP_K_CHUNKS
        self.retrieval_strategy = retrieval_strategy
        self.max_adaptive_attempts = max_adaptive_attempts

        self.last_retrieved_count = 0
        self.last_full_document_size = 0
        self.adaptive_attempts_used = 0

        if retrieval_strategy not in ["fixed_k", "adaptive_k"]:
            raise ValueError(
                f"Invalid retrieval_strategy: {retrieval_strategy}. Must be 'fixed_k' or 'adaptive_k'."
            )

        logger.info(f"HybridRLMAgent initialized: strategy={retrieval_strategy}, top_k={top_k}")

    def _is_low_confidence(self, answer: str) -> bool:
        """Check if an answer contains low-confidence phrases."""
        if not answer:
            return True
        answer_lower = answer.lower()
        for phrase in LOW_CONFIDENCE_PHRASES:
            if phrase in answer_lower:
                logger.debug(f"Low confidence detected: '{phrase}' found in answer")
                return True
        return False

    def _build_document_from_chunks(self, search_results: List[SearchResult]) -> str:
        """Build a document string from retrieved chunks."""
        if not search_results:
            return ""

        chunks_text = []
        for i, result in enumerate(search_results, 1):
            chunk_text = result.text or f"[Chunk {i}]"
            chunks_text.append(f"--- Chunk {i} (score: {result.score:.3f}) ---\n{chunk_text}")

        return "\n\n".join(chunks_text)

    def run(self, query: str) -> Dict[str, Any]:
        """Run the hybrid retrieval-gated RLM pipeline."""
        logger.info(f"Hybrid RLM run: '{query[:100]}...' (strategy={self.retrieval_strategy})")

        current_k = self.top_k
        adaptive_attempts = 0
        final_result = None

        while adaptive_attempts < self.max_adaptive_attempts:
            # Step 1: Embed query
            query_embedding = self.embedding_provider.embed_query(query)
            logger.debug(f"Query embedding generated (dim={len(query_embedding)})")

            # Step 2: Retrieve chunks
            search_results = self.vector_store.search(query_embedding, top_k=current_k)
            self.last_retrieved_count = len(search_results)
            self.last_full_document_size = self.vector_store.size

            logger.info(f"Retrieved {self.last_retrieved_count} chunks (k={current_k})")

            if not search_results:
                logger.warning("No chunks retrieved from vector store. Falling back to empty document.")
                document_text = ""
            else:
                document_text = self._build_document_from_chunks(search_results)

            # Step 3: Delegate to RLM agent
            logger.debug(f"Delegating to RLMAgent with {len(document_text)} chars of context")
            raw_result = self.rlm_agent.run(query=query, document=document_text)

            # RLMAgent returns either a string or dict. Convert to string.
            final_answer = raw_result if isinstance(raw_result, str) else str(raw_result)

            # Step 4: Check confidence for adaptive strategy
            if self.retrieval_strategy == "fixed_k":
                final_result = {
                    "answer": final_answer,
                    "total_cost": self.rlm_agent.total_cost_usd,
                    "total_tokens": self.rlm_agent.total_tokens,
                    "total_latency_ms": self.rlm_agent.total_latency_ms,
                    "chunks_retrieved": self.last_retrieved_count,
                    "adaptive_attempts": 0,
                    "sub_calls": self.rlm_agent.sub_call_records,
                    "full_document_size": self.last_full_document_size,
                    "retrieval_strategy": "fixed_k",
                }
                break

            # Adaptive strategy: check confidence
            if self._is_low_confidence(final_answer):
                adaptive_attempts += 1
                self.adaptive_attempts_used = adaptive_attempts
                current_k = min(current_k * 2, self.last_full_document_size or current_k * 2)
                logger.warning(
                    f"Low confidence detected. Increasing k to {current_k} (attempt {adaptive_attempts})"
                )

                if current_k >= self.last_full_document_size and self.last_full_document_size > 0:
                    logger.info(
                        f"k ({current_k}) >= full document size ({self.last_full_document_size}). Falling through."
                    )
                    final_result = {
                        "answer": final_answer,
                        "total_cost": self.rlm_agent.total_cost_usd,
                        "total_tokens": self.rlm_agent.total_tokens,
                        "total_latency_ms": self.rlm_agent.total_latency_ms,
                        "chunks_retrieved": self.last_retrieved_count,
                        "adaptive_attempts": adaptive_attempts,
                        "sub_calls": self.rlm_agent.sub_call_records,
                        "full_document_size": self.last_full_document_size,
                        "retrieval_strategy": "adaptive_k",
                        "max_adaptive_reached": True,
                    }
                    break
                continue
            else:
                final_result = {
                    "answer": final_answer,
                    "total_cost": self.rlm_agent.total_cost_usd,
                    "total_tokens": self.rlm_agent.total_tokens,
                    "total_latency_ms": self.rlm_agent.total_latency_ms,
                    "chunks_retrieved": self.last_retrieved_count,
                    "adaptive_attempts": adaptive_attempts,
                    "sub_calls": self.rlm_agent.sub_call_records,
                    "full_document_size": self.last_full_document_size,
                    "retrieval_strategy": "adaptive_k",
                    "max_adaptive_reached": False,
                }
                break

        if final_result is None:
            final_result = {
                "answer": "Error: Max adaptive attempts reached without a valid result.",
                "total_cost": self.rlm_agent.total_cost_usd,
                "total_tokens": self.rlm_agent.total_tokens,
                "total_latency_ms": self.rlm_agent.total_latency_ms,
                "chunks_retrieved": 0,
                "adaptive_attempts": adaptive_attempts,
                "sub_calls": self.rlm_agent.sub_call_records,
                "full_document_size": self.last_full_document_size,
                "retrieval_strategy": self.retrieval_strategy,
                "error": True,
            }

        logger.info(
            f"Hybrid RLM complete: cost=${final_result['total_cost']:.6f}, "
            f"chunks={final_result['chunks_retrieved']}, "
            f"adaptive_attempts={final_result.get('adaptive_attempts', 0)}"
        )

        return final_result