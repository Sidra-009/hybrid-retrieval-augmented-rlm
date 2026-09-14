"""
Retrieval Gating Module for HRA-RLM
Selectively retrieves relevant passages before recursive reasoning
using semantic embeddings + HNSW approximate nearest-neighbor search.

Requires:
    pip install sentence-transformers hnswlib numpy
"""

from typing import List, Dict, Any
import re

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

try:
    import hnswlib
except ImportError:
    hnswlib = None

import numpy as np


class RetrievalGating:
    """
    Retrieval-gated recursion controller.

    Builds an HNSW index over sentence-level chunks of the document using
    dense embeddings, and retrieves the top_k most semantically similar
    chunks to the query. Falls back to keyword overlap only if the
    embedding dependencies are not installed, and clearly reports which
    mode was used so results are never silently mislabeled.
    """

    def __init__(self, top_k: int = 5, model_name: str = "all-MiniLM-L6-v2",
                 chunk_size: int = 500, ef_construction: int = 200, M: int = 16):
        self.top_k = top_k
        self.chunk_size = chunk_size
        self.total_tokens_retrieved = 0
        self.ef_construction = ef_construction
        self.M = M

        self.mode = "hnsw_semantic"
        self.model = None
        if SentenceTransformer is not None and hnswlib is not None:
            self.model = SentenceTransformer(model_name)
        else:
            self.mode = "keyword_fallback"

    def retrieve(self, context: str, query: str) -> List[str]:
        """
        Retrieve the most relevant passages from context for a query.

        Args:
            context: Full document text
            query: User question / sub-query

        Returns:
            List of relevant passages (top_k), ranked most to least relevant
        """
        chunks = self._chunk_text(context)

        if not chunks:
            return []

        if self.mode == "hnsw_semantic":
            retrieved = self._retrieve_hnsw(chunks, query)
        else:
            retrieved = self._retrieve_keyword(chunks, query)

        self.total_tokens_retrieved += sum(len(c) / 4 for c in retrieved)  # approx tokens
        return retrieved

    def _retrieve_hnsw(self, chunks: List[str], query: str) -> List[str]:
        """Semantic retrieval: embed chunks + query, search via HNSW index."""
        k = min(self.top_k, len(chunks))

        chunk_embeddings = self.model.encode(chunks, normalize_embeddings=True)
        query_embedding = self.model.encode([query], normalize_embeddings=True)

        dim = chunk_embeddings.shape[1]
        index = hnswlib.Index(space="cosine", dim=dim)
        # ef_construction/M control index build quality vs. speed; small
        # corpora don't need tuning, but exposing them keeps this honest
        # about being a real ANN index rather than a brute-force scan.
        index.init_index(max_elements=len(chunks), ef_construction=self.ef_construction, M=self.M)
        index.add_items(chunk_embeddings, np.arange(len(chunks)))
        index.set_ef(max(self.ef_construction, k + 1))

        labels, distances = index.knn_query(query_embedding, k=k)
        ranked_indices = labels[0]

        return [chunks[i] for i in ranked_indices]

    def _retrieve_keyword(self, chunks: List[str], query: str) -> List[str]:
        """Fallback used only when embedding deps aren't installed."""
        query_keywords = self._extract_keywords(query)
        scored_chunks = [(self._score_chunk(c, query_keywords), c) for c in chunks]
        scored_chunks.sort(reverse=True, key=lambda x: x[0])
        return [chunk for _, chunk in scored_chunks[: self.top_k]]

    def _extract_keywords(self, query: str) -> List[str]:
        stopwords = {'what', 'is', 'are', 'was', 'were', 'how', 'why', 'where', 'when',
                     'the', 'a', 'an', 'in', 'on', 'at', 'for', 'with', 'without', 'by'}
        words = query.lower().split()
        keywords = [w for w in words if w not in stopwords and len(w) > 3]
        return keywords[:10]

    def _chunk_text(self, text: str) -> List[str]:
        """Split text into overlap-free chunks. For real documents, prefer
        sentence-boundary chunking over a fixed character window."""
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        chunks, current = [], ""
        for sent in sentences:
            if len(current) + len(sent) <= self.chunk_size:
                current = f"{current} {sent}".strip()
            else:
                if current:
                    chunks.append(current)
                current = sent
        if current:
            chunks.append(current)
        return chunks

    def _score_chunk(self, chunk: str, keywords: List[str]) -> int:
        score = 0
        chunk_lower = chunk.lower()
        for kw in keywords:
            score += chunk_lower.count(kw)
        return score

    def get_stats(self) -> Dict[str, Any]:
        return {
            "mode": self.mode,
            "top_k": self.top_k,
            "total_tokens_retrieved_approx": self.total_tokens_retrieved,
        }