"""
Retrieval Gating Module for HRA-RLM
Selectively retrieves relevant passages before recursive reasoning
"""

from typing import List, Dict, Any
import re


class RetrievalGating:
    """Retrieval-gated recursion controller"""
    
    def __init__(self, top_k: int = 5):
        self.top_k = top_k
        self.total_tokens_retrieved = 0
    
    def retrieve(self, context: str, query: str) -> List[str]:
        """
        Retrieve most relevant passages from context
        
        Args:
            context: Full document text
            query: User question
        
        Returns:
            List of relevant passages (top_k)
        """
        # Simple keyword-based retrieval (for now)
        # In production, use vector DB or BM25
        query_keywords = self._extract_keywords(query)
        
        # Split context into chunks
        chunks = self._chunk_text(context)
        
        # Score each chunk
        scored_chunks = []
        for chunk in chunks:
            score = self._score_chunk(chunk, query_keywords)
            scored_chunks.append((score, chunk))
        
        # Sort by score and return top_k
        scored_chunks.sort(reverse=True, key=lambda x: x[0])
        retrieved = [chunk for _, chunk in scored_chunks[:self.top_k]]
        
        self.total_tokens_retrieved += sum(len(c) / 4 for c in retrieved)  # Approx tokens
        
        return retrieved
    
    def _extract_keywords(self, query: str) -> List[str]:
        """Extract important keywords from query"""
        # Remove stopwords and keep meaningful words
        stopwords = {'what', 'is', 'are', 'was', 'were', 'how', 'why', 'where', 'when',
                     'the', 'a', 'an', 'in', 'on', 'at', 'for', 'with', 'without', 'by'}
        words = query.lower().split()
        keywords = [w for w in words if w not in stopwords and len(w) > 3]
        return keywords[:10]  # Top 10 keywords
    
    def _chunk_text(self, text: str, chunk_size: int = 1000) -> List[str]:
        """Split text into chunks"""
        chunks = []
        for i in range(0, len(text), chunk_size):
            chunks.append(text[i:i+chunk_size])
        return chunks
    
    def _score_chunk(self, chunk: str, keywords: List[str]) -> int:
        """Score chunk based on keyword presence"""
        score = 0
        chunk_lower = chunk.lower()
        for kw in keywords:
            score += chunk_lower.count(kw)
        return score