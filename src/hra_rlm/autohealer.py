"""
AutoHealer Module for HRA-RLM
Detects context rot and automatically recovers
"""

from typing import Dict, Any, Optional


class AutoHealer:
    """Detects and mitigates reasoning quality degradation"""
    
    def __init__(self, confidence_threshold: float = 0.7):
        self.confidence_threshold = confidence_threshold
        self.rot_detected = False
        self.recovery_attempts = 0
        self.answer_history = []
    
    def detect_rot(self, current_answer: str, previous_answer: Optional[str] = None) -> bool:
        """
        Detect if context rot is happening
        
        Args:
            current_answer: Latest LLM response
            previous_answer: Previous LLM response (if any)
        
        Returns:
            True if rot detected, False otherwise
        """
        # Heuristic 1: Answer too short
        if len(current_answer.strip()) < 10:
            self.rot_detected = True
            return True
        
        # Heuristic 2: Previous answer exists and length changed dramatically
        if previous_answer:
            prev_len = len(previous_answer)
            curr_len = len(current_answer)
            if prev_len > 0:
                ratio = abs(curr_len - prev_len) / max(prev_len, 1)
                if ratio > 0.8:  # 80% change
                    self.rot_detected = True
                    return True
        
        # Heuristic 3: Answer contains repetition or gibberish
        words = current_answer.split()
        if len(words) > 5:
            # Check if same word repeated many times
            from collections import Counter
            counter = Counter(words)
            most_common = counter.most_common(1)[0]
            if most_common[1] / len(words) > 0.5:  # 50% same word
                self.rot_detected = True
                return True
        
        self.rot_detected = False
        return False
    
    def heal(self, query: str, context_chunks: list) -> str:
        """
        Attempt to recover from context rot
        
        Args:
            query: Original user query
            context_chunks: Retrieved context chunks
        
        Returns:
            Healed/regenerated answer
        """
        self.recovery_attempts += 1
        
        # Strategy: Use smaller, more focused chunks
        smaller_chunks = []
        for chunk in context_chunks[:5]:  # Limit to 5 chunks
            if len(chunk) > 500:
                # Split into smaller chunks
                mid = len(chunk) // 2
                smaller_chunks.append(chunk[:mid])
                smaller_chunks.append(chunk[mid:])
            else:
                smaller_chunks.append(chunk)
        
        # Build healed prompt
        context_text = "\n---\n".join(smaller_chunks[:8])  # Limit to 8 chunks
        
        return f"""
[AUTOHEALER - Recovery Attempt #{self.recovery_attempts}]

Context rot was detected in previous answer. 
Re-answering with more focused, smaller chunks:

--- CONTEXT ---
{context_text}

--- QUESTION ---
{query}

--- INSTRUCTIONS ---
Please provide a clear, concise, and accurate answer.
Focus only on the most relevant information.
Be specific and avoid repetition.

Answer:
"""
    
    def get_stats(self) -> Dict[str, Any]:
        """Return healing statistics"""
        return {
            "rot_detected": self.rot_detected,
            "recovery_attempts": self.recovery_attempts,
            "confidence_threshold": self.confidence_threshold,
            "history_size": len(self.answer_history)
        }