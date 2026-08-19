"""Auto-Healer for RLM agents with context rot detection.

Why this module exists:
Wraps the HybridRLMAgent with rot detection and auto-healing capabilities.
When context rot is detected, automatically switches strategy:
- Increase top_k for retrieval
- Reduce recursion depth
- Switch from fixed_k to adaptive_k
- Or any combination of these

Every switch decision is logged for auditability.
"""

import logging
from typing import Any, Dict, List, Optional

from src.hra_rlm.config.settings import get_settings
from src.hra_rlm.rlm.hybrid import HybridRLMAgent
from src.hra_rlm.rlm.models import SubCallRecord
from src.hra_rlm.rot_detector.detector import ContextRotDetector, RotScore

logger = logging.getLogger(__name__)


class AutoHealer:
    """Auto-Healer that wraps HybridRLMAgent with rot detection and healing.

    The healer runs a query, scores the result, and if rot is detected,
    retries with a different strategy (e.g., increase k, reduce depth).
    All decisions are logged for auditability.
    """

    def __init__(
        self,
        hybrid_agent: HybridRLMAgent,
        detector: Optional[ContextRotDetector] = None,
        max_healing_attempts: int = 3,
        k_increase_factor: float = 2.0,
        allow_depth_reduction: bool = True,
        allow_strategy_switch: bool = True,
    ):
        """Initialize the AutoHealer.

        Args:
            hybrid_agent: The HybridRLMAgent to wrap.
            detector: ContextRotDetector instance. Defaults to new instance.
            max_healing_attempts: Maximum number of healing attempts before giving up.
            k_increase_factor: Factor by which to increase k on each healing attempt.
            allow_depth_reduction: Whether to reduce recursion depth as a healing strategy.
            allow_strategy_switch: Whether to switch retrieval strategy as a healing strategy.
        """
        self.hybrid_agent = hybrid_agent
        self.detector = detector or ContextRotDetector(
            threshold=get_settings().ROT_THRESHOLD
        )
        self.max_healing_attempts = max_healing_attempts
        self.k_increase_factor = k_increase_factor
        self.allow_depth_reduction = allow_depth_reduction
        self.allow_strategy_switch = allow_strategy_switch

        # Track healing attempts for audit
        self.healing_history: List[Dict[str, Any]] = []
        self.last_rot_score: Optional[RotScore] = None
        self.healing_attempts_made = 0

        # Store original configuration for reset
        self._original_top_k = hybrid_agent.top_k
        self._original_strategy = hybrid_agent.retrieval_strategy
        self._original_max_depth = hybrid_agent.rlm_agent.max_recursion_depth

    def _log_healing_decision(self, attempt: int, old_score: RotScore, new_config: Dict[str, Any]) -> None:
        """Log a healing decision for auditability."""
        entry = {
            "attempt": attempt,
            "old_confidence": old_score.confidence,
            "old_reason": old_score.reason,
            "new_config": new_config,
        }
        self.healing_history.append(entry)
        logger.info(
            f"Healing attempt {attempt}: confidence={old_score.confidence:.3f}, "
            f"reason={old_score.reason}, new_config={new_config}"
        )

    def _apply_healing_strategy(self, attempt: int) -> Dict[str, Any]:
        """Apply a healing strategy based on the attempt number.

        Strategy progression:
        1. Increase top_k
        2. Increase top_k further + switch to adaptive_k (if not already)
        3. Reduce recursion depth + increase top_k

        Returns:
            Dict with the new configuration applied.
        """
        config = {
            "top_k": self.hybrid_agent.top_k,
            "retrieval_strategy": self.hybrid_agent.retrieval_strategy,
            "max_recursion_depth": self.hybrid_agent.rlm_agent.max_recursion_depth,
        }

        # Strategy progression
        if attempt == 1:
            # First attempt: increase k
            new_k = int(self.hybrid_agent.top_k * self.k_increase_factor)
            self.hybrid_agent.top_k = new_k
            config["top_k"] = new_k
            logger.info(f"Healing strategy: increased top_k to {new_k}")

        elif attempt == 2:
            # Second attempt: increase k more + switch to adaptive_k
            new_k = int(self.hybrid_agent.top_k * self.k_increase_factor)
            self.hybrid_agent.top_k = new_k
            config["top_k"] = new_k

            if self.allow_strategy_switch:
                self.hybrid_agent.retrieval_strategy = "adaptive_k"
                config["retrieval_strategy"] = "adaptive_k"
                logger.info(f"Healing strategy: increased top_k to {new_k}, switched to adaptive_k")

        elif attempt == 3:
            # Third attempt: increase k + reduce depth
            new_k = int(self.hybrid_agent.top_k * self.k_increase_factor)
            self.hybrid_agent.top_k = new_k
            config["top_k"] = new_k

            if self.allow_depth_reduction:
                new_depth = max(1, self.hybrid_agent.rlm_agent.max_recursion_depth // 2)
                self.hybrid_agent.rlm_agent.max_recursion_depth = new_depth
                config["max_recursion_depth"] = new_depth
                logger.info(f"Healing strategy: increased top_k to {new_k}, reduced depth to {new_depth}")

        return config

    def _reset_configuration(self) -> None:
        """Reset the hybrid agent to its original configuration."""
        self.hybrid_agent.top_k = self._original_top_k
        self.hybrid_agent.retrieval_strategy = self._original_strategy
        self.hybrid_agent.rlm_agent.max_recursion_depth = self._original_max_depth
        logger.debug("Configuration reset to original values")

    def run(self, query: str) -> Dict[str, Any]:
        """Run the query with auto-healing.

        Args:
            query: The user's question.

        Returns:
            Dictionary with answer, cost, tokens, and healing metadata.
        """
        logger.info(f"AutoHealer running query: '{query[:100]}...'")

        self.healing_attempts_made = 0
        self.healing_history = []
        self.last_rot_score = None

        # First run
        result = self.hybrid_agent.run(query)
        answer = result.get("answer", "")

        # Score the response
        sub_calls = result.get("sub_calls", [])
        rot_score = self.detector.score_response(
            response=answer,
            sub_call_records=sub_calls,
        )
        self.last_rot_score = rot_score

        # Check if healing is needed
        if not rot_score.is_rotting:
            logger.info(f"No rot detected. Confidence: {rot_score.confidence:.3f}")
            result["healing_attempts"] = 0
            result["healing_history"] = []
            result["rot_score"] = {
                "confidence": rot_score.confidence,
                "is_rotting": rot_score.is_rotting,
                "reason": rot_score.reason,
            }
            return result

        # Rot detected — attempt healing
        logger.warning(f"Rot detected! Confidence: {rot_score.confidence:.3f}. Attempting healing...")

        for attempt in range(1, self.max_healing_attempts + 1):
            self.healing_attempts_made = attempt

            # Apply healing strategy
            new_config = self._apply_healing_strategy(attempt)
            self._log_healing_decision(attempt, rot_score, new_config)

            # Re-run with new configuration
            logger.info(f"Healing attempt {attempt}: re-running query...")
            healed_result = self.hybrid_agent.run(query)

            # Score the new result
            healed_answer = healed_result.get("answer", "")
            healed_sub_calls = healed_result.get("sub_calls", [])
            new_rot_score = self.detector.score_response(
                response=healed_answer,
                sub_call_records=healed_sub_calls,
            )
            self.last_rot_score = new_rot_score

            # Check if healing worked
            if not new_rot_score.is_rotting:
                logger.info(f"Healing succeeded on attempt {attempt}! Confidence: {new_rot_score.confidence:.3f}")
                healed_result["healing_attempts"] = attempt
                healed_result["healing_history"] = self.healing_history
                healed_result["rot_score"] = {
                    "confidence": new_rot_score.confidence,
                    "is_rotting": new_rot_score.is_rotting,
                    "reason": new_rot_score.reason,
                }
                # Keep the new configuration (don't reset)
                return healed_result

            # Still rotting — prepare for next attempt
            rot_score = new_rot_score
            logger.warning(f"Healing attempt {attempt} failed. Confidence: {new_rot_score.confidence:.3f}")

        # All healing attempts failed
        logger.error(f"All {self.max_healing_attempts} healing attempts failed. Returning best result.")

        # Reset configuration
        self._reset_configuration()

        # Return the last result (even if still rotting)
        result["healing_attempts"] = self.max_healing_attempts
        result["healing_history"] = self.healing_history
        result["rot_score"] = {
            "confidence": self.last_rot_score.confidence if self.last_rot_score else 0.0,
            "is_rotting": True,
            "reason": "All healing attempts failed",
        }
        result["healing_failed"] = True

        return result

    def get_healing_summary(self) -> Dict[str, Any]:
        """Get a summary of all healing operations performed.

        Returns:
            Dict with healing statistics.
        """
        total_attempts = self.healing_attempts_made
        successful_attempts = sum(
            1 for h in self.healing_history
            if h.get("new_config") is not None
        )

        return {
            "total_healing_attempts": total_attempts,
            "successful_healing_attempts": successful_attempts,
            "history": self.healing_history,
            "last_rot_score": {
                "confidence": self.last_rot_score.confidence if self.last_rot_score else 0.0,
                "is_rotting": self.last_rot_score.is_rotting if self.last_rot_score else False,
                "reason": self.last_rot_score.reason if self.last_rot_score else "No rot score",
            } if self.last_rot_score else None,
        }