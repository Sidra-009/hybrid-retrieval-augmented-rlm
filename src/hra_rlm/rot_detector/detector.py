"""Context Rot Detector for RLM agents.

Why this module exists:
Detects when the model's effective accuracy/confidence is degrading as context grows
("context rot"). Uses heuristic signals like:
- Response length and hedging language
- Number of sub-calls needed
- Disagreement between sub-calls
- Self-reported uncertainty phrases

IMPORTANT: This is a HEURISTIC proxy, not a ground-truth accuracy measure.
This is a known research limitation to be stated explicitly.
"""

import logging
import re
from dataclasses import dataclass
from typing import List, Optional

from src.hra_rlm.rlm.models import SubCallRecord

logger = logging.getLogger(__name__)

# Phrases that indicate uncertainty or hedging
UNCERTAINTY_PHRASES = [
    r"\b(?:i (?:am|'m) not (?:sure|certain|confident)\b)",
    r"\b(?:i (?:do|don't|do not) (?:know|have enough information)\b)",
    r"\b(?:maybe|perhaps|possibly|probably|it seems|it appears|i think|i believe)\b",
    r"\b(?:not (?:clear|obvious|well-defined|specified))\b",
    r"\b(?:difficult to (?:determine|say|know|ascertain))\b",
    r"\b(?:cannot (?:be determined|be confirmed|say with certainty))\b",
    r"\b(?:insufficient (?:information|context|data))\b",
    r"\b(?:unclear|ambiguous|uncertain)\b",
    r"\b(?:not enough (?:information|context|data))\b",
]


@dataclass
class RotScore:
    """Confidence score and rot detection result.

    Attributes:
        confidence: Float between 0.0 and 1.0 indicating estimated confidence.
        is_rotting: Boolean indicating whether context rot is detected.
        signals: Dictionary of individual signal values used in the score.
        reason: Human-readable explanation of the score.
    """
    confidence: float
    is_rotting: bool
    signals: dict
    reason: str


class ContextRotDetector:
    """Detects context rot in RLM responses using heuristic signals."""

    def __init__(
        self,
        threshold: float = 0.4,
        min_response_length: int = 50,
        max_sub_calls: int = 3,
    ):
        self.threshold = threshold
        self.min_response_length = min_response_length
        self.max_sub_calls = max_sub_calls

    def _check_uncertainty_phrases(self, response: str) -> float:
        """Check for uncertainty/hedging phrases in response."""
        if not response:
            return 1.0

        response_lower = response.lower()
        matches = 0

        for pattern in UNCERTAINTY_PHRASES:
            if re.search(pattern, response_lower, re.IGNORECASE):
                matches += 1

        return min(1.0, matches / max(1, len(UNCERTAINTY_PHRASES) // 2))

    def _check_response_length(self, response: str) -> float:
        """Check if response is suspiciously short."""
        if not response:
            return 1.0

        length = len(response)
        if length >= self.min_response_length:
            return 0.0

        return max(0.0, 1.0 - (length / self.min_response_length))

    def _check_sub_call_count(self, sub_calls: List[SubCallRecord]) -> float:
        """Check if number of sub-calls is unusually high."""
        count = len(sub_calls)
        if count <= self.max_sub_calls:
            return 0.0

        if count >= self.max_sub_calls * 2:
            return 1.0

        return (count - self.max_sub_calls) / self.max_sub_calls

    def _check_sub_call_consistency(self, sub_calls: List[SubCallRecord]) -> float:
        """Check if sub-call responses are consistent."""
        if len(sub_calls) < 2:
            return 0.0

        responses = [r.response.lower().strip() for r in sub_calls]

        unique_count = len(set(responses))
        if len(responses) > 1 and unique_count == 1:
            return 0.8

        lengths = [len(r) for r in responses]
        if len(lengths) > 1:
            avg_len = sum(lengths) / len(lengths)
            variance = sum((l - avg_len) ** 2 for l in lengths) / len(lengths)
            return min(1.0, variance / 1000)

        return 0.0

    def score_response(
        self,
        response: str,
        sub_call_records: List[SubCallRecord],
        additional_context: Optional[str] = None,
    ) -> RotScore:
        """Score a response for confidence and rot detection."""
        uncertainty_score = self._check_uncertainty_phrases(response)
        length_score = self._check_response_length(response)
        sub_call_count_score = self._check_sub_call_count(sub_call_records)
        sub_call_consistency_score = self._check_sub_call_consistency(sub_call_records)

        weights = {
            "uncertainty": 0.35,
            "response_length": 0.25,
            "sub_call_count": 0.20,
            "sub_call_consistency": 0.20,
        }

        signals = {
            "uncertainty_score": uncertainty_score,
            "length_score": length_score,
            "sub_call_count_score": sub_call_count_score,
            "sub_call_consistency_score": sub_call_consistency_score,
            "sub_call_count": len(sub_call_records),
            "response_length": len(response),
        }

        rot_score = (
            weights["uncertainty"] * uncertainty_score
            + weights["response_length"] * length_score
            + weights["sub_call_count"] * sub_call_count_score
            + weights["sub_call_consistency"] * sub_call_consistency_score
        )

        confidence = 1.0 - rot_score
        is_rotting = confidence < self.threshold

        # Build reason string
        reason_parts = []
        if uncertainty_score > 0.3:
            reason_parts.append("high uncertainty detected")
        if length_score > 0.3:
            reason_parts.append("response too short")
        if sub_call_count_score > 0.3:
            reason_parts.append(f"unusually high sub-call count ({len(sub_call_records)})")
        if sub_call_consistency_score > 0.3:
            reason_parts.append("sub-calls show inconsistency")

        # If confidence is below threshold but no individual signal triggered,
        # set a generic reason
        if is_rotting and not reason_parts:
            reason_parts.append(f"combined confidence ({confidence:.3f}) below threshold ({self.threshold})")

        if not reason_parts:
            reason = "No rot signals detected."
        else:
            reason = f"Rot detected: {', '.join(reason_parts)}"

        logger.debug(
            f"Rot score: confidence={confidence:.3f}, rot={is_rotting}, "
            f"signals={signals}, reason={reason}"
        )

        return RotScore(
            confidence=confidence,
            is_rotting=is_rotting,
            signals=signals,
            reason=reason,
        )

    def should_switch_strategy(self, rot_score: RotScore) -> bool:
        """Determine if strategy should be switched based on rot score."""
        return rot_score.is_rotting