"""Context Rot Detector and Auto-Healer module.

Why this module exists:
Detects when the model's effective accuracy/confidence is degrading as context grows
("context rot"), and automatically switches strategy to mitigate it.
"""

from src.hra_rlm.rot_detector.detector import ContextRotDetector, RotScore
from src.hra_rlm.rot_detector.healer import AutoHealer

__all__ = ["ContextRotDetector", "RotScore", "AutoHealer"]