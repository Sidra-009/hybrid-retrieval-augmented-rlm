"""Orchestrator module for parallel execution using Metaflow.

Why this module exists:
Enables parallel fan-out of independent RLM sub-calls using Metaflow,
reducing overall latency compared to sequential execution.
"""

from src.hra_rlm.orchestrator.parallel_hybrid import ParallelHybridRLMAgent

__all__ = ["ParallelHybridRLMAgent"]