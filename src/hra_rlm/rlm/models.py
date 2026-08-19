"""Pydantic models for RLM execution tracking.

Why this module exists:
Defines the data structures for tracking what the RLM does:
- ExecutionResult: What happened when code ran (success/fail/output/error)
- SubCallRecord: Every recursive llm_query call (for cost/latency auditing)
"""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class ExecutionResult(BaseModel):
    """Result of executing a code block in the sandbox."""

    success: bool
    output: str
    error: Optional[str] = None
    execution_time_ms: float = Field(default=0.0)


class SubCallRecord(BaseModel):
    """Record of a single llm_query() call made from inside the sandbox."""

    timestamp: datetime = Field(default_factory=datetime.now)
    prompt: str
    response: str
    tokens_used: Optional[int] = None
    cost_usd: float = Field(default=0.0)
    latency_ms: float = Field(default=0.0)