"""Metaflow flow for parallel RLM sub-call execution.

Why this module exists:
Metaflow provides a simple way to parallelize independent tasks.
This flow takes a batch of sub-queries, processes each in parallel,
and aggregates the results.

Usage:
    python flow.py run
"""

import logging
from typing import Any, Dict, List, Optional

from metaflow import FlowSpec, Parameter, step

from src.hra_rlm.rlm.core import LLMClient
from src.hra_rlm.rlm.models import SubCallRecord
from src.hra_rlm.rlm.repl import SandboxREPL

logger = logging.getLogger(__name__)


class ParallelRLMFlow(FlowSpec):
    """Metaflow flow that processes multiple sub-queries in parallel.

    Inputs:
        sub_queries: List of (chunk_id, query, context) tuples.
        llm_client: Serialized LLM client (or parameters to instantiate one).

    Outputs:
        results: List of (chunk_id, answer, cost, tokens) for each query.
    """

    sub_queries = Parameter(
        "sub_queries",
        help="List of (chunk_id, query, context) to process.",
        default=[],
    )

    @step
    def start(self):
        """Start step: set up and fan out."""
        self.results = []
        self.next(self.process_sub_query, foreach="sub_queries")

    @step
    def process_sub_query(self):
        """Process a single sub-query in parallel."""
        chunk_id, query, context = self.input
        # Here we would invoke the LLM with the context.
        # For now, we use a placeholder.
        # In production, you'd instantiate an LLM client and SandboxREPL here.
        # Since Metaflow serializes steps, we need to ensure llm_client is available.
        # We'll use a simple approach: llm_client is passed as a global parameter.
        try:
            # This is a simplified version; actual implementation would call LLM.
            # For demonstration, we just echo back.
            answer = f"Processed {chunk_id}: {query} with context length {len(context)}"
            cost = 0.001
            tokens = 100
        except Exception as e:
            answer = f"Error: {e}"
            cost = 0.0
            tokens = 0

        self.result = {
            "chunk_id": chunk_id,
            "answer": answer,
            "cost": cost,
            "tokens": tokens,
        }
        self.next(self.join)

    @step
    def join(self, inputs):
        """Join all parallel results."""
        self.results = [inp.result for inp in inputs]
        # Sort by chunk_id to maintain order
        self.results.sort(key=lambda x: x["chunk_id"])
        self.next(self.end)

    @step
    def end(self):
        """End step."""
        pass


if __name__ == "__main__":
    ParallelRLMFlow()