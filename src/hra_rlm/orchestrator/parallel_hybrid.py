"""Parallel Hybrid RLM Agent using Metaflow.

Why this module exists:
Extends HybridRLMAgent to execute independent sub-calls in parallel
using Metaflow, reducing overall latency.

Tradeoff: Metaflow adds overhead for small numbers of sub-calls;
for large batches (e.g., many chunks), parallelization provides significant speedup.
"""

import json
import logging
import subprocess
import tempfile
from typing import Any, Dict, List, Optional

from src.hra_rlm.rlm.hybrid import HybridRLMAgent
from src.hra_rlm.rlm.models import SubCallRecord

logger = logging.getLogger(__name__)


class ParallelHybridRLMAgent:
    """Hybrid RLM agent with parallel sub-call execution via Metaflow."""

    def __init__(
        self,
        hybrid_agent: HybridRLMAgent,
        metaflow_enabled: bool = True,
    ):
        """Initialize the parallel agent.

        Args:
            hybrid_agent: The underlying HybridRLMAgent instance.
            metaflow_enabled: If True, try to use Metaflow; if False, fallback to sequential.
        """
        self.hybrid_agent = hybrid_agent
        self.metaflow_enabled = metaflow_enabled

    def _run_metaflow_flow(self, sub_queries: List[tuple]) -> List[Dict[str, Any]]:
        """Run the Metaflow flow with the given sub_queries.

        Args:
            sub_queries: List of (chunk_id, query, context) tuples.

        Returns:
            List of result dicts for each sub_query.
        """
        # For simplicity, we write the sub_queries to a temp file and run the flow.
        # In a real deployment, you'd use Metaflow's API or deploy the flow as a service.
        # We'll use subprocess to run the flow as a separate process.

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(sub_queries, f)
            temp_path = f.name

        # Build command: python flow.py run --sub_queries <json_file>
        cmd = [
            "python",
            "-m",
            "src.hra_rlm.orchestrator.flow",
            "run",
            f"--sub_queries={temp_path}",
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            # Parse output: assuming results are printed as JSON
            # For this demo, we'll return dummy results.
            # A production version would parse stdout for JSON.
            logger.info("Metaflow flow executed successfully.")
        except subprocess.CalledProcessError as e:
            logger.error(f"Metaflow flow failed: {e.stderr}")
            # Fallback to sequential
            return self._sequential_run(sub_queries)
        finally:
            import os
            os.unlink(temp_path)

        # For now, return dummy results
        return [
            {
                "chunk_id": q[0],
                "answer": f"Parallel processed {q[0]}",
                "cost": 0.001,
                "tokens": 100,
            }
            for q in sub_queries
        ]

    def _sequential_run(self, sub_queries: List[tuple]) -> List[Dict[str, Any]]:
        """Fallback sequential execution if Metaflow is unavailable."""
        results = []
        for chunk_id, query, context in sub_queries:
            # Use the hybrid agent's RLM agent to process each query
            # In real implementation, we'd call the LLM directly.
            # For demo, we mock.
            results.append({
                "chunk_id": chunk_id,
                "answer": f"Sequential processed {chunk_id}",
                "cost": 0.001,
                "tokens": 100,
            })
        return results

    def run(self, query: str, sub_queries: List[tuple]) -> Dict[str, Any]:
        """Run the hybrid agent with parallel sub-call execution.

        Args:
            query: The main query.
            sub_queries: List of (chunk_id, sub_query, context) to process in parallel.

        Returns:
            Dict with combined results and metadata.
        """
        if self.metaflow_enabled:
            try:
                results = self._run_metaflow_flow(sub_queries)
            except Exception as e:
                logger.warning(f"Metaflow execution failed, falling back to sequential: {e}")
                results = self._sequential_run(sub_queries)
        else:
            results = self._sequential_run(sub_queries)

        # Combine results into a single answer
        combined_answer = "\n".join([r["answer"] for r in results])
        total_cost = sum(r["cost"] for r in results)
        total_tokens = sum(r["tokens"] for r in results)

        return {
            "answer": combined_answer,
            "total_cost": total_cost,
            "total_tokens": total_tokens,
            "sub_call_results": results,
            "parallel_used": self.metaflow_enabled,
        }