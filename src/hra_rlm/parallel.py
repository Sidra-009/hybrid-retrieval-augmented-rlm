"""
Parallel Execution Module for HRA-RLM
Runs retrieval and generation steps in parallel, and measures the
benefit correctly: as total batch wall-clock time, not per-query latency.

The README's own Ablation Notes flagged this: per-query latency doesn't
capture what parallel execution is meant to speed up. This version adds
a `run_batch_comparison` helper that runs the same batch of tasks both
sequentially and concurrently and reports the real wall-clock speedup.
"""

from typing import List, Dict, Any, Callable
import concurrent.futures
import time


class ParallelExecutor:
    """Parallel execution pipeline"""

    def __init__(self, max_workers: int = 4):
        self.max_workers = max_workers
        self.execution_time = 0.0

    def execute_parallel(self, tasks: List[Dict]) -> List[Dict]:
        """
        Execute multiple tasks concurrently.

        Args:
            tasks: List of task dicts with 'function' and 'args'

        Returns:
            List of results (order not guaranteed to match input order,
            since results are collected as they complete)
        """
        start_time = time.time()
        results = []

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_task = {
                executor.submit(task['function'], *task.get('args', [])): task
                for task in tasks
            }
            for future in concurrent.futures.as_completed(future_to_task):
                try:
                    result = future.result()
                    results.append({"success": True, "result": result, "task": future_to_task[future]})
                except Exception as e:
                    results.append({"success": False, "error": str(e), "task": future_to_task[future]})

        self.execution_time = time.time() - start_time
        return results

    def execute_sequential(self, tasks: List[Dict]) -> List[Dict]:
        """Run the same tasks one at a time, for a fair baseline comparison."""
        start_time = time.time()
        results = []
        for task in tasks:
            try:
                result = task['function'](*task.get('args', []))
                results.append({"success": True, "result": result, "task": task})
            except Exception as e:
                results.append({"success": False, "error": str(e), "task": task})
        self.execution_time = time.time() - start_time
        return results

    def run_batch_comparison(self, tasks: List[Dict]) -> Dict[str, Any]:
        """
        Run the identical batch both sequentially and in parallel, and
        report true wall-clock time for each. This is the metric the
        original README's Ablation Notes said was missing: batch-level
        speedup rather than single-query latency.

        Note: tasks that share mutable state (e.g. a rate-limited API
        client) may not show the full theoretical speedup here — network
        rate limits, not the executor, become the bottleneck. Report both
        numbers rather than only the favorable one.
        """
        sequential_results = self.execute_sequential(tasks)
        sequential_time = self.execution_time

        parallel_results = self.execute_parallel(tasks)
        parallel_time = self.execution_time

        speedup = sequential_time / parallel_time if parallel_time > 0 else float("nan")

        return {
            "sequential_wall_clock_s": sequential_time,
            "parallel_wall_clock_s": parallel_time,
            "speedup_x": speedup,
            "num_tasks": len(tasks),
            "max_workers": self.max_workers,
            "sequential_results": sequential_results,
            "parallel_results": parallel_results,
        }

    def get_stats(self) -> Dict[str, Any]:
        return {
            "execution_time_ms": self.execution_time * 1000,
            "max_workers": self.max_workers,
            "parallel_enabled": True
        }