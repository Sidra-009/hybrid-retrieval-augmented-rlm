"""
Parallel Execution Module for HRA-RLM
Runs retrieval and generation steps in parallel
"""

from typing import List, Dict, Any
import concurrent.futures
import time


class ParallelExecutor:
    """Parallel execution pipeline"""
    
    def __init__(self, max_workers: int = 4):
        self.max_workers = max_workers
        self.execution_time = 0.0
    
    def execute_parallel(self, tasks: List[Dict]) -> List[Dict]:
        """
        Execute multiple tasks in parallel
        
        Args:
            tasks: List of task dicts with 'function' and 'args'
        
        Returns:
            List of results
        """
        start_time = time.time()
        results = []
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_task = {
                executor.submit(task['function'], *task.get('args', [])): task
                for task in tasks
            }
            
            # Collect results
            for future in concurrent.futures.as_completed(future_to_task):
                try:
                    result = future.result()
                    results.append({
                        "success": True,
                        "result": result,
                        "task": future_to_task[future]
                    })
                except Exception as e:
                    results.append({
                        "success": False,
                        "error": str(e),
                        "task": future_to_task[future]
                    })
        
        self.execution_time = time.time() - start_time
        return results
    
    def get_stats(self) -> Dict[str, Any]:
        """Return execution statistics"""
        return {
            "execution_time_ms": self.execution_time * 1000,
            "max_workers": self.max_workers,
            "parallel_enabled": True
        }