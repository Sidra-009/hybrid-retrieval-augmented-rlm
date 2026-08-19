"""Root RLM recursive reasoning agent.

Why this module exists:
This is the core of the RLM — it orchestrates the recursive loop where the LLM
generates code, the sandbox executes it, and the result is fed back to the LLM.
This continues until a final answer is produced or max recursion depth is hit.
"""

import logging
import re
import time
from typing import Any, Callable, List, Optional, Tuple

from tenacity import retry, stop_after_attempt, wait_exponential

from src.hra_rlm.config.settings import get_settings
from src.hra_rlm.rlm.models import ExecutionResult, SubCallRecord
from src.hra_rlm.rlm.prompts import FINAL_ANSWER_PROMPT, RETRY_PROMPT, SYSTEM_PROMPT
from src.hra_rlm.rlm.repl import SandboxREPL

logger = logging.getLogger(__name__)


class LLMClient:
    """Abstract interface for an LLM client."""

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> Tuple[str, float, int]:
        raise NotImplementedError


class MockLLMClient(LLMClient):
    """Mock LLM client for testing."""

    def __init__(self, responses: List[str]):
        self.responses = responses
        self.call_count = 0
        self.cost_usd = 0.0
        self.tokens_used = 0

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> Tuple[str, float, int]:
        if self.call_count < len(self.responses):
            response = self.responses[self.call_count]
        else:
            response = "print('Fallback mock response')"
        self.call_count += 1
        self.cost_usd += 0.001
        self.tokens_used += 100
        return response, 0.001, 100


class RLMAgent:
    """Root RLM agent that orchestrates the recursive reasoning loop."""

    def __init__(
        self,
        llm_client: LLMClient,
        repl: Optional[SandboxREPL] = None,
        max_recursion_depth: Optional[int] = None,
    ):
        self.llm_client = llm_client
        self.repl = repl or SandboxREPL(timeout_seconds=get_settings().REPL_TIMEOUT_SECONDS)
        self.max_recursion_depth = max_recursion_depth or get_settings().MAX_RECURSION_DEPTH

        self.sub_call_records: List[SubCallRecord] = []
        self.total_cost_usd = 0.0
        self.total_tokens = 0
        self.total_latency_ms = 0.0

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=5))
    def _generate_code(self, query: str, error: Optional[str] = None, previous_code: Optional[str] = None) -> str:
        if error and previous_code:
            prompt = RETRY_PROMPT.format(
                error=error,
                original_query=query,
                previous_code=previous_code,
            )
        else:
            prompt = SYSTEM_PROMPT.format(query=query)

        response, cost, tokens = self.llm_client.generate(prompt, system_prompt=None)
        self.total_cost_usd += cost
        self.total_tokens += tokens

        code = self._extract_code(response)
        return code or response

    def _extract_code(self, response: str) -> Optional[str]:
        patterns = [
            r"```python\s*\n(.*?)\n```",
            r"```\s*\n(.*?)\n```",
        ]
        for pattern in patterns:
            match = re.search(pattern, response, re.DOTALL | re.IGNORECASE)
            if match:
                return match.group(1).strip()
        return None

    def _get_final_answer(self, query: str, execution_output: str) -> str:
        prompt = FINAL_ANSWER_PROMPT.format(
            execution_output=execution_output,
            original_query=query,
        )
        response, cost, tokens = self.llm_client.generate(prompt, system_prompt=None)
        self.total_cost_usd += cost
        self.total_tokens += tokens
        return response

    def _handle_sub_call(self, prompt: str) -> str:
        start_time = time.time()
        response, cost, tokens = self.llm_client.generate(prompt, system_prompt=None)
        latency_ms = (time.time() - start_time) * 1000

        record = SubCallRecord(
            prompt=prompt,
            response=response,
            tokens_used=tokens,
            cost_usd=cost,
            latency_ms=latency_ms,
        )
        self.sub_call_records.append(record)
        self.total_cost_usd += cost
        self.total_tokens += tokens
        self.total_latency_ms += latency_ms
        return response

    def run(self, query: str, document: Any) -> Any:
        logger.info(f"Starting RLM agent with query: {query[:100]}...")
        start_time = time.time()
        code = None
        execution_result: Optional[ExecutionResult] = None
        error = None
        previous_code = None

        for depth in range(self.max_recursion_depth):
            try:
                code = self._generate_code(query, error, previous_code)
            except Exception as e:
                logger.error(f"Code generation failed: {e}")
                return f"Error: Failed to generate code: {e}"

            try:
                execution_result = self.repl.execute(
                    code=code,
                    document=document,
                    llm_query_callback=self._handle_sub_call,
                )
            except Exception as e:
                execution_result = ExecutionResult(
                    success=False,
                    output="",
                    error=str(e),
                    execution_time_ms=0,
                )

            if execution_result.success and execution_result.output:
                try:
                    final_answer = self._get_final_answer(query, execution_result.output)
                    total_latency_ms = (time.time() - start_time) * 1000
                    logger.info(f"Completed at depth {depth+1}, cost ${self.total_cost_usd:.6f}")
                    return final_answer
                except Exception as e:
                    return f"Error: Failed to get final answer: {e}"

            error = execution_result.error or "Unknown execution error"
            previous_code = code
            logger.warning(f"Execution failed at depth {depth+1}: {error[:200]}")

        logger.warning(f"Max recursion depth ({self.max_recursion_depth}) reached")
        if execution_result and execution_result.success:
            return execution_result.output
        return f"Max recursion depth reached. Last error: {error}"