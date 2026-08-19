"""Unit tests for the RLMAgent recursive reasoning core."""

import pytest

from src.hra_rlm.rlm.core import MockLLMClient, RLMAgent
from src.hra_rlm.rlm.models import ExecutionResult
from src.hra_rlm.rlm.repl import SandboxREPL


class MockSandboxREPL(SandboxREPL):
    """Mock sandbox that returns predefined results."""

    def __init__(self, results: list, timeout_seconds: int = 2):
        super().__init__(timeout_seconds=timeout_seconds)
        self.results = results
        self.call_count = 0

    def execute(self, code: str, document: str, llm_query_callback) -> ExecutionResult:
        if self.call_count < len(self.results):
            result = self.results[self.call_count]
        else:
            result = ExecutionResult(success=True, output="mock output")
        self.call_count += 1
        return result


def test_loop_terminates_on_final_answer() -> None:
    llm_responses = [
        "```python\nprint('Hello from RLM')\n```",
        "This is the final answer to the user's question.",
    ]
    llm = MockLLMClient(responses=llm_responses)
    repl = MockSandboxREPL(results=[
        ExecutionResult(success=True, output="Hello from RLM"),
    ])
    agent = RLMAgent(llm_client=llm, repl=repl, max_recursion_depth=3)
    result = agent.run(query="What is the answer?", document="Test document")
    assert result == "This is the final answer to the user's question."
    assert agent.total_cost_usd > 0


def test_respects_max_recursion_depth() -> None:
    llm_responses = [
        "```python\nprint('Attempt 1')\n```",
        "```python\nprint('Attempt 2')\n```",
        "```python\nprint('Attempt 3')\n```",
        "```python\nprint('Attempt 4')\n```",
    ]
    llm = MockLLMClient(responses=llm_responses)
    repl = MockSandboxREPL(results=[
        ExecutionResult(success=False, output="", error="Mock error"),
        ExecutionResult(success=False, output="", error="Mock error"),
        ExecutionResult(success=False, output="", error="Mock error"),
        ExecutionResult(success=False, output="", error="Mock error"),
    ])
    agent = RLMAgent(llm_client=llm, repl=repl, max_recursion_depth=3)
    result = agent.run(query="What is the answer?", document="Test document")
    assert "max recursion depth" in result.lower()


def test_sub_call_records_accumulate() -> None:
    # Override the agent's _handle_sub_call to record without real LLM
    llm = MockLLMClient(responses=[
        "```python\nresult = llm_query('What is 2+2?')\nprint(result)\n```",
        "The final answer is 4.",
    ])
    repl = SandboxREPL(timeout_seconds=2)
    agent = RLMAgent(llm_client=llm, repl=repl, max_recursion_depth=3)

    # Mock the sub-call handler to record and return a fixed response
    recorded_prompts = []

    def mock_sub(prompt: str) -> str:
        recorded_prompts.append(prompt)
        return "4"

    agent._handle_sub_call = mock_sub  # type: ignore

    # Execute
    result = agent.run(query="What is 2+2?", document="Test document")
    assert len(recorded_prompts) >= 1
    assert recorded_prompts[0] == "What is 2+2?"
    assert result == "The final answer is 4."


def test_code_extraction() -> None:
    llm_responses = [
        "Here is my code:\n```python\nprint('hello')\n```\nThat should work.",
        "This is the final answer.",
    ]
    llm = MockLLMClient(responses=llm_responses)
    repl = MockSandboxREPL(results=[
        ExecutionResult(success=True, output="hello"),
    ])
    agent = RLMAgent(llm_client=llm, repl=repl, max_recursion_depth=2)
    result = agent.run(query="Test query", document="Test document")
    assert result == "This is the final answer."


def test_error_handling_in_loop() -> None:
    llm_responses = [
        "```python\nprint('Working code')\n```",
        "Final answer after error recovery.",
    ]
    llm = MockLLMClient(responses=llm_responses)
    repl = MockSandboxREPL(results=[
        ExecutionResult(success=False, output="", error="Division by zero"),
        ExecutionResult(success=True, output="Success after retry"),
    ])
    agent = RLMAgent(llm_client=llm, repl=repl, max_recursion_depth=3)
    result = agent.run(query="Test query", document="Test document")
    assert result == "Final answer after error recovery."