"""Unit tests for the RLM sandbox REPL."""

import pytest

from src.hra_rlm.rlm import SandboxREPL


def test_safe_code_executes() -> None:
    """Test that safe code runs and captures stdout."""
    repl = SandboxREPL(timeout_seconds=2)

    def mock_llm(prompt: str) -> str:
        return "mock response"

    result = repl.execute(
        code="print('hello world')\nx = 5\ny = 10\nprint(x + y)",
        document="test doc",
        llm_query_callback=mock_llm,
    )

    assert result.success is True
    assert "hello world" in result.output
    assert "15" in result.output
    assert result.error is None


def test_blocked_import_os() -> None:
    """Test that importing os is blocked."""
    repl = SandboxREPL(timeout_seconds=2)

    def mock_llm(prompt: str) -> str:
        return "mock"

    result = repl.execute(
        code="import os\nos.listdir('.')",
        document="test doc",
        llm_query_callback=mock_llm,
    )

    assert result.success is False
    assert "import" in result.error or "os" in result.error


def test_blocked_file_open() -> None:
    """Test that file I/O is blocked."""
    repl = SandboxREPL(timeout_seconds=2)

    def mock_llm(prompt: str) -> str:
        return "mock"

    result = repl.execute(
        code="open('test.txt', 'w')",
        document="test doc",
        llm_query_callback=mock_llm,
    )

    assert result.success is False
    assert "open" in result.error or "builtins" in result.error


def test_timeout_infinite_loop() -> None:
    """Test that infinite loops time out cleanly."""
    repl = SandboxREPL(timeout_seconds=1)

    def mock_llm(prompt: str) -> str:
        return "mock"

    result = repl.execute(
        code="while True: pass",
        document="test doc",
        llm_query_callback=mock_llm,
    )

    assert result.success is False
    assert "timed out" in result.error


def test_llm_query_interception() -> None:
    """Test that llm_query() inside sandbox calls the callback."""
    repl = SandboxREPL(timeout_seconds=2)

    call_count = 0

    def mock_llm(prompt: str) -> str:
        nonlocal call_count
        call_count += 1
        return f"Response to: {prompt}"

    result = repl.execute(
        code="result = llm_query('What is the answer?')\nprint(result)",
        document="test doc",
        llm_query_callback=mock_llm,
    )

    assert result.success is True
    assert "Response to: What is the answer?" in result.output
    assert call_count == 1


def test_document_variable_accessible() -> None:
    """Test that the 'document' variable is injected into the sandbox."""
    repl = SandboxREPL(timeout_seconds=2)

    def mock_llm(prompt: str) -> str:
        return "mock"

    result = repl.execute(
        code="print(type(document))\nprint(str(document)[:10])",
        document="This is a long test document for RLM.",
        llm_query_callback=mock_llm,
    )

    assert result.success is True
    assert "str" in result.output
    assert "This is a " in result.output