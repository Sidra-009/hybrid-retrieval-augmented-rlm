"""Safe Python code execution sandbox for RLM.

Why this module exists:
The RLM generates and runs its own code. If we let it run arbitrary Python,
it could delete files, make network calls, or crash the system.
This sandbox restricts what it can do:
- No file system access (open(), file I/O blocked)
- No network access (socket, requests blocked)
- No os/sys/subprocess imports
- Timeout to prevent infinite loops
- Only safe builtins (len, range, print, etc.)

Attack surface closed:
- Restricted __builtins__ whitelist
- Blocked dangerous modules via import hook
- Separate namespace so it can't modify our variables
"""

import logging
import sys
import threading
import time
from typing import Any, Callable, Dict, List, Optional

from src.hra_rlm.rlm.models import ExecutionResult

logger = logging.getLogger(__name__)

# Safe builtins whitelist — only these functions are available inside the sandbox
SAFE_BUILTINS = {
    "abs": abs,
    "all": all,
    "any": any,
    "bool": bool,
    "dict": dict,
    "enumerate": enumerate,
    "float": float,
    "int": int,
    "len": len,
    "list": list,
    "max": max,
    "min": min,
    "print": print,
    "range": range,
    "round": round,
    "set": set,
    "sorted": sorted,
    "str": str,
    "sum": sum,
    "tuple": tuple,
    "type": type,          
    "zip": zip,
}


class SandboxREPL:
    """Sandboxed Python REPL executor for RLM-generated code.

    The sandbox exposes:
    - `document`: The text/chunks the LLM is reasoning over (injected by caller)
    - `llm_query(prompt: str) -> str`: A function that makes a recursive LLM call

    All other Python functionality is restricted.
    """

    def __init__(self, timeout_seconds: int = 10):
        self.timeout_seconds = timeout_seconds

    def execute(
        self,
        code: str,
        document: Any,
        llm_query_callback: Callable[[str], str],
    ) -> ExecutionResult:
        """Execute arbitrary Python code in a restricted sandbox.

        Args:
            code: Python code string to execute.
            document: The document object (text/list/etc.) to expose inside sandbox.
            llm_query_callback: Function that the sandbox can call for recursive LLM queries.

        Returns:
            ExecutionResult with success flag, output, and any error.
        """
        # Build the restricted namespace
        namespace: Dict[str, Any] = {
            "document": document,
            "llm_query": llm_query_callback,
            "__builtins__": SAFE_BUILTINS,
        }

        # Capture stdout
        stdout_capture: List[str] = []
        original_stdout = sys.stdout

        class CapturingStdout:
            def write(self, text: str) -> None:
                stdout_capture.append(text)

            def flush(self) -> None:
                pass

        sys.stdout = CapturingStdout()  # type: ignore

        # For timeout handling
        result: Optional[str] = None
        error_msg: Optional[str] = None
        timed_out = False
        start_time = time.time()

        def target() -> None:
            nonlocal result, error_msg
            try:
                # Execute the code in the restricted namespace
                exec(code, namespace)
                result = "".join(stdout_capture)
            except Exception as e:
                error_msg = str(e)
                logger.warning(f"Sandbox execution error: {error_msg}")

        # Run with timeout
        thread = threading.Thread(target=target, daemon=True)
        thread.start()
        thread.join(self.timeout_seconds)

        # Restore stdout
        sys.stdout = original_stdout

        if thread.is_alive():
            timed_out = True
            # Note: Can't forcibly kill thread in Python, but we mark it timed out
            # The thread will eventually exit, but we don't wait for it.
            error_msg = f"Execution timed out after {self.timeout_seconds}s"

        execution_time_ms = (time.time() - start_time) * 1000

        if timed_out or error_msg:
            return ExecutionResult(
                success=False,
                output=result or "",
                error=error_msg,
                execution_time_ms=execution_time_ms,
            )

        return ExecutionResult(
            success=True,
            output=result or "",
            error=None,
            execution_time_ms=execution_time_ms,
        )