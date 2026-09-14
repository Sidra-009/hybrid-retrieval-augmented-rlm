"""
Real LLM Client for HRA-RLM
Supports Groq (free tier) and OpenAI (optional)

Cost note:
Groq's free tier reports $0.00 for every call, which makes real cost
comparisons impossible. This client still tracks that as `actual_cost`
(what you were billed), but also reports `estimated_cost` computed
against a published reference price for an equivalent model, so cost
comparisons in benchmarks are meaningful rather than trivially zero.
Update REFERENCE_PRICING if you want to compare against a different
provider's list price.
"""

import os
import time
from typing import Optional, Dict, Any
from dataclasses import dataclass

try:
    from dotenv import load_dotenv
    load_dotenv()  # reads .env from project root (or nearest parent) into os.environ
except ImportError:
    pass  # python-dotenv not installed; rely on real environment variables instead

try:
    from groq import Groq
except ImportError:
    Groq = None

try:
    import openai
except ImportError:
    openai = None


# Reference $ / 1M tokens, used ONLY to estimate what a metered provider
# would have charged for the same token counts. These are illustrative
# reference prices, not Groq's actual billing (Groq free tier = $0).
# Update to the current published rate before citing this number anywhere.
REFERENCE_PRICING = {
    "openai/gpt-oss-20b": {"input": 0.10, "output": 0.30},
    "llama-3.1-8b-instant": {"input": 0.05, "output": 0.08},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
}


@dataclass
class LLMResponse:
    """Standardized response from LLM"""
    content: str
    tokens_used: int
    actual_cost: float
    estimated_cost: float
    latency_ms: float
    model: str


class LLMClient:
    """
    Unified LLM client with cost tracking.
    Supports: Groq (free), OpenAI (paid)
    """

    def __init__(self, provider: str = "groq", model: str = None):
        self.provider = provider
        self.total_tokens = 0
        self.total_actual_cost = 0.0
        self.total_estimated_cost = 0.0
        self.total_requests = 0

        if provider == "groq":
            api_key = os.environ.get("GROQ_API_KEY")
            if not api_key:
                raise ValueError("GROQ_API_KEY environment variable not set")
            if Groq is None:
                raise ImportError("groq package not installed. Run: pip install groq")

            self.client = Groq(api_key=api_key)
            self.model = model or "openai/gpt-oss-20b"

            # Groq free tier: actual billed cost is always $0.
            self.cost_per_1m_input = 0.0
            self.cost_per_1m_output = 0.0

        elif provider == "openai":
            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY environment variable not set")
            if openai is None:
                raise ImportError("openai package not installed. Run: pip install openai")

            self.client = openai.OpenAI(api_key=api_key)
            self.model = model or "gpt-4o-mini"

            pricing = REFERENCE_PRICING.get(self.model, {"input": 0.15, "output": 0.60})
            self.cost_per_1m_input = pricing["input"]
            self.cost_per_1m_output = pricing["output"]

        else:
            raise ValueError(f"Unknown provider: {provider}")

        # Reference pricing used for the *estimated* cost figure, independent
        # of what the provider actually charged.
        ref = REFERENCE_PRICING.get(self.model, {"input": 0.10, "output": 0.30})
        self.ref_cost_per_1m_input = ref["input"]
        self.ref_cost_per_1m_output = ref["output"]

    def query(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 800
    ) -> LLMResponse:
        """
        Note: openai/gpt-oss-20b on Groq spends a chunk of the token
        budget on internal reasoning before writing the final answer
        (observed ~150-250 reasoning tokens for short prompts). If
        max_tokens is too low, the response can come back empty even
        though the call "succeeded" (finish_reason='stop'). 800 gives
        enough headroom for reasoning + a short final answer.
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        start_time = time.time()

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )

        latency_ms = (time.time() - start_time) * 1000

        usage = response.usage
        input_tokens = usage.prompt_tokens
        output_tokens = usage.completion_tokens
        total_tokens = input_tokens + output_tokens

        actual_cost = (
            (input_tokens / 1_000_000) * self.cost_per_1m_input +
            (output_tokens / 1_000_000) * self.cost_per_1m_output
        )
        estimated_cost = (
            (input_tokens / 1_000_000) * self.ref_cost_per_1m_input +
            (output_tokens / 1_000_000) * self.ref_cost_per_1m_output
        )

        self.total_tokens += total_tokens
        self.total_actual_cost += actual_cost
        self.total_estimated_cost += estimated_cost
        self.total_requests += 1

        return LLMResponse(
            content=response.choices[0].message.content,
            tokens_used=total_tokens,
            actual_cost=actual_cost,
            estimated_cost=estimated_cost,
            latency_ms=latency_ms,
            model=self.model
        )

    def batch_query(self, prompts: list, **kwargs) -> list:
        return [self.query(p, **kwargs) for p in prompts]

    def get_stats(self) -> Dict[str, Any]:
        return {
            "total_requests": self.total_requests,
            "total_tokens": self.total_tokens,
            "total_actual_cost": self.total_actual_cost,
            "total_estimated_cost": self.total_estimated_cost,
            "avg_estimated_cost_per_request": self.total_estimated_cost / max(1, self.total_requests),
            "provider": self.provider,
            "model": self.model
        }