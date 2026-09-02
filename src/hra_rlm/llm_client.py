"""
Real LLM Client for HRA-RLM
Supports Groq (free tier) and OpenAI (optional)
"""

import os
import time
from typing import Optional, Dict, Any
from dataclasses import dataclass

try:
    from groq import Groq
except ImportError:
    Groq = None

try:
    import openai
except ImportError:
    openai = None


@dataclass
class LLMResponse:
    """Standardized response from LLM"""
    content: str
    tokens_used: int
    cost: float
    latency_ms: float
    model: str


class LLMClient:
    """
    Unified LLM client with cost tracking
    Supports: Groq (free), OpenAI (paid)
    """
    
    def __init__(self, provider: str = "groq", model: str = None):
        """
        Args:
            provider: "groq" or "openai"
            model: model name (auto-select if None)
        """
        self.provider = provider
        self.total_tokens = 0
        self.total_cost = 0.0
        self.total_requests = 0
        
        # Groq (FREE tier)
        if provider == "groq":
            api_key = os.environ.get("GROQ_API_KEY")
            if not api_key:
                raise ValueError("GROQ_API_KEY environment variable not set")
            
            if Groq is None:
                raise ImportError("groq package not installed. Run: pip install groq")
            
            self.client = Groq(api_key=api_key)
            
            self.model = model or "openai/gpt-oss-20b"
            
            
            # Cost per 1M tokens 
            self.cost_per_1m_input = 0.0
            self.cost_per_1m_output = 0.0
        
        # OpenAI (PAID - fallback)
        elif provider == "openai":
            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY environment variable not set")
            
            if openai is None:
                raise ImportError("openai package not installed. Run: pip install openai")
            
            self.client = openai.OpenAI(api_key=api_key)
            self.model = model or "gpt-4o-mini"
            
            # Cost per 1M tokens (GPT-4o-mini)
            self.cost_per_1m_input = 0.15
            self.cost_per_1m_output = 0.60
        
        else:
            raise ValueError(f"Unknown provider: {provider}")
    
    def query(
        self, 
        prompt: str, 
        system_prompt: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 500
    ) -> LLMResponse:
        """
        Send a query to the LLM and get response with cost tracking
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        start_time = time.time()
        
        if self.provider == "groq":
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
        
        elif self.provider == "openai":
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
        
        cost = (
            (input_tokens / 1_000_000) * self.cost_per_1m_input +
            (output_tokens / 1_000_000) * self.cost_per_1m_output
        )
        
        self.total_tokens += total_tokens
        self.total_cost += cost
        self.total_requests += 1
        
        return LLMResponse(
            content=response.choices[0].message.content,
            tokens_used=total_tokens,
            cost=cost,
            latency_ms=latency_ms,
            model=self.model
        )
    
    def batch_query(self, prompts: list, **kwargs) -> list:
        return [self.query(p, **kwargs) for p in prompts]
    
    def get_stats(self) -> Dict[str, Any]:
        return {
            "total_requests": self.total_requests,
            "total_tokens": self.total_tokens,
            "total_cost": self.total_cost,
            "avg_cost_per_request": self.total_cost / max(1, self.total_requests),
            "provider": self.provider,
            "model": self.model
        }