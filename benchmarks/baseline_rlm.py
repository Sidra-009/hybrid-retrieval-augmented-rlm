"""Baseline RLM benchmark runner."""

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List

from src.hra_rlm.config.settings import get_settings
from src.hra_rlm.rlm.core import RLMAgent
from src.hra_rlm.rlm.repl import SandboxREPL

logger = logging.getLogger(__name__)


class BaselineRLMBenchmark:
    """Run baseline RLM benchmark without retrieval gating."""

    def __init__(self, use_mock: bool = True):
        self.use_mock = use_mock
        self.settings = get_settings()
        self.results: List[Dict[str, Any]] = []

    def _get_llm_client(self):
        if not self.use_mock:
            settings = get_settings()
            provider = settings.LLM_PROVIDER.lower()

            if provider == "groq":
                from groq import Groq
                client = Groq(api_key=settings.GROQ_API_KEY)

                class GroqLLMClient:
                    def __init__(self, client, model):
                        self.client = client
                        self.model = model
                        self.call_count = 0

                    def generate(self, prompt: str, system_prompt=None):
                        messages = []
                        if system_prompt:
                            messages.append({"role": "system", "content": system_prompt})
                        messages.append({"role": "user", "content": prompt})
                        response = self.client.chat.completions.create(
                            model=self.model,
                            messages=messages,
                            temperature=0.7,
                        )
                        content = response.choices[0].message.content
                        tokens = getattr(response.usage, 'total_tokens', 100)
                        cost = tokens * 0.000001
                        return content, cost, tokens

                return GroqLLMClient(client, settings.GROQ_MODEL)
            elif provider == "openai":
                raise NotImplementedError("OpenAI real client not yet implemented.")
            elif provider == "ollama":
                raise NotImplementedError("Ollama real client not yet implemented.")
            else:
                raise NotImplementedError(f"Real LLM client for {provider} not implemented.")
        else:
            class DynamicMockLLMClient:
                def __init__(self):
                    self.call_count = 0

                def generate(self, prompt: str, system_prompt=None):
                    return "```python\nprint('Baseline answer')\n```", 0.001, 100

            return DynamicMockLLMClient()

    def run_single_query(self, query: str, document_text: str) -> Dict[str, Any]:
        llm = self._get_llm_client()
        repl = SandboxREPL(timeout_seconds=self.settings.REPL_TIMEOUT_SECONDS)
        agent = RLMAgent(llm_client=llm, repl=repl, max_recursion_depth=self.settings.MAX_RECURSION_DEPTH)
        start_time = time.time()
        answer = agent.run(query=query, document=document_text)
        latency_ms = (time.time() - start_time) * 1000
        return {
            "query": query,
            "answer": answer,
            "cost": agent.total_cost_usd,
            "tokens": agent.total_tokens,
            "latency_ms": latency_ms,
            "sub_calls": len(agent.sub_call_records),
        }

    def run_benchmark(self, dataset_path: Path, output_dir: Path) -> None:
        with open(dataset_path, "r", encoding="utf-8") as f:
            dataset = json.load(f)

        all_results = []
        for doc in dataset:
            for qa in doc["qa_pairs"]:
                query = qa["question"]
                expected = qa["answer"]
                result = self.run_single_query(query, doc["text"])
                result["doc_id"] = doc["doc_id"]
                result["expected_answer"] = expected
                result["correct"] = expected.lower() in (result.get("answer", "").lower())
                all_results.append(result)
        self.results = all_results
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        out_file = output_dir / f"baseline_{timestamp}.json"
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2)
        logger.info(f"Baseline results saved to {out_file}")