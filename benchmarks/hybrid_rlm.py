"""Hybrid RLM benchmark runner."""

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List

from src.hra_rlm.config.settings import get_settings
from src.hra_rlm.rlm.core import RLMAgent
from src.hra_rlm.rlm.hybrid import HybridRLMAgent
from src.hra_rlm.rlm.repl import SandboxREPL
from src.hra_rlm.rot_detector.healer import AutoHealer
from src.hra_rlm.vectordb.embeddings import EmbeddingProvider
from src.hra_rlm.vectordb.models import Chunk
from src.hra_rlm.vectordb.store import InMemoryVectorStore

logger = logging.getLogger(__name__)


class HybridRLMBenchmark:
    def __init__(self, use_mock: bool = True, top_k: int = 5):
        self.use_mock = use_mock
        self.top_k = top_k
        self.settings = get_settings()
        self.results: Dict[str, List[Dict]] = {
            "hybrid": [],
            "autohealer": [],
            "parallel": [],
        }

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
            else:
                raise NotImplementedError(f"Real LLM client for {provider} not implemented.")
        else:
            class DynamicMockLLMClient:
                def __init__(self):
                    self.call_count = 0

                def generate(self, prompt: str, system_prompt=None):
                    return "```python\nprint('Hybrid answer')\n```", 0.001, 100

            return DynamicMockLLMClient()

    def _build_vector_store(self, documents: List[Dict]) -> InMemoryVectorStore:
        store = InMemoryVectorStore()
        embedder = EmbeddingProvider()
        for doc in documents:
            chunk = Chunk(
                chunk_id=doc["doc_id"],
                text=doc["text"],
                embedding=embedder.embed_query(doc["text"]),
                metadata={"doc_id": doc["doc_id"]},
            )
            store.add(chunk)
        return store

    def run_single_query(self, query: str, expected: str, store: InMemoryVectorStore,
                         config: str = "hybrid") -> Dict[str, Any]:
        llm = self._get_llm_client()
        repl = SandboxREPL(timeout_seconds=self.settings.REPL_TIMEOUT_SECONDS)
        rlm = RLMAgent(llm_client=llm, repl=repl, max_recursion_depth=self.settings.MAX_RECURSION_DEPTH)

        hybrid = HybridRLMAgent(
            vector_store=store,
            rlm_agent=rlm,
            embedding_provider=EmbeddingProvider(),
            top_k=self.top_k,
            retrieval_strategy="fixed_k",
        )

        if config == "autohealer":
            agent = AutoHealer(hybrid_agent=hybrid)
        elif config == "parallel":
            from src.hra_rlm.orchestrator.parallel_hybrid import ParallelHybridRLMAgent
            agent = ParallelHybridRLMAgent(hybrid_agent=hybrid, metaflow_enabled=False)
            sub_queries = [(store.size, query, store.size)]
            start_time = time.time()
            result = agent.run(query=query, sub_queries=sub_queries)
            latency_ms = (time.time() - start_time) * 1000
            return {
                "query": query,
                "answer": result["answer"],
                "expected_answer": expected,
                "correct": expected.lower() in result["answer"].lower(),
                "cost": result["total_cost"],
                "tokens": result["total_tokens"],
                "latency_ms": latency_ms,
                "sub_calls": len(result.get("sub_call_results", [])),
            }
        else:
            agent = hybrid

        start_time = time.time()
        result = agent.run(query=query)
        latency_ms = (time.time() - start_time) * 1000

        return {
            "query": query,
            "answer": result.get("answer", ""),
            "expected_answer": expected,
            "correct": expected.lower() in result.get("answer", "").lower(),
            "cost": result.get("total_cost", 0.0),
            "tokens": result.get("total_tokens", 0),
            "latency_ms": latency_ms,
            "sub_calls": len(result.get("sub_calls", [])),
        }

    def run_benchmark(self, dataset_path: Path, output_dir: Path) -> None:
        with open(dataset_path, "r", encoding="utf-8") as f:
            dataset = json.load(f)

        store = self._build_vector_store(dataset)

        for config in ["hybrid", "autohealer", "parallel"]:
            results = []
            for doc in dataset:
                for qa in doc["qa_pairs"]:
                    result = self.run_single_query(qa["question"], qa["answer"], store, config)
                    result["doc_id"] = doc["doc_id"]
                    results.append(result)
            self.results[config] = results

        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        for config, results in self.results.items():
            out_file = output_dir / f"hybrid_{config}_{timestamp}.json"
            with open(out_file, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2)
            logger.info(f"Hybrid {config} results saved to {out_file}")