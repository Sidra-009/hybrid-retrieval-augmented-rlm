#!/usr/bin/env python3
"""
Run HRA-RLM benchmarks with real or mock LLM calls.

This version actually differentiates the four methods instead of calling
the LLM the same way for all of them:

  Baseline RLM          -> full source document passed as context every time
  Hybrid (fixed_k)      -> retrieval-gated: only top-k relevant chunks passed,
                           using the real HNSW semantic retriever
  Hybrid + AutoHealer    -> fixed_k retrieval, but retries with full context
                           if the gated answer looks degraded (context rot)
  Hybrid + Parallel     -> fixed_k retrieval, but retrieval + generation for
                           all queries in a batch run concurrently

Additions in this version:
  - GlobalPacer: enforces a minimum spacing between real API calls (across
    all threads) so we stay under Groq free-tier TPM limits instead of
    bursting into 429s.
  - Mock-fallback contamination tracking: if a real call fails and falls
    back to a mock answer, that is now counted, printed, and surfaced as
    a WARNING per method, instead of silently blending into "real" results.
  - batch_wall_clock_s: true wall-clock time for the whole batch of queries
    per method, in addition to the old per-query p50 latency.
  - --trials N: run the full 4-method suite N times and report mean +
    [min-max] range per method.
  - Configurable --top-k and --embedding-model.
  - A longer, 10-section (~2000-word) source document instead of the
    single ~300-word abstract, so top_k actually has room to matter.
  - A 27-question dataset spanning all 10 sections (was 10 questions
    against one short paragraph).
  - Dual scoring: the original keyword-overlap heuristic PLUS a semantic
    embedding-similarity score (using the same embedding model as
    retrieval), reported side by side rather than replacing one with the
    other. The embedding score is used as the primary "accuracy" metric
    when the retriever is running in real (non-fallback) semantic mode;
    otherwise keyword-overlap remains the only signal.
  - --num-questions N: use a subset of the dataset (useful for cheap
    --use-mock smoke-tests of code changes before spending real API quota
    on the full 27-question set).
"""

import json
import time
import argparse
import random
import re
import threading
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import statistics

import numpy as np

RATE_LIMIT_RETRY_PATTERN = re.compile(r"try again in ([\d.]+)s", re.IGNORECASE)
MAX_RETRIES_ON_RATE_LIMIT = 4
EMBEDDING_MATCH_THRESHOLD = 0.55

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.hra_rlm.llm_client import LLMClient
from src.hra_rlm.retriever import RetrievalGating  # real HNSW retriever


# ---------------------------------------------------------------------------
# Source document: 10 distinct sections (~2000 words total) so that top_k
# retrieval has to genuinely choose between passages, unlike the earlier
# ~300-word single-paragraph version where top_k=7 retrieved almost
# everything anyway.
# ---------------------------------------------------------------------------
SOURCE_DOCUMENT = """
Large language models increasingly serve as the reasoning engine behind long-document
question answering, multi-hop retrieval pipelines, and agentic workflows that operate
over large text corpora. A common architectural pattern for this setting is the
recursive language model, which improves answer quality by iteratively re-processing
the available context across multiple reasoning passes, using each pass to refine or
verify the answer produced by the previous one. While this iterative strategy improves
accuracy on complex questions, it comes at a steep computational cost: every recursive
pass re-consumes the entire context window, so token usage, inference latency, and
dollar cost all scale with the number of passes rather than staying fixed. As document
length grows or as more reasoning depth is required, this cost grows correspondingly,
which limits how practical purely recursive reasoning is for real-world deployments
operating under a fixed compute or cost budget. The central design question this
project investigates is whether most of that re-processing cost can be avoided without
giving up the accuracy benefits of recursion, by being selective about which part of
the document is actually re-processed at each step rather than always re-processing
all of it.

The retrieval-gated recursion controller is the component responsible for deciding
what portion of the source document is passed to the language model at each reasoning
step. Rather than forwarding the entire document on every pass, the controller first
encodes the document into sentence-boundary-aware chunks and builds an approximate
nearest-neighbor index over dense embeddings of those chunks using the HNSW algorithm.
When a sub-query arrives, the controller embeds the query with the same embedding
model, searches the index, and retrieves only the top-k most semantically similar
chunks, which are then concatenated and passed to the language model as the working
context for that step. The value of k is a tunable parameter that directly controls
the trade-off between token savings and answer quality: a small k discards more of the
document and saves more tokens, but risks omitting a chunk that is actually needed to
answer the question; a large k retains more of the document, recovers accuracy, but
proportionally reduces the token savings the mechanism is meant to provide. On short
documents, a large k can retrieve nearly the entire document, at which point the
token-reduction benefit largely disappears even though accuracy is preserved.

The AutoHealer component exists to catch the failure mode where retrieval gating
discards a chunk of the document that turns out to be necessary to answer the current
sub-query, a phenomenon the project refers to as context rot. After the language model
produces an answer using the retrieved, reduced context, AutoHealer inspects that
answer for signs of degradation: an unusually short response, or the presence of
hedging language such as expressions of uncertainty or an explicit statement that the
answer cannot be determined from the given context. If degradation is detected,
AutoHealer triggers a second generation pass, this time supplying the full, unfiltered
source document as context instead of the retrieved subset, and returns that healed
answer instead of the original one. Because this fallback pass reprocesses the entire
document, it carries the same token and latency cost as a single baseline pass, so
AutoHealer should only trigger occasionally, on the subset of questions where retrieval
genuinely failed, rather than routinely, or its cost savings relative to the baseline
erode. Whether AutoHealer meaningfully improves accuracy without eroding those savings
depends heavily on how often it actually fires, which in turn depends on how
well-tuned the retrieval step is and how demanding the underlying questions are.

The parallel execution pipeline is designed to reduce the wall-clock time needed to
answer a batch of independent queries, as opposed to reducing the cost or token usage
of any single query. Because different sub-queries against the same document do not
depend on each other's answers, their retrieval and generation steps can in principle
be issued concurrently rather than one after another, using a thread pool to dispatch
several requests to the language model provider at the same time. The benefit this
pipeline is meant to provide is strictly about total batch completion time: for a
fixed number of independent queries, running them concurrently should reduce the total
wall-clock time relative to running them one at a time, even though the total amount
of computation performed, and therefore the total tokens consumed, does not change.
This benefit is only realized, however, if the underlying API provider actually
processes concurrent requests in parallel rather than queuing or rate-limiting them;
if a provider enforces a strict cap on tokens processed per unit time, incoming
concurrent requests are effectively serialized by that cap regardless of how many are
issued in parallel by the client, which removes the wall-clock advantage the pipeline
is intended to provide.

The baseline used throughout this project's evaluation is a conventional recursive
language model that does not perform any retrieval gating: at every reasoning pass,
the full source document is passed to the language model as context, regardless of
which sub-query is currently being answered. This baseline represents the accuracy
ceiling the retrieval-gated variants are compared against, since it always has access
to the complete document and therefore cannot fail due to a missing or discarded
passage the way a retrieval-gated method can. It also represents the token and cost
ceiling, since every pass reprocesses the entire document rather than a reduced subset
of it. Any retrieval-gated configuration is therefore evaluated along two axes
simultaneously relative to this baseline: how much of the baseline's token and cost
footprint it manages to avoid, and how much, if any, of the baseline's accuracy it
sacrifices in exchange. A configuration that reduces token usage without sacrificing
measurable accuracy relative to this baseline is the target outcome; a configuration
that reduces token usage but at a large accuracy cost is a genuine trade-off rather
than an unambiguous improvement.

All real-mode experiments in this evaluation use openai/gpt-oss-20b served through
Groq's hosted API on its free usage tier. This model is a reasoning model, meaning it
allocates part of its output token budget to an internal chain-of-thought process
before producing the final, user-visible answer; if the maximum output token budget
configured for a request is too small, the internal reasoning process can consume the
entire budget and leave nothing for the visible answer, which returns as an empty
string even though the underlying API call reports success. Care was taken to
configure a generous enough token budget to avoid this failure mode. The evaluation is
run using a benchmark harness that supports running the same set of questions multiple
times as separate trials, so that a mean and a minimum-to-maximum range can be
reported for every metric rather than a single, noisy, single-run number. The harness
also enforces a minimum spacing between consecutive real API calls, coordinated across
all concurrently running threads, to respect the provider's rate limits, and
automatically retries a request after a rate-limit error before falling back to a
simulated answer if retries are exhausted.

Two independent methods are used to judge whether a generated answer is correct, and
both are reported side by side rather than collapsing to a single score. The first is
a keyword-overlap heuristic: each question has an associated set of expected keywords,
and an answer is marked correct if it contains at least a fixed proportion of those
keywords after removing common stopwords. This method is fast and requires no
additional model calls, but it is a weak proxy for correctness, since it can mark a
correct answer wrong if it is phrased without using the expected keywords, and can
mark an incorrect answer right if it happens to repeat enough of the expected
vocabulary without actually answering the question. The second method uses semantic
embedding similarity: both the generated answer and a full reference answer are
embedded using the same sentence-embedding model used for retrieval, and the cosine
similarity between the two embeddings is compared against a fixed threshold to decide
correctness. This second method is intended to be more tolerant of paraphrasing than
the keyword method, though it introduces its own dependency on the embedding model's
quality and on the chosen similarity threshold, both of which are reported alongside
the results so the scoring methodology remains inspectable.

Because the evaluation is run against Groq's free usage tier, the billed cost reported
directly by the provider for every call is zero, which makes the provider's own cost
figure useless for comparing methods against each other. To produce a cost comparison
that is still meaningful, the evaluation harness instead computes an estimated cost for
every call, using the number of input and output tokens consumed by that call
multiplied against a fixed, published per-token price for an equivalent commercial
model, entirely independent of what Groq actually billed. This estimated figure is
reported alongside, and clearly labeled separately from, the real, actual cost billed
by the provider, so that a reader is never able to mistake the estimate for a verified
real-world dollar figure. A genuine, verified cost comparison would require running the
same evaluation against a metered, non-free-tier provider that bills per token used,
which has not yet been done as part of this evaluation.

This project's design sits at the intersection of two established lines of work:
retrieval-augmented generation, which grounds a language model's output in passages
retrieved from an external corpus rather than relying solely on the model's parametric
knowledge, and recursive or iterative reasoning over long documents, which improves
answer quality by allowing the model multiple passes over the available context. Most
retrieval-augmented generation work is motivated primarily by improving answer
accuracy or reducing hallucination, treating retrieval as a way to supply the model
with information it would not otherwise have access to. Comparatively less prior work
treats retrieval specifically as a mechanism for reducing the token cost and latency of
an iterative or recursive reasoning pipeline that would otherwise reprocess the full
document at every step; this efficiency-oriented framing, rather than a pure
accuracy-oriented framing, is the specific angle this project adopts. Separately, prior
work on splitting inference between a cloud-hosted large model and a smaller
client-side model has explored a different axis of the same general efficiency
problem, trading network latency against local compute rather than trading retrieved
context size against re-processing cost, which is the axis this project focuses on
instead.

The evaluation reported so far has notable limitations that any reader should weigh
before generalizing its findings. The source document used for evaluation, while now
substantially longer and organized into multiple distinct sections than the earlier
single-paragraph pilot, is still short relative to the long, multi-document corpora
the retrieval-gating approach is ultimately intended to help with, so the
token-reduction percentages observed here may not directly transfer to a much longer
or multi-document setting. The number of trials run so far is still small enough that
reported ranges reflect real run-to-run noise rather than a tight, statistically
confident estimate. Testing on a metered, rate-limit-free provider tier has not yet
been done, so the parallel-execution pipeline's real benefit, and a fully verified
real-dollar cost comparison, both remain open questions rather than settled findings.
Planned future work includes evaluating against considerably longer, multi-passage
documents, deliberately constructing question sets designed to trigger the AutoHealer
fallback path rather than relying on it triggering incidentally, and extending the
number of trials run per configuration once daily API quota constraints allow for it.
"""

STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "of", "to", "in", "on",
    "and", "or", "for", "with", "by", "at", "as", "that", "this", "it",
    "what", "how", "does", "do", "did", "which", "when", "where", "who",
}


def keywordize(text: str) -> List[str]:
    words = re.findall(r"[a-zA-Z][a-zA-Z\-]+", text.lower())
    return [w for w in words if w not in STOPWORDS and len(w) > 2]


# Single shared retriever instance so the HNSW index / embedding model
# loads once, not once per query. Set in main() from CLI args (--top-k,
# --embedding-model) so different configs can be swept without editing code.
RETRIEVER = None


def retrieve_context(question: str) -> str:
    """Real retrieval step: HNSW semantic search over SOURCE_DOCUMENT."""
    chunks = RETRIEVER.retrieve(SOURCE_DOCUMENT, question)
    return " ".join(chunks)


def build_prompt(question: str, context: str) -> str:
    return (
        "Answer the question using only the context below. "
        "Be concise (1-2 sentences).\n\n"
        f"Context: {context}\n\n"
        f"Question: {question}"
    )


def looks_degraded(answer: str) -> bool:
    if not answer or len(answer.strip()) < 8:
        return True
    lowered = answer.lower()
    hedge_phrases = [
        "i don't know", "i do not know", "cannot answer", "not mentioned",
        "no information", "unable to determine", "not enough context",
    ]
    return any(p in lowered for p in hedge_phrases)


class GlobalPacer:
    """
    Thread-safe pacer that enforces a minimum interval between the END of
    one real API call and the START of the next -- across ALL threads,
    not per-thread. This keeps us under Groq's free-tier TPM limit instead
    of bursting requests when running the parallel method.
    """

    def __init__(self, min_interval: float = 1.0):
        self.min_interval = min_interval
        self._lock = threading.Lock()
        self._last_call_ts = 0.0

    def wait(self):
        with self._lock:
            now = time.monotonic()
            elapsed = now - self._last_call_ts
            if elapsed < self.min_interval:
                time.sleep(self.min_interval - elapsed)
            self._last_call_ts = time.monotonic()


PACER = GlobalPacer(min_interval=1.0)


@dataclass
class BenchmarkResult:
    method: str
    accuracy: float                # primary: embedding-based when available
    accuracy_keyword: float        # always computed, reported for comparison
    avg_cost: float
    p50_latency: float
    avg_tokens: int
    total_queries: int
    correct_answers: int
    mock_fallback_count: int = 0
    batch_wall_clock_s: float = 0.0


class BenchmarkRunner:
    def __init__(self, use_mock: bool = False, num_questions: Optional[int] = None):
        self.use_mock = use_mock
        self.results = []

        if not use_mock:
            try:
                self.llm = LLMClient(provider="groq")
                print(f"Using REAL Groq API (free tier) | Retriever mode: {RETRIEVER.mode} | top_k={RETRIEVER.top_k}")
            except Exception as e:
                print(f"Error initializing Groq: {e}")
                print("   Falling back to mock mode...")
                self.use_mock = True
                self.llm = None
        else:
            self.llm = None
            print("Using MOCK mode (accuracy/cost are simulated, not real)")

        self.dataset = self._create_dataset()
        if num_questions is not None:
            self.dataset = self.dataset[:num_questions]
        print(f"Dataset: {len(self.dataset)} questions")

    def _create_dataset(self) -> List[Dict]:
        # Each entry: "question", "answer" (keywords, for the keyword-overlap
        # heuristic) and "reference" (a full sentence, for embedding-similarity
        # scoring). Spans all 10 sections of SOURCE_DOCUMENT.
        return [
            # Section 1 -- Introduction and motivation
            {"question": "What computational cost problem does recursive reasoning have?",
             "answer": "reprocesses entire context every pass token cost scale",
             "reference": "Recursive reasoning re-processes the entire context window on every pass, so token usage, latency, and cost all scale with the number of passes."},
            {"question": "What central question does this project investigate?",
             "answer": "selective retrieval avoid reprocessing cost keep accuracy",
             "reference": "Whether most of the re-processing cost can be avoided without losing recursion's accuracy benefits by being selective about what is re-processed."},

            # Section 2 -- Retrieval-gated recursion controller
            {"question": "How does the retrieval-gated recursion controller decide what to pass to the model?",
             "answer": "HNSW index dense embeddings top-k similar chunks",
             "reference": "It builds an HNSW index over dense embeddings of document chunks and retrieves only the top-k most semantically similar chunks to the query."},
            {"question": "What trade-off does the top-k parameter control?",
             "answer": "small k saves tokens risk missing large k recovers accuracy less savings",
             "reference": "A small top-k saves more tokens but risks missing needed context, while a large top-k recovers accuracy but reduces token savings."},
            {"question": "What happens to token savings when top-k is large on a short document?",
             "answer": "large top-k retrieves nearly entire document savings disappear",
             "reference": "A large top-k can retrieve nearly the entire short document, so the token-reduction benefit largely disappears even though accuracy is preserved."},

            # Section 3 -- AutoHealer
            {"question": "What failure mode does AutoHealer address?",
             "answer": "context rot discarded chunk needed answer",
             "reference": "AutoHealer addresses context rot, where retrieval gating discards a chunk of the document that was actually needed to answer the question."},
            {"question": "How does AutoHealer detect a degraded answer?",
             "answer": "short response hedging language uncertainty cannot determined",
             "reference": "It looks for an unusually short response or hedging language such as expressions of uncertainty or statements the answer cannot be determined."},
            {"question": "What does AutoHealer do once degradation is detected?",
             "answer": "second pass full unfiltered document instead retrieved subset",
             "reference": "It triggers a second generation pass using the full, unfiltered source document instead of the retrieved subset."},

            # Section 4 -- Parallel execution pipeline
            {"question": "What is the parallel execution pipeline meant to reduce?",
             "answer": "reduce wall-clock batch time not token usage",
             "reference": "It is meant to reduce the wall-clock time needed to answer a batch of independent queries, not the token usage of any single query."},
            {"question": "Why can independent queries be run concurrently?",
             "answer": "queries independent no dependency concurrent issue",
             "reference": "Different sub-queries against the same document do not depend on each other's answers, so their retrieval and generation steps can be issued concurrently."},
            {"question": "When does the parallel pipeline fail to provide a wall-clock benefit?",
             "answer": "provider rate cap serializes concurrent requests removes benefit",
             "reference": "If the provider enforces a strict cap on tokens processed per unit time, concurrent requests are effectively serialized regardless of client-side parallelism."},

            # Section 5 -- Baseline recursive language model
            {"question": "What context does the baseline recursive model use at every pass?",
             "answer": "full source document every pass no retrieval gating",
             "reference": "The baseline passes the full source document to the language model as context at every reasoning pass, with no retrieval gating."},
            {"question": "What does the baseline represent in the comparison?",
             "answer": "accuracy ceiling token cost ceiling comparison baseline",
             "reference": "The baseline represents both the accuracy ceiling and the token/cost ceiling that retrieval-gated methods are compared against."},

            # Section 6 -- Experimental setup
            {"question": "What model and provider are used for real-mode experiments?",
             "answer": "openai gpt-oss-20b Groq free tier",
             "reference": "The experiments use openai/gpt-oss-20b served through Groq's hosted API on its free usage tier."},
            {"question": "Why must the output token budget be generous for this model?",
             "answer": "reasoning model chain-of-thought consumes budget empty answer",
             "reference": "The model is a reasoning model that spends part of its token budget on internal chain-of-thought, and a too-small budget can leave nothing for the visible answer."},
            {"question": "What does the benchmark harness do to respect provider rate limits?",
             "answer": "minimum spacing calls retries rate limit mock fallback",
             "reference": "It enforces a minimum spacing between consecutive real API calls across all threads and retries after a rate-limit error before falling back to a simulated answer."},

            # Section 7 -- Evaluation metrics and scoring methodology
            {"question": "What are the two methods used to judge answer correctness?",
             "answer": "keyword overlap embedding similarity reference answer reported",
             "reference": "A keyword-overlap heuristic and a semantic embedding similarity comparison against a reference answer are both used and reported side by side."},
            {"question": "What is a weakness of the keyword-overlap scoring method?",
             "answer": "weak proxy wrong keywords right vocabulary without answering",
             "reference": "It can mark a correctly phrased answer wrong if it avoids the expected keywords, or mark an incorrect answer right if it repeats enough expected vocabulary."},
            {"question": "How does the embedding similarity scoring method work?",
             "answer": "embed answer reference cosine similarity fixed threshold",
             "reference": "It embeds the generated answer and a reference answer with the same embedding model and compares their cosine similarity against a fixed threshold."},

            # Section 8 -- Cost accounting and pricing estimation
            {"question": "Why is Groq's own reported cost useless for comparing methods?",
             "answer": "free tier bills zero cost useless comparison",
             "reference": "Because the free usage tier bills zero for every call, so the provider's own cost figure cannot distinguish between methods."},
            {"question": "How is a meaningful estimated cost computed instead?",
             "answer": "tokens consumed published per-token price equivalent model",
             "reference": "By multiplying the tokens consumed by a call against a fixed, published per-token price for an equivalent commercial model."},
            {"question": "What would be needed for a fully verified cost comparison?",
             "answer": "metered non-free-tier provider bills per token",
             "reference": "Running the same evaluation against a metered, non-free-tier provider that actually bills per token used."},

            # Section 9 -- Related work
            {"question": "What two lines of work does this project sit at the intersection of?",
             "answer": "retrieval-augmented generation recursive iterative reasoning long documents",
             "reference": "Retrieval-augmented generation and recursive or iterative reasoning over long documents."},
            {"question": "What efficiency-oriented framing does this project adopt that most RAG work does not?",
             "answer": "retrieval reduce token cost latency not just accuracy",
             "reference": "Treating retrieval as a mechanism for reducing token cost and latency of a recursive pipeline, rather than purely for improving accuracy."},

            # Section 10 -- Limitations and future work
            {"question": "Why might token-reduction percentages not transfer to a longer setting?",
             "answer": "document still short relative target long multi-document corpora",
             "reference": "The evaluation document, while longer than the earlier pilot, is still short relative to the long, multi-document corpora the method is ultimately intended for."},
            {"question": "What remains an open question about the parallel-execution pipeline?",
             "answer": "parallel benefit open no metered rate-limit-free testing",
             "reference": "Its real benefit remains open because testing on a metered, rate-limit-free provider tier has not yet been done."},
            {"question": "What is planned to properly evaluate AutoHealer in future work?",
             "answer": "deliberately construct questions trigger AutoHealer fallback",
             "reference": "Deliberately constructing question sets designed to trigger the AutoHealer fallback path rather than relying on it triggering incidentally."},
        ]

    def _mock_query(self, prompt: str) -> Dict:
        return {
            "content": f"[MOCK] Answer to: {prompt[:50]}...",
            "tokens_used": random.randint(100, 300),
            "cost": 0.001 + random.random() * 0.002,
            "latency_ms": random.uniform(5, 15),
            "model": "mock",
        }

    def _normalize_response(self, llm_response, used_mock_fallback: bool = False) -> Dict:
        """
        Turn either a dict (mock) or a real LLMResponse object into a plain
        dict. Real LLMResponse now exposes actual_cost (what Groq billed,
        always $0 on free tier) and estimated_cost (reference-priced
        equivalent). We report estimated_cost as 'cost' here so benchmark
        comparisons are meaningful instead of trivially zero.

        used_mock_fallback marks whether this specific answer is a mock
        substitution due to a real-call failure (as opposed to intentional
        --use-mock mode) -- this is tracked and surfaced, never hidden.
        """
        if isinstance(llm_response, dict):
            return {
                "content": llm_response.get("content", "") or "",
                "tokens_used": llm_response.get("tokens_used", 200),
                "cost": llm_response.get("cost", 0.001),
                "latency_ms": llm_response.get("latency_ms", 10),
                "model": llm_response.get("model", "unknown"),
                "used_mock_fallback": used_mock_fallback,
            }
        return {
            "content": getattr(llm_response, "content", "") or "",
            "tokens_used": getattr(llm_response, "tokens_used", 200),
            "cost": getattr(llm_response, "estimated_cost", 0.0),
            "actual_cost": getattr(llm_response, "actual_cost", 0.0),
            "latency_ms": getattr(llm_response, "latency_ms", 10),
            "model": getattr(llm_response, "model", "unknown"),
            "used_mock_fallback": used_mock_fallback,
        }

    def _call_llm(self, prompt: str) -> Dict:
        if self.use_mock:
            return self._normalize_response(self._mock_query(prompt))

        last_error = None
        for attempt in range(MAX_RETRIES_ON_RATE_LIMIT):
            try:
                PACER.wait()
                raw = self.llm.query(prompt)
                return self._normalize_response(raw)
            except Exception as e:
                last_error = e
                msg = str(e)
                match = RATE_LIMIT_RETRY_PATTERN.search(msg)
                if match:
                    wait_s = float(match.group(1)) + 0.5
                    print(f"     Rate limited, waiting {wait_s:.1f}s (attempt {attempt + 1}/{MAX_RETRIES_ON_RATE_LIMIT})...")
                    time.sleep(wait_s)
                    continue
                break

        print(f"     Error: {last_error}, using mock fallback")
        return self._normalize_response(self._mock_query(prompt), used_mock_fallback=True)

    def _score_keyword(self, expected: str, actual: str) -> bool:
        expected_kw = [w for w in keywordize(expected)]
        if not expected_kw:
            return False
        actual_lower = actual.lower()
        hits = sum(1 for kw in expected_kw if kw in actual_lower)
        ratio = hits / len(expected_kw)
        return ratio >= 0.34

    def _score_embedding(self, reference: str, actual: str) -> Optional[bool]:
        """
        Semantic scoring: embed both the reference answer and the model's
        actual answer with the same embedding model used for retrieval, and
        compare cosine similarity against a fixed threshold. Returns None
        (score unavailable) if the retriever isn't in real semantic mode or
        the answer is empty -- callers should fall back to keyword scoring
        in that case rather than silently marking it wrong.
        """
        if RETRIEVER.mode != "hnsw_semantic" or not actual.strip():
            return None
        try:
            emb = RETRIEVER.model.encode([reference, actual], normalize_embeddings=True)
            similarity = float(np.dot(emb[0], emb[1]))
            return similarity >= EMBEDDING_MATCH_THRESHOLD
        except Exception:
            return None

    def _answer_baseline(self, question: str) -> Dict:
        prompt = build_prompt(question, SOURCE_DOCUMENT)
        return self._call_llm(prompt)

    def _answer_hybrid_fixed_k(self, question: str) -> Dict:
        context = retrieve_context(question)
        prompt = build_prompt(question, context)
        return self._call_llm(prompt)

    def _answer_hybrid_autohealer(self, question: str) -> Dict:
        context = retrieve_context(question)
        response = self._call_llm(build_prompt(question, context))
        if looks_degraded(response["content"]):
            healed = self._call_llm(build_prompt(question, SOURCE_DOCUMENT))
            healed["tokens_used"] += response["tokens_used"]
            healed["cost"] += response["cost"]
            healed["latency_ms"] += response["latency_ms"]
            healed["used_mock_fallback"] = (
                response.get("used_mock_fallback", False)
                or healed.get("used_mock_fallback", False)
            )
            return healed
        return response

    def _answer_hybrid_parallel(self, questions: List[str]) -> List[Dict]:
        def work(q, stagger_delay):
            time.sleep(stagger_delay)
            context = retrieve_context(q)
            return self._call_llm(build_prompt(q, context))

        results = [None] * len(questions)
        max_workers = 2 if not self.use_mock else min(8, len(questions))
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            future_to_idx = {
                pool.submit(work, q, i * 0.3): i
                for i, q in enumerate(questions)
            }
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                results[idx] = future.result()
        return results

    def _grade(self, query: Dict, content: str) -> Dict[str, bool]:
        """Compute both scores for one answer. Primary 'is_correct' uses the
        embedding score when available, falling back to keyword-overlap."""
        if self.use_mock:
            simulated = random.random() < 0.70
            return {"is_correct": simulated, "is_correct_keyword": simulated}

        is_correct_keyword = self._score_keyword(query.get("answer", ""), content)
        is_correct_embedding = self._score_embedding(query.get("reference", ""), content)
        is_correct = is_correct_embedding if is_correct_embedding is not None else is_correct_keyword
        return {"is_correct": is_correct, "is_correct_keyword": is_correct_keyword}

    def _run_method(self, method_name: str, queries: List[Dict]) -> Dict:
        results = []
        wall_start = time.perf_counter()

        if method_name == "Hybrid + Parallel":
            questions = [q["question"] for q in queries]
            responses = self._answer_hybrid_parallel(questions)
            for query, response in zip(queries, responses):
                grade = self._grade(query, response["content"])
                results.append({"query": query, "response": response, **grade})
        else:
            for query in queries:
                question = query["question"]
                if method_name == "Baseline RLM":
                    response = self._answer_baseline(question)
                elif method_name == "Hybrid + AutoHealer":
                    response = self._answer_hybrid_autohealer(question)
                else:
                    response = self._answer_hybrid_fixed_k(question)

                grade = self._grade(query, response["content"])
                results.append({"query": query, "response": response, **grade})

        wall_elapsed = time.perf_counter() - wall_start

        total = len(results)
        correct = sum(1 for r in results if r["is_correct"])
        correct_keyword = sum(1 for r in results if r["is_correct_keyword"])
        accuracy = (correct / total) * 100 if total > 0 else 0
        accuracy_keyword = (correct_keyword / total) * 100 if total > 0 else 0

        costs = [r["response"]["cost"] for r in results]
        latencies = [r["response"]["latency_ms"] for r in results]
        tokens = [r["response"]["tokens_used"] for r in results]
        mock_fallback_count = sum(1 for r in results if r["response"].get("used_mock_fallback"))

        if mock_fallback_count > 0 and not self.use_mock:
            print(f"     WARNING: {mock_fallback_count}/{total} queries fell back to mock "
                  f"due to real-call errors -- this result is PARTIALLY simulated, not fully real.")

        return {
            "method": method_name,
            "accuracy": accuracy,
            "accuracy_keyword": accuracy_keyword,
            "avg_cost": statistics.mean(costs) if costs else 0,
            "p50_latency": statistics.median(latencies) if latencies else 0,
            "avg_tokens": int(statistics.mean(tokens)) if tokens else 0,
            "total_queries": total,
            "correct_answers": correct,
            "mock_fallback_count": mock_fallback_count,
            "batch_wall_clock_s": wall_elapsed,
        }

    def run_all(self) -> List[BenchmarkResult]:
        self.results = []
        queries = self.dataset

        print(f"\nRunning benchmarks on {len(queries)} queries...")
        print(f"   Mode: {'REAL (Groq)' if not self.use_mock else 'MOCK'}\n")

        methods = [
            "Baseline RLM",
            "Hybrid (fixed_k)",
            "Hybrid + AutoHealer",
            "Hybrid + Parallel",
        ]

        for name in methods:
            print(f"   Testing: {name}...")
            result = self._run_method(name, queries)
            self.results.append(BenchmarkResult(**result))
            print(f"     Accuracy (embedding): {result['accuracy']:.1f}%  |  Accuracy (keyword): {result['accuracy_keyword']:.1f}%")

        return self.results

    def print_table(self):
        print("\n" + "=" * 108)
        print("BENCHMARK RESULTS" + ("  (MOCK - not real numbers)" if self.use_mock else ""))
        print("=" * 108)

        print(f"{'Method':<22} {'Acc(emb)':<10} {'Acc(kw)':<9} {'Est.Cost':<12} {'p50 Lat':<10} {'Tokens':<8} {'Wall(s)':<9} {'MockFB':<7}")
        print("-" * 108)

        for r in self.results:
            print(
                f"{r.method:<22} "
                f"{r.accuracy:>6.1f}%   "
                f"{r.accuracy_keyword:>5.1f}%   "
                f"${r.avg_cost:>8.5f}  "
                f"{r.p50_latency:>7.2f}ms  "
                f"{r.avg_tokens:>6}  "
                f"{r.batch_wall_clock_s:>7.2f}  "
                f"{r.mock_fallback_count:>5}"
            )

        print("=" * 108)

        if self.results:
            best_cost = min(self.results, key=lambda x: x.avg_cost)
            fastest = min(self.results, key=lambda x: x.p50_latency)
            fastest_batch = min(self.results, key=lambda x: x.batch_wall_clock_s)
            most_acc = max(self.results, key=lambda x: x.accuracy)

            print(f"\nCheapest (estimated): {best_cost.method} (${best_cost.avg_cost:.5f})")
            print(f"Fastest (p50 per-query): {fastest.method} ({fastest.p50_latency:.2f}ms)")
            print(f"Fastest (batch wall-clock): {fastest_batch.method} ({fastest_batch.batch_wall_clock_s:.2f}s)")
            print(f"Most accurate (embedding score): {most_acc.method} ({most_acc.accuracy:.1f}%)")

            any_fallback = any(r.mock_fallback_count > 0 for r in self.results)
            if any_fallback and not self.use_mock:
                print("\nWARNING: one or more methods above include mock-fallback answers "
                      "(real API call failed and a simulated answer was substituted). "
                      "See MockFB column -- do not treat these rows as fully real.")

    def save_results(self, output_path: str = None):
        if output_path is None:
            tag = f"topk{RETRIEVER.top_k}_{RETRIEVER.mode}"
            output_path = f"benchmarks/results/results_{tag}.json"
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        data = {
            "mode": "real" if not self.use_mock else "mock",
            "retriever_mode": RETRIEVER.mode,
            "top_k": RETRIEVER.top_k,
            "num_questions": len(self.dataset),
            "embedding_match_threshold": EMBEDDING_MATCH_THRESHOLD,
            "timestamp": time.time(),
            "results": [asdict(r) for r in self.results],
        }

        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)

        print(f"\nResults saved to: {output_path}")


def _agg(values: List[float]) -> Dict[str, float]:
    return {
        "mean": statistics.mean(values),
        "min": min(values),
        "max": max(values),
    }


def run_multi_trial(runner: BenchmarkRunner, trials: int):
    all_trials: List[List[Dict]] = []

    for t in range(1, trials + 1):
        print(f"\n{'#' * 108}\n# TRIAL {t}/{trials}\n{'#' * 108}")
        results = runner.run_all()
        runner.print_table()
        all_trials.append([asdict(r) for r in results])

    method_order = [r["method"] for r in all_trials[0]]
    aggregated = {}

    for method in method_order:
        rows = [next(r for r in trial if r["method"] == method) for trial in all_trials]
        aggregated[method] = {
            "accuracy": _agg([r["accuracy"] for r in rows]),
            "accuracy_keyword": _agg([r["accuracy_keyword"] for r in rows]),
            "avg_cost": _agg([r["avg_cost"] for r in rows]),
            "p50_latency": _agg([r["p50_latency"] for r in rows]),
            "avg_tokens": _agg([r["avg_tokens"] for r in rows]),
            "batch_wall_clock_s": _agg([r["batch_wall_clock_s"] for r in rows]),
            "mock_fallback_total": sum(r["mock_fallback_count"] for r in rows),
        }

    print("\n" + "=" * 108)
    print(f"SUMMARY ACROSS {trials} TRIALS (mean [min-max])" + ("  (MOCK)" if runner.use_mock else ""))
    print("=" * 108)

    for method, agg in aggregated.items():
        print(f"\n{method}:")
        a = agg["accuracy"]; ak = agg["accuracy_keyword"]; c = agg["avg_cost"]; l = agg["p50_latency"]
        tk = agg["avg_tokens"]; w = agg["batch_wall_clock_s"]
        print(f"  Accuracy (embedding): {a['mean']:.1f}%   [{a['min']:.1f}-{a['max']:.1f}]")
        print(f"  Accuracy (keyword):   {ak['mean']:.1f}%   [{ak['min']:.1f}-{ak['max']:.1f}]")
        print(f"  Est. cost:            ${c['mean']:.5f}  [{c['min']:.5f}-{c['max']:.5f}]")
        print(f"  p50 latency:          {l['mean']:.2f}ms  [{l['min']:.2f}-{l['max']:.2f}]")
        print(f"  Avg tokens:           {tk['mean']:.1f}   [{tk['min']:.1f}-{tk['max']:.1f}]")
        print(f"  Batch wall-clock:     {w['mean']:.2f}s   [{w['min']:.2f}-{w['max']:.2f}]")
        if agg["mock_fallback_total"] > 0 and not runner.use_mock:
            print(f"  WARNING: {agg['mock_fallback_total']} mock-fallback answers across all "
                  f"trials for this method -- treat with caution, not fully real data.")

    tag = f"topk{RETRIEVER.top_k}_{RETRIEVER.mode}"
    output_path = f"benchmarks/results/results_multi_trial_{tag}.json"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump({
            "mode": "real" if not runner.use_mock else "mock",
            "retriever_mode": RETRIEVER.mode,
            "top_k": RETRIEVER.top_k,
            "num_questions": len(runner.dataset),
            "embedding_match_threshold": EMBEDDING_MATCH_THRESHOLD,
            "trials": trials,
            "timestamp": time.time(),
            "per_trial_results": all_trials,
            "aggregated": aggregated,
        }, f, indent=2)

    print(f"\nMulti-trial results saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true", help="Run all benchmarks")
    parser.add_argument("--use-mock", action="store_true", default=False,
                        help="Use mock mode (no real API calls)")
    parser.add_argument("--real", action="store_true", default=False,
                        help="Use real Groq API (free tier)")
    parser.add_argument("--plots", action="store_true",
                        help="Generate plots after running")
    parser.add_argument("--trials", type=int, default=1,
                        help="Run the full suite N times and report mean + [min-max] range per method")
    parser.add_argument("--top-k", type=int, default=3,
                        help="Number of chunks the retriever returns per query (default: 3)")
    parser.add_argument("--embedding-model", type=str, default="all-MiniLM-L6-v2",
                        help="sentence-transformers model name/id, e.g. all-MiniLM-L6-v2, "
                             "sentence-transformers/all-mpnet-base-v2, BAAI/bge-small-en-v1.5")
    parser.add_argument("--num-questions", type=int, default=None,
                        help="Use only the first N questions from the dataset (default: all 27). "
                             "Useful for a cheap --use-mock smoke test before spending real API quota.")
    args = parser.parse_args()

    if args.real:
        use_mock = False
    elif args.use_mock:
        use_mock = True
    else:
        use_mock = False

    global RETRIEVER
    print(f"Initializing retriever: top_k={args.top_k}, model={args.embedding_model} ...")
    RETRIEVER = RetrievalGating(top_k=args.top_k, model_name=args.embedding_model)

    runner = BenchmarkRunner(use_mock=use_mock, num_questions=args.num_questions)

    if args.all or args.plots:
        if args.trials and args.trials > 1:
            run_multi_trial(runner, args.trials)
        else:
            runner.run_all()
            runner.print_table()
            runner.save_results()


if __name__ == "__main__":
    main()