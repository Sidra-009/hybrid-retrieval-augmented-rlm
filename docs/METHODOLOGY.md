# Methodology

This document describes the architecture of HRA-RLM (Hybrid Retrieval-Augmented
Recursive Language Model) and how it differs from a baseline recursive LM
pipeline. It is intended for readers who want to understand *why* the system
is built this way, not just what numbers it produces — see
[`RESULTS.md`](./RESULTS.md) for the numbers, and
[`BENCHMARKS.md`](./BENCHMARKS.md) for how those numbers were produced.

## 1. Problem

A naive recursive LLM pipeline re-passes large amounts of document context to
the model at every recursive step, regardless of whether that context is
relevant to the current sub-query. This inflates token usage, cost, and
latency roughly linearly with document length and recursion depth.

## 2. Core idea: retrieval-gated recursion

HRA-RLM inserts a **retrieval gate** in front of every recursive call. Instead
of forwarding the full document, the system:

1. Embeds the incoming query.
2. Retrieves only the top-`k` semantically relevant chunks from the source
   document.
3. Passes just those chunks (plus the query) to the LLM for that step.

Recursion still happens when the pipeline determines a sub-query is needed,
but each recursive call now operates on a small, relevant context window
instead of the full document.

## 3. Components

| Component | File | Role |
|---|---|---|
| Retriever | `src/hra_rlm/retriever.py` | Chunks the source document and serves top-`k` relevant chunks per query via HNSW semantic search |
| LLM client | `src/hra_rlm/llm_client.py` | Wraps the Groq API call, reports both actual and estimated cost, handles reasoning-model token budgeting |
| Parallel executor | `src/hra_rlm/parallel.py` | Provides sequential vs. batched/parallel execution paths for fair latency comparison |
| Benchmark runner | `benchmarks/run_benchmark.py` | Orchestrates all four methods below over the question set and reports metrics |

## 4. Retrieval: HNSW semantic search

- **Chunking:** sentence-boundary-aware (not fixed-character-window), so
  chunks don't cut sentences mid-way.
- **Embeddings:** `sentence-transformers` (`all-MiniLM-L6-v2` by default,
  swappable via `--embedding-model`).
- **Index:** `hnswlib` approximate nearest-neighbor search over the chunk
  embeddings.
- **Fallback:** if `sentence-transformers`/`hnswlib` are unavailable in the
  environment, the retriever falls back to plain keyword matching. The active
  mode is always exposed via `retriever.mode` (`"hnsw_semantic"` or
  `"keyword_fallback"`) and printed at the start of every benchmark run, so
  results are never silently produced in a degraded mode without a visible
  signal.

## 5. Methods compared

The benchmark suite compares four configurations head-to-head on the same
question set:

1. **Baseline RLM** — full-document recursive prompting, no retrieval gate.
   This is the pre-existing approach HRA-RLM is measured against.
2. **Hybrid (fixed_k)** — retrieval-gated recursion with a fixed top-`k`
   chunk count per query. This is the core HRA-RLM proposal.
3. **Hybrid + AutoHealer** — adds a self-correction step: if the initial
   answer looks incomplete or low-confidence, the pipeline retrieves an
   additional chunk and retries before returning.
4. **Hybrid + Parallel** — same retrieval gating as (2), but queries within a
   batch are dispatched concurrently rather than one-at-a-time, to test
   whether wall-clock latency improves under parallel execution.

## 6. Cost and latency measurement

Two design decisions here are worth stating explicitly, because they were the
result of debugging real measurement bugs earlier in development:

- **Actual vs. estimated cost.** Groq's free tier returns `$0` for actual
  billing, which makes cost comparisons meaningless on their own. The client
  therefore also computes an `estimated_cost` from a `REFERENCE_PRICING`
  table of published per-token rates for a comparable metered model, so that
  relative cost reduction between methods is still measurable even on a free
  tier.
- **Per-query latency vs. batch wall-clock.** Per-query p50 latency does not
  reveal the benefit of parallel execution, since it measures one query at a
  time. The benchmark therefore also reports `batch_wall_clock_s` — the total
  time to answer the full question set for each method — which is the metric
  that actually reflects parallel speedup.

## 7. Known architectural limitations

- Dual accuracy scoring (embedding similarity vs. keyword overlap) exists
  because a single scoring method was found to disagree with itself across
  methods in early trials (see [`RESULTS.md`](./RESULTS.md)). Both are
  reported rather than picking whichever looks better.
- Retrieval quality depends on the embedding model and `top_k`; both are
  exposed as CLI flags (`--embedding-model`, `--top-k`) specifically so that
  their effect on the accuracy/cost trade-off can be measured, not assumed.
