# Hybrid Retrieval-Augmented RLM (HRA-RLM)

[![CI](https://github.com/Sidra-009/hybrid-retrieval-augmented-rlm/actions/workflows/ci.yml/badge.svg)](https://github.com/Sidra-009/hybrid-retrieval-augmented-rlm/actions/workflows/ci.yml)
[![Benchmarks](https://github.com/Sidra-009/hybrid-retrieval-augmented-rlm/actions/workflows/benchmark.yml/badge.svg)](https://github.com/Sidra-009/hybrid-retrieval-augmented-rlm/actions/workflows/benchmark.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

> **Cost-efficient recursive reasoning over massive documents by gating recursion through retrieval.**
> A retrieval-gated extension of MIT's Recursive Language Models (Zhang, Kraska & Khattab, 2025) — built to cut cost, latency, and context rot without sacrificing multi-hop reasoning quality.

---

## Why this exists

Recursive Language Models (RLMs) let an LLM treat a huge document as an external variable in a Python REPL and recursively query pieces of it, instead of stuffing everything into the context window. This solves *context rot* — but the [original RLM approach](https://arxiv.org/abs/2512.24601) explores the **raw, unfiltered document** at every recursion step. On benchmarks like BrowseComp-Plus, this costs an average of **$0.99 per query** and runs sub-calls **sequentially**, which the authors themselves flag as a limitation (heavy-tailed cost distribution, synchronous execution, limited recursion depth).

**HRA-RLM** asks a simple question: *what if the model never had to look at irrelevant text in the first place?*

We gate every recursive step behind a retrieval pass over a custom HNSW vector index, so the RLM only ever reasons over the subset of the document that's actually relevant to the query — then parallelizes the remaining sub-calls and continuously monitors for context rot so it can fall back to pure retrieval when recursion isn't paying off.

## What's actually new here (vs. the MIT baseline)

| | MIT RLM (baseline) | HRA-RLM (this repo) |
|---|---|---|
| Input to recursion | Full raw document string | Top-*k* semantically relevant chunks (HNSW-filtered) |
| Sub-call execution | Sequential | Parallelized via a step-orchestrator |
| Degradation handling | None (static strategy) | Live context-rot detector with automatic RAG fallback |
| Cost profile | ~$0.99/query (BrowseComp-Plus, GPT-5) | Target: 5–10x reduction, benchmarked below |
| Retrieval | None (pure REPL exploration) | Custom from-scratch HNSW vector index |

This is **not** a fork or a re-skin of `alexzhang13/rlm`. It's an independent implementation that treats retrieval and recursion as complementary, tunable stages in a single pipeline, with the engineering (tests, CI, benchmarks) to back up every claim made in this README.

## Architecture

```
                     ┌──────────────────────────┐
        query ──────►│   HRA-RLM Orchestrator    │
                     └────────────┬─────────────┘
                                  │
                 ┌────────────────┼────────────────┐
                 ▼                                  ▼
        ┌─────────────────┐              ┌─────────────────────┐
        │  HNSW Vector DB   │              │  Context Rot Monitor  │
        │  (retrieval gate) │              │  (accuracy vs. depth) │
        └────────┬──────────┘              └──────────┬───────────┘
                 │ top-k relevant chunks                │ mode switch
                 ▼                                       ▼
        ┌───────────────────────────────────────────────────────┐
        │            Recursive Reasoning Engine (REPL)            │
        │   root LLM writes code → executes → spawns sub-calls    │
        │        sub-calls parallelized across N workers          │
        └───────────────────────────────────────────────────────┘
                                  │
                                  ▼
                             final answer
```

Full design rationale and trade-off discussion: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)

## Benchmarks

Reproducible benchmark suite in [`benchmarks/`](benchmarks/), results and methodology in [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md). Numbers below are filled in as each module lands — this is a living table, not a marketing claim.

| Metric | MIT RLM baseline | HRA-RLM | Δ |
|---|---|---|---|
| Avg. cost / query | *pending* | *pending* | — |
| Latency (p50 / p95) | *pending* | *pending* | — |
| Accuracy (test set) | *pending* | *pending* | — |
| Tokens / query | *pending* | *pending* | — |

## Project status

🚧 Actively under development, built in public, module by module. See the [project board](../../projects) for what's shipped and what's next.

## Installation

```bash
git clone https://github.com/Sidra-009/hybrid-retrieval-augmented-rlm.git
cd hybrid-retrieval-augmented-rlm
pip install -e ".[dev]"
```

## Quickstart

```python
from hydrarlm import HydraRLM

engine = HydraRLM.from_config("configs/default.yaml")
answer = engine.query(
    document_path="examples/annual_report_2023.pdf",
    question="Compare Company A and Company B revenue growth in 2023."
)
print(answer)
```

*(Full working example lands once the `rlm` and `vectordb` modules are merged — see roadmap.)*

## Repository layout

```
hydrarlm/
├── src/hydrarlm/
│   ├── vectordb/       # from-scratch HNSW implementation
│   ├── rlm/             # REPL sandbox + recursive reasoning core
│   ├── rot_detector/    # context-rot tracking + auto-heal
│   ├── orchestrator/    # parallel sub-call scheduling
│   └── utils/           # config, logging, shared helpers
├── benchmarks/          # reproducible cost/latency/accuracy comparisons
├── tests/                # unit + integration tests
├── docs/                 # architecture + benchmark writeups
└── examples/             # runnable demos
```

## Research background

This project builds directly on:

- Zhang, A. L., Kraska, T., & Khattab, O. (2025). *Recursive Language Models.* arXiv:2512.24601.
- Wang, D. (2026). *Think, But Don't Overthink: Reproducing Recursive Language Models.* arXiv:2603.02615.

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for a full related-work discussion and how each design decision here diverges from the baseline.

## Contributing

Contributions, issues, and benchmark reproductions are welcome — see [`CONTRIBUTING.md`](CONTRIBUTING.md).

## License

MIT — see [`LICENSE`](LICENSE).

## Citation

```bibtex
@software{hra_rlm_2026,
  title  = {Hybrid Retrieval-Augmented RLM},
  author = {Sidra},
  year   = {2026},
  url    = {https://github.com/Sidra-009/hybrid-retrieval-augmented-rlm}
}
```