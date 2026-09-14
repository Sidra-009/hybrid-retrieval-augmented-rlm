# Contributing

Thanks for your interest in HRA-RLM. This is an active undergraduate research
project; contributions, issues, and suggestions are welcome.

## Setup

```bash
git clone https://github.com/Sidra-009/hybrid-retrieval-augmented-rlm.git
cd hybrid-retrieval-augmented-rlm
pip install -r requirements.txt
```

Real benchmark runs require a Groq API key. Create a `.env` file in the repo
root (never commit this file — it's gitignored):
GROQ_API_KEY=your_key_here


## Running the benchmark suite

```bash
# Free smoke test, no API calls
python benchmarks/run_benchmark.py --all --use-mock --num-questions 3

# Real run
python benchmarks/run_benchmark.py --all --real
```

See [`docs/BENCHMARKING.md`](./docs/BENCHMARKING.md) for full details,
including Groq's rate-limit constraints and this project's data validity
policy — please read the validity policy before submitting benchmark
results in a PR.

## Making changes

1. Fork the repo and create a branch off `main`.
2. Keep changes focused — one logical change per PR.
3. If you change `src/hra_rlm/retriever.py`, `llm_client.py`, or
   `parallel.py`, run at least the mock smoke test before opening a PR.
4. If you change benchmark logic in `benchmarks/run_benchmark.py`, explain
   *why* in the PR description — this file's scoring and pacing logic is
   deliberately conservative (see `docs/BENCHMARKING.md`), and changes to it
   are held to a higher bar of justification than other files.
5. Open a PR against `main` with a clear description of what changed and
   why.

## Reporting issues

Open a GitHub issue with:
- What you ran (exact command)
- What you expected vs. what happened
- Your environment (OS, Python version)

## Code of conduct

Be respectful and constructive. This is a student research project — good
-faith questions and beginner-friendly PRs are welcome.