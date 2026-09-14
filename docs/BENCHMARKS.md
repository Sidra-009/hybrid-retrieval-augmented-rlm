# Benchmarking

This document explains how the benchmark suite is run, what it measures, and
— importantly — which results are valid to cite and why. See
[`RESULTS.md`](./RESULTS.md) for the actual numbers and
[`METHODOLOGY.md`](./METHODOLOGY.md) for the architecture being measured.

## 1. Dataset

27 question/answer pairs spanning the full source document, used identically
across all four methods (see `benchmarks/datasets/`). An earlier 10-question
pilot dataset (scoped to a ~300-word excerpt) was used during early
development and is superseded by this 27-question set — it should not be
cited going forward.

## 2. Methods under test

See [`METHODOLOGY.md`](./METHODOLOGY.md#5-methods-compared) for what each of
the four methods does. All four are run over the same 27 questions in every
trial.

## 3. Metrics reported

| Metric | Meaning |
|---|---|
| Accuracy (embedding) | Semantic similarity between model answer and reference answer |
| Accuracy (keyword) | Keyword/token overlap between model answer and reference answer |
| Estimated cost | Per-query cost computed from published reference pricing (see `METHODOLOGY.md`) |
| Tokens | Total tokens consumed per query (prompt + completion) |
| p50 latency | Median per-query response time |
| Batch wall-clock | Total time to answer all 27 questions for that method — the metric that reflects real parallel speedup |
| Mock-fallback count | Number of queries in that method's row that fell back to simulated output due to a real-API error (rate limit, timeout, etc.) |

**Both accuracy metrics are always reported together, never just one.** Early
runs showed them disagreeing in direction (one metric improving while the
other dropped for the same method) — see `RESULTS.md`. Reporting only the
favorable one would be cherry-picking.

## 4. Running it yourself

```bash
# Cheap smoke test — no API calls, verifies the pipeline runs end-to-end
python benchmarks/run_benchmark.py --all --use-mock --num-questions 3

# Full real run, single trial
python benchmarks/run_benchmark.py --all --real

# Full real run, multiple trials (mean + [min-max] range per method)
python benchmarks/run_benchmark.py --all --real --trials 2

# Optimization sweep example
python benchmarks/run_benchmark.py --all --real --trials 2 \
    --top-k 5 --embedding-model BAAI/bge-small-en-v1.5
```

Results are written to `benchmarks/results/*.json` (gitignored — regenerate
locally rather than relying on committed snapshots, except for the automated
weekly mock-mode refresh described below).

## 5. Rate limits and what "real" actually requires

The benchmark runs against Groq's free tier, which enforces two independent
caps:

- **8,000 tokens/minute** — handled automatically by an internal pacer that
  enforces a minimum delay between real API calls.
- **200,000 tokens/day** — a hard daily budget. A single full 27-question × 4-
  method run consumes a large fraction of this on its own. This is a *daily*
  cap, not something the pacer can work around; the only options when it is
  hit are to wait for the daily reset or upgrade to a paid tier.

**Do not attempt to bypass the daily cap with a second account or key** — this
would violate Groq's terms of service and is out of scope for this project
regardless of the deadline pressure.

## 6. Data validity policy

**A trial is only valid and citable if it has zero mock-fallback queries
across all four methods.** When a real API call fails (typically a 429 rate
limit), the runner falls back to a simulated answer so the run doesn't crash
— but that row's accuracy/cost numbers are then partially synthetic, not
measured. The runner always prints and saves the mock-fallback count per
method precisely so this can never be silently missed.

This project has logged one fully valid real trial (**Trial 1**, zero mock
fallback across all 108 queries) and two/three follow-up attempts that hit
the daily token cap partway through and are **excluded from all reported
results** as a result. See [`RESULTS.md`](./RESULTS.md#invalid-trial-attempts)
for the specifics — they're documented rather than deleted, for transparency
about what was tried and why it didn't produce usable data.

## 7. Automated weekly run (CI)

`.github/workflows/benchmark-bot.yml` runs the suite in **mock mode** every
Monday (plus manual trigger), and commits refreshed `benchmarks/results/`
JSON as `github-actions[bot]`. This is intentionally mock-only — it exists to
keep the results artifacts fresh and CI green, not to produce citable
numbers. Mock-mode output must never be presented as, or mixed into, the real
trial data in `RESULTS.md`.