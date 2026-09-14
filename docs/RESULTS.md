# Results

See [`BENCHMARKS.md`](./BENCHMARKS.md) for how these numbers were
produced and what "valid trial" means for this project.

## Trial 1 — the current citable result

27 questions × 4 methods = 108 real Groq API calls, **zero mock-fallback**.
Retriever mode: `hnsw_semantic`. Embedding model: `all-MiniLM-L6-v2`. `top_k`:
3.

| Method | Acc (embedding) | Acc (keyword) | Est. cost/query | Tokens/query | Batch wall-clock (27 Qs) |
|---|---|---|---|---|---|
| Baseline RLM | 81.5% | 92.6% | $0.00030 | 2653 | 414.57s |
| Hybrid (fixed_k) | 85.2% | 85.2% | $0.00008 | 481 | 97.99s |
| Hybrid + AutoHealer | 81.5% | 85.2% | $0.00008 | 483 | 96.18s |
| Hybrid + Parallel | 77.8% | 85.2% | $0.00009 | 498 | 99.52s |

<p align="center">
  <img src="../assets/chart_tokens.png" width="48%" alt="Tokens per query by method">
  <img src="../assets/chart_cost.png" width="48%" alt="Estimated cost per query by method">
</p>
<p align="center">
  <img src="../assets/chart_accuracy.png" width="48%" alt="Accuracy by method, embedding vs keyword">
  <img src="../assets/chart_latency.png" width="48%" alt="Batch wall-clock latency by method">
</p>

### How this compares to the originally submitted abstract claims

| Claim | Trial 1 result | Verdict |
|---|---|---|
| ~50% lower token usage | ~82% lower (2653 → 481) | Exceeds claim |
| Up to 5× lower cost | ~3.75× lower ($0.00030 → $0.00008) | Close, slightly short |
| ≤2% accuracy loss | Metric-dependent: embedding score *improves* (81.5% → 85.2%); keyword score *drops* 7.4 points (92.6% → 85.2%) | Not a single clean number — both reported, neither cherry-picked |
| Up to 2× latency improvement via parallel execution | Batch wall-clock: baseline 414.57s → fastest (AutoHealer) 96.18s ≈ 4.3× | Exceeds claim, but note below |

**Note on the latency line:** in this trial, *AutoHealer* posted the fastest
batch wall-clock, not literally the *Parallel* configuration. Per-query p50
latency is the wrong metric for demonstrating parallel benefit (see
`BENCHMARKS.md`); batch wall-clock is reported here for exactly that
reason, and it's worth being upfront that the fastest method by that metric
wasn't the one the abstract's phrasing implies.

## Invalid trial attempts

Two further real-mode trials were attempted and are **excluded from the
table above**. Documenting them here rather than deleting them, since the
failure mode is a legitimate finding in itself (Groq free-tier daily quota,
not a bug in the method):

- Both attempts hit Groq's 200,000 tokens/day cap partway through, with the
  vast majority of queries per method falling back to mock (observed
  mock-fallback counts ranging from ~16/27 up to 27/27 depending on the
  method/row).
- Resulting accuracy figures in those runs (e.g. numbers as low as 0%–15% for
  some rows) are simulation artifacts of the mock fallback path, not measured
  model performance, and must not be averaged in with Trial 1 or cited
  anywhere.
- Status: **pending** either (a) a quota reset window with a clean run, or
  (b) an upgrade to Groq's paid Dev Tier, before a second valid trial can be
  added here.

## What would strengthen this further

- A second and third valid trial, to report a mean and [min–max] range per
  method rather than a single run (`--trials N` supports this already; it's
  a quota problem, not a tooling problem).
- An ablation over `--top-k` and `--embedding-model`, since both are exposed
  specifically to let the accuracy/cost trade-off be tuned and reported
  honestly rather than assumed.
