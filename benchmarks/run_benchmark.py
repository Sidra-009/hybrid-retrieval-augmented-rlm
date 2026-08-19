#!/usr/bin/env python
"""CLI entrypoint for running the full benchmark suite.

Usage:
    python benchmarks/run_benchmark.py --all
    python benchmarks/run_benchmark.py --baseline --hybrid
    python benchmarks/run_benchmark.py --plots
"""

import argparse
import json
import logging
from pathlib import Path
from typing import List, Dict, Any

from rich.console import Console
from rich.table import Table
from rich.progress import track

from benchmarks.baseline_rlm import BaselineRLMBenchmark
from benchmarks.hybrid_rlm import HybridRLMBenchmark
from benchmarks.plots import generate_plots

console = Console()
logger = logging.getLogger(__name__)


def load_dataset(dataset_path: Path) -> List[Dict]:
    with open(dataset_path, "r", encoding="utf-8") as f:
        return json.load(f)


def compute_summary(results: List[Dict]) -> Dict[str, float]:
    """Compute summary statistics from results."""
    if not results:
        return {}
    total = len(results)
    correct = sum(1 for r in results if r.get("correct", False))
    accuracy = correct / total if total > 0 else 0.0
    avg_cost = sum(r.get("cost", 0.0) for r in results) / total
    avg_tokens = sum(r.get("tokens", 0) for r in results) / total
    latencies = [r.get("latency_ms", 0) for r in results]
    latencies_sorted = sorted(latencies)
    p50 = latencies_sorted[int(0.5 * total)] if total > 0 else 0
    p95 = latencies_sorted[int(0.95 * total)] if total > 0 else 0
    return {
        "accuracy": accuracy,
        "avg_cost": avg_cost,
        "avg_tokens": avg_tokens,
        "p50_latency": p50,
        "p95_latency": p95,
    }


def print_summary_table(summaries: Dict[str, Dict[str, float]]) -> None:
    """Print a rich table of benchmark results."""
    table = Table(title="Benchmark Results")
    table.add_column("Method", style="cyan")
    table.add_column("Accuracy %", justify="right")
    table.add_column("Avg Cost ($)", justify="right")
    table.add_column("Avg Tokens", justify="right")
    table.add_column("P50 Latency (ms)", justify="right")
    table.add_column("P95 Latency (ms)", justify="right")

    for method, stats in summaries.items():
        table.add_row(
            method,
            f"{stats.get('accuracy', 0)*100:.1f}%",
            f"${stats.get('avg_cost', 0):.6f}",
            f"{stats.get('avg_tokens', 0):.0f}",
            f"{stats.get('p50_latency', 0):.2f}",
            f"{stats.get('p95_latency', 0):.2f}",
        )
    console.print(table)


def main():
    parser = argparse.ArgumentParser(description="Run HRA-RLM benchmarks")
    parser.add_argument("--baseline", action="store_true", help="Run baseline RLM")
    parser.add_argument("--hybrid", action="store_true", help="Run hybrid variants")
    parser.add_argument("--all", action="store_true", help="Run all benchmarks")
    parser.add_argument("--plots", action="store_true", help="Generate plots from results")
    parser.add_argument("--use-mock", action="store_true", default=False, help="Use mock LLM (default: False, i.e., use real LLM)")
    parser.add_argument("--dataset", type=str, default="benchmarks/datasets/dataset.json",
                        help="Path to dataset JSON")
    parser.add_argument("--output", type=str, default="benchmarks/results",
                        help="Directory to save results")
    parser.add_argument("--top-k", type=int, default=5, help="Top-K chunks for retrieval")
    args = parser.parse_args()

    # Set up logging
    logging.basicConfig(level=logging.INFO)

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        console.print(f"[red]Dataset not found: {dataset_path}. Run build_dataset.py first.[/red]")
        return

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    summaries = {}

    if args.all or args.baseline:
        console.print("[bold blue]Running Baseline RLM...[/bold blue]")
        baseline = BaselineRLMBenchmark(use_mock=args.use_mock)
        baseline.run_benchmark(dataset_path, output_dir)
        summaries["Baseline RLM"] = compute_summary(baseline.results)

    if args.all or args.hybrid:
        console.print("[bold blue]Running Hybrid RLM variants...[/bold blue]")
        hybrid = HybridRLMBenchmark(use_mock=args.use_mock, top_k=args.top_k)
        hybrid.run_benchmark(dataset_path, output_dir)
        for config, results in hybrid.results.items():
            summary = compute_summary(results)
            summaries[f"Hybrid ({config})"] = summary

    if summaries:
        print_summary_table(summaries)
        # Save summary to JSON
        summary_file = output_dir / "summary.json"
        with open(summary_file, "w") as f:
            json.dump(summaries, f, indent=2)
        console.print(f"[green]Summary saved to {summary_file}[/green]")

    if args.plots:
        console.print("[bold blue]Generating plots...[/bold blue]")
        plot_dir = Path("benchmarks/plots")
        generate_plots(output_dir, plot_dir)


if __name__ == "__main__":
    main()