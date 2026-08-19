"""Plot generation for benchmark results.

Why this module exists:
Visualize benchmark results with matplotlib to compare cost vs accuracy,
latency distribution, and accuracy vs document length.
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any

import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)


def load_results(results_dir: Path) -> Dict[str, List[Dict]]:
    """Load all result JSON files from the results directory."""
    results = {}
    for json_file in results_dir.glob("*.json"):
        if json_file.name == "summary.json":
            continue
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            # Determine config from filename
            name = json_file.stem
            results[name] = data
    return results


def generate_plots(results_dir: Path, output_dir: Path) -> None:
    """Generate plots from benchmark results."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load all results
    all_results = load_results(results_dir)
    if not all_results:
        logger.warning("No result files found.")
        return

    # 1. Cost vs Accuracy scatter plot
    fig, ax = plt.subplots(figsize=(8, 6))
    methods = []
    accuracies = []
    costs = []
    for name, data in all_results.items():
        if not data:
            continue
        # Compute average cost and accuracy
        total_cost = sum(d.get("cost", 0.0) for d in data)
        correct = sum(1 for d in data if d.get("correct", False))
        acc = correct / len(data) if data else 0
        avg_cost = total_cost / len(data) if data else 0
        methods.append(name.replace("_", " "))
        accuracies.append(acc)
        costs.append(avg_cost)

    # Use colors for different methods
    colors = plt.cm.tab10(np.linspace(0, 1, len(methods)))
    ax.scatter(costs, accuracies, c=colors, s=100, alpha=0.7)
    for i, name in enumerate(methods):
        ax.annotate(name, (costs[i], accuracies[i]), fontsize=8)

    ax.set_xlabel("Average Cost ($)")
    ax.set_ylabel("Accuracy")
    ax.set_title("Cost vs Accuracy")
    ax.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(output_dir / "cost_vs_accuracy.png", dpi=150)
    plt.close()

    # 2. Latency distribution box plot
    fig, ax = plt.subplots(figsize=(10, 6))
    latency_data = []
    labels = []
    for name, data in all_results.items():
        if not data:
            continue
        latencies = [d.get("latency_ms", 0) for d in data if d.get("latency_ms") is not None]
        if latencies:
            latency_data.append(latencies)
            labels.append(name.replace("_", " "))
    if latency_data:
        ax.boxplot(latency_data, labels=labels, showmeans=True)
        ax.set_ylabel("Latency (ms)")
        ax.set_title("Latency Distribution by Method")
        ax.grid(True, linestyle='--', alpha=0.3)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(output_dir / "latency_boxplot.png", dpi=150)
    plt.close()

    # 3. Accuracy vs document length (if we have document sizes)
    # For simplicity, we'll just plot accuracy per method with error bars from bootstrap
    # Instead, we can use the number of sub-calls as proxy for complexity
    # But we'll skip this for now.

    logger.info(f"Plots saved to {output_dir}")