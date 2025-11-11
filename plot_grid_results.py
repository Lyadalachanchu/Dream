#!/usr/bin/env python3
"""Plot grid test results with different aggregation modes."""
from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib

# Use a non-interactive backend for headless environments.
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("logs/grid_test_results.json"),
        help="Path to grid test results JSON file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("logs/grid_test_plot.png"),
        help="Where to save the generated plot.",
    )
    parser.add_argument(
        "--mode",
        choices=("raw", "batch"),
        default="batch",
        help="Aggregation mode: 'raw' keeps every logged step; 'batch' averages the 100-step GPU batches.",
    )
    parser.add_argument(
        "--n",
        type=int,
        nargs="+",
        help="Only include entries with these total particle counts n.",
    )
    parser.add_argument(
        "--reward-particles",
        type=int,
        nargs="+",
        help="Filter to these reward particle (m) counts.",
    )
    return parser.parse_args()


def aggregate_raw_steps(runs: Iterable[Dict]) -> Tuple[List[int], List[float], List[float]]:
    per_step: Dict[int, List[float]] = {}
    for run in runs:
        for record in run.get("steps", []):
            per_step.setdefault(record["step"], []).append(record["average_reward"])

    step_ids = sorted(per_step.keys())
    means = [statistics.fmean(per_step[idx]) for idx in step_ids]
    stds = [
        statistics.pstdev(per_step[idx]) if len(per_step[idx]) > 1 else 0.0
        for idx in step_ids
    ]
    return step_ids, means, stds


def aggregate_batch_profile(entry: Dict) -> Tuple[List[int], List[float], List[float]]:
    runs = entry.get("runs", [])
    n = entry.get("n")
    max_gpu_n = entry.get("max_gpu_n")
    if not runs or not n or not max_gpu_n:
        raise ValueError("Entry must include runs, n, and max_gpu_n for batch aggregation.")

    batches_per_run = max(1, math.ceil(n / max_gpu_n))
    per_position: Dict[int, List[float]] = {}
    steps_per_batch = None

    for run in runs:
        records = sorted(run.get("steps", []), key=lambda r: r["step"])
        total_records = len(records)
        if total_records == 0:
            continue
        if total_records % batches_per_run != 0:
            raise ValueError(
                f"Run {run.get('run_id')} has {total_records} records which is not divisible "
                f"by batches_per_run={batches_per_run}."
            )
        run_steps_per_batch = total_records // batches_per_run
        steps_per_batch = steps_per_batch or run_steps_per_batch
        if steps_per_batch != run_steps_per_batch:
            raise ValueError(
                f"Inconsistent steps-per-batch: expected {steps_per_batch}, "
                f"got {run_steps_per_batch} for run {run.get('run_id')}."
            )
        for batch_idx in range(batches_per_run):
            start = batch_idx * run_steps_per_batch
            batch_records = records[start : start + run_steps_per_batch]
            for pos, record in enumerate(batch_records):
                per_position.setdefault(pos, []).append(record["average_reward"])

    if not per_position or steps_per_batch is None:
        raise ValueError("No batch data collected.")

    positions = list(range(steps_per_batch))
    steps = [pos + 1 for pos in positions]
    means = [statistics.fmean(per_position[pos]) for pos in positions]
    stds = [
        statistics.pstdev(per_position[pos]) if len(per_position[pos]) > 1 else 0.0
        for pos in positions
    ]
    return steps, means, stds


def main() -> None:
    args = parse_args()
    with args.input.open() as f:
        data = json.load(f)

    if args.n:
        allowed_n = set(args.n)
        data = [entry for entry in data if entry.get("n") in allowed_n]
    if args.reward_particles:
        allowed = set(args.reward_particles)
        data = [entry for entry in data if entry.get("reward_particles") in allowed]

    if not data:
        raise SystemExit("No entries found in the grid test results with the given filters.")

    plt.figure(figsize=(10, 6))
    colors = plt.cm.tab10.colors
    xlabel = "Diffusion Step Within Batch" if args.mode == "batch" else "Step"

    for idx, entry in enumerate(data):
        color = colors[idx % len(colors)]
        if args.mode == "batch":
            steps, means, stds = aggregate_batch_profile(entry)
        else:
            steps, means, stds = aggregate_raw_steps(entry.get("runs", []))

        label = f"rp={entry.get('reward_particles')} n={entry.get('n')}"
        plt.plot(steps, means, label=label, color=color, linewidth=1.8)
        lower = [m - s for m, s in zip(means, stds)]
        upper = [m + s for m, s in zip(means, stds)]
        plt.fill_between(steps, lower, upper, color=color, alpha=0.2)

    plt.xlabel(xlabel)
    plt.ylabel("Average Reward")
    plt.title("Grid Test Results (Mean ± 1 Std)")
    plt.legend()
    plt.grid(alpha=0.3, linestyle="--", linewidth=0.5)
    plt.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.output, dpi=200)
    print(f"Saved plot to {args.output}")


if __name__ == "__main__":
    main()
