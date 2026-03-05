#!/usr/bin/env python3
"""
ARC-AGI Full Benchmark Evaluation Script

Downloads (if needed) and evaluates the Four Pillars solver on the
full ARC-AGI-1 dataset (400 training + 400 evaluation tasks).

Usage:
    # Evaluate on ARC-AGI-1 training set
    python -m arc_agent.evaluate --data-dir path/to/ARC-AGI/data/training

    # Evaluate on evaluation set
    python -m arc_agent.evaluate --data-dir path/to/ARC-AGI/data/evaluation

    # Save results to file
    python -m arc_agent.evaluate --data-dir data/training --output results.json

    # With persistence (save learned toolkit for later)
    python -m arc_agent.evaluate --data-dir data/training --save-toolkit toolkit.json

    # Run on first N tasks only (for quick testing)
    python -m arc_agent.evaluate --data-dir data/training --limit 20

Setup:
    1. Clone ARC-AGI dataset:
       git clone https://github.com/fchollet/ARC-AGI.git

    2. Run evaluation:
       python -m arc_agent.evaluate --data-dir ARC-AGI/data/training
"""
import argparse
import json
import os
import random
import time
from .dataset import load_dataset, evaluate_dataset
from .persistence import save_toolkit, save_archive
from .solver import FourPillarsSolver


def main():
    parser = argparse.ArgumentParser(
        description="Four Pillars AGI Agent — ARC-AGI Benchmark Evaluation"
    )
    parser.add_argument(
        "--data-dir", type=str, required=True,
        help="Path to directory with ARC-AGI task JSON files"
    )
    parser.add_argument(
        "--output", type=str, default="",
        help="Save results to this JSON file"
    )
    parser.add_argument(
        "--save-toolkit", type=str, default="",
        help="Save learned toolkit to this JSON file after evaluation"
    )
    parser.add_argument(
        "--save-archive", type=str, default="",
        help="Save archive to this JSON file after evaluation"
    )
    parser.add_argument(
        "--limit", type=int, default=0,
        help="Evaluate only the first N tasks (0 = all)"
    )
    parser.add_argument(
        "--population", type=int, default=60,
        help="Evolutionary population size (default: 60)"
    )
    parser.add_argument(
        "--generations", type=int, default=30,
        help="Max generations per task (default: 30)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)"
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress per-task output"
    )
    args = parser.parse_args()

    if not os.path.isdir(args.data_dir):
        print(f"Error: {args.data_dir} is not a directory")
        return 1

    # Load dataset
    print(f"Loading tasks from: {args.data_dir}")
    tasks = load_dataset(args.data_dir)
    print(f"Found {len(tasks)} tasks")

    if args.limit > 0:
        # Take first N tasks (sorted by ID for reproducibility)
        sorted_ids = sorted(tasks.keys())[:args.limit]
        tasks = {tid: tasks[tid] for tid in sorted_ids}
        print(f"Limited to first {args.limit} tasks")

    if not tasks:
        print("No tasks found. Check the --data-dir path.")
        return 1

    # Run evaluation
    print()
    results = evaluate_dataset(
        tasks,
        population_size=args.population,
        max_generations=args.generations,
        verbose=not args.quiet,
        output_path=args.output,
        seed=args.seed,
    )

    if args.output:
        print(f"\nResults saved to: {args.output}")

    return 0


if __name__ == "__main__":
    exit(main())
