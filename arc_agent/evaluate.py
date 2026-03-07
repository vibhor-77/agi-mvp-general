#!/usr/bin/env python3
"""
ARC-AGI Full Benchmark Evaluation CLI

Runs the Four Pillars solver on the ARC-AGI dataset and reports metrics.

Quick start:
    git clone https://github.com/fchollet/ARC-AGI.git
    python -m arc_agent.evaluate --data-dir ARC-AGI/data/training

Usage examples:
    # Full training set, auto-detect worker count (default)
    python -m arc_agent.evaluate --data-dir ARC-AGI/data/training

    # Held-out evaluation set
    python -m arc_agent.evaluate --data-dir ARC-AGI/data/evaluation

    # Limit to first 20 tasks for a quick sanity check
    python -m arc_agent.evaluate --data-dir ARC-AGI/data/training --limit 20

    # Force single-process mode (easiest to debug)
    python -m arc_agent.evaluate --data-dir ARC-AGI/data/training --workers 1

    # Save results JSON
    python -m arc_agent.evaluate --data-dir ARC-AGI/data/training --output results.json

    # Two-phase pipeline: learn from training, apply to evaluation
    python -m arc_agent.evaluate --data-dir ARC-AGI/data/training --save-culture culture.json
    python -m arc_agent.evaluate --data-dir ARC-AGI/data/evaluation --load-culture culture.json

    # Save learned toolkit for later
    python -m arc_agent.evaluate --data-dir ARC-AGI/data/training --save-toolkit toolkit.json

Reproducibility:
    All runs are seeded. Use --seed to change the seed; the same (seed, workers)
    pair always produces identical results.
"""
import argparse
import os
import sys
from .dataset import load_dataset, evaluate_dataset
from .cpu_utils import default_workers, describe_cpu


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Four Pillars AGI Agent — ARC-AGI Benchmark Evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--data-dir", required=True,
        help="Path to directory containing ARC-AGI task JSON files",
    )
    parser.add_argument(
        "--output", default="",
        help="Save full results to this JSON file",
    )
    parser.add_argument(
        "--save-toolkit", default="",
        help="Save the learned toolkit to this JSON file after evaluation",
    )
    parser.add_argument(
        "--limit", type=int, default=0,
        help="Evaluate only the first N tasks (0 = all; tasks are sorted by ID)",
    )
    parser.add_argument(
        "--workers", type=int, default=0,
        help=(
            "Number of parallel worker processes. "
            "0 (default) = auto-detect performance cores. "
            "1 = single-process mode (easiest to debug). "
            f"Auto on this machine = {default_workers()} ({describe_cpu()})"
        ),
    )
    parser.add_argument(
        "--population", type=int, default=60,
        help="Evolutionary population size per worker (default: 60)",
    )
    parser.add_argument(
        "--generations", type=int, default=30,
        help="Max evolutionary generations per task (default: 30)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help=(
            "Global random seed for reproducibility (default: 42). "
            "Worker seeds are derived as seed + worker_index * 1000."
        ),
    )
    parser.add_argument(
        "--save-culture", default="",
        help=(
            "Save learned culture (concepts, programs) to this JSON file "
            "after evaluation. Use with training set to build a culture file."
        ),
    )
    parser.add_argument(
        "--load-culture", default="",
        help=(
            "Load pre-trained culture from this JSON file before evaluation. "
            "Use with evaluation set to apply knowledge learned from training."
        ),
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress per-task output; only print the final summary",
    )
    args = parser.parse_args()

    if not os.path.isdir(args.data_dir):
        print(f"Error: {args.data_dir!r} is not a directory", file=sys.stderr)
        return 1

    print(f"Loading tasks from: {args.data_dir}")
    tasks = load_dataset(args.data_dir)
    print(f"Found {len(tasks)} tasks")

    if not tasks:
        print("No tasks found. Check the --data-dir path.", file=sys.stderr)
        return 1

    if args.limit > 0:
        sorted_ids = sorted(tasks.keys())[: args.limit]
        tasks = {tid: tasks[tid] for tid in sorted_ids}
        print(f"Limited to first {args.limit} tasks (sorted by ID)")

    print()
    evaluate_dataset(
        tasks,
        population_size=args.population,
        max_generations=args.generations,
        verbose=not args.quiet,
        output_path=args.output,
        seed=args.seed,
        workers=args.workers,
        load_culture_path=args.load_culture,
        save_culture_path=args.save_culture,
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
