#!/usr/bin/env python3
"""
ARC-AGI Full Benchmark Evaluation CLI

Two modes:
  train — Full access to answers. Learns culture (concepts, programs) and
          saves it to a culture file for later use during eval.
  eval  — Uses answers ONLY for scoring/reporting. Loads culture from
          a previously saved culture file but does NOT learn from answers.

Quick start:
    git clone https://github.com/fchollet/ARC-AGI.git

    # Train: learn culture from training set
    python -m arc_agent.evaluate train \\
        --data-dir ARC-AGI/data/training \\
        --culture-file culture.json \\
        --output results_train.json

    # Eval: apply culture to held-out evaluation set
    python -m arc_agent.evaluate eval \\
        --data-dir ARC-AGI/data/evaluation \\
        --culture-file culture.json \\
        --output results_eval.json

Additional options:
    --limit N       Evaluate only the first N tasks (sorted by ID)
    --workers N     Parallel workers (0 = auto, 1 = debug mode)
    --population N  Evolutionary population size (default: 60)
    --generations N Max evolutionary generations (default: 30)
    --seed N        Random seed for reproducibility (default: 42)
    --quiet         Suppress per-task output

Reproducibility:
    All runs are seeded. Use --seed to change the seed; the same (seed, workers)
    pair always produces identical results.
"""
import argparse
import os
import sys
from .dataset import load_dataset, evaluate_dataset
from .cpu_utils import default_workers, describe_cpu


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    """Add arguments shared by both train and eval subcommands."""
    parser.add_argument(
        "--data-dir", required=True,
        help="Path to directory containing ARC-AGI task JSON files",
    )
    parser.add_argument(
        "--culture-file", default="",
        help=(
            "Path to culture JSON file. "
            "Train mode: saves learned culture here after run. "
            "Eval mode: loads pre-trained culture from here before run."
        ),
    )
    parser.add_argument(
        "--output", default="",
        help="Save full results to this JSON file",
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
        "--quiet", action="store_true",
        help="Suppress per-task output; only print the final summary",
    )


def _run(args: argparse.Namespace, mode: str) -> int:
    """Shared run logic for both train and eval modes."""
    if not os.path.isdir(args.data_dir):
        print(f"Error: {args.data_dir!r} is not a directory", file=sys.stderr)
        return 1

    print(f"Mode: {mode.upper()}")
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

    # In train mode: culture-file is where we SAVE culture after the run.
    # In eval mode:  culture-file is where we LOAD culture from before the run.
    load_culture_path = ""
    save_culture_path = ""

    if mode == "train":
        save_culture_path = args.culture_file
        if save_culture_path:
            print(f"Will save culture to: {save_culture_path}")
    elif mode == "eval":
        load_culture_path = args.culture_file
        if load_culture_path:
            if not os.path.isfile(load_culture_path):
                print(
                    f"Warning: culture file {load_culture_path!r} not found. "
                    f"Running eval without culture transfer.",
                    file=sys.stderr,
                )
                load_culture_path = ""
            else:
                print(f"Loading culture from: {load_culture_path}")

    print()
    evaluate_dataset(
        tasks,
        population_size=args.population,
        max_generations=args.generations,
        verbose=not args.quiet,
        output_path=args.output,
        seed=args.seed,
        workers=args.workers,
        load_culture_path=load_culture_path,
        save_culture_path=save_culture_path,
        mode=mode,
    )

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Four Pillars AGI Agent — ARC-AGI Benchmark Evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    subparsers = parser.add_subparsers(dest="mode", help="Operating mode")

    # Train subcommand
    train_parser = subparsers.add_parser(
        "train",
        help=(
            "Train mode: full access to answers. Learns culture and "
            "saves it to --culture-file for later eval use."
        ),
    )
    _add_common_args(train_parser)

    # Eval subcommand
    eval_parser = subparsers.add_parser(
        "eval",
        help=(
            "Eval mode: answers used ONLY for scoring/reporting. "
            "Loads culture from --culture-file but does NOT learn from answers."
        ),
    )
    _add_common_args(eval_parser)

    args = parser.parse_args()

    if not args.mode:
        parser.error("please specify a mode: 'train' or 'eval'")
        return 1  # unreachable, parser.error exits

    return _run(args, args.mode)


if __name__ == "__main__":
    sys.exit(main())
