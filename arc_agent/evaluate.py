#!/usr/bin/env python3
"""
ARC-AGI Benchmark CLI — Train, Infer, Eval

Three modes with clean separation of concerns:

  train — Full access to training answers. Learns culture (concepts,
          programs) and saves to a culture file. Runs exhaustive search.
          Use on the training set to build up knowledge.

  infer — Runs the solver and outputs ranked candidates per task.
          Never looks at test output. Loads culture from training.
          Output is a JSON file with candidates + programs that can
          be submitted to private eval or scored locally with `eval`.

  eval  — Scores inference results against expected test output.
          Produces the final scoreboard. Can run inline (solver + score)
          or score a previously saved inference output file.

Typical workflow:
    git clone https://github.com/fchollet/ARC-AGI.git

    # 1. Train: learn culture from training set
    python -m arc_agent.evaluate train \\
        --data-dir ARC-AGI/data/training \\
        --culture-file culture.json \\
        --output results_train.json

    # 2. Infer: generate candidates for eval set (no peeking at answers)
    python -m arc_agent.evaluate infer \\
        --data-dir ARC-AGI/data/evaluation \\
        --culture-file culture.json \\
        --output predictions_eval.json

    # 3. Eval: score predictions against answers
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
    --top-k N       Candidates to submit per task (default: 3)
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
    """Add arguments shared by all subcommands."""
    parser.add_argument(
        "--data-dir", required=True,
        help="Path to directory containing ARC-AGI task JSON files",
    )
    parser.add_argument(
        "--culture-file", default="",
        help=(
            "Path to culture JSON file. "
            "Train mode: saves learned culture here after run. "
            "Infer/Eval mode: loads pre-trained culture from here."
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
        "--tasks", nargs="+", default=None,
        help=(
            "Run only specific task IDs (space-separated). "
            "E.g.: --tasks 0b148d64 2204b7a8 3c9b0459"
        ),
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
    parser.add_argument(
        "--top-k", type=int, default=3, dest="top_k",
        help=(
            "Number of diverse candidates to submit per task (default: 3). "
            "Each pixel-perfect candidate is tested independently against "
            "the held-out test output; if ANY candidate passes, the task "
            "counts as solved. Higher values increase test-confirmation "
            "rate at no extra compute cost."
        ),
    )
    parser.add_argument(
        "--compute-cap", type=int, default=1_500_000, dest="compute_cap",
        help=(
            "Cell-normalized compute cap (default: 1,500,000). "
            "Per-task eval budget = compute_cap / avg_cells. "
            "Normalizes for ~200x variation in eval cost by grid "
            "size. Recommended: 200,000 for fast iteration "
            "(~2 min with 8 workers), 1,500,000 for full "
            "deterministic search, 0 to disable (unlimited)."
        ),
    )
    parser.add_argument(
        "--time-limit", type=float, default=0.0, dest="time_limit",
        help=(
            "Maximum wall-clock seconds per task (default: 0 = unlimited). "
            "When set, search phases are skipped once the time limit is "
            "reached. WARNING: non-deterministic, use only for "
            "development speed, not reproducible benchmarks."
        ),
    )


def _run(args: argparse.Namespace, mode: str) -> int:
    """Shared run logic for train, infer, and eval modes."""
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

    # Filter tasks by specific IDs (--tasks) or by count (--limit)
    if args.tasks:
        requested = set(args.tasks)
        matched = {tid: tasks[tid] for tid in args.tasks if tid in tasks}
        missing = requested - set(matched.keys())
        if missing:
            print(f"Warning: task IDs not found: {', '.join(sorted(missing))}",
                  file=sys.stderr)
        tasks = matched
        print(f"Selected {len(tasks)} specific tasks: {', '.join(sorted(tasks.keys()))}")
    elif args.limit > 0:
        sorted_ids = sorted(tasks.keys())[: args.limit]
        tasks = {tid: tasks[tid] for tid in sorted_ids}
        print(f"Limited to first {args.limit} tasks (sorted by ID)")

    # Culture file handling depends on mode:
    #   train: saves culture after run
    #   infer: loads culture before run (read-only)
    #   eval:  loads culture before run (read-only)
    load_culture_path = ""
    save_culture_path = ""

    if mode == "train":
        save_culture_path = args.culture_file
        if save_culture_path:
            print(f"Will save culture to: {save_culture_path}")
    else:
        # infer and eval both load culture
        load_culture_path = args.culture_file
        if load_culture_path:
            if not os.path.isfile(load_culture_path):
                print(
                    f"Warning: culture file {load_culture_path!r} not found. "
                    f"Running {mode} without culture transfer.",
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
        top_k=args.top_k,
        compute_cap=args.compute_cap,
        time_limit=args.time_limit,
    )

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Four Pillars AGI Agent — ARC-AGI Benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    subparsers = parser.add_subparsers(dest="mode", help="Operating mode")

    # Train subcommand
    train_parser = subparsers.add_parser(
        "train",
        help=(
            "Train mode: full access to answers. Learns culture and "
            "saves it to --culture-file for later use."
        ),
    )
    _add_common_args(train_parser)

    # Infer subcommand
    infer_parser = subparsers.add_parser(
        "infer",
        help=(
            "Inference mode: runs solver, outputs ranked candidates per task. "
            "Never looks at test output. Loads culture from --culture-file."
        ),
    )
    _add_common_args(infer_parser)

    # Eval subcommand
    eval_parser = subparsers.add_parser(
        "eval",
        help=(
            "Eval mode: runs solver and scores against expected test output. "
            "Loads culture from --culture-file. Produces final scoreboard."
        ),
    )
    _add_common_args(eval_parser)

    args = parser.parse_args()

    if not args.mode:
        parser.error("please specify a mode: 'train', 'infer', or 'eval'")
        return 1  # unreachable, parser.error exits

    return _run(args, args.mode)


if __name__ == "__main__":
    sys.exit(main())
