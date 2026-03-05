"""
ARC-AGI Dataset Loader and Evaluation Harness

Loads the official ARC-AGI dataset (JSON files) and runs the
Four Pillars solver on them with full metrics tracking.

Dataset structure (ARC-AGI-1):
  data/training/   — 400 tasks for development
  data/evaluation/  — 400 tasks for held-out evaluation

Each task is a JSON file with:
  {
    "train": [{"input": [[int]], "output": [[int]]}],
    "test":  [{"input": [[int]], "output": [[int]]}]
  }

Grids are rectangular matrices of ints 0-9, sizes 1x1 to 30x30.

Usage:
    python -m arc_agent.evaluate --data-dir path/to/ARC-AGI/data/training
"""
from __future__ import annotations
import json
import os
import time
import random
from typing import Optional
from .solver import FourPillarsSolver
from .scorer import validate_on_test


def load_task(path: str) -> dict:
    """Load a single ARC-AGI task from a JSON file.

    Args:
        path: Path to a task JSON file.

    Returns:
        Task dict with 'train' and 'test' lists of examples.
    """
    with open(path, "r") as f:
        return json.load(f)


def load_dataset(directory: str) -> dict[str, dict]:
    """Load all ARC-AGI tasks from a directory.

    Reads every .json file in the directory, using the filename
    (without extension) as the task ID.

    Args:
        directory: Path to directory containing task JSON files.

    Returns:
        Dict mapping task_id → task dict.
    """
    tasks = {}
    if not os.path.isdir(directory):
        return tasks

    for filename in sorted(os.listdir(directory)):
        if not filename.endswith(".json"):
            continue
        task_id = filename[:-5]  # strip .json
        task_path = os.path.join(directory, filename)
        try:
            tasks[task_id] = load_task(task_path)
        except (json.JSONDecodeError, IOError) as e:
            print(f"  Warning: skipping {filename}: {e}")

    return tasks


def evaluate_dataset(
    tasks: dict[str, dict],
    population_size: int = 60,
    max_generations: int = 30,
    max_program_length: int = 4,
    verbose: bool = True,
    output_path: str = "",
    seed: int = 42,
) -> dict:
    """Run the Four Pillars solver on a dataset and collect metrics.

    The solver accumulates knowledge across tasks (cumulative culture),
    so task ordering matters. Tasks are sorted by ID for reproducibility.

    Args:
        tasks: Dict mapping task_id → task dict.
        population_size: Evolutionary population size.
        max_generations: Max generations per task.
        max_program_length: Max program chain length.
        verbose: Print progress.
        output_path: If set, save results JSON to this path.
        seed: Random seed for reproducibility.

    Returns:
        Dict with 'task_results' (per-task) and 'summary' (aggregate).
    """
    random.seed(seed)

    solver = FourPillarsSolver(
        population_size=population_size,
        max_generations=max_generations,
        max_program_length=max_program_length,
        verbose=False,  # Per-task output controlled below
    )

    task_results = {}
    solved_count = 0
    partial_count = 0
    test_correct = 0
    test_total = 0
    total_time = 0.0

    sorted_ids = sorted(tasks.keys())
    n_tasks = len(sorted_ids)

    if verbose:
        print(f"Evaluating {n_tasks} tasks...")
        print(f"Initial toolkit: {solver.toolkit.size} concepts")
        print()

    for i, task_id in enumerate(sorted_ids):
        task = tasks[task_id]
        start = time.time()
        result = solver.solve_task(task, task_id)
        elapsed = time.time() - start
        total_time += elapsed

        # Test validation
        test_passed = False
        test_score = 0.0
        if result["solved"]:
            solved_count += 1
            programs = solver.archive.task_solutions.get(task_id, [])
            if programs:
                exact, ts = validate_on_test(programs[0], task)
                test_score = ts
                test_total += 1
                if exact:
                    test_correct += 1
                    test_passed = True
        else:
            test_total += 1
            if result["score"] > 0.8:
                partial_count += 1

        task_results[task_id] = {
            "solved": result["solved"],
            "score": result["score"],
            "test_passed": test_passed,
            "test_score": test_score,
            "program": result["program"],
            "program_length": result["program_length"],
            "method": result["method"],
            "time_seconds": elapsed,
            "toolkit_size": result["toolkit_size"],
        }

        if verbose:
            status = "✓" if result["solved"] else ("~" if result["score"] > 0.8 else "✗")
            test_str = f"test={'✓' if test_passed else '✗'}" if result["solved"] else ""
            print(f"  [{i+1:3d}/{n_tasks}] {status} {task_id[:20]:20s} "
                  f"score={result['score']:.3f} {test_str} "
                  f"({elapsed:.2f}s) tk={result['toolkit_size']}")

    # Summary
    summary = {
        "total_tasks": n_tasks,
        "solved_exact": solved_count,
        "solve_rate": solved_count / max(n_tasks, 1),
        "partial_solved": partial_count,
        "test_correct": test_correct,
        "test_total": test_total,
        "test_rate": test_correct / max(test_total, 1),
        "total_time_seconds": total_time,
        "avg_time_per_task": total_time / max(n_tasks, 1),
        "final_toolkit_size": solver.toolkit.size,
        "concepts_learned": len(solver.toolkit.programs),
    }

    if verbose:
        print(f"\n{'='*60}")
        print("BENCHMARK RESULTS")
        print(f"{'='*60}")
        print(f"Tasks:          {n_tasks}")
        print(f"Solved (exact): {solved_count} ({100*summary['solve_rate']:.1f}%)")
        print(f"Partial (>80%): {partial_count}")
        print(f"Test correct:   {test_correct}/{test_total} "
              f"({100*summary['test_rate']:.1f}%)")
        print(f"Total time:     {total_time:.1f}s "
              f"(avg {summary['avg_time_per_task']:.2f}s/task)")
        print(f"Final toolkit:  {solver.toolkit.size} concepts "
              f"({len(solver.toolkit.programs)} learned)")
        print(f"{'='*60}")

    output = {
        "task_results": task_results,
        "summary": summary,
    }

    if output_path:
        with open(output_path, "w") as f:
            json.dump(output, f, indent=2)

    return output
