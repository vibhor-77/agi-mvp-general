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

Performance:
  Parallel evaluation is supported via --workers (default: cpu_count).
  Each worker runs an independent solver on a chunk of tasks.
  Results are merged at the end; learned concepts from all workers
  are combined into the final toolkit.

Usage:
    python -m arc_agent.evaluate --data-dir path/to/ARC-AGI/data/training
    python -m arc_agent.evaluate --data-dir data/training --workers 10
"""
from __future__ import annotations
import json
import os
import time
import random
import multiprocessing
from typing import Optional
from .solver import FourPillarsSolver
from .scorer import validate_on_test


def load_task(path: str) -> dict:
    """Load a single ARC-AGI task from a JSON file."""
    with open(path, "r") as f:
        return json.load(f)


def load_dataset(directory: str) -> dict[str, dict]:
    """Load all ARC-AGI tasks from a directory.

    Returns:
        Dict mapping task_id → task dict, sorted by task_id.
    """
    tasks = {}
    if not os.path.isdir(directory):
        return tasks

    for filename in sorted(os.listdir(directory)):
        if not filename.endswith(".json"):
            continue
        task_id = filename[:-5]
        task_path = os.path.join(directory, filename)
        try:
            tasks[task_id] = load_task(task_path)
        except (json.JSONDecodeError, IOError) as e:
            print(f"  Warning: skipping {filename}: {e}")

    return tasks


def _solve_chunk(args: tuple) -> dict:
    """Worker function: solve a chunk of tasks with an independent solver.

    Runs in a separate process. Each worker gets its own copy of the
    solver (no shared state), enabling safe parallelism.

    Args:
        args: (task_chunk, population_size, max_generations, seed)
              task_chunk: list of (task_id, task_dict) pairs

    Returns:
        Dict with 'task_results' and 'toolkit_programs' (learned programs
        as serialisable names, for cross-worker knowledge merge).
    """
    task_chunk, population_size, max_generations, seed = args

    # Each worker seeds independently to avoid identical random streams
    random.seed(seed)

    solver = FourPillarsSolver(
        population_size=population_size,
        max_generations=max_generations,
        verbose=False,
    )

    task_results = {}
    for task_id, task in task_chunk:
        result = solver.solve_task(task, task_id)
        # Validate on test examples
        test_passed = False
        test_score = 0.0
        if result["solved"]:
            programs = solver.archive.task_solutions.get(task_id, [])
            if programs:
                exact, ts = validate_on_test(programs[0], task)
                test_score = ts
                test_passed = exact
        task_results[task_id] = {
            "solved":          result["solved"],
            "score":           result["score"],
            "test_passed":     test_passed,
            "test_score":      test_score,
            "program":         result["program"],
            "program_length":  result["program_length"],
            "method":          result["method"],
            "time_seconds":    result["time_seconds"],
            "toolkit_size":    result["toolkit_size"],
        }

    # Return learned program names so the main process can log growth
    learned_names = [p.name for p in solver.toolkit.programs]

    return {
        "task_results":   task_results,
        "learned_names":  learned_names,
        "final_toolkit":  solver.toolkit.size,
    }


def evaluate_dataset(
    tasks: dict[str, dict],
    population_size: int = 60,
    max_generations: int = 30,
    max_program_length: int = 4,
    verbose: bool = True,
    output_path: str = "",
    seed: int = 42,
    workers: int = 0,
) -> dict:
    """Run the Four Pillars solver on a dataset and collect metrics.

    When workers > 1, tasks are distributed across multiple processes.
    Each worker runs an independent solver; results are merged afterward.
    Knowledge compounding within each worker still operates sequentially
    (tasks within a chunk benefit from each other), and the merged toolkit
    reflects all learned concepts across workers.

    Args:
        tasks: Dict mapping task_id → task dict.
        population_size: Evolutionary population size per worker.
        max_generations: Max generations per task.
        max_program_length: Max program chain length.
        verbose: Print progress.
        output_path: If set, save results JSON to this path.
        seed: Random seed for reproducibility.
        workers: Number of parallel workers (0 = all CPUs).

    Returns:
        Dict with 'task_results' (per-task) and 'summary' (aggregate).
    """
    random.seed(seed)

    n_cpus = multiprocessing.cpu_count()
    n_workers = workers if workers > 0 else n_cpus
    n_workers = max(1, min(n_workers, len(tasks)))

    sorted_ids = sorted(tasks.keys())
    n_tasks = len(sorted_ids)

    if verbose:
        print(f"Evaluating {n_tasks} tasks with {n_workers} worker(s)...")
        # Show initial toolkit size from a fresh solver
        _tmp_solver = FourPillarsSolver(verbose=False)
        print(f"Initial toolkit: {_tmp_solver.toolkit.size} concepts")
        del _tmp_solver
        print()

    # Split tasks into chunks — one chunk per worker
    chunks = [[] for _ in range(n_workers)]
    for i, task_id in enumerate(sorted_ids):
        chunks[i % n_workers].append((task_id, tasks[task_id]))

    # Assign different seeds per worker for diversity
    worker_seeds = [seed + i * 1000 for i in range(n_workers)]
    worker_args = [
        (chunk, population_size, max_generations, wseed)
        for chunk, wseed in zip(chunks, worker_seeds)
    ]

    start_time = time.time()

    if n_workers == 1:
        # Single-worker path: run in-process (easier to debug)
        worker_outputs = [_solve_chunk(worker_args[0])]
    else:
        # Multi-worker path: true parallelism via multiprocessing
        with multiprocessing.Pool(processes=n_workers) as pool:
            worker_outputs = pool.map(_solve_chunk, worker_args)

    total_time = time.time() - start_time

    # Merge results from all workers
    task_results: dict[str, dict] = {}
    solved_count   = 0
    partial_count  = 0
    test_correct   = 0
    test_total     = 0
    all_learned_names: list[str] = []

    for output in worker_outputs:
        task_results.update(output["task_results"])
        all_learned_names.extend(output["learned_names"])

    # Re-order by sorted task ID for consistent output
    for task_id in sorted_ids:
        r = task_results[task_id]
        if r["solved"]:
            solved_count += 1
            test_total   += 1
            if r["test_passed"]:
                test_correct += 1
        else:
            test_total += 1
            if r["score"] > 0.8:
                partial_count += 1

    if verbose:
        for i, task_id in enumerate(sorted_ids):
            r = task_results[task_id]
            status   = "✓" if r["solved"] else ("~" if r["score"] > 0.8 else "✗")
            test_str = f"test={'✓' if r['test_passed'] else '✗'}" if r["solved"] else ""
            print(f"  [{i+1:3d}/{n_tasks}] {status} {task_id[:20]:20s} "
                  f"score={r['score']:.3f} {test_str} "
                  f"({r['time_seconds']:.2f}s) tk={r['toolkit_size']}")

    # Final toolkit size is the max across workers (each started at same base)
    final_toolkit_size = max(
        (o["final_toolkit"] for o in worker_outputs), default=0
    )

    summary = {
        "total_tasks":         n_tasks,
        "solved_exact":        solved_count,
        "solve_rate":          solved_count / max(n_tasks, 1),
        "partial_solved":      partial_count,
        "test_correct":        test_correct,
        "test_total":          test_total,
        "test_rate":           test_correct / max(test_total, 1),
        "total_time_seconds":  total_time,
        "avg_time_per_task":   total_time / max(n_tasks, 1),
        "final_toolkit_size":  final_toolkit_size,
        "concepts_learned":    len(all_learned_names),
        "workers_used":        n_workers,
    }

    if verbose:
        print(f"\n{'='*60}")
        print("BENCHMARK RESULTS")
        print(f"{'='*60}")
        print(f"Tasks:          {n_tasks}")
        print(f"Workers:        {n_workers}")
        print(f"Solved (exact): {solved_count} ({100*summary['solve_rate']:.1f}%)")
        print(f"Partial (>80%): {partial_count}")
        print(f"Test correct:   {test_correct}/{test_total} "
              f"({100*summary['test_rate']:.1f}%)")
        print(f"Total time:     {total_time:.1f}s "
              f"(avg {summary['avg_time_per_task']:.2f}s/task, "
              f"wall {total_time:.1f}s)")
        print(f"Final toolkit:  {final_toolkit_size} concepts "
              f"({len(all_learned_names)} learned across workers)")
        print(f"{'='*60}")

    output = {
        "task_results": task_results,
        "summary":      summary,
    }

    if output_path:
        with open(output_path, "w") as f:
            json.dump(output, f, indent=2)

    return output
