"""
ARC-AGI Dataset Loader and Parallel Evaluation Harness

Loads the official ARC-AGI dataset (JSON files) and runs the Four Pillars
solver on them, collecting full metrics.

Dataset structure (ARC-AGI-1):
  data/training/    — 400 tasks for development
  data/evaluation/  — 400 tasks for held-out evaluation

Each task is a JSON file:
  {"train": [{"input": [[int]], "output": [[int]]}],
   "test":  [{"input": [[int]], "output": [[int]]}]}

Grids are rectangular matrices of ints 0-9, sizes 1×1 to 30×30.

Parallel evaluation:
  Tasks are split round-robin across worker processes. Each worker runs
  an independent FourPillarsSolver so there is no shared mutable state.
  Worker seeds are derived deterministically from the global seed, so
  results are reproducible for any given (seed, workers) combination.
  Results are merged and sorted by task_id for consistent output.

  Default worker count = performance core count (see cpu_utils.py).
  On Apple Silicon M-series this is the P-core count, not the total.
"""
from __future__ import annotations
import json
import os
import time
import random
import multiprocessing
from .solver import FourPillarsSolver
from .scorer import validate_on_test
from .cpu_utils import default_workers, describe_cpu


def load_task(path: str) -> dict:
    """Load a single ARC-AGI task from a JSON file."""
    with open(path) as f:
        return json.load(f)


def load_dataset(directory: str) -> dict[str, dict]:
    """Load all ARC-AGI tasks from a directory.

    Returns a dict mapping task_id → task, sorted alphabetically by task_id
    so that --limit N always picks the same N tasks regardless of OS.
    """
    tasks: dict[str, dict] = {}
    if not os.path.isdir(directory):
        return tasks

    for filename in sorted(os.listdir(directory)):
        if not filename.endswith(".json"):
            continue
        task_id = filename[:-5]
        try:
            tasks[task_id] = load_task(os.path.join(directory, filename))
        except (json.JSONDecodeError, OSError) as exc:
            print(f"  Warning: skipping {filename}: {exc}")

    return tasks


def _solve_chunk(args: tuple) -> dict:
    """Worker function: solve a chunk of tasks with an independent solver.

    Designed to run in a subprocess via multiprocessing.Pool.  Each worker
    creates its own FourPillarsSolver so there is zero shared mutable state
    between workers.

    The random seed is derived from (global_seed + worker_index * 1000) so
    results are fully reproducible for any given (seed, workers) pair.

    Args:
        args: (task_chunk, population_size, max_generations, seed)
              task_chunk: list of (task_id, task_dict) in sorted order

    Returns:
        Dict:
          task_results  — {task_id: result_dict} for each task in the chunk
          learned_names — names of programs promoted into the toolkit
          final_toolkit — toolkit size at end of this chunk
    """
    task_chunk, population_size, max_generations, seed = args

    random.seed(seed)

    solver = FourPillarsSolver(
        population_size=population_size,
        max_generations=max_generations,
        verbose=False,
    )

    task_results: dict[str, dict] = {}
    for task_id, task in task_chunk:
        result = solver.solve_task(task, task_id)

        test_passed = False
        test_score  = 0.0
        if result["solved"]:
            programs = solver.archive.task_solutions.get(task_id, [])
            if programs:
                test_passed, test_score = validate_on_test(programs[0], task)

        task_results[task_id] = {
            "solved":         result["solved"],
            "score":          result["score"],
            "test_passed":    test_passed,
            "test_score":     test_score,
            "program":        result["program"],
            "program_length": result["program_length"],
            "method":         result["method"],
            "time_seconds":   result["time_seconds"],
            "toolkit_size":   result["toolkit_size"],
        }

    return {
        "task_results":  task_results,
        "learned_names": [p.name for p in solver.toolkit.programs],
        "final_toolkit": solver.toolkit.size,
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

    Args:
        tasks:              task_id → task dict (from load_dataset)
        population_size:    evolutionary population size per worker
        max_generations:    max generations per task
        max_program_length: max program chain length (unused directly here,
                            passed through for future use)
        verbose:            print per-task and summary output
        output_path:        if set, save results JSON to this path
        seed:               global random seed — results are fully reproducible
                            for any fixed (seed, workers) combination
        workers:            number of parallel processes
                            0  → default_workers() (performance cores only)
                            1  → single-process, easiest to debug
                            N  → exactly N processes

    Returns:
        Dict with keys:
          "task_results" — {task_id: per-task metrics}
          "summary"      — aggregate benchmark metrics
    """
    random.seed(seed)

    n_workers = workers if workers > 0 else default_workers()
    n_workers = max(1, min(n_workers, len(tasks)))

    sorted_ids = sorted(tasks.keys())
    n_tasks    = len(sorted_ids)

    if verbose:
        _tmp = FourPillarsSolver(verbose=False)
        print(f"CPU: {describe_cpu()}")
        print(f"Workers: {n_workers}  |  Tasks: {n_tasks}  |  "
              f"Seed: {seed}  |  "
              f"Initial toolkit: {_tmp.toolkit.size} concepts")
        del _tmp
        print()

    # Distribute tasks round-robin so chunks are roughly equal size
    chunks: list[list] = [[] for _ in range(n_workers)]
    for i, task_id in enumerate(sorted_ids):
        chunks[i % n_workers].append((task_id, tasks[task_id]))

    # Deterministic per-worker seeds: seed + worker_index * 1000
    worker_args = [
        (chunk, population_size, max_generations, seed + idx * 1000)
        for idx, chunk in enumerate(chunks)
    ]

    start_time = time.time()

    if n_workers == 1:
        # In-process path — useful for debugging and single-core machines
        worker_outputs = [_solve_chunk(worker_args[0])]
    else:
        with multiprocessing.Pool(processes=n_workers) as pool:
            worker_outputs = pool.map(_solve_chunk, worker_args)

    total_time = time.time() - start_time

    # Merge: collect task_results in sorted order for deterministic output
    task_results: dict[str, dict] = {}
    for output in worker_outputs:
        task_results.update(output["task_results"])

    solved_count  = 0
    partial_count = 0
    test_correct  = 0
    all_learned:  list[str] = []

    for output in worker_outputs:
        all_learned.extend(output["learned_names"])

    for task_id in sorted_ids:
        r = task_results[task_id]
        if r["solved"]:
            solved_count += 1
            if r["test_passed"]:
                test_correct += 1
        elif r["score"] > 0.8:
            partial_count += 1

    if verbose:
        for i, task_id in enumerate(sorted_ids):
            r = task_results[task_id]
            status   = "✓" if r["solved"] else ("~" if r["score"] > 0.8 else "✗")
            test_str = f"test={'✓' if r['test_passed'] else '✗'}" if r["solved"] else ""
            print(f"  [{i+1:3d}/{n_tasks}] {status} {task_id[:20]:20s} "
                  f"score={r['score']:.3f} {test_str} "
                  f"({r['time_seconds']:.2f}s) tk={r['toolkit_size']}")

    final_toolkit = max((o["final_toolkit"] for o in worker_outputs), default=0)

    summary = {
        "total_tasks":        n_tasks,
        "solved_exact":       solved_count,
        "solve_rate":         solved_count / max(n_tasks, 1),
        "partial_solved":     partial_count,
        "test_correct":       test_correct,
        "test_rate":          test_correct / max(n_tasks, 1),
        "total_time_seconds": total_time,
        "avg_time_per_task":  total_time / max(n_tasks, 1),
        "final_toolkit_size": final_toolkit,
        "concepts_learned":   len(all_learned),
        "workers_used":       n_workers,
        "seed":               seed,
    }

    if verbose:
        print(f"\n{'='*60}")
        print("BENCHMARK RESULTS")
        print(f"{'='*60}")
        print(f"Solved (exact): {solved_count}/{n_tasks} "
              f"({100*summary['solve_rate']:.1f}%)")
        print(f"Partial (>80%): {partial_count}/{n_tasks}")
        print(f"Test correct:   {test_correct}/{n_tasks} "
              f"({100*summary['test_rate']:.1f}%)")
        print(f"Total time:     {total_time:.1f}s "
              f"({summary['avg_time_per_task']:.2f}s/task avg)")
        print(f"Workers:        {n_workers}")
        print(f"Toolkit:        {final_toolkit} concepts "
              f"({len(all_learned)} learned across workers)")
        print(f"{'='*60}")

    result = {"task_results": task_results, "summary": summary}

    if output_path:
        with open(output_path, "w") as f:
            json.dump(result, f, indent=2)
        if verbose:
            print(f"\nResults saved to: {output_path}")

    return result
