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
  Tasks are distributed across worker processes using imap_unordered,
  so results stream back as workers finish — you see live output and can
  Ctrl-C to abort at any time with a clean summary of work done so far.

  Worker seeds are derived deterministically from (global_seed + worker_index
  * 1000) so results are reproducible for any fixed (seed, workers) pair.
  Results are sorted by task_id in the final JSON output.

  Default worker count = performance core count (see cpu_utils.py).
  On Apple Silicon M-series this is the P-core count, not the total.
"""
from __future__ import annotations

import json
import os
import random
import statistics
import time
from typing import Iterator

import multiprocessing

from .solver import FourPillarsSolver
from .scorer import validate_on_test
from .cpu_utils import default_workers, describe_cpu


# ── Dataset loading ────────────────────────────────────────────────────────

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


# ── Worker ─────────────────────────────────────────────────────────────────

def _solve_one(args: tuple) -> dict:
    """Worker function: solve a single task in a subprocess.

    Using one-task-at-a-time (via imap_unordered) instead of chunked
    pool.map lets results stream back as soon as each task finishes,
    enabling live progress display and clean Ctrl-C abort.

    Args:
        args: (task_id, task_dict, population_size, max_generations, seed,
               solver_state_unused)
              The solver is created fresh per worker call so there is no
              shared mutable state between workers.

    Returns:
        Dict with task_id and all per-task metrics, plus a worker_seed
        for reproducibility bookkeeping.
    """
    task_id, task, population_size, max_generations, seed = args

    random.seed(seed)

    solver = FourPillarsSolver(
        population_size=population_size,
        max_generations=max_generations,
        verbose=False,
    )

    result = solver.solve_task(task, task_id)

    test_passed = False
    test_score  = 0.0
    if result["solved"]:
        programs = solver.archive.task_solutions.get(task_id, [])
        if programs:
            test_passed, test_score = validate_on_test(programs[0], task)

    return {
        "task_id":        task_id,
        "solved":         result["solved"],
        "score":          result["score"],
        "test_passed":    test_passed,
        "test_score":     test_score,
        "program":        result["program"],
        "program_length": result["program_length"],
        "method":         result["method"],
        "time_seconds":   result["time_seconds"],
        "toolkit_size":   result["toolkit_size"],
        "worker_seed":    seed,
    }


# ── Progress display ────────────────────────────────────────────────────────

class _ProgressTracker:
    """Accumulates per-task results and prints a live status line.

    Designed to be called from the main process as results stream in
    from imap_unordered, so it sees completed tasks one at a time.

    Output columns:
      [idx/total] symbol task_id   score=X.XXX  Xs  method  |  running stats
    """

    def __init__(self, n_tasks: int, n_workers: int, start_time: float):
        self.n_tasks    = n_tasks
        self.n_workers  = n_workers
        self.start_time = start_time

        # Accumulators
        self.done       = 0
        self.solved     = 0
        self.partial    = 0   # score > 0.80
        self.test_ok    = 0
        self.scores:    list[float] = []
        self.times:     list[float] = []

    def update(self, r: dict) -> None:
        """Record a completed task result and print one progress line."""
        self.done += 1
        self.scores.append(r["score"])
        self.times.append(r["time_seconds"])

        if r["solved"]:
            self.solved += 1
            if r["test_passed"]:
                self.test_ok += 1
        elif r["score"] > 0.80:
            self.partial += 1

        self._print_task_line(r)

        # Print a rolling summary every 10 tasks (or at the very end).
        if self.done % 10 == 0 or self.done == self.n_tasks:
            self._print_rolling_summary()

    def _print_task_line(self, r: dict) -> None:
        """Print one line per completed task with all relevant metrics."""
        status   = "✓" if r["solved"] else ("~" if r["score"] > 0.80 else "✗")
        elapsed  = time.time() - self.start_time
        # Projected finish time
        rate     = self.done / max(elapsed, 0.001)          # tasks/s wall-clock
        remaining_tasks = self.n_tasks - self.done
        eta_sec  = remaining_tasks / rate if rate > 0 else 0
        eta_str  = _fmt_duration(eta_sec)

        test_tag = f" test={'✓' if r['test_passed'] else '✗'}" if r["solved"] else ""
        method   = r.get("method", "")[:12]

        print(
            f"  [{self.done:3d}/{self.n_tasks}] {status} "
            f"{r['task_id'][:16]:16s} "
            f"score={r['score']:.3f}{test_tag:7s} "
            f"{r['time_seconds']:5.2f}s  "
            f"{method:14s} "
            f"tk={r['toolkit_size']:3d}  "
            f"ETA {eta_str}"
        )

    def _print_rolling_summary(self) -> None:
        """Print a compact running-total statistics block."""
        elapsed    = time.time() - self.start_time
        mean_score = statistics.mean(self.scores)
        med_score  = statistics.median(self.scores)
        mean_time  = statistics.mean(self.times)
        solve_pct  = 100 * self.solved  / self.done
        partial_pct= 100 * self.partial / self.done
        above80    = self.solved + self.partial

        # Rolling score distribution buckets
        buckets = {"≥0.99": 0, "0.80-0.99": 0, "0.50-0.80": 0, "<0.50": 0}
        for s in self.scores:
            if   s >= 0.99: buckets["≥0.99"]    += 1
            elif s >= 0.80: buckets["0.80-0.99"] += 1
            elif s >= 0.50: buckets["0.50-0.80"] += 1
            else:           buckets["<0.50"]      += 1

        print(f"\n  ┌─ [{self.done}/{self.n_tasks} done  {_fmt_duration(elapsed)} elapsed] ─────────────────")
        print(f"  │  Solved (exact):     {self.solved:3d}  ({solve_pct:.1f}%)")
        print(f"  │  Partial (>80%):     {self.partial:3d}  ({partial_pct:.1f}%)   above 80%: {above80} ({100*above80/self.done:.0f}%)")
        print(f"  │  Test confirmed:     {self.test_ok:3d}")
        print(f"  │  Mean / median score:  {mean_score:.3f} / {med_score:.3f}")
        print(f"  │  Score dist:  "
              f"✓{buckets['≥0.99']}  "
              f"~{buckets['0.80-0.99']}  "
              f"△{buckets['0.50-0.80']}  "
              f"✗{buckets['<0.50']}")
        print(f"  │  Avg task time:  {mean_time:.2f}s  "
              f"(min {min(self.times):.2f}s  max {max(self.times):.2f}s)")
        print(f"  │  Wall-clock rate:  {self.done/max(elapsed,0.001):.2f} tasks/s  "
              f"({self.n_workers} workers)")
        print(f"  └──────────────────────────────────────────────────────────\n")


def _fmt_duration(seconds: float) -> str:
    """Format seconds as '1h23m', '4m32s', or '45s'."""
    s = int(seconds)
    if s >= 3600:
        return f"{s // 3600}h{(s % 3600) // 60:02d}m"
    if s >= 60:
        return f"{s // 60}m{s % 60:02d}s"
    return f"{s}s"


# ── Main evaluation harness ────────────────────────────────────────────────

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

    Results stream to stdout as tasks finish, so you can monitor progress
    live and Ctrl-C to abort cleanly at any point.

    Args:
        tasks:              task_id → task dict (from load_dataset)
        population_size:    evolutionary population size per worker
        max_generations:    max generations per task
        max_program_length: max program chain length (reserved for future use)
        verbose:            print per-task and summary output
        output_path:        if set, save results JSON to this path
        seed:               global random seed (fully reproducible for fixed
                            (seed, workers) pair)
        workers:            0 → auto (performance cores), 1 → single-process,
                            N → exactly N processes

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
        init_tk = _tmp.toolkit.size
        del _tmp
        print(f"CPU: {describe_cpu()}")
        print(f"Workers: {n_workers}  |  Tasks: {n_tasks}  |  "
              f"Seed: {seed}  |  "
              f"Initial toolkit: {init_tk} concepts")
        print()

    # Each task gets a deterministic seed derived from its position in the
    # sorted list, so (seed, workers) always gives identical results.
    worker_args = [
        (task_id, tasks[task_id], population_size, max_generations,
         seed + i * 1000)
        for i, task_id in enumerate(sorted_ids)
    ]

    start_time = time.time()
    tracker    = _ProgressTracker(n_tasks, n_workers, start_time)
    task_results: dict[str, dict] = {}

    try:
        if n_workers == 1:
            # In-process path — same results, easier to debug/profile
            for args in worker_args:
                r = _solve_one(args)
                task_results[r["task_id"]] = r
                if verbose:
                    tracker.update(r)
        else:
            # Parallel path — imap_unordered streams results as they arrive
            # so the progress display is live even with many workers.
            with multiprocessing.Pool(processes=n_workers) as pool:
                for r in pool.imap_unordered(_solve_one, worker_args):
                    task_results[r["task_id"]] = r
                    if verbose:
                        tracker.update(r)

    except KeyboardInterrupt:
        print("\n\n  ⚠  Aborted by user — partial results below.\n")
        # Fall through to print whatever we have so far

    total_time = time.time() - start_time

    # ── Final summary ────────────────────────────────────────────────────
    solved_count  = sum(1 for r in task_results.values() if r["solved"])
    partial_count = sum(1 for r in task_results.values()
                        if not r["solved"] and r["score"] > 0.8)
    test_correct  = sum(1 for r in task_results.values() if r.get("test_passed"))
    completed     = len(task_results)

    if verbose and completed > 0:
        scores = [r["score"] for r in task_results.values()]
        times  = [r["time_seconds"] for r in task_results.values()]
        above80 = solved_count + partial_count

        print(f"\n{'='*60}")
        print("BENCHMARK RESULTS")
        print(f"{'='*60}")
        print(f"Tasks completed:    {completed}/{n_tasks}")
        print(f"Solved (exact):     {solved_count}/{completed} "
              f"({100*solved_count/max(completed,1):.1f}%)")
        print(f"Partial (>80%):     {partial_count}/{completed} "
              f"({100*partial_count/max(completed,1):.1f}%)")
        print(f"Above 80% total:    {above80}/{completed} "
              f"({100*above80/max(completed,1):.1f}%)")
        print(f"Test confirmed:     {test_correct}/{completed} "
              f"({100*test_correct/max(completed,1):.1f}%)")
        print(f"Mean score:         {statistics.mean(scores):.3f}")
        print(f"Median score:       {statistics.median(scores):.3f}")
        print(f"Score std-dev:      {statistics.stdev(scores) if len(scores)>1 else 0:.3f}")
        print(f"Total time:         {_fmt_duration(total_time)} "
              f"({total_time/max(completed,1):.2f}s/task avg)")
        print(f"Wall-clock rate:    "
              f"{completed/max(total_time,0.001):.2f} tasks/s  "
              f"({n_workers} workers)")
        print(f"{'='*60}")

    summary = {
        "total_tasks":        n_tasks,
        "completed_tasks":    completed,
        "solved_exact":       solved_count,
        "solve_rate":         solved_count / max(completed, 1),
        "partial_solved":     partial_count,
        "above_80pct":        solved_count + partial_count,
        "test_correct":       test_correct,
        "test_rate":          test_correct / max(completed, 1),
        "mean_score":         statistics.mean(r["score"] for r in task_results.values())
                              if task_results else 0.0,
        "total_time_seconds": total_time,
        "avg_time_per_task":  total_time / max(completed, 1),
        "workers_used":       n_workers,
        "seed":               seed,
    }

    result = {"task_results": task_results, "summary": summary}

    if output_path and task_results:
        # Sort task_results by task_id for deterministic JSON output
        result["task_results"] = dict(sorted(task_results.items()))
        with open(output_path, "w") as f:
            json.dump(result, f, indent=2)
        if verbose:
            print(f"\nResults saved to: {output_path}")

    return result
