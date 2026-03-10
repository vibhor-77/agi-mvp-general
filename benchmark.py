#!/usr/bin/env python3
"""
Four Pillars AGI — Performance & Accuracy Benchmark
====================================================
Full pipeline (train + eval in one command):

    python benchmark.py --pipeline

Run training only:

    python benchmark.py --data-dir ARC-AGI/data/training

Run eval with a specific culture file:

    python benchmark.py --data-dir ARC-AGI/data/evaluation \\
        --culture-file cultures/<timestamp>_training.json

Quick subset:

    python benchmark.py --data-dir ARC-AGI/data/training --tasks 20

Single-process (easier to debug):

    python benchmark.py --data-dir ARC-AGI/data/training --workers 1

All runs automatically produce:
  - logs/<timestamp>_<mode>.log      — full console output
  - results/<timestamp>_<mode>.json  — per-task results
  - cultures/<timestamp>_<mode>.json — learned culture snapshot
"""
from __future__ import annotations

import argparse
import glob
import json
import multiprocessing
import os
import random
import statistics
import sys
import time
import threading
from datetime import datetime
from io import TextIOBase

import numpy as np


# ---------------------------------------------------------------------------
# Tee writer — duplicates stdout to both console and a log file
# ---------------------------------------------------------------------------

class _TeeWriter(TextIOBase):
    """Write to both the original stdout and a log file simultaneously."""

    def __init__(self, log_path: str, original_stdout):
        super().__init__()
        self._original = original_stdout
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        self._log = open(log_path, "w", buffering=1)  # line-buffered

    def write(self, s):
        self._original.write(s)
        self._log.write(s)
        return len(s)

    def flush(self):
        self._original.flush()
        self._log.flush()

    def close(self):
        self._log.close()

    @property
    def encoding(self):
        return self._original.encoding


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _hline(char="─", width=72):
    print(char * width)


def _section(title: str):
    print()
    _hline("─")
    print(f"  {title}")
    _hline("─")


def _fmt_duration(seconds: float) -> str:
    """Format seconds as '1h23m', '4m32s', or '12.3s'."""
    s = seconds
    if s >= 3600:
        return f"{int(s) // 3600}h{(int(s) % 3600) // 60:02d}m"
    if s >= 60:
        return f"{int(s) // 60}m{int(s) % 60:02d}s"
    return f"{s:.1f}s"


def _pct(n: int, total: int) -> str:
    """Format n/total as truncated percentage string, e.g. '5.75%'.

    Uses 2 decimal places and truncates (floors) rather than rounding,
    to avoid misleading over-reporting. E.g. 23/400 = 5.75%, not 6%.
    """
    if total == 0:
        return "0.00%"
    import math
    value = 100 * n / total
    # Truncate to 2 decimal places (floor, not round)
    truncated = math.floor(value * 100) / 100
    return f"{truncated:.2f}%"


def _task_dimensions(task: dict) -> tuple[str, str]:
    """Extract train and test dimensions as formatted strings.

    Returns (train_dims, test_dims) like "3x3->9x9, 5x5->5x5".
    """
    train_parts = []
    for ex in task.get("train", []):
        inp = ex.get("input", [[]])
        out = ex.get("output", [[]])
        ih, iw = len(inp), len(inp[0]) if inp else 0
        oh, ow = len(out), len(out[0]) if out else 0
        train_parts.append(f"{ih}x{iw}->{oh}x{ow}")

    test_parts = []
    for ex in task.get("test", []):
        inp = ex.get("input", [[]])
        ih, iw = len(inp), len(inp[0]) if inp else 0
        out = ex.get("output")
        if out:
            oh, ow = len(out), len(out[0]) if out else 0
            test_parts.append(f"{ih}x{iw}->{oh}x{ow}")
        else:
            test_parts.append(f"{ih}x{iw}->?")

    return ", ".join(train_parts), ", ".join(test_parts)


def _task_grid_size(task: dict) -> int:
    """Total number of cells across all train+test inputs+outputs.

    Used to estimate computational cost — larger grids take longer.
    """
    total = 0
    for split in ("train", "test"):
        for ex in task.get(split, []):
            for key in ("input", "output"):
                g = ex.get(key, [[]])
                total += len(g) * (len(g[0]) if g else 0)
    return total


def _aggregate_culture(all_results: dict, save_path: str) -> None:
    """Aggregate learned culture from all worker results and save to JSON.

    Each worker independently discovers concepts and programs. This merges
    them into a single culture file, deduplicating by name.
    """
    all_concepts: dict[str, dict] = {}
    all_programs: list[dict] = []
    all_features: dict[str, dict] = {}

    for wr in all_results.values():
        for concept in wr.get("_learned_concepts", []):
            name = concept["name"]
            if name not in all_concepts:
                all_concepts[name] = concept

        all_programs.extend(wr.get("_solved_programs", []))

        for tid, features in wr.get("_task_features", {}).items():
            if tid not in all_features:
                serializable = {}
                for k, v in features.items():
                    if isinstance(v, (bool, int, float, str)):
                        serializable[k] = v
                all_features[tid] = serializable

    # Deduplicate programs by (task_id, name) pair
    seen = set()
    unique_programs = []
    for prog in all_programs:
        key = (prog.get("task_id", ""), prog.get("name", ""))
        if key not in seen:
            seen.add(key)
            unique_programs.append(prog)

    culture = {
        "version": "0.9",
        "learned_concepts": list(all_concepts.values()),
        "successful_programs": unique_programs,
        "task_features": all_features,
    }

    with open(save_path, "w") as f:
        json.dump(culture, f, indent=2)

    print(f"    Learned concepts: {len(all_concepts)}")
    print(f"    Successful programs: {len(unique_programs)}")
    print(f"    Task features: {len(all_features)}")


# ---------------------------------------------------------------------------
# 1. Environment report
# ---------------------------------------------------------------------------

def _report_environment() -> bool:
    """Print environment info. Returns True if Numba is available."""
    _section("Environment")
    import platform
    print(f"  Python:   {sys.version.split()[0]}")
    print(f"  Platform: {platform.platform()}")
    print(f"  Machine:  {platform.machine()}")
    print(f"  NumPy:    {np.__version__}")

    try:
        import numba
        print(f"  Numba:    {numba.__version__}  (JIT active)")
        numba_ok = True
    except ImportError:
        print(f"  Numba:    NOT installed  (pip install numba for ~20x speedup)")
        numba_ok = False

    from arc_agent.cpu_utils import default_workers, describe_cpu
    print(f"  CPU:      {describe_cpu()}")
    print(f"  Workers:  {default_workers()} (auto-detected performance cores)")

    return numba_ok


# ---------------------------------------------------------------------------
# 2. Micro-benchmarks
# ---------------------------------------------------------------------------

def _benchmark_operations() -> tuple[float, bool]:
    """Run micro-benchmarks on hot-path operations. Returns (fo_us, numba_active)."""
    _section("Micro-benchmarks (us per call)")

    from arc_agent.objects import find_objects, _USE_NUMBA
    from arc_agent.scorer import _structural_similarity_np

    grid = [[0] * 30 for _ in range(30)]
    for r in range(3, 12):
        for c in range(2, 10):
            grid[r][c] = 1
    for r in range(15, 20):
        for c in range(15, 22):
            grid[r][c] = 3
    for r in range(22, 28):
        for c in range(5, 9):
            grid[r][c] = 7

    for _ in range(10):
        find_objects(grid)
    N = 500
    t0 = time.perf_counter()
    for _ in range(N):
        find_objects(grid)
    dt_fo = (time.perf_counter() - t0) / N * 1e6
    impl = "Numba JIT" if _USE_NUMBA else "pure Python"
    print(f"  find_objects (30x30, {impl}):  {dt_fo:.0f} us")

    sample = [[r * c % 10 for c in range(30)] for r in range(30)]
    N = 2000
    t0 = time.perf_counter()
    for _ in range(N):
        np.array(sample, dtype=np.uint8)
    dt_arr = (time.perf_counter() - t0) / N * 1e6
    print(f"  np.array 30x30 conversion:    {dt_arr:.1f} us")

    p = np.array(sample, dtype=np.uint8)
    h, w = p.shape
    N = 5000
    t0 = time.perf_counter()
    for _ in range(N):
        _structural_similarity_np(p, p, h, w, h, w)
    dt_score = (time.perf_counter() - t0) / N * 1e6
    print(f"  structural_similarity 30x30:  {dt_score:.1f} us")

    return dt_fo, _USE_NUMBA


# ---------------------------------------------------------------------------
# 3. Worker function for multiprocessing
# ---------------------------------------------------------------------------

def _solve_one(args):
    """Solve a single task in a worker process.

    Creates a fresh solver per call — no shared mutable state between workers.
    Prints a Started line at actual execution time (not queue time) so that
    in parallel mode the user sees when work truly begins on each task.
    """
    task_id, task, population_size, max_generations, seed, culture_path, \
        idx, n_tasks, train_dims, test_dims, cells, evals_budget = args

    # Print Started at actual execution time (from the worker process)
    print(f"  >> [{idx:3d}/{n_tasks}] {task_id}  "
          f"cells={cells:5d}  train=[{train_dims}]  test=[{test_dims}]",
          flush=True)

    random.seed(seed)
    np.random.seed(seed)

    from arc_agent.solver import FourPillarsSolver

    solver = FourPillarsSolver(
        population_size=population_size,
        max_generations=max_generations,
        verbose=False,
    )

    if culture_path:
        try:
            from arc_agent.culture import load_culture
            load_culture(solver.toolkit, culture_path, solver.archive)
        except Exception:
            pass

    cpu0 = time.process_time()
    t0 = time.perf_counter()
    result = solver.solve_task(task, task_id, evals_budget=evals_budget)
    elapsed = time.perf_counter() - t0
    cpu_time = time.process_time() - cpu0

    # Collect culture data for aggregation across workers
    learned_concepts = []
    for name, concept in solver.toolkit.concepts.items():
        if name.startswith("learned_"):
            from arc_agent.culture import _extract_step_names
            step_names = _extract_step_names(concept)
            learned_concepts.append({
                "name": name,
                "steps": step_names,
                "kind": concept.kind,
                "usage_count": concept.usage_count,
                "success_count": concept.success_count,
            })

    solved_programs = []
    for tid, programs in solver.archive.task_solutions.items():
        for prog in programs:
            step_names = [s.name for s in prog.steps]
            all_ok = all(sn in solver.toolkit.concepts for sn in step_names)
            if all_ok and step_names:
                solved_programs.append({
                    "task_id": tid,
                    "steps": step_names,
                    "fitness": prog.fitness,
                    "name": prog.name,
                })

    return {
        "task_id": task_id,
        "result": result,
        "elapsed": elapsed,
        "cpu_time": cpu_time,
        "toolkit_size": solver.toolkit.size,
        "cells": cells,
        "train_dims": train_dims,
        "test_dims": test_dims,
        # Culture data for cross-worker aggregation
        "_learned_concepts": learned_concepts,
        "_solved_programs": solved_programs,
        "_task_features": dict(solver.archive.task_features),
    }


# ---------------------------------------------------------------------------
# 4. Progress tracker with straggler detection
# ---------------------------------------------------------------------------

# Status symbols for compact display
_STATUS_ICON = {
    "exact":   "✓",
    "overfit": "◇",
    "fluke":   "△",
    "fail":    "✗",
}


class _BenchmarkTracker:
    """Thread-safe progress tracker with straggler detection.

    Prints a Started line when dispatched, a Done line with running stats
    when complete, and flags potential stragglers in rolling summaries.
    """

    def __init__(self, n_tasks: int, n_workers: int, task_sizes: dict,
                 results_live: str = "", culture_live: str = ""):
        self.n_tasks = n_tasks
        self.n_workers = n_workers
        self.task_sizes = task_sizes  # task_id -> total_cells
        self.results_live = results_live  # JSONL path for live results
        self.culture_live = culture_live  # JSONL path for live culture
        self.lock = threading.Lock()
        # Initialize live files (truncate any previous content)
        if results_live:
            open(results_live, "w").close()
        if culture_live:
            open(culture_live, "w").close()

        # Running stats
        self.done = 0
        self.exact = 0
        self.flukes = 0
        self.overfits = 0
        self.fails = 0

        self.total_evals = 0  # total program evaluations across all tasks
        self.total_cpu_time = 0.0  # total CPU time across all tasks
        self.budget_exceeded_count = 0  # tasks that hit evals budget
        self.scores: list[float] = []
        self.times: list[float] = []
        self.by_method: dict[str, int] = {}
        self.start_time = time.time()

        # Fluke train accuracy tracking
        self.fluke_train_hit = 0   # train examples that fluke programs got right
        self.fluke_train_total = 0 # total train examples across fluke tasks

        # Near-miss: tasks where the primary candidate failed test but
        # at least one non-primary candidate would have passed.
        self.near_misses = 0
        self.near_miss_tasks: list[str] = []

        # Per-task tracking for straggler detection
        self.completed: set[str] = set()
        # All task results for saving
        self.all_results: dict[str, dict] = {}

    def task_done(self, worker_result: dict):
        """Log when a task completes with running stats."""
        with self.lock:
            self.done += 1

            r = worker_result["result"]
            elapsed = worker_result["elapsed"]
            cpu_time = worker_result.get("cpu_time", 0.0)
            task_id = worker_result["task_id"]
            cells = worker_result.get("cells", 0)
            train_dims = worker_result.get("train_dims", "?")
            test_dims = worker_result.get("test_dims", "?")
            n_evals = r.get("n_evals", 0)
            self.total_evals += n_evals
            self.total_cpu_time += cpu_time
            if r.get("budget_exceeded", False):
                self.budget_exceeded_count += 1

            self.completed.add(task_id)
            self.all_results[task_id] = worker_result

            self.scores.append(r["score"])
            self.times.append(elapsed)

            # Classify result
            tc = r.get("test_confirmed", False)
            fl = r.get("fluke", False)
            solved = r["solved"]

            if tc:
                self.exact += 1
                status = "exact"
            elif fl:
                self.flukes += 1
                status = "fluke"
                # Track per-example train accuracy for flukes
                tex = r.get("train_example_exact", [])
                n_train = r.get("n_train", 0)
                self.fluke_train_total += n_train
                self.fluke_train_hit += sum(tex)
            elif solved and not tc:
                self.overfits += 1
                status = "overfit"
            else:
                self.fails += 1
                status = "fail"

            if solved:
                method = r.get("method", "unknown")
                self.by_method[method] = self.by_method.get(method, 0) + 1

            # Near-miss detection: primary candidate failed test,
            # but at least one non-primary candidate would have passed.
            if not tc and not fl:
                cands = r.get("candidates", [])
                any_alt_passed = any(
                    c.get("test_exact", False)
                    for c in cands
                )
                if any_alt_passed:
                    self.near_misses += 1
                    self.near_miss_tasks.append(task_id)

            icon = _STATUS_ICON[status]
            score = r["score"]
            method_str = r.get("method", "") if solved else ""
            pending = self.n_tasks - self.done

            # Is this task slow? Compare against running median
            slow_tag = ""
            if len(self.times) >= 5:
                med = statistics.median(self.times)
                if elapsed > med * 3:
                    slow_tag = "  *** SLOW ***"

            # Fluke detail: show train hit/miss
            fluke_detail = ""
            if status == "fluke":
                tex = r.get("train_example_exact", [])
                n_train = r.get("n_train", 0)
                n_hit = sum(tex)
                fluke_detail = f"  (train {n_hit}/{n_train} exact)"

            print(
                f"  {icon} [{self.done:3d}/{self.n_tasks}] {task_id}  "
                f"{_fmt_duration(elapsed):>7s}  "
                f"score={score:.3f}  {status:<7s} "
                f"{method_str}{fluke_detail}{slow_tag}")
            # Program tree: show the steps used
            program_steps = r.get("program_steps", [])
            if program_steps:
                steps_str = " → ".join(program_steps)
                print(f"       program: {steps_str}")
            # Show all candidates and their test results
            cands = r.get("candidates", [])
            if len(cands) > 1 or (len(cands) == 1 and cands[0].get("test_exact") is not None):
                for ci, c in enumerate(cands):
                    c_icon = "✓" if c.get("test_exact") else "✗"
                    c_steps = " → ".join(c.get("steps", []))
                    print(f"       candidate[{ci}] {c_icon} test={c.get('test_score', 0):.3f}  "
                          f"{c.get('method', '')}  {c_steps}")
            print(
                f"       cells={cells}  "
                f"evals={n_evals:,}  "
                f"train=[{train_dims}]  test=[{test_dims}]")
            # Reorder: exact+fluke first (accuracy), then overfit+fail
            print(
                f"       done={self.done}/{self.n_tasks}  "
                f"exact={self.exact}/{self.done}  "
                f"fluke={self.flukes}/{self.done}  "
                f"overfit={self.overfits}/{self.done}  "
                f"fail={self.fails}/{self.done}  "
                f"pending={pending}")

            # Live-append result to JSONL file (one JSON object per line)
            if self.results_live:
                try:
                    live_record = {
                        "task_id": task_id,
                        "score": round(score, 4),
                        "status": status,
                        "method": method_str or None,
                        "elapsed": round(elapsed, 2),
                        "cpu_time": round(cpu_time, 2),
                        "cells": cells,
                        "n_evals": n_evals,
                        "budget_exceeded": r.get("budget_exceeded", False),
                        "program": program_steps,
                        "candidates": len(r.get("candidates", [])),
                    }
                    with open(self.results_live, "a") as f:
                        f.write(json.dumps(live_record) + "\n")
                except OSError:
                    pass  # don't crash if live file can't be written

            # Live-append culture entries (concepts + programs)
            if self.culture_live:
                try:
                    entries = []
                    for concept in worker_result.get("_learned_concepts", []):
                        entries.append({"type": "concept", "data": concept})
                    for prog in worker_result.get("_solved_programs", []):
                        entries.append({"type": "program", "data": prog})
                    if entries:
                        with open(self.culture_live, "a") as f:
                            for entry in entries:
                                f.write(json.dumps(entry) + "\n")
                except OSError:
                    pass

            # Rolling summary every 25 tasks (or at the end)
            if self.done % 25 == 0 or self.done == self.n_tasks:
                self._print_summary()

    def _print_summary(self):
        """Print a rolling summary (lock already held)."""
        elapsed = time.time() - self.start_time
        mean_score = statistics.mean(self.scores) if self.scores else 0
        med_time = statistics.median(self.times) if self.times else 0
        rate = self.done / max(elapsed, 0.001)
        remaining = self.n_tasks - self.done
        eta = remaining / rate if rate > 0 else 0

        print()
        print(f"  ┌── Progress: {self.done}/{self.n_tasks}  "
              f"{_fmt_duration(elapsed)} elapsed  "
              f"ETA {_fmt_duration(eta)}  "
              f"{rate:.1f} tasks/s ──")
        # Reorder: exact+fluke (accuracy) first, then overfit+fail
        fluke_train_str = ""
        if self.flukes > 0 and self.fluke_train_total > 0:
            fluke_train_str = (f"  (fluke train: "
                               f"{self.fluke_train_hit}/{self.fluke_train_total})")
        print(f"  │  ✓ exact={self.exact}/{self.done}  "
              f"△ fluke={self.flukes}/{self.done}  "
              f"◇ overfit={self.overfits}/{self.done}  "
              f"✗ fail={self.fails}/{self.done}"
              f"{fluke_train_str}")
        print(f"  │  Score: mean={mean_score:.3f}  "
              f"Time: median={med_time:.1f}s  mean={statistics.mean(self.times):.1f}s  "
              f"Evals: {self.total_evals:,}")
        if self.by_method:
            methods = "  ".join(f"{m}:{c}" for m, c in
                                sorted(self.by_method.items(),
                                       key=lambda x: -x[1]))
            print(f"  │  Methods: {methods}")
        if self.near_misses > 0:
            print(f"  │  Near-misses: {self.near_misses} "
                  f"(non-primary candidate would have passed test)")

        # Straggler post-mortem: show the slowest completed tasks so far
        if len(self.times) >= 10:
            task_time_pairs = [
                (tid, wr["elapsed"], self.task_sizes.get(tid, 0))
                for tid, wr in self.all_results.items()
            ]
            task_time_pairs.sort(key=lambda x: -x[1])
            top3 = task_time_pairs[:3]
            slowest_str = "  ".join(
                f"{t[0][:8]}({_fmt_duration(t[1])},c={t[2]})"
                for t in top3)
            print(f"  │  Slowest: {slowest_str}")

        print(f"  └{'─' * 60}")
        print()


# ---------------------------------------------------------------------------
# 5. Solver benchmark (parallel with auto-save)
# ---------------------------------------------------------------------------

def benchmark_solver(
    data_dir: str,
    n_tasks: int = 0,
    seed: int = 42,
    population_size: int = 60,
    max_generations: int = 30,
    culture_file: str | None = None,
    save_culture: str | None = None,
    workers: int = 0,
    results_path: str | None = None,
    run_timestamp: str = "",
    compute_cap: int = 400_000_000,
) -> dict | None:
    """Run the solver on tasks with parallel execution.

    Returns dict with 'times', 'scores', 'results_path', 'culture_path',
    or None if no tasks found.

    Args:
        run_timestamp: Shared timestamp for all artifacts from this run.
                       If empty, generates one at call time.
    """
    if not run_timestamp:
        run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    from arc_agent.cpu_utils import default_workers

    task_files = sorted(glob.glob(os.path.join(data_dir, "*.json")))
    if n_tasks > 0:
        task_files = task_files[:n_tasks]
    n_total = len(task_files)

    n_workers = workers if workers > 0 else default_workers()
    n_workers = max(1, min(n_workers, n_total))

    if not task_files:
        print(f"\n  ERROR: no .json files found in {data_dir}")
        return None

    # Load all tasks
    tasks = {}
    task_dims = {}
    task_sizes = {}
    for tf in task_files:
        task_id = os.path.basename(tf).replace(".json", "")
        with open(tf) as f:
            task = json.load(f)
        tasks[task_id] = task
        task_dims[task_id] = _task_dimensions(task)
        task_sizes[task_id] = _task_grid_size(task)

    # Detect mode from data dir path
    if "eval" in data_dir.lower():
        mode = "evaluation"
    elif "test" in data_dir.lower() and "training" not in data_dir.lower():
        mode = "test"
    else:
        mode = "training"

    # ── Resolve artifact paths early (so we can print and live-write) ──
    os.makedirs("results", exist_ok=True)
    os.makedirs("cultures", exist_ok=True)
    if not results_path:
        results_path = f"results/{run_timestamp}_{mode}.json"
    if not save_culture:
        save_culture = f"cultures/{run_timestamp}_{mode}.json"
    results_live = results_path.replace(".json", ".jsonl")
    culture_live = save_culture.replace(".json", ".jsonl")

    # ── Print all run parameters ──────────────────────────────────────
    _section(f"Benchmark Configuration")
    print(f"  Mode:             {mode}")
    print(f"  Data dir:         {data_dir}")
    print(f"  Tasks:            {n_total}"
          f"{'  (subset)' if n_tasks > 0 else '  (all)'}")
    print(f"  Workers:          {n_workers}"
          f"{'  (auto)' if workers == 0 else '  (manual)'}")
    print(f"  Seed:             {seed}")
    print(f"  Population size:  {population_size}")
    print(f"  Max generations:  {max_generations}")
    print(f"  Culture input:    {culture_file or '(none)'}")
    if compute_cap > 0:
        print(f"  Compute cap:      {compute_cap:,} (evals×cells)")
        max_evals = compute_cap // 800
        print(f"  Evals budget:     min(compute_cap / cells, {max_evals:,}) per task")
    else:
        print(f"  Evals budget:     unlimited (compute cap disabled)")

    # Grid size statistics
    sizes = list(task_sizes.values())
    print(f"  Grid sizes:       "
          f"min={min(sizes)}  median={statistics.median(sizes):.0f}  "
          f"max={max(sizes)}  total={sum(sizes):,} cells")

    # Print artifact paths upfront so users can tail -f immediately
    _section("Output Files (available now for tail -f)")
    print(f"  Results (live):   {results_live}")
    print(f"  Results (final):  {results_path}")
    print(f"  Culture (live):   {culture_live}")
    print(f"  Culture (final):  {save_culture}")

    # Load culture
    if culture_file and os.path.exists(culture_file):
        print(f"  Culture loaded:   {culture_file}")
    elif culture_file:
        print(f"  WARNING: culture file not found: {culture_file}")
        culture_file = None

    # Build worker args (includes idx, dims, cells for Started line in worker)
    sorted_ids = sorted(tasks.keys())
    culture_path = culture_file or ""
    worker_args = []
    for i, task_id in enumerate(sorted_ids):
        cells = task_sizes[task_id]
        # Cell-normalized budget: large grids get fewer evals because each
        # eval is proportionally more expensive. This is deterministic and
        # machine-independent (unlike wall-clock timeouts).
        DEFAULT_CELLS = 800  # Median grid size in ARC training
        if compute_cap > 0 and cells > 0:
            max_evals = compute_cap // DEFAULT_CELLS
            effective_budget = min(max(compute_cap // cells, 500),
                                  max(max_evals, 500))
        else:
            effective_budget = 10_000_000  # Effectively unlimited
        worker_args.append((
            task_id, tasks[task_id], population_size, max_generations,
            seed + i * 1000, culture_path,
            i + 1, n_total,  # idx, n_tasks
            task_dims[task_id][0], task_dims[task_id][1],  # train_dims, test_dims
            cells,
            effective_budget,  # cell-normalized computational budget
        ))

    _section(f"Running {n_total} tasks on {n_workers} workers")

    tracker = _BenchmarkTracker(n_total, n_workers, task_sizes,
                                results_live=results_live,
                                culture_live=culture_live)

    try:
        if n_workers == 1:
            for args in worker_args:
                r = _solve_one(args)  # prints Started inside worker
                tracker.task_done(r)
        else:
            with multiprocessing.Pool(processes=n_workers) as pool:
                for r in pool.imap_unordered(_solve_one, worker_args):
                    tracker.task_done(r)

    except KeyboardInterrupt:
        print("\n\n  Aborted by user — partial results below.\n")

    # ── Final summary ─────────────────────────────────────────────────
    print()
    _hline("═")
    print("  FINAL RESULTS")
    _hline("═")

    if not tracker.scores:
        print("  No tasks completed.")
        return None

    done = tracker.done
    train_perfect = tracker.exact + tracker.overfits

    print(f"  Tasks:             {done}/{n_total}")
    # Accuracy metrics first (exact + fluke = passed test)
    passed_test = tracker.exact + tracker.flukes
    print(f"  ✓ Solved:          {tracker.exact}/{done}  "
          f"({_pct(tracker.exact, done)})  ← train+test confirmed")
    if tracker.flukes > 0:
        fluke_train_str = ""
        if tracker.fluke_train_total > 0:
            fluke_train_str = (f", train accuracy: "
                               f"{tracker.fluke_train_hit}/"
                               f"{tracker.fluke_train_total}")
        print(f"    △ Fluke:         {tracker.flukes}/{done}  "
              f"({_pct(tracker.flukes, done)})  "
              f"(test-pass, train-fail{fluke_train_str})")
    if passed_test != tracker.exact:
        print(f"    Passed test:     {passed_test}/{done}  "
              f"({_pct(passed_test, done)})")
    # Then overfit + fail
    print(f"    Train-perfect:   {train_perfect}/{done}  "
          f"({_pct(train_perfect, done)})")
    if tracker.overfits > 0:
        print(f"    ◇ Overfit:       {tracker.overfits}/{done}  "
              f"({_pct(tracker.overfits, done)})  "
              f"(train-perfect, test-fail)")
    print(f"    ✗ Fail:          {tracker.fails}/{done}  "
          f"({_pct(tracker.fails, done)})")
    print(f"  Mean score:        {statistics.mean(tracker.scores):.3f}")
    print(f"  Total evaluations: {tracker.total_evals:,}")
    print(f"  Total CPU time:    {_fmt_duration(tracker.total_cpu_time)}")
    if tracker.budget_exceeded_count > 0:
        cap_str = f"compute_cap={compute_cap:,}" if compute_cap > 0 else "unlimited"
        print(f"  Budget exceeded:   {tracker.budget_exceeded_count}/{done} tasks "
              f"({cap_str})")
    print(f"  Median task time:  {statistics.median(tracker.times):.2f}s")
    print(f"  Mean task time:    {statistics.mean(tracker.times):.2f}s")
    total_wall = time.time() - tracker.start_time
    print(f"  Wall-clock time:   {_fmt_duration(total_wall)}")
    print(f"  Throughput:        {done / max(total_wall, 0.001):.1f} tasks/s "
          f"({n_workers} workers)")

    if tracker.by_method:
        print(f"  By method:")
        for m, c in sorted(tracker.by_method.items(), key=lambda x: -x[1]):
            print(f"    {m}: {c}")

    # Near-miss report: tasks where a non-primary candidate would have passed
    if tracker.near_misses > 0:
        print(f"  Near-misses:       {tracker.near_misses}  "
              f"(non-primary candidate passed test)")
        for tid in tracker.near_miss_tasks[:10]:
            wr = tracker.all_results[tid]
            cands = wr["result"].get("candidates", [])
            passing = [c for c in cands if c.get("test_exact")]
            methods = ", ".join(c.get("method", "?") for c in passing)
            print(f"    {tid}  (would pass via: {methods})")

    # Show top-5 slowest tasks (straggler post-mortem)
    if len(tracker.times) >= 10:
        task_time_pairs = [
            (tid, wr["elapsed"], wr["result"]["score"],
             wr.get("cells", 0), wr["result"].get("n_evals", 0))
            for tid, wr in tracker.all_results.items()
        ]
        task_time_pairs.sort(key=lambda x: -x[1])
        print(f"  Slowest tasks:")
        for tid, dur, sc, cells, evals in task_time_pairs[:5]:
            print(f"    {tid}  {_fmt_duration(dur):>7s}  "
                  f"cells={cells:5d}  evals={evals:,}  score={sc:.3f}")

    print(f"  Legend: ✓=solved (train+test)  ◇=overfit (train only)  "
          f"△=fluke (test only)  ✗=fail")

    # ── Save results JSON ─────────────────────────────────────────────
    results_data = {
        "meta": {
            "timestamp": datetime.now().isoformat(),
            "mode": mode,
            "data_dir": data_dir,
            "n_tasks": n_total,
            "n_completed": done,
            "workers": n_workers,
            "seed": seed,
            "population_size": population_size,
            "max_generations": max_generations,
            "culture_file": culture_file,
            "wall_clock_seconds": total_wall,
        },
        "summary": {
            "solved": tracker.exact,
            "train_perfect": train_perfect,
            "overfits": tracker.overfits,
            "flukes": tracker.flukes,
            "fails": tracker.fails,
            "mean_score": round(statistics.mean(tracker.scores), 4),
            "median_time": round(statistics.median(tracker.times), 2),
            "total_evals": tracker.total_evals,
            "total_cpu_time": round(tracker.total_cpu_time, 2),
            "budget_exceeded_count": tracker.budget_exceeded_count,
            "compute_cap_formula": "compute_cap / cells",
            "compute_cap": compute_cap,
            "fluke_train_hit": tracker.fluke_train_hit,
            "fluke_train_total": tracker.fluke_train_total,
            "near_misses": tracker.near_misses,
            "near_miss_tasks": tracker.near_miss_tasks,
            "by_method": tracker.by_method,
        },
        "tasks": {
            tid: {
                "score": wr["result"]["score"],
                "solved": wr["result"]["solved"],
                "test_confirmed": wr["result"].get("test_confirmed", False),
                "test_score": wr["result"].get("test_score", 0.0),
                "fluke": wr["result"].get("fluke", False),
                "method": wr["result"].get("method", ""),
                "program_steps": wr["result"].get("program_steps", []),
                "candidates": wr["result"].get("candidates", []),
                "elapsed": round(wr["elapsed"], 3),
                "n_evals": wr["result"].get("n_evals", 0),
                "n_train": wr["result"].get("n_train", 0),
                "train_example_exact": wr["result"].get("train_example_exact", []),
                "cpu_time": round(wr.get("cpu_time", 0.0), 3),
                "budget_exceeded": wr["result"].get("budget_exceeded", False),
                "cells": wr.get("cells", 0),
                "toolkit_size": wr["toolkit_size"],
            }
            for tid, wr in tracker.all_results.items()
        },
    }

    # results_path and save_culture already resolved at top of function
    with open(results_path, "w") as f:
        json.dump(results_data, f, indent=2)
    print(f"\n  Results saved:     {results_path}")
    if os.path.exists(results_live):
        print(f"  Results (live):    {results_live}")

    # ── Save culture (aggregated from all workers) ─────────────────
    _aggregate_culture(tracker.all_results, save_culture)
    print(f"  Culture saved:     {save_culture}")
    if os.path.exists(culture_live):
        print(f"  Culture (live):    {culture_live}")

    return {
        "times": tracker.times,
        "scores": tracker.scores,
        "solved": tracker.exact,
        "overfits": tracker.overfits,
        "flukes": tracker.flukes,
        "fails": tracker.fails,
        "total": done,
        "results_path": results_path,
        "culture_path": save_culture,
    }


# ---------------------------------------------------------------------------
# 6. Extrapolation (only shown for subset runs)
# ---------------------------------------------------------------------------

def _extrapolate(task_times: list[float], numba_active: bool):
    _section("Full benchmark projection (400 tasks)")

    from arc_agent.cpu_utils import default_workers

    mean_1w = statistics.mean(task_times)
    n_workers = default_workers()

    fo_fraction = 0.25
    numba_fo_speedup = 20
    overall_speedup = 1 / (1 - fo_fraction + fo_fraction / numba_fo_speedup)

    if numba_active:
        mean_with_numba = mean_1w
        note = "(active)"
    else:
        mean_with_numba = mean_1w / overall_speedup
        note = "(estimated)"

    yn = mean_with_numba * 400 / n_workers / 60
    wn = mean_1w * 400 / n_workers / 60
    print(f"  400-task estimate ({n_workers} workers): "
          f"~{wn:.0f} min current, ~{yn:.0f} min with Numba {note}")

    if not numba_active:
        print(f"  Tip: pip install numba  (first run compiles ~5s, then cached)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Four Pillars AGI — performance & accuracy benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--data-dir",
        default="ARC-AGI/data/training",
        help="Path to ARC-AGI JSON files (default: ARC-AGI/data/training)",
    )
    parser.add_argument(
        "--tasks", type=int, default=0,
        help="Number of tasks (0 = all, default: 0)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--workers", type=int, default=0,
        help="Parallel workers (0 = auto, 1 = single-process, default: 0)",
    )
    parser.add_argument(
        "--population-size", type=int, default=60,
        help="Evolutionary population size (default: 60)",
    )
    parser.add_argument(
        "--max-generations", type=int, default=30,
        help="Max evolutionary generations (default: 30)",
    )
    parser.add_argument(
        "--culture-file", default=None,
        help="Load culture from this JSON file",
    )
    parser.add_argument(
        "--save-culture", default=None,
        help="Save culture to this path (default: auto-timestamped in cultures/)",
    )
    parser.add_argument(
        "--results", default=None,
        help="Save results to this path (default: auto-timestamped in results/)",
    )
    parser.add_argument(
        "--log-file", default=None,
        help="Log file path (default: auto-timestamped in logs/)",
    )
    parser.add_argument(
        "--no-log", action="store_true",
        help="Disable log file output (console only)",
    )
    parser.add_argument(
        "--pipeline", action="store_true",
        help=(
            "Run full train→eval pipeline in one command. "
            "Trains on ARC-AGI/data/training, then evaluates on "
            "ARC-AGI/data/evaluation with the culture learned from training. "
            "Overrides --data-dir and --culture-file."
        ),
    )
    parser.add_argument(
        "--train-dir",
        default="ARC-AGI/data/training",
        help="Training data dir for --pipeline mode (default: ARC-AGI/data/training)",
    )
    parser.add_argument(
        "--eval-dir",
        default="ARC-AGI/data/evaluation",
        help="Evaluation data dir for --pipeline mode (default: ARC-AGI/data/evaluation)",
    )
    parser.add_argument(
        "--compute-cap", type=int, default=400_000_000,
        help="Cell-normalized compute cap (default: 400,000,000). "
             "Per-task budget = min(compute_cap/cells, compute_cap/800). "
             "At 400M: ceiling ~500K. At 8M: ~10K. 0 to disable.",
    )
    parser.add_argument(
        "--contest", action="store_true",
        help="Contest mode: disable compute cap entirely to maximize solves. "
             "Uses unlimited evals per task regardless of grid size. "
             "Equivalent to --compute-cap 0.",
    )
    args = parser.parse_args()

    # Contest mode: uncapped — maximize solves, don't save time
    if args.contest:
        args.compute_cap = 0

    if args.pipeline:
        _run_pipeline(args)
    else:
        _run_single(args)


def _run_single(args):
    """Run a single benchmark (training OR evaluation)."""
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tee, log_path = _setup_logging(args, run_timestamp)

    try:
        _print_header()
        numba_active = _report_environment()
        _benchmark_operations()

        result = benchmark_solver(
            data_dir=args.data_dir,
            n_tasks=args.tasks,
            seed=args.seed,
            population_size=args.population_size,
            max_generations=args.max_generations,
            culture_file=args.culture_file,
            save_culture=args.save_culture,
            workers=args.workers,
            results_path=args.results,
            run_timestamp=run_timestamp,
                        compute_cap=args.compute_cap,
        )
        if result is not None and args.tasks > 0:
            _extrapolate(result["times"], numba_active)

        _print_artifacts(log_path, result, args.no_log)

        # Suggest next command (helpful: no more digging for culture file)
        if result and "training" in (result.get("culture_path") or ""):
            print()
            print("  Next step — run evaluation with this culture:")
            print(f"    python benchmark.py --data-dir ARC-AGI/data/evaluation \\")
            print(f"        --culture-file {result['culture_path']}")

        _hline("═")
        print("  Done.")
        _hline("═")
        print()

    finally:
        if tee:
            sys.stdout = tee._original
            tee.close()


def _run_pipeline(args):
    """Run full train→eval pipeline: train first, then eval with learned culture."""
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Pipeline uses a single log file for both phases
    tee = None
    log_path = None
    if not args.no_log:
        os.makedirs("logs", exist_ok=True)
        log_path = f"logs/{run_timestamp}_pipeline.log"
        tee = _TeeWriter(log_path, sys.stdout)
        sys.stdout = tee

    try:
        _print_header()
        print("  Mode: PIPELINE (train → eval)")
        _hline("─")
        numba_active = _report_environment()
        _benchmark_operations()

        # ── Phase 1: Training ──────────────────────────────────────────
        _section("PHASE 1: TRAINING")
        train_result = benchmark_solver(
            data_dir=args.train_dir,
            n_tasks=args.tasks,
            seed=args.seed,
            population_size=args.population_size,
            max_generations=args.max_generations,
            culture_file=args.culture_file,
            workers=args.workers,
            run_timestamp=run_timestamp,
                        compute_cap=args.compute_cap,
        )

        if train_result is None:
            print("\n  ERROR: Training failed — no tasks completed.")
            return

        culture_path = train_result["culture_path"]
        print(f"\n  Training complete. Culture: {culture_path}")

        # ── Phase 2: Evaluation ────────────────────────────────────────
        _section("PHASE 2: EVALUATION (with culture from training)")
        eval_result = benchmark_solver(
            data_dir=args.eval_dir,
            n_tasks=args.tasks,
            seed=args.seed,
            population_size=args.population_size,
            max_generations=args.max_generations,
            culture_file=culture_path,
            workers=args.workers,
            run_timestamp=run_timestamp,
                        compute_cap=args.compute_cap,
        )

        # ── Pipeline summary ──────────────────────────────────────────
        _section("PIPELINE SUMMARY")
        for label, res in [("Training", train_result), ("Evaluation", eval_result)]:
            if res:
                s = res["solved"]
                t = res["total"]
                ov = res["overfits"]
                fl = res["flukes"]
                extras = []
                if ov:
                    extras.append(f"+{ov} overfit")
                if fl:
                    extras.append(f"+{fl} fluke")
                extra_str = f"  [{', '.join(extras)}]" if extras else ""
                print(f"  {label:12s} {s}/{t} solved "
                      f"({_pct(s, t)}){extra_str}")

        _print_artifacts(log_path, train_result, args.no_log, label="Train")
        _print_artifacts(None, eval_result, True, label="Eval")

        _hline("═")
        print("  Pipeline complete.")
        _hline("═")
        print()

    finally:
        if tee:
            sys.stdout = tee._original
            tee.close()


def _setup_logging(args, run_timestamp: str):
    """Set up tee logging. Returns (tee, log_path) or (None, None)."""
    if args.no_log:
        return None, None
    os.makedirs("logs", exist_ok=True)
    if args.log_file:
        log_path = args.log_file
    else:
        if "eval" in args.data_dir.lower():
            mode_tag = "evaluation"
        elif "test" in args.data_dir.lower() and "training" not in args.data_dir.lower():
            mode_tag = "test"
        else:
            mode_tag = "training"
        log_path = f"logs/{run_timestamp}_{mode_tag}.log"
    tee = _TeeWriter(log_path, sys.stdout)
    sys.stdout = tee
    return tee, log_path


def _print_header():
    _hline("═")
    print("  Four Pillars AGI — Benchmark")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    _hline("═")


def _print_artifacts(log_path, result, no_log, label=""):
    _section(f"Artifacts{' (' + label + ')' if label else ''}")
    if log_path and not no_log:
        print(f"  Log:      {log_path}")
    if result:
        print(f"  Results:  {result['results_path']}")
        print(f"  Culture:  {result['culture_path']}")


if __name__ == "__main__":
    main()
