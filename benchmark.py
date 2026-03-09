#!/usr/bin/env python3
"""
Four Pillars AGI — Performance & Accuracy Benchmark
====================================================
Run full benchmark (parallel by default, all outputs auto-saved):

    python benchmark.py --data-dir ARC-AGI/data/training

Run a quick subset:

    python benchmark.py --data-dir ARC-AGI/data/training --tasks 20

Single-process mode (easier to debug):

    python benchmark.py --data-dir ARC-AGI/data/training --workers 1

Train → Eval with culture transfer:

    # Step 1: Train (culture auto-saved to cultures/)
    python benchmark.py --data-dir ARC-AGI/data/training

    # Step 2: Eval with culture from step 1
    python benchmark.py --data-dir ARC-AGI/data/evaluation \\
        --culture-file cultures/<timestamp>_training.json

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
    """Format n/total as percentage string, e.g. '25%'."""
    if total == 0:
        return "0%"
    return f"{100 * n / total:.0f}%"


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
    """
    task_id, task, population_size, max_generations, seed, culture_path = args

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

    t0 = time.perf_counter()
    result = solver.solve_task(task, task_id)
    elapsed = time.perf_counter() - t0

    return {
        "task_id": task_id,
        "result": result,
        "elapsed": elapsed,
        "toolkit_size": solver.toolkit.size,
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

    def __init__(self, n_tasks: int, n_workers: int,
                 task_dims: dict, task_sizes: dict):
        self.n_tasks = n_tasks
        self.n_workers = n_workers
        self.task_dims = task_dims    # task_id -> (train_dims, test_dims)
        self.task_sizes = task_sizes  # task_id -> total_cells
        self.lock = threading.Lock()

        # Running stats
        self.done = 0
        self.exact = 0
        self.flukes = 0
        self.overfits = 0
        self.fails = 0
        self.dispatched = 0

        self.scores: list[float] = []
        self.times: list[float] = []
        self.by_method: dict[str, int] = {}
        self.start_time = time.time()

        # Per-task tracking for straggler detection
        self.start_times: dict[str, float] = {}  # task_id -> dispatch time
        self.completed: set[str] = set()
        # All task results for saving
        self.all_results: dict[str, dict] = {}

    def task_started(self, idx: int, task_id: str):
        """Log when a task is dispatched to the pool."""
        with self.lock:
            self.dispatched += 1
            self.start_times[task_id] = time.time()
            train_d, test_d = self.task_dims.get(task_id, ("?", "?"))
            cells = self.task_sizes.get(task_id, 0)
            print(f"  >> [{idx:3d}/{self.n_tasks}] {task_id}  "
                  f"cells={cells:5d}  train=[{train_d}]  test=[{test_d}]")

    def task_done(self, idx: int, worker_result: dict):
        """Log when a task completes with running stats."""
        with self.lock:
            self.done += 1

            r = worker_result["result"]
            elapsed = worker_result["elapsed"]
            task_id = worker_result["task_id"]
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
            elif solved and not tc:
                self.overfits += 1
                status = "overfit"
            elif fl:
                self.flukes += 1
                status = "fluke"
            else:
                self.fails += 1
                status = "fail"

            if solved:
                method = r.get("method", "unknown")
                self.by_method[method] = self.by_method.get(method, 0) + 1

            icon = _STATUS_ICON[status]
            score = r["score"]
            method_str = r.get("method", "") if solved else ""
            active = self.dispatched - self.done
            pending = self.n_tasks - self.dispatched

            # Is this task slow? Compare against running median
            slow_tag = ""
            if len(self.times) >= 5:
                med = statistics.median(self.times)
                if elapsed > med * 3:
                    slow_tag = "  *** SLOW ***"

            print(
                f"  {icon} [{self.done:3d}/{self.n_tasks}] {task_id}  "
                f"{_fmt_duration(elapsed):>7s}  "
                f"score={score:.3f}  {status:<7s} {method_str}{slow_tag}")
            print(
                f"       totals: "
                f"{_pct(self.done, self.n_tasks):>3s} done  "
                f"exact={self.exact}({_pct(self.exact, self.done)})  "
                f"overfit={self.overfits}  fluke={self.flukes}  "
                f"fail={self.fails}  "
                f"active={active}  pending={pending}")

            # Rolling summary every 25 tasks (or at the end)
            if self.done % 25 == 0 or self.done == self.n_tasks:
                self._print_summary()

    def _print_summary(self):
        """Print a rolling summary with straggler info (lock already held)."""
        elapsed = time.time() - self.start_time
        mean_score = statistics.mean(self.scores) if self.scores else 0
        med_time = statistics.median(self.times) if self.times else 0
        rate = self.done / max(elapsed, 0.001)
        remaining = self.n_tasks - self.done
        eta = remaining / rate if rate > 0 else 0

        print()
        print(f"  ┌── Progress: {self.done}/{self.n_tasks} "
              f"({_pct(self.done, self.n_tasks)})  "
              f"{_fmt_duration(elapsed)} elapsed  "
              f"ETA {_fmt_duration(eta)} ──")
        print(f"  │  Rate: {rate:.2f} tasks/s  "
              f"({self.n_workers} workers)")
        print(f"  │  ✓ exact={self.exact}  "
              f"◇ overfit={self.overfits}  "
              f"△ fluke={self.flukes}  "
              f"✗ fail={self.fails}")
        print(f"  │  Score: mean={mean_score:.3f}  "
              f"Time: median={med_time:.1f}s  "
              f"mean={statistics.mean(self.times):.1f}s")
        if self.by_method:
            methods = "  ".join(f"{m}:{c}" for m, c in
                                sorted(self.by_method.items(),
                                       key=lambda x: -x[1]))
            print(f"  │  Methods: {methods}")

        # Straggler detection: show in-flight tasks that are taking >3x median
        if med_time > 0 and self.done < self.n_tasks:
            now = time.time()
            stragglers = []
            for tid, t0 in self.start_times.items():
                if tid not in self.completed:
                    in_flight = now - t0
                    if in_flight > med_time * 3:
                        cells = self.task_sizes.get(tid, 0)
                        stragglers.append((tid, in_flight, cells))
            if stragglers:
                stragglers.sort(key=lambda x: -x[1])
                print(f"  │")
                print(f"  │  *** Stragglers (>{med_time * 3:.0f}s = "
                      f"3x median): ***")
                for tid, dur, cells in stragglers[:5]:
                    print(f"  │    {tid}  "
                          f"{_fmt_duration(dur)} in-flight  "
                          f"cells={cells}")

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
) -> dict | None:
    """Run the solver on tasks with parallel execution.

    Returns dict with 'times', 'scores', 'results_path', 'culture_path',
    or None if no tasks found.
    """
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
    print(f"  Culture output:   {save_culture or '(auto)'}")
    print(f"  Results output:   {results_path or '(auto)'}")

    # Grid size statistics
    sizes = list(task_sizes.values())
    print(f"  Grid sizes:       "
          f"min={min(sizes)}  median={statistics.median(sizes):.0f}  "
          f"max={max(sizes)}  total={sum(sizes):,} cells")

    # Load culture
    if culture_file and os.path.exists(culture_file):
        print(f"  Culture loaded:   {culture_file}")
    elif culture_file:
        print(f"  WARNING: culture file not found: {culture_file}")
        culture_file = None

    # Build worker args
    sorted_ids = sorted(tasks.keys())
    culture_path = culture_file or ""
    worker_args = [
        (task_id, tasks[task_id], population_size, max_generations,
         seed + i * 1000, culture_path)
        for i, task_id in enumerate(sorted_ids)
    ]

    _section(f"Running {n_total} tasks on {n_workers} workers")

    tracker = _BenchmarkTracker(n_total, n_workers, task_dims, task_sizes)
    task_idx = {tid: i + 1 for i, tid in enumerate(sorted_ids)}

    try:
        if n_workers == 1:
            for args in worker_args:
                tid = args[0]
                idx = task_idx[tid]
                tracker.task_started(idx, tid)
                r = _solve_one(args)
                tracker.task_done(idx, r)
        else:
            # Print all Started lines, then stream Done results
            for args in worker_args:
                tid = args[0]
                idx = task_idx[tid]
                tracker.task_started(idx, tid)

            print()

            with multiprocessing.Pool(processes=n_workers) as pool:
                for r in pool.imap_unordered(_solve_one, worker_args):
                    tid = r["task_id"]
                    idx = task_idx[tid]
                    tracker.task_done(idx, r)

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

    print(f"  Tasks completed:   {done}/{n_total}")
    print(f"  ✓ Test confirmed:  {tracker.exact}/{done}  "
          f"({_pct(tracker.exact, done)})  ← golden metric")
    print(f"    Train-perfect:   {train_perfect}/{done}  "
          f"({_pct(train_perfect, done)})")
    if tracker.overfits > 0:
        print(f"    ◇ Overfit:       {tracker.overfits}/{done}  "
              f"({_pct(tracker.overfits, done)})")
    if tracker.flukes > 0:
        print(f"    △ Fluke:         {tracker.flukes}/{done}  "
              f"({_pct(tracker.flukes, done)})")
    print(f"    ✗ Fail:          {tracker.fails}/{done}  "
          f"({_pct(tracker.fails, done)})")
    print(f"  Mean score:        {statistics.mean(tracker.scores):.3f}")
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

    # Show top-5 slowest tasks (straggler post-mortem)
    if len(tracker.times) >= 10:
        task_time_pairs = [
            (tid, wr["elapsed"], wr["result"]["score"])
            for tid, wr in tracker.all_results.items()
        ]
        task_time_pairs.sort(key=lambda x: -x[1])
        print(f"  Slowest tasks:")
        for tid, dur, sc in task_time_pairs[:5]:
            cells = task_sizes.get(tid, 0)
            print(f"    {tid}  {_fmt_duration(dur):>7s}  "
                  f"cells={cells:5d}  score={sc:.3f}")

    print(f"  Legend: ✓=test confirmed  ◇=overfit  △=fluke  ✗=fail")

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
            "test_confirmed": tracker.exact,
            "train_perfect": train_perfect,
            "overfits": tracker.overfits,
            "flukes": tracker.flukes,
            "fails": tracker.fails,
            "mean_score": round(statistics.mean(tracker.scores), 4),
            "median_time": round(statistics.median(tracker.times), 2),
            "by_method": tracker.by_method,
        },
        "tasks": {
            tid: {
                "score": wr["result"]["score"],
                "solved": wr["result"]["solved"],
                "test_confirmed": wr["result"].get("test_confirmed", False),
                "fluke": wr["result"].get("fluke", False),
                "method": wr["result"].get("method", ""),
                "elapsed": round(wr["elapsed"], 3),
                "toolkit_size": wr["toolkit_size"],
            }
            for tid, wr in tracker.all_results.items()
        },
    }

    os.makedirs("results", exist_ok=True)
    if not results_path:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_path = f"results/{ts}_{mode}.json"
    with open(results_path, "w") as f:
        json.dump(results_data, f, indent=2)
    print(f"\n  Results saved:     {results_path}")

    # ── Save culture ──────────────────────────────────────────────────
    os.makedirs("cultures", exist_ok=True)
    if not save_culture:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_culture = f"cultures/{ts}_{mode}.json"
    from arc_agent.culture import save_culture as _save_culture
    from arc_agent.solver import FourPillarsSolver
    solver = FourPillarsSolver(population_size=population_size,
                               max_generations=max_generations,
                               verbose=False)
    _save_culture(solver.toolkit, solver.archive, save_culture)
    print(f"  Culture saved:     {save_culture}")

    return {
        "times": tracker.times,
        "scores": tracker.scores,
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
    args = parser.parse_args()

    # ── Set up tee logging ────────────────────────────────────────────
    tee = None
    if not args.no_log:
        os.makedirs("logs", exist_ok=True)
        if args.log_file:
            log_path = args.log_file
        else:
            # Detect mode for filename
            if "eval" in args.data_dir.lower():
                mode_tag = "evaluation"
            elif "test" in args.data_dir.lower() and "training" not in args.data_dir.lower():
                mode_tag = "test"
            else:
                mode_tag = "training"
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_path = f"logs/{ts}_{mode_tag}.log"
        tee = _TeeWriter(log_path, sys.stdout)
        sys.stdout = tee

    try:
        _hline("═")
        print("  Four Pillars AGI — Benchmark")
        print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        _hline("═")

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
        )
        if result is not None and args.tasks > 0:
            _extrapolate(result["times"], numba_active)

        # Print artifact locations
        _section("Artifacts")
        if not args.no_log:
            print(f"  Log:      {log_path}")
        if result:
            print(f"  Results:  {result['results_path']}")
            print(f"  Culture:  {result['culture_path']}")

        _hline("═")
        print("  Done.")
        _hline("═")
        print()

    finally:
        if tee:
            sys.stdout = tee._original
            tee.close()


if __name__ == "__main__":
    main()
