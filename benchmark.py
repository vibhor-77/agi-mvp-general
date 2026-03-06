"""
Four Pillars AGI — Performance & Accuracy Benchmark
====================================================
Run this on your machine to validate performance and accuracy:

    python benchmark.py --data-dir ARC-AGI/data/training

It will:
  1. Report your CPU configuration and Numba availability
  2. Micro-benchmark each hot-path operation (find_objects, scoring, np.array)
  3. Run the full solver on 20 tasks (single-process for reproducibility)
  4. Extrapolate expected time for the full 400-task benchmark
"""
from __future__ import annotations
import argparse
import glob
import json
import os
import statistics
import sys
import time

import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _hline(char="─", width=60):
    print(char * width)


def _section(title: str):
    print()
    _hline("─")
    print(f"  {title}")
    _hline("─")


# ---------------------------------------------------------------------------
# 1. Environment report
# ---------------------------------------------------------------------------

def report_environment():
    _section("Environment")
    import platform
    print(f"  Python:   {sys.version.split()[0]}")
    print(f"  Platform: {platform.platform()}")
    print(f"  Machine:  {platform.machine()}")

    # NumPy
    print(f"  NumPy:    {np.__version__}")

    # Numba
    try:
        import numba
        print(f"  Numba:    {numba.__version__}  ✓  (JIT active)")
        _numba_ok = True
    except ImportError:
        print(f"  Numba:    NOT installed  (pip install numba  for ~20x speedup on find_objects)")
        _numba_ok = False

    # CPU topology
    from arc_agent.cpu_utils import default_workers, describe_cpu, _PERFORMANCE_CORES
    import os as _os
    total = _os.cpu_count() or 1
    print(f"  CPU:      {describe_cpu()}")
    print(f"  Default workers: {default_workers()}")

    return _numba_ok


# ---------------------------------------------------------------------------
# 2. Micro-benchmarks
# ---------------------------------------------------------------------------

def benchmark_operations():
    _section("Micro-benchmarks (µs per call)")

    from arc_agent.objects import find_objects, _USE_NUMBA
    from arc_agent.scorer import _structural_similarity_np

    # Build a representative grid: 30×30 with several objects
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

    # --- find_objects ---
    # Warm up
    for _ in range(10):
        find_objects(grid)
    N = 500
    t0 = time.perf_counter()
    for _ in range(N):
        find_objects(grid)
    dt_fo = (time.perf_counter() - t0) / N * 1e6
    impl = "Numba JIT" if _USE_NUMBA else "pure Python"
    print(f"  find_objects (30×30, 3 objects, {impl}): {dt_fo:.0f} µs")

    # --- np.array() conversion ---
    sample = [[r * c % 10 for c in range(30)] for r in range(30)]
    N = 2000
    t0 = time.perf_counter()
    for _ in range(N):
        np.array(sample, dtype=np.uint8)
    dt_arr = (time.perf_counter() - t0) / N * 1e6
    print(f"  np.array() 30×30 list→ndarray:          {dt_arr:.1f} µs")

    # --- _structural_similarity_np ---
    p = np.array(sample, dtype=np.uint8)
    e = np.array(sample, dtype=np.uint8)
    h, w = p.shape
    N = 5000
    t0 = time.perf_counter()
    for _ in range(N):
        _structural_similarity_np(p, e, h, w, h, w)
    dt_score = (time.perf_counter() - t0) / N * 1e6
    print(f"  _structural_similarity_np (30×30):      {dt_score:.1f} µs")

    return dt_fo, _USE_NUMBA


# ---------------------------------------------------------------------------
# 3. Solver accuracy & timing on N tasks
# ---------------------------------------------------------------------------

def benchmark_solver(data_dir: str, n_tasks: int = 20, seed: int = 42):
    _section(f"Solver benchmark ({n_tasks} tasks, seed={seed}, 1 worker)")

    task_files = sorted(glob.glob(os.path.join(data_dir, "*.json")))[:n_tasks]
    if not task_files:
        print(f"  ERROR: no .json files found in {data_dir}")
        return None

    from arc_agent.solver import FourPillarsSolver
    import random
    random.seed(seed)
    np.random.seed(seed)

    solver = FourPillarsSolver(population_size=60, max_generations=30, verbose=False)

    task_times = []
    scores = []
    solved = 0
    partial = 0  # >0.80

    for i, tf in enumerate(task_files, 1):
        task_id = os.path.basename(tf).replace(".json", "")
        with open(tf) as f:
            task = json.load(f)

        t0 = time.perf_counter()
        result = solver.solve_task(task, task_id)
        elapsed = time.perf_counter() - t0

        task_times.append(elapsed)
        scores.append(result["score"])
        if result["solved"]:
            solved += 1
        elif result["score"] > 0.80:
            partial += 1

        status = "✓" if result["solved"] else ("~" if result["score"] > 0.80 else "✗")
        print(f"  [{i:2d}/{n_tasks}] {status} {task_id}  score={result['score']:.3f}  {elapsed:.2f}s")

    print()
    print(f"  Solved (exact):    {solved}/{n_tasks}  ({100*solved/n_tasks:.0f}%)")
    print(f"  Partial (>80%):    {partial}/{n_tasks}  ({100*partial/n_tasks:.0f}%)")
    print(f"  Mean score:        {statistics.mean(scores):.3f}")
    print(f"  Median task time:  {statistics.median(task_times):.2f}s")
    print(f"  Mean task time:    {statistics.mean(task_times):.2f}s")
    print(f"  Total time:        {sum(task_times):.1f}s")
    print(f"  Toolkit size:      {solver.toolkit.size} concepts")

    return task_times, scores


# ---------------------------------------------------------------------------
# 4. Extrapolation
# ---------------------------------------------------------------------------

def extrapolate(task_times: list[float], dt_fo_us: float, numba_active: bool):
    _section("Full benchmark projection (400 tasks)")

    from arc_agent.cpu_utils import default_workers

    mean_1w = statistics.mean(task_times)
    n_workers = default_workers()

    # If Numba is not installed, project what it would look like
    # find_objects is ~25% of compute on object-heavy tasks
    fo_fraction = 0.25
    numba_fo_speedup = 20  # conservative; measured ~5-20x on similar flood-fill kernels
    overall_speedup = 1 / (1 - fo_fraction + fo_fraction / numba_fo_speedup)

    if numba_active:
        mean_with_numba = mean_1w
        mean_without_numba = mean_1w * overall_speedup  # would be slower without
        note = "(Numba already active in your timings above)"
    else:
        mean_with_numba = mean_1w / overall_speedup
        mean_without_numba = mean_1w
        note = "(estimated; install numba to activate)"

    print(f"  Your machine: {n_workers} performance workers")
    print()
    print(f"  Per-task time (1 worker):")
    print(f"    Without Numba:    {mean_without_numba:.2f}s")
    print(f"    With Numba {note}:    {mean_with_numba:.2f}s  ({overall_speedup:.1f}x from JIT)")
    print()
    print(f"  400-task ARC benchmark wall-clock time:")
    print(f"    Without Numba, {n_workers} workers:  {mean_without_numba * 400 / n_workers / 60:.0f} min")
    print(f"    With Numba,    {n_workers} workers:  {mean_with_numba    * 400 / n_workers / 60:.0f} min")
    print()

    if not numba_active:
        print(f"  ► To enable Numba JIT: pip install numba")
        print(f"    First run will compile & cache (~5s). All subsequent runs are instant.")


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
        help="Path to ARC-AGI training JSON files (default: ARC-AGI/data/training)",
    )
    parser.add_argument(
        "--tasks",
        type=int,
        default=20,
        help="Number of tasks to evaluate (default: 20)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    args = parser.parse_args()

    _hline("═")
    print("  Four Pillars AGI — Benchmark")
    _hline("═")

    numba_active = report_environment()
    dt_fo, _ = benchmark_operations()

    result = benchmark_solver(args.data_dir, args.tasks, args.seed)
    if result is not None:
        task_times, scores = result
        extrapolate(task_times, dt_fo, numba_active)

    _hline("═")
    print("  Done.")
    _hline("═")
    print()


if __name__ == "__main__":
    main()
