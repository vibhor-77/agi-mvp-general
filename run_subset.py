#!/usr/bin/env python3
"""Run the solver on a specific subset of tasks for quick hypothesis testing.

Usage:
    python run_subset.py TASK_IDS [--culture FILE] [--data-dir DIR] [--workers N]

    TASK_IDS: comma-separated task IDs (e.g., "91714a58,3631a71a,73251a56")

Example:
    python run_subset.py 91714a58,3631a71a --culture cultures/latest_training.json
"""
import argparse
import json
import os
import sys
import time
from multiprocessing import Pool, cpu_count


def solve_task(args):
    """Solve a single task and return results."""
    task_id, task_path, culture_path = args

    from arc_agent.solver import FourPillarsSolver

    solver = FourPillarsSolver(
        population_size=60,
        max_generations=30,
        verbose=False,
    )

    if culture_path:
        from arc_agent.culture import load_culture
        load_culture(solver.toolkit, culture_path, solver.archive)

    with open(task_path) as f:
        task = json.load(f)

    start = time.time()
    result = solver.solve_task(task, task_id=task_id)
    elapsed = time.time() - start

    return {
        "task_id": task_id,
        "score": result.get("score", 0),
        "test_score": result.get("test_score", 0),
        "test_confirmed": result.get("test_confirmed", False),
        "solved": result.get("solved", False),
        "method": result.get("method", "?"),
        "program": result.get("program", "?"),
        "elapsed": elapsed,
    }


def main():
    parser = argparse.ArgumentParser(description="Run solver on specific tasks")
    parser.add_argument("task_ids", help="Comma-separated task IDs")
    parser.add_argument("--culture", help="Culture file to load")
    parser.add_argument("--data-dir", default="ARC-AGI/data/training")
    parser.add_argument("--workers", type=int, default=0, help="0=auto")
    args = parser.parse_args()

    task_ids = [t.strip() for t in args.task_ids.split(",") if t.strip()]
    workers = args.workers or min(cpu_count(), len(task_ids))

    # Verify tasks exist
    task_args = []
    for tid in task_ids:
        path = os.path.join(args.data_dir, f"{tid}.json")
        if not os.path.exists(path):
            print(f"  ⚠ Task {tid} not found at {path}, skipping")
            continue
        task_args.append((tid, path, args.culture))

    print(f"Running {len(task_args)} tasks on {workers} workers...")
    start = time.time()

    if workers == 1:
        results = [solve_task(a) for a in task_args]
    else:
        with Pool(workers) as pool:
            results = pool.map(solve_task, task_args)

    elapsed = time.time() - start

    # Print results
    exact = sum(1 for r in results if r["test_confirmed"])
    overfit = sum(1 for r in results if r["solved"] and not r["test_confirmed"])
    fail = sum(1 for r in results if not r["solved"])

    print(f"\n{'='*70}")
    print(f"  Results: {len(results)} tasks in {elapsed:.1f}s")
    print(f"  exact={exact}  overfit={overfit}  fail={fail}")
    print(f"{'='*70}")

    for r in sorted(results, key=lambda x: x["score"], reverse=True):
        cat = "exact" if r["test_confirmed"] else ("overfit" if r["solved"] else "fail")
        icon = "✓" if cat == "exact" else ("◇" if cat == "overfit" else "✗")
        print(f"  {icon} {r['task_id'][:8]}  train={r['score']:.4f}  test={r['test_score']:.4f}  "
              f"{r['elapsed']:5.1f}s  {cat:8s}  {r['method']:20s}  {r['program'][:50]}")

    return results


if __name__ == "__main__":
    main()
