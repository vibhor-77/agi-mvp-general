"""Validate solver on a subset of real ARC tasks.

Runs the full solver pipeline (including DSL synthesis) on 30 tasks
to verify correctness and measure solve rate.
"""
import glob
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from arc_agent.solver import FourPillarsSolver
from arc_agent.scorer import TaskCache

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "ARC-AGI", "data", "training")

N_TASKS = 30

task_files = sorted(glob.glob(os.path.join(DATA_DIR, "*.json")))[:N_TASKS]
if not task_files:
    print(f"ERROR: no .json files found in {DATA_DIR}")
    sys.exit(1)

solver = FourPillarsSolver(population_size=60, max_generations=30, verbose=False)

solved = 0
dsl_solved = 0
total_time = 0.0

for i, tf in enumerate(task_files, 1):
    task_id = os.path.basename(tf).replace(".json", "")
    with open(tf) as f:
        task = json.load(f)

    t0 = time.perf_counter()
    result = solver.solve_task(task, task_id)
    elapsed = time.perf_counter() - t0
    total_time += elapsed

    status = "X" if result["solved"] else ("~" if result["score"] > 0.80 else ".")
    method = result.get("method", "")

    # Check if DSL was involved
    if "dsl" in method.lower():
        if result["solved"]:
            dsl_solved += 1

    # Check all candidates for DSL contributions
    dsl_cands = [c for c in result.get("candidates", []) if "dsl" in c.get("method", "")]
    dsl_tag = f" [DSL:{len(dsl_cands)}]" if dsl_cands else ""

    if result["solved"]:
        solved += 1

    print(f"  [{i:2d}/{N_TASKS}] {status} {task_id}  "
          f"score={result['score']:.3f}  {elapsed:.1f}s  {method}{dsl_tag}")

print(f"\nSolved: {solved}/{N_TASKS} ({100*solved/N_TASKS:.0f}%)")
print(f"DSL solves: {dsl_solved}")
print(f"Total time: {total_time:.1f}s")
print(f"Mean time: {total_time/N_TASKS:.1f}s")
print(f"Toolkit size: {solver.toolkit.size} concepts")
