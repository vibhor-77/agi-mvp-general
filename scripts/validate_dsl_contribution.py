"""Check DSL's real contribution: run full solver and count DSL-method wins."""
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

# Run on 50 evenly-spaced tasks for a representative sample
task_files = sorted(glob.glob(os.path.join(DATA_DIR, "*.json")))
step = max(1, len(task_files) // 50)
task_files = task_files[::step][:50]

solver = FourPillarsSolver(population_size=60, max_generations=30, verbose=False)

solved = 0
by_method = {}
dsl_candidates = 0

for i, tf in enumerate(task_files, 1):
    task_id = os.path.basename(tf).replace(".json", "")
    with open(tf) as f:
        task = json.load(f)

    result = solver.solve_task(task, task_id)
    method = result["method"]

    if result["solved"]:
        solved += 1
        by_method[method] = by_method.get(method, 0) + 1

    # Count tasks where DSL found at least one candidate
    cands = result.get("candidates", [])
    dsl_cands = [c for c in cands if "dsl" in c.get("method", "")]
    if dsl_cands:
        dsl_candidates += 1

    status = "X" if result["solved"] else ("~" if result["score"] > 0.80 else ".")
    dsl_tag = f" [DSL:{len(dsl_cands)}]" if dsl_cands else ""
    print(f"  [{i:2d}/{len(task_files)}] {status} {task_id}  "
          f"score={result['score']:.3f}  {method}{dsl_tag}")

print(f"\nSolved: {solved}/{len(task_files)} ({100*solved/len(task_files):.0f}%)")
print(f"DSL candidates found: {dsl_candidates} tasks")
print(f"\nBy method:")
for m, c in sorted(by_method.items(), key=lambda x: -x[1]):
    print(f"  {m}: {c}")
