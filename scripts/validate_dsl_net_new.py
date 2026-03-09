"""Check which DSL-unique tasks are NET NEW (not solved by full solver otherwise).

Runs the full solver WITHOUT DSL synthesis on DSL-unique tasks to see
which ones would be missed without the DSL.
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

DSL_UNIQUE = [
    "0d3d703e", "3aa6fb7a", "50cb2852", "7468f01a", "7e0986d6",
    "7f4411dc", "94f9d214", "a699fb00", "a740d043", "aabf363d",
    "b1948b0a", "bb43febb", "bda2d7a6", "c8f0f002", "d511f180",
    "dae9d2b5", "f25fbde4", "fafffa47",
]

# Run full solver on these tasks and check if they're solved
solver = FourPillarsSolver(population_size=60, max_generations=30, verbose=False)

net_new = []
already_solved = []

for task_id in DSL_UNIQUE:
    tf = os.path.join(DATA_DIR, f"{task_id}.json")
    with open(tf) as f:
        task = json.load(f)

    result = solver.solve_task(task, task_id)

    method = result["method"]
    score = result["score"]
    if result["solved"]:
        already_solved.append((task_id, method))
    else:
        net_new.append((task_id, score))

    status = "SOLVED" if result["solved"] else f"score={score:.3f}"
    print(f"  {task_id}: {status}  method={method}")

print(f"\nOf {len(DSL_UNIQUE)} DSL-unique tasks:")
print(f"  Already solved by full solver: {len(already_solved)}")
print(f"  NET NEW from DSL: {len(net_new)}")
if net_new:
    print("\nNet-new tasks (would be missed without DSL):")
    for tid, score in net_new:
        print(f"  {tid} (best score without DSL: {score:.3f})")
