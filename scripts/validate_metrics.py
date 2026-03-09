"""Validate the updated metrics (test_confirmed, flukes, overfit) on 10 tasks."""
import glob
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from arc_agent.solver import FourPillarsSolver

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "ARC-AGI", "data", "training")

task_files = sorted(glob.glob(os.path.join(DATA_DIR, "*.json")))[:10]
solver = FourPillarsSolver(population_size=60, max_generations=30, verbose=False)

train_perfect = 0
test_confirmed = 0
flukes = 0

for i, tf in enumerate(task_files, 1):
    task_id = os.path.basename(tf).replace(".json", "")
    with open(tf) as f:
        task = json.load(f)

    result = solver.solve_task(task, task_id)

    tp = result["solved"]
    tc = result.get("test_confirmed", False)
    fl = result.get("fluke", False)
    te = result.get("test_exact", False)

    if tp:
        train_perfect += 1
    if tc:
        test_confirmed += 1
    if fl:
        flukes += 1

    # Status symbol
    if tc:
        status = "V"  # test confirmed
    elif tp:
        status = "O"  # overfit (train only)
    elif fl:
        status = "F"  # fluke
    elif result["score"] > 0.80:
        status = "~"
    else:
        status = "."

    print(f"  [{i:2d}] {status} {task_id}  train={tp}  test={te}  "
          f"confirmed={tc}  fluke={fl}  score={result['score']:.3f}")

print(f"\nTrain-perfect: {train_perfect}/10")
print(f"Test-confirmed: {test_confirmed}/10")
print(f"Flukes: {flukes}/10")
print(f"Overfit: {train_perfect - test_confirmed}/10")
