"""Check DSL synthesis on a broad sample of ARC tasks.

Runs ONLY the DSL synthesis engine (not the full solver) on all 400
tasks to see which ones it can solve independently.
"""
import glob
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from arc_agent.dsl_synth import synthesize_dsl_program
from arc_agent.scorer import TaskCache

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "ARC-AGI", "data", "training")

task_files = sorted(glob.glob(os.path.join(DATA_DIR, "*.json")))
if not task_files:
    print(f"ERROR: no .json files found in {DATA_DIR}")
    sys.exit(1)

solved = 0
total = len(task_files)
t_start = time.perf_counter()

for tf in task_files:
    task_id = os.path.basename(tf).replace(".json", "")
    with open(tf) as f:
        task = json.load(f)

    cache = TaskCache(task)
    result = synthesize_dsl_program(task, cache, time_budget=5.0, max_depth=2)
    if result is not None and cache.is_pixel_perfect(result):
        solved += 1
        print(f"  DSL solved: {task_id}")

elapsed = time.perf_counter() - t_start
print(f"\nDSL synthesis: {solved}/{total} tasks solved ({100*solved/total:.1f}%)")
print(f"Total time: {elapsed:.1f}s ({elapsed/total:.2f}s/task)")
