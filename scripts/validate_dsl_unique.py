"""Check which DSL solves are unique (not found by single primitives).

Compares DSL synthesis results against single-primitive search to
identify tasks where DSL provides the ONLY solution.
"""
import glob
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from arc_agent.dsl_synth import synthesize_dsl_program
from arc_agent.scorer import TaskCache
from arc_agent.solver import FourPillarsSolver

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "ARC-AGI", "data", "training")

# DSL-solved task IDs from the previous run
DSL_SOLVED = [
    "0d3d703e", "1cf80156", "1e0a9b12", "3906de3d", "3aa6fb7a",
    "3c9b0459", "4347f46a", "496994bd", "50cb2852", "6150a2bd",
    "67a3c6ac", "68b16354", "7468f01a", "74dd1130", "7e0986d6",
    "7f4411dc", "9172f3a0", "94f9d214", "9dfd6313", "a416b8f3",
    "a699fb00", "a740d043", "aabf363d", "b1948b0a", "bb43febb",
    "bda2d7a6", "c59eb873", "c8f0f002", "d511f180", "dae9d2b5",
    "ed36ccf7", "f25fbde4", "f25ffba3", "fafffa47",
]

# For each DSL-solved task, check if a single primitive also solves it
solver = FourPillarsSolver(population_size=60, max_generations=30, verbose=False)

dsl_only = []
both = []

for task_id in DSL_SOLVED:
    tf = os.path.join(DATA_DIR, f"{task_id}.json")
    with open(tf) as f:
        task = json.load(f)

    cache = TaskCache(task)

    # Try single primitives only
    single = solver._try_single_primitives(task, cache)
    if single and cache.is_pixel_perfect(single):
        both.append(task_id)
    else:
        dsl_only.append(task_id)

print(f"DSL solved: {len(DSL_SOLVED)} tasks")
print(f"Also solved by single primitive: {len(both)}")
print(f"DSL-ONLY (unique contribution): {len(dsl_only)}")
print()
if dsl_only:
    print("DSL-unique tasks:")
    for tid in dsl_only:
        print(f"  {tid}")
