"""Quick DSL synthesis benchmark on sample tasks.

Tests DSL synthesis on all sample tasks to see how many it can solve
independently (without the full solver pipeline).
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from arc_agent.sample_tasks import SAMPLE_TASKS
from arc_agent.dsl_synth import synthesize_dsl_program
from arc_agent.scorer import TaskCache

dsl_solved = 0
total = len(SAMPLE_TASKS)

for tid, task in sorted(SAMPLE_TASKS.items()):
    cache = TaskCache(task)
    result = synthesize_dsl_program(task, cache, time_budget=5.0, max_depth=2)
    if result is not None and cache.is_pixel_perfect(result):
        dsl_solved += 1
        print(f"  DSL solved: {tid}")

print(f"\nDSL synthesis: {dsl_solved}/{total} sample tasks solved")
