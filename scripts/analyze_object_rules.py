#!/usr/bin/env python3
"""Analyze ARC tasks to understand what object-level rules would help.

For each same-dims task, runs the scene pipeline and reports:
- How many objects are detected in input/output
- What types of diffs are found (recolor, remove, add, move, shape change)
- Whether consistent rules are found
- Whether the rules produce correct output
- What's MISSING if rules fail

This guides Phase 2 development by showing what rule types we need.
"""
import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from arc_agent.scene import (
    build_scene, diff_scenes, find_consistent_rules, apply_rules,
    solve_with_object_rules
)
from arc_agent.dataset import load_dataset


def analyze_task(task_id: str, task: dict) -> dict:
    """Analyze one task for object-level patterns."""
    train = task.get("train", [])
    if not train:
        return {"skip": "no_train"}

    # Check same-dims
    same_dims = all(
        len(ex["input"]) == len(ex["output"])
        and len(ex["input"][0]) == len(ex["output"][0])
        for ex in train
    )
    if not same_dims:
        return {"type": "diff_dims"}

    # Build scenes and diffs
    diffs = []
    scenes = []
    for ex in train:
        src = build_scene(ex["input"])
        dst = build_scene(ex["output"])
        scenes.append((src, dst))
        diffs.append(diff_scenes(src, dst))

    # Categorize what's happening
    has_recolor = any(any(m.new_color is not None for m in d.matched) for d in diffs)
    has_removal = any(len(d.removed) > 0 for d in diffs)
    has_addition = any(len(d.added) > 0 for d in diffs)
    has_movement = any(any(m.movement != (0, 0) for m in d.matched) for d in diffs)
    has_shape_change = any(any(not m.shape_preserved for m in d.matched) for d in diffs)
    has_size_change = any(any(m.size_delta != 0 for m in d.matched) for d in diffs)

    # Try to find rules
    rules = find_consistent_rules(diffs)
    rule_kinds = [r.kind for r in rules]

    # Try full pipeline
    transform = solve_with_object_rules(task)
    solved = transform is not None

    # If solved, check test too
    test_correct = False
    if solved and "test" in task:
        for tex in task["test"]:
            pred = transform(tex["input"])
            if pred == tex.get("output"):
                test_correct = True

    # Count objects
    avg_input_objs = sum(len(s.objects) for s, _ in scenes) / len(scenes)
    avg_output_objs = sum(len(d.objects) for _, d in scenes) / len(scenes)

    # Categorize the diff pattern
    patterns = []
    if has_recolor: patterns.append("recolor")
    if has_removal: patterns.append("removal")
    if has_addition: patterns.append("addition")
    if has_movement: patterns.append("movement")
    if has_shape_change: patterns.append("shape_change")
    if has_size_change: patterns.append("size_change")

    return {
        "type": "same_dims",
        "solved": solved,
        "test_correct": test_correct,
        "patterns": patterns,
        "rule_kinds": rule_kinds,
        "avg_input_objs": round(avg_input_objs, 1),
        "avg_output_objs": round(avg_output_objs, 1),
        "num_examples": len(train),
    }


def main():
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "ARC-AGI/data/training"
    if not os.path.isdir(data_dir):
        print(f"Usage: python {sys.argv[0]} <data-dir>")
        print(f"  e.g., python {sys.argv[0]} ARC-AGI/data/training")
        sys.exit(1)

    tasks = load_dataset(data_dir)
    print(f"Loaded {len(tasks)} tasks from {data_dir}\n")

    results = {}
    pattern_counts = Counter()
    solved_count = 0
    same_dims_count = 0

    for tid in sorted(tasks):
        r = analyze_task(tid, tasks[tid])
        results[tid] = r
        if r.get("type") == "same_dims":
            same_dims_count += 1
            for p in r.get("patterns", []):
                pattern_counts[p] += 1
            if r.get("solved"):
                solved_count += 1

    # Summary
    diff_dims = sum(1 for r in results.values() if r.get("type") == "diff_dims")
    print(f"Same-dims tasks: {same_dims_count}/{len(tasks)} ({100*same_dims_count/len(tasks):.0f}%)")
    print(f"Diff-dims tasks: {diff_dims}/{len(tasks)} ({100*diff_dims/len(tasks):.0f}%)")
    print(f"\nSolved by object rules: {solved_count}/{same_dims_count} same-dims tasks")

    print(f"\nPattern distribution (in same-dims tasks):")
    for pattern, count in pattern_counts.most_common():
        print(f"  {pattern}: {count} ({100*count/same_dims_count:.0f}%)")

    # Group unsolved by pattern combination
    print(f"\nUnsolved same-dims tasks by pattern combination:")
    combo_counts = Counter()
    for r in results.values():
        if r.get("type") == "same_dims" and not r.get("solved"):
            combo = "+".join(sorted(r.get("patterns", []))) or "no_change_detected"
            combo_counts[combo] += 1

    for combo, count in combo_counts.most_common(15):
        print(f"  {combo}: {count}")

    # Show solved tasks
    solved_tasks = [tid for tid, r in results.items() if r.get("solved")]
    if solved_tasks:
        print(f"\nSolved tasks ({len(solved_tasks)}):")
        for tid in solved_tasks:
            r = results[tid]
            test_str = " (test correct!)" if r.get("test_correct") else " (test N/A or wrong)"
            print(f"  {tid}: rules={r['rule_kinds']}{test_str}")


if __name__ == "__main__":
    main()
