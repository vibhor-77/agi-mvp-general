#!/usr/bin/env python3
"""Debug why pure-recolor tasks aren't being solved by object rules."""
import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from arc_agent.scene import (
    build_scene, diff_scenes, find_consistent_rules, apply_rules,
    solve_with_object_rules, ObjectRule
)
from arc_agent.dataset import load_dataset


def debug_task(tid, task):
    """Deep dive into one task."""
    train = task["train"]

    # Check same-dims
    same_dims = all(
        len(ex["input"]) == len(ex["output"])
        and len(ex["input"][0]) == len(ex["output"][0])
        for ex in train
    )
    if not same_dims:
        return None

    diffs = []
    for i, ex in enumerate(train):
        src = build_scene(ex["input"])
        dst = build_scene(ex["output"])
        diff = diff_scenes(src, dst)
        diffs.append(diff)

        # Check if this has ONLY recolor (no movement, addition, removal, shape change)
        has_non_recolor = (
            len(diff.removed) > 0 or
            len(diff.added) > 0 or
            any(m.movement != (0, 0) for m in diff.matched) or
            any(not m.shape_preserved for m in diff.matched) or
            any(m.size_delta != 0 for m in diff.matched)
        )
        has_recolor = any(m.new_color is not None for m in diff.matched)

        if has_non_recolor:
            return None  # Not pure recolor
        if not has_recolor:
            return None  # Nothing changes?

    # This IS a pure recolor task. Why isn't it solved?
    rules = find_consistent_rules(diffs)

    # Show what colors change to what
    color_changes = Counter()
    for diff in diffs:
        for m in diff.matched:
            if m.new_color is not None:
                color_changes[(m.src.color, m.new_color)] += 1

    # Try applying rules
    if rules:
        all_correct = True
        for ex in train:
            bg = build_scene(ex["input"]).bg_color
            pred = apply_rules(ex["input"], rules, bg)
            if pred != ex["output"]:
                all_correct = False
                # Find first pixel diff
                for r in range(len(pred)):
                    for c in range(len(pred[0])):
                        if pred[r][c] != ex["output"][r][c]:
                            print(f"    Pixel diff at ({r},{c}): pred={pred[r][c]} expected={ex['output'][r][c]} input={ex['input'][r][c]}")
                            break
                    else:
                        continue
                    break
        return {
            "color_changes": dict(color_changes),
            "rules": [(r.kind, r.src_color, r.dst_color) for r in rules],
            "all_correct": all_correct,
        }
    else:
        return {
            "color_changes": dict(color_changes),
            "rules": [],
            "reason": "no consistent rules found",
        }


def main():
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "ARC-AGI/data/training"
    tasks = load_dataset(data_dir)

    print("Pure recolor tasks (recolor only, no movement/addition/removal/shape change):\n")

    count = 0
    for tid in sorted(tasks):
        result = debug_task(tid, tasks[tid])
        if result is not None:
            count += 1
            status = "SOLVED" if result.get("all_correct") else "FAILED"
            print(f"{tid}: {status}")
            print(f"  Color changes: {result['color_changes']}")
            print(f"  Rules found: {result['rules']}")
            if "reason" in result:
                print(f"  Reason: {result['reason']}")
            print()

    print(f"\nTotal pure recolor tasks: {count}")


if __name__ == "__main__":
    main()
