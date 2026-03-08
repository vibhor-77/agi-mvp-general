#!/usr/bin/env python3
"""
Inspect specific ARC tasks: show inputs/outputs as ASCII grids.

Helps debug why the solver fails on specific tasks by making the
transformation visually obvious (as a human would see it).

Usage:
    python scripts/inspect_task.py ARC-AGI/data/training/0b148d64.json
    python scripts/inspect_task.py ARC-AGI/data/training/ --tasks 0b148d64 2204b7a8

IMPORTANT: Only use on TRAINING tasks, never eval. See RESEARCH_PLAN.md
Section 4.4 (Data Integrity Policy).
"""
import json
import sys
import os
import glob

# Color codes for terminal display (0-9 map to ARC colors)
# 0=black, 1=blue, 2=red, 3=green, 4=yellow, 5=gray, 6=pink, 7=orange, 8=cyan, 9=maroon
SYMBOLS = "·12345678 "
COLORS = {
    0: "\033[90m·\033[0m",   # dark gray dot for background
    1: "\033[34m1\033[0m",   # blue
    2: "\033[31m2\033[0m",   # red
    3: "\033[32m3\033[0m",   # green
    4: "\033[33m4\033[0m",   # yellow
    5: "\033[37m5\033[0m",   # gray
    6: "\033[35m6\033[0m",   # magenta/pink
    7: "\033[91m7\033[0m",   # orange (bright red)
    8: "\033[36m8\033[0m",   # cyan
    9: "\033[35m9\033[0m",   # maroon
}


def render_grid(grid: list[list[int]], colored: bool = True) -> str:
    """Render a grid as an ASCII string."""
    lines = []
    for row in grid:
        if colored:
            line = " ".join(COLORS.get(cell, str(cell)) for cell in row)
        else:
            line = " ".join(SYMBOLS[cell] if 0 <= cell <= 9 else str(cell)
                           for cell in row)
        lines.append(line)
    return "\n".join(lines)


def render_side_by_side(input_grid, output_grid, colored=True) -> str:
    """Render input and output grids side by side."""
    in_lines = render_grid(input_grid, colored).split("\n")
    out_lines = render_grid(output_grid, colored).split("\n")

    # Pad to same height
    max_h = max(len(in_lines), len(out_lines))
    in_w = max(len(line) for line in in_lines) if in_lines else 0
    # Calculate actual char width (without ANSI escape codes)
    in_w_raw = max(len(row) * 2 - 1 for row in input_grid) if input_grid else 0

    while len(in_lines) < max_h:
        in_lines.append(" " * in_w_raw)
    while len(out_lines) < max_h:
        out_lines.append("")

    result = []
    for i, (il, ol) in enumerate(zip(in_lines, out_lines)):
        # Pad input line to consistent width
        padding = " " * max(0, in_w_raw - len(il.replace("\033[90m", "").replace("\033[34m", "").replace("\033[31m", "").replace("\033[32m", "").replace("\033[33m", "").replace("\033[37m", "").replace("\033[35m", "").replace("\033[91m", "").replace("\033[36m", "").replace("\033[0m", "")))
        arrow = " → " if i == max_h // 2 else "   "
        result.append(f"{il}{padding}{arrow}{ol}")

    return "\n".join(result)


def inspect_task(task_path: str, colored: bool = True) -> None:
    """Load and display a single ARC task."""
    with open(task_path) as f:
        task = json.load(f)

    task_id = os.path.splitext(os.path.basename(task_path))[0]
    print(f"\n{'='*60}")
    print(f"Task: {task_id}")
    print(f"{'='*60}")

    # Basic features
    train = task.get("train", [])
    test = task.get("test", [])

    print(f"Train examples: {len(train)}")
    print(f"Test examples: {len(test)}")

    # Analyze dimensions
    for i, ex in enumerate(train):
        in_h, in_w = len(ex["input"]), len(ex["input"][0])
        out_h, out_w = len(ex["output"]), len(ex["output"][0])
        dim_change = ""
        if in_h == out_h and in_w == out_w:
            dim_change = "same-dims"
        elif in_h * 2 == out_h and in_w * 2 == out_w:
            dim_change = "2x scale"
        elif in_h * 3 == out_h and in_w * 3 == out_w:
            dim_change = "3x scale"
        elif out_h < in_h or out_w < in_w:
            dim_change = "shrinks"
        else:
            dim_change = f"{in_h}x{in_w} → {out_h}x{out_w}"
        print(f"\n  Train example {i+1} ({dim_change}):")
        print(render_side_by_side(ex["input"], ex["output"], colored))

    for i, ex in enumerate(test):
        in_h, in_w = len(ex["input"]), len(ex["input"][0])
        if "output" in ex:
            out_h, out_w = len(ex["output"]), len(ex["output"][0])
            print(f"\n  Test example {i+1} ({in_h}x{in_w} → {out_h}x{out_w}):")
            print(render_side_by_side(ex["input"], ex["output"], colored))
        else:
            print(f"\n  Test example {i+1} ({in_h}x{in_w} → ?):")
            print(render_grid(ex["input"], colored))

    # Quick analysis: what kind of transformation is this?
    print(f"\n  Quick analysis:")
    if train:
        all_same_dims = all(
            len(ex["input"]) == len(ex["output"])
            and len(ex["input"][0]) == len(ex["output"][0])
            for ex in train
        )
        print(f"    Same dimensions: {all_same_dims}")

        # Color analysis
        in_colors = set()
        out_colors = set()
        for ex in train:
            for row in ex["input"]:
                in_colors.update(row)
            for row in ex["output"]:
                out_colors.update(row)
        new_colors = out_colors - in_colors
        removed_colors = in_colors - out_colors
        print(f"    Input colors: {sorted(in_colors)}")
        print(f"    Output colors: {sorted(out_colors)}")
        if new_colors:
            print(f"    New colors in output: {sorted(new_colors)}")
        if removed_colors:
            print(f"    Colors removed: {sorted(removed_colors)}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Inspect ARC tasks")
    parser.add_argument("path", help="Task JSON file or directory")
    parser.add_argument("--tasks", nargs="+", default=None,
                       help="Specific task IDs to inspect")
    parser.add_argument("--no-color", action="store_true",
                       help="Disable colored output")
    args = parser.parse_args()

    colored = not args.no_color

    if os.path.isfile(args.path):
        inspect_task(args.path, colored)
    elif os.path.isdir(args.path):
        if args.tasks:
            for tid in args.tasks:
                fpath = os.path.join(args.path, f"{tid}.json")
                if os.path.isfile(fpath):
                    inspect_task(fpath, colored)
                else:
                    print(f"Warning: task {tid} not found at {fpath}")
        else:
            # Show first 5 tasks
            files = sorted(glob.glob(os.path.join(args.path, "*.json")))[:5]
            for fpath in files:
                inspect_task(fpath, colored)
            if len(files) < len(glob.glob(os.path.join(args.path, "*.json"))):
                print(f"\n... and more. Use --tasks to specify specific IDs.")
    else:
        print(f"Error: {args.path} not found")
        sys.exit(1)


if __name__ == "__main__":
    main()
