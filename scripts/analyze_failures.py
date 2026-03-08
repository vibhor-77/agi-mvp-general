#!/usr/bin/env python3
"""
Analyze benchmark results to understand failure patterns.

Reads a results JSON file (from evaluate.py) and categorizes failures
by score range, method, and task features. This helps form data-driven
hypotheses for the next improvement.

Usage:
    python scripts/analyze_failures.py results_v024_train.json

IMPORTANT: Only use on TRAINING results, never eval. See RESEARCH_PLAN.md
Section 4.4 (Data Integrity Policy).
"""
import json
import sys
import os
from collections import Counter


def analyze(results_path: str) -> None:
    """Analyze a results file and print failure categories."""
    with open(results_path) as f:
        data = json.load(f)

    task_results = data.get("task_results", [])
    if not task_results:
        print("No task results found.")
        return

    # ── Score distribution ─────────────────────────────────────────────
    solved = [r for r in task_results if r.get("solved")]
    overfits = [r for r in task_results
                if r.get("solved") and not r.get("test_passed")]
    # Actually: solved=pixel_perfect on train, so "solved and not test_passed" = overfit
    # Let's be precise: result["solved"] means pixel-perfect on train
    #   test_passed means test confirmed

    exact = [r for r in task_results
             if r.get("solved") and r.get("test_passed")]
    overfits_list = [r for r in task_results
                     if r.get("solved") and not r.get("test_passed")]
    flukes = [r for r in task_results
              if not r.get("solved") and r.get("test_passed")]

    failed = [r for r in task_results if not r.get("solved")]

    print(f"\n{'='*70}")
    print(f"FAILURE ANALYSIS — {os.path.basename(results_path)}")
    print(f"{'='*70}")
    print(f"\nTotal tasks: {len(task_results)}")
    print(f"Solved exact (train AND test): {len(exact)}")
    print(f"Overfits (train only): {len(overfits_list)}")
    print(f"Flukes (test only): {len(flukes)}")
    print(f"Failed (neither): {len(failed) - len(flukes)}")

    # ── Score bands for failed tasks ───────────────────────────────────
    print(f"\n{'─'*40}")
    print("FAILED TASKS BY SCORE BAND:")
    print(f"{'─'*40}")
    bands = {
        "0.95-0.99 (near miss)": [],
        "0.90-0.95 (close)": [],
        "0.80-0.90 (partial)": [],
        "0.60-0.80 (some progress)": [],
        "0.40-0.60 (low)": [],
        "0.00-0.40 (minimal)": [],
    }
    for r in failed:
        s = r.get("score", 0)
        if s >= 0.95: bands["0.95-0.99 (near miss)"].append(r)
        elif s >= 0.90: bands["0.90-0.95 (close)"].append(r)
        elif s >= 0.80: bands["0.80-0.90 (partial)"].append(r)
        elif s >= 0.60: bands["0.60-0.80 (some progress)"].append(r)
        elif s >= 0.40: bands["0.40-0.60 (low)"].append(r)
        else: bands["0.00-0.40 (minimal)"].append(r)

    for band, tasks in bands.items():
        print(f"  {band}: {len(tasks)} tasks")
        # Show top 5 by score (closest to solving)
        if tasks and band.startswith("0.95"):
            for r in sorted(tasks, key=lambda x: -x.get("score", 0))[:10]:
                print(f"    {r['task_id']}  score={r['score']:.4f}  "
                      f"method={r.get('method', '?')}  "
                      f"len={r.get('program_length', 0)}")

    # ── Overfit analysis ────────────────────────────────────────────────
    if overfits_list:
        print(f"\n{'─'*40}")
        print("OVERFIT TASKS (pixel-perfect train, failed test):")
        print(f"{'─'*40}")
        method_counts = Counter(r.get("method", "?") for r in overfits_list)
        print(f"  By method: {dict(method_counts)}")
        for r in sorted(overfits_list, key=lambda x: x.get("task_id", "")):
            print(f"    {r['task_id']}  method={r.get('method', '?')}  "
                  f"len={r.get('program_length', 0)}  "
                  f"test_score={r.get('test_score', 0):.3f}  "
                  f"n_cand={r.get('n_candidates', 0)}")

    # ── Method distribution ─────────────────────────────────────────────
    print(f"\n{'─'*40}")
    print("SOLVING METHODS (for solved tasks):")
    print(f"{'─'*40}")
    method_counts = Counter(r.get("method", "?") for r in solved)
    for method, count in method_counts.most_common():
        print(f"  {method}: {count}")

    print(f"\n{'─'*40}")
    print("BEST METHOD FOR FAILED TASKS (what got closest):")
    print(f"{'─'*40}")
    method_counts = Counter(r.get("method", "?") for r in failed)
    for method, count in method_counts.most_common():
        print(f"  {method}: {count}")

    # ── Near-misses: tasks scoring 0.95+ but not pixel-perfect ──────────
    near_misses = [r for r in failed if r.get("score", 0) >= 0.90]
    if near_misses:
        print(f"\n{'─'*40}")
        print(f"NEAR-MISS TASKS (score >= 0.90, {len(near_misses)} tasks):")
        print(f"These are the LOWEST-HANGING FRUIT for improvement.")
        print(f"{'─'*40}")
        for r in sorted(near_misses, key=lambda x: -x.get("score", 0))[:20]:
            print(f"  {r['task_id']}  score={r['score']:.4f}  "
                  f"method={r.get('method', '?')}  "
                  f"len={r.get('program_length', 0)}")

    # ── Program length distribution ─────────────────────────────────────
    print(f"\n{'─'*40}")
    print("PROGRAM LENGTH DISTRIBUTION (solved tasks):")
    print(f"{'─'*40}")
    len_counts = Counter(r.get("program_length", 0) for r in solved)
    for length, count in sorted(len_counts.items()):
        bar = "█" * count
        print(f"  length {length}: {count:3d}  {bar}")

    # ── Summary recommendations ─────────────────────────────────────────
    print(f"\n{'='*70}")
    print("RECOMMENDATIONS:")
    print(f"{'='*70}")

    n_near = len([r for r in failed if r.get("score", 0) >= 0.95])
    n_close = len([r for r in failed if 0.90 <= r.get("score", 0) < 0.95])
    n_partial = len([r for r in failed if 0.80 <= r.get("score", 0) < 0.90])
    n_low = len([r for r in failed if r.get("score", 0) < 0.80])

    print(f"  1. {n_near} near-misses (0.95+): these need tiny fixes")
    print(f"  2. {n_close} close tasks (0.90-0.95): worth investigating")
    print(f"  3. {n_partial} partial tasks (0.80-0.90): need new approach")
    print(f"  4. {n_low} low-scoring tasks (<0.80): fundamentally different")
    print(f"  5. {len(overfits_list)} overfits: programs memorize but don't generalize")
    print()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/analyze_failures.py results_file.json")
        sys.exit(1)
    analyze(sys.argv[1])
