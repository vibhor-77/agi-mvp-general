#!/usr/bin/env python3
"""
Four Pillars AGI Agent — Main Entry Point

Demonstrates Vibhor Jain's 4 Pillars framework on ARC-AGI tasks:
  1. Feedback Loops: Tight scoring against training examples
  2. Approximability: Evolutionary refinement converges toward solutions
  3. Composability: Programs are built from composable concept primitives
  4. Exploration: UCB-based explore/exploit tradeoff

Usage:
    python -m arc_agent.main                    # Full evaluation
    python -m arc_agent.main --task mirror_h    # Single task
    python -m arc_agent.main --save-toolkit t.json  # Save after eval
    python -m arc_agent.main --load-toolkit t.json  # Resume from saved
"""
import argparse
import json
import os
import random
from .solver import FourPillarsSolver
from .sample_tasks import SAMPLE_TASKS
from .scorer import validate_on_test
from .persistence import save_toolkit, load_toolkit, save_archive


def run_evaluation(
    save_path: str = "",
    load_path: str = "",
    archive_path: str = "",
) -> dict:
    """Run the full evaluation on sample ARC-AGI tasks.

    Args:
        save_path: If set, save the Toolkit to this path after evaluation.
        load_path: If set, load a pre-trained Toolkit from this path.
        archive_path: If set, save the Archive to this path after evaluation.

    Returns:
        Dict mapping task_id → result dict.
    """
    print("=" * 60)
    print("FOUR PILLARS AGI AGENT — ARC-AGI Evaluation")
    print("Based on the research of Vibhor Jain")
    print("=" * 60)
    print()
    print("Pillars:")
    print("  1. Feedback Loops  — Tight scoring against training examples")
    print("  2. Approximability — Evolutionary refinement toward truth")
    print("  3. Composability   — Recursive concept building blocks")
    print("  4. Exploration     — UCB explore/exploit tradeoff")
    print()

    # Set random seed for reproducibility
    random.seed(42)

    # Create the solver (optionally loading a saved toolkit)
    solver = FourPillarsSolver(
        population_size=60,
        max_generations=30,
        max_program_length=4,
        verbose=True,
    )

    # If loading a pre-trained toolkit, replace the default one
    if load_path and os.path.exists(load_path):
        print(f"Loading toolkit from: {load_path}")
        solver.toolkit = load_toolkit(load_path)
        # Re-wire synthesizer and explorer to use loaded toolkit
        solver.synthesizer.toolkit = solver.toolkit
        solver.explorer.toolkit = solver.toolkit
        print(f"Loaded toolkit: {solver.toolkit.size} concepts")
    else:
        print(f"Initial toolkit: {solver.toolkit.size} concepts")

    print()

    # Solve all sample tasks
    initial_size = solver.toolkit.size
    results = solver.solve_batch(SAMPLE_TASKS)

    # Validate on test examples
    print("\n" + "=" * 60)
    print("TEST VALIDATION (held-out examples)")
    print("=" * 60)

    test_correct = 0
    test_total = 0

    for task_id, result in results.items():
        task = SAMPLE_TASKS[task_id]
        if result["solved"]:
            programs = solver.archive.task_solutions.get(task_id, [])
            if programs:
                program = programs[0]
                exact, test_score = validate_on_test(program, task)
                test_total += 1
                if exact:
                    test_correct += 1
                    print(f"  ✓ {task_id}: test PASSED (score={test_score:.3f})")
                else:
                    print(f"  ~ {task_id}: test partial (score={test_score:.3f})")
        else:
            test_total += 1
            print(f"  ✗ {task_id}: not solved (train score={result['score']:.3f})")

    if test_total > 0:
        print(f"\nTest accuracy: {test_correct}/{test_total} "
              f"({100*test_correct/test_total:.1f}%)")

    # Print the concept library growth (cumulative culture metric)
    print(f"\n{'='*60}")
    print("CONCEPT LIBRARY GROWTH (Cumulative Culture Metric)")
    print(f"{'='*60}")
    for i, size in enumerate(solver.concept_growth):
        task_id = list(SAMPLE_TASKS.keys())[i]
        bar = "█" * (size // 2)
        print(f"  After task {i+1:2d} ({task_id:20s}): {size:3d} concepts {bar}")

    # Summary of the 4 pillars in action
    print(f"\n{'='*60}")
    print("FOUR PILLARS IN ACTION")
    print(f"{'='*60}")
    print(f"  Pillar 1 (Feedback):      {solver.tasks_attempted} tasks scored")
    print(f"  Pillar 2 (Approximation): Avg score = "
          f"{sum(r['score'] for r in results.values())/len(results):.3f}")
    print(f"  Pillar 3 (Composability): "
          f"{solver.toolkit.size - initial_size} new concepts learned")
    print(f"  Pillar 4 (Exploration):   epsilon = {solver.explorer.epsilon:.3f}")

    # Save toolkit and archive if requested
    if save_path:
        save_toolkit(solver.toolkit, save_path)
        print(f"\nToolkit saved to: {save_path}")
    if archive_path:
        save_archive(solver.archive, archive_path)
        print(f"Archive saved to: {archive_path}")

    return results


def run_single_task(task_name: str):
    """Run on a single task for debugging."""
    if task_name not in SAMPLE_TASKS:
        print(f"Unknown task: {task_name}")
        print(f"Available: {list(SAMPLE_TASKS.keys())}")
        return

    random.seed(42)
    solver = FourPillarsSolver(
        population_size=40,
        max_generations=20,
        verbose=True,
    )

    task = SAMPLE_TASKS[task_name]
    result = solver.solve_task(task, task_name)

    print(f"\nResult: {json.dumps(result, indent=2)}")


def main():
    """CLI entry point with argument parsing."""
    parser = argparse.ArgumentParser(
        description="Four Pillars AGI Agent — ARC-AGI Evaluation"
    )
    parser.add_argument(
        "--task", type=str, default="",
        help="Run a single task by name (e.g., 'mirror_h')"
    )
    parser.add_argument(
        "--save-toolkit", type=str, default="",
        help="Save the toolkit to this JSON file after evaluation"
    )
    parser.add_argument(
        "--load-toolkit", type=str, default="",
        help="Load a pre-trained toolkit from this JSON file"
    )
    parser.add_argument(
        "--save-archive", type=str, default="",
        help="Save the archive to this JSON file after evaluation"
    )
    args = parser.parse_args()

    if args.task:
        run_single_task(args.task)
    else:
        run_evaluation(
            save_path=args.save_toolkit,
            load_path=args.load_toolkit,
            archive_path=args.save_archive,
        )


if __name__ == "__main__":
    main()
