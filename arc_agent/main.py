#!/usr/bin/env python3
"""
Four Pillars AGI Agent — Main Entry Point

Demonstrates Vibhor Jain's 4 Pillars framework on ARC-AGI tasks:
  1. Feedback Loops: Tight scoring against training examples
  2. Approximability: Evolutionary refinement converges toward solutions
  3. Composability: Programs are built from composable concept primitives
  4. Exploration: UCB-based explore/exploit tradeoff

Usage:
    python -m arc_agent.main
"""
import json
import time
import random
from .solver import FourPillarsSolver
from .sample_tasks import SAMPLE_TASKS
from .scorer import validate_on_test


def run_evaluation():
    """Run the full evaluation on sample ARC-AGI tasks."""
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

    # Create the solver
    solver = FourPillarsSolver(
        population_size=60,
        max_generations=30,
        max_program_length=4,
        verbose=True,
    )

    print(f"Initial toolkit: {solver.toolkit.size} concepts")
    print()

    # Solve all sample tasks
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
            # Find the actual program to validate
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
    print(f"  Pillar 3 (Composability): {solver.toolkit.size - 50} new concepts learned")
    print(f"  Pillar 4 (Exploration):   epsilon = {solver.explorer.epsilon:.3f}")

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


if __name__ == "__main__":
    run_evaluation()
