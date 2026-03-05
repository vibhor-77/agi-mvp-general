#!/usr/bin/env python3
"""
Test runner — works with or without pytest.

Usage:
    # With pytest (recommended):
    pip install pytest
    pytest tests/ -v

    # Without pytest (fallback):
    python run_tests.py
"""
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def run_with_pytest():
    """Run tests using pytest."""
    import pytest
    return pytest.main(["-v", "tests/"])


def run_without_pytest():
    """Fallback: run a subset of tests using unittest (no pytest fixtures)."""
    import unittest
    import random
    random.seed(42)

    # We can't easily run pytest-fixture-based tests without pytest,
    # so we run a focused integration test instead
    from arc_agent.solver import FourPillarsSolver
    from arc_agent.sample_tasks import SAMPLE_TASKS
    from arc_agent.scorer import validate_on_test
    from arc_agent.primitives import (
        rotate_90_cw, mirror_horizontal, scale_2x, gravity_down,
        fill_enclosed, outline, identity, crop_to_nonzero,
        build_initial_toolkit,
    )
    from arc_agent.concepts import Concept, Program, Toolkit, Archive
    from arc_agent.synthesizer import ProgramSynthesizer
    from arc_agent.explorer import ExplorationEngine
    from arc_agent.scorer import pixel_accuracy, structural_similarity

    passed = 0
    failed = 0
    errors = []

    def check(name, condition):
        nonlocal passed, failed
        if condition:
            passed += 1
            print(f"  ✓ {name}")
        else:
            failed += 1
            errors.append(name)
            print(f"  ✗ {name}")

    # ── Primitives Tests ──
    print("\n=== Primitives ===")
    check("rotate_90_cw", rotate_90_cw([[1,2],[3,4]]) == [[3,1],[4,2]])
    check("mirror_horizontal", mirror_horizontal([[1,2,3]]) == [[3,2,1]])
    check("scale_2x", scale_2x([[1]]) == [[1,1],[1,1]])
    check("gravity_down", gravity_down([[1,0],[0,2]]) == [[0,0],[1,2]])
    check("fill_enclosed", fill_enclosed([[1,1,1],[1,0,1],[1,1,1]]) == [[1,1,1],[1,1,1],[1,1,1]])
    check("outline", outline([[1,1,1],[1,1,1],[1,1,1]]) == [[1,1,1],[1,0,1],[1,1,1]])
    check("identity_is_copy", identity([[1]])[0][0] == 1)
    check("crop_to_nonzero", crop_to_nonzero([[0,0],[0,5]]) == [[5]])
    check("toolkit_has_concepts", build_initial_toolkit().size > 20)

    # ── Concepts Tests ──
    print("\n=== Concepts ===")
    c = Concept(kind="operator", name="id", implementation=lambda g: g)
    c.apply([[1]])
    check("concept_usage_tracking", c.usage_count == 1)
    c.reinforce(True)
    check("concept_reinforcement", c.success_count == 1)

    p = Program([c, c])
    check("program_chaining", p.execute([[1]]) == [[1]])
    check("program_length", len(p) == 2)

    tk = Toolkit()
    tk.add_concept(c)
    check("toolkit_add", tk.size == 1)

    archive = Archive()
    archive.record_solution("t1", p, 1.0)
    check("archive_records", len(archive.history) == 1)

    # ── Scorer Tests ──
    print("\n=== Scorer ===")
    check("pixel_accuracy_perfect", pixel_accuracy([[1,2],[3,4]], [[1,2],[3,4]]) == 1.0)
    check("pixel_accuracy_partial", pixel_accuracy([[1,2],[3,0]], [[1,2],[3,4]]) == 0.75)
    check("pixel_accuracy_none", pixel_accuracy([[9,9],[9,9]], [[1,2],[3,4]]) == 0.0)
    check("structural_similarity_range",
          0.0 <= structural_similarity([[1]], [[2]]) <= 1.0)

    # ── Synthesizer Tests ──
    print("\n=== Synthesizer ===")
    random.seed(42)
    tk2 = build_initial_toolkit()
    synth = ProgramSynthesizer(tk2, population_size=20, max_program_length=3)
    pop = synth.generate_initial_population()
    check("population_size", len(pop) == 20)
    check("has_single_programs", any(len(p) == 1 for p in pop))
    mutated = synth.mutate(pop[0])
    check("mutation_produces_program", isinstance(mutated, Program))

    task = {"train": [{"input": [[1,2],[3,4]], "output": [[1,2],[3,4]]}]}
    best, history = synth.synthesize(task, max_generations=5)
    check("synthesize_finds_identity", best.fitness >= 0.99)

    # ── Explorer Tests ──
    print("\n=== Explorer ===")
    random.seed(42)
    tk3 = build_initial_toolkit()
    ar = Archive()
    exp = ExplorationEngine(tk3, ar, epsilon=0.3)
    concept = exp.select_concept_ucb()
    check("ucb_returns_concept", isinstance(concept, Concept))
    novels = exp.generate_novel_programs(5)
    check("generates_novel_programs", len(novels) > 0)
    check("epsilon_decay", (exp.decay_epsilon() or True) and exp.epsilon < 0.3)

    c1 = Concept(kind="operator", name="s1", implementation=lambda g: g)
    c2 = Concept(kind="operator", name="s2", implementation=lambda g: g)
    new_c = exp.discover_new_concept(Program([c1, c2]), "test")
    check("discover_multi_step", new_c is not None and new_c.kind == "composed")
    check("discover_skips_single", exp.discover_new_concept(Program([c1]), "t") is None)

    # ── Integration Tests ──
    print("\n=== Integration (Full Pipeline) ===")
    random.seed(42)
    solver = FourPillarsSolver(population_size=40, max_generations=20, verbose=False)

    for task_name in ["mirror_h", "rotate_90", "scale_2x", "gravity_down",
                      "fill_enclosed", "outline_task", "color_swap_1_to_2"]:
        result = solver.solve_task(SAMPLE_TASKS[task_name], task_name)
        check(f"solves_{task_name}", result["solved"])

    result = solver.solve_task(SAMPLE_TASKS["crop_then_mirror"], "crop_then_mirror")
    check("composition_partial", result["score"] > 0.5)
    check("archive_has_history", len(solver.archive.history) >= 2)
    check("result_format", all(k in result for k in
          ["task_id", "solved", "score", "program", "time_seconds"]))

    # Summary
    total = passed + failed
    print(f"\n{'='*50}")
    print(f"Results: {passed}/{total} passed, {failed} failed")
    if errors:
        print(f"Failures: {', '.join(errors)}")
    print(f"{'='*50}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    try:
        sys.exit(run_with_pytest())
    except ImportError:
        print("pytest not installed, running fallback test suite...")
        print("Install pytest for full test coverage: pip install pytest")
        sys.exit(run_without_pytest())
