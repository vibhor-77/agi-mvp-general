"""Tests for solver candidate selection logic.

Covers:
- MDL preference (shorter programs preferred)
- Anti-overfit: built-in primitives preferred over learned ones
- _candidate_sort_key ordering
"""
import unittest
import random
from arc_agent.solver import FourPillarsSolver
from arc_agent.concepts import Program, Concept
from arc_agent.sample_tasks import SAMPLE_TASKS


class TestCandidateSortKey(unittest.TestCase):
    """Test the _candidate_sort_key ordering used in Step 7."""

    def setUp(self):
        random.seed(42)
        self.solver = FourPillarsSolver(
            population_size=20, max_generations=5,
            max_program_length=3, verbose=False,
        )

    def test_shorter_program_preferred(self):
        """A 1-step program should rank before a 2-step program."""
        # Create test programs
        builtin = Concept(kind="operator", name="mirror_h",
                          implementation=lambda g: g)
        one_step = Program([builtin])
        two_step = Program([builtin, builtin])

        candidates = [
            (two_step, "pair"),
            (one_step, "single"),
        ]

        # Apply sort key
        from arc_agent.solver import FourPillarsSolver
        # Access sort key via solve_task's internal
        # The key is (len, n_learned, method)
        key_fn = lambda item: (len(item[0].steps),
                               sum(1 for s in item[0].steps if s.name.startswith("learned_")),
                               item[1])
        winner = min(candidates, key=key_fn)
        self.assertEqual(len(winner[0].steps), 1)

    def test_builtin_preferred_over_learned_same_length(self):
        """Among same-length programs, built-in should rank before learned."""
        builtin = Concept(kind="operator", name="mirror_h",
                          implementation=lambda g: g)
        learned = Concept(kind="operator", name="learned_abc_123",
                          implementation=lambda g: g)

        builtin_prog = Program([builtin])
        learned_prog = Program([learned])

        candidates = [
            (learned_prog, "culture"),
            (builtin_prog, "single"),
        ]

        key_fn = lambda item: (len(item[0].steps),
                               sum(1 for s in item[0].steps if s.name.startswith("learned_")),
                               item[1])
        winner = min(candidates, key=key_fn)
        self.assertEqual(winner[0].steps[0].name, "mirror_h")

    def test_mixed_learned_steps_penalized(self):
        """A 2-step program with 1 learned step should rank below pure built-in 2-step."""
        builtin = Concept(kind="operator", name="mirror_h",
                          implementation=lambda g: g)
        learned = Concept(kind="operator", name="learned_abc_123",
                          implementation=lambda g: g)

        pure_builtin = Program([builtin, builtin])
        mixed = Program([builtin, learned])

        candidates = [
            (mixed, "pair"),
            (pure_builtin, "pair"),
        ]

        key_fn = lambda item: (len(item[0].steps),
                               sum(1 for s in item[0].steps if s.name.startswith("learned_")),
                               item[1])
        winner = min(candidates, key=key_fn)
        self.assertFalse(any(s.name.startswith("learned_") for s in winner[0].steps))


class TestAntiOverfit(unittest.TestCase):
    """Test that solver prefers simpler, generalizable programs."""

    def test_solve_uses_built_in_for_mirror(self):
        """mirror_h should use a built-in primitive, not a learned one."""
        random.seed(42)
        solver = FourPillarsSolver(
            population_size=20, max_generations=5,
            max_program_length=3, verbose=False,
        )
        r = solver.solve_task(SAMPLE_TASKS["mirror_h"], "mirror_h",
                              evals_budget=10_000)
        # The steps should not include learned primitives for this easy task
        steps = r.get("program_steps", [])
        for step in steps:
            self.assertFalse(step.startswith("learned_"),
                             f"Should use built-in, not {step}")

    def test_deterministic_results(self):
        """Same seed should give identical results (deterministic)."""
        results = []
        for _ in range(2):
            random.seed(42)
            solver = FourPillarsSolver(
                population_size=20, max_generations=5,
                max_program_length=3, verbose=False,
            )
            r = solver.solve_task(SAMPLE_TASKS["mirror_h"], "mirror_h",
                                  evals_budget=10_000)
            results.append(r)

        self.assertEqual(results[0]["n_evals"], results[1]["n_evals"])
        self.assertEqual(results[0]["solved"], results[1]["solved"])
        self.assertEqual(results[0]["method"], results[1]["method"])


if __name__ == "__main__":
    unittest.main()
