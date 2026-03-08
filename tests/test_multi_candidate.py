"""Unit tests for multiple candidate submission (top-k diverse predictions).

Tests the full pipeline: solver collects candidates → result carries them →
dataset validates top-k against test → best across candidates is reported.

TDD: these tests were written BEFORE the implementation.
"""
import random
import unittest

from arc_agent.concepts import Concept, Program, Grid
from arc_agent.solver import FourPillarsSolver
from arc_agent.scorer import TaskCache, validate_on_test, validate_candidates_on_test
from arc_agent.sample_tasks import SAMPLE_TASKS


# ── Solver candidate collection ──────────────────────────────────────────

class TestSolverCandidateCollection(unittest.TestCase):
    """Verify solver returns all pixel-perfect candidates in the result."""

    def setUp(self):
        random.seed(42)
        self.solver = FourPillarsSolver(
            population_size=40, max_generations=15, verbose=False
        )

    def test_result_has_candidates_field(self):
        """Result dict must include a 'candidates' list."""
        result = self.solver.solve_task(SAMPLE_TASKS["mirror_h"], "mirror_h")
        self.assertIn("candidates", result)
        self.assertIsInstance(result["candidates"], list)

    def test_solved_task_has_at_least_one_candidate(self):
        """A solved task must have >= 1 candidate."""
        result = self.solver.solve_task(SAMPLE_TASKS["mirror_h"], "mirror_h")
        if result["solved"]:
            self.assertGreaterEqual(len(result["candidates"]), 1)

    def test_candidates_are_program_method_tuples(self):
        """Each candidate is a dict with 'program' and 'method' keys."""
        result = self.solver.solve_task(SAMPLE_TASKS["mirror_h"], "mirror_h")
        if result["candidates"]:
            for cand in result["candidates"]:
                self.assertIn("program", cand)
                self.assertIn("method", cand)
                self.assertIn("steps", cand)

    def test_n_candidates_matches_candidates_length(self):
        """n_candidates field should match len(candidates)."""
        result = self.solver.solve_task(SAMPLE_TASKS["mirror_h"], "mirror_h")
        self.assertEqual(result["n_candidates"], len(result["candidates"]))

    def test_winner_is_shortest_candidate(self):
        """The selected winner should be the shortest (MDL) among candidates."""
        result = self.solver.solve_task(SAMPLE_TASKS["mirror_h"], "mirror_h")
        if len(result["candidates"]) > 1:
            winner_len = result["program_length"]
            for cand in result["candidates"]:
                self.assertGreaterEqual(len(cand["steps"]), winner_len)


# ── Multi-candidate test validation ──────────────────────────────────────

class TestValidateCandidatesOnTest(unittest.TestCase):
    """Test the new validate_candidates_on_test function."""

    def _make_program(self, fn):
        """Helper to create a single-step program from a function."""
        concept = Concept(kind="operator", name="test_fn",
                          implementation=fn)
        return Program([concept])

    def test_single_candidate_matches_validate_on_test(self):
        """With 1 candidate, result should match validate_on_test."""
        task = SAMPLE_TASKS["mirror_h"]
        # Create a perfect program (mirror_h)
        from arc_agent.primitives import build_initial_toolkit
        tk = build_initial_toolkit()
        prog = Program([tk.concepts["mirror_h"]])

        single_result = validate_on_test(prog, task)
        multi_result = validate_candidates_on_test([prog], task)

        self.assertEqual(single_result[0], multi_result[0])  # passed
        self.assertAlmostEqual(single_result[1], multi_result[1])  # score

    def test_best_candidate_wins(self):
        """If one candidate passes test and another doesn't, best wins."""
        task = SAMPLE_TASKS["mirror_h"]
        from arc_agent.primitives import build_initial_toolkit
        tk = build_initial_toolkit()

        good_prog = Program([tk.concepts["mirror_h"]])   # correct
        bad_prog = Program([tk.concepts["rotate_90_cw"]])    # wrong

        passed, score = validate_candidates_on_test([bad_prog, good_prog], task)
        self.assertTrue(passed)
        self.assertAlmostEqual(score, 1.0)

    def test_empty_candidates_returns_false(self):
        """Empty candidate list should return (False, 0.0)."""
        task = SAMPLE_TASKS["mirror_h"]
        passed, score = validate_candidates_on_test([], task)
        self.assertFalse(passed)
        self.assertEqual(score, 0.0)

    def test_top_k_limits_candidates(self):
        """Only the first top_k candidates should be tested."""
        task = SAMPLE_TASKS["mirror_h"]
        from arc_agent.primitives import build_initial_toolkit
        tk = build_initial_toolkit()

        good_prog = Program([tk.concepts["mirror_h"]])
        bad_prog = Program([tk.concepts["rotate_90_cw"]])

        # good_prog is at index 1, but top_k=1 only tests index 0 (bad_prog)
        passed, score = validate_candidates_on_test(
            [bad_prog, good_prog], task, top_k=1
        )
        self.assertFalse(passed)

    def test_returns_best_score_across_candidates(self):
        """Score should be the maximum across all tested candidates."""
        task = SAMPLE_TASKS["mirror_h"]
        from arc_agent.primitives import build_initial_toolkit
        tk = build_initial_toolkit()

        prog1 = Program([tk.concepts["rotate_90_cw"]])   # wrong
        prog2 = Program([tk.concepts["mirror_h"]])     # correct

        _, score = validate_candidates_on_test([prog1, prog2], task)
        self.assertAlmostEqual(score, 1.0)


# ── Candidate deduplication ──────────────────────────────────────────────

class TestCandidateDedup(unittest.TestCase):
    """Test that duplicate candidates are removed."""

    def test_solver_deduplicates_candidates(self):
        """Same program from different methods shouldn't appear twice."""
        random.seed(42)
        solver = FourPillarsSolver(
            population_size=40, max_generations=15, verbose=False
        )
        result = solver.solve_task(SAMPLE_TASKS["mirror_h"], "mirror_h")

        if result["candidates"]:
            # Check no duplicate step sequences
            step_seqs = [tuple(c["steps"]) for c in result["candidates"]]
            self.assertEqual(len(step_seqs), len(set(step_seqs)),
                             f"Duplicate candidates found: {step_seqs}")


# ── Integration: multi-candidate in collect_result ────────────────────────

class TestCollectResultMultiCandidate(unittest.TestCase):
    """Test that _collect_result passes candidates through."""

    def test_collect_result_includes_candidates(self):
        """_collect_result should propagate candidates from solver result."""
        from arc_agent.dataset import _collect_result
        random.seed(42)
        solver = FourPillarsSolver(
            population_size=40, max_generations=15, verbose=False
        )
        task = SAMPLE_TASKS["mirror_h"]
        result = solver.solve_task(task, "mirror_h")

        collected = _collect_result(solver, result, "mirror_h", task, seed=42)
        # After our changes, collected should have test validation
        # using all candidates (not just the first program)
        self.assertIn("test_passed", collected)

    def test_multi_candidate_test_improves_tc(self):
        """With multiple candidates, test_passed should be at least as good
        as single-candidate test_passed."""
        # This tests the principle: more candidates → higher test confirmation rate
        random.seed(42)
        solver = FourPillarsSolver(
            population_size=40, max_generations=15, verbose=False
        )
        task = SAMPLE_TASKS["mirror_h"]
        result = solver.solve_task(task, "mirror_h")

        if result["solved"] and result["n_candidates"] > 0:
            from arc_agent.dataset import _collect_result
            collected = _collect_result(solver, result, "mirror_h", task, seed=42)
            # If we solved it on train, at least one candidate should pass test
            # (for this simple task)
            self.assertTrue(collected["test_passed"])


if __name__ == "__main__":
    unittest.main()
