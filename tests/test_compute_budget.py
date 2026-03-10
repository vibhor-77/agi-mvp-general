"""Tests for the cell-normalized compute budget system.

Covers:
- Cell-normalized budget calculation: compute_cap / cells
- Budget gating of search phases in the solver
- Budget exceeded flag in solver output
- Contest mode (uncapped: compute_cap=0)
- TaskCache.n_evals counter
- Edge cases (zero cells, zero cap, very small budgets)
"""
import unittest
import random
from arc_agent.solver import FourPillarsSolver
from arc_agent.scorer import TaskCache
from arc_agent.sample_tasks import SAMPLE_TASKS


class TestCellNormalizedBudget(unittest.TestCase):
    """Test the cell-normalization formula: compute_cap / cells.

    The single user-facing control is --compute-cap. The per-task eval
    budget is derived as max(compute_cap // cells, 500). When
    compute_cap=0 (contest mode), evals are effectively unlimited.
    """

    def _effective(self, compute_cap, cells):
        """Replicate the budget calculation from dataset.py / benchmark.py."""
        if compute_cap > 0 and cells > 0:
            return max(compute_cap // cells, 500)
        return 10_000_000  # Effectively unlimited

    def test_small_grid_high_cap(self):
        """Small grids with generous cap get many evals."""
        # 1.5M / 9 = 166K evals
        self.assertEqual(self._effective(1_500_000, 9), 166_666)

    def test_large_grid_fewer_evals(self):
        """Large grids should get proportionally fewer evals."""
        # 1.5M / 900 = 1666 evals
        eff = self._effective(1_500_000, 900)
        self.assertEqual(eff, 1666)

    def test_contest_mode_unlimited(self):
        """Contest mode (compute_cap=0) should give effectively unlimited evals."""
        eff = self._effective(0, 9000)
        self.assertGreater(eff, 1_000_000)

    def test_zero_cells_fallback(self):
        """Zero cells (edge case) should give effectively unlimited evals."""
        eff = self._effective(1_500_000, 0)
        self.assertGreater(eff, 1_000_000)

    def test_budget_proportional_to_grid_size(self):
        """Larger grids should get proportionally fewer evals."""
        eff_small = self._effective(1_500_000, 100)
        eff_large = self._effective(1_500_000, 10_000)
        self.assertEqual(eff_small, 15_000)
        self.assertEqual(eff_large, 500)  # Minimum 500
        self.assertGreater(eff_small, eff_large)

    def test_fast_iteration_cap(self):
        """200K compute-cap should give tight budgets for fast iteration."""
        # 200K / 9 cells = 22K evals
        self.assertEqual(self._effective(200_000, 9), 22222)
        # 200K / 100 cells = 2K evals
        self.assertEqual(self._effective(200_000, 100), 2000)
        # 200K / 900 cells < 500, so minimum 500 applies
        self.assertEqual(self._effective(200_000, 900), 500)

    def test_minimum_500_evals(self):
        """Even with tiny compute-cap, minimum 500 evals is enforced."""
        self.assertEqual(self._effective(100, 10_000), 500)


class TestBudgetGating(unittest.TestCase):
    """Test that the solver skips expensive phases when budget is exceeded."""

    def test_tiny_budget_uses_far_fewer_evals(self):
        """A tiny budget should use far fewer evals than a normal budget."""
        random.seed(42)
        solver1 = FourPillarsSolver(population_size=20, max_generations=5,
                                    max_program_length=3, verbose=False)
        r_normal = solver1.solve_task(
            SAMPLE_TASKS["mirror_h"], "mirror_h", evals_budget=150_000
        )

        random.seed(42)
        solver2 = FourPillarsSolver(population_size=20, max_generations=5,
                                    max_program_length=3, verbose=False)
        r_tiny = solver2.solve_task(
            SAMPLE_TASKS["mirror_h"], "mirror_h", evals_budget=10
        )

        # Tiny budget should use significantly fewer evals
        self.assertLess(r_tiny["n_evals"], r_normal["n_evals"])
        # At least 50% reduction (single prims always run, but pairs/triples skipped)
        self.assertLess(r_tiny["n_evals"], r_normal["n_evals"] * 0.5)

    def test_budget_500_caps_between_phases(self):
        """Budget of 500 should let some phases run but skip later ones."""
        random.seed(42)
        solver = FourPillarsSolver(population_size=20, max_generations=5,
                                   max_program_length=3, verbose=False)
        r = solver.solve_task(
            SAMPLE_TASKS["mirror_h"], "mirror_h", evals_budget=500
        )
        self.assertTrue(r["budget_exceeded"])
        # Should use more evals than budget=10 but fewer than uncapped
        self.assertGreater(r["n_evals"], 100)
        self.assertLess(r["n_evals"], 1500)

    def test_easy_task_solves_with_any_budget(self):
        """mirror_h is solved by single primitive, so any budget works."""
        random.seed(42)
        solver = FourPillarsSolver(population_size=20, max_generations=5,
                                   max_program_length=3, verbose=False)
        r = solver.solve_task(
            SAMPLE_TASKS["mirror_h"], "mirror_h", evals_budget=10
        )
        # Should still solve because single_primitive runs before budget check
        self.assertTrue(r["solved"])


class TestBudgetExceeded(unittest.TestCase):
    """Test that budget_exceeded flag is set correctly."""

    def setUp(self):
        random.seed(42)
        self.solver = FourPillarsSolver(
            population_size=20, max_generations=5,
            max_program_length=3, verbose=False,
        )

    def test_result_contains_budget_fields(self):
        """Solver result should include n_evals and budget_exceeded."""
        result = self.solver.solve_task(
            SAMPLE_TASKS["mirror_h"], "mirror_h", evals_budget=150_000
        )
        self.assertIn("n_evals", result)
        self.assertIn("budget_exceeded", result)
        self.assertIsInstance(result["n_evals"], int)
        self.assertIsInstance(result["budget_exceeded"], bool)

    def test_easy_task_within_budget(self):
        """Easy tasks solved deterministically should stay within budget."""
        result = self.solver.solve_task(
            SAMPLE_TASKS["mirror_h"], "mirror_h", evals_budget=150_000
        )
        self.assertTrue(result["solved"])
        self.assertFalse(result["budget_exceeded"])
        self.assertLess(result["n_evals"], 10_000)

    def test_tiny_budget_triggers_exceeded(self):
        """With a tiny budget, budget_exceeded should be True."""
        result = self.solver.solve_task(
            SAMPLE_TASKS["mirror_h"], "mirror_h", evals_budget=10
        )
        self.assertTrue(result["budget_exceeded"])

    def test_n_evals_deterministic(self):
        """n_evals should be deterministic across runs with same seed."""
        random.seed(42)
        r1 = self.solver.solve_task(
            SAMPLE_TASKS["mirror_h"], "mirror_h", evals_budget=150_000
        )

        random.seed(42)
        solver2 = FourPillarsSolver(
            population_size=20, max_generations=5,
            max_program_length=3, verbose=False,
        )
        r2 = solver2.solve_task(
            SAMPLE_TASKS["mirror_h"], "mirror_h", evals_budget=150_000
        )
        self.assertEqual(r1["n_evals"], r2["n_evals"])


class TestTaskCacheEvalCounter(unittest.TestCase):
    """Test that TaskCache.n_evals correctly counts evaluations."""

    def test_score_program_increments(self):
        """Each score_program call should increment n_evals by 1."""
        task = SAMPLE_TASKS["mirror_h"]
        cache = TaskCache(task)
        self.assertEqual(cache.n_evals, 0)

        from arc_agent.concepts import Program, Concept
        noop = Program([Concept(kind="operator", name="identity",
                                implementation=lambda g: g)])
        cache.score_program(noop)
        self.assertEqual(cache.n_evals, 1)
        cache.score_program(noop)
        self.assertEqual(cache.n_evals, 2)

    def test_score_population_increments_by_count(self):
        """score_population should increment n_evals by population size."""
        task = SAMPLE_TASKS["mirror_h"]
        cache = TaskCache(task)

        from arc_agent.concepts import Program, Concept
        noop = Program([Concept(kind="operator", name="identity",
                                implementation=lambda g: g)])
        pop = [noop] * 10

        cache.score_population(pop)
        self.assertEqual(cache.n_evals, 10)
        cache.score_population(pop)
        self.assertEqual(cache.n_evals, 20)

    def test_initial_count_zero(self):
        """Fresh TaskCache should have n_evals = 0."""
        cache = TaskCache(SAMPLE_TASKS["mirror_h"])
        self.assertEqual(cache.n_evals, 0)


if __name__ == "__main__":
    unittest.main()
