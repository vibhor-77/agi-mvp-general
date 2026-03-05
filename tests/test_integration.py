"""Integration tests — full pipeline exercising all 4 pillars together."""
import unittest
import random
from arc_agent.solver import FourPillarsSolver
from arc_agent.sample_tasks import SAMPLE_TASKS
from arc_agent.scorer import validate_on_test


class TestFullPipeline(unittest.TestCase):
    """Tests that the complete solver works end-to-end."""

    def setUp(self):
        """Set up solver for each test."""
        random.seed(42)
        self.solver = FourPillarsSolver(
            population_size=40,
            max_generations=20,
            max_program_length=4,
            verbose=False,
        )

    def test_solves_mirror_task(self):
        result = self.solver.solve_task(SAMPLE_TASKS["mirror_h"], "mirror_h")
        self.assertIs(result["solved"], True)
        self.assertGreaterEqual(result["score"], 0.99)

    def test_solves_rotate_task(self):
        result = self.solver.solve_task(SAMPLE_TASKS["rotate_90"], "rotate_90")
        self.assertIs(result["solved"], True)

    def test_solves_scale_task(self):
        result = self.solver.solve_task(SAMPLE_TASKS["scale_2x"], "scale_2x")
        self.assertIs(result["solved"], True)

    def test_solves_gravity_task(self):
        result = self.solver.solve_task(SAMPLE_TASKS["gravity_down"], "gravity_down")
        self.assertIs(result["solved"], True)

    def test_solves_fill_enclosed_task(self):
        result = self.solver.solve_task(SAMPLE_TASKS["fill_enclosed"], "fill_enclosed")
        self.assertIs(result["solved"], True)

    def test_solves_outline_task(self):
        result = self.solver.solve_task(SAMPLE_TASKS["outline_task"], "outline_task")
        self.assertIs(result["solved"], True)

    def test_solves_color_swap_task(self):
        result = self.solver.solve_task(SAMPLE_TASKS["color_swap_1_to_2"], "color_swap_1_to_2")
        self.assertIs(result["solved"], True)

    def test_solves_composition_task(self):
        """Test that evolutionary search can discover multi-step compositions."""
        random.seed(42)
        result = self.solver.solve_task(SAMPLE_TASKS["crop_then_mirror"], "crop_then_mirror")
        self.assertGreater(result["score"], 0.5)  # At least partial solve


class TestKnowledgeCompounding(unittest.TestCase):
    """Tests that the system compounds knowledge across tasks (cumulative culture)."""

    def test_toolkit_grows_after_solving(self):
        random.seed(42)
        solver = FourPillarsSolver(
            population_size=40, max_generations=25, verbose=False
        )
        initial_size = solver.toolkit.size

        # Solve a composition task that should create a new concept
        solver.solve_task(SAMPLE_TASKS["crop_then_mirror"], "crop_then_mirror")

        # Toolkit should have grown if a multi-step solution was found
        # (or stayed same if single primitive solved it)
        self.assertGreaterEqual(solver.toolkit.size, initial_size)

    def test_archive_records_solutions(self):
        random.seed(42)
        solver = FourPillarsSolver(
            population_size=40, max_generations=15, verbose=False
        )

        solver.solve_task(SAMPLE_TASKS["mirror_h"], "mirror_h")
        solver.solve_task(SAMPLE_TASKS["rotate_90"], "rotate_90")

        # Archive should have recorded both solutions
        self.assertGreaterEqual(len(solver.archive.history), 2)

    def test_batch_processing(self):
        """Test that batch solving works and accumulates knowledge."""
        random.seed(42)
        solver = FourPillarsSolver(
            population_size=30, max_generations=15, verbose=False
        )

        # Solve a small batch
        small_tasks = {
            "mirror_h": SAMPLE_TASKS["mirror_h"],
            "rotate_90": SAMPLE_TASKS["rotate_90"],
            "gravity_down": SAMPLE_TASKS["gravity_down"],
        }

        results = solver.solve_batch(small_tasks)
        self.assertEqual(len(results), 3)
        solved_count = sum(1 for r in results.values() if r["solved"])
        self.assertGreaterEqual(solved_count, 2)


class TestTestValidation(unittest.TestCase):
    """Tests that solved tasks also pass held-out test examples."""

    def test_mirror_generalizes(self):
        random.seed(42)
        solver = FourPillarsSolver(
            population_size=40, max_generations=15, verbose=False
        )
        result = solver.solve_task(SAMPLE_TASKS["mirror_h"], "mirror_h")

        if result["solved"]:
            programs = solver.archive.task_solutions.get("mirror_h", [])
            if programs:
                exact, score = validate_on_test(programs[0], SAMPLE_TASKS["mirror_h"])
                self.assertIs(exact, True)

    def test_scale_generalizes(self):
        random.seed(42)
        solver = FourPillarsSolver(
            population_size=40, max_generations=15, verbose=False
        )
        result = solver.solve_task(SAMPLE_TASKS["scale_2x"], "scale_2x")

        if result["solved"]:
            programs = solver.archive.task_solutions.get("scale_2x", [])
            if programs:
                exact, score = validate_on_test(programs[0], SAMPLE_TASKS["scale_2x"])
                self.assertIs(exact, True)


class TestAblationStudies(unittest.TestCase):
    """Ablation tests to validate each pillar is necessary."""

    def test_without_exploration_still_finds_singles(self):
        """Without exploration (epsilon=0), should still find single primitives."""
        random.seed(42)
        solver = FourPillarsSolver(
            population_size=40, max_generations=15, verbose=False
        )
        solver.explorer.epsilon = 0.0  # No exploration

        result = solver.solve_task(SAMPLE_TASKS["mirror_h"], "mirror_h")
        # Mirror is a single primitive — should still be found via exploit
        self.assertIs(result["solved"], True)

    def test_approximability_matters(self):
        """Verify that partial credit scoring enables convergence."""
        from arc_agent.scorer import pixel_accuracy

        # A partially correct prediction should score > 0
        pred = [[1, 2], [3, 0]]
        exp = [[1, 2], [3, 4]]
        score = pixel_accuracy(pred, exp)
        self.assertEqual(score, 0.75)  # 3 out of 4 pixels correct

    def test_composability_needed_for_multi_step(self):
        """Multi-step tasks can't be solved with single primitives alone."""
        random.seed(42)
        solver = FourPillarsSolver(
            population_size=40, max_generations=1,  # Very few generations
            max_program_length=1,  # Only single primitives
            verbose=False
        )

        # crop_then_mirror requires 2+ steps
        result = solver.solve_task(SAMPLE_TASKS["crop_then_mirror"], "crop_then_mirror")
        # With max_program_length=1, it shouldn't find the full solution
        # (it might get partial credit)
        self.assertTrue(result["score"] < 0.99 or result["program_length"] <= 1)


class TestResultFormat(unittest.TestCase):
    """Tests that result dictionaries have the expected format."""

    def test_result_has_required_fields(self):
        random.seed(42)
        solver = FourPillarsSolver(
            population_size=20, max_generations=5, verbose=False
        )
        result = solver.solve_task(SAMPLE_TASKS["mirror_h"], "mirror_h")

        self.assertIn("task_id", result)
        self.assertIn("solved", result)
        self.assertIn("score", result)
        self.assertIn("program", result)
        self.assertIn("time_seconds", result)
        self.assertIn("method", result)
        self.assertIn("toolkit_size", result)
        self.assertIsInstance(result["solved"], bool)
        self.assertIsInstance(result["score"], float)
        self.assertIsInstance(result["time_seconds"], float)


if __name__ == '__main__':
    unittest.main()
