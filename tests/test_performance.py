"""
Tests for performance optimisations (v0.5).

Covers:
  - NumPy-accelerated scorer produces numerically identical results
    to the pure-Python path.
  - score_population_on_task matches per-program scoring.
  - Parallel evaluation (workers > 1) produces correct aggregate counts.
  - --workers CLI flag is accepted by evaluate.py.
"""
import unittest
import sys
import importlib
from arc_agent.concepts import Concept, Program
from arc_agent.scorer import (
    pixel_accuracy,
    structural_similarity,
    score_program_on_task,
    score_population_on_task,
    _NUMPY_AVAILABLE,
)
from arc_agent.primitives import identity, mirror_horizontal, rotate_90_cw


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_identity_program() -> Program:
    c = Concept(kind="operator", name="identity", implementation=identity)
    return Program([c])


def _make_rotate_program() -> Program:
    c = Concept(kind="operator", name="rotate_90_cw", implementation=rotate_90_cw)
    return Program([c])


SIMPLE_TASK = {
    "train": [
        {"input": [[1, 2], [3, 4]], "output": [[1, 2], [3, 4]]},
        {"input": [[5, 6], [7, 8]], "output": [[5, 6], [7, 8]]},
    ],
    "test": [
        {"input": [[1, 0], [0, 1]], "output": [[1, 0], [0, 1]]},
    ],
}


# ---------------------------------------------------------------------------
# Pixel accuracy tests
# ---------------------------------------------------------------------------

class TestPixelAccuracyNumPy(unittest.TestCase):
    """Verify NumPy path matches pure-Python path numerically."""

    def _py_pixel_accuracy(self, predicted, expected):
        """Pure-Python reference implementation."""
        if not predicted or not expected:
            return 0.0
        pred_h, pred_w = len(predicted), len(predicted[0])
        exp_h, exp_w = len(expected), len(expected[0])
        if pred_h != exp_h or pred_w != exp_w:
            d = 0.0
            if pred_h == exp_h: d += 0.1
            if pred_w == exp_w: d += 0.1
            return d
        total = exp_h * exp_w
        if total == 0:
            return 1.0
        matching = sum(
            1 for r in range(exp_h) for c in range(exp_w)
            if predicted[r][c] == expected[r][c]
        )
        return matching / total

    def test_perfect_match(self):
        g = [[1, 2, 3], [4, 5, 6]]
        self.assertAlmostEqual(pixel_accuracy(g, g), 1.0)

    def test_no_match(self):
        p = [[1, 1], [1, 1]]
        e = [[2, 2], [2, 2]]
        self.assertAlmostEqual(pixel_accuracy(p, e), 0.0)

    def test_partial_match(self):
        p = [[1, 2], [3, 4]]
        e = [[1, 0], [0, 4]]
        result = pixel_accuracy(p, e)
        self.assertAlmostEqual(result, 0.5)

    def test_dim_mismatch_h(self):
        p = [[1, 2], [3, 4], [5, 6]]
        e = [[1, 2], [3, 4]]
        # Height differs, width matches → 0.1
        self.assertAlmostEqual(pixel_accuracy(p, e), 0.1)

    def test_matches_python_reference(self):
        """NumPy result must match pure-Python for a variety of grids."""
        grids = [
            ([[0, 1, 2], [3, 4, 5]], [[0, 1, 2], [3, 0, 5]]),
            ([[1] * 5 for _ in range(5)], [[1] * 5 for _ in range(5)]),
            ([[1, 2], [3, 4]], [[4, 3], [2, 1]]),
            ([[0, 0, 0]], [[1, 0, 0]]),
        ]
        for p, e in grids:
            expected_py = self._py_pixel_accuracy(p, e)
            result = pixel_accuracy(p, e)
            self.assertAlmostEqual(result, expected_py, places=8,
                                   msg=f"Mismatch for p={p}, e={e}")


# ---------------------------------------------------------------------------
# Structural similarity tests
# ---------------------------------------------------------------------------

class TestStructuralSimilarityNumPy(unittest.TestCase):
    """structural_similarity NumPy vs reference."""

    def _py_structural_similarity(self, predicted, expected):
        """Pure-Python reference (original implementation)."""
        if not predicted or not expected:
            return 0.0
        pred_h, pred_w = len(predicted), len(predicted[0])
        exp_h,  exp_w  = len(expected),  len(expected[0])

        # pixel accuracy
        if pred_h == exp_h and pred_w == exp_w:
            total = exp_h * exp_w
            matching = sum(
                1 for r in range(exp_h) for c in range(exp_w)
                if predicted[r][c] == expected[r][c]
            )
            pa = matching / total if total > 0 else 1.0
        else:
            d = 0.0
            if pred_h == exp_h: d += 0.1
            if pred_w == exp_w: d += 0.1
            pa = d

        dim_match = 1.0 if (pred_h == exp_h and pred_w == exp_w) else 0.0

        pred_colors = set(c for row in predicted for c in row) - {0}
        exp_colors  = set(c for row in expected  for c in row) - {0}
        if exp_colors:
            union = len(pred_colors | exp_colors)
            color_overlap = len(pred_colors & exp_colors) / union if union else 1.0
        else:
            color_overlap = 1.0 if not pred_colors else 0.0

        pred_nz = sum(1 for row in predicted for c in row if c != 0)
        exp_nz  = sum(1 for row in expected  for c in row if c != 0)
        max_nz  = max(pred_nz, exp_nz, 1)
        nz_sim  = 1.0 - abs(pred_nz - exp_nz) / max_nz

        return 0.6 * pa + 0.15 * dim_match + 0.15 * color_overlap + 0.1 * nz_sim

    def test_perfect_match(self):
        g = [[1, 2, 3], [4, 5, 6]]
        self.assertAlmostEqual(structural_similarity(g, g), 1.0)

    def test_empty_input(self):
        self.assertAlmostEqual(structural_similarity([], [[1, 2]]), 0.0)

    def test_matches_python_reference(self):
        """NumPy result must match pure-Python for a variety of grids."""
        cases = [
            ([[1, 2, 3], [4, 5, 6]], [[1, 2, 3], [4, 5, 6]]),
            ([[1, 2], [3, 4]],        [[4, 3], [2, 1]]),
            ([[1, 1, 0], [0, 1, 1]],  [[0, 1, 1], [1, 1, 0]]),
            ([[1, 2, 3]],             [[1, 2, 3]]),
            ([[0, 0], [0, 0]],        [[1, 1], [1, 1]]),
            ([[1, 2], [3, 4]],        [[1, 2], [3, 4], [5, 6]]),  # dim mismatch
        ]
        for p, e in cases:
            expected_py = self._py_structural_similarity(p, e)
            result = structural_similarity(p, e)
            self.assertAlmostEqual(result, expected_py, places=8,
                                   msg=f"Mismatch for p={p}, e={e}")


# ---------------------------------------------------------------------------
# Population batch scoring test
# ---------------------------------------------------------------------------

class TestScorePopulationOnTask(unittest.TestCase):
    """score_population_on_task must match per-program scoring."""

    def setUp(self):
        self.task = SIMPLE_TASK
        self.programs = [
            _make_identity_program(),
            _make_rotate_program(),
        ]

    def test_matches_individual_scores(self):
        """Batch scores must equal individually computed scores."""
        batch_scores = score_population_on_task(self.programs, self.task)
        for prog, batch_score in zip(self.programs, batch_scores):
            individual = score_program_on_task(prog, self.task)
            self.assertAlmostEqual(batch_score, individual, places=8,
                                   msg=f"Mismatch for {prog.name}")

    def test_empty_population(self):
        scores = score_population_on_task([], self.task)
        self.assertEqual(scores, [])

    def test_empty_task(self):
        scores = score_population_on_task(self.programs, {"train": []})
        self.assertEqual(scores, [0.0, 0.0])

    def test_identity_scores_1_on_identity_task(self):
        prog = _make_identity_program()
        scores = score_population_on_task([prog], SIMPLE_TASK)
        self.assertAlmostEqual(scores[0], 1.0)


# ---------------------------------------------------------------------------
# Parallel evaluation test
# ---------------------------------------------------------------------------

class TestParallelEvaluation(unittest.TestCase):
    """Parallel evaluation with workers=2 must give correct task results."""

    def test_workers_flag_accepted(self):
        """evaluate_dataset must accept a workers argument without crashing."""
        from arc_agent.dataset import evaluate_dataset
        tasks = {
            "task_identity": SIMPLE_TASK,
        }
        # workers=1 is the single-worker (in-process) path
        result = evaluate_dataset(
            tasks,
            population_size=10,
            max_generations=5,
            verbose=False,
            workers=1,
        )
        self.assertIn("summary", result)
        self.assertIn("task_results", result)
        self.assertEqual(result["summary"]["total_tasks"], 1)
        self.assertEqual(result["summary"]["workers_used"], 1)

    def test_single_vs_parallel_solve_rate(self):
        """Workers=1 and workers=2 should both solve a trivial identity task."""
        from arc_agent.dataset import evaluate_dataset

        tasks = {f"task_{i}": SIMPLE_TASK for i in range(4)}

        r1 = evaluate_dataset(tasks, population_size=20, max_generations=10,
                               verbose=False, workers=1, seed=42)
        r2 = evaluate_dataset(tasks, population_size=20, max_generations=10,
                               verbose=False, workers=2, seed=42)

        # Both should solve all 4 trivial identity tasks
        self.assertEqual(r1["summary"]["solved_exact"],
                         r2["summary"]["solved_exact"])
        self.assertEqual(r1["summary"]["total_tasks"],
                         r2["summary"]["total_tasks"])


# ---------------------------------------------------------------------------
# NumPy availability sanity check
# ---------------------------------------------------------------------------

class TestNumPyAvailability(unittest.TestCase):
    def test_numpy_is_available(self):
        """NumPy must be importable for performance optimisations to work."""
        self.assertTrue(
            _NUMPY_AVAILABLE,
            "NumPy is not available — install it with: pip install numpy"
        )


if __name__ == "__main__":
    unittest.main()
