"""
Tests for performance-sensitive paths (v0.5+).

Covers:
  - Scorer correctness (pixel_accuracy, structural_similarity).
  - score_population_on_task matches per-program scoring.
  - Parallel evaluation (workers=2) gives the same aggregate as workers=1.
  - CPU topology detection returns a sane value.
  - --workers CLI flag is accepted.
"""
import unittest
from arc_agent.concepts import Concept, Program
from arc_agent.scorer import (
    pixel_accuracy,
    structural_similarity,
    score_program_on_task,
    score_population_on_task,
)
from arc_agent.primitives import identity, mirror_horizontal, rotate_90_cw
from arc_agent.cpu_utils import default_workers, describe_cpu


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _prog(fn, name: str) -> Program:
    return Program([Concept(kind="operator", name=name, implementation=fn)])


IDENTITY_TASK = {
    "train": [
        {"input": [[1, 2], [3, 4]], "output": [[1, 2], [3, 4]]},
        {"input": [[5, 6], [7, 8]], "output": [[5, 6], [7, 8]]},
    ],
    "test": [
        {"input": [[1, 0], [0, 1]], "output": [[1, 0], [0, 1]]},
    ],
}


# ---------------------------------------------------------------------------
# pixel_accuracy
# ---------------------------------------------------------------------------

class TestPixelAccuracy(unittest.TestCase):

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
        self.assertAlmostEqual(pixel_accuracy(p, e), 0.5)

    def test_dim_mismatch_h(self):
        # Height differs, width matches → 0.1
        p = [[1, 2], [3, 4], [5, 6]]
        e = [[1, 2], [3, 4]]
        self.assertAlmostEqual(pixel_accuracy(p, e), 0.1)

    def test_dim_mismatch_both(self):
        # Both dimensions differ → 0.0
        p = [[1, 2, 3]]
        e = [[1, 2], [3, 4]]
        self.assertAlmostEqual(pixel_accuracy(p, e), 0.0)

    def test_empty_grids(self):
        self.assertAlmostEqual(pixel_accuracy([], [[1]]), 0.0)
        self.assertAlmostEqual(pixel_accuracy([[1]], []), 0.0)

    def test_single_cell_match(self):
        self.assertAlmostEqual(pixel_accuracy([[5]], [[5]]), 1.0)

    def test_single_cell_mismatch(self):
        self.assertAlmostEqual(pixel_accuracy([[5]], [[3]]), 0.0)

    def test_larger_grid(self):
        g = [[i for i in range(10)] for _ in range(10)]
        self.assertAlmostEqual(pixel_accuracy(g, g), 1.0)


# ---------------------------------------------------------------------------
# structural_similarity
# ---------------------------------------------------------------------------

class TestStructuralSimilarity(unittest.TestCase):

    def test_perfect_match(self):
        g = [[1, 2, 3], [4, 5, 6]]
        self.assertAlmostEqual(structural_similarity(g, g), 1.0)

    def test_empty_input(self):
        self.assertAlmostEqual(structural_similarity([], [[1, 2]]), 0.0)

    def test_all_zeros_vs_all_zeros(self):
        # Identical all-zero grids: pixel=1, dim=1, color=1 (no non-bg), nz=1
        g = [[0, 0], [0, 0]]
        self.assertAlmostEqual(structural_similarity(g, g), 1.0)

    def test_color_mismatch_penalty(self):
        # Same shape, different colors → low score but not zero
        p = [[1, 1], [1, 1]]
        e = [[2, 2], [2, 2]]
        score = structural_similarity(p, e)
        self.assertGreater(score, 0.0)
        self.assertLess(score, 0.5)

    def test_dim_mismatch(self):
        p = [[1, 2], [3, 4]]
        e = [[1, 2], [3, 4], [5, 6]]
        score = structural_similarity(p, e)
        # Should be positive (color overlap, nz sim) but < 0.5
        self.assertGreater(score, 0.0)
        self.assertLess(score, 0.5)

    def test_score_range(self):
        import random
        rng = random.Random(0)
        for _ in range(20):
            h, w = rng.randint(1, 8), rng.randint(1, 8)
            p = [[rng.randint(0, 9) for _ in range(w)] for _ in range(h)]
            e = [[rng.randint(0, 9) for _ in range(w)] for _ in range(h)]
            score = structural_similarity(p, e)
            self.assertGreaterEqual(score, 0.0, f"negative score for p={p}, e={e}")
            self.assertLessEqual(score, 1.0 + 1e-9, f"score > 1 for p={p}, e={e}")


# ---------------------------------------------------------------------------
# score_population_on_task
# ---------------------------------------------------------------------------

class TestScorePopulationOnTask(unittest.TestCase):

    def setUp(self):
        self.task  = IDENTITY_TASK
        self.progs = [
            _prog(identity,           "identity"),
            _prog(rotate_90_cw,       "rotate_90_cw"),
            _prog(mirror_horizontal,  "mirror_horizontal"),
        ]

    def test_matches_individual_scores(self):
        """Batch scores must equal per-program scores exactly."""
        batch = score_population_on_task(self.progs, self.task)
        for prog, b_score in zip(self.progs, batch):
            individual = score_program_on_task(prog, self.task)
            self.assertAlmostEqual(b_score, individual, places=10,
                                   msg=f"Mismatch for {prog.name}")

    def test_empty_population(self):
        self.assertEqual(score_population_on_task([], self.task), [])

    def test_empty_task_train(self):
        scores = score_population_on_task(self.progs, {"train": []})
        self.assertEqual(scores, [0.0, 0.0, 0.0])

    def test_identity_scores_one(self):
        p = _prog(identity, "identity")
        self.assertAlmostEqual(score_population_on_task([p], IDENTITY_TASK)[0], 1.0)


# ---------------------------------------------------------------------------
# Parallel evaluation
# ---------------------------------------------------------------------------

class TestParallelEvaluation(unittest.TestCase):
    """workers=1 and workers=2 must give identical aggregate counts
    on deterministic tasks."""

    def _run(self, n_workers: int) -> dict:
        from arc_agent.dataset import evaluate_dataset
        tasks = {f"t{i}": IDENTITY_TASK for i in range(4)}
        return evaluate_dataset(
            tasks, population_size=20, max_generations=10,
            verbose=False, workers=n_workers, seed=42,
        )

    def test_workers_flag_accepted(self):
        r = self._run(1)
        self.assertIn("summary", r)
        self.assertIn("task_results", r)
        self.assertEqual(r["summary"]["total_tasks"], 4)
        self.assertEqual(r["summary"]["workers_used"], 1)

    def test_single_vs_parallel_solve_count(self):
        r1 = self._run(1)
        r2 = self._run(2)
        self.assertEqual(r1["summary"]["total_tasks"], r2["summary"]["total_tasks"])
        self.assertEqual(r1["summary"]["solved_exact"], r2["summary"]["solved_exact"])

    def test_results_keys_sorted(self):
        """task_results keys must be present for all task IDs."""
        from arc_agent.dataset import evaluate_dataset
        tasks = {f"t{i:02d}": IDENTITY_TASK for i in range(6)}
        r = evaluate_dataset(tasks, population_size=10, max_generations=5,
                             verbose=False, workers=1, seed=0)
        self.assertEqual(sorted(r["task_results"].keys()), sorted(tasks.keys()))

    def test_seed_in_summary(self):
        r = self._run(1)
        self.assertEqual(r["summary"]["seed"], 42)


# ---------------------------------------------------------------------------
# CPU topology detection
# ---------------------------------------------------------------------------

class TestCpuUtils(unittest.TestCase):

    def test_default_workers_positive(self):
        n = default_workers()
        self.assertGreater(n, 0)

    def test_default_workers_reasonable(self):
        import os
        n = default_workers()
        # Should never exceed total logical CPU count
        self.assertLessEqual(n, os.cpu_count() or 1)

    def test_describe_cpu_is_string(self):
        desc = describe_cpu()
        self.assertIsInstance(desc, str)
        self.assertGreater(len(desc), 0)


if __name__ == "__main__":
    unittest.main()
