"""
TDD tests for pattern inpainting primitives.

These primitives fill in zero-cells (holes) in grids where the surrounding
context has a detectable repeating or structured pattern. This targets
22 near-miss training tasks that are pattern-completion tasks.

Strategy: detect tiling period OR use row/column context to infer missing values.
"""
import unittest
import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestInpaintTiledPattern(unittest.TestCase):
    """Test inpainting grids with a periodic tiling pattern."""

    def test_simple_row_periodic(self):
        """A grid with row period 2, holes replaced with 0."""
        from arc_agent.primitives import inpaint_tiled
        inp = [
            [1, 2, 1, 2, 1, 2],
            [3, 4, 3, 4, 3, 4],
            [1, 2, 0, 0, 1, 2],  # holes at (2,2) and (2,3)
            [3, 4, 3, 4, 3, 4],
        ]
        expected = [
            [1, 2, 1, 2, 1, 2],
            [3, 4, 3, 4, 3, 4],
            [1, 2, 1, 2, 1, 2],
            [3, 4, 3, 4, 3, 4],
        ]
        result = inpaint_tiled(inp)
        self.assertEqual(result, expected)

    def test_column_periodic(self):
        """Periodic in columns but not rows."""
        from arc_agent.primitives import inpaint_tiled
        inp = [
            [1, 1, 1],
            [2, 2, 2],
            [3, 3, 3],
            [1, 0, 1],
            [2, 0, 2],
            [3, 3, 3],
        ]
        expected = [
            [1, 1, 1],
            [2, 2, 2],
            [3, 3, 3],
            [1, 1, 1],
            [2, 2, 2],
            [3, 3, 3],
        ]
        result = inpaint_tiled(inp)
        self.assertEqual(result, expected)

    def test_no_holes_unchanged(self):
        """Grid with no zeros should be returned unchanged."""
        from arc_agent.primitives import inpaint_tiled
        inp = [
            [1, 2, 3],
            [4, 5, 6],
        ]
        result = inpaint_tiled(inp)
        self.assertEqual(result, inp)

    def test_no_period_detected(self):
        """If no period is found, return grid unchanged."""
        from arc_agent.primitives import inpaint_tiled
        inp = [
            [1, 2, 3, 4, 5],
            [6, 0, 0, 9, 10],
            [11, 12, 13, 14, 15],
        ]
        # No obvious period; should return as-is or best effort
        result = inpaint_tiled(inp)
        # At minimum, should not crash and should return a grid of same dims
        self.assertEqual(len(result), len(inp))
        self.assertEqual(len(result[0]), len(inp[0]))

    def test_2d_periodic(self):
        """Grid periodic in both row and column with period 3x2."""
        from arc_agent.primitives import inpaint_tiled
        inp = [
            [1, 2, 1, 2, 1, 2],
            [3, 4, 3, 4, 3, 4],
            [5, 6, 5, 6, 5, 6],
            [1, 2, 1, 2, 0, 0],
            [3, 4, 3, 4, 0, 0],
            [5, 6, 5, 6, 5, 6],
        ]
        expected = [
            [1, 2, 1, 2, 1, 2],
            [3, 4, 3, 4, 3, 4],
            [5, 6, 5, 6, 5, 6],
            [1, 2, 1, 2, 1, 2],
            [3, 4, 3, 4, 3, 4],
            [5, 6, 5, 6, 5, 6],
        ]
        result = inpaint_tiled(inp)
        self.assertEqual(result, expected)


class TestInpaintFromContext(unittest.TestCase):
    """Test inpainting using row/column context (non-periodic patterns)."""

    def test_row_context_simple(self):
        """Each row has a pattern; infer missing cell from row context."""
        from arc_agent.primitives import inpaint_from_context
        # Row 1 should be [1,2,3,4] but has a 0 at position 2
        inp = [
            [1, 2, 3, 4],
            [1, 2, 0, 4],
            [1, 2, 3, 4],
        ]
        expected = [
            [1, 2, 3, 4],
            [1, 2, 3, 4],
            [1, 2, 3, 4],
        ]
        result = inpaint_from_context(inp)
        self.assertEqual(result, expected)

    def test_column_context_simple(self):
        """Each column has a pattern; infer from column."""
        from arc_agent.primitives import inpaint_from_context
        inp = [
            [1, 1, 1],
            [2, 0, 2],
            [3, 3, 3],
        ]
        expected = [
            [1, 1, 1],
            [2, 2, 2],
            [3, 3, 3],
        ]
        result = inpaint_from_context(inp)
        self.assertEqual(result, expected)

    def test_no_zeros_unchanged(self):
        """No zeros means no inpainting needed."""
        from arc_agent.primitives import inpaint_from_context
        inp = [[1, 2], [3, 4]]
        result = inpaint_from_context(inp)
        self.assertEqual(result, inp)


class TestInpaintOnRealTasks(unittest.TestCase):
    """Test inpainting on actual ARC training tasks (pattern completion type).

    Only tests against training data — never eval (data integrity policy).
    """

    def _load_task(self, task_id: str):
        """Load a training task from the ARC dataset."""
        fpath = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "ARC-AGI", "data", "training", f"{task_id}.json"
        )
        if not os.path.exists(fpath):
            self.skipTest(f"ARC dataset not available at {fpath}")
        with open(fpath) as f:
            return json.load(f)

    def _check_inpaint_solves(self, task_id: str):
        """Check if inpainting solves all training examples of a task."""
        from arc_agent.primitives import inpaint_tiled, inpaint_from_context
        task = self._load_task(task_id)

        for ex_i, ex in enumerate(task["train"]):
            inp, expected = ex["input"], ex["output"]
            # Try tiled first, then context
            result = inpaint_tiled(inp)
            if result == expected:
                continue
            result = inpaint_from_context(inp)
            if result == expected:
                continue
            return False, ex_i
        return True, -1

    def test_tiled_29ec7d0e(self):
        """Task 29ec7d0e: periodic tiling with holes (period 5x5)."""
        solved, ex = self._check_inpaint_solves("29ec7d0e")
        self.assertTrue(solved, f"Failed on example {ex}")

    def test_tiled_0dfd9992(self):
        """Task 0dfd9992: periodic tiling with holes (period 6x6)."""
        solved, ex = self._check_inpaint_solves("0dfd9992")
        self.assertTrue(solved, f"Failed on example {ex}")

    def test_tiled_c3f564a4(self):
        """Task c3f564a4: periodic tiling with holes (period 5x5)."""
        solved, ex = self._check_inpaint_solves("c3f564a4")
        self.assertTrue(solved, f"Failed on example {ex}")


class TestInpaintIntegration(unittest.TestCase):
    """Test that inpainting primitives are registered in the toolkit."""

    def test_inpaint_tiled_in_toolkit(self):
        """inpaint_tiled should be available as a toolkit concept."""
        from arc_agent.primitives import build_initial_toolkit
        tk = build_initial_toolkit()
        self.assertIn("inpaint_tiled", tk.concepts)

    def test_inpaint_from_context_in_toolkit(self):
        """inpaint_from_context should be available as a toolkit concept."""
        from arc_agent.primitives import build_initial_toolkit
        tk = build_initial_toolkit()
        self.assertIn("inpaint_from_context", tk.concepts)


if __name__ == "__main__":
    unittest.main()
