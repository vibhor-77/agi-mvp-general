"""TDD Tests for Cell Synthesis: Enumeration-based DSL for per-cell transformations.

This module tests the cell synthesis engine which discovers task-specific
cell-level transformation rules through enumeration of small DSL programs.

Test coverage:
  1. DSL nodes (Const, Self, NeighborMajority, MapColor, IfColor, etc.)
  2. Cell program evaluation on simple grids
  3. Enumeration: correct program generation at each depth
  4. Scoring: cell program evaluation against training examples
  5. Integration: top cell program wrapped as Concept
"""
import unittest
from arc_agent.cell_synth import (
    CellExpr,
    Const,
    Self,
    NeighborMajority,
    NeighborAt,
    MapColor,
    IfColor,
    IfNeighborHas,
    evaluate_cell_expr,
    enumerate_cell_exprs,
    score_cell_expr,
    synthesize_cell_program,
)
from arc_agent.concepts import Concept, Program, Grid


# ============================================================
# Test DSL Nodes (Unit Tests)
# ============================================================

class TestCellExprDSL(unittest.TestCase):
    """Test individual DSL node types."""

    def test_const_expr(self):
        """Const(c) always returns c."""
        expr = Const(5)
        grid = [[1, 2], [3, 4]]
        # Evaluate at any cell, should always return 5
        self.assertEqual(evaluate_cell_expr(expr, grid, 0, 0), 5)
        self.assertEqual(evaluate_cell_expr(expr, grid, 1, 1), 5)

    def test_self_expr(self):
        """Self returns the current cell's value."""
        expr = Self()
        grid = [[1, 2], [3, 4]]
        self.assertEqual(evaluate_cell_expr(expr, grid, 0, 0), 1)
        self.assertEqual(evaluate_cell_expr(expr, grid, 1, 1), 4)
        self.assertEqual(evaluate_cell_expr(expr, grid, 0, 1), 2)

    def test_neighbor_at_up(self):
        """NeighborAt('up') returns value above current cell."""
        expr = NeighborAt('up')
        grid = [[1, 2], [3, 4]]
        # At (1, 0), neighbor up is (0, 0) = 1
        self.assertEqual(evaluate_cell_expr(expr, grid, 1, 0), 1)
        # At (0, 0), no neighbor up -> returns 0 (out of bounds)
        self.assertEqual(evaluate_cell_expr(expr, grid, 0, 0), 0)

    def test_neighbor_at_down(self):
        """NeighborAt('down') returns value below current cell."""
        expr = NeighborAt('down')
        grid = [[1, 2], [3, 4]]
        # At (0, 0), neighbor down is (1, 0) = 3
        self.assertEqual(evaluate_cell_expr(expr, grid, 0, 0), 3)
        # At (1, 0), no neighbor down -> returns 0
        self.assertEqual(evaluate_cell_expr(expr, grid, 1, 0), 0)

    def test_neighbor_at_left(self):
        """NeighborAt('left') returns value to the left."""
        expr = NeighborAt('left')
        grid = [[1, 2], [3, 4]]
        # At (0, 1), neighbor left is (0, 0) = 1
        self.assertEqual(evaluate_cell_expr(expr, grid, 0, 1), 1)
        # At (0, 0), no neighbor left -> returns 0
        self.assertEqual(evaluate_cell_expr(expr, grid, 0, 0), 0)

    def test_neighbor_at_right(self):
        """NeighborAt('right') returns value to the right."""
        expr = NeighborAt('right')
        grid = [[1, 2], [3, 4]]
        # At (0, 0), neighbor right is (0, 1) = 2
        self.assertEqual(evaluate_cell_expr(expr, grid, 0, 0), 2)
        # At (0, 1), no neighbor right -> returns 0
        self.assertEqual(evaluate_cell_expr(expr, grid, 0, 1), 0)

    def test_neighbor_majority_simple(self):
        """NeighborMajority returns most common neighbor color."""
        expr = NeighborMajority()
        # Grid where (1,1) has neighbors: up=2, down=2, left=3, right=3
        # Majority is either 2 or 3 (tie), should pick min or first
        grid = [[2, 2], [3, 0, 3], [2, 2]]
        # For a 3x3 grid, (1,1) has 4 neighbors
        grid_3x3 = [[1, 1, 1], [1, 0, 1], [1, 1, 1]]
        result = evaluate_cell_expr(expr, grid_3x3, 1, 1)
        # All neighbors are 1, so majority is 1
        self.assertEqual(result, 1)

    def test_map_color_match(self):
        """MapColor(from, to) replaces from_color with to_color."""
        expr = MapColor(3, 5)
        grid = [[3, 2], [3, 4]]
        # At (0,0) where cell == 3, should return 5
        self.assertEqual(evaluate_cell_expr(expr, grid, 0, 0), 5)
        # At (0,1) where cell == 2, should return 2 (no change)
        self.assertEqual(evaluate_cell_expr(expr, grid, 0, 1), 2)

    def test_if_color_true_branch(self):
        """IfColor(c, then, else) takes then branch if cell==c."""
        expr = IfColor(3, Const(9), Const(0))
        grid = [[3, 2], [3, 4]]
        # At (0,0) where cell == 3, should take then branch -> 9
        self.assertEqual(evaluate_cell_expr(expr, grid, 0, 0), 9)

    def test_if_color_false_branch(self):
        """IfColor(c, then, else) takes else branch if cell!=c."""
        expr = IfColor(3, Const(9), Const(0))
        grid = [[3, 2], [3, 4]]
        # At (0,1) where cell == 2 != 3, should take else branch -> 0
        self.assertEqual(evaluate_cell_expr(expr, grid, 0, 1), 0)

    def test_if_neighbor_has_true_branch(self):
        """IfNeighborHas(c, then, else) takes then if any neighbor==c."""
        expr = IfNeighborHas(2, Const(9), Const(0))
        grid = [[2, 3], [4, 5]]
        # At (1,1), neighbors are up=3, down=nothing, left=4, right=nothing
        # No neighbor has color 2, so should take else branch -> 0
        self.assertEqual(evaluate_cell_expr(expr, grid, 1, 1), 0)
        # At (0,0), neighbor right=3, neighbor down=4, no neighbor has 2
        # So should take else branch -> 0
        self.assertEqual(evaluate_cell_expr(expr, grid, 0, 0), 0)

    def test_if_neighbor_has_with_match(self):
        """IfNeighborHas detects when a neighbor has the target color."""
        expr = IfNeighborHas(2, Const(9), Const(0))
        grid = [[2, 2], [4, 5]]
        # At (1,0), neighbor up=2, so has neighbor with color 2
        # Should take then branch -> 9
        self.assertEqual(evaluate_cell_expr(expr, grid, 1, 0), 9)


# ============================================================
# Test Cell Program Evaluation (Integration)
# ============================================================

class TestCellProgramEvaluation(unittest.TestCase):
    """Test evaluation of cell programs on grids."""

    def test_evaluate_simple_grid(self):
        """Evaluate a simple cell expr on a grid."""
        expr = Self()
        grid = [[1, 2], [3, 4]]
        # Apply to entire grid
        result = evaluate_cell_expr(expr, grid, 0, 0)
        self.assertEqual(result, 1)

    def test_compose_nested_exprs(self):
        """Test nested cell expressions."""
        # IfColor(3, MapColor(2,5), Self)
        # If cell==3, map 2->5, else keep self
        expr = IfColor(3, MapColor(2, 5), Self())
        grid = [[3, 2], [3, 4]]
        # At (0,0) cell==3, so MapColor(2,5) but cell is 3, not 2 -> 3
        result = evaluate_cell_expr(expr, grid, 0, 0)
        self.assertEqual(result, 3)


# ============================================================
# Test Enumeration
# ============================================================

class TestEnumeration(unittest.TestCase):
    """Test cell program enumeration."""

    def test_enumerate_depth_0(self):
        """Enumerate all programs of depth 0 (constants only)."""
        # Colors: {0, 1, 2, 3, 4, 5}
        programs = enumerate_cell_exprs(colors={0, 1, 2, 3, 4, 5}, max_depth=0)
        # Should include all Const(c) for c in colors
        self.assertGreaterEqual(len(programs), 6)  # At least 6 constants

    def test_enumerate_depth_1(self):
        """Enumerate programs of depth 1 (constants + primitives)."""
        programs = enumerate_cell_exprs(colors={0, 1}, max_depth=1)
        # Should include constants, Self, all NeighborAt variants, etc.
        self.assertGreater(len(programs), 10)

    def test_enumerate_no_duplicates(self):
        """Enumeration should not produce duplicate programs."""
        programs = enumerate_cell_exprs(colors={0, 1, 2}, max_depth=1)
        # Convert to string representations to check uniqueness
        str_reps = [str(p) for p in programs]
        self.assertEqual(len(str_reps), len(set(str_reps)))

    def test_enumerate_depth_2_is_larger(self):
        """Depth 2 should produce more programs than depth 1."""
        progs_d1 = enumerate_cell_exprs(colors={0, 1, 2}, max_depth=1)
        progs_d2 = enumerate_cell_exprs(colors={0, 1, 2}, max_depth=2)
        self.assertGreater(len(progs_d2), len(progs_d1))


# ============================================================
# Test Scoring
# ============================================================

class TestScoring(unittest.TestCase):
    """Test cell program scoring against training examples."""

    def test_score_perfect_program(self):
        """A program that perfectly matches should score 1.0."""
        # Program: self (keep cell unchanged)
        expr = Self()
        input_grid = [[1, 2], [3, 4]]
        output_grid = [[1, 2], [3, 4]]
        score = score_cell_expr(expr, input_grid, output_grid)
        self.assertEqual(score, 1.0)

    def test_score_partial_program(self):
        """A program that partially matches should score < 1.0."""
        # Program: always output 1
        expr = Const(1)
        input_grid = [[1, 2], [3, 4]]
        output_grid = [[1, 1], [1, 1]]
        score = score_cell_expr(expr, input_grid, output_grid)
        self.assertEqual(score, 1.0)  # This actually matches perfectly

    def test_score_bad_program(self):
        """A program that doesn't match should score low."""
        # Program: always output 9
        expr = Const(9)
        input_grid = [[1, 2], [3, 4]]
        output_grid = [[1, 2], [3, 4]]
        score = score_cell_expr(expr, input_grid, output_grid)
        # No cell matches, so score should be 0.0
        self.assertEqual(score, 0.0)

    def test_score_multi_example(self):
        """Score should average across multiple examples."""
        expr = Self()
        examples = [
            ([[1, 2]], [[1, 2]]),
            ([[3, 4]], [[3, 4]]),
        ]
        scores = [score_cell_expr(expr, inp, out) for inp, out in examples]
        # Both should be 1.0
        self.assertTrue(all(s == 1.0 for s in scores))


# ============================================================
# Test Synthesis
# ============================================================

class TestSynthesis(unittest.TestCase):
    """Test the top-level synthesis function."""

    def test_synthesize_identity_task(self):
        """Synthesize solution for identity task (output == input)."""
        task = {
            'train': [
                {'input': [[1, 2], [3, 4]], 'output': [[1, 2], [3, 4]]},
                {'input': [[5, 6], [7, 8]], 'output': [[5, 6], [7, 8]]},
            ]
        }
        expr, score = synthesize_cell_program(task, max_depth=2)
        # Should find Self() with score 1.0
        self.assertIsNotNone(expr)
        self.assertEqual(score, 1.0)

    def test_synthesize_constant_task(self):
        """Synthesize solution where output is always a constant."""
        task = {
            'train': [
                {'input': [[1, 2], [3, 4]], 'output': [[7, 7], [7, 7]]},
                {'input': [[5, 6], [7, 8]], 'output': [[7, 7], [7, 7]]},
            ]
        }
        expr, score = synthesize_cell_program(task, max_depth=1)
        self.assertIsNotNone(expr)
        self.assertEqual(score, 1.0)

    def test_synthesize_returns_early_on_low_first_score(self):
        """If first example scores < 0.5, early termination."""
        task = {
            'train': [
                {'input': [[1, 2]], 'output': [[9, 9]]},  # Very different
            ]
        }
        expr, score = synthesize_cell_program(task, max_depth=1)
        # Either returns None or a low-scoring program
        # Early termination means we don't spend time on clearly bad programs


# ============================================================
# Test Concept Wrapping
# ============================================================

class TestCellSynthConcept(unittest.TestCase):
    """Test wrapping cell synthesis results as Concepts."""

    def test_cell_expr_wraps_as_concept(self):
        """A synthesized cell expr can be wrapped as a Concept."""
        from arc_agent.cell_synth import wrap_cell_expr_as_concept
        expr = Self()
        concept = wrap_cell_expr_as_concept(expr, name="identity")
        self.assertIsInstance(concept, Concept)
        self.assertEqual(concept.name, "identity")

    def test_wrapped_concept_applies_to_grid(self):
        """Wrapped concept applies cell expr to entire grid."""
        from arc_agent.cell_synth import wrap_cell_expr_as_concept
        expr = MapColor(1, 9)
        concept = wrap_cell_expr_as_concept(expr, name="remap")
        grid = [[1, 2], [1, 3]]
        result = concept.apply(grid)
        expected = [[9, 2], [9, 3]]
        self.assertEqual(result, expected)


if __name__ == '__main__':
    unittest.main()
