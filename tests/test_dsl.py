"""TDD tests for the DSL expression tree, interpreter, and synthesis engine.

Tests are organized by layer:
  1. DSL types and expression construction
  2. Atomic operations (cell-level, query, spatial, object, color, geometric)
  3. Combinators (compose, map_objects, if_then_else)
  4. Interpreter execution
  5. Bottom-up synthesis engine
"""
import unittest


# ============================================================
# 1. Expression tree construction and types
# ============================================================

class TestDSLTypes(unittest.TestCase):
    """Test DSL type system and expression tree construction."""

    def test_create_literal_expr(self):
        from arc_agent.dsl import DSLExpr, DSLType
        expr = DSLExpr.literal(5, DSLType.INT)
        self.assertEqual(expr.op, "literal")
        self.assertEqual(expr.value, 5)
        self.assertEqual(expr.return_type, DSLType.INT)

    def test_create_op_expr(self):
        from arc_agent.dsl import DSLExpr, DSLType
        grid_input = DSLExpr.input_grid()
        expr = DSLExpr.make_op("grid_height", [grid_input], DSLType.INT)
        self.assertEqual(expr.op, "grid_height")
        self.assertEqual(expr.return_type, DSLType.INT)
        self.assertEqual(len(expr.args), 1)

    def test_type_enum(self):
        from arc_agent.dsl import DSLType
        # All required types exist
        self.assertIsNotNone(DSLType.GRID)
        self.assertIsNotNone(DSLType.COLOR)
        self.assertIsNotNone(DSLType.INT)
        self.assertIsNotNone(DSLType.BOOL)

    def test_expr_depth(self):
        from arc_agent.dsl import DSLExpr, DSLType
        lit = DSLExpr.literal(3, DSLType.INT)
        self.assertEqual(lit.depth, 0)
        grid = DSLExpr.input_grid()
        h = DSLExpr.make_op("grid_height", [grid], DSLType.INT)
        self.assertEqual(h.depth, 1)

    def test_expr_size(self):
        """Size counts total nodes in the expression tree."""
        from arc_agent.dsl import DSLExpr, DSLType
        grid = DSLExpr.input_grid()
        self.assertEqual(grid.size, 1)
        h = DSLExpr.make_op("grid_height", [grid], DSLType.INT)
        self.assertEqual(h.size, 2)


# ============================================================
# 2. Atomic operations
# ============================================================

class TestDSLAtomicOps(unittest.TestCase):
    """Test individual DSL atomic operations via the interpreter."""

    def setUp(self):
        from arc_agent.dsl import DSLInterpreter
        self.interp = DSLInterpreter()
        self.grid_3x3 = [
            [1, 0, 2],
            [0, 3, 0],
            [1, 0, 1],
        ]

    def test_grid_height(self):
        from arc_agent.dsl import DSLExpr, DSLType
        expr = DSLExpr.make_op("grid_height", [DSLExpr.input_grid()], DSLType.INT)
        result = self.interp.evaluate(expr, self.grid_3x3)
        self.assertEqual(result, 3)

    def test_grid_width(self):
        from arc_agent.dsl import DSLExpr, DSLType
        expr = DSLExpr.make_op("grid_width", [DSLExpr.input_grid()], DSLType.INT)
        result = self.interp.evaluate(expr, self.grid_3x3)
        self.assertEqual(result, 3)

    def test_most_common_color(self):
        from arc_agent.dsl import DSLExpr, DSLType
        expr = DSLExpr.make_op("most_common_color", [DSLExpr.input_grid()], DSLType.COLOR)
        result = self.interp.evaluate(expr, self.grid_3x3)
        self.assertEqual(result, 0)  # 0 appears 5 times

    def test_least_common_color(self):
        from arc_agent.dsl import DSLExpr, DSLType
        expr = DSLExpr.make_op("least_common_color", [DSLExpr.input_grid()], DSLType.COLOR)
        result = self.interp.evaluate(expr, self.grid_3x3)
        # color 2 and 3 each appear once; either is valid
        self.assertIn(result, [2, 3])

    def test_replace_color(self):
        from arc_agent.dsl import DSLExpr, DSLType
        grid_in = DSLExpr.input_grid()
        old = DSLExpr.literal(1, DSLType.COLOR)
        new = DSLExpr.literal(5, DSLType.COLOR)
        expr = DSLExpr.make_op("replace_color", [grid_in, old, new], DSLType.GRID)
        result = self.interp.evaluate(expr, self.grid_3x3)
        expected = [
            [5, 0, 2],
            [0, 3, 0],
            [5, 0, 5],
        ]
        self.assertEqual(result, expected)

    def test_transpose(self):
        from arc_agent.dsl import DSLExpr, DSLType
        expr = DSLExpr.make_op("transpose", [DSLExpr.input_grid()], DSLType.GRID)
        grid_2x3 = [[1, 2, 3], [4, 5, 6]]
        result = self.interp.evaluate(expr, grid_2x3)
        self.assertEqual(result, [[1, 4], [2, 5], [3, 6]])

    def test_flip_h(self):
        from arc_agent.dsl import DSLExpr, DSLType
        expr = DSLExpr.make_op("flip_h", [DSLExpr.input_grid()], DSLType.GRID)
        grid = [[1, 2, 3], [4, 5, 6]]
        result = self.interp.evaluate(expr, grid)
        self.assertEqual(result, [[3, 2, 1], [6, 5, 4]])

    def test_flip_v(self):
        from arc_agent.dsl import DSLExpr, DSLType
        expr = DSLExpr.make_op("flip_v", [DSLExpr.input_grid()], DSLType.GRID)
        grid = [[1, 2], [3, 4]]
        result = self.interp.evaluate(expr, grid)
        self.assertEqual(result, [[3, 4], [1, 2]])

    def test_rotate_90(self):
        from arc_agent.dsl import DSLExpr, DSLType
        expr = DSLExpr.make_op("rotate_90", [DSLExpr.input_grid()], DSLType.GRID)
        grid = [[1, 2], [3, 4]]
        result = self.interp.evaluate(expr, grid)
        self.assertEqual(result, [[3, 1], [4, 2]])

    def test_count_color(self):
        from arc_agent.dsl import DSLExpr, DSLType
        grid_in = DSLExpr.input_grid()
        color = DSLExpr.literal(1, DSLType.COLOR)
        expr = DSLExpr.make_op("count_color", [grid_in, color], DSLType.INT)
        result = self.interp.evaluate(expr, self.grid_3x3)
        self.assertEqual(result, 3)  # three 1s

    def test_apply_color_map(self):
        """Apply a learned color mapping to a grid."""
        from arc_agent.dsl import DSLExpr, DSLType
        grid_in = DSLExpr.input_grid()
        # Build a color map: 1→5, 2→5, 3→5 (all foreground → 5)
        cmap = DSLExpr.literal({1: 5, 2: 5, 3: 5}, DSLType.COLOR_MAP)
        expr = DSLExpr.make_op("apply_color_map", [grid_in, cmap], DSLType.GRID)
        result = self.interp.evaluate(expr, self.grid_3x3)
        expected = [
            [5, 0, 5],
            [0, 5, 0],
            [5, 0, 5],
        ]
        self.assertEqual(result, expected)


# ============================================================
# 3. Combinators
# ============================================================

class TestDSLCombinators(unittest.TestCase):
    """Test DSL combinators: compose, map_objects, if_then_else."""

    def setUp(self):
        from arc_agent.dsl import DSLInterpreter
        self.interp = DSLInterpreter()

    def test_compose_two_transforms(self):
        """compose(flip_h, flip_v) = rotate 180."""
        from arc_agent.dsl import DSLExpr, DSLType
        grid_in = DSLExpr.input_grid()
        flip_h = DSLExpr.make_op("flip_h", [grid_in], DSLType.GRID)
        composed = DSLExpr.make_op("flip_v", [flip_h], DSLType.GRID)

        grid = [[1, 2], [3, 4]]
        result = self.interp.evaluate(composed, grid)
        self.assertEqual(result, [[4, 3], [2, 1]])

    def test_compose_replace_then_flip(self):
        """Replace color 1→5, then flip horizontally."""
        from arc_agent.dsl import DSLExpr, DSLType
        grid_in = DSLExpr.input_grid()
        old = DSLExpr.literal(1, DSLType.COLOR)
        new = DSLExpr.literal(5, DSLType.COLOR)
        replaced = DSLExpr.make_op("replace_color", [grid_in, old, new], DSLType.GRID)
        final = DSLExpr.make_op("flip_h", [replaced], DSLType.GRID)

        grid = [[1, 0], [0, 1]]
        result = self.interp.evaluate(final, grid)
        self.assertEqual(result, [[0, 5], [5, 0]])

    def test_map_objects_recolor(self):
        """Apply a recolor transform to each object independently."""
        from arc_agent.dsl import DSLExpr, DSLType

        # Grid with two separate objects (color 1 and color 2)
        grid = [
            [0, 1, 0, 0],
            [0, 1, 0, 2],
            [0, 0, 0, 2],
        ]
        # map_objects should apply a transform to each object's subgrid
        # The transform: replace any non-zero color with 5
        grid_in = DSLExpr.input_grid()
        expr = DSLExpr.make_op("map_objects", [
            grid_in,
            DSLExpr.lambda_expr("replace_all_fg_with",
                                [DSLExpr.literal(5, DSLType.COLOR)],
                                DSLType.GRID)
        ], DSLType.GRID)

        result = self.interp.evaluate(expr, grid)
        expected = [
            [0, 5, 0, 0],
            [0, 5, 0, 5],
            [0, 0, 0, 5],
        ]
        self.assertEqual(result, expected)


# ============================================================
# 4. Interpreter edge cases
# ============================================================

class TestDSLInterpreterEdgeCases(unittest.TestCase):
    """Test interpreter robustness."""

    def test_empty_grid(self):
        from arc_agent.dsl import DSLExpr, DSLType, DSLInterpreter
        interp = DSLInterpreter()
        expr = DSLExpr.make_op("grid_height", [DSLExpr.input_grid()], DSLType.INT)
        result = interp.evaluate(expr, [])
        self.assertEqual(result, 0)

    def test_single_cell_grid(self):
        from arc_agent.dsl import DSLExpr, DSLType, DSLInterpreter
        interp = DSLInterpreter()
        expr = DSLExpr.make_op("most_common_color", [DSLExpr.input_grid()], DSLType.COLOR)
        result = interp.evaluate(expr, [[7]])
        self.assertEqual(result, 7)

    def test_invalid_op_returns_none(self):
        """Unknown operations should raise or return None gracefully."""
        from arc_agent.dsl import DSLExpr, DSLType, DSLInterpreter
        interp = DSLInterpreter()
        expr = DSLExpr.make_op("nonexistent_op", [DSLExpr.input_grid()], DSLType.GRID)
        result = interp.evaluate(expr, [[1]])
        self.assertIsNone(result)

    def test_replace_color_preserves_dims(self):
        from arc_agent.dsl import DSLExpr, DSLType, DSLInterpreter
        interp = DSLInterpreter()
        grid_in = DSLExpr.input_grid()
        expr = DSLExpr.make_op("replace_color", [
            grid_in,
            DSLExpr.literal(0, DSLType.COLOR),
            DSLExpr.literal(9, DSLType.COLOR),
        ], DSLType.GRID)
        grid = [[0, 1], [2, 0], [0, 0]]
        result = interp.evaluate(expr, grid)
        self.assertEqual(len(result), 3)
        self.assertEqual(len(result[0]), 2)


# ============================================================
# 5. Bottom-up synthesis
# ============================================================

class TestDSLSynthesis(unittest.TestCase):
    """Test bottom-up synthesis engine."""

    def test_synthesize_identity(self):
        """Trivial task: output == input. Should find identity."""
        from arc_agent.dsl_synth import synthesize_dsl_program
        from arc_agent.scorer import TaskCache

        inp = [[1, 2], [3, 4]]
        task = {"train": [{"input": inp, "output": inp}]}
        cache = TaskCache(task)
        result = synthesize_dsl_program(task, cache, time_budget=2.0)
        # Should find the input_grid expression (identity)
        self.assertIsNotNone(result)

    def test_synthesize_flip_h(self):
        """Task: output is horizontal flip of input."""
        from arc_agent.dsl_synth import synthesize_dsl_program
        from arc_agent.scorer import TaskCache

        task = {"train": [
            {"input": [[1, 2, 3]], "output": [[3, 2, 1]]},
            {"input": [[4, 5]], "output": [[5, 4]]},
        ]}
        cache = TaskCache(task)
        result = synthesize_dsl_program(task, cache, time_budget=5.0)
        self.assertIsNotNone(result)
        self.assertTrue(cache.is_pixel_perfect(result))

    def test_synthesize_color_replace(self):
        """Task: replace color 1 with color 5."""
        from arc_agent.dsl_synth import synthesize_dsl_program
        from arc_agent.scorer import TaskCache

        task = {"train": [
            {"input": [[1, 0], [0, 1]], "output": [[5, 0], [0, 5]]},
            {"input": [[0, 1, 1]], "output": [[0, 5, 5]]},
        ]}
        cache = TaskCache(task)
        result = synthesize_dsl_program(task, cache, time_budget=5.0)
        self.assertIsNotNone(result)
        self.assertTrue(cache.is_pixel_perfect(result))

    def test_synthesize_color_map(self):
        """Task: apply a color mapping (1→3, 2→4)."""
        from arc_agent.dsl_synth import synthesize_dsl_program
        from arc_agent.scorer import TaskCache

        task = {"train": [
            {"input": [[1, 2], [0, 1]], "output": [[3, 4], [0, 3]]},
            {"input": [[2, 0, 1]], "output": [[4, 0, 3]]},
        ]}
        cache = TaskCache(task)
        result = synthesize_dsl_program(task, cache, time_budget=5.0)
        self.assertIsNotNone(result)
        self.assertTrue(cache.is_pixel_perfect(result))

    def test_synthesize_returns_none_for_impossible(self):
        """Contradictory mapping should not be solved."""
        from arc_agent.dsl_synth import synthesize_dsl_program
        from arc_agent.scorer import TaskCache

        # Same pixel (1) maps to different values in different examples
        task = {"train": [
            {"input": [[1, 2], [3, 4]], "output": [[7, 8], [9, 0]]},
            {"input": [[1, 2], [3, 4]], "output": [[0, 8], [9, 7]]},
        ]}
        cache = TaskCache(task)
        result = synthesize_dsl_program(task, cache, time_budget=1.0)
        # Should either return None or a non-perfect program
        if result is not None:
            self.assertFalse(cache.is_pixel_perfect(result))

    def test_synthesize_compose_two(self):
        """Task requiring composition: flip_h then flip_v = rotate 180."""
        from arc_agent.dsl_synth import synthesize_dsl_program
        from arc_agent.scorer import TaskCache

        task = {"train": [
            {"input": [[1, 2], [3, 4]], "output": [[4, 3], [2, 1]]},
            {"input": [[5, 6, 7], [8, 9, 0]], "output": [[0, 9, 8], [7, 6, 5]]},
        ]}
        cache = TaskCache(task)
        result = synthesize_dsl_program(task, cache, time_budget=5.0)
        self.assertIsNotNone(result)
        self.assertTrue(cache.is_pixel_perfect(result))


# ============================================================
# 6. New DSL operations: neighbor rule, crop, fill
# ============================================================

class TestDSLNeighborOps(unittest.TestCase):
    """Test neighborhood-based DSL operations."""

    def setUp(self):
        from arc_agent.dsl import DSLInterpreter
        self.interp = DSLInterpreter()

    def test_crop_to_content(self):
        """Crop grid to bounding box of non-background cells."""
        from arc_agent.dsl import DSLExpr, DSLType
        expr = DSLExpr.make_op("crop_to_content", [DSLExpr.input_grid()],
                               DSLType.GRID)
        grid = [
            [0, 0, 0, 0],
            [0, 1, 2, 0],
            [0, 3, 0, 0],
            [0, 0, 0, 0],
        ]
        result = self.interp.evaluate(expr, grid)
        self.assertEqual(result, [[1, 2], [3, 0]])

    def test_crop_to_content_full(self):
        """If all cells are non-zero, return full grid."""
        from arc_agent.dsl import DSLExpr, DSLType
        expr = DSLExpr.make_op("crop_to_content", [DSLExpr.input_grid()],
                               DSLType.GRID)
        grid = [[1, 2], [3, 4]]
        result = self.interp.evaluate(expr, grid)
        self.assertEqual(result, [[1, 2], [3, 4]])

    def test_fill_background(self):
        """Fill background (most common color) with a new color."""
        from arc_agent.dsl import DSLExpr, DSLType
        grid_in = DSLExpr.input_grid()
        new_color = DSLExpr.literal(5, DSLType.COLOR)
        expr = DSLExpr.make_op("fill_background", [grid_in, new_color],
                               DSLType.GRID)
        grid = [[0, 1, 0], [0, 0, 2]]
        result = self.interp.evaluate(expr, grid)
        self.assertEqual(result, [[5, 1, 5], [5, 5, 2]])

    def test_apply_neighbor_rule(self):
        """Apply a learned neighborhood rule table."""
        from arc_agent.dsl import DSLExpr, DSLType
        grid_in = DSLExpr.input_grid()
        # Rule: (cell_color, n_nonbg_neighbors_4) → new_color
        # "If cell is 0 and has exactly 2 non-bg neighbors, become 5"
        rule = DSLExpr.literal(
            {(0, 2): 5},  # (cell_color, n_nonbg_4) → new_color
            DSLType.COLOR_MAP,
        )
        expr = DSLExpr.make_op("apply_neighbor_rule", [grid_in, rule],
                               DSLType.GRID)
        grid = [
            [0, 1, 0],
            [1, 0, 1],
            [0, 1, 0],
        ]
        result = self.interp.evaluate(expr, grid)
        # Center cell (1,1) is 0 with 4 non-bg neighbors → not matched (4≠2)
        # Cells (0,1), (1,0), (1,2), (2,1) are non-zero → unchanged
        # Corners (0,0), (0,2), (2,0), (2,2) are 0 with 2 non-bg neighbors → 5
        expected = [
            [5, 1, 5],
            [1, 0, 1],
            [5, 1, 5],
        ]
        self.assertEqual(result, expected)


class TestDSLSynthesisNeighbor(unittest.TestCase):
    """Test synthesis with neighbor-based operations."""

    def test_synthesize_crop_to_content(self):
        """Task: crop grid to bounding box of non-zero cells."""
        from arc_agent.dsl_synth import synthesize_dsl_program
        from arc_agent.scorer import TaskCache

        task = {"train": [
            {"input": [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
             "output": [[1]]},
            {"input": [[0, 0, 0, 0], [0, 2, 3, 0], [0, 0, 0, 0]],
             "output": [[2, 3]]},
        ]}
        cache = TaskCache(task)
        result = synthesize_dsl_program(task, cache, time_budget=5.0)
        self.assertIsNotNone(result)
        self.assertTrue(cache.is_pixel_perfect(result))

    def test_synthesize_neighbor_rule(self):
        """Task solvable by a neighborhood rule."""
        from arc_agent.dsl_synth import synthesize_dsl_program
        from arc_agent.scorer import TaskCache

        # Task: fill corners of a cross pattern
        # Input has a cross of 1s, output fills the empty corners with 5
        task = {"train": [
            {"input": [[0, 1, 0], [1, 1, 1], [0, 1, 0]],
             "output": [[5, 1, 5], [1, 1, 1], [5, 1, 5]]},
            {"input": [[0, 0, 1, 0, 0],
                       [0, 0, 1, 0, 0],
                       [1, 1, 1, 1, 1],
                       [0, 0, 1, 0, 0],
                       [0, 0, 1, 0, 0]],
             "output": [[5, 5, 1, 5, 5],
                        [5, 5, 1, 5, 5],
                        [1, 1, 1, 1, 1],
                        [5, 5, 1, 5, 5],
                        [5, 5, 1, 5, 5]]},
        ]}
        cache = TaskCache(task)
        result = synthesize_dsl_program(task, cache, time_budget=5.0)
        self.assertIsNotNone(result)
        self.assertTrue(cache.is_pixel_perfect(result))


if __name__ == "__main__":
    unittest.main()
