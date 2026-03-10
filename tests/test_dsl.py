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


class TestDSLSpatialOps(unittest.TestCase):
    """Test spatial/structural DSL operations: tiling, scaling, gravity, symmetry, denoise."""

    def setUp(self):
        from arc_agent.dsl import DSLInterpreter
        self.interp = DSLInterpreter()

    def test_tile_2x2(self):
        """Tile grid in a 2x2 arrangement."""
        from arc_agent.dsl import DSLExpr, DSLType
        expr = DSLExpr.make_op("tile_2x2", [DSLExpr.input_grid()], DSLType.GRID)
        grid = [[1, 2], [3, 4]]
        result = self.interp.evaluate(expr, grid)
        expected = [
            [1, 2, 1, 2],
            [3, 4, 3, 4],
            [1, 2, 1, 2],
            [3, 4, 3, 4],
        ]
        self.assertEqual(result, expected)

    def test_tile_3x3(self):
        """Tile grid in a 3x3 arrangement."""
        from arc_agent.dsl import DSLExpr, DSLType
        expr = DSLExpr.make_op("tile_3x3", [DSLExpr.input_grid()], DSLType.GRID)
        grid = [[1]]
        result = self.interp.evaluate(expr, grid)
        expected = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
        self.assertEqual(result, expected)

    def test_scale_2x(self):
        """Scale each cell to a 2x2 block."""
        from arc_agent.dsl import DSLExpr, DSLType
        expr = DSLExpr.make_op("scale_2x", [DSLExpr.input_grid()], DSLType.GRID)
        grid = [[1, 2], [3, 4]]
        result = self.interp.evaluate(expr, grid)
        expected = [
            [1, 1, 2, 2],
            [1, 1, 2, 2],
            [3, 3, 4, 4],
            [3, 3, 4, 4],
        ]
        self.assertEqual(result, expected)

    def test_scale_3x(self):
        """Scale each cell to a 3x3 block."""
        from arc_agent.dsl import DSLExpr, DSLType
        expr = DSLExpr.make_op("scale_3x", [DSLExpr.input_grid()], DSLType.GRID)
        grid = [[1]]
        result = self.interp.evaluate(expr, grid)
        expected = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
        self.assertEqual(result, expected)

    def test_gravity_down(self):
        """Move non-zero cells to bottom of columns."""
        from arc_agent.dsl import DSLExpr, DSLType
        expr = DSLExpr.make_op("gravity_down", [DSLExpr.input_grid()], DSLType.GRID)
        grid = [[1, 0, 2], [0, 3, 0], [0, 0, 0]]
        result = self.interp.evaluate(expr, grid)
        expected = [[0, 0, 0], [0, 0, 0], [1, 3, 2]]
        self.assertEqual(result, expected)

    def test_gravity_up(self):
        """Move non-zero cells to top of columns."""
        from arc_agent.dsl import DSLExpr, DSLType
        expr = DSLExpr.make_op("gravity_up", [DSLExpr.input_grid()], DSLType.GRID)
        grid = [[0, 0, 0], [0, 3, 0], [1, 0, 2]]
        result = self.interp.evaluate(expr, grid)
        expected = [[1, 3, 2], [0, 0, 0], [0, 0, 0]]
        self.assertEqual(result, expected)

    def test_gravity_left(self):
        """Move non-zero cells to left of rows."""
        from arc_agent.dsl import DSLExpr, DSLType
        expr = DSLExpr.make_op("gravity_left", [DSLExpr.input_grid()], DSLType.GRID)
        grid = [[0, 1, 0], [2, 0, 3]]
        result = self.interp.evaluate(expr, grid)
        expected = [[1, 0, 0], [2, 3, 0]]
        self.assertEqual(result, expected)

    def test_gravity_right(self):
        """Move non-zero cells to right of rows."""
        from arc_agent.dsl import DSLExpr, DSLType
        expr = DSLExpr.make_op("gravity_right", [DSLExpr.input_grid()], DSLType.GRID)
        grid = [[0, 1, 0], [2, 0, 3]]
        result = self.interp.evaluate(expr, grid)
        expected = [[0, 0, 1], [0, 2, 3]]
        self.assertEqual(result, expected)

    def test_complete_symmetry_h(self):
        """Complete horizontal symmetry: mirror the denser half."""
        from arc_agent.dsl import DSLExpr, DSLType
        expr = DSLExpr.make_op("complete_symmetry_h", [DSLExpr.input_grid()],
                               DSLType.GRID)
        # Left half has content, right is empty
        grid = [[1, 2, 0, 0], [3, 4, 0, 0]]
        result = self.interp.evaluate(expr, grid)
        expected = [[1, 2, 2, 1], [3, 4, 4, 3]]
        self.assertEqual(result, expected)

    def test_complete_symmetry_v(self):
        """Complete vertical symmetry: mirror the denser half."""
        from arc_agent.dsl import DSLExpr, DSLType
        expr = DSLExpr.make_op("complete_symmetry_v", [DSLExpr.input_grid()],
                               DSLType.GRID)
        # Top half has content, bottom is empty
        grid = [[1, 2], [3, 4], [0, 0], [0, 0]]
        result = self.interp.evaluate(expr, grid)
        expected = [[1, 2], [3, 4], [3, 4], [1, 2]]
        self.assertEqual(result, expected)

    def test_denoise_3x3(self):
        """3x3 majority vote denoising."""
        from arc_agent.dsl import DSLExpr, DSLType
        expr = DSLExpr.make_op("denoise_3x3", [DSLExpr.input_grid()],
                               DSLType.GRID)
        # A grid of 1s with a single noisy 2 in the middle
        grid = [[1, 1, 1], [1, 2, 1], [1, 1, 1]]
        result = self.interp.evaluate(expr, grid)
        expected = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
        self.assertEqual(result, expected)


class TestDSLSynthesisSpatial(unittest.TestCase):
    """Test synthesis with spatial operations."""

    def test_synthesize_tile_2x2(self):
        """Task: tile input grid 2x2."""
        from arc_agent.dsl_synth import synthesize_dsl_program
        from arc_agent.scorer import TaskCache
        task = {"train": [
            {"input": [[1, 2], [3, 4]],
             "output": [[1, 2, 1, 2], [3, 4, 3, 4],
                        [1, 2, 1, 2], [3, 4, 3, 4]]},
            {"input": [[5]],
             "output": [[5, 5], [5, 5]]},
        ]}
        cache = TaskCache(task)
        result = synthesize_dsl_program(task, cache, time_budget=5.0)
        self.assertIsNotNone(result)
        self.assertTrue(cache.is_pixel_perfect(result))

    def test_synthesize_scale_2x(self):
        """Task: scale each cell to 2x2 block."""
        from arc_agent.dsl_synth import synthesize_dsl_program
        from arc_agent.scorer import TaskCache
        task = {"train": [
            {"input": [[1, 2]],
             "output": [[1, 1, 2, 2], [1, 1, 2, 2]]},
            {"input": [[3]],
             "output": [[3, 3], [3, 3]]},
        ]}
        cache = TaskCache(task)
        result = synthesize_dsl_program(task, cache, time_budget=5.0)
        self.assertIsNotNone(result)
        self.assertTrue(cache.is_pixel_perfect(result))

    def test_synthesize_gravity_down(self):
        """Task: gravity pushes all cells down."""
        from arc_agent.dsl_synth import synthesize_dsl_program
        from arc_agent.scorer import TaskCache
        task = {"train": [
            {"input": [[1, 0], [0, 2]],
             "output": [[0, 0], [1, 2]]},
            {"input": [[3, 4, 0], [0, 0, 5], [0, 0, 0]],
             "output": [[0, 0, 0], [0, 0, 0], [3, 4, 5]]},
        ]}
        cache = TaskCache(task)
        result = synthesize_dsl_program(task, cache, time_budget=5.0)
        self.assertIsNotNone(result)
        self.assertTrue(cache.is_pixel_perfect(result))

    def test_synthesize_complete_symmetry_h(self):
        """Task: complete horizontal symmetry."""
        from arc_agent.dsl_synth import synthesize_dsl_program
        from arc_agent.scorer import TaskCache
        task = {"train": [
            {"input": [[1, 2, 0, 0], [3, 4, 0, 0]],
             "output": [[1, 2, 2, 1], [3, 4, 4, 3]]},
            {"input": [[5, 0], [6, 0]],
             "output": [[5, 5], [6, 6]]},
        ]}
        cache = TaskCache(task)
        result = synthesize_dsl_program(task, cache, time_budget=5.0)
        self.assertIsNotNone(result)
        self.assertTrue(cache.is_pixel_perfect(result))


class TestDSLHalvingOps(unittest.TestCase):
    """Test grid halving and boolean overlay operations."""

    def setUp(self):
        from arc_agent.dsl import DSLInterpreter
        self.interp = DSLInterpreter()

    def test_get_top_half(self):
        from arc_agent.dsl import DSLExpr, DSLType
        expr = DSLExpr.make_op("get_top_half", [DSLExpr.input_grid()], DSLType.GRID)
        grid = [[1, 2], [3, 4], [5, 6], [7, 8]]
        result = self.interp.evaluate(expr, grid)
        self.assertEqual(result, [[1, 2], [3, 4]])

    def test_get_bottom_half(self):
        from arc_agent.dsl import DSLExpr, DSLType
        expr = DSLExpr.make_op("get_bottom_half", [DSLExpr.input_grid()], DSLType.GRID)
        grid = [[1, 2], [3, 4], [5, 6], [7, 8]]
        result = self.interp.evaluate(expr, grid)
        self.assertEqual(result, [[5, 6], [7, 8]])

    def test_get_left_half(self):
        from arc_agent.dsl import DSLExpr, DSLType
        expr = DSLExpr.make_op("get_left_half", [DSLExpr.input_grid()], DSLType.GRID)
        grid = [[1, 2, 3, 4], [5, 6, 7, 8]]
        result = self.interp.evaluate(expr, grid)
        self.assertEqual(result, [[1, 2], [5, 6]])

    def test_get_right_half(self):
        from arc_agent.dsl import DSLExpr, DSLType
        expr = DSLExpr.make_op("get_right_half", [DSLExpr.input_grid()], DSLType.GRID)
        grid = [[1, 2, 3, 4], [5, 6, 7, 8]]
        result = self.interp.evaluate(expr, grid)
        self.assertEqual(result, [[3, 4], [7, 8]])

    def test_xor_halves_v(self):
        """XOR top and bottom halves vertically."""
        from arc_agent.dsl import DSLExpr, DSLType
        expr = DSLExpr.make_op("xor_halves_v", [DSLExpr.input_grid()], DSLType.GRID)
        # Top: [[1, 0]], Bottom: [[0, 2]]
        grid = [[1, 0], [0, 2]]
        result = self.interp.evaluate(expr, grid)
        # XOR: both non-zero → 0, one non-zero → keep it
        self.assertEqual(result, [[1, 2]])

    def test_or_halves_v(self):
        """OR top and bottom halves vertically."""
        from arc_agent.dsl import DSLExpr, DSLType
        expr = DSLExpr.make_op("or_halves_v", [DSLExpr.input_grid()], DSLType.GRID)
        grid = [[1, 0], [0, 2]]
        result = self.interp.evaluate(expr, grid)
        # OR: prefer top if non-zero, else bottom
        self.assertEqual(result, [[1, 2]])

    def test_and_halves_v(self):
        """AND top and bottom halves vertically."""
        from arc_agent.dsl import DSLExpr, DSLType
        expr = DSLExpr.make_op("and_halves_v", [DSLExpr.input_grid()], DSLType.GRID)
        grid = [[1, 0], [0, 2]]
        result = self.interp.evaluate(expr, grid)
        # AND: both must be non-zero to keep
        self.assertEqual(result, [[0, 0]])

    def test_and_halves_v_overlap(self):
        """AND with overlapping non-zero cells."""
        from arc_agent.dsl import DSLExpr, DSLType
        expr = DSLExpr.make_op("and_halves_v", [DSLExpr.input_grid()], DSLType.GRID)
        grid = [[1, 3], [2, 4]]
        result = self.interp.evaluate(expr, grid)
        # AND: both non-zero → keep top value
        self.assertEqual(result, [[1, 3]])

    def test_xor_halves_h(self):
        """XOR left and right halves horizontally."""
        from arc_agent.dsl import DSLExpr, DSLType
        expr = DSLExpr.make_op("xor_halves_h", [DSLExpr.input_grid()], DSLType.GRID)
        grid = [[1, 0, 0, 2]]
        result = self.interp.evaluate(expr, grid)
        self.assertEqual(result, [[1, 2]])


class TestDSLSynthesisHalving(unittest.TestCase):
    """Test synthesis with halving operations."""

    def test_synthesize_xor_halves_v(self):
        """Task: XOR top and bottom halves."""
        from arc_agent.dsl_synth import synthesize_dsl_program
        from arc_agent.scorer import TaskCache
        task = {"train": [
            {"input": [[1, 0, 0], [0, 0, 2]],
             "output": [[1, 0, 2]]},
            {"input": [[3, 0], [0, 4]],
             "output": [[3, 4]]},
        ]}
        cache = TaskCache(task)
        result = synthesize_dsl_program(task, cache, time_budget=5.0)
        self.assertIsNotNone(result)
        self.assertTrue(cache.is_pixel_perfect(result))

    def test_synthesize_get_top_half(self):
        """Task: extract top half of grid."""
        from arc_agent.dsl_synth import synthesize_dsl_program
        from arc_agent.scorer import TaskCache
        task = {"train": [
            {"input": [[1, 2], [3, 4], [5, 6], [7, 8]],
             "output": [[1, 2], [3, 4]]},
            {"input": [[9, 0], [1, 2]],
             "output": [[9, 0]]},
        ]}
        cache = TaskCache(task)
        result = synthesize_dsl_program(task, cache, time_budget=5.0)
        self.assertIsNotNone(result)
        self.assertTrue(cache.is_pixel_perfect(result))

    def test_dimension_shortcut_tile_with_color_map(self):
        """Task: tile 2x2 then swap colors."""
        from arc_agent.dsl_synth import synthesize_dsl_program
        from arc_agent.scorer import TaskCache
        # Input: [[1]], Output: [[2, 2], [2, 2]] (tile + color map 1→2)
        task = {"train": [
            {"input": [[1]],
             "output": [[2, 2], [2, 2]]},
            {"input": [[3]],
             "output": [[4, 4], [4, 4]]},
        ]}
        cache = TaskCache(task)
        result = synthesize_dsl_program(task, cache, time_budget=5.0)
        # This may or may not find it depending on whether tile+color_map
        # or scale+color_map is tried. Both are valid.
        if result is not None:
            self.assertTrue(cache.is_pixel_perfect(result))


class TestDSLLoocv(unittest.TestCase):
    """Test LOOCV generalization check for DSL neighbor rules."""

    def test_loocv_passes_consistent_rule(self):
        """A consistent neighbor rule should pass LOOCV."""
        from arc_agent.dsl_synth import _loocv_neighbor_rule
        from arc_agent.dsl import DSLInterpreter

        interp = DSLInterpreter()
        # Simple rule: isolated cells (0 non-bg neighbors) become color 2
        # This rule should generalize across examples
        inputs = [
            [[0, 1, 0], [0, 0, 0], [0, 1, 0]],
            [[0, 0, 0], [1, 0, 1], [0, 0, 0]],
            [[1, 0, 0], [0, 0, 0], [0, 0, 1]],
        ]
        outputs = [
            [[0, 2, 0], [0, 0, 0], [0, 2, 0]],
            [[0, 0, 0], [2, 0, 2], [0, 0, 0]],
            [[2, 0, 0], [0, 0, 0], [0, 0, 2]],
        ]
        self.assertTrue(_loocv_neighbor_rule(inputs, outputs, interp))

    def test_loocv_fails_overfitting_rule(self):
        """A rule that fits training but doesn't generalize should fail.

        The rule learned from all 3 examples may be consistent but overly
        specific — when learned from 2 examples it doesn't predict the 3rd.
        """
        from arc_agent.dsl_synth import _loocv_neighbor_rule, _learn_neighbor_rule
        from arc_agent.dsl import DSLInterpreter

        interp = DSLInterpreter()
        # Task where each example has a different pattern:
        # Example 1: cell (1,0) neighbors → specific output
        # Example 2: cell (1,1) neighbors → different output
        # These won't produce consistent LOOCV because
        # the neighbor count→color mapping changes per example
        inputs = [
            [[1, 0], [0, 1]],
            [[0, 1], [1, 0]],
        ]
        # Output swaps in a way that depends on position, not just neighbors
        outputs = [
            [[0, 1], [1, 0]],
            [[1, 0], [0, 1]],
        ]
        # First check: the rule from all examples should be None
        # (inconsistent) so LOOCV won't even be reached
        rule = _learn_neighbor_rule(inputs, outputs)
        if rule is not None:
            # If a rule is learned from all, LOOCV should catch it
            result = _loocv_neighbor_rule(inputs, outputs, interp)
            self.assertFalse(result)

    def test_loocv_single_example_passes(self):
        """With only 1 training example, LOOCV can't validate — passes."""
        from arc_agent.dsl_synth import _loocv_neighbor_rule
        from arc_agent.dsl import DSLInterpreter

        interp = DSLInterpreter()
        inputs = [[[1, 0], [0, 1]]]
        outputs = [[[2, 0], [0, 2]]]
        self.assertTrue(_loocv_neighbor_rule(inputs, outputs, interp))

    def test_loocv_two_examples_generalizes(self):
        """Two-example LOOCV: rule learned from 1 predicts the other.

        Rule: color-1 cells with exactly 2 non-bg 4-neighbors → color 3.
        Both examples share the same neighbor-count distribution so
        a rule learned from one generalizes to the other.
        """
        from arc_agent.dsl_synth import _loocv_neighbor_rule
        from arc_agent.dsl import DSLInterpreter

        interp = DSLInterpreter()
        # Both grids: same size, same structure of neighbor counts
        # 3x3 grid with center cell having 4 non-bg neighbors, etc.
        inputs = [
            [[0, 1, 0], [1, 1, 1], [0, 1, 0]],  # cross pattern
            [[0, 1, 0], [1, 1, 1], [0, 1, 0]],  # same pattern
        ]
        # Rule: (1, 2) → 3, (1, 4) → 3, (1, 1) → 3
        # (color 1 cells with any non-bg neighbors become 3)
        outputs = [
            [[0, 3, 0], [3, 3, 3], [0, 3, 0]],
            [[0, 3, 0], [3, 3, 3], [0, 3, 0]],
        ]
        self.assertTrue(_loocv_neighbor_rule(inputs, outputs, interp))


class TestDSLSymmetryOps(unittest.TestCase):
    """Test diagonal and 4-way symmetry completion DSL operations."""

    def test_complete_symmetry_diagonal(self):
        """Complete diagonal symmetry: fill zeros using reflected value."""
        from arc_agent.dsl import DSLExpr, DSLType, DSLInterpreter

        interp = DSLInterpreter()
        grid = DSLExpr.input_grid()
        expr = DSLExpr.make_op("complete_symmetry_diagonal", [grid], DSLType.GRID)

        # Grid with partial diagonal symmetry: (0,1)=3 but (1,0)=0
        inp = [[0, 3, 0], [0, 0, 0], [0, 0, 0]]
        result = interp.evaluate(expr, inp)
        # (1,0) should be filled with value from (0,1) = 3
        self.assertEqual(result[1][0], 3)
        # Existing non-zero values stay
        self.assertEqual(result[0][1], 3)

    def test_complete_symmetry_diagonal_no_overwrite(self):
        """Diagonal completion should not overwrite existing non-zero values."""
        from arc_agent.dsl import DSLExpr, DSLType, DSLInterpreter

        interp = DSLInterpreter()
        grid = DSLExpr.input_grid()
        expr = DSLExpr.make_op("complete_symmetry_diagonal", [grid], DSLType.GRID)

        inp = [[1, 2], [3, 4]]
        result = interp.evaluate(expr, inp)
        # All cells non-zero, nothing should change
        self.assertEqual(result, [[1, 2], [3, 4]])

    def test_complete_symmetry_4way(self):
        """4-way symmetry: complete across both axes and both diagonals."""
        from arc_agent.dsl import DSLExpr, DSLType, DSLInterpreter

        interp = DSLInterpreter()
        grid = DSLExpr.input_grid()
        expr = DSLExpr.make_op("complete_symmetry_4way", [grid], DSLType.GRID)

        # 5x5 grid with a few non-zero cells, should fill symmetrically
        inp = [
            [0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ]
        result = interp.evaluate(expr, inp)
        # (1,2)=1 should reflect to:
        # H: (3,2), V: (1,2) (same), Diag: (2,1), AntiDiag: (2,3)
        # And their reflections...
        self.assertEqual(result[1][2], 1)  # original
        self.assertEqual(result[3][2], 1)  # horizontal mirror
        self.assertEqual(result[2][1], 1)  # diagonal
        self.assertEqual(result[2][3], 1)  # anti-diagonal

    def test_complete_symmetry_diagonal_in_registry(self):
        """Verify diagonal symmetry is in the DSL_OPS registry."""
        from arc_agent.dsl import DSL_OPS, DSLType
        self.assertIn("complete_symmetry_diagonal", DSL_OPS)
        arg_types, ret = DSL_OPS["complete_symmetry_diagonal"]
        self.assertEqual(arg_types, [DSLType.GRID])
        self.assertEqual(ret, DSLType.GRID)

    def test_complete_symmetry_4way_in_registry(self):
        """Verify 4-way symmetry is in the DSL_OPS registry."""
        from arc_agent.dsl import DSL_OPS, DSLType
        self.assertIn("complete_symmetry_4way", DSL_OPS)
        arg_types, ret = DSL_OPS["complete_symmetry_4way"]
        self.assertEqual(arg_types, [DSLType.GRID])
        self.assertEqual(ret, DSLType.GRID)


class TestDSLObjectOps(unittest.TestCase):
    """Test object-level DSL operations."""

    def test_extract_largest_object(self):
        """Extract the largest connected foreground object as a cropped grid."""
        from arc_agent.dsl import DSLExpr, DSLType, DSLInterpreter

        interp = DSLInterpreter()
        grid = DSLExpr.input_grid()
        expr = DSLExpr.make_op("extract_largest_object", [grid], DSLType.GRID)

        # Grid with two objects: a 3x3 block (9 cells) and a single cell
        inp = [
            [0, 0, 0, 0, 0],
            [0, 1, 1, 1, 0],
            [0, 1, 1, 1, 0],
            [0, 1, 1, 1, 0],
            [0, 0, 0, 0, 2],
        ]
        result = interp.evaluate(expr, inp)
        self.assertIsNotNone(result)
        # Should extract the 3x3 block of 1s
        self.assertEqual(len(result), 3)
        self.assertEqual(len(result[0]), 3)
        self.assertTrue(all(result[r][c] == 1 for r in range(3) for c in range(3)))

    def test_extract_largest_object_in_registry(self):
        """Verify extract_largest_object is in the DSL_OPS registry."""
        from arc_agent.dsl import DSL_OPS, DSLType
        self.assertIn("extract_largest_object", DSL_OPS)
        arg_types, ret = DSL_OPS["extract_largest_object"]
        self.assertEqual(arg_types, [DSLType.GRID])
        self.assertEqual(ret, DSLType.GRID)

    def test_extract_largest_object_empty_grid(self):
        """Extract from all-zero grid should return the grid unchanged."""
        from arc_agent.dsl import DSLExpr, DSLType, DSLInterpreter

        interp = DSLInterpreter()
        grid = DSLExpr.input_grid()
        expr = DSLExpr.make_op("extract_largest_object", [grid], DSLType.GRID)

        inp = [[0, 0], [0, 0]]
        result = interp.evaluate(expr, inp)
        self.assertIsNotNone(result)


class TestDSLSortOps(unittest.TestCase):
    """Test sort/reorder DSL operations."""

    def test_sort_rows_by_color_count(self):
        """Sort rows by number of non-zero cells (ascending)."""
        from arc_agent.dsl import DSLExpr, DSLType, DSLInterpreter

        interp = DSLInterpreter()
        grid = DSLExpr.input_grid()
        expr = DSLExpr.make_op("sort_rows_by_nonzero", [grid], DSLType.GRID)

        inp = [
            [1, 1, 1],  # 3 non-zero
            [1, 0, 0],  # 1 non-zero
            [1, 1, 0],  # 2 non-zero
        ]
        result = interp.evaluate(expr, inp)
        self.assertIsNotNone(result)
        # Should be sorted: 1 non-zero, 2 non-zero, 3 non-zero
        self.assertEqual(result[0], [1, 0, 0])
        self.assertEqual(result[1], [1, 1, 0])
        self.assertEqual(result[2], [1, 1, 1])

    def test_sort_rows_by_nonzero_in_registry(self):
        """Verify sort_rows_by_nonzero is in the DSL_OPS registry."""
        from arc_agent.dsl import DSL_OPS, DSLType
        self.assertIn("sort_rows_by_nonzero", DSL_OPS)


class TestDSLSynthesisNewOps(unittest.TestCase):
    """Test that new DSL ops compose correctly in synthesis."""

    def test_replace_color_then_complete_diagonal(self):
        """Compose replace_color + complete_symmetry_diagonal."""
        from arc_agent.dsl import DSLExpr, DSLType, DSLInterpreter

        interp = DSLInterpreter()
        # replace_color(input, 9, 0) then complete_symmetry_diagonal
        inp_expr = DSLExpr.input_grid()
        step1 = DSLExpr.make_op(
            "replace_color",
            [inp_expr, DSLExpr.literal(9, DSLType.COLOR),
             DSLExpr.literal(0, DSLType.COLOR)],
            DSLType.GRID,
        )
        step2 = DSLExpr.make_op(
            "complete_symmetry_diagonal", [step1], DSLType.GRID,
        )

        # Grid with 9s as noise and partial diagonal symmetry
        inp = [[0, 3, 9], [0, 0, 0], [0, 0, 0]]
        result = interp.evaluate(step2, inp)
        # After replace 9→0: [[0,3,0],[0,0,0],[0,0,0]]
        # After diag symm: (0,1)=3 → (1,0)=3
        self.assertEqual(result[0][1], 3)
        self.assertEqual(result[1][0], 3)
        self.assertEqual(result[0][2], 0)  # 9 was erased

    def test_crop_then_extract_largest(self):
        """Compose crop_to_content + extract_largest_object."""
        from arc_agent.dsl import DSLExpr, DSLType, DSLInterpreter

        interp = DSLInterpreter()
        inp_expr = DSLExpr.input_grid()
        step1 = DSLExpr.make_op("crop_to_content", [inp_expr], DSLType.GRID)
        step2 = DSLExpr.make_op("extract_largest_object", [step1], DSLType.GRID)

        inp = [
            [0, 0, 0, 0],
            [0, 1, 1, 0],
            [0, 1, 1, 0],
            [0, 0, 2, 0],
            [0, 0, 0, 0],
        ]
        result = interp.evaluate(step2, inp)
        self.assertIsNotNone(result)


class TestFillFrameInterior(unittest.TestCase):
    """Test the fill_frame_interior primitive."""

    def test_fill_frame_with_markers(self):
        """Fill inside a 2-border preserving marker bounding box."""
        from arc_agent.primitives import fill_frame_interior

        inp = [
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 2, 2, 2, 2, 2, 2, 2, 2, 0, 0],
            [0, 2, 0, 0, 0, 0, 0, 0, 2, 0, 0],
            [0, 2, 0, 5, 0, 0, 5, 0, 2, 0, 0],
            [0, 2, 0, 0, 0, 5, 0, 0, 2, 0, 0],
            [0, 2, 0, 0, 0, 0, 0, 0, 2, 0, 0],
            [0, 2, 2, 2, 2, 2, 2, 2, 2, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ]
        expected = [
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 2, 2, 2, 2, 2, 2, 2, 2, 0, 0],
            [0, 2, 2, 2, 2, 2, 2, 2, 2, 0, 0],
            [0, 2, 2, 5, 0, 0, 5, 2, 2, 0, 0],
            [0, 2, 2, 0, 0, 5, 0, 2, 2, 0, 0],
            [0, 2, 2, 2, 2, 2, 2, 2, 2, 0, 0],
            [0, 2, 2, 2, 2, 2, 2, 2, 2, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ]
        result = fill_frame_interior(inp)
        self.assertEqual(result, expected)

    def test_fill_frame_no_markers(self):
        """Fill inside a frame with no markers — fill everything."""
        from arc_agent.primitives import fill_frame_interior

        # Frame must be minority color (bg=0 is majority)
        inp = [
            [0, 0, 0, 0, 0, 0],
            [0, 3, 3, 3, 3, 0],
            [0, 3, 0, 0, 3, 0],
            [0, 3, 0, 0, 3, 0],
            [0, 3, 3, 3, 3, 0],
            [0, 0, 0, 0, 0, 0],
        ]
        expected = [
            [0, 0, 0, 0, 0, 0],
            [0, 3, 3, 3, 3, 0],
            [0, 3, 3, 3, 3, 0],
            [0, 3, 3, 3, 3, 0],
            [0, 3, 3, 3, 3, 0],
            [0, 0, 0, 0, 0, 0],
        ]
        result = fill_frame_interior(inp)
        self.assertEqual(result, expected)

    def test_no_frame_returns_unchanged(self):
        """Grid with no rectangular frame returns unchanged."""
        from arc_agent.primitives import fill_frame_interior

        inp = [[1, 0, 0], [0, 2, 0], [0, 0, 3]]
        result = fill_frame_interior(inp)
        self.assertEqual(result, inp)


class TestNeighborRule8Shortcut(unittest.TestCase):
    """Test 8-connected neighbor rule in DSL synthesis."""

    def test_8neighbor_rule_learns(self):
        """8-neighbor rule learns consistent mapping."""
        from arc_agent.dsl_synth import _learn_neighbor_rule_8

        # Simple task: fill bg cells with >=3 non-bg 8-neighbors
        inp1 = [[0, 1, 0], [1, 0, 1], [0, 1, 0]]
        out1 = [[0, 1, 0], [1, 1, 1], [0, 1, 0]]
        inp2 = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
        out2 = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]

        rule = _learn_neighbor_rule_8([inp1, inp2], [out1, out2])
        # Center cell (0, 4 diag neighbors) -> 1, rest stay
        self.assertIsNotNone(rule)


class TestNeighborRuleParityShortcut(unittest.TestCase):
    """Test parity-aware neighbor rule in DSL synthesis."""

    def test_parity_rule_learns_checkerboard(self):
        """Parity rule learns position-dependent patterns."""
        from arc_agent.dsl_synth import _learn_neighbor_rule_parity

        # Checkerboard fill: even positions get color 1, odd stay 0
        inp1 = [[0, 0, 0, 0], [0, 0, 0, 0]]
        out1 = [[1, 0, 1, 0], [0, 1, 0, 1]]

        rule = _learn_neighbor_rule_parity([inp1], [out1])
        self.assertIsNotNone(rule)

    def test_parity_rule_inconsistent_returns_none(self):
        """Inconsistent mapping returns None."""
        from arc_agent.dsl_synth import _learn_neighbor_rule_parity

        # Same key maps to different outputs
        inp1 = [[0, 0], [0, 0]]
        out1 = [[1, 0], [0, 1]]
        inp2 = [[0, 0], [0, 0]]
        out2 = [[2, 0], [0, 2]]  # conflicts with out1

        rule = _learn_neighbor_rule_parity([inp1, inp2], [out1, out2])
        self.assertIsNone(rule)


if __name__ == "__main__":
    unittest.main()
