"""Unit tests for DSL primitives (grid transformations)."""
import unittest
from arc_agent.primitives import (
    rotate_90_cw, rotate_90_ccw, rotate_180,
    mirror_horizontal, mirror_vertical, transpose, identity,
    crop_to_nonzero, tile_2x2, scale_2x, scale_3x,
    gravity_down, gravity_up, gravity_left, gravity_right,
    flood_fill_background, outline, fill_enclosed, invert_colors,
    extract_unique_colors, count_nonzero_per_row,
    is_symmetric_h, is_symmetric_v, is_square, has_single_color,
    build_initial_toolkit,
)


class TestGeometricTransforms(unittest.TestCase):
    def test_rotate_90_cw(self):
        grid = [[1, 2], [3, 4]]
        result = rotate_90_cw(grid)
        self.assertEqual(result, [[3, 1], [4, 2]])

    def test_rotate_90_ccw(self):
        grid = [[1, 2], [3, 4]]
        result = rotate_90_ccw(grid)
        self.assertEqual(result, [[2, 4], [1, 3]])

    def test_rotate_180(self):
        grid = [[1, 2], [3, 4]]
        result = rotate_180(grid)
        self.assertEqual(result, [[4, 3], [2, 1]])

    def test_rotate_360_is_identity(self):
        grid = [[1, 2, 3], [4, 5, 6]]
        r1 = rotate_90_cw(grid)
        r2 = rotate_90_cw(r1)
        r3 = rotate_90_cw(r2)
        r4 = rotate_90_cw(r3)
        self.assertEqual(r4, grid)

    def test_mirror_horizontal(self):
        grid = [[1, 2, 3], [4, 5, 6]]
        result = mirror_horizontal(grid)
        self.assertEqual(result, [[3, 2, 1], [6, 5, 4]])

    def test_mirror_vertical(self):
        grid = [[1, 2], [3, 4]]
        result = mirror_vertical(grid)
        self.assertEqual(result, [[3, 4], [1, 2]])

    def test_transpose(self):
        grid = [[1, 2, 3], [4, 5, 6]]
        result = transpose(grid)
        self.assertEqual(result, [[1, 4], [2, 5], [3, 6]])

    def test_identity(self):
        grid = [[1, 2], [3, 4]]
        result = identity(grid)
        self.assertEqual(result, grid)
        # Ensure it's a copy, not the same object
        result[0][0] = 99
        self.assertEqual(grid[0][0], 1)

    def test_empty_grid(self):
        self.assertEqual(rotate_90_cw([]), [])
        self.assertEqual(mirror_horizontal([]), [])


class TestColorTransforms(unittest.TestCase):
    def test_invert_colors(self):
        grid = [[1, 0], [0, 2]]
        result = invert_colors(grid)
        self.assertEqual(result, [[0, 1], [1, 0]])

    def test_extract_unique_colors(self):
        grid = [[1, 2, 1], [3, 0, 2]]
        result = extract_unique_colors(grid)
        self.assertEqual(result, [[1, 2, 3]])

    def test_extract_unique_colors_empty(self):
        grid = [[0, 0], [0, 0]]
        result = extract_unique_colors(grid)
        self.assertEqual(result, [[0]])


class TestSpatialTransforms(unittest.TestCase):
    def test_crop_to_nonzero(self):
        grid = [
            [0, 0, 0, 0],
            [0, 1, 2, 0],
            [0, 3, 4, 0],
            [0, 0, 0, 0],
        ]
        result = crop_to_nonzero(grid)
        self.assertEqual(result, [[1, 2], [3, 4]])

    def test_crop_single_cell(self):
        grid = [[0, 0], [0, 5]]
        result = crop_to_nonzero(grid)
        self.assertEqual(result, [[5]])

    def test_crop_all_zero(self):
        grid = [[0, 0], [0, 0]]
        result = crop_to_nonzero(grid)
        self.assertEqual(result, [[0]])

    def test_tile_2x2(self):
        grid = [[1, 2], [3, 4]]
        result = tile_2x2(grid)
        self.assertEqual(len(result), 4)
        self.assertEqual(len(result[0]), 4)
        self.assertEqual(result[0], [1, 2, 1, 2])
        self.assertEqual(result[2], [1, 2, 1, 2])

    def test_scale_2x(self):
        grid = [[1, 2], [3, 4]]
        result = scale_2x(grid)
        expected = [
            [1, 1, 2, 2],
            [1, 1, 2, 2],
            [3, 3, 4, 4],
            [3, 3, 4, 4],
        ]
        self.assertEqual(result, expected)

    def test_scale_3x(self):
        grid = [[5]]
        result = scale_3x(grid)
        expected = [[5, 5, 5], [5, 5, 5], [5, 5, 5]]
        self.assertEqual(result, expected)


class TestGravity(unittest.TestCase):
    def test_gravity_down(self):
        grid = [[1, 0, 0], [0, 2, 0], [0, 0, 3]]
        result = gravity_down(grid)
        self.assertEqual(result, [[0, 0, 0], [0, 0, 0], [1, 2, 3]])

    def test_gravity_up(self):
        grid = [[0, 0, 0], [0, 0, 0], [1, 2, 3]]
        result = gravity_up(grid)
        self.assertEqual(result, [[1, 2, 3], [0, 0, 0], [0, 0, 0]])

    def test_gravity_left(self):
        grid = [[0, 0, 1], [0, 2, 0], [3, 0, 0]]
        result = gravity_left(grid)
        self.assertEqual(result, [[1, 0, 0], [2, 0, 0], [3, 0, 0]])

    def test_gravity_right(self):
        grid = [[1, 0, 0], [0, 2, 0], [0, 0, 3]]
        result = gravity_right(grid)
        self.assertEqual(result, [[0, 0, 1], [0, 0, 2], [0, 0, 3]])


class TestFillOperations(unittest.TestCase):
    def test_fill_enclosed(self):
        grid = [
            [1, 1, 1],
            [1, 0, 1],
            [1, 1, 1],
        ]
        result = fill_enclosed(grid)
        self.assertEqual(result, [[1, 1, 1], [1, 1, 1], [1, 1, 1]])

    def test_fill_enclosed_not_border_connected(self):
        grid = [
            [2, 2, 2, 0],
            [2, 0, 2, 0],
            [2, 2, 2, 0],
        ]
        result = fill_enclosed(grid)
        # Interior 0 is enclosed, border 0s are not
        self.assertEqual(result[1][1], 2)
        self.assertEqual(result[0][3], 0)

    def test_outline(self):
        grid = [
            [1, 1, 1, 1],
            [1, 1, 1, 1],
            [1, 1, 1, 1],
            [1, 1, 1, 1],
        ]
        result = outline(grid)
        expected = [
            [1, 1, 1, 1],
            [1, 0, 0, 1],
            [1, 0, 0, 1],
            [1, 1, 1, 1],
        ]
        self.assertEqual(result, expected)

    def test_flood_fill_background(self):
        grid = [
            [0, 0, 1],
            [0, 1, 1],
            [1, 1, 1],
        ]
        result = flood_fill_background(grid)
        self.assertEqual(result[0][0], 1)
        self.assertEqual(result[0][1], 1)
        self.assertEqual(result[1][0], 1)


class TestPredicates(unittest.TestCase):
    def test_is_symmetric_h(self):
        self.assertIs(is_symmetric_h([[1, 2, 1], [3, 4, 3]]), True)
        self.assertIs(is_symmetric_h([[1, 2, 3]]), False)

    def test_is_symmetric_v(self):
        self.assertIs(is_symmetric_v([[1, 2], [3, 4], [1, 2]]), True)
        self.assertIs(is_symmetric_v([[1, 2], [3, 4]]), False)

    def test_is_square(self):
        self.assertIs(is_square([[1, 2], [3, 4]]), True)
        self.assertIs(is_square([[1, 2, 3], [4, 5, 6]]), False)

    def test_has_single_color(self):
        self.assertIs(has_single_color([[1, 0, 1], [0, 1, 0]]), True)
        self.assertIs(has_single_color([[1, 2], [3, 4]]), False)
        self.assertIs(has_single_color([[0, 0], [0, 0]]), True)


class TestBuildToolkit(unittest.TestCase):
    def test_toolkit_has_primitives(self):
        tk = build_initial_toolkit()
        self.assertGreater(tk.size, 20)
        self.assertIn("rotate_90_cw", tk.concepts)
        self.assertIn("mirror_h", tk.concepts)
        self.assertIn("gravity_down", tk.concepts)
        self.assertIn("identity", tk.concepts)

    def test_toolkit_has_color_swaps(self):
        tk = build_initial_toolkit()
        self.assertIn("swap_1_to_2", tk.concepts)
        self.assertIn("swap_2_to_1", tk.concepts)

    def test_toolkit_has_recolor_ops(self):
        tk = build_initial_toolkit()
        self.assertIn("recolor_to_1", tk.concepts)
        self.assertIn("recolor_to_5", tk.concepts)

    def test_all_concepts_are_callable(self):
        tk = build_initial_toolkit()
        grid = [[1, 2], [3, 4]]
        for name, concept in tk.concepts.items():
            # Every concept should be able to process a grid without crashing
            result = concept.apply(grid)
            # Result should be a list of lists or None
            self.assertTrue(
                result is None or isinstance(result, list),
                f"Concept {name} returned {type(result)}"
            )


if __name__ == '__main__':
    unittest.main()
