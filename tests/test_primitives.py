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
    complete_symmetry_h, complete_symmetry_v, complete_symmetry_4,
    denoise_3x3, denoise_5x5,
    xor_halves_v, or_halves_v, and_halves_v,
    xor_halves_h, or_halves_h, and_halves_h,
    swap_most_least, recolor_least_common,
    repeat_rows_2x, repeat_cols_2x,
    stack_with_mirror_v, stack_with_mirror_h,
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


class TestGridPartitioning(unittest.TestCase):
    """Tests for grid partitioning and pattern primitives (v0.3)."""

    def test_get_top_half(self):
        from arc_agent.primitives import get_top_half
        grid = [[1, 2], [3, 4], [5, 6], [7, 8]]
        self.assertEqual(get_top_half(grid), [[1, 2], [3, 4]])

    def test_get_bottom_half(self):
        from arc_agent.primitives import get_bottom_half
        grid = [[1, 2], [3, 4], [5, 6], [7, 8]]
        self.assertEqual(get_bottom_half(grid), [[5, 6], [7, 8]])

    def test_get_left_half(self):
        from arc_agent.primitives import get_left_half
        grid = [[1, 2, 3, 4], [5, 6, 7, 8]]
        self.assertEqual(get_left_half(grid), [[1, 2], [5, 6]])

    def test_get_right_half(self):
        from arc_agent.primitives import get_right_half
        grid = [[1, 2, 3, 4], [5, 6, 7, 8]]
        self.assertEqual(get_right_half(grid), [[3, 4], [7, 8]])

    def test_get_border(self):
        from arc_agent.primitives import get_border
        grid = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        result = get_border(grid)
        self.assertEqual(result, [[1, 2, 3], [4, 0, 6], [7, 8, 9]])

    def test_get_interior(self):
        from arc_agent.primitives import get_interior
        grid = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        self.assertEqual(get_interior(grid), [[5]])

    def test_get_interior_small_grid(self):
        from arc_agent.primitives import get_interior
        grid = [[1, 2], [3, 4]]
        # Too small for interior, returns copy
        self.assertEqual(get_interior(grid), [[1, 2], [3, 4]])

    def test_replace_color(self):
        from arc_agent.primitives import replace_color
        grid = [[1, 2, 1], [2, 1, 2]]
        self.assertEqual(replace_color(grid, 1, 3), [[3, 2, 3], [2, 3, 2]])

    def test_most_common_color(self):
        from arc_agent.primitives import most_common_color
        grid = [[1, 2, 1], [1, 2, 0]]
        self.assertEqual(most_common_color(grid), 1)

    def test_least_common_color(self):
        from arc_agent.primitives import least_common_color
        grid = [[1, 2, 1], [1, 2, 0]]
        self.assertEqual(least_common_color(grid), 2)

    def test_most_common_color_empty(self):
        from arc_agent.primitives import most_common_color
        self.assertEqual(most_common_color([[0, 0], [0, 0]]), 0)

    def test_recolor_to_most_common(self):
        from arc_agent.primitives import recolor_to_most_common
        grid = [[1, 2, 1], [1, 0, 3]]
        result = recolor_to_most_common(grid)
        self.assertEqual(result, [[1, 1, 1], [1, 0, 1]])

    def test_deduplicate_rows(self):
        from arc_agent.primitives import deduplicate_rows
        grid = [[1, 2], [1, 2], [3, 4], [3, 4], [3, 4]]
        self.assertEqual(deduplicate_rows(grid), [[1, 2], [3, 4]])

    def test_deduplicate_cols(self):
        from arc_agent.primitives import deduplicate_cols
        grid = [[1, 1, 2], [3, 3, 4]]
        self.assertEqual(deduplicate_cols(grid), [[1, 2], [3, 4]])

    def test_sort_rows_by_color_count(self):
        from arc_agent.primitives import sort_rows_by_color_count
        grid = [[1, 2, 3], [0, 0, 1], [1, 0, 0]]
        result = sort_rows_by_color_count(grid)
        # Sorted by number of non-zero: [0,0,1] and [1,0,0] have 1 each, [1,2,3] has 3
        self.assertEqual(result[-1], [1, 2, 3])

    def test_reverse_rows(self):
        from arc_agent.primitives import reverse_rows
        grid = [[1, 2], [3, 4], [5, 6]]
        self.assertEqual(reverse_rows(grid), [[5, 6], [3, 4], [1, 2]])

    def test_reverse_cols(self):
        from arc_agent.primitives import reverse_cols
        grid = [[1, 2, 3], [4, 5, 6]]
        self.assertEqual(reverse_cols(grid), [[3, 2, 1], [6, 5, 4]])

    def test_upscale_to_max_small(self):
        from arc_agent.primitives import upscale_to_max
        grid = [[1, 2], [3, 4]]
        result = upscale_to_max(grid)
        # 2x2 → 3x scale → 6x6
        self.assertEqual(len(result), 6)
        self.assertEqual(len(result[0]), 6)

    def test_upscale_to_max_medium(self):
        from arc_agent.primitives import upscale_to_max
        grid = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 1, 2, 3], [4, 5, 6, 7]]
        result = upscale_to_max(grid)
        # 4x4 → 2x scale → 8x8
        self.assertEqual(len(result), 8)

    def test_upscale_to_max_large(self):
        from arc_agent.primitives import upscale_to_max
        grid = [[i for i in range(10)] for _ in range(10)]
        result = upscale_to_max(grid)
        # 10x10 → no scale, returns copy
        self.assertEqual(len(result), 10)


class TestToolkitNewPrimitives(unittest.TestCase):
    """Verify new primitives are registered in the toolkit."""

    def test_toolkit_has_partitioning_ops(self):
        tk = build_initial_toolkit()
        for name in ["get_top_half", "get_bottom_half", "get_left_half",
                      "get_right_half", "get_border", "get_interior"]:
            self.assertIn(name, tk.concepts, f"Missing: {name}")

    def test_toolkit_has_pattern_ops(self):
        tk = build_initial_toolkit()
        for name in ["recolor_to_most_common", "deduplicate_rows",
                      "deduplicate_cols", "reverse_rows", "reverse_cols",
                      "sort_rows_by_color_count", "upscale_to_max"]:
            self.assertIn(name, tk.concepts, f"Missing: {name}")

    def test_toolkit_has_fill_bg_ops(self):
        tk = build_initial_toolkit()
        for color in range(1, 10):
            self.assertIn(f"fill_bg_{color}", tk.concepts)

    def test_toolkit_has_erase_ops(self):
        tk = build_initial_toolkit()
        for color in range(1, 10):
            self.assertIn(f"erase_{color}", tk.concepts)

    def test_toolkit_size_increased(self):
        tk = build_initial_toolkit()
        # Was 73 in v0.2, should now be 73 + 13 partitioning + 9 fill_bg + 9 erase = 104+
        self.assertGreaterEqual(tk.size, 100)


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
            # Skip predicates (they return booleans, not grids)
            if concept.kind == "predicate":
                continue
            # Every non-predicate concept should be able to process a grid
            result = concept.apply(grid)
            # Result should be a list of lists or None
            self.assertTrue(
                result is None or isinstance(result, list),
                f"Concept {name} returned {type(result)}"
            )


class TestSymmetryCompletion(unittest.TestCase):
    def test_complete_symmetry_h_basic(self):
        # Left has content, right is empty -> mirror left onto right
        grid = [[1, 2, 0, 0]]
        result = complete_symmetry_h(grid)
        self.assertEqual(result, [[1, 2, 2, 1]])

    def test_complete_symmetry_h_right_heavier(self):
        # Right has more content -> mirror right onto left
        grid = [[0, 0, 3, 4]]
        result = complete_symmetry_h(grid)
        self.assertEqual(result, [[4, 3, 3, 4]])

    def test_complete_symmetry_v_basic(self):
        # Top has content, bottom is empty
        grid = [[1, 2], [3, 4], [0, 0], [0, 0]]
        result = complete_symmetry_v(grid)
        self.assertEqual(result, [[1, 2], [3, 4], [3, 4], [1, 2]])

    def test_complete_symmetry_4(self):
        grid = [[1, 0], [0, 0]]
        result = complete_symmetry_4(grid)
        # After H: [[1,1],[0,0]], after V: [[1,1],[1,1]]
        self.assertEqual(result, [[1, 1], [1, 1]])


class TestDenoise(unittest.TestCase):
    def test_denoise_3x3_fixes_noise(self):
        # Single noisy pixel in a field of 1s
        grid = [
            [1, 1, 1],
            [1, 0, 1],  # center pixel is "noise"
            [1, 1, 1],
        ]
        result = denoise_3x3(grid)
        # Majority of 3x3 neighborhood is 1 -> center becomes 1
        self.assertEqual(result[1][1], 1)

    def test_denoise_3x3_preserves_real_boundary(self):
        # Large region boundary should not be erased
        grid = [
            [1, 1, 0],
            [1, 1, 0],
            [0, 0, 0],
        ]
        result = denoise_3x3(grid)
        self.assertEqual(result[0][0], 1)
        self.assertEqual(result[2][2], 0)


class TestGridOverlay(unittest.TestCase):
    def test_xor_halves_v(self):
        grid = [
            [1, 0, 2],
            [0, 3, 0],
            [1, 0, 0],  # bottom half
            [0, 0, 2],
        ]
        result = xor_halves_v(grid)
        # Row 0: xor([1,0,2], [1,0,0]) -> [0,0,2]
        # Row 1: xor([0,3,0], [0,0,2]) -> [0,3,2]
        self.assertEqual(result, [[0, 0, 2], [0, 3, 2]])

    def test_or_halves_v(self):
        grid = [
            [1, 0],
            [0, 2],
        ]
        result = or_halves_v(grid)
        # or([1,0], [0,2]) -> [1,2]
        self.assertEqual(result, [[1, 2]])

    def test_and_halves_h(self):
        grid = [
            [1, 0, 1, 2],
        ]
        result = and_halves_h(grid)
        # and([1,0], [1,2]) -> [1,0]
        self.assertEqual(result, [[1, 0]])

    def test_xor_halves_h(self):
        grid = [
            [1, 0, 0, 2],
        ]
        result = xor_halves_h(grid)
        # xor([1,0], [0,2]) -> [1,2]
        self.assertEqual(result, [[1, 2]])


class TestColorFrequency(unittest.TestCase):
    def test_swap_most_least(self):
        # Color 1 appears 3x, color 2 appears 1x
        grid = [[1, 1, 1], [2, 0, 0]]
        result = swap_most_least(grid)
        self.assertEqual(result, [[2, 2, 2], [1, 0, 0]])

    def test_recolor_least_common(self):
        grid = [[1, 1, 1], [2, 0, 0]]
        result = recolor_least_common(grid)
        self.assertEqual(result, [[1, 1, 1], [1, 0, 0]])


class TestPatternStacking(unittest.TestCase):
    def test_repeat_rows_2x(self):
        grid = [[1, 2], [3, 4]]
        result = repeat_rows_2x(grid)
        self.assertEqual(len(result), 4)
        self.assertEqual(result, [[1, 2], [3, 4], [1, 2], [3, 4]])

    def test_repeat_cols_2x(self):
        grid = [[1, 2], [3, 4]]
        result = repeat_cols_2x(grid)
        self.assertEqual(result, [[1, 2, 1, 2], [3, 4, 3, 4]])

    def test_stack_with_mirror_v(self):
        grid = [[1, 2], [3, 4]]
        result = stack_with_mirror_v(grid)
        self.assertEqual(result, [[1, 2], [3, 4], [3, 4], [1, 2]])

    def test_stack_with_mirror_h(self):
        grid = [[1, 2, 3]]
        result = stack_with_mirror_h(grid)
        self.assertEqual(result, [[1, 2, 3, 3, 2, 1]])


class TestNewToolkitSize(unittest.TestCase):
    def test_toolkit_has_new_ops(self):
        tk = build_initial_toolkit()
        self.assertGreaterEqual(tk.size, 140)
        # Check new ops are registered
        for name in ["complete_symmetry_h", "denoise_3x3", "xor_halves_v",
                     "swap_most_least", "repeat_rows_2x", "stack_with_mirror_v",
                     "recolor_smallest_to_1", "recolor_all_to_most_common_obj"]:
            self.assertIn(name, tk.concepts, f"Missing: {name}")


if __name__ == '__main__':
    unittest.main()
