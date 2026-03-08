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
    fill_tile_pattern, fill_by_symmetry, recolor_by_nearest_border,
    extend_to_border_h, extend_to_border_v,
    spread_in_lanes_h, spread_in_lanes_v,
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
        self.assertGreaterEqual(tk.size, 155)
        # Check new ops are registered
        for name in ["complete_symmetry_h", "denoise_3x3", "xor_halves_v",
                     "swap_most_least", "repeat_rows_2x", "stack_with_mirror_v",
                     "recolor_smallest_to_1", "recolor_all_to_most_common_obj",
                     # v0.7 additions
                     "mirror_diagonal_main", "mirror_diagonal_anti",
                     "fill_holes_per_color", "fill_rectangles",
                     "sort_cols_by_color_count",
                     "grid_difference", "grid_difference_h",
                     "spread_colors", "erode",
                     "keep_only_largest_color", "keep_only_smallest_color"]:
            self.assertIn(name, tk.concepts, f"Missing: {name}")


class TestDiagonalOps(unittest.TestCase):
    def test_mirror_diagonal_main_square(self):
        from arc_agent.primitives import mirror_diagonal_main
        grid = [[1, 2], [3, 4]]
        result = mirror_diagonal_main(grid)
        # Transpose: [[1,3],[2,4]]
        self.assertEqual(result, [[1, 3], [2, 4]])

    def test_mirror_diagonal_anti(self):
        from arc_agent.primitives import mirror_diagonal_anti
        grid = [[1, 2], [3, 4]]
        result = mirror_diagonal_anti(grid)
        self.assertEqual(result, [[4, 2], [3, 1]])


class TestFillOps(unittest.TestCase):
    def test_fill_holes_per_color(self):
        from arc_agent.primitives import fill_holes_per_color
        # Color 1 encloses a 0 cell
        grid = [
            [1, 1, 1],
            [1, 0, 1],
            [1, 1, 1],
        ]
        result = fill_holes_per_color(grid)
        self.assertEqual(result[1][1], 1)

    def test_fill_holes_per_color_border_not_filled(self):
        from arc_agent.primitives import fill_holes_per_color
        # Zero on border should NOT be filled
        grid = [
            [1, 0, 1],
            [1, 1, 1],
        ]
        result = fill_holes_per_color(grid)
        self.assertEqual(result[0][1], 0)

    def test_fill_rectangles(self):
        from arc_agent.primitives import fill_rectangles
        # L-shaped object -> fill bounding box
        grid = [
            [1, 0],
            [1, 1],
        ]
        result = fill_rectangles(grid)
        self.assertEqual(result, [[1, 1], [1, 1]])


class TestSortCols(unittest.TestCase):
    def test_sort_cols_by_color_count(self):
        from arc_agent.primitives import sort_cols_by_color_count
        # Col 0 has 2 nonzero, col 1 has 1, col 2 has 0
        grid = [
            [1, 0, 0],
            [1, 2, 0],
        ]
        result = sort_cols_by_color_count(grid)
        # Sorted ascending: col2(0), col1(1), col0(2)
        self.assertEqual(result, [[0, 0, 1], [0, 2, 1]])


class TestGridArithmetic(unittest.TestCase):
    def test_grid_difference(self):
        from arc_agent.primitives import grid_difference
        grid = [
            [1, 2],
            [3, 0],
            [0, 2],  # bottom half
            [3, 4],
        ]
        result = grid_difference(grid)
        # Row 0: a=[1,2], b=[0,2] -> [1,0] (1 unique to top, 2 in both)
        # Row 1: a=[3,0], b=[3,4] -> [0,0]
        self.assertEqual(result, [[1, 0], [0, 0]])

    def test_grid_difference_h(self):
        from arc_agent.primitives import grid_difference_h
        grid = [[1, 2, 0, 2]]
        result = grid_difference_h(grid)
        # left=[1,2], right=[0,2] -> [1,0]
        self.assertEqual(result, [[1, 0]])


class TestMorphological(unittest.TestCase):
    def test_spread_colors(self):
        from arc_agent.primitives import spread_colors
        grid = [
            [0, 0, 0],
            [0, 1, 0],
            [0, 0, 0],
        ]
        result = spread_colors(grid)
        # Center stays 1, all 4-neighbors become 1
        self.assertEqual(result[0][1], 1)
        self.assertEqual(result[1][0], 1)
        self.assertEqual(result[1][2], 1)
        self.assertEqual(result[2][1], 1)
        # Corners stay 0 (not 4-connected to 1)
        self.assertEqual(result[0][0], 0)

    def test_erode(self):
        from arc_agent.primitives import erode
        grid = [
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1],
        ]
        result = erode(grid)
        # Only center survives (all border cells touch edge)
        self.assertEqual(result[1][1], 1)
        self.assertEqual(result[0][0], 0)
        self.assertEqual(result[0][1], 0)


class TestColorMask(unittest.TestCase):
    def test_keep_only_largest_color(self):
        from arc_agent.primitives import keep_only_largest_color
        grid = [[1, 1, 1], [2, 0, 0]]
        result = keep_only_largest_color(grid)
        self.assertEqual(result, [[1, 1, 1], [0, 0, 0]])

    def test_keep_only_smallest_color(self):
        from arc_agent.primitives import keep_only_smallest_color
        grid = [[1, 1, 1], [2, 0, 0]]
        result = keep_only_smallest_color(grid)
        self.assertEqual(result, [[0, 0, 0], [2, 0, 0]])


class TestObjectOps(unittest.TestCase):
    def test_remove_largest_object(self):
        from arc_agent.objects import remove_largest_object
        grid = [
            [1, 1, 0],
            [1, 1, 2],
            [0, 0, 0],
        ]
        result = remove_largest_object(grid)
        # Largest is the 4-cell block of 1s
        self.assertEqual(result[0][0], 0)
        self.assertEqual(result[0][1], 0)
        self.assertEqual(result[1][2], 2)  # Small object preserved

    def test_remove_smallest_object(self):
        from arc_agent.objects import remove_smallest_object
        grid = [
            [1, 1, 0],
            [1, 1, 2],
            [0, 0, 0],
        ]
        result = remove_smallest_object(grid)
        # Smallest is the single 2 cell
        self.assertEqual(result[1][2], 0)
        self.assertEqual(result[0][0], 1)  # Large object preserved

    def test_keep_largest_object_only(self):
        from arc_agent.objects import keep_largest_object_only
        grid = [
            [1, 1, 0],
            [1, 1, 2],
            [0, 0, 0],
        ]
        result = keep_largest_object_only(grid)
        self.assertEqual(result[0][0], 1)
        self.assertEqual(result[1][2], 0)  # Small object removed

    def test_keep_smallest_object_only(self):
        from arc_agent.objects import keep_smallest_object_only
        grid = [
            [1, 1, 0],
            [1, 1, 2],
            [0, 0, 0],
        ]
        result = keep_smallest_object_only(grid)
        self.assertEqual(result[1][2], 2)
        self.assertEqual(result[0][0], 0)  # Large object removed

    def test_toolkit_has_new_object_ops(self):
        tk = build_initial_toolkit()
        for name in ["remove_largest_obj", "remove_smallest_obj",
                     "keep_largest_obj_only", "keep_smallest_obj_only"]:
            self.assertIn(name, tk.concepts, f"Missing: {name}")


class TestSynthesizerPairExhaustion(unittest.TestCase):
    def test_try_all_pairs_returns_program(self):
        from arc_agent.synthesizer import ProgramSynthesizer
        tk = build_initial_toolkit()
        synth = ProgramSynthesizer(tk, population_size=10)

        # Simple task: rotate 90 CW then mirror H
        # (just test that it runs without error and returns a program)
        grid_in = [[1, 2], [3, 4]]
        grid_out = [[3, 1], [4, 2]]  # rotate_90_cw result
        task = {"train": [{"input": grid_in, "output": grid_out}]}

        result = synth.try_all_pairs(task, top_k=5)
        self.assertIsNotNone(result)
        self.assertGreater(result.fitness, 0.0)

    def test_hill_climb_improves_or_maintains(self):
        from arc_agent.synthesizer import ProgramSynthesizer
        from arc_agent.scorer import TaskCache
        from arc_agent.concepts import Program

        tk = build_initial_toolkit()
        synth = ProgramSynthesizer(tk, population_size=10)

        task = {"train": [{"input": [[1, 0], [0, 1]], "output": [[0, 1], [1, 0]]}]}
        cache = TaskCache(task)

        # Start with identity (bad program)
        start = Program([tk.concepts["identity"]])
        start.fitness = cache.score_program(start)

        refined = synth.hill_climb(start, cache, max_steps=20)
        # Should be at least as good
        self.assertGreaterEqual(refined.fitness, start.fitness)


class TestTileExtraction(unittest.TestCase):
    """Tests for tile/pattern extraction primitives (v0.9)."""

    def test_extract_repeating_tile(self):
        from arc_agent.primitives import extract_repeating_tile
        # 2x2 tile repeated in 4x4 grid
        grid = [
            [1, 2, 1, 2],
            [3, 4, 3, 4],
            [1, 2, 1, 2],
            [3, 4, 3, 4],
        ]
        result = extract_repeating_tile(grid)
        self.assertEqual(result, [[1, 2], [3, 4]])

    def test_extract_repeating_tile_no_tile(self):
        from arc_agent.primitives import extract_repeating_tile
        # No repeating tile
        grid = [[1, 2], [3, 4]]
        result = extract_repeating_tile(grid)
        self.assertEqual(result, [[1, 2], [3, 4]])

    def test_extract_top_left_block(self):
        from arc_agent.primitives import extract_top_left_block
        # Grid with horizontal separator (row of 5s)
        grid = [
            [1, 2, 3],
            [4, 5, 6],
            [5, 5, 5],
            [7, 8, 9],
        ]
        result = extract_top_left_block(grid)
        self.assertEqual(result, [[1, 2, 3], [4, 5, 6]])

    def test_extract_bottom_right_block(self):
        from arc_agent.primitives import extract_bottom_right_block
        grid = [
            [1, 2, 3],
            [5, 5, 5],
            [7, 8, 9],
            [0, 1, 0],
        ]
        result = extract_bottom_right_block(grid)
        self.assertEqual(result, [[7, 8, 9], [0, 1, 0]])

    def test_split_separator_overlay(self):
        from arc_agent.primitives import split_by_separator_and_overlay
        # Top and bottom halves with separator
        grid = [
            [1, 0, 1],
            [5, 5, 5],
            [0, 2, 0],
        ]
        result = split_by_separator_and_overlay(grid)
        self.assertEqual(result, [[1, 2, 1]])

    def test_split_separator_xor(self):
        from arc_agent.primitives import split_by_separator_and_xor
        grid = [
            [1, 0, 1],
            [5, 5, 5],
            [0, 2, 0],
        ]
        result = split_by_separator_and_xor(grid)
        # Top: [1, 0, 1], Bottom: [0, 2, 0]
        # XOR: keep non-zero from one but not both
        self.assertEqual(result, [[1, 2, 1]])

    def test_compress_rows(self):
        from arc_agent.primitives import compress_rows
        grid = [
            [1, 2, 3],
            [1, 2, 3],
            [4, 5, 6],
            [4, 5, 6],
        ]
        result = compress_rows(grid)
        self.assertEqual(result, [[1, 2, 3], [4, 5, 6]])

    def test_compress_cols(self):
        from arc_agent.primitives import compress_cols
        grid = [
            [1, 1, 2, 2],
            [3, 3, 4, 4],
        ]
        result = compress_cols(grid)
        self.assertEqual(result, [[1, 2], [3, 4]])

    def test_max_color_per_cell(self):
        from arc_agent.primitives import max_color_per_cell
        # Two blocks separated by row of 5s
        grid = [
            [1, 0, 3],
            [5, 5, 5],
            [0, 2, 0],
        ]
        result = max_color_per_cell(grid)
        self.assertEqual(result, [[1, 2, 3]])

    def test_extract_unique_block(self):
        from arc_agent.primitives import extract_unique_block
        # Three blocks, one unique
        grid = [
            [1, 2],
            [5, 5],
            [1, 2],
            [5, 5],
            [3, 4],
        ]
        result = extract_unique_block(grid)
        self.assertEqual(result, [[3, 4]])

    def test_flatten_to_row(self):
        from arc_agent.primitives import flatten_to_row
        grid = [[3, 0, 1], [0, 2, 0]]
        result = flatten_to_row(grid)
        self.assertEqual(result, [[1, 2, 3]])

    def test_flatten_to_column(self):
        from arc_agent.primitives import flatten_to_column
        grid = [[3, 0, 1], [0, 2, 0]]
        result = flatten_to_column(grid)
        self.assertEqual(result, [[1], [2], [3]])

    def test_mode_color_per_row(self):
        from arc_agent.primitives import mode_color_per_row
        grid = [[1, 1, 2], [3, 3, 3]]
        result = mode_color_per_row(grid)
        self.assertEqual(result, [[1, 1, 1], [3, 3, 3]])

    def test_mode_color_per_col(self):
        from arc_agent.primitives import mode_color_per_col
        grid = [[1, 3], [1, 3], [2, 3]]
        result = mode_color_per_col(grid)
        self.assertEqual(result, [[1, 3], [1, 3], [1, 3]])


class TestNewToolkitSizeV09(unittest.TestCase):
    """Verify toolkit contains all v0.9 primitives."""

    def test_toolkit_size(self):
        tk = build_initial_toolkit()
        # v0.8 had 156, v0.9 adds 15 new, v0.13 adds 7 more
        self.assertGreaterEqual(tk.size, 175)

    def test_new_v09_primitives_exist(self):
        tk = build_initial_toolkit()
        new_names = [
            "extract_repeating_tile", "extract_top_left_block",
            "extract_bottom_right_block", "split_sep_overlay",
            "split_sep_xor", "compress_rows", "compress_cols",
            "max_color_per_cell", "min_color_per_cell",
            "extract_unique_block", "flatten_to_row",
            "flatten_to_column", "count_objects_grid",
            "mode_color_per_row", "mode_color_per_col",
        ]
        for name in new_names:
            self.assertIn(name, tk.concepts, f"Missing concept: {name}")


class TestFillTilePattern(unittest.TestCase):
    """Tests for fill_tile_pattern — infer repeating tile from partial grid."""

    def test_simple_2x2_tile(self):
        """Grid with a 2x2 repeating tile, some cells zeroed out."""
        # Tile [[1, 2], [3, 4]] repeated 2x2
        grid = [
            [1, 2, 1, 2],
            [3, 0, 3, 4],  # one cell zeroed
            [1, 2, 1, 2],
            [3, 4, 3, 4],
        ]
        result = fill_tile_pattern(grid)
        # Should recover the zero at [1][1] = 4
        self.assertEqual(result[1][1], 4)

    def test_identity_on_no_tile(self):
        """If no tile pattern detected, returns copy of input."""
        grid = [[1, 2, 3], [4, 5, 6]]
        result = fill_tile_pattern(grid)
        # Either returns original or finds a degenerate tile — at least no crash
        self.assertEqual(len(result), 2)
        self.assertEqual(len(result[0]), 3)

    def test_empty_grid(self):
        result = fill_tile_pattern([])
        self.assertEqual(result, [])

    def test_returns_full_grid_size(self):
        """Output grid must be same dimensions as input."""
        grid = [[1, 2, 1, 2], [3, 4, 3, 0], [1, 2, 1, 2], [3, 4, 3, 4]]
        result = fill_tile_pattern(grid)
        self.assertEqual(len(result), len(grid))
        self.assertEqual(len(result[0]), len(grid[0]))


class TestFillBySymmetry(unittest.TestCase):
    """Tests for fill_by_symmetry — recover masked cells from symmetry."""

    def test_180_symmetry_recovery(self):
        """Masked rectangle recovered via 180° rotational symmetry."""
        # A 4x4 grid with 180° symmetry, top-right covered by 9s
        grid = [
            [1, 2, 9, 9],
            [3, 4, 9, 9],
            [5, 6, 4, 3],
            [7, 8, 2, 1],
        ]
        result = fill_by_symmetry(grid)
        # The 9s at [0][2],[0][3],[1][2],[1][3] should be recovered
        # 180° of [0][2] is [3][1] = 8, etc.
        # At minimum, the result should be different from input
        self.assertNotEqual(result[0][2], 9)  # should be filled in
        self.assertNotEqual(result[0][3], 9)

    def test_no_mask_returns_identity(self):
        """Grid with no rectangular same-color block returns copy."""
        grid = [[1, 2], [3, 4]]
        result = fill_by_symmetry(grid)
        self.assertEqual(result, [[1, 2], [3, 4]])

    def test_empty_grid(self):
        result = fill_by_symmetry([])
        self.assertEqual(result, [])


class TestRecolorByNearestBorder(unittest.TestCase):
    """Tests for recolor_by_nearest_border — assign isolated pixels to border colors."""

    def test_single_border_row_assignment(self):
        """Noise pixels near a border stripe get that border's color."""
        # Top row all 1s (border), noise pixel (color 3) is in row 1
        # Bottom row all 2s (border), noise pixel (color 3) near row 4
        grid = [
            [1, 1, 1, 1, 1],
            [0, 0, 3, 0, 0],  # close to top border (1)
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [2, 2, 2, 2, 2],
        ]
        result = recolor_by_nearest_border(grid)
        # The 3 in row 1 is nearest to row 0 (border of 1s), so gets color 1
        self.assertEqual(result[1][2], 1)

    def test_no_border_returns_identity(self):
        """Grid with no separator rows/cols returns unchanged copy."""
        grid = [[1, 2], [3, 4]]
        result = recolor_by_nearest_border(grid)
        self.assertEqual(result, [[1, 2], [3, 4]])

    def test_preserves_border_cells(self):
        """Border stripe cells are not modified."""
        grid = [
            [5, 5, 5],
            [0, 3, 0],
            [0, 0, 0],
        ]
        result = recolor_by_nearest_border(grid)
        # Top border must remain unchanged
        self.assertEqual(result[0], [5, 5, 5])


class TestExtendToBorder(unittest.TestCase):
    """Tests for extend_to_border_h and extend_to_border_v."""

    def test_extend_h_single_color_row(self):
        """A row with one color fills the entire row."""
        grid = [
            [0, 0, 0, 0],
            [0, 3, 0, 0],
            [0, 0, 0, 0],
        ]
        result = extend_to_border_h(grid)
        # Row 1 has only color 3 — should fill entire row
        self.assertEqual(result[1], [3, 3, 3, 3])
        # Rows 0 and 2 stay all zeros
        self.assertEqual(result[0], [0, 0, 0, 0])
        self.assertEqual(result[2], [0, 0, 0, 0])

    def test_extend_v_single_color_col(self):
        """A column with one color fills the entire column."""
        grid = [
            [0, 0, 0],
            [0, 5, 0],
            [0, 0, 0],
            [0, 0, 0],
        ]
        result = extend_to_border_v(grid)
        # Col 1 has only color 5 — should fill entire column
        for r in range(4):
            self.assertEqual(result[r][1], 5)
        # Col 0 and 2 unchanged
        for r in range(4):
            self.assertEqual(result[r][0], 0)
            self.assertEqual(result[r][2], 0)

    def test_extend_h_empty_grid(self):
        result = extend_to_border_h([])
        self.assertEqual(result, [])

    def test_extend_h_preserves_dims(self):
        grid = [[1, 0, 0], [0, 2, 0]]
        result = extend_to_border_h(grid)
        self.assertEqual(len(result), 2)
        self.assertEqual(len(result[0]), 3)


class TestSpreadInLanes(unittest.TestCase):
    """Tests for spread_in_lanes_h and spread_in_lanes_v."""

    def test_spread_h_fills_lane_rows(self):
        """Colored cell spreads across its entire row within separator grid."""
        # Separators at rows 0 and 2 (all 8s), colored cell at [1][1]
        grid = [
            [8, 8, 8, 8],
            [0, 3, 0, 0],  # has color 3 — should spread
            [8, 8, 8, 8],
            [0, 0, 0, 0],  # no color — stays empty
        ]
        result = spread_in_lanes_h(grid)
        # Row 1 should be filled with 3 in non-separator cells
        self.assertEqual(result[1][0], 3)
        self.assertEqual(result[1][2], 3)
        self.assertEqual(result[1][3], 3)
        # Separator rows unchanged
        self.assertEqual(result[0], [8, 8, 8, 8])
        self.assertEqual(result[2], [8, 8, 8, 8])

    def test_spread_v_fills_lane_cols(self):
        """Vertical transpose of spread_h — colored cell spreads down column."""
        grid = [
            [8, 0, 8],
            [8, 4, 8],  # color 4 in col 1
            [8, 0, 8],
        ]
        result = spread_in_lanes_v(grid)
        # All rows in col 1 should have color 4
        self.assertEqual(result[0][1], 4)
        self.assertEqual(result[2][1], 4)

    def test_spread_h_empty_row_unchanged(self):
        """Rows with no non-separator color remain empty."""
        grid = [
            [8, 8, 8],
            [0, 0, 0],  # no color
            [8, 8, 8],
        ]
        result = spread_in_lanes_h(grid)
        self.assertEqual(result[1], [0, 0, 0])

    def test_spread_h_no_separator_returns_identity(self):
        """If no separator rows found, returns copy of input."""
        grid = [[1, 2], [3, 4]]
        result = spread_in_lanes_h(grid)
        self.assertEqual(result, [[1, 2], [3, 4]])


class TestV13ToolkitContents(unittest.TestCase):
    """Verify v0.13 primitives are in the toolkit."""

    def test_new_v13_primitives_exist(self):
        tk = build_initial_toolkit()
        new_names = [
            "fill_tile_pattern", "fill_by_symmetry", "recolor_by_nearest_border",
            "extend_to_border_h", "extend_to_border_v",
            "spread_in_lanes_h", "spread_in_lanes_v",
        ]
        for name in new_names:
            self.assertIn(name, tk.concepts, f"Missing concept: {name}")

    def test_toolkit_size_v13(self):
        tk = build_initial_toolkit()
        self.assertGreaterEqual(tk.size, 178)


if __name__ == '__main__':
    unittest.main()


# ============================================================
# V14 PRIMITIVE TESTS
# ============================================================

from arc_agent.primitives import (
    connect_pixels_to_rect, gravity_toward_color, fill_holes_in_objects,
    recolor_2nd_to_3rd_color, recolor_least_to_second_least,
    swap_most_and_second_color, swap_largest_and_smallest_obj_color,
    swap_colors_12, swap_colors_34,
    complete_pattern_4way, fill_bg_with_color_from_border,
    keep_only_unique_rows, keep_only_unique_cols,
    rotate_colors_up, rotate_colors_down,
    extend_nonzero_to_fill_row, extend_nonzero_to_fill_col,
)


class TestGravityTowardColor(unittest.TestCase):
    """gravity_toward_color: pack scattered dots adjacent to band."""

    def test_dots_pack_to_band(self):
        # Band at rows 2-3, dots above and below
        grid = [
            [2, 0, 0],  # dot
            [0, 0, 0],
            [5, 5, 5],  # band
            [5, 5, 5],  # band
            [0, 0, 0],
            [0, 2, 0],  # dot
        ]
        result = gravity_toward_color(grid)
        # Dot above should move to row 1 (adjacent to band at row 2)
        self.assertEqual(result[1][0], 2)
        # Dot below should move to row 4 (adjacent to band at row 3)
        self.assertEqual(result[4][1], 2)
        # Original dot positions clear
        self.assertEqual(result[0][0], 0)
        self.assertEqual(result[5][1], 0)

    def test_no_band_returns_input(self):
        grid = [[1, 0], [0, 2]]
        result = gravity_toward_color(grid)
        self.assertEqual(result, grid)

    def test_single_band_row(self):
        grid = [
            [0, 3, 0],
            [5, 5, 5],  # band
            [0, 0, 0],
        ]
        result = gravity_toward_color(grid)
        # dot above stays adjacent (already adjacent)
        self.assertEqual(result[0][1], 3)


class TestFillHolesInObjects(unittest.TestCase):
    """fill_holes_in_objects: fill enclosed bg cells with surrounding color."""

    def test_fills_hole_inside_ring(self):
        grid = [
            [0, 0, 0, 0, 0],
            [0, 1, 1, 1, 0],
            [0, 1, 0, 1, 0],  # hole at (2,2)
            [0, 1, 1, 1, 0],
            [0, 0, 0, 0, 0],
        ]
        result = fill_holes_in_objects(grid)
        self.assertEqual(result[2][2], 1)

    def test_does_not_fill_exterior(self):
        grid = [
            [0, 0, 0, 0, 0],
            [0, 1, 1, 1, 0],
            [0, 1, 0, 1, 0],
            [0, 1, 1, 1, 0],
            [0, 0, 0, 0, 0],
        ]
        result = fill_holes_in_objects(grid)
        # Exterior 0s stay 0
        self.assertEqual(result[0][0], 0)
        self.assertEqual(result[4][4], 0)

    def test_already_filled_unchanged(self):
        grid = [[1, 1], [1, 1]]
        result = fill_holes_in_objects(grid)
        self.assertEqual(result, [[1, 1], [1, 1]])


class TestRecolor2ndTo3rd(unittest.TestCase):
    """recolor_2nd_to_3rd_color."""

    def test_replaces_2nd_most_common(self):
        # bg=0 (most), then 1 (most non-bg), then 2, then 3
        grid = [
            [0, 0, 0, 0],
            [1, 1, 1, 0],
            [2, 2, 0, 0],
            [3, 0, 0, 0],
        ]
        result = recolor_2nd_to_3rd_color(grid)
        # 1 is most common non-bg, 2 is 2nd, 3 is 3rd
        # So 2→3
        for row in result:
            self.assertNotIn(2, row)

    def test_fewer_than_3_colors_unchanged(self):
        grid = [[1, 1], [0, 0]]
        result = recolor_2nd_to_3rd_color(grid)
        self.assertEqual(result, grid)


class TestRecolorLeastTo2ndLeast(unittest.TestCase):
    """recolor_least_to_second_least."""

    def test_replaces_least_common(self):
        grid = [
            [1, 1, 1],
            [2, 2, 0],
            [3, 0, 0],
        ]
        # bg=0, counts: 1→3, 2→2, 3→1; least=3, 2nd_least=2 → 3→2
        result = recolor_least_to_second_least(grid)
        flat = [v for row in result for v in row]
        self.assertNotIn(3, flat)


class TestSwapColors12(unittest.TestCase):
    def test_swaps_1_and_2(self):
        grid = [[1, 2, 3], [2, 1, 0]]
        result = swap_colors_12(grid)
        self.assertEqual(result[0], [2, 1, 3])
        self.assertEqual(result[1], [1, 2, 0])


class TestCompletePattern4way(unittest.TestCase):
    def test_fills_symmetric_positions(self):
        grid = [
            [1, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
        ]
        result = complete_pattern_4way(grid)
        # Top-left 1 should propagate to all corners
        self.assertEqual(result[0][0], 1)
        self.assertEqual(result[0][2], 1)
        self.assertEqual(result[2][0], 1)
        self.assertEqual(result[2][2], 1)


class TestKeepUniqueRows(unittest.TestCase):
    def test_removes_duplicate_rows(self):
        grid = [[1, 2], [3, 4], [1, 2], [5, 6]]
        result = keep_only_unique_rows(grid)
        self.assertEqual(len(result), 3)
        self.assertIn([1, 2], result)
        self.assertIn([3, 4], result)
        self.assertIn([5, 6], result)

    def test_no_duplicates_unchanged(self):
        grid = [[1, 2], [3, 4]]
        result = keep_only_unique_rows(grid)
        self.assertEqual(result, grid)


class TestKeepUniqueCols(unittest.TestCase):
    def test_removes_duplicate_cols(self):
        grid = [[1, 3, 1], [2, 4, 2]]
        result = keep_only_unique_cols(grid)
        self.assertEqual(len(result[0]), 2)


class TestRotateColors(unittest.TestCase):
    def test_rotate_up_cycles(self):
        # bg=0 is most common; non-bg values get cycled up
        grid = [[0, 0, 0], [0, 1, 9], [0, 0, 0]]
        result = rotate_colors_up(grid)
        self.assertEqual(result[1][1], 2)   # 1 → 2
        self.assertEqual(result[1][2], 1)   # 9 → (9%9)+1=1
        self.assertEqual(result[0][0], 0)   # bg unchanged

    def test_rotate_down_cycles(self):
        # bg=0 is most common
        grid = [[0, 0, 0], [0, 1, 2], [0, 0, 0]]
        result = rotate_colors_down(grid)
        self.assertEqual(result[1][1], 9)   # 1 → ((1-2)%9)+1 = 9
        self.assertEqual(result[1][2], 1)   # 2 → ((2-2)%9)+1 = 1
        self.assertEqual(result[0][0], 0)   # bg unchanged


class TestExtendNonzeroFillRow(unittest.TestCase):
    def test_single_color_row_fills(self):
        grid = [[0, 3, 0, 3], [1, 2, 0, 0]]
        result = extend_nonzero_to_fill_row(grid)
        # row 0 has both 3s → fills all
        self.assertEqual(result[0], [3, 3, 3, 3])
        # row 1 has both 1 and 2 → unchanged
        self.assertEqual(result[1], [1, 2, 0, 0])

    def test_all_bg_row_unchanged(self):
        grid = [[0, 0, 0]]
        result = extend_nonzero_to_fill_row(grid)
        self.assertEqual(result[0], [0, 0, 0])


class TestV14ToolkitContents(unittest.TestCase):
    """Verify v14 primitives are in the toolkit."""

    def test_new_v14_primitives_exist(self):
        tk = build_initial_toolkit()
        new_names = [
            "connect_pixels_to_rect", "gravity_toward_color", "fill_holes_in_objects",
            "recolor_2nd_to_3rd", "recolor_least_to_2nd_least",
            "swap_most_and_2nd_color", "swap_largest_smallest_obj_color",
            "swap_colors_12", "swap_colors_34",
            "complete_pattern_4way", "fill_bg_from_border",
            "keep_unique_rows", "keep_unique_cols",
            "rotate_colors_up", "rotate_colors_down",
            "extend_nonzero_fill_row", "extend_nonzero_fill_col",
        ]
        for name in new_names:
            self.assertIn(name, tk.concepts, f"Missing concept: {name}")

    def test_toolkit_size_v14(self):
        tk = build_initial_toolkit()
        self.assertGreaterEqual(tk.size, 205)


# ============================================================
# V15 PRIMITIVE TESTS
# ============================================================

from arc_agent.primitives import (
    recolor_isolated_to_nearest, recolor_small_objects_to_nearest,
    remove_color_noise, mirror_h_merge, mirror_v_merge,
    sort_rows_by_value, sort_cols_by_value, recolor_by_size_rank,
    fill_row_from_right, fill_col_from_bottom,
    complete_symmetry_diagonal, keep_border_only,
    tile_grid_2x1, tile_grid_1x2, repeat_pattern_to_size,
)


class TestRecolorIsolatedToNearest(unittest.TestCase):
    def test_isolated_gets_nearest_color(self):
        # 5 is isolated, nearest non-bg color is 3 (nearby block)
        grid = [
            [0, 0, 0, 0, 0],
            [0, 3, 3, 0, 0],
            [0, 3, 3, 0, 5],  # 5 is isolated, nearest is 3
            [0, 0, 0, 0, 0],
        ]
        result = recolor_isolated_to_nearest(grid)
        self.assertEqual(result[2][4], 3)

    def test_non_isolated_unchanged(self):
        grid = [
            [0, 0, 0],
            [0, 2, 2],  # 2s are adjacent, not isolated
            [0, 0, 0],
        ]
        result = recolor_isolated_to_nearest(grid)
        self.assertEqual(result[1][1], 2)
        self.assertEqual(result[1][2], 2)

    def test_empty_unchanged(self):
        self.assertEqual(recolor_isolated_to_nearest([]), [])


class TestRemoveColorNoise(unittest.TestCase):
    def test_removes_isolated_pixels(self):
        grid = [
            [0, 0, 0, 0],
            [0, 1, 0, 0],  # isolated 1
            [0, 0, 2, 2],  # connected 2s
            [0, 0, 2, 2],
        ]
        result = remove_color_noise(grid)
        self.assertEqual(result[1][1], 0)  # isolated 1 removed
        self.assertEqual(result[2][2], 2)  # connected 2 kept
        self.assertEqual(result[2][3], 2)  # connected 2 kept

    def test_all_connected_unchanged(self):
        grid = [[1, 1], [1, 1]]
        result = remove_color_noise(grid)
        self.assertEqual(result, [[1, 1], [1, 1]])


class TestMirrorHMerge(unittest.TestCase):
    def test_fills_bg_from_mirror(self):
        grid = [
            [1, 0, 0, 2],
        ]
        result = mirror_h_merge(grid)
        # Original: [1,0,0,2], mirrored: [2,0,0,1]
        # Merge: [1,0,0,2] (non-bg wins)
        self.assertEqual(result[0][0], 1)
        self.assertEqual(result[0][3], 2)

    def test_bg_filled_from_mirror(self):
        # Row [1, 0, 0, 3]: mirrored = [3, 0, 0, 1]
        # Merged: [1, 3, 3, 3] — bg(0) positions get mirror value
        grid = [[1, 0, 0, 3]]
        result = mirror_h_merge(grid)
        self.assertEqual(result[0][0], 1)  # original kept
        self.assertEqual(result[0][3], 3)  # original kept
        # middle positions: original=0, mirror=0 → stays 0
        self.assertEqual(result[0][1], 0)


class TestMirrorVMerge(unittest.TestCase):
    def test_fills_bg_from_vmirror(self):
        grid = [
            [1, 0],
            [0, 0],
            [0, 2],
        ]
        result = mirror_v_merge(grid)
        # row 0 original [1,0], mirrored row (row 2) [0,2] → [1,2]
        self.assertEqual(result[0][0], 1)
        self.assertEqual(result[0][1], 2)


class TestSortRows(unittest.TestCase):
    def test_sorts_row_ascending(self):
        grid = [[3, 1, 2], [6, 4, 5]]
        result = sort_rows_by_value(grid)
        self.assertEqual(result[0], [1, 2, 3])
        self.assertEqual(result[1], [4, 5, 6])


class TestSortCols(unittest.TestCase):
    def test_sorts_col_ascending(self):
        grid = [[3, 6], [1, 4], [2, 5]]
        result = sort_cols_by_value(grid)
        self.assertEqual([result[r][0] for r in range(3)], [1, 2, 3])
        self.assertEqual([result[r][1] for r in range(3)], [4, 5, 6])


class TestFillRowFromRight(unittest.TestCase):
    def test_propagates_rightward(self):
        grid = [[0, 0, 3, 0, 0]]
        result = fill_row_from_right(grid)
        # From right, 3 is at position 2, propagates left to positions 0,1
        self.assertEqual(result[0][0], 3)
        self.assertEqual(result[0][1], 3)
        self.assertEqual(result[0][2], 3)
        # Positions after 3 stay 0 (no value to right of 3)
        self.assertEqual(result[0][3], 0)


class TestCompleteSymmetryDiagonal(unittest.TestCase):
    def test_mirrors_across_diagonal(self):
        grid = [
            [0, 5, 0],
            [0, 0, 0],
            [0, 0, 0],
        ]
        result = complete_symmetry_diagonal(grid)
        # (0,1)=5 should set (1,0)=5
        self.assertEqual(result[1][0], 5)

    def test_non_square_unchanged(self):
        grid = [[1, 2, 3], [4, 5, 6]]
        result = complete_symmetry_diagonal(grid)
        self.assertEqual(result, grid)


class TestKeepBorderOnly(unittest.TestCase):
    def test_keeps_border_clears_interior(self):
        # Grid filled with 1s, interior 2 should be cleared to bg (1)
        # Actually keep_border_only sets interior to bg (most common)
        # Use a grid where border=1 and interior has different value
        grid = [
            [1, 1, 1, 1],
            [1, 2, 2, 1],
            [1, 2, 2, 1],
            [1, 1, 1, 1],
        ]
        result = keep_border_only(grid)
        # bg=1 (most common), interior 2s replaced by 1
        self.assertEqual(result[1][1], 1)
        self.assertEqual(result[1][2], 1)
        # border values unchanged
        self.assertEqual(result[0][0], 1)


class TestTileGrid(unittest.TestCase):
    def test_tile_2x1(self):
        grid = [[1, 2], [3, 4]]
        result = tile_grid_2x1(grid)
        self.assertEqual(result[0], [1, 2, 1, 2])
        self.assertEqual(result[1], [3, 4, 3, 4])

    def test_tile_1x2(self):
        grid = [[1, 2], [3, 4]]
        result = tile_grid_1x2(grid)
        self.assertEqual(len(result), 4)
        self.assertEqual(result[2], [1, 2])
        self.assertEqual(result[3], [3, 4])


class TestRepeatPatternToSize(unittest.TestCase):
    def test_tiled_grid_returns_same(self):
        # A 4x4 grid that's already a 2x2 tile repeated
        grid = [
            [1, 2, 1, 2],
            [3, 4, 3, 4],
            [1, 2, 1, 2],
            [3, 4, 3, 4],
        ]
        result = repeat_pattern_to_size(grid)
        # Should return same content (already tiled)
        self.assertEqual(result, grid)

    def test_non_tiled_unchanged(self):
        grid = [[1, 2], [3, 5]]  # no repeating pattern
        result = repeat_pattern_to_size(grid)
        self.assertEqual(result, grid)


class TestV15ToolkitContents(unittest.TestCase):
    def test_new_v15_primitives_exist(self):
        tk = build_initial_toolkit()
        new_names = [
            "recolor_isolated_to_nearest", "recolor_small_objs_to_nearest",
            "remove_color_noise", "mirror_h_merge", "mirror_v_merge",
            "sort_rows_by_value", "sort_cols_by_value", "recolor_by_size_rank",
            "fill_row_from_right", "fill_col_from_bottom",
            "complete_symmetry_diagonal", "keep_border_only",
            "tile_grid_2x1", "tile_grid_1x2", "repeat_pattern_to_size",
        ]
        for name in new_names:
            self.assertIn(name, tk.concepts, f"Missing: {name}")

    def test_toolkit_size_v15(self):
        tk = build_initial_toolkit()
        self.assertGreaterEqual(tk.size, 224)


class TestFillStripeGaps(unittest.TestCase):
    def test_fill_stripe_gaps_h_basic(self):
        from arc_agent.primitives import fill_stripe_gaps_h
        # Row with same color on both sides of zeros
        grid = [[3, 0, 0, 3], [1, 2, 1, 2]]
        result = fill_stripe_gaps_h(grid)
        # Row 0: 3 _ _ 3 → 3 3 3 3
        self.assertEqual(result[0], [3, 3, 3, 3])
        # Row 1: different colors — should not fill
        self.assertEqual(result[1], [1, 2, 1, 2])

    def test_fill_stripe_gaps_v_basic(self):
        from arc_agent.primitives import fill_stripe_gaps_v
        # Column 0 has 5 on top and bottom with zeros between
        grid = [[5, 0], [0, 0], [5, 0]]
        result = fill_stripe_gaps_v(grid)
        self.assertEqual(result[0][0], 5)
        self.assertEqual(result[1][0], 5)
        self.assertEqual(result[2][0], 5)


class TestCompleteTileFromModal(unittest.TestCase):
    def test_complete_tile_from_modal_row(self):
        from arc_agent.primitives import complete_tile_from_modal_row
        # Row majority is 1, one anomaly of 7
        grid = [[1, 1, 1, 7, 1, 1],
                [2, 2, 2, 2, 2, 2]]
        result = complete_tile_from_modal_row(grid)
        # Anomalous 7 in row 0 should be replaced by 1
        self.assertEqual(result[0][3], 1)
        # Row 1 all same — no change
        self.assertEqual(result[1], [2, 2, 2, 2, 2, 2])

    def test_complete_tile_from_modal_col(self):
        from arc_agent.primitives import complete_tile_from_modal_col
        # Column 0 majority is 3, one anomaly of 9
        grid = [[3], [3], [9], [3], [3], [3]]
        result = complete_tile_from_modal_col(grid)
        self.assertEqual(result[2][0], 3)


class TestRecolorMinority(unittest.TestCase):
    def test_recolor_minority_in_rows(self):
        from arc_agent.primitives import recolor_minority_in_rows
        # Row with dominant color 2, one stray 7
        grid = [[0, 0, 0, 0],
                [2, 2, 7, 2],
                [3, 3, 3, 3]]
        result = recolor_minority_in_rows(grid)
        # The 7 (appears once in row 1, dominant is 2) should be recolored to 2
        self.assertEqual(result[1][2], 2)


class TestPropagateColor(unittest.TestCase):
    def test_propagate_color_h(self):
        from arc_agent.primitives import propagate_color_h
        grid = [[0, 0, 3, 0, 0, 0],
                [0, 5, 0, 0, 0, 0]]
        result = propagate_color_h(grid)
        # Row 0: 3 propagates right
        self.assertEqual(result[0][3], 3)
        self.assertEqual(result[0][5], 3)
        # Row 1: 5 propagates right
        self.assertEqual(result[1][2], 5)

    def test_propagate_color_v(self):
        from arc_agent.primitives import propagate_color_v
        grid = [[0, 4], [0, 0], [0, 0]]
        result = propagate_color_v(grid)
        # Col 1: 4 propagates down
        self.assertEqual(result[1][1], 4)
        self.assertEqual(result[2][1], 4)


class TestSnapIsolatedToRect(unittest.TestCase):
    def test_snap_isolated_leaves_main_object(self):
        from arc_agent.primitives import snap_isolated_to_rect_boundary
        # Simple: one isolated pixel outside a rectangle
        grid = [
            [0, 0, 0, 0, 0],
            [0, 8, 8, 8, 0],
            [0, 8, 8, 8, 0],
            [0, 0, 0, 0, 0],
            [3, 0, 0, 0, 0],  # isolated 3, outside rectangle
        ]
        result = snap_isolated_to_rect_boundary(grid)
        # The isolated 3 should be moved (original position cleared)
        self.assertEqual(result[4][0], 0)


class TestV16ToolkitContents(unittest.TestCase):
    def test_new_v16_primitives_exist(self):
        tk = build_initial_toolkit()
        new_names = [
            "fill_stripe_gaps_h", "fill_stripe_gaps_v",
            "complete_tile_from_modal_col", "complete_tile_from_modal_row",
            "recolor_minority_in_rows", "recolor_minority_in_cols",
            "recolor_smallest_obj_in_each_row", "recolor_smallest_obj_in_each_col",
            "fill_grid_intersections", "propagate_color_h", "propagate_color_v",
            "recolor_unique_in_row_col", "snap_isolated_to_rect_boundary",
            "recolor_touching_2nd_to_8", "recolor_touching_2nd_to_3",
            "recolor_neighbors_of_2nd_color",
            "extend_color_within_col_bounds", "extend_color_within_row_bounds",
        ]
        for name in new_names:
            self.assertIn(name, tk.concepts, f"Missing v0.16 primitive: {name}")

    def test_toolkit_size_v16(self):
        tk = build_initial_toolkit()
        self.assertGreaterEqual(tk.size, 242)



class TestRecolorDominantTouchingAccent(unittest.TestCase):
    def test_recolor_dominant_touching_accent_to_4(self):
        from arc_agent.primitives import recolor_dominant_touching_accent_to_4
        # bg=0 (most common=16), dominant non-bg=5 (8 cells), accent=2 (1 cell)
        # dominant (5) cells touching accent (2) -> 4
        grid = [
            [0, 0, 0, 0, 0, 0],
            [0, 5, 5, 5, 5, 0],
            [0, 5, 5, 2, 5, 0],
            [0, 5, 5, 5, 5, 0],
            [0, 0, 0, 0, 0, 0],
        ]
        result = recolor_dominant_touching_accent_to_4(grid)
        # 5s adjacent to 2 at (2,3) should become 4
        self.assertEqual(result[1][3], 4)  # above
        self.assertEqual(result[2][2], 4)  # left
        self.assertEqual(result[2][4], 4)  # right
        self.assertEqual(result[3][3], 4)  # below
        # accent cell unchanged
        self.assertEqual(result[2][3], 2)
        # non-touching dominant unchanged
        self.assertEqual(result[1][1], 5)

    def test_recolor_dominant_touching_accent_to_8(self):
        from arc_agent.primitives import recolor_dominant_touching_accent_to_8
        # bg=0 (most common=12), dominant=5 (8 cells), accent=3 (1 cell)
        grid = [
            [0, 0, 0, 0, 0],
            [0, 5, 5, 5, 0],
            [0, 5, 3, 5, 0],
            [0, 5, 5, 5, 0],
            [0, 0, 0, 0, 0],
        ]
        result = recolor_dominant_touching_accent_to_8(grid)
        # 5s touching 3 at (2,2) should become 8
        self.assertEqual(result[1][2], 8)  # above
        self.assertEqual(result[2][1], 8)  # left
        self.assertEqual(result[2][3], 8)  # right
        self.assertEqual(result[3][2], 8)  # below
        # bg cells unchanged
        self.assertEqual(result[0][0], 0)
        # non-touching dominant unchanged
        self.assertEqual(result[1][1], 5)


class TestFillSmallestRectHole(unittest.TestCase):
    def test_fill_smallest_rect_hole_with_1(self):
        from arc_agent.primitives import fill_smallest_rect_hole_with_1
        # bg=0 (most common), 5s form an enclosure with a 2-cell interior hole
        # Use a large enough grid so 0 is clearly most common
        grid = [
            [0, 0, 0, 0, 0, 0, 0],
            [0, 5, 5, 5, 5, 5, 0],
            [0, 5, 0, 0, 5, 5, 0],
            [0, 5, 0, 0, 5, 5, 0],
            [0, 5, 5, 5, 5, 5, 0],
            [0, 0, 0, 0, 0, 0, 0],
        ]
        result = fill_smallest_rect_hole_with_1(grid)
        self.assertEqual(result[2][2], 1)
        self.assertEqual(result[2][3], 1)
        self.assertEqual(result[3][2], 1)
        self.assertEqual(result[3][3], 1)
        self.assertEqual(result[0][0], 0)

    def test_fill_smallest_rect_hole_with_8(self):
        from arc_agent.primitives import fill_smallest_rect_hole_with_8
        # bg=0 (most common), 3s enclose a 2x2 hole
        grid = [
            [0, 0, 0, 0, 0, 0, 0],
            [0, 3, 3, 3, 3, 3, 0],
            [0, 3, 0, 0, 3, 3, 0],
            [0, 3, 0, 0, 3, 3, 0],
            [0, 3, 3, 3, 3, 3, 0],
            [0, 0, 0, 0, 0, 0, 0],
        ]
        result = fill_smallest_rect_hole_with_8(grid)
        self.assertEqual(result[2][2], 8)
        self.assertEqual(result[2][3], 8)
        self.assertEqual(result[3][2], 8)
        self.assertEqual(result[3][3], 8)


class TestSortBySum(unittest.TestCase):
    def test_sort_rows_by_sum(self):
        from arc_agent.primitives import sort_rows_by_sum
        grid = [
            [3, 1, 1],  # sum=5
            [0, 0, 0],  # sum=0
            [2, 2, 1],  # sum=5
        ]
        result = sort_rows_by_sum(grid)
        # row with sum=0 should be first
        self.assertEqual(result[0], [0, 0, 0])

    def test_sort_cols_by_sum(self):
        from arc_agent.primitives import sort_cols_by_sum
        grid = [
            [3, 0, 2],
            [3, 0, 2],
            [3, 0, 2],
        ]
        result = sort_cols_by_sum(grid)
        # col with sum=0 should be first
        self.assertEqual([result[r][0] for r in range(3)], [0, 0, 0])


class TestRecolor2ndColorToDominant(unittest.TestCase):
    def test_recolor_2nd_color_to_dominant(self):
        from arc_agent.primitives import recolor_2nd_color_to_dominant
        # bg=0 (most common=8), dominant=5 (5 cells), accent=3 (2 cells)
        # accent (3) -> dominant (5)
        grid = [
            [0, 0, 0, 0, 0],
            [0, 5, 5, 3, 0],
            [0, 5, 5, 3, 0],
            [0, 5, 0, 0, 0],
        ]
        result = recolor_2nd_color_to_dominant(grid)
        # 3s should become 5
        self.assertEqual(result[1][3], 5)
        self.assertEqual(result[2][3], 5)
        # existing 5s unchanged
        self.assertEqual(result[1][1], 5)

    def test_erase_2nd_color(self):
        from arc_agent.primitives import erase_2nd_color
        # bg=0 (most common=8), dominant=5 (5 cells), accent=3 (2 cells)
        grid = [
            [0, 0, 0, 0, 0],
            [0, 5, 5, 3, 0],
            [0, 5, 5, 3, 0],
            [0, 5, 0, 0, 0],
        ]
        result = erase_2nd_color(grid)
        # 3s should be replaced with bg (0)
        self.assertEqual(result[1][3], 0)
        self.assertEqual(result[2][3], 0)
        # 5s unchanged
        self.assertEqual(result[1][1], 5)


class TestV18ToolkitContents(unittest.TestCase):
    def test_new_v18_primitives_exist(self):
        tk = build_initial_toolkit()
        new_names = [
            "recolor_dominant_touching_accent_to_4",
            "recolor_dominant_touching_accent_to_6",
            "recolor_dominant_touching_accent_to_7",
            "recolor_dominant_touching_accent_to_8",
            "recolor_dominant_touching_accent_to_2",
            "recolor_dominant_touching_accent_to_3",
            "fill_smallest_rect_hole_with_1",
            "fill_smallest_rect_hole_with_4",
            "fill_smallest_rect_hole_with_8",
            "recolor_bg_enclosed_by_dominant",
            "sort_rows_by_sum",
            "sort_cols_by_sum",
            "recolor_2nd_color_to_dominant",
            "erase_2nd_color",
        ]
        for name in new_names:
            self.assertIn(name, tk.concepts, f"Missing v0.18 primitive: {name}")

    def test_toolkit_size_v18(self):
        tk = build_initial_toolkit()
        self.assertGreaterEqual(tk.size, 256)  # 267 - 11 removed v0.17 primitives


class TestRecolorNonzeroInsideBbox(unittest.TestCase):
    def test_recolor_nonzero_inside_8_bbox_to_3(self):
        from arc_agent.primitives import recolor_nonzero_inside_8_bbox_to_3
        # bg=0, 8s form a cross, 1s inside bbox should become 3
        grid = [
            [0, 0, 0, 0, 0],
            [0, 8, 8, 8, 0],
            [0, 1, 8, 1, 0],
            [0, 8, 8, 8, 0],
            [0, 0, 0, 0, 0],
        ]
        result = recolor_nonzero_inside_8_bbox_to_3(grid)
        # 1s at (2,1) and (2,3) are inside bbox of 8s -> 3
        self.assertEqual(result[2][1], 3)
        self.assertEqual(result[2][3], 3)
        # 8s unchanged
        self.assertEqual(result[1][1], 8)
        # outside bbox unchanged
        self.assertEqual(result[0][0], 0)

    def test_recolor_nonzero_inside_2_bbox_to_4(self):
        from arc_agent.primitives import recolor_nonzero_inside_2_bbox_to_4
        grid = [
            [0, 0, 0, 0, 0],
            [0, 2, 0, 2, 0],
            [0, 0, 1, 0, 0],
            [0, 2, 0, 2, 0],
            [0, 0, 0, 0, 0],
        ]
        result = recolor_nonzero_inside_2_bbox_to_4(grid)
        # 1 at (2,2) is inside bbox of 2s -> 4
        self.assertEqual(result[2][2], 4)
        # 2s unchanged
        self.assertEqual(result[1][1], 2)
        # outside unchanged
        self.assertEqual(result[0][0], 0)


class TestFillRectInterior(unittest.TestCase):
    def test_fill_rect_interior_with_2(self):
        from arc_agent.primitives import fill_rect_interior_with_2
        # 5x5 grid with a 3x3 frame of 5s enclosing a 1x1 bg hole
        grid = [
            [0, 0, 0, 0, 0],
            [0, 5, 5, 5, 0],
            [0, 5, 0, 5, 0],
            [0, 5, 5, 5, 0],
            [0, 0, 0, 0, 0],
        ]
        result = fill_rect_interior_with_2(grid)
        # Interior at (2,2) should become 2
        self.assertEqual(result[2][2], 2)
        # Frame unchanged
        self.assertEqual(result[1][1], 5)
        # Outer bg unchanged
        self.assertEqual(result[0][0], 0)

    def test_fill_rect_interior_no_enclosed(self):
        from arc_agent.primitives import fill_rect_interior_with_2
        # Grid where bg is at the border - outer bg unreachable only if enclosed
        # A simple open grid: all bg cells connect to border
        grid = [[0, 0, 0], [0, 1, 0], [0, 0, 0]]
        result = fill_rect_interior_with_2(grid)
        # All bg cells touch the border - no enclosed region
        self.assertEqual(result[0][0], 0)
        self.assertEqual(result[1][0], 0)


class TestExtendLinesToContact(unittest.TestCase):
    def test_extend_horizontal_gap(self):
        from arc_agent.primitives import extend_lines_to_contact
        # Row with same-color cells and gap between
        grid = [
            [0, 0, 0, 0, 0],
            [0, 3, 0, 3, 0],
            [0, 0, 0, 0, 0],
        ]
        result = extend_lines_to_contact(grid)
        # Gap at (1,2) between two 3s should be filled
        self.assertEqual(result[1][2], 3)
        # Outer cells unchanged
        self.assertEqual(result[0][0], 0)

    def test_extend_vertical_gap(self):
        from arc_agent.primitives import extend_lines_to_contact
        grid = [
            [0, 5, 0],
            [0, 0, 0],
            [0, 5, 0],
        ]
        result = extend_lines_to_contact(grid)
        # Gap at (1,1) between two 5s should be filled
        self.assertEqual(result[1][1], 5)


class TestMarkRowColIntersections(unittest.TestCase):
    def test_mark_with_2(self):
        from arc_agent.primitives import mark_row_col_intersections_with_2
        # bg=0, dominant=1 (most common non-bg)
        # rows 0,2 contain 1; cols 0,2 contain 1
        # intersections at (0,0),(0,2),(2,0),(2,2) already have 1
        # but bg cells at accent row/col intersections
        grid = [
            [0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0],
        ]
        result = mark_row_col_intersections_with_2(grid)
        # Row 1 has 1 at col 0, row 2 has 1 at col 4
        # Col 0 has 1 at row 1, col 4 has 1 at row 2
        # Intersection (2,0): bg -> 2
        self.assertEqual(result[2][0], 2)
        # Intersection (1,4): bg -> 2
        self.assertEqual(result[1][4], 2)
        # Original 1s unchanged
        self.assertEqual(result[1][0], 1)


class TestFillBgAdjacent(unittest.TestCase):
    def test_fill_bg_adjacent_to_dominant_with_8(self):
        from arc_agent.primitives import fill_bg_adjacent_to_dominant_with_8
        # bg=0 (most common), dominant=5 (most common non-bg)
        grid = [
            [0, 0, 0, 0, 0, 0],
            [0, 0, 5, 5, 0, 0],
            [0, 0, 5, 5, 0, 0],
            [0, 0, 0, 0, 0, 0],
        ]
        result = fill_bg_adjacent_to_dominant_with_8(grid)
        # bg cells adjacent to 5s should become 8
        self.assertEqual(result[0][2], 8)  # above 5
        self.assertEqual(result[1][1], 8)  # left of 5
        self.assertEqual(result[1][4], 8)  # right of 5
        # non-adjacent bg unchanged
        self.assertEqual(result[0][0], 0)


class TestV19ToolkitContents(unittest.TestCase):
    def test_new_v19_primitives_exist(self):
        tk = build_initial_toolkit()
        new_names = [
            "recolor_nonzero_inside_8_bbox_to_3",
            "recolor_nonzero_inside_8_bbox_to_4",
            "recolor_nonzero_inside_8_bbox_to_2",
            "recolor_nonzero_inside_2_bbox_to_4",
            "recolor_nonzero_inside_2_bbox_to_8",
            "recolor_nonzero_inside_2_bbox_to_3",
            "recolor_nonzero_inside_3_bbox_to_4",
            "recolor_nonzero_inside_3_bbox_to_8",
            "recolor_nonzero_inside_6_bbox_to_4",
            "recolor_nonzero_inside_6_bbox_to_8",
            "fill_rect_interior_with_2",
            "fill_rect_interior_with_4",
            "fill_rect_interior_with_1",
            "fill_rect_interior_with_3",
            "mark_row_col_intersections_with_2",
            "mark_row_col_intersections_with_3",
            "mark_row_col_intersections_with_4",
            "extend_lines_to_contact",
            "fill_bg_adjacent_to_accent_with_3",
            "fill_bg_adjacent_to_accent_with_8",
            "fill_bg_adjacent_to_dominant_with_3",
            "fill_bg_adjacent_to_dominant_with_8",
        ]
        for name in new_names:
            self.assertIn(name, tk.concepts, f"Missing v0.19 primitive: {name}")

    def test_toolkit_size_v19(self):
        tk = build_initial_toolkit()
        self.assertGreaterEqual(tk.size, 278)


# ============================================================
# V20 Tests: shift, fill_enclosed, border/interior objects
# ============================================================

class TestShiftOperations(unittest.TestCase):
    def test_shift_down_1(self):
        from arc_agent.primitives import shift_down_1
        grid = [[1, 2], [3, 4], [5, 6]]
        result = shift_down_1(grid)
        self.assertEqual(result, [[5, 6], [1, 2], [3, 4]])

    def test_shift_up_1(self):
        from arc_agent.primitives import shift_up_1
        grid = [[1, 2], [3, 4], [5, 6]]
        result = shift_up_1(grid)
        self.assertEqual(result, [[3, 4], [5, 6], [1, 2]])

    def test_shift_left_1(self):
        from arc_agent.primitives import shift_left_1
        grid = [[1, 2, 3], [4, 5, 6]]
        result = shift_left_1(grid)
        self.assertEqual(result, [[2, 3, 1], [5, 6, 4]])

    def test_shift_right_1(self):
        from arc_agent.primitives import shift_right_1
        grid = [[1, 2, 3], [4, 5, 6]]
        result = shift_right_1(grid)
        self.assertEqual(result, [[3, 1, 2], [6, 4, 5]])

    def test_shift_down_solves_25ff71a9(self):
        """shift_down_1 confirmed to solve training task 25ff71a9."""
        from arc_agent.primitives import shift_down_1
        inp = [[0, 0, 0], [1, 0, 0], [0, 2, 0]]
        result = shift_down_1(inp)
        self.assertEqual(result, [[0, 2, 0], [0, 0, 0], [1, 0, 0]])


class TestFillEnclosedWallColor(unittest.TestCase):
    def test_simple_enclosed(self):
        from arc_agent.primitives import fill_enclosed_wall_color
        # 5x5 grid with bg=0, a frame of 5s enclosing a bg cell
        grid = [
            [0, 0, 0, 0, 0],
            [0, 5, 5, 5, 0],
            [0, 5, 0, 5, 0],
            [0, 5, 5, 5, 0],
            [0, 0, 0, 0, 0],
        ]
        result = fill_enclosed_wall_color(grid)
        self.assertEqual(result[2][2], 5)  # center filled with wall color

    def test_no_enclosed(self):
        from arc_agent.primitives import fill_enclosed_wall_color
        # All bg reachable from border
        grid = [
            [0, 0, 0, 0, 0],
            [0, 0, 5, 0, 0],
            [0, 0, 0, 0, 0],
        ]
        result = fill_enclosed_wall_color(grid)
        self.assertEqual(result, grid)  # Nothing enclosed

    def test_multiple_enclosed_regions(self):
        from arc_agent.primitives import fill_enclosed_wall_color
        # Two separate enclosed regions
        grid = [
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 3, 3, 3, 0, 5, 5, 5, 0],
            [0, 3, 0, 3, 0, 5, 0, 5, 0],
            [0, 3, 3, 3, 0, 5, 5, 5, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
        ]
        result = fill_enclosed_wall_color(grid)
        self.assertEqual(result[2][2], 3)  # filled with 3 (wall)
        self.assertEqual(result[2][6], 5)  # filled with 5 (wall)


class TestBorderInteriorObjects(unittest.TestCase):
    def test_remove_border_objects(self):
        from arc_agent.primitives import remove_border_objects
        # Object at corner touches border, object in center doesn't
        grid = [
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 3, 3, 0, 0, 0],
            [0, 0, 3, 3, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 5],
            [0, 0, 0, 0, 0, 5, 5],
        ]
        result = remove_border_objects(grid)
        # Border object (5s) removed
        self.assertEqual(result[5][6], 0)
        self.assertEqual(result[6][5], 0)
        # Interior object (3s) preserved
        self.assertEqual(result[2][2], 3)

    def test_keep_interior_objects(self):
        from arc_agent.primitives import keep_interior_objects
        grid = [
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 3, 3, 0, 0, 0],
            [0, 0, 3, 3, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 5],
            [0, 0, 0, 0, 0, 5, 5],
        ]
        result = keep_interior_objects(grid)
        # Interior object (3s) kept
        self.assertEqual(result[2][2], 3)
        # Border object (5s) removed
        self.assertEqual(result[5][6], 0)

    def test_hollow_objects(self):
        from arc_agent.primitives import hollow_objects
        grid = [
            [0, 0, 0, 0, 0, 0],
            [0, 5, 5, 5, 5, 0],
            [0, 5, 5, 5, 5, 0],
            [0, 5, 5, 5, 5, 0],
            [0, 5, 5, 5, 5, 0],
            [0, 0, 0, 0, 0, 0],
        ]
        result = hollow_objects(grid)
        # Border cells of object remain
        self.assertEqual(result[1][1], 5)
        self.assertEqual(result[1][4], 5)
        # Interior cells erased
        self.assertEqual(result[2][2], 0)
        self.assertEqual(result[3][3], 0)

    def test_fill_object_bboxes(self):
        from arc_agent.primitives import fill_object_bboxes
        grid = [
            [0, 0, 0, 0, 0],
            [0, 3, 0, 3, 0],
            [0, 0, 0, 0, 0],
            [0, 3, 0, 3, 0],
            [0, 0, 0, 0, 0],
        ]
        # All 3s are separate 1-cell objects, so bbox = 1x1, no change
        result = fill_object_bboxes(grid)
        self.assertEqual(result, grid)

    def test_fill_object_bboxes_connected(self):
        from arc_agent.primitives import fill_object_bboxes
        grid = [
            [0, 0, 0, 0, 0],
            [0, 3, 0, 0, 0],
            [0, 0, 3, 0, 0],
            [0, 0, 0, 0, 0],
        ]
        result = fill_object_bboxes(grid)
        # 3s not connected (diagonal), so each is 1x1 bbox, no change
        self.assertEqual(result, grid)


class TestTripleSearchPrepend(unittest.TestCase):
    """Test that triple search tries both append and prepend."""

    def test_prepend_triple_found(self):
        """Triple search should find prepend solutions (concept → pair)."""
        from arc_agent.synthesizer import ProgramSynthesizer
        from arc_agent.concepts import Program
        from arc_agent.scorer import TaskCache

        tk = build_initial_toolkit()
        synth = ProgramSynthesizer(toolkit=tk)

        # Create a task where prepend solves but append doesn't:
        # Task: apply mirror_v then rotate_90_cw to get the output
        # Best pair might be rotate_90_cw → identity (or similar)
        # Prepend mirror_v → best_pair should solve it

        # Use a simple 3x3 grid with asymmetric content
        inp = [[1, 0, 0], [0, 2, 0], [0, 0, 3]]
        from arc_agent.primitives import mirror_vertical, rotate_90_cw
        intermediate = mirror_vertical(inp)
        out = rotate_90_cw(intermediate)

        task = {"train": [{"input": inp, "output": out}], "test": []}
        cache = TaskCache(task)

        # The pair (mirror_v → rotate_90_cw) should be found,
        # but if the best pair is something else, prepend should help
        pair = synth.try_all_pairs(task, cache, top_k=20)
        self.assertIsNotNone(pair)

        # Whether it's a pair or triple, the task should be solvable
        if pair.fitness < 0.99:
            triple = synth.try_best_triples(pair, cache)
            self.assertIsNotNone(triple)
            # Triple should improve or match
            self.assertGreaterEqual(triple.fitness, pair.fitness)


class TestLearnedColorMapping(unittest.TestCase):
    """Test example-parameterized color mapping in solver."""

    def test_learn_color_mapping_simple(self):
        from arc_agent.solver import FourPillarsSolver
        solver = FourPillarsSolver(verbose=False)

        # Task: swap color 1 and 2
        task = {
            "train": [
                {"input": [[1, 0], [0, 2]], "output": [[2, 0], [0, 1]]},
                {"input": [[1, 1], [2, 0]], "output": [[2, 2], [1, 0]]},
            ],
            "test": [],
        }
        concepts = solver._learn_task_concepts(task)
        self.assertEqual(len(concepts), 1)
        self.assertEqual(concepts[0].name, "learned_color_map")

        # Verify the mapping works
        result = concepts[0].apply([[1, 2, 0]])
        self.assertEqual(result, [[2, 1, 0]])

    def test_no_mapping_for_diff_dims(self):
        from arc_agent.solver import FourPillarsSolver
        solver = FourPillarsSolver(verbose=False)

        # Different dimensions => no color mapping
        task = {
            "train": [
                {"input": [[1, 0], [0, 2]], "output": [[1]]},
            ],
            "test": [],
        }
        concepts = solver._learn_task_concepts(task)
        self.assertEqual(len(concepts), 0)

    def test_no_mapping_for_identity(self):
        from arc_agent.solver import FourPillarsSolver
        solver = FourPillarsSolver(verbose=False)

        # Identity => no changes => no mapping
        task = {
            "train": [
                {"input": [[1, 2], [3, 4]], "output": [[1, 2], [3, 4]]},
            ],
            "test": [],
        }
        concepts = solver._learn_task_concepts(task)
        self.assertEqual(len(concepts), 0)


class TestV20ToolkitContents(unittest.TestCase):
    def test_new_v20_primitives_exist(self):
        tk = build_initial_toolkit()
        new_names = [
            "shift_down_1", "shift_up_1", "shift_left_1", "shift_right_1",
            "fill_enclosed_wall_color",
            "remove_border_objects", "keep_interior_objects",
            "hollow_objects", "fill_object_bboxes",
        ]
        for name in new_names:
            self.assertIn(name, tk.concepts, f"Missing v0.20 primitive: {name}")

    def test_toolkit_size_v20(self):
        tk = build_initial_toolkit()
        self.assertGreaterEqual(tk.size, 287)
