"""Unit tests for DSL primitives (grid transformations)."""
import pytest
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


class TestGeometricTransforms:
    def test_rotate_90_cw(self):
        grid = [[1, 2], [3, 4]]
        result = rotate_90_cw(grid)
        assert result == [[3, 1], [4, 2]]

    def test_rotate_90_ccw(self):
        grid = [[1, 2], [3, 4]]
        result = rotate_90_ccw(grid)
        assert result == [[2, 4], [1, 3]]

    def test_rotate_180(self):
        grid = [[1, 2], [3, 4]]
        result = rotate_180(grid)
        assert result == [[4, 3], [2, 1]]

    def test_rotate_360_is_identity(self):
        grid = [[1, 2, 3], [4, 5, 6]]
        r1 = rotate_90_cw(grid)
        r2 = rotate_90_cw(r1)
        r3 = rotate_90_cw(r2)
        r4 = rotate_90_cw(r3)
        assert r4 == grid

    def test_mirror_horizontal(self):
        grid = [[1, 2, 3], [4, 5, 6]]
        result = mirror_horizontal(grid)
        assert result == [[3, 2, 1], [6, 5, 4]]

    def test_mirror_vertical(self):
        grid = [[1, 2], [3, 4]]
        result = mirror_vertical(grid)
        assert result == [[3, 4], [1, 2]]

    def test_transpose(self):
        grid = [[1, 2, 3], [4, 5, 6]]
        result = transpose(grid)
        assert result == [[1, 4], [2, 5], [3, 6]]

    def test_identity(self):
        grid = [[1, 2], [3, 4]]
        result = identity(grid)
        assert result == grid
        # Ensure it's a copy, not the same object
        result[0][0] = 99
        assert grid[0][0] == 1

    def test_empty_grid(self):
        assert rotate_90_cw([]) == []
        assert mirror_horizontal([]) == []


class TestColorTransforms:
    def test_invert_colors(self):
        grid = [[1, 0], [0, 2]]
        result = invert_colors(grid)
        assert result == [[0, 1], [1, 0]]

    def test_extract_unique_colors(self):
        grid = [[1, 2, 1], [3, 0, 2]]
        result = extract_unique_colors(grid)
        assert result == [[1, 2, 3]]

    def test_extract_unique_colors_empty(self):
        grid = [[0, 0], [0, 0]]
        result = extract_unique_colors(grid)
        assert result == [[0]]


class TestSpatialTransforms:
    def test_crop_to_nonzero(self):
        grid = [
            [0, 0, 0, 0],
            [0, 1, 2, 0],
            [0, 3, 4, 0],
            [0, 0, 0, 0],
        ]
        result = crop_to_nonzero(grid)
        assert result == [[1, 2], [3, 4]]

    def test_crop_single_cell(self):
        grid = [[0, 0], [0, 5]]
        result = crop_to_nonzero(grid)
        assert result == [[5]]

    def test_crop_all_zero(self):
        grid = [[0, 0], [0, 0]]
        result = crop_to_nonzero(grid)
        assert result == [[0]]

    def test_tile_2x2(self):
        grid = [[1, 2], [3, 4]]
        result = tile_2x2(grid)
        assert len(result) == 4
        assert len(result[0]) == 4
        assert result[0] == [1, 2, 1, 2]
        assert result[2] == [1, 2, 1, 2]

    def test_scale_2x(self):
        grid = [[1, 2], [3, 4]]
        result = scale_2x(grid)
        expected = [
            [1, 1, 2, 2],
            [1, 1, 2, 2],
            [3, 3, 4, 4],
            [3, 3, 4, 4],
        ]
        assert result == expected

    def test_scale_3x(self):
        grid = [[5]]
        result = scale_3x(grid)
        expected = [[5, 5, 5], [5, 5, 5], [5, 5, 5]]
        assert result == expected


class TestGravity:
    def test_gravity_down(self):
        grid = [[1, 0, 0], [0, 2, 0], [0, 0, 3]]
        result = gravity_down(grid)
        assert result == [[0, 0, 0], [0, 0, 0], [1, 2, 3]]

    def test_gravity_up(self):
        grid = [[0, 0, 0], [0, 0, 0], [1, 2, 3]]
        result = gravity_up(grid)
        assert result == [[1, 2, 3], [0, 0, 0], [0, 0, 0]]

    def test_gravity_left(self):
        grid = [[0, 0, 1], [0, 2, 0], [3, 0, 0]]
        result = gravity_left(grid)
        assert result == [[1, 0, 0], [2, 0, 0], [3, 0, 0]]

    def test_gravity_right(self):
        grid = [[1, 0, 0], [0, 2, 0], [0, 0, 3]]
        result = gravity_right(grid)
        assert result == [[0, 0, 1], [0, 0, 2], [0, 0, 3]]


class TestFillOperations:
    def test_fill_enclosed(self):
        grid = [
            [1, 1, 1],
            [1, 0, 1],
            [1, 1, 1],
        ]
        result = fill_enclosed(grid)
        assert result == [[1, 1, 1], [1, 1, 1], [1, 1, 1]]

    def test_fill_enclosed_not_border_connected(self):
        grid = [
            [2, 2, 2, 0],
            [2, 0, 2, 0],
            [2, 2, 2, 0],
        ]
        result = fill_enclosed(grid)
        # Interior 0 is enclosed, border 0s are not
        assert result[1][1] == 2
        assert result[0][3] == 0

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
        assert result == expected

    def test_flood_fill_background(self):
        grid = [
            [0, 0, 1],
            [0, 1, 1],
            [1, 1, 1],
        ]
        result = flood_fill_background(grid)
        assert result[0][0] == 1
        assert result[0][1] == 1
        assert result[1][0] == 1


class TestPredicates:
    def test_is_symmetric_h(self):
        assert is_symmetric_h([[1, 2, 1], [3, 4, 3]]) is True
        assert is_symmetric_h([[1, 2, 3]]) is False

    def test_is_symmetric_v(self):
        assert is_symmetric_v([[1, 2], [3, 4], [1, 2]]) is True
        assert is_symmetric_v([[1, 2], [3, 4]]) is False

    def test_is_square(self):
        assert is_square([[1, 2], [3, 4]]) is True
        assert is_square([[1, 2, 3], [4, 5, 6]]) is False

    def test_has_single_color(self):
        assert has_single_color([[1, 0, 1], [0, 1, 0]]) is True
        assert has_single_color([[1, 2], [3, 4]]) is False
        assert has_single_color([[0, 0], [0, 0]]) is True


class TestBuildToolkit:
    def test_toolkit_has_primitives(self):
        tk = build_initial_toolkit()
        assert tk.size > 20
        assert "rotate_90_cw" in tk.concepts
        assert "mirror_h" in tk.concepts
        assert "gravity_down" in tk.concepts
        assert "identity" in tk.concepts

    def test_toolkit_has_color_swaps(self):
        tk = build_initial_toolkit()
        assert "swap_1_to_2" in tk.concepts
        assert "swap_2_to_1" in tk.concepts

    def test_toolkit_has_recolor_ops(self):
        tk = build_initial_toolkit()
        assert "recolor_to_1" in tk.concepts
        assert "recolor_to_5" in tk.concepts

    def test_all_concepts_are_callable(self):
        tk = build_initial_toolkit()
        grid = [[1, 2], [3, 4]]
        for name, concept in tk.concepts.items():
            # Every concept should be able to process a grid without crashing
            result = concept.apply(grid)
            # Result should be a list of lists or None
            assert result is None or isinstance(result, list), \
                f"Concept {name} returned {type(result)}"
