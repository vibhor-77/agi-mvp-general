"""Unit tests for project_markers_to_block primitive."""
import unittest
from arc_agent.primitives import project_markers_to_block


class TestProjectMarkersToBlock(unittest.TestCase):
    """Test line projection from block edges to isolated marker cells."""

    def test_simple_horizontal_projection_right(self):
        """Test drawing line right from block to marker (task 2c608aff example 1)."""
        # 3x3 block of color 3 at rows 1-3, cols 2-4
        # marker at row 3, col 9, color 4
        # should draw line from right edge of block (col 5) to marker (col 9)
        input_grid = [
            [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
            [8, 8, 3, 3, 3, 8, 8, 8, 8, 8, 8, 8],
            [8, 8, 3, 3, 3, 8, 8, 8, 8, 8, 8, 8],
            [8, 8, 3, 3, 3, 8, 8, 8, 8, 4, 8, 8],
            [8, 8, 3, 3, 3, 8, 8, 8, 8, 8, 8, 8],
            [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
            [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
            [8, 8, 8, 8, 8, 8, 8, 4, 8, 8, 8, 8],
            [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
        ]

        expected = [
            [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
            [8, 8, 3, 3, 3, 8, 8, 8, 8, 8, 8, 8],
            [8, 8, 3, 3, 3, 8, 8, 8, 8, 8, 8, 8],
            [8, 8, 3, 3, 3, 4, 4, 4, 4, 4, 8, 8],  # line drawn right
            [8, 8, 3, 3, 3, 8, 8, 8, 8, 8, 8, 8],
            [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
            [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
            [8, 8, 8, 8, 8, 8, 8, 4, 8, 8, 8, 8],  # no line (marker too far)
            [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
        ]

        result = project_markers_to_block(input_grid)
        self.assertEqual(result, expected)

    def test_vertical_projection_down(self):
        """Test drawing line down from block to marker (task 2c608aff example 2)."""
        # 3x3 block of color 1 at rows 2-4, cols 3-5
        # marker at row 8, col 3, color 8
        # should draw line from bottom edge of block (row 5) down to marker (row 8)
        input_grid = [
            [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
            [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
            [2, 2, 2, 1, 1, 1, 2, 2, 2, 2, 2, 2],
            [2, 2, 2, 1, 1, 1, 2, 2, 2, 2, 2, 2],
            [2, 2, 2, 1, 1, 1, 2, 2, 2, 2, 2, 2],
            [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
            [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
            [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
            [2, 2, 2, 8, 2, 2, 2, 2, 2, 2, 2, 2],
            [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
        ]

        expected = [
            [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
            [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
            [2, 2, 2, 1, 1, 1, 2, 2, 2, 2, 2, 2],
            [2, 2, 2, 1, 1, 1, 2, 2, 2, 2, 2, 2],
            [2, 2, 2, 1, 1, 1, 2, 2, 2, 2, 2, 2],
            [2, 2, 2, 8, 2, 2, 2, 2, 2, 2, 2, 2],  # line drawn down
            [2, 2, 2, 8, 2, 2, 2, 2, 2, 2, 2, 2],
            [2, 2, 2, 8, 2, 2, 2, 2, 2, 2, 2, 2],
            [2, 2, 2, 8, 2, 2, 2, 2, 2, 2, 2, 2],
            [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
        ]

        result = project_markers_to_block(input_grid)
        self.assertEqual(result, expected)

    def test_multiple_markers_different_edges(self):
        """Test multiple markers projecting to different edges."""
        # 4x4 block of color 4 at rows 4-7, cols 3-6
        # markers of color 2 at various locations
        input_grid = [
            [1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 4, 4, 4, 4, 1, 1, 1, 1, 1],
            [1, 1, 1, 4, 4, 4, 4, 1, 1, 1, 1, 1],
            [1, 1, 1, 4, 4, 4, 4, 1, 1, 1, 2, 1],
            [1, 1, 1, 4, 4, 4, 4, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        ]

        expected = [
            [1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1],  # marker at (0,4), projects up (but block is above)
            [1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 4, 4, 4, 4, 1, 1, 1, 1, 1],
            [1, 1, 1, 4, 4, 4, 4, 1, 1, 1, 1, 1],
            [1, 1, 1, 4, 4, 4, 4, 2, 2, 2, 2, 1],  # line right from (6,10)
            [1, 1, 1, 4, 4, 4, 4, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # marker at (11,1), not adjacent
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        ]

        result = project_markers_to_block(input_grid)
        self.assertEqual(result, expected)

    def test_no_markers(self):
        """Test grid with block but no markers."""
        grid = [
            [0, 0, 0, 0, 0],
            [0, 1, 1, 0, 0],
            [0, 1, 1, 0, 0],
            [0, 0, 0, 0, 0],
        ]
        result = project_markers_to_block(grid)
        self.assertEqual(result, grid)

    def test_only_background(self):
        """Test grid with only background color."""
        grid = [
            [8, 8, 8],
            [8, 8, 8],
            [8, 8, 8],
        ]
        result = project_markers_to_block(grid)
        self.assertEqual(result, grid)

    def test_single_cell_block(self):
        """Test with a single-cell block aligned with marker."""
        input_grid = [
            [0, 0, 0, 0],
            [0, 1, 0, 2],
            [0, 0, 0, 0],
        ]
        expected = [
            [0, 0, 0, 0],
            [0, 1, 2, 2],  # line right from block at (1,1) to marker at (1,3)
            [0, 0, 0, 0],
        ]
        result = project_markers_to_block(input_grid)
        self.assertEqual(result, expected)

    def test_marker_on_same_row_as_block_bottom(self):
        """Test marker on same row as block's bottom edge."""
        input_grid = [
            [0, 0, 0, 0, 0, 0],
            [0, 1, 1, 0, 0, 0],
            [0, 1, 1, 0, 2, 0],
        ]
        expected = [
            [0, 0, 0, 0, 0, 0],
            [0, 1, 1, 0, 0, 0],
            [0, 1, 1, 2, 2, 0],  # line right from block at row 2
        ]
        result = project_markers_to_block(input_grid)
        self.assertEqual(result, expected)


if __name__ == '__main__':
    unittest.main()
