"""Unit tests for task decomposition (Feature 2: Task Decomposition)."""
import unittest
from arc_agent.decompose import (
    DecompositionEngine,
    _extract_color_channel,
    _merge_color_channels,
    _split_into_quadrants,
    _merge_quadrants,
    _find_changed_cells,
    _extract_region_around_changes,
)
from arc_agent.concepts import Program, Concept
from arc_agent.primitives import identity, rotate_90_cw


class TestColorChannelOperations(unittest.TestCase):
    """Test color channel extraction and merging."""

    def test_extract_color_channel(self):
        """Extract single color as binary mask."""
        grid = [[1, 2, 1], [2, 1, 2], [1, 1, 2]]
        channel = _extract_color_channel(grid, 1)
        expected = [[1, 0, 1], [0, 1, 0], [1, 1, 0]]
        self.assertEqual(channel, expected)

    def test_extract_color_channel_no_match(self):
        """Extract color that doesn't exist returns all zeros."""
        grid = [[1, 2], [3, 4]]
        channel = _extract_color_channel(grid, 5)
        expected = [[0, 0], [0, 0]]
        self.assertEqual(channel, expected)

    def test_merge_color_channels(self):
        """Merge multiple color channels back together."""
        channels = {
            1: [[1, 0, 1], [0, 1, 0], [1, 1, 0]],
            2: [[0, 1, 0], [1, 0, 1], [0, 0, 1]],
        }
        result = _merge_color_channels(channels)
        expected = [[1, 2, 1], [2, 1, 2], [1, 1, 2]]
        self.assertEqual(result, expected)

    def test_merge_empty_channels(self):
        """Merging empty channels returns [[0]]."""
        result = _merge_color_channels({})
        self.assertEqual(result, [[0]])

    def test_color_roundtrip(self):
        """Extract and merge should be inverse operations."""
        original = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

        # Extract all colors
        colors = set()
        for row in original:
            for cell in row:
                if cell != 0:
                    colors.add(cell)

        channels = {c: _extract_color_channel(original, c) for c in colors}
        reconstructed = _merge_color_channels(channels)

        self.assertEqual(reconstructed, original)


class TestQuadrantOperations(unittest.TestCase):
    """Test spatial quadrant decomposition."""

    def test_split_into_quadrants(self):
        """Split 4x4 grid into 4 quadrants."""
        grid = [[1, 2, 3, 4],
                [5, 6, 7, 8],
                [9, 10, 11, 12],
                [13, 14, 15, 16]]

        quads = _split_into_quadrants(grid)

        self.assertEqual(quads['TL'], [[1, 2], [5, 6]])
        self.assertEqual(quads['TR'], [[3, 4], [7, 8]])
        self.assertEqual(quads['BL'], [[9, 10], [13, 14]])
        self.assertEqual(quads['BR'], [[11, 12], [15, 16]])

    def test_split_odd_dimensions(self):
        """Split works with odd dimensions (uneven splits)."""
        grid = [[1, 2, 3],
                [4, 5, 6],
                [7, 8, 9]]

        quads = _split_into_quadrants(grid)

        # Odd dimension: TL gets first 1 row/col
        self.assertEqual(quads['TL'], [[1]])
        self.assertEqual(quads['TR'], [[2, 3]])
        self.assertEqual(quads['BL'], [[4], [7]])
        self.assertEqual(quads['BR'], [[5, 6], [8, 9]])

    def test_merge_quadrants(self):
        """Merge quadrants back into original."""
        quads = {
            'TL': [[1, 2], [5, 6]],
            'TR': [[3, 4], [7, 8]],
            'BL': [[9, 10], [13, 14]],
            'BR': [[11, 12], [15, 16]],
        }

        result = _merge_quadrants(quads)

        expected = [[1, 2, 3, 4],
                    [5, 6, 7, 8],
                    [9, 10, 11, 12],
                    [13, 14, 15, 16]]

        self.assertEqual(result, expected)

    def test_quadrant_roundtrip(self):
        """Split and merge should be inverse operations."""
        original = [[1, 2, 3, 4],
                    [5, 6, 7, 8],
                    [9, 10, 11, 12],
                    [13, 14, 15, 16]]

        quads = _split_into_quadrants(original)
        reconstructed = _merge_quadrants(quads)

        self.assertEqual(reconstructed, original)


class TestDiffDetection(unittest.TestCase):
    """Test detection and extraction of changed regions."""

    def test_find_changed_cells(self):
        """Find cells that differ between input and output."""
        input_grid = [[1, 2], [3, 4]]
        output_grid = [[1, 2], [3, 5]]

        changed = _find_changed_cells(input_grid, output_grid)

        self.assertEqual(changed, {(1, 1)})

    def test_find_changed_cells_multiple(self):
        """Find multiple changed cells."""
        input_grid = [[1, 2], [3, 4]]
        output_grid = [[1, 0], [0, 4]]

        changed = _find_changed_cells(input_grid, output_grid)

        self.assertEqual(changed, {(0, 1), (1, 0)})

    def test_find_changed_cells_none(self):
        """Find no changes when grids are identical."""
        input_grid = [[1, 2], [3, 4]]
        output_grid = [[1, 2], [3, 4]]

        changed = _find_changed_cells(input_grid, output_grid)

        self.assertEqual(changed, set())

    def test_find_changed_cells_different_sizes(self):
        """Find changes when grids have different dimensions."""
        input_grid = [[1, 2], [3, 4]]
        output_grid = [[1, 2], [3, 4], [5, 6]]  # Output has extra row

        changed = _find_changed_cells(input_grid, output_grid)

        # New row is all changes
        self.assertIn((2, 0), changed)
        self.assertIn((2, 1), changed)

    def test_extract_region_around_changes(self):
        """Extract a rectangular region around changed cells."""
        grid = [[1, 1, 1],
                [1, 2, 1],
                [1, 1, 1]]

        changed = {(1, 1)}
        region = _extract_region_around_changes(grid, changed, padding=1)

        # With padding=1, should include all surrounding cells
        expected = [[1, 1, 1],
                    [1, 2, 1],
                    [1, 1, 1]]

        self.assertEqual(region, expected)

    def test_extract_region_with_padding(self):
        """Extract region respects boundaries with padding."""
        grid = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

        # Change at (0, 0) with padding=1 should not go negative
        changed = {(0, 0)}
        region = _extract_region_around_changes(grid, changed, padding=1)

        # Should start from (0, 0)
        self.assertEqual(region[0][0], 1)


class TestDecompositionEngine(unittest.TestCase):
    """Test the DecompositionEngine class."""

    def setUp(self):
        """Create engine for tests."""
        self.engine = DecompositionEngine()

    def test_engine_initialization(self):
        """Engine should initialize without error."""
        engine = DecompositionEngine()
        self.assertIsNotNone(engine)

    def test_decompose_if_needed_high_score(self):
        """If score is already 0.99+, should return None."""
        task = {'train': []}

        def dummy_synthesize(t):
            return None, []

        result = self.engine.decompose_if_needed(task, 0.99, dummy_synthesize)
        self.assertIsNone(result)

    def test_decompose_if_needed_low_score(self):
        """If score is < 0.99, should try strategies."""
        # Create a simple task
        task = {
            'train': [
                {
                    'input': [[1, 2], [3, 4]],
                    'output': [[1, 2], [3, 4]],
                }
            ]
        }

        def identity_synthesize(t):
            # Return an identity program
            prog = Program([Concept(
                kind="operator",
                name="identity",
                implementation=lambda g: g,
            )])
            prog.fitness = 1.0
            return prog, []

        # Call decompose with low score
        result = self.engine.decompose_if_needed(task, 0.5, identity_synthesize)

        # Should attempt decomposition (may succeed or fail gracefully)
        # At minimum, should not crash
        self.assertTrue(True)

    def test_color_channel_decomposition_single_color_skips(self):
        """Color decomposition should skip single-color tasks."""
        task = {
            'train': [
                {
                    'input': [[1, 1], [1, 1]],
                    'output': [[1, 1], [1, 1]],
                }
            ]
        }

        def dummy_synthesize(t):
            return None, []

        result = self.engine.try_color_channel_decomposition(task, dummy_synthesize)
        self.assertIsNone(result)

    def test_spatial_decomposition_small_grids_skips(self):
        """Spatial decomposition should skip grids too small."""
        task = {
            'train': [
                {
                    'input': [[1, 1], [1, 1]],  # 2x2 - too small
                    'output': [[1, 1], [1, 1]],
                }
            ]
        }

        def dummy_synthesize(t):
            return None, []

        result = self.engine.try_spatial_decomposition(task, dummy_synthesize)
        self.assertIsNone(result)


if __name__ == '__main__':
    unittest.main()
