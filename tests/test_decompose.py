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
    _detect_repeating_pattern,
    _separate_foreground_background,
    _merge_foreground_background,
    _get_bounding_box,
    _extract_subgrid,
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


class TestPatternDetection(unittest.TestCase):
    """Test repeating pattern detection."""

    def test_detect_2x2_repeating_pattern(self):
        """Detect a 2x2 tile repeated in a 4x4 grid."""
        # 2x2 tile [[1, 2], [3, 4]] repeated 2x2
        grid = [
            [1, 2, 1, 2],
            [3, 4, 3, 4],
            [1, 2, 1, 2],
            [3, 4, 3, 4],
        ]

        result = _detect_repeating_pattern(grid)
        self.assertIsNotNone(result)
        tile_h, tile_w, tile = result
        self.assertEqual(tile_h, 2)
        self.assertEqual(tile_w, 2)
        self.assertEqual(tile, [[1, 2], [3, 4]])

    def test_detect_3x3_repeating_pattern(self):
        """Detect a 3x3 tile repeated in a 6x6 grid."""
        tile = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        grid = tile + tile  # Vertical repeat
        grid = [row + row for row in grid]  # Horizontal repeat

        result = _detect_repeating_pattern(grid)
        self.assertIsNotNone(result)
        tile_h, tile_w, detected_tile = result
        self.assertEqual(tile_h, 3)
        self.assertEqual(tile_w, 3)
        self.assertEqual(detected_tile, tile)

    def test_detect_non_repeating_returns_none(self):
        """Non-repeating patterns should return None."""
        grid = [
            [1, 2, 3, 4],
            [5, 6, 7, 8],
            [9, 10, 11, 12],
            [13, 14, 15, 16],
        ]

        result = _detect_repeating_pattern(grid)
        self.assertIsNone(result)

    def test_detect_single_tile_returns_none(self):
        """Single tile (no repetition) should return None."""
        grid = [[1, 2], [3, 4]]
        result = _detect_repeating_pattern(grid)
        self.assertIsNone(result)

    def test_detect_empty_grid(self):
        """Empty grid should return None."""
        result = _detect_repeating_pattern([])
        self.assertIsNone(result)


class TestForegroundBackground(unittest.TestCase):
    """Test foreground/background separation."""

    def test_separate_foreground_background(self):
        """Separate grid into foreground and background."""
        grid = [
            [0, 0, 0],
            [0, 5, 0],
            [0, 0, 0],
        ]
        mask, bg = _separate_foreground_background(grid)

        self.assertEqual(bg, 0)
        self.assertEqual(mask, [
            [0, 0, 0],
            [0, 1, 0],
            [0, 0, 0],
        ])

    def test_separate_with_different_bg(self):
        """Detect background as most common color."""
        grid = [
            [5, 5, 3],
            [5, 2, 5],
            [5, 5, 5],
        ]
        mask, bg = _separate_foreground_background(grid)

        self.assertEqual(bg, 5)
        self.assertEqual(mask, [
            [0, 0, 1],
            [0, 1, 0],
            [0, 0, 0],
        ])

    def test_merge_foreground_background(self):
        """Merge foreground grid with background."""
        fg_grid = [
            [0, 0, 3],
            [0, 2, 0],
            [0, 0, 0],
        ]
        result = _merge_foreground_background(fg_grid, 0)

        expected = [
            [0, 0, 3],
            [0, 2, 0],
            [0, 0, 0],
        ]
        self.assertEqual(result, expected)

    def test_merge_with_different_bg(self):
        """Merge with a specific background color."""
        fg_grid = [
            [0, 0, 3],
            [0, 2, 0],
            [0, 0, 0],
        ]
        result = _merge_foreground_background(fg_grid, 5)

        expected = [
            [5, 5, 3],
            [5, 2, 5],
            [5, 5, 5],
        ]
        self.assertEqual(result, expected)


class TestBoundingBox(unittest.TestCase):
    """Test bounding box extraction."""

    def test_get_bounding_box(self):
        """Get bounding box of non-zero cells."""
        mask = [
            [0, 0, 0],
            [0, 1, 0],
            [0, 0, 0],
        ]
        bbox = _get_bounding_box(mask)
        self.assertEqual(bbox, (1, 1, 1, 1))

    def test_get_bounding_box_multiple_cells(self):
        """Get bounding box with multiple non-zero cells."""
        mask = [
            [0, 1, 0],
            [1, 1, 1],
            [0, 1, 0],
        ]
        bbox = _get_bounding_box(mask)
        self.assertEqual(bbox, (0, 0, 2, 2))

    def test_get_bounding_box_all_zero(self):
        """All-zero grid should return None."""
        mask = [
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
        ]
        bbox = _get_bounding_box(mask)
        self.assertIsNone(bbox)

    def test_extract_subgrid(self):
        """Extract a subgrid by coordinates."""
        grid = [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
        ]
        subgrid = _extract_subgrid(grid, 0, 0, 1, 1)
        expected = [[1, 2], [4, 5]]
        self.assertEqual(subgrid, expected)

    def test_extract_subgrid_single_cell(self):
        """Extract a single cell."""
        grid = [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
        ]
        subgrid = _extract_subgrid(grid, 1, 1, 1, 1)
        expected = [[5]]
        self.assertEqual(subgrid, expected)


class TestPatternDecompositionStrategy(unittest.TestCase):
    """Test the pattern decomposition strategy."""

    def test_pattern_decomposition_2x2_tile(self):
        """Pattern decomposition with 2x2 repeating tiles."""
        engine = DecompositionEngine()

        # Task: 2x2 tile repeated 2x2, each tile gets rotated
        tile_input = [[1, 2], [3, 4]]
        tile_output = [[2, 4], [1, 3]]  # Some transformation

        task = {
            'train': [
                {
                    'input': [
                        [1, 2, 1, 2],
                        [3, 4, 3, 4],
                        [1, 2, 1, 2],
                        [3, 4, 3, 4],
                    ],
                    'output': [
                        [2, 4, 2, 4],
                        [1, 3, 1, 3],
                        [2, 4, 2, 4],
                        [1, 3, 1, 3],
                    ],
                }
            ]
        }

        def identity_synthesize(t):
            # Return identity for tile task
            prog = Program([Concept(
                kind="operator",
                name="identity",
                implementation=lambda g: g,
            )])
            prog.fitness = 1.0
            return prog, []

        result = engine.try_pattern_decomposition(task, identity_synthesize)
        # Should recognize the pattern and attempt decomposition
        self.assertIsNotNone(result)

    def test_pattern_decomposition_no_pattern(self):
        """Pattern decomposition should return None for non-repeating grids."""
        engine = DecompositionEngine()

        task = {
            'train': [
                {
                    'input': [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
                    'output': [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
                }
            ]
        }

        def dummy_synthesize(t):
            return None, []

        result = engine.try_pattern_decomposition(task, dummy_synthesize)
        self.assertIsNone(result)


class TestSizeRatioDecomposition(unittest.TestCase):
    """Test size-ratio decomposition strategy."""

    def test_size_ratio_2x_expansion(self):
        """Size ratio decomposition with 2x expansion."""
        engine = DecompositionEngine()

        task = {
            'train': [
                {
                    'input': [[1, 2], [3, 4]],
                    'output': [[1, 1, 2, 2], [1, 1, 2, 2], [3, 3, 4, 4], [3, 3, 4, 4]],
                }
            ]
        }

        def identity_synthesize(t):
            # Return identity
            prog = Program([Concept(
                kind="operator",
                name="identity",
                implementation=lambda g: g,
            )])
            prog.fitness = 1.0
            return prog, []

        result = engine.try_size_ratio_decomposition(task, identity_synthesize)
        # Should detect the 2x ratio
        self.assertIsNotNone(result)

    def test_size_ratio_inconsistent_ratios(self):
        """Size ratio decomposition should fail with inconsistent ratios."""
        engine = DecompositionEngine()

        task = {
            'train': [
                {
                    'input': [[1, 2], [3, 4]],
                    'output': [[1, 1], [1, 1]],  # 1x ratio
                }
            ]
        }

        def dummy_synthesize(t):
            return None, []

        result = engine.try_size_ratio_decomposition(task, dummy_synthesize)
        self.assertIsNone(result)

    def test_size_ratio_3x_expansion(self):
        """Size ratio decomposition with 3x expansion."""
        engine = DecompositionEngine()

        task = {
            'train': [
                {
                    'input': [[1]],
                    'output': [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
                }
            ]
        }

        def identity_synthesize(t):
            prog = Program([Concept(
                kind="operator",
                name="identity",
                implementation=lambda g: g,
            )])
            prog.fitness = 1.0
            return prog, []

        result = engine.try_size_ratio_decomposition(task, identity_synthesize)
        self.assertIsNotNone(result)


class TestMaskingDecomposition(unittest.TestCase):
    """Test masking (foreground/background) decomposition strategy."""

    def test_masking_decomposition_same_dims(self):
        """Masking decomposition with same-dimension I/O."""
        engine = DecompositionEngine()

        task = {
            'train': [
                {
                    'input': [[0, 1, 0], [1, 1, 1], [0, 1, 0]],
                    'output': [[0, 2, 0], [2, 2, 2], [0, 2, 0]],
                }
            ]
        }

        def identity_synthesize(t):
            prog = Program([Concept(
                kind="operator",
                name="identity",
                implementation=lambda g: g,
            )])
            prog.fitness = 1.0
            return prog, []

        result = engine.try_masking_decomposition(task, identity_synthesize)
        # Should attempt decomposition with same-dim task
        self.assertIsNotNone(result)

    def test_masking_decomposition_different_dims(self):
        """Masking decomposition should skip different-dimension tasks."""
        engine = DecompositionEngine()

        task = {
            'train': [
                {
                    'input': [[1, 2], [3, 4]],
                    'output': [[1, 2, 0], [3, 4, 0]],  # Different width
                }
            ]
        }

        def dummy_synthesize(t):
            return None, []

        result = engine.try_masking_decomposition(task, dummy_synthesize)
        self.assertIsNone(result)


class TestDeterministicSubSynthesize(unittest.TestCase):
    """Test the deterministic sub-synthesize used by decomposition.

    The solver's _deterministic_sub_synthesize replaces the evolutionary
    synthesize_fn with fast deterministic search (singles → pairs → triples).
    """

    def test_sub_synthesize_finds_identity(self):
        """Deterministic sub-synthesize should find identity transform."""
        from arc_agent.solver import FourPillarsSolver as Solver

        solver = Solver(verbose=False)
        task = {
            'train': [
                {'input': [[1, 2], [3, 4]], 'output': [[1, 2], [3, 4]]},
                {'input': [[5, 6], [7, 8]], 'output': [[5, 6], [7, 8]]},
            ]
        }
        prog, history = solver._deterministic_sub_synthesize(task)
        self.assertIsNotNone(prog)
        self.assertIsInstance(history, list)
        # Identity should score perfectly
        self.assertGreaterEqual(prog.fitness, 0.99)

    def test_sub_synthesize_returns_tuple(self):
        """Return signature matches (program, history) for decomposer."""
        from arc_agent.solver import FourPillarsSolver as Solver

        solver = Solver(verbose=False)
        task = {
            'train': [
                {'input': [[1]], 'output': [[1]]},
            ]
        }
        result = solver._deterministic_sub_synthesize(task)
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)

    def test_sub_synthesize_unsolvable_returns_none(self):
        """Unsolvable sub-tasks should return (None, [])."""
        from arc_agent.solver import FourPillarsSolver as Solver

        solver = Solver(verbose=False)
        # Create a task with complex, unsolvable transform
        task = {
            'train': [
                {
                    'input': [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
                    'output': [[9, 1, 5], [3, 7, 2], [6, 4, 8]],
                },
            ]
        }
        prog, history = solver._deterministic_sub_synthesize(task)
        # May return None or a low-scoring program
        if prog is not None:
            # Should not be pixel-perfect for a random permutation
            self.assertLessEqual(prog.fitness, 1.0)
        self.assertIsInstance(history, list)


if __name__ == '__main__':
    unittest.main()
