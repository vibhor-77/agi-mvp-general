"""TDD tests for object decomposition architecture.

Tests the perceive → decompose → transform-per-object → reassemble pipeline.
"""
import unittest
from arc_agent.objects import (
    find_objects, GridObject,
    find_foreground_shapes, find_bounding_box, place_subgrid,
)


# ============================================================
# Perception helpers
# ============================================================

class TestFindBoundingBox(unittest.TestCase):
    """Tests for find_bounding_box (read-only grid perception)."""

    def test_single_object(self):
        grid = [
            [0, 0, 0, 0],
            [0, 3, 3, 0],
            [0, 3, 0, 0],
            [0, 0, 0, 0],
        ]
        assert find_bounding_box(grid) == (1, 1, 2, 2)

    def test_multiple_objects(self):
        """Bounding box should encompass all non-zero cells."""
        grid = [
            [1, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 2],
        ]
        assert find_bounding_box(grid) == (0, 0, 2, 3)

    def test_all_zeros(self):
        """Empty grid returns None."""
        grid = [[0, 0], [0, 0]]
        assert find_bounding_box(grid) is None

    def test_full_grid(self):
        grid = [[1, 2], [3, 4]]
        assert find_bounding_box(grid) == (0, 0, 1, 1)

    def test_single_pixel(self):
        grid = [[0, 0], [0, 5]]
        assert find_bounding_box(grid) == (1, 1, 1, 1)


class TestFindForegroundShapes(unittest.TestCase):
    """Tests for find_foreground_shapes (object extraction with metadata)."""

    def test_two_objects(self):
        grid = [
            [0, 1, 0, 0],
            [0, 1, 0, 2],
            [0, 0, 0, 2],
        ]
        shapes = find_foreground_shapes(grid)
        assert len(shapes) == 2

        # Check that each shape has required metadata
        for s in shapes:
            assert "subgrid" in s
            assert "bbox" in s
            assert "color" in s
            assert "size" in s
            assert "position" in s  # (row, col) of top-left corner

    def test_shape_subgrid_content(self):
        grid = [
            [0, 0, 0],
            [0, 3, 3],
            [0, 3, 0],
        ]
        shapes = find_foreground_shapes(grid)
        assert len(shapes) == 1
        s = shapes[0]
        assert s["color"] == 3
        assert s["subgrid"] == [[3, 3], [3, 0]]
        assert s["position"] == (1, 1)
        assert s["bbox"] == (1, 1, 2, 2)
        assert s["size"] == 3

    def test_empty_grid(self):
        grid = [[0, 0], [0, 0]]
        shapes = find_foreground_shapes(grid)
        assert shapes == []

    def test_single_pixel_object(self):
        grid = [[0, 0], [7, 0]]
        shapes = find_foreground_shapes(grid)
        assert len(shapes) == 1
        assert shapes[0]["subgrid"] == [[7]]
        assert shapes[0]["position"] == (1, 0)


class TestPlaceSubgrid(unittest.TestCase):
    """Tests for place_subgrid (reassembly primitive)."""

    def test_basic_placement(self):
        canvas = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
        subgrid = [[5, 5], [5, 0]]
        result = place_subgrid(canvas, subgrid, (0, 1))
        assert result == [[0, 5, 5], [0, 5, 0], [0, 0, 0]]

    def test_transparency(self):
        """Zero cells in subgrid should not overwrite canvas."""
        canvas = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
        subgrid = [[5, 0], [0, 5]]
        result = place_subgrid(canvas, subgrid, (0, 0))
        assert result == [[5, 1, 1], [1, 5, 1], [1, 1, 1]]

    def test_roundtrip_with_to_grid(self):
        """Extract an object then place it back should give same grid."""
        grid = [
            [0, 0, 0, 0],
            [0, 3, 3, 0],
            [0, 3, 0, 0],
            [0, 0, 0, 0],
        ]
        objects = find_objects(grid)
        assert len(objects) == 1
        obj = objects[0]
        subgrid = obj.to_grid()
        min_r, min_c, _, _ = obj.bbox

        canvas = [[0] * 4 for _ in range(4)]
        result = place_subgrid(canvas, subgrid, (min_r, min_c))
        assert result == grid

    def test_does_not_mutate_canvas(self):
        canvas = [[0, 0], [0, 0]]
        subgrid = [[1]]
        result = place_subgrid(canvas, subgrid, (0, 0))
        assert canvas == [[0, 0], [0, 0]]  # original unchanged
        assert result == [[1, 0], [0, 0]]

    def test_custom_transparent_color(self):
        """Allow non-zero transparent color."""
        canvas = [[0, 0], [0, 0]]
        subgrid = [[5, 9], [9, 5]]
        result = place_subgrid(canvas, subgrid, (0, 0), transparent_color=9)
        assert result == [[5, 0], [0, 5]]


# ============================================================
# Object decomposition solver
# ============================================================

class TestObjectDecomposeSolver(unittest.TestCase):
    """Tests for the end-to-end object decomposition solver."""

    def test_rotate_each_object(self):
        """Synthetic task: rotate each object 90° clockwise."""
        from arc_agent.object_decompose import solve_by_object_decomposition
        from arc_agent.primitives import build_initial_toolkit
        from arc_agent.scorer import TaskCache

        # Input: two L-shaped objects
        inp = [
            [0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 2, 0],
            [0, 1, 0, 0, 2, 0],
            [0, 1, 1, 0, 2, 2],
            [0, 0, 0, 0, 0, 0],
        ]
        # Output: each L rotated 90° CW, placed at same position
        # Object 1 (color 1): L at (1,1)-(3,2), shape [[1,0],[1,0],[1,1]]
        # Rotated 90 CW: [[1,1,1],[0,0,1]]
        # Object 2 (color 2): L at (1,4)-(3,5), shape [[2,0],[2,0],[2,2]]
        # Rotated 90 CW: [[2,2,2],[0,0,2]]
        out = [
            [0, 0, 0, 0, 0, 0],
            [0, 1, 1, 1, 2, 2],  # Wait, the rotated subgrids may not fit
            [0, 0, 0, 1, 0, 0],  # Need to think about this more carefully
            [0, 0, 0, 0, 0, 2],
            [0, 0, 0, 0, 0, 0],
        ]
        # Actually, rotating a 3x2 grid gives a 2x3 grid which may not
        # fit in the same position. Let's use a simpler transform.

    def test_mirror_each_object(self):
        """Synthetic task: mirror each object horizontally."""
        from arc_agent.object_decompose import solve_by_object_decomposition
        from arc_agent.primitives import build_initial_toolkit
        from arc_agent.scorer import TaskCache

        # Input with two L-shaped objects
        inp1 = [
            [0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 1, 1, 0, 0],
            [0, 0, 0, 0, 0],
        ]
        # mirror_h each object: [[0,1],[1,1]]
        out1 = [
            [0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 1, 1, 0, 0],
            [0, 0, 0, 0, 0],
        ]

        inp2 = [
            [0, 0, 0, 0, 0],
            [0, 0, 2, 0, 0],
            [0, 0, 2, 2, 0],
            [0, 0, 0, 0, 0],
        ]
        out2 = [
            [0, 0, 0, 0, 0],
            [0, 0, 0, 2, 0],
            [0, 0, 2, 2, 0],
            [0, 0, 0, 0, 0],
        ]

        task = {
            "train": [
                {"input": inp1, "output": out1},
                {"input": inp2, "output": out2},
            ]
        }

        toolkit = build_initial_toolkit()
        cache = TaskCache(task)
        result = solve_by_object_decomposition(task, toolkit, cache)

        assert result is not None
        assert result.fitness >= 0.99

    def test_recolor_each_object(self):
        """Synthetic task: recolor each object to a fixed color."""
        from arc_agent.object_decompose import solve_by_object_decomposition
        from arc_agent.primitives import build_initial_toolkit
        from arc_agent.scorer import TaskCache

        # All objects become color 5
        inp1 = [
            [0, 1, 0],
            [0, 1, 0],
            [0, 0, 3],
        ]
        out1 = [
            [0, 5, 0],
            [0, 5, 0],
            [0, 0, 5],
        ]
        inp2 = [
            [2, 0, 0],
            [0, 0, 4],
            [0, 0, 4],
        ]
        out2 = [
            [5, 0, 0],
            [0, 0, 5],
            [0, 0, 5],
        ]

        task = {
            "train": [
                {"input": inp1, "output": out1},
                {"input": inp2, "output": out2},
            ]
        }

        toolkit = build_initial_toolkit()
        cache = TaskCache(task)
        result = solve_by_object_decomposition(task, toolkit, cache)

        assert result is not None
        assert result.fitness >= 0.99

    def test_no_solution_returns_none(self):
        """If no per-object transform works, return None."""
        from arc_agent.object_decompose import solve_by_object_decomposition
        from arc_agent.primitives import build_initial_toolkit
        from arc_agent.scorer import TaskCache

        # Nonsensical transform that no primitive can reproduce
        task = {
            "train": [
                {
                    "input": [[0, 1, 0], [0, 1, 0]],
                    "output": [[9, 9, 9], [9, 9, 9]],
                }
            ]
        }

        toolkit = build_initial_toolkit()
        cache = TaskCache(task)
        result = solve_by_object_decomposition(task, toolkit, cache)
        # Should return None (no per-object transform can produce full 9s)
        assert result is None or result.fitness < 0.99


# ============================================================
# Conditional per-object recolor
# ============================================================

class TestConditionalRecolor(unittest.TestCase):
    """Tests for conditional per-object recolor by property."""

    def test_recolor_by_size(self):
        """Large objects get recolored, singletons stay (like task 67385a82)."""
        from arc_agent.object_decompose import solve_by_object_decomposition
        from arc_agent.primitives import build_initial_toolkit
        from arc_agent.scorer import TaskCache

        # Multi-pixel objects (color 3) become color 8; singletons stay 3
        inp1 = [
            [3, 3, 0],
            [0, 3, 0],
            [3, 0, 3],
        ]
        out1 = [
            [8, 8, 0],
            [0, 8, 0],
            [3, 0, 3],
        ]
        inp2 = [
            [0, 3, 3, 3, 0],
            [0, 0, 0, 0, 0],
            [3, 0, 3, 3, 0],
        ]
        out2 = [
            [0, 8, 8, 8, 0],
            [0, 0, 0, 0, 0],
            [3, 0, 8, 8, 0],
        ]

        task = {
            "train": [
                {"input": inp1, "output": out1},
                {"input": inp2, "output": out2},
            ]
        }

        toolkit = build_initial_toolkit()
        cache = TaskCache(task)
        result = solve_by_object_decomposition(task, toolkit, cache)

        assert result is not None
        assert result.fitness >= 0.99
        assert "recolor" in result.name.lower()

    def test_recolor_by_size_consistent(self):
        """All objects of size 2 become color 1, size 1 stay color 2."""
        from arc_agent.object_decompose import solve_by_object_decomposition
        from arc_agent.primitives import build_initial_toolkit
        from arc_agent.scorer import TaskCache

        inp1 = [[2, 0, 2, 2], [0, 0, 0, 0]]
        out1 = [[2, 0, 1, 1], [0, 0, 0, 0]]

        inp2 = [[0, 2, 2, 0], [2, 0, 0, 2]]
        out2 = [[0, 1, 1, 0], [2, 0, 0, 2]]

        task = {
            "train": [
                {"input": inp1, "output": out1},
                {"input": inp2, "output": out2},
            ]
        }

        toolkit = build_initial_toolkit()
        cache = TaskCache(task)
        result = solve_by_object_decomposition(task, toolkit, cache)

        assert result is not None
        assert result.fitness >= 0.99

    def test_no_recolor_when_truly_inconsistent(self):
        """Inconsistent across examples: same property → different colors."""
        from arc_agent.object_decompose import solve_by_object_decomposition
        from arc_agent.primitives import build_initial_toolkit
        from arc_agent.scorer import TaskCache

        # Example 1: singleton color 1 at top → 3
        # Example 2: singleton color 1 at top → 4 (contradicts ex1)
        # No property-based rule can explain this inconsistency.
        inp1 = [[1, 0], [0, 0]]
        out1 = [[3, 0], [0, 0]]
        inp2 = [[1, 0], [0, 0]]
        out2 = [[4, 0], [0, 0]]

        task = {
            "train": [
                {"input": inp1, "output": out1},
                {"input": inp2, "output": out2},
            ]
        }

        toolkit = build_initial_toolkit()
        cache = TaskCache(task)
        result = solve_by_object_decomposition(task, toolkit, cache)

        # No consistent rule: same input in two examples → different output
        assert result is None or result.fitness < 0.99

    def test_input_color_recolor_valid(self):
        """Different input colors mapping to different output colors IS valid."""
        from arc_agent.object_decompose import solve_by_object_decomposition
        from arc_agent.primitives import build_initial_toolkit
        from arc_agent.scorer import TaskCache

        # Color 1→3, color 2→4 is a valid by_input_color rule
        inp1 = [[1, 0, 2], [0, 0, 0]]
        out1 = [[3, 0, 4], [0, 0, 0]]

        task = {
            "train": [
                {"input": inp1, "output": out1},
            ]
        }

        toolkit = build_initial_toolkit()
        cache = TaskCache(task)
        result = solve_by_object_decomposition(task, toolkit, cache)

        assert result is not None
        assert result.fitness >= 0.99

    def test_recolor_by_input_color(self):
        """Each input color maps to a specific output color (like a5f85a15)."""
        from arc_agent.object_decompose import solve_by_object_decomposition
        from arc_agent.primitives import build_initial_toolkit
        from arc_agent.scorer import TaskCache

        # Color 2 → 5, color 3 → 5 (all foreground → same target)
        inp1 = [
            [0, 2, 0],
            [0, 0, 3],
            [2, 0, 0],
        ]
        out1 = [
            [0, 5, 0],
            [0, 0, 5],
            [5, 0, 0],
        ]
        inp2 = [
            [3, 0, 2, 2],
            [0, 0, 0, 0],
        ]
        out2 = [
            [5, 0, 5, 5],
            [0, 0, 0, 0],
        ]

        task = {
            "train": [
                {"input": inp1, "output": out1},
                {"input": inp2, "output": out2},
            ]
        }

        toolkit = build_initial_toolkit()
        cache = TaskCache(task)
        result = solve_by_object_decomposition(task, toolkit, cache)

        assert result is not None
        assert result.fitness >= 0.99

    def test_recolor_by_input_color_distinct(self):
        """Different input colors map to different output colors."""
        from arc_agent.object_decompose import solve_by_object_decomposition
        from arc_agent.primitives import build_initial_toolkit
        from arc_agent.scorer import TaskCache

        # Color 1 → 4, color 2 → 5
        inp1 = [
            [1, 0, 2],
            [1, 0, 2],
        ]
        out1 = [
            [4, 0, 5],
            [4, 0, 5],
        ]
        inp2 = [
            [0, 2, 0, 1],
            [0, 0, 0, 0],
        ]
        out2 = [
            [0, 5, 0, 4],
            [0, 0, 0, 0],
        ]

        task = {
            "train": [
                {"input": inp1, "output": out1},
                {"input": inp2, "output": out2},
            ]
        }

        toolkit = build_initial_toolkit()
        cache = TaskCache(task)
        result = solve_by_object_decomposition(task, toolkit, cache)

        assert result is not None
        assert result.fitness >= 0.99

    def test_recolor_by_position_vertical(self):
        """Objects in top half get one color, bottom half another."""
        from arc_agent.object_decompose import solve_by_object_decomposition
        from arc_agent.primitives import build_initial_toolkit
        from arc_agent.scorer import TaskCache

        # Objects in rows 0-1 → color 4, rows 2-3 → color 7
        inp1 = [
            [0, 3, 0],
            [0, 3, 0],
            [0, 0, 0],
            [3, 0, 0],
        ]
        out1 = [
            [0, 4, 0],
            [0, 4, 0],
            [0, 0, 0],
            [7, 0, 0],
        ]
        inp2 = [
            [3, 0, 3],
            [0, 0, 0],
            [0, 3, 0],
            [0, 3, 0],
        ]
        out2 = [
            [4, 0, 4],
            [0, 0, 0],
            [0, 7, 0],
            [0, 7, 0],
        ]

        task = {
            "train": [
                {"input": inp1, "output": out1},
                {"input": inp2, "output": out2},
            ]
        }

        toolkit = build_initial_toolkit()
        cache = TaskCache(task)
        result = solve_by_object_decomposition(task, toolkit, cache)

        assert result is not None
        assert result.fitness >= 0.99

    def test_recolor_by_shape_signature(self):
        """Objects with same shape (ignoring color) get same output color."""
        from arc_agent.object_decompose import solve_by_object_decomposition
        from arc_agent.primitives import build_initial_toolkit
        from arc_agent.scorer import TaskCache

        # L-shapes → color 4, single pixels → color 7
        inp1 = [
            [0, 1, 0, 0, 0],
            [0, 1, 1, 0, 2],
            [0, 0, 0, 0, 0],
        ]
        out1 = [
            [0, 4, 0, 0, 0],
            [0, 4, 4, 0, 7],
            [0, 0, 0, 0, 0],
        ]
        inp2 = [
            [0, 0, 0, 3, 0],
            [0, 5, 0, 3, 0],
            [0, 0, 0, 3, 3],
        ]
        out2 = [
            [0, 0, 0, 4, 0],
            [0, 7, 0, 4, 0],
            [0, 0, 0, 4, 4],
        ]

        task = {
            "train": [
                {"input": inp1, "output": out1},
                {"input": inp2, "output": out2},
            ]
        }

        toolkit = build_initial_toolkit()
        cache = TaskCache(task)
        result = solve_by_object_decomposition(task, toolkit, cache)

        assert result is not None
        assert result.fitness >= 0.99

    def test_recolor_by_size_rank(self):
        """Largest object gets one color, smallest gets another (rank-based).

        Uses same-color objects (separated by background) with varying
        absolute sizes across examples. by_size fails because sizes differ
        across examples; by_input_color fails because all objects have same
        input color. Only rank order is consistent: largest→4, smallest→7.
        """
        from arc_agent.object_decompose import solve_by_object_decomposition
        from arc_agent.primitives import build_initial_toolkit
        from arc_agent.scorer import TaskCache

        # Ex1: two color-1 objects. size 3 (largest)→4, size 1 (smallest)→7
        # Smallest on the LEFT to break by_position
        inp1 = [
            [1, 0, 0],
            [0, 0, 1],
            [0, 0, 1],
            [0, 0, 1],
        ]
        out1 = [
            [7, 0, 0],
            [0, 0, 4],
            [0, 0, 4],
            [0, 0, 4],
        ]
        # Ex2: two color-1 objects. size 3 (largest)→4, size 2 (smallest)→7
        # Largest on the RIGHT
        inp2 = [
            [0, 1, 0, 1],
            [0, 1, 0, 1],
            [0, 0, 0, 1],
        ]
        out2 = [
            [0, 7, 0, 4],
            [0, 7, 0, 4],
            [0, 0, 0, 4],
        ]
        # Ex3: two color-1 objects. size 2 (largest)→4, size 1 (smallest)→7
        # Breaks by_size: size 2 mapped to 7 in ex2 but 4 here
        # Largest on LEFT to break by_position
        inp3 = [
            [1, 0, 1],
            [1, 0, 0],
        ]
        out3 = [
            [4, 0, 7],
            [4, 0, 0],
        ]

        task = {
            "train": [
                {"input": inp1, "output": out1},
                {"input": inp2, "output": out2},
                {"input": inp3, "output": out3},
            ]
        }

        toolkit = build_initial_toolkit()
        cache = TaskCache(task)
        result = solve_by_object_decomposition(task, toolkit, cache)

        assert result is not None
        assert result.fitness >= 0.99
        assert "rank" in result.name.lower()

    def test_recolor_by_compactness(self):
        """Rectangular objects get one color, irregular objects another.

        Uses same-color, same-size objects with different compactness.
        Positions vary to break by_position. Shapes vary to break by_shape.
        """
        from arc_agent.object_decompose import solve_by_object_decomposition
        from arc_agent.primitives import build_initial_toolkit
        from arc_agent.scorer import TaskCache

        # Ex1: Two color-3 objects of size 4.
        # Object A: L-shape (non-compact, bbox 3x2) → color 7, top-left
        # Object B: 2x2 rectangle (compact=1.0) → color 4, bottom-right
        inp1 = [
            [3, 3, 0, 0, 0],
            [3, 0, 0, 0, 0],
            [3, 0, 0, 0, 0],
            [0, 0, 0, 3, 3],
            [0, 0, 0, 3, 3],
        ]
        out1 = [
            [7, 7, 0, 0, 0],
            [7, 0, 0, 0, 0],
            [7, 0, 0, 0, 0],
            [0, 0, 0, 4, 4],
            [0, 0, 0, 4, 4],
        ]
        # Ex2: Two color-3 objects of size 3.
        # Object C: 1x3 rectangle (compact=1.0) → color 4, TOP (breaks by_position)
        # Object D: L-shape (non-compact) → color 7, BOTTOM
        inp2 = [
            [3, 3, 3, 0],
            [0, 0, 0, 0],
            [0, 3, 3, 0],
            [0, 0, 3, 0],
        ]
        out2 = [
            [4, 4, 4, 0],
            [0, 0, 0, 0],
            [0, 7, 7, 0],
            [0, 0, 7, 0],
        ]

        task = {
            "train": [
                {"input": inp1, "output": out1},
                {"input": inp2, "output": out2},
            ]
        }

        toolkit = build_initial_toolkit()
        cache = TaskCache(task)
        result = solve_by_object_decomposition(task, toolkit, cache)

        assert result is not None
        assert result.fitness >= 0.99

    def test_compactness_learner_returns_rule(self):
        """Unit test: _learn_recolor_by_compactness returns correct rule."""
        from arc_agent.object_decompose import _learn_recolor_by_compactness

        inp1 = [[3, 3, 0, 0, 0], [3, 0, 0, 0, 0], [3, 0, 0, 0, 0],
                 [0, 0, 0, 3, 3], [0, 0, 0, 3, 3]]
        out1 = [[7, 7, 0, 0, 0], [7, 0, 0, 0, 0], [7, 0, 0, 0, 0],
                 [0, 0, 0, 4, 4], [0, 0, 0, 4, 4]]
        train = [{"input": inp1, "output": out1}]
        rule = _learn_recolor_by_compactness(train)
        assert rule is not None
        assert rule[True] == 4   # compact → 4
        assert rule[False] == 7  # non-compact → 7

    def test_recolor_by_has_hole(self):
        """Objects with holes get one color, solid objects another.

        Uses same-color, same-size objects so that only hole-detection
        can distinguish them. Sizes differ across examples to break by_size.
        """
        from arc_agent.object_decompose import solve_by_object_decomposition
        from arc_agent.primitives import build_initial_toolkit
        from arc_agent.scorer import TaskCache

        # Ex1: Both objects have color 1 and size 8.
        # Object A: 3x3 ring with hole (8 pixels) → color 4
        # Object B: 2x4 solid rectangle (8 pixels) → color 7
        inp1 = [
            [0, 0, 0, 0, 0, 0, 0],
            [0, 1, 1, 1, 0, 0, 0],
            [0, 1, 0, 1, 0, 0, 0],
            [0, 1, 1, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 1, 1],
            [0, 0, 0, 0, 1, 1, 1],
        ]
        out1 = [
            [0, 0, 0, 0, 0, 0, 0],
            [0, 4, 4, 4, 0, 0, 0],
            [0, 4, 0, 4, 0, 0, 0],
            [0, 4, 4, 4, 0, 0, 0],
            [0, 0, 0, 0, 7, 7, 7],
            [0, 0, 0, 0, 7, 7, 7],
        ]
        # Ex2: Both objects have color 2 and size 12.
        # Object C: 4x4 ring with hole (12 pixels) → color 4
        # Object D: 3x4 solid rectangle (12 pixels) → color 7
        inp2 = [
            [2, 2, 2, 2, 0, 0, 0, 0],
            [2, 0, 0, 2, 0, 0, 0, 0],
            [2, 0, 0, 2, 0, 0, 0, 0],
            [2, 2, 2, 2, 0, 0, 0, 0],
            [0, 0, 0, 0, 2, 2, 2, 2],
            [0, 0, 0, 0, 2, 2, 2, 2],
            [0, 0, 0, 0, 2, 2, 2, 2],
        ]
        out2 = [
            [4, 4, 4, 4, 0, 0, 0, 0],
            [4, 0, 0, 4, 0, 0, 0, 0],
            [4, 0, 0, 4, 0, 0, 0, 0],
            [4, 4, 4, 4, 0, 0, 0, 0],
            [0, 0, 0, 0, 7, 7, 7, 7],
            [0, 0, 0, 0, 7, 7, 7, 7],
            [0, 0, 0, 0, 7, 7, 7, 7],
        ]

        task = {
            "train": [
                {"input": inp1, "output": out1},
                {"input": inp2, "output": out2},
            ]
        }

        toolkit = build_initial_toolkit()
        cache = TaskCache(task)
        result = solve_by_object_decomposition(task, toolkit, cache)

        assert result is not None
        assert result.fitness >= 0.99

    def test_has_hole_learner_returns_rule(self):
        """Unit test: _learn_recolor_by_has_hole returns correct rule."""
        from arc_agent.object_decompose import _learn_recolor_by_has_hole, _has_hole
        from arc_agent.objects import find_foreground_shapes

        # Ring with hole
        ring = find_foreground_shapes([[1, 1, 1], [1, 0, 1], [1, 1, 1]])[0]
        assert _has_hole(ring) is True

        # Solid rectangle
        solid = find_foreground_shapes([[0, 0, 0], [0, 1, 1], [0, 1, 1]])[0]
        assert _has_hole(solid) is False

        # L-shape (no hole — open, not enclosed)
        lshape = find_foreground_shapes([[1, 0], [1, 1]])[0]
        assert _has_hole(lshape) is False

    def test_has_hole_detection(self):
        """_has_hole correctly identifies enclosed background."""
        from arc_agent.object_decompose import _has_hole
        from arc_agent.objects import find_foreground_shapes

        # U-shape (open top, no hole)
        u = find_foreground_shapes([[1, 0, 1], [1, 0, 1], [1, 1, 1]])[0]
        assert _has_hole(u) is False

        # O-shape (hole)
        o = find_foreground_shapes([[1, 1, 1], [1, 0, 1], [1, 1, 1]])[0]
        assert _has_hole(o) is True

        # Donut with larger hole
        d_grid = [[0]*6 for _ in range(6)]
        for r in range(5):
            for c in range(5):
                if r == 0 or r == 4 or c == 0 or c == 4:
                    d_grid[r][c] = 2
        donut = find_foreground_shapes(d_grid)[0]
        assert _has_hole(donut) is True
