"""
Tests for object-level primitives (TDD — tests written before implementation).

Object-level reasoning is the #1 gap in v0.1. Many ARC tasks require:
- Finding connected components (objects) in a grid
- Detecting object properties (size, color, bounding box, position)
- Transforming individual objects (move, recolor, duplicate)
- Reasoning about object relationships (same color, touching, aligned)
"""
import unittest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestConnectedComponents(unittest.TestCase):
    """Test finding connected components (objects) in grids."""

    def test_single_object(self):
        from arc_agent.objects import find_objects
        grid = [
            [0, 0, 0, 0],
            [0, 1, 1, 0],
            [0, 1, 1, 0],
            [0, 0, 0, 0],
        ]
        objects = find_objects(grid)
        self.assertEqual(len(objects), 1)
        self.assertEqual(objects[0].color, 1)
        self.assertEqual(objects[0].size, 4)

    def test_two_separate_objects(self):
        from arc_agent.objects import find_objects
        grid = [
            [1, 1, 0, 0],
            [1, 1, 0, 0],
            [0, 0, 2, 2],
            [0, 0, 2, 2],
        ]
        objects = find_objects(grid)
        self.assertEqual(len(objects), 2)
        colors = {obj.color for obj in objects}
        self.assertEqual(colors, {1, 2})

    def test_no_objects(self):
        from arc_agent.objects import find_objects
        grid = [[0, 0], [0, 0]]
        objects = find_objects(grid)
        self.assertEqual(len(objects), 0)

    def test_diagonal_not_connected(self):
        """By default, use 4-connectivity (not diagonal)."""
        from arc_agent.objects import find_objects
        grid = [
            [1, 0],
            [0, 1],
        ]
        objects = find_objects(grid)
        self.assertEqual(len(objects), 2)

    def test_l_shaped_object(self):
        from arc_agent.objects import find_objects
        grid = [
            [3, 0],
            [3, 0],
            [3, 3],
        ]
        objects = find_objects(grid)
        self.assertEqual(len(objects), 1)
        self.assertEqual(objects[0].size, 4)

    def test_multiple_colors_separate(self):
        """Different colors form different objects even if adjacent."""
        from arc_agent.objects import find_objects
        grid = [
            [1, 2],
            [1, 2],
        ]
        objects = find_objects(grid)
        self.assertEqual(len(objects), 2)


class TestObjectProperties(unittest.TestCase):
    """Test object property detection."""

    def test_bounding_box(self):
        from arc_agent.objects import find_objects
        grid = [
            [0, 0, 0, 0],
            [0, 5, 5, 0],
            [0, 5, 0, 0],
            [0, 0, 0, 0],
        ]
        obj = find_objects(grid)[0]
        self.assertEqual(obj.bbox, (1, 1, 2, 2))  # (min_r, min_c, max_r, max_c)

    def test_object_pixels(self):
        from arc_agent.objects import find_objects
        grid = [
            [0, 1, 0],
            [1, 1, 1],
            [0, 1, 0],
        ]
        obj = find_objects(grid)[0]
        self.assertEqual(obj.size, 5)
        self.assertIn((0, 1), obj.pixels)
        self.assertIn((1, 0), obj.pixels)

    def test_object_mask(self):
        """Extract just the object as a sub-grid."""
        from arc_agent.objects import find_objects
        grid = [
            [0, 0, 0],
            [0, 2, 2],
            [0, 2, 0],
        ]
        obj = find_objects(grid)[0]
        mask = obj.to_grid()
        self.assertEqual(mask, [[2, 2], [2, 0]])

    def test_object_center(self):
        from arc_agent.objects import find_objects
        grid = [
            [0, 0, 0],
            [0, 4, 0],
            [0, 0, 0],
        ]
        obj = find_objects(grid)[0]
        self.assertEqual(obj.center, (1, 1))


class TestObjectOperations(unittest.TestCase):
    """Test grid-level operations that use objects."""

    def test_extract_largest_object(self):
        from arc_agent.objects import extract_largest_object
        grid = [
            [1, 1, 0, 2],
            [1, 1, 0, 0],
            [1, 1, 0, 0],
        ]
        result = extract_largest_object(grid)
        self.assertEqual(result, [[1, 1], [1, 1], [1, 1]])

    def test_extract_smallest_object(self):
        from arc_agent.objects import extract_smallest_object
        grid = [
            [1, 1, 0, 2],
            [1, 1, 0, 0],
            [1, 1, 0, 0],
        ]
        result = extract_smallest_object(grid)
        self.assertEqual(result, [[2]])

    def test_count_objects(self):
        from arc_agent.objects import count_objects
        grid = [
            [1, 0, 2, 0, 3],
            [0, 0, 0, 0, 0],
        ]
        self.assertEqual(count_objects(grid), 3)

    def test_sort_objects_by_size(self):
        from arc_agent.objects import find_objects
        grid = [
            [1, 0, 2, 2],
            [0, 0, 2, 2],
            [3, 3, 3, 0],
        ]
        objects = find_objects(grid)
        sorted_objs = sorted(objects, key=lambda o: o.size, reverse=True)
        self.assertEqual(sorted_objs[0].size, 4)  # color 2
        self.assertEqual(sorted_objs[1].size, 3)  # color 3
        self.assertEqual(sorted_objs[2].size, 1)  # color 1

    def test_remove_object_by_color(self):
        from arc_agent.objects import remove_color
        grid = [
            [1, 2, 1],
            [2, 1, 2],
        ]
        result = remove_color(grid, 2)
        self.assertEqual(result, [[1, 0, 1], [0, 1, 0]])

    def test_isolate_color(self):
        """Keep only objects of a given color, zero everything else."""
        from arc_agent.objects import isolate_color
        grid = [
            [1, 2, 3],
            [2, 1, 2],
        ]
        result = isolate_color(grid, 2)
        self.assertEqual(result, [[0, 2, 0], [2, 0, 2]])


class TestObjectGridTransforms(unittest.TestCase):
    """Test Concept-compatible grid transforms using objects."""

    def test_extract_largest_as_concept(self):
        """extract_largest_object should work as a Concept (Grid → Grid)."""
        from arc_agent.objects import extract_largest_object
        grid = [[1, 0, 2, 2], [0, 0, 2, 2]]
        result = extract_largest_object(grid)
        self.assertIsNotNone(result)
        self.assertIsInstance(result, list)
        self.assertTrue(all(isinstance(row, list) for row in result))

    def test_recolor_largest_object(self):
        from arc_agent.objects import recolor_largest_object
        grid = [
            [1, 0, 2, 2],
            [0, 0, 2, 2],
        ]
        result = recolor_largest_object(grid, new_color=5)
        self.assertEqual(result, [[1, 0, 5, 5], [0, 0, 5, 5]])

    def test_mirror_each_object(self):
        """Mirror each object in-place within its bounding box."""
        from arc_agent.objects import mirror_objects_horizontal
        grid = [
            [0, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 1, 1, 0],
            [0, 0, 0, 0],
        ]
        result = mirror_objects_horizontal(grid)
        # The L-shape [1,0],[1,1] should become [0,1],[1,1]
        self.assertEqual(result[1][2], 1)
        self.assertEqual(result[1][1], 0)
        self.assertEqual(result[2][1], 1)
        self.assertEqual(result[2][2], 1)


if __name__ == "__main__":
    unittest.main()
