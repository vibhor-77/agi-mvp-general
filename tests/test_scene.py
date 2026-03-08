"""Tests for object-centric scene graph reasoning (TDD).

Written BEFORE the implementation — these define the expected behavior
of the scene perception, diff, and rule inference pipeline.
"""
import unittest


class TestBuildScene(unittest.TestCase):
    """Test scene graph construction from grids."""

    def test_single_object(self):
        from arc_agent.scene import build_scene
        grid = [[0, 0, 0],
                [0, 1, 1],
                [0, 1, 1]]
        scene = build_scene(grid)
        self.assertEqual(scene.bg_color, 0)
        self.assertEqual(len(scene.objects), 1)
        self.assertEqual(scene.objects[0].color, 1)
        self.assertEqual(scene.objects[0].size, 4)

    def test_multiple_objects(self):
        from arc_agent.scene import build_scene
        grid = [[1, 0, 2],
                [0, 0, 0],
                [3, 0, 4]]
        scene = build_scene(grid)
        self.assertEqual(scene.bg_color, 0)
        self.assertEqual(len(scene.objects), 4)
        colors = {o.color for o in scene.objects}
        self.assertEqual(colors, {1, 2, 3, 4})

    def test_connected_components_not_diagonal(self):
        """Diagonal pixels of same color are separate objects (4-connectivity)."""
        from arc_agent.scene import build_scene
        grid = [[1, 0],
                [0, 1]]
        scene = build_scene(grid)
        self.assertEqual(len(scene.objects), 2)

    def test_shape_signature(self):
        """Two objects with same shape have same shape_signature."""
        from arc_agent.scene import build_scene
        # L-shape at top-left
        grid1 = [[1, 0, 0],
                 [1, 1, 0],
                 [0, 0, 0]]
        # Same L-shape at bottom-right
        grid2 = [[0, 0, 0],
                 [0, 2, 0],
                 [0, 2, 2]]
        s1 = build_scene(grid1)
        s2 = build_scene(grid2)
        self.assertEqual(s1.objects[0].shape_signature,
                         s2.objects[0].shape_signature)

    def test_different_shapes(self):
        """Objects with different shapes have different shape_signatures."""
        from arc_agent.scene import build_scene
        grid = [[1, 1, 0, 2],
                [0, 0, 0, 2],
                [0, 0, 0, 2]]
        scene = build_scene(grid)
        sigs = [o.shape_signature for o in scene.objects]
        self.assertNotEqual(sigs[0], sigs[1])

    def test_bg_color_most_frequent(self):
        """Background is the most frequent color."""
        from arc_agent.scene import build_scene
        # 5 has more cells than 0
        grid = [[5, 5, 5],
                [5, 1, 5],
                [5, 5, 5]]
        scene = build_scene(grid)
        self.assertEqual(scene.bg_color, 5)
        # The "1" pixel is the only object
        self.assertEqual(len(scene.objects), 1)
        self.assertEqual(scene.objects[0].color, 1)

    def test_empty_grid(self):
        from arc_agent.scene import build_scene
        grid = [[0, 0], [0, 0]]
        scene = build_scene(grid)
        self.assertEqual(len(scene.objects), 0)
        self.assertEqual(scene.bg_color, 0)


class TestDiffScenes(unittest.TestCase):
    """Test structured diffs between input and output scenes."""

    def test_color_change(self):
        """Detect when an object changes color."""
        from arc_agent.scene import build_scene, diff_scenes
        inp = [[0, 1, 1],
               [0, 1, 1],
               [0, 0, 0]]
        out = [[0, 2, 2],
               [0, 2, 2],
               [0, 0, 0]]
        diff = diff_scenes(build_scene(inp), build_scene(out))
        self.assertEqual(len(diff.matched), 1)
        self.assertEqual(diff.matched[0].new_color, 2)
        self.assertTrue(diff.matched[0].shape_preserved)
        self.assertEqual(diff.matched[0].movement, (0, 0))

    def test_object_moved(self):
        """Detect when an object shifts position."""
        from arc_agent.scene import build_scene, diff_scenes
        inp = [[1, 0, 0],
               [0, 0, 0],
               [0, 0, 0]]
        out = [[0, 0, 0],
               [0, 0, 0],
               [0, 0, 1]]
        diff = diff_scenes(build_scene(inp), build_scene(out))
        self.assertEqual(len(diff.matched), 1)
        self.assertEqual(diff.matched[0].movement, (2, 2))
        self.assertIsNone(diff.matched[0].new_color)

    def test_object_removed(self):
        """Detect when an object disappears."""
        from arc_agent.scene import build_scene, diff_scenes
        inp = [[1, 0, 2],
               [0, 0, 0],
               [0, 0, 0]]
        out = [[0, 0, 2],
               [0, 0, 0],
               [0, 0, 0]]
        diff = diff_scenes(build_scene(inp), build_scene(out))
        self.assertEqual(len(diff.removed), 1)
        self.assertEqual(diff.removed[0].color, 1)

    def test_object_added(self):
        """Detect when a new object appears."""
        from arc_agent.scene import build_scene, diff_scenes
        inp = [[0, 0, 0],
               [0, 0, 0],
               [0, 0, 0]]
        out = [[0, 0, 0],
               [0, 3, 0],
               [0, 0, 0]]
        diff = diff_scenes(build_scene(inp), build_scene(out))
        self.assertEqual(len(diff.added), 1)
        self.assertEqual(diff.added[0].color, 3)

    def test_shape_change_detected(self):
        """Detect when object changes shape (not just position/color)."""
        from arc_agent.scene import build_scene, diff_scenes
        inp = [[1, 0],
               [1, 0]]
        out = [[1, 1],
               [0, 0]]
        diff = diff_scenes(build_scene(inp), build_scene(out))
        # Same color, same size, but different shape
        self.assertEqual(len(diff.matched), 1)
        self.assertFalse(diff.matched[0].shape_preserved)

    def test_multiple_changes(self):
        """Multiple objects change simultaneously."""
        from arc_agent.scene import build_scene, diff_scenes
        inp = [[1, 0, 2],
               [0, 0, 0],
               [3, 0, 0]]
        out = [[4, 0, 5],
               [0, 0, 0],
               [6, 0, 0]]
        diff = diff_scenes(build_scene(inp), build_scene(out))
        # All three objects recolored
        self.assertEqual(len(diff.matched), 3)
        new_colors = {d.new_color for d in diff.matched}
        self.assertEqual(new_colors, {4, 5, 6})


class TestConsistentDiffs(unittest.TestCase):
    """Test finding consistent patterns across multiple examples."""

    def test_consistent_color_rule(self):
        """Detect consistent 'color X → color Y' across examples."""
        from arc_agent.scene import build_scene, diff_scenes, find_consistent_rules
        examples = [
            ([[0, 1, 0], [0, 0, 0]], [[0, 2, 0], [0, 0, 0]]),
            ([[1, 0, 0], [0, 0, 0]], [[2, 0, 0], [0, 0, 0]]),
            ([[0, 0, 1], [0, 1, 0]], [[0, 0, 2], [0, 2, 0]]),
        ]
        diffs = [diff_scenes(build_scene(inp), build_scene(out))
                 for inp, out in examples]
        rules = find_consistent_rules(diffs)
        # Should find: recolor 1→2
        self.assertTrue(any(
            r.kind == "recolor" and r.src_color == 1 and r.dst_color == 2
            for r in rules
        ))

    def test_consistent_removal_rule(self):
        """Detect consistent 'remove color X' across examples."""
        from arc_agent.scene import build_scene, diff_scenes, find_consistent_rules
        examples = [
            ([[1, 2, 0], [0, 0, 0]], [[0, 2, 0], [0, 0, 0]]),
            ([[0, 2, 1], [0, 0, 0]], [[0, 2, 0], [0, 0, 0]]),
        ]
        diffs = [diff_scenes(build_scene(inp), build_scene(out))
                 for inp, out in examples]
        rules = find_consistent_rules(diffs)
        self.assertTrue(any(
            r.kind == "remove" and r.src_color == 1
            for r in rules
        ))

    def test_no_consistent_rule(self):
        """When changes are inconsistent, no rule is returned."""
        from arc_agent.scene import build_scene, diff_scenes, find_consistent_rules
        examples = [
            ([[1, 0], [0, 0]], [[2, 0], [0, 0]]),  # 1→2
            ([[1, 0], [0, 0]], [[3, 0], [0, 0]]),  # 1→3 (inconsistent!)
        ]
        diffs = [diff_scenes(build_scene(inp), build_scene(out))
                 for inp, out in examples]
        rules = find_consistent_rules(diffs)
        # Should find no consistent recolor rule for color 1
        recolor_1 = [r for r in rules if r.kind == "recolor" and r.src_color == 1]
        self.assertEqual(len(recolor_1), 0)


class TestRuleApplication(unittest.TestCase):
    """Test applying inferred rules to produce output grids."""

    def test_apply_recolor_rule(self):
        """Apply a recolor rule to a new grid."""
        from arc_agent.scene import ObjectRule, apply_rules
        rules = [ObjectRule(kind="recolor", src_color=1, dst_color=2)]
        grid = [[0, 1, 0],
                [1, 1, 0],
                [0, 0, 0]]
        result = apply_rules(grid, rules, bg_color=0)
        expected = [[0, 2, 0],
                    [2, 2, 0],
                    [0, 0, 0]]
        self.assertEqual(result, expected)

    def test_apply_removal_rule(self):
        """Apply a removal rule to a new grid."""
        from arc_agent.scene import ObjectRule, apply_rules
        rules = [ObjectRule(kind="remove", src_color=1)]
        grid = [[1, 2, 0],
                [1, 0, 0],
                [0, 0, 2]]
        result = apply_rules(grid, rules, bg_color=0)
        expected = [[0, 2, 0],
                    [0, 0, 0],
                    [0, 0, 2]]
        self.assertEqual(result, expected)

    def test_apply_multiple_rules(self):
        """Apply multiple rules in sequence."""
        from arc_agent.scene import ObjectRule, apply_rules
        rules = [
            ObjectRule(kind="recolor", src_color=1, dst_color=3),
            ObjectRule(kind="remove", src_color=2),
        ]
        grid = [[1, 2],
                [0, 0]]
        result = apply_rules(grid, rules, bg_color=0)
        expected = [[3, 0],
                    [0, 0]]
        self.assertEqual(result, expected)


class TestEndToEnd(unittest.TestCase):
    """Test the full pipeline: task → scene → diff → rules → apply → validate."""

    def test_simple_recolor_task(self):
        """Full pipeline solves a simple recolor task."""
        from arc_agent.scene import solve_with_object_rules
        task = {
            "train": [
                {"input": [[0, 1, 0], [0, 1, 0]], "output": [[0, 3, 0], [0, 3, 0]]},
                {"input": [[1, 0, 0], [0, 0, 0]], "output": [[3, 0, 0], [0, 0, 0]]},
                {"input": [[0, 0, 1], [1, 0, 0]], "output": [[0, 0, 3], [3, 0, 0]]},
            ],
            "test": [
                {"input": [[1, 1, 0], [0, 0, 0]], "output": [[3, 3, 0], [0, 0, 0]]},
            ],
        }
        result = solve_with_object_rules(task)
        self.assertIsNotNone(result)
        # result should be a callable that transforms input → output
        test_output = result(task["test"][0]["input"])
        self.assertEqual(test_output, task["test"][0]["output"])

    def test_removal_task(self):
        """Full pipeline solves a removal task."""
        from arc_agent.scene import solve_with_object_rules
        task = {
            "train": [
                {"input": [[1, 2, 0], [0, 0, 0]], "output": [[0, 2, 0], [0, 0, 0]]},
                {"input": [[0, 2, 1], [1, 0, 0]], "output": [[0, 2, 0], [0, 0, 0]]},
            ],
            "test": [
                {"input": [[1, 0, 2], [0, 1, 0]], "output": [[0, 0, 2], [0, 0, 0]]},
            ],
        }
        result = solve_with_object_rules(task)
        self.assertIsNotNone(result)
        test_output = result(task["test"][0]["input"])
        self.assertEqual(test_output, task["test"][0]["output"])

    def test_returns_none_for_unsolvable(self):
        """Returns None when no consistent object rule is found."""
        from arc_agent.scene import solve_with_object_rules
        task = {
            "train": [
                {"input": [[1, 0], [0, 0]], "output": [[2, 0], [0, 0]]},
                {"input": [[1, 0], [0, 0]], "output": [[3, 0], [0, 0]]},  # Inconsistent
            ],
            "test": [
                {"input": [[1, 0], [0, 0]], "output": [[2, 0], [0, 0]]},
            ],
        }
        result = solve_with_object_rules(task)
        self.assertIsNone(result)


class TestGlobalColorMap(unittest.TestCase):
    """Test global pixel-level color mapping."""

    def test_simple_color_swap(self):
        """Global color mapping: 1→2, 3→4."""
        from arc_agent.scene import solve_with_object_rules
        task = {
            "train": [
                {"input": [[1, 3, 0], [0, 1, 3]], "output": [[2, 4, 0], [0, 2, 4]]},
                {"input": [[3, 0, 1], [1, 3, 0]], "output": [[4, 0, 2], [2, 4, 0]]},
            ],
            "test": [
                {"input": [[0, 1, 3], [3, 0, 1]], "output": [[0, 2, 4], [4, 0, 2]]},
            ],
        }
        result = solve_with_object_rules(task)
        self.assertIsNotNone(result)
        test_output = result(task["test"][0]["input"])
        self.assertEqual(test_output, task["test"][0]["output"])

    def test_non_deterministic_rejected(self):
        """Non-deterministic mapping is rejected."""
        from arc_agent.scene import solve_with_object_rules
        task = {
            "train": [
                {"input": [[1, 0], [0, 0]], "output": [[2, 0], [0, 0]]},
                {"input": [[1, 0], [0, 0]], "output": [[3, 0], [0, 0]]},  # 1→2 and 1→3
            ],
            "test": [{"input": [[1, 0], [0, 0]], "output": [[2, 0], [0, 0]]}],
        }
        result = solve_with_object_rules(task)
        self.assertIsNone(result)


class TestSizeConditionalRecolor(unittest.TestCase):
    """Test recoloring objects based on their size."""

    def test_size_determines_color(self):
        """Objects of same color get different colors based on size."""
        from arc_agent.scene import solve_with_object_rules
        # size=1 → color 2, size=2 → color 3
        task = {
            "train": [
                {
                    "input": [[5, 0, 0], [0, 5, 5], [0, 0, 0]],
                    "output": [[2, 0, 0], [0, 3, 3], [0, 0, 0]],
                },
                {
                    "input": [[0, 5, 0], [5, 5, 0], [0, 0, 5]],
                    "output": [[0, 3, 0], [3, 3, 0], [0, 0, 2]],  # size-3 obj → 3? No...
                },
            ],
            "test": [
                {
                    "input": [[5, 0, 5], [0, 0, 5], [0, 0, 0]],
                    "output": [[2, 0, 3], [0, 0, 3], [0, 0, 0]],
                },
            ],
        }
        result = solve_with_object_rules(task)
        self.assertIsNotNone(result)
        test_output = result(task["test"][0]["input"])
        self.assertEqual(test_output, task["test"][0]["output"])

    def test_real_size_pattern(self):
        """Pattern: size 4→1, size 3→2, size 2→3 (objects well-separated)."""
        from arc_agent.scene import solve_with_object_rules
        # Objects of color 5 get recolored based on size:
        # size 4 → color 1, size 3 → color 2, size 2 → color 3
        task = {
            "train": [
                {
                    # size-4 block (2x2), size-2 bar, size-3 bar (all separated)
                    "input": [
                        [5, 5, 0, 0, 0, 0],
                        [5, 5, 0, 0, 0, 0],
                        [0, 0, 0, 5, 5, 0],
                        [0, 0, 0, 0, 0, 0],
                        [0, 5, 5, 5, 0, 0],
                        [0, 0, 0, 0, 0, 0],
                    ],
                    "output": [
                        [1, 1, 0, 0, 0, 0],
                        [1, 1, 0, 0, 0, 0],
                        [0, 0, 0, 3, 3, 0],
                        [0, 0, 0, 0, 0, 0],
                        [0, 2, 2, 2, 0, 0],
                        [0, 0, 0, 0, 0, 0],
                    ],
                },
                {
                    # size-3 bar, size-4 row, size-2 bar
                    "input": [
                        [5, 5, 5, 0, 0],
                        [0, 0, 0, 0, 0],
                        [5, 5, 5, 5, 0],
                        [0, 0, 0, 0, 0],
                        [0, 0, 5, 5, 0],
                    ],
                    "output": [
                        [2, 2, 2, 0, 0],
                        [0, 0, 0, 0, 0],
                        [1, 1, 1, 1, 0],
                        [0, 0, 0, 0, 0],
                        [0, 0, 3, 3, 0],
                    ],
                },
            ],
            "test": [
                {
                    "input": [
                        [5, 5, 0, 0],
                        [0, 0, 0, 0],
                        [5, 5, 5, 0],
                        [0, 0, 0, 0],
                    ],
                    "output": [
                        [3, 3, 0, 0],
                        [0, 0, 0, 0],
                        [2, 2, 2, 0],
                        [0, 0, 0, 0],
                    ],
                },
            ],
        }
        result = solve_with_object_rules(task)
        self.assertIsNotNone(result)
        test_output = result(task["test"][0]["input"])
        self.assertEqual(test_output, task["test"][0]["output"])


if __name__ == "__main__":
    unittest.main()
