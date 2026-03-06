"""Unit tests for conditional logic (Feature 1: Conditional Logic)."""
import unittest
from arc_agent.concepts import Concept, ConditionalConcept, Program, Grid
from arc_agent.primitives import (
    identity, rotate_90_cw, mirror_horizontal,
    is_square, is_symmetric_h, is_tall, is_wide,
    has_single_color, has_many_colors, is_small, is_large,
    has_background_majority,
)


class TestPredicates(unittest.TestCase):
    """Test the new predicates for conditional logic."""

    def test_is_tall(self):
        """Grid with height > width should return True."""
        grid = [[1, 2], [3, 4], [5, 6]]  # 3x2
        self.assertTrue(is_tall(grid))

        grid = [[1, 2, 3], [4, 5, 6]]  # 2x3
        self.assertFalse(is_tall(grid))

    def test_is_wide(self):
        """Grid with width > height should return True."""
        grid = [[1, 2, 3], [4, 5, 6]]  # 2x3
        self.assertTrue(is_wide(grid))

        grid = [[1, 2], [3, 4], [5, 6]]  # 3x2
        self.assertFalse(is_wide(grid))

    def test_has_many_colors(self):
        """Grid with more than 3 non-zero colors should return True."""
        grid = [[1, 2, 3, 4], [5, 6, 7, 8]]  # 8 colors
        self.assertTrue(has_many_colors(grid))

        grid = [[1, 2, 3], [1, 2, 3]]  # 3 colors
        self.assertFalse(has_many_colors(grid))

        grid = [[1], [2], [3]]  # 3 colors
        self.assertFalse(has_many_colors(grid))

    def test_is_small(self):
        """Grid with < 50 total cells should return True."""
        grid = [[1] * 7 for _ in range(7)]  # 49 cells
        self.assertTrue(is_small(grid))

        grid = [[1] * 8 for _ in range(8)]  # 64 cells
        self.assertFalse(is_small(grid))

    def test_is_large(self):
        """Grid with > 200 total cells should return True."""
        grid = [[1] * 15 for _ in range(15)]  # 225 cells
        self.assertTrue(is_large(grid))

        grid = [[1] * 10 for _ in range(10)]  # 100 cells
        self.assertFalse(is_large(grid))

    def test_has_background_majority(self):
        """Grid with > 50% background (0) cells should return True."""
        # Most cells are 0
        grid = [[0, 0, 0], [0, 0, 1], [0, 1, 1]]  # 7/9 zeros
        self.assertTrue(has_background_majority(grid))

        # Most cells are non-zero
        grid = [[1, 1, 1], [1, 1, 0], [1, 0, 0]]  # 4/9 zeros
        self.assertFalse(has_background_majority(grid))


class TestConditionalConcept(unittest.TestCase):
    """Test the ConditionalConcept class."""

    def test_conditional_applies_then_branch(self):
        """If predicate is True, should apply then_concept."""
        # Create predicates and concepts
        def is_even_height(grid):
            return len(grid) % 2 == 0

        then_c = Concept(kind="operator", name="rotate", implementation=rotate_90_cw)
        else_c = Concept(kind="operator", name="identity", implementation=identity)

        cond = ConditionalConcept(is_even_height, then_c, else_c)

        # Grid with even height (2 rows) -> should rotate
        grid = [[1, 2], [3, 4]]
        result = cond.apply(grid)
        expected = [[3, 1], [4, 2]]
        self.assertEqual(result, expected)

    def test_conditional_applies_else_branch(self):
        """If predicate is False, should apply else_concept."""
        def is_even_height(grid):
            return len(grid) % 2 == 0

        then_c = Concept(kind="operator", name="rotate", implementation=rotate_90_cw)
        else_c = Concept(kind="operator", name="identity", implementation=identity)

        cond = ConditionalConcept(is_even_height, then_c, else_c)

        # Grid with odd height (1 row) -> should apply identity
        grid = [[1, 2, 3]]
        result = cond.apply(grid)
        self.assertEqual(result, [[1, 2, 3]])

    def test_conditional_with_is_square(self):
        """Test conditional using is_square predicate."""
        then_c = Concept(kind="operator", name="rotate", implementation=rotate_90_cw)
        else_c = Concept(kind="operator", name="mirror", implementation=mirror_horizontal)

        cond = ConditionalConcept(is_square, then_c, else_c)

        # Square grid (2x2) -> rotate
        grid = [[1, 2], [3, 4]]
        result = cond.apply(grid)
        self.assertEqual(result, [[3, 1], [4, 2]])

        # Non-square grid (2x3) -> mirror
        grid = [[1, 2, 3], [4, 5, 6]]
        result = cond.apply(grid)
        self.assertEqual(result, [[3, 2, 1], [6, 5, 4]])

    def test_conditional_usage_count(self):
        """ConditionalConcept should increment usage_count on apply."""
        cond = ConditionalConcept(is_square,
                                   Concept(kind="op", name="r", implementation=rotate_90_cw),
                                   Concept(kind="op", name="i", implementation=identity))
        self.assertEqual(cond.usage_count, 0)

        grid = [[1, 2], [3, 4]]
        cond.apply(grid)
        self.assertEqual(cond.usage_count, 1)

        cond.apply(grid)
        self.assertEqual(cond.usage_count, 2)

    def test_conditional_returns_none_on_exception(self):
        """ConditionalConcept should return None if implementation raises."""
        def bad_predicate(grid):
            raise ValueError("broken predicate")

        cond = ConditionalConcept(bad_predicate,
                                   Concept(kind="op", name="r", implementation=rotate_90_cw),
                                   Concept(kind="op", name="i", implementation=identity))
        grid = [[1, 2], [3, 4]]
        result = cond.apply(grid)
        self.assertIsNone(result)

    def test_conditional_name_generation(self):
        """ConditionalConcept should auto-generate name from components."""
        then_c = Concept(kind="operator", name="rotate", implementation=rotate_90_cw)
        else_c = Concept(kind="operator", name="identity", implementation=identity)

        cond = ConditionalConcept(is_square, then_c, else_c)
        self.assertIn("is_square", cond.name)
        self.assertIn("rotate", cond.name)
        self.assertIn("identity", cond.name)

    def test_conditional_custom_name(self):
        """ConditionalConcept should use provided name if given."""
        then_c = Concept(kind="operator", name="rotate", implementation=rotate_90_cw)
        else_c = Concept(kind="operator", name="identity", implementation=identity)

        cond = ConditionalConcept(is_square, then_c, else_c, name="custom_cond")
        self.assertEqual(cond.name, "custom_cond")

    def test_conditional_kind(self):
        """ConditionalConcept should have kind 'conditional'."""
        cond = ConditionalConcept(is_square,
                                   Concept(kind="op", name="r", implementation=rotate_90_cw),
                                   Concept(kind="op", name="i", implementation=identity))
        self.assertEqual(cond.kind, "conditional")

    def test_conditional_in_program(self):
        """ConditionalConcept should work as a step in a Program."""
        then_c = Concept(kind="operator", name="rotate", implementation=rotate_90_cw)
        else_c = Concept(kind="operator", name="identity", implementation=identity)
        cond = ConditionalConcept(is_square, then_c, else_c)

        identity_c = Concept(kind="operator", name="id", implementation=identity)

        # Create a program with a conditional step
        program = Program([cond, identity_c])

        grid = [[1, 2], [3, 4]]
        result = program.execute(grid)

        # Should apply conditional (rotate), then identity
        expected = [[3, 1], [4, 2]]
        self.assertEqual(result, expected)

    def test_conditional_children_stored(self):
        """ConditionalConcept should store branch concepts as children."""
        then_c = Concept(kind="operator", name="rotate", implementation=rotate_90_cw)
        else_c = Concept(kind="operator", name="identity", implementation=identity)
        cond = ConditionalConcept(is_square, then_c, else_c)

        self.assertEqual(len(cond.children), 2)
        self.assertIs(cond.children[0], then_c)
        self.assertIs(cond.children[1], else_c)


if __name__ == '__main__':
    unittest.main()
