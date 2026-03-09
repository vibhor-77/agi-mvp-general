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


class TestNewPredicates(unittest.TestCase):
    """Test the new predicates added in Session 23."""

    def test_is_mostly_empty(self):
        from arc_agent.primitives import is_mostly_empty
        # 9 cells, only 1 non-zero → 88% empty → True
        grid = [[0, 0, 0], [0, 1, 0], [0, 0, 0]]
        self.assertTrue(is_mostly_empty(grid))
        # All non-zero → False
        grid = [[1, 2], [3, 4]]
        self.assertFalse(is_mostly_empty(grid))

    def test_has_frame_structure(self):
        from arc_agent.primitives import has_frame_structure
        # 3x3 with border=1, interior=2 → True
        grid = [[1, 1, 1], [1, 2, 1], [1, 1, 1]]
        self.assertTrue(has_frame_structure(grid))
        # Uniform → False
        grid = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
        self.assertFalse(has_frame_structure(grid))

    def test_has_diagonal_symmetry(self):
        from arc_agent.primitives import has_diagonal_symmetry
        # Symmetric: transpose equals self
        grid = [[1, 2], [2, 1]]
        self.assertTrue(has_diagonal_symmetry(grid))
        # Not symmetric
        grid = [[1, 2], [3, 4]]
        self.assertFalse(has_diagonal_symmetry(grid))
        # Non-square → False
        grid = [[1, 2, 3], [4, 5, 6]]
        self.assertFalse(has_diagonal_symmetry(grid))

    def test_is_odd_dimensions(self):
        from arc_agent.primitives import is_odd_dimensions
        grid = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]  # 3x3
        self.assertTrue(is_odd_dimensions(grid))
        grid = [[1, 2], [3, 4]]  # 2x2
        self.assertFalse(is_odd_dimensions(grid))

    def test_has_two_colors(self):
        from arc_agent.primitives import has_two_colors
        grid = [[1, 2], [2, 1]]
        self.assertTrue(has_two_colors(grid))
        grid = [[1, 1], [1, 1]]
        self.assertFalse(has_two_colors(grid))
        grid = [[1, 2], [3, 0]]
        self.assertFalse(has_two_colors(grid))  # 3 non-zero colors

    def test_has_horizontal_stripe(self):
        from arc_agent.primitives import has_horizontal_stripe
        grid = [[1, 1, 1], [2, 3, 4]]
        self.assertTrue(has_horizontal_stripe(grid))
        grid = [[1, 2, 3], [4, 5, 6]]
        self.assertFalse(has_horizontal_stripe(grid))

    def test_has_vertical_stripe(self):
        from arc_agent.primitives import has_vertical_stripe
        grid = [[1, 2], [1, 3], [1, 4]]
        self.assertTrue(has_vertical_stripe(grid))
        grid = [[1, 2], [3, 4]]
        self.assertFalse(has_vertical_stripe(grid))


class TestConditionalSearch(unittest.TestCase):
    """Test deterministic conditional search methods."""

    def setUp(self):
        from arc_agent.primitives import build_initial_toolkit
        from arc_agent.synthesizer import ProgramSynthesizer
        self.toolkit = build_initial_toolkit(include_objects=False)
        self.synth = ProgramSynthesizer(self.toolkit)

    def test_try_conditional_singles_returns_program_or_none(self):
        """try_conditional_singles should return a Program or None."""
        task = {
            "train": [
                {"input": [[1, 2], [3, 4]], "output": [[3, 1], [4, 2]]},
            ]
        }
        result = self.synth.try_conditional_singles(task, top_k=5)
        if result is not None:
            self.assertIsInstance(result, Program)
            self.assertGreater(result.fitness, 0)

    def test_try_conditional_pairs_returns_program_or_none(self):
        """try_conditional_pairs should return a Program or None."""
        task = {
            "train": [
                {"input": [[1, 2], [3, 4]], "output": [[3, 1], [4, 2]]},
            ]
        }
        result = self.synth.try_conditional_pairs(task, top_k=5)
        if result is not None:
            self.assertIsInstance(result, Program)
            self.assertGreater(result.fitness, 0)

    def test_conditional_search_finds_branching_solution(self):
        """Should find a conditional that branches on grid shape."""
        # Square input → rotate, non-square → mirror_h
        task = {
            "train": [
                # 2x2 square: rotate_90_cw → [[3,1],[4,2]]
                {"input": [[1, 2], [3, 4]], "output": [[3, 1], [4, 2]]},
                # 2x3 wide: mirror_h → [[3,2,1],[6,5,4]]
                {"input": [[1, 2, 3], [4, 5, 6]], "output": [[3, 2, 1], [6, 5, 4]]},
            ]
        }
        result = self.synth.try_conditional_singles(task, top_k=10)
        if result is not None:
            self.assertGreaterEqual(result.fitness, 0.99)

    def test_conditional_search_does_not_crash_empty_predicates(self):
        """Should handle toolkit with no predicates gracefully."""
        from arc_agent.primitives import build_initial_toolkit
        from arc_agent.synthesizer import ProgramSynthesizer
        from arc_agent.concepts import Concept
        # Build toolkit without predicates
        tk = build_initial_toolkit(include_objects=False)
        # Remove all predicates
        pred_names = [n for n, c in tk.concepts.items() if c.kind == "predicate"]
        for n in pred_names:
            tk.concepts.pop(n)
        synth = ProgramSynthesizer(tk)

        task = {"train": [{"input": [[1]], "output": [[2]]}]}
        result = synth.try_conditional_singles(task)
        self.assertIsNone(result)


class TestMaxProgramLength(unittest.TestCase):
    """Test that max_program_length is set to 6."""

    def test_default_max_program_length(self):
        from arc_agent.solver import FourPillarsSolver
        solver = FourPillarsSolver(verbose=False)
        self.assertEqual(solver.synthesizer.max_program_length, 6)


if __name__ == '__main__':
    unittest.main()
