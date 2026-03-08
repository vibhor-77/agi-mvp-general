"""Unit tests for Cell Rule DSL (per-cell conditional transformations).

This module tests the cell rule system that enables context-dependent,
per-cell transformations. The DSL allows defining predicates (conditions)
and actions on individual cells, enabling more sophisticated patterns
than whole-grid operations.

Tests follow TDD principles:
  1. Cell predicates (test conditions on single cells)
  2. Cell actions (test transformations of single cells)
  3. CellRuleConcept (test rule application to entire grids)
  4. Integration with toolkit (test registration and search)
"""
import unittest
from arc_agent.cell_rules import (
    CellPredicate,
    CellAction,
    CellRule,
    CellRuleConcept,
    has_neighbor_color,
    is_color,
    is_zero,
    is_nonzero,
    is_border,
    count_neighbors_of_color,
    is_adjacent_to_nonzero,
    set_color,
    copy_neighbor_color,
    copy_neighbor_matching,
)
from arc_agent.concepts import Grid


class TestCellPredicates(unittest.TestCase):
    """Test individual cell predicate functions."""

    def setUp(self):
        """Set up test grids."""
        # Simple 3x3 grid for predicate testing
        # [1, 2, 1]
        # [2, 3, 2]
        # [1, 2, 1]
        self.grid_3x3 = [
            [1, 2, 1],
            [2, 3, 2],
            [1, 2, 1],
        ]

    def test_is_color_true(self):
        """Test is_color predicate returns True for matching color."""
        pred = is_color(2)
        self.assertTrue(pred(self.grid_3x3, 0, 1))  # (0,1) has color 2

    def test_is_color_false(self):
        """Test is_color predicate returns False for non-matching color."""
        pred = is_color(2)
        self.assertFalse(pred(self.grid_3x3, 0, 0))  # (0,0) has color 1, not 2

    def test_is_zero_true(self):
        """Test is_zero predicate returns True for zero cells."""
        grid = [[0, 1], [2, 0]]
        pred = is_zero()
        self.assertTrue(pred(grid, 0, 0))
        self.assertTrue(pred(grid, 1, 1))

    def test_is_zero_false(self):
        """Test is_zero predicate returns False for non-zero cells."""
        pred = is_zero()
        self.assertFalse(pred(self.grid_3x3, 0, 0))

    def test_is_nonzero_true(self):
        """Test is_nonzero predicate returns True for non-zero cells."""
        pred = is_nonzero()
        self.assertTrue(pred(self.grid_3x3, 0, 0))
        self.assertTrue(pred(self.grid_3x3, 1, 1))

    def test_is_nonzero_false(self):
        """Test is_nonzero predicate returns False for zero cells."""
        grid = [[0, 1], [2, 0]]
        pred = is_nonzero()
        self.assertFalse(pred(grid, 0, 0))
        self.assertFalse(pred(grid, 1, 1))

    def test_is_border_corners(self):
        """Test is_border predicate for corner cells."""
        pred = is_border()
        self.assertTrue(pred(self.grid_3x3, 0, 0))  # top-left
        self.assertTrue(pred(self.grid_3x3, 0, 2))  # top-right
        self.assertTrue(pred(self.grid_3x3, 2, 0))  # bottom-left
        self.assertTrue(pred(self.grid_3x3, 2, 2))  # bottom-right

    def test_is_border_edges(self):
        """Test is_border predicate for edge cells."""
        pred = is_border()
        self.assertTrue(pred(self.grid_3x3, 0, 1))   # top edge
        self.assertTrue(pred(self.grid_3x3, 1, 0))   # left edge
        self.assertTrue(pred(self.grid_3x3, 1, 2))   # right edge
        self.assertTrue(pred(self.grid_3x3, 2, 1))   # bottom edge

    def test_is_border_center(self):
        """Test is_border predicate returns False for center cells."""
        pred = is_border()
        self.assertFalse(pred(self.grid_3x3, 1, 1))  # center

    def test_has_neighbor_color_true(self):
        """Test has_neighbor_color predicate when neighbor exists."""
        pred = has_neighbor_color(2)
        # (1,1) center cell with color 3 has neighbors with color 2
        self.assertTrue(pred(self.grid_3x3, 1, 1))

    def test_has_neighbor_color_false(self):
        """Test has_neighbor_color predicate when neighbor doesn't exist."""
        pred = has_neighbor_color(9)  # No 9 in grid
        self.assertFalse(pred(self.grid_3x3, 1, 1))

    def test_count_neighbors_of_color_exact(self):
        """Test count_neighbors_of_color predicate for exact match."""
        pred = count_neighbors_of_color(2, exactly=4)
        # (1,1) center is surrounded by 4 cells of color 2
        self.assertTrue(pred(self.grid_3x3, 1, 1))

    def test_count_neighbors_of_color_at_least(self):
        """Test count_neighbors_of_color predicate with at_least."""
        pred = count_neighbors_of_color(2, at_least=3)
        self.assertTrue(pred(self.grid_3x3, 1, 1))

    def test_count_neighbors_of_color_at_most(self):
        """Test count_neighbors_of_color predicate with at_most."""
        pred = count_neighbors_of_color(2, at_most=4)
        self.assertTrue(pred(self.grid_3x3, 1, 1))

    def test_is_adjacent_to_nonzero_true(self):
        """Test is_adjacent_to_nonzero when neighbors are nonzero."""
        pred = is_adjacent_to_nonzero()
        # All cells in this grid are nonzero
        self.assertTrue(pred(self.grid_3x3, 0, 0))

    def test_is_adjacent_to_nonzero_false(self):
        """Test is_adjacent_to_nonzero when neighbors are zero."""
        grid_with_zeros = [
            [0, 0, 0],
            [0, 1, 0],
            [0, 0, 0],
        ]
        pred = is_adjacent_to_nonzero()
        # (1,1) has one nonzero value and only zero neighbors
        self.assertFalse(pred(grid_with_zeros, 0, 0))


class TestCellActions(unittest.TestCase):
    """Test individual cell action functions."""

    def setUp(self):
        """Set up test grids."""
        self.grid_3x3 = [
            [1, 2, 1],
            [2, 3, 2],
            [1, 2, 1],
        ]

    def test_set_color_action(self):
        """Test set_color action modifies cell."""
        action = set_color(5)
        grid_copy = [row[:] for row in self.grid_3x3]
        action(grid_copy, 0, 0)
        self.assertEqual(grid_copy[0][0], 5)
        # Check other cells unchanged
        self.assertEqual(grid_copy[0][1], 2)

    def test_copy_neighbor_color_up(self):
        """Test copy_neighbor_color from up neighbor."""
        action = copy_neighbor_color("up")
        grid_copy = [row[:] for row in self.grid_3x3]
        grid_copy[1][1] = 0  # Clear center
        action(grid_copy, 1, 1)  # Copy from cell above
        self.assertEqual(grid_copy[1][1], self.grid_3x3[0][1])

    def test_copy_neighbor_color_down(self):
        """Test copy_neighbor_color from down neighbor."""
        action = copy_neighbor_color("down")
        grid_copy = [row[:] for row in self.grid_3x3]
        grid_copy[1][1] = 0
        action(grid_copy, 1, 1)
        self.assertEqual(grid_copy[1][1], self.grid_3x3[2][1])

    def test_copy_neighbor_color_left(self):
        """Test copy_neighbor_color from left neighbor."""
        action = copy_neighbor_color("left")
        grid_copy = [row[:] for row in self.grid_3x3]
        grid_copy[1][1] = 0
        action(grid_copy, 1, 1)
        self.assertEqual(grid_copy[1][1], self.grid_3x3[1][0])

    def test_copy_neighbor_color_right(self):
        """Test copy_neighbor_color from right neighbor."""
        action = copy_neighbor_color("right")
        grid_copy = [row[:] for row in self.grid_3x3]
        grid_copy[1][1] = 0
        action(grid_copy, 1, 1)
        self.assertEqual(grid_copy[1][1], self.grid_3x3[1][2])

    def test_copy_neighbor_color_out_of_bounds(self):
        """Test copy_neighbor_color gracefully handles out-of-bounds."""
        action = copy_neighbor_color("up")
        grid_copy = [row[:] for row in self.grid_3x3]
        original_value = grid_copy[0][0]
        action(grid_copy, 0, 0)  # Try to copy from above first row
        # Should remain unchanged
        self.assertEqual(grid_copy[0][0], original_value)

    def test_copy_neighbor_matching_true(self):
        """Test copy_neighbor_matching finds and copies matching color."""
        action = copy_neighbor_matching(2)
        grid_copy = [row[:] for row in self.grid_3x3]
        grid_copy[0][0] = 0
        action(grid_copy, 0, 0)
        # (0,0) has neighbor (0,1) with color 2
        self.assertEqual(grid_copy[0][0], 2)

    def test_copy_neighbor_matching_no_match(self):
        """Test copy_neighbor_matching when no neighbor matches."""
        action = copy_neighbor_matching(9)  # No 9 in grid
        grid_copy = [row[:] for row in self.grid_3x3]
        original_value = grid_copy[0][0]
        action(grid_copy, 0, 0)
        # Should remain unchanged
        self.assertEqual(grid_copy[0][0], original_value)


class TestCellRule(unittest.TestCase):
    """Test CellRule data structure."""

    def test_cell_rule_creation(self):
        """Test basic CellRule creation."""
        pred = is_color(1)
        action = set_color(2)
        rule = CellRule(predicate=pred, action=action)
        self.assertEqual(rule.predicate, pred)
        self.assertEqual(rule.action, action)

    def test_cell_rule_name_generation(self):
        """Test auto-generated name for CellRule."""
        pred = is_color(1)
        action = set_color(2)
        rule = CellRule(predicate=pred, action=action)
        self.assertIsInstance(rule.name, str)
        self.assertGreater(len(rule.name), 0)

    def test_cell_rule_custom_name(self):
        """Test custom name for CellRule."""
        pred = is_color(1)
        action = set_color(2)
        rule = CellRule(predicate=pred, action=action, name="custom_rule")
        self.assertEqual(rule.name, "custom_rule")


class TestCellRuleConcept(unittest.TestCase):
    """Test CellRuleConcept application to grids."""

    def setUp(self):
        """Set up test grids."""
        self.grid_3x3 = [
            [1, 2, 1],
            [2, 3, 2],
            [1, 2, 1],
        ]

    def test_cell_rule_concept_apply_single_rule(self):
        """Test CellRuleConcept applies rule to all matching cells."""
        # Rule: if cell is color 1, change to color 5
        pred = is_color(1)
        action = set_color(5)
        rule = CellRule(predicate=pred, action=action)
        concept = CellRuleConcept([rule])

        result = concept.apply(self.grid_3x3)
        self.assertIsNotNone(result)

        # All 1s should become 5s
        self.assertEqual(result[0][0], 5)
        self.assertEqual(result[0][2], 5)
        self.assertEqual(result[2][0], 5)
        self.assertEqual(result[2][2], 5)

        # Other cells unchanged
        self.assertEqual(result[0][1], 2)
        self.assertEqual(result[1][1], 3)

    def test_cell_rule_concept_multiple_rules(self):
        """Test CellRuleConcept applies multiple rules in sequence."""
        # Rule 1: if color 1, change to 5
        rule1 = CellRule(is_color(1), set_color(5))
        # Rule 2: if color 2, change to 6
        rule2 = CellRule(is_color(2), set_color(6))
        concept = CellRuleConcept([rule1, rule2])

        result = concept.apply(self.grid_3x3)
        self.assertIsNotNone(result)

        # Check transformations
        self.assertEqual(result[0][0], 5)  # Was 1
        self.assertEqual(result[0][1], 6)  # Was 2
        self.assertEqual(result[1][1], 3)  # Was 3, unchanged

    def test_cell_rule_concept_copy_from_neighbor(self):
        """Test CellRuleConcept with neighbor copying."""
        grid = [
            [1, 2, 1],
            [2, 0, 2],  # Center is 0
            [1, 2, 1],
        ]

        # Rule: if cell is 0, copy from any neighbor with color 2
        rule = CellRule(is_color(0), copy_neighbor_matching(2))
        concept = CellRuleConcept([rule])

        result = concept.apply(grid)
        self.assertIsNotNone(result)
        self.assertEqual(result[1][1], 2)  # Center should be 2

    def test_cell_rule_concept_preserves_grid_dimensions(self):
        """Test that CellRuleConcept preserves grid dimensions."""
        rule = CellRule(is_color(1), set_color(5))
        concept = CellRuleConcept([rule])
        result = concept.apply(self.grid_3x3)

        self.assertEqual(len(result), len(self.grid_3x3))
        self.assertEqual(len(result[0]), len(self.grid_3x3[0]))

    def test_cell_rule_concept_name(self):
        """Test CellRuleConcept has meaningful name."""
        rule = CellRule(is_color(1), set_color(5), name="test_rule")
        concept = CellRuleConcept([rule])
        self.assertIn("cell_rule", concept.name.lower())

    def test_cell_rule_concept_kind(self):
        """Test CellRuleConcept has correct kind."""
        rule = CellRule(is_color(1), set_color(5))
        concept = CellRuleConcept([rule])
        self.assertEqual(concept.kind, "cell_rule")

    def test_cell_rule_concept_border_predicate(self):
        """Test CellRuleConcept with border predicate."""
        # Color borders with 9
        rule = CellRule(is_border(), set_color(9))
        concept = CellRuleConcept([rule])

        result = concept.apply(self.grid_3x3)
        self.assertIsNotNone(result)

        # Borders should be 9
        self.assertEqual(result[0][0], 9)  # top-left
        self.assertEqual(result[0][1], 9)  # top-edge
        self.assertEqual(result[1][0], 9)  # left-edge
        self.assertEqual(result[1][1], 3)  # center unchanged


class TestCellRuleEdgeCases(unittest.TestCase):
    """Test edge cases and error handling."""

    def test_single_cell_grid(self):
        """Test cell rules on 1x1 grid."""
        grid = [[5]]
        rule = CellRule(is_color(5), set_color(7))
        concept = CellRuleConcept([rule])
        result = concept.apply(grid)
        self.assertEqual(result[0][0], 7)

    def test_single_row_grid(self):
        """Test cell rules on 1xN grid."""
        grid = [[1, 2, 3]]
        rule = CellRule(is_color(2), set_color(9))
        concept = CellRuleConcept([rule])
        result = concept.apply(grid)
        self.assertEqual(result[0][1], 9)

    def test_single_column_grid(self):
        """Test cell rules on Nx1 grid."""
        grid = [[1], [2], [3]]
        rule = CellRule(is_color(2), set_color(9))
        concept = CellRuleConcept([rule])
        result = concept.apply(grid)
        self.assertEqual(result[1][0], 9)

    def test_empty_rules_list(self):
        """Test CellRuleConcept with no rules applies identity."""
        grid = [[1, 2], [3, 4]]
        concept = CellRuleConcept([])
        result = concept.apply(grid)
        self.assertEqual(result, grid)

    def test_complex_predicate_chain(self):
        """Test complex predicates with multiple conditions."""
        grid = [
            [0, 0, 0],
            [0, 1, 0],
            [0, 0, 0],
        ]

        # Rule: if cell is 0 and is NOT border, change to 5
        # (Only center can be 0 and not-border, but center is 1 not 0)
        # So nothing changes
        rule = CellRule(
            is_color(0),
            set_color(5)
        )
        concept = CellRuleConcept([rule])
        result = concept.apply(grid)

        # All 0s should become 5s
        for i in range(3):
            for j in range(3):
                if grid[i][j] == 0:
                    self.assertEqual(result[i][j], 5)


class TestCellRuleIntegration(unittest.TestCase):
    """Integration tests with Concept system."""

    def test_cell_rule_concept_is_concept(self):
        """Test that CellRuleConcept is a proper Concept."""
        from arc_agent.concepts import Concept
        rule = CellRule(is_color(1), set_color(2))
        concept = CellRuleConcept([rule])
        self.assertIsInstance(concept, Concept)

    def test_cell_rule_concept_apply_method(self):
        """Test that CellRuleConcept.apply method works correctly."""
        grid = [[1, 2, 3]]
        rule = CellRule(is_color(2), set_color(9))
        concept = CellRuleConcept([rule])

        result = concept.apply(grid)
        self.assertIsNotNone(result)
        self.assertEqual(result[0][1], 9)

    def test_cell_rule_concept_usage_tracking(self):
        """Test that CellRuleConcept tracks usage."""
        grid = [[1, 2, 3]]
        rule = CellRule(is_color(2), set_color(9))
        concept = CellRuleConcept([rule])

        self.assertEqual(concept.usage_count, 0)
        concept.apply(grid)
        self.assertEqual(concept.usage_count, 1)

    def test_cell_rule_concept_success_tracking(self):
        """Test that CellRuleConcept tracks success."""
        grid = [[1, 2, 3]]
        rule = CellRule(is_color(2), set_color(9))
        concept = CellRuleConcept([rule])

        concept.apply(grid)
        concept.reinforce(True)
        self.assertEqual(concept.success_count, 1)


if __name__ == '__main__':
    unittest.main()
