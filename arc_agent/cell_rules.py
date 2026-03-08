"""Cell Rule DSL: Per-Cell Conditional Transformations

This module implements a Domain-Specific Language (DSL) for cell-level
conditional transformations. Rather than applying whole-grid operations,
cell rules enable context-dependent changes to individual cells based on
their properties and neighbors.

Architecture:
  - CellPredicate: Callable that tests a condition on a single cell
  - CellAction: Callable that transforms a single cell
  - CellRule: Pairs a predicate with an action
  - CellRuleConcept: Wraps CellRule(s) as a reusable Concept

Example usage:
  # Color all border cells with 9
  rule = CellRule(is_border(), set_color(9))
  concept = CellRuleConcept([rule])
  result = concept.apply(grid)

  # Color cells with color 0 based on their neighbors
  rule = CellRule(
    is_color(0),
    copy_neighbor_matching(2)  # Copy color 2 from neighbor if exists
  )
  concept = CellRuleConcept([rule])
  result = concept.apply(grid)
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Optional, Any
from .concepts import Concept, Grid


# Type aliases for clarity
CellPredicate = Callable[['Grid', int, int], bool]
CellAction = Callable[['Grid', int, int], None]


# ============================================================
# CELL PREDICATES: Conditions on individual cells
# ============================================================

def is_zero() -> CellPredicate:
    """Create a predicate that checks if a cell is zero (empty).

    Returns:
        A predicate function that returns True if cell value is 0
    """
    def predicate(grid: Grid, row: int, col: int) -> bool:
        return grid[row][col] == 0
    return predicate


def is_nonzero() -> CellPredicate:
    """Create a predicate that checks if a cell is non-zero.

    Returns:
        A predicate function that returns True if cell value != 0
    """
    def predicate(grid: Grid, row: int, col: int) -> bool:
        return grid[row][col] != 0
    return predicate


def is_color(color: int) -> CellPredicate:
    """Create a predicate that checks if a cell has a specific color.

    Args:
        color: The color value to match

    Returns:
        A predicate function that takes (grid, row, col) and returns
        True if grid[row][col] == color, False otherwise
    """
    def predicate(grid: Grid, row: int, col: int) -> bool:
        return grid[row][col] == color
    return predicate


def is_border() -> CellPredicate:
    """Create a predicate that checks if a cell is on the border.

    A cell is on the border if it's in the first/last row or
    first/last column.

    Returns:
        A predicate function that takes (grid, row, col) and returns
        True if the cell is on any border
    """
    def predicate(grid: Grid, row: int, col: int) -> bool:
        h = len(grid)
        w = len(grid[0]) if h > 0 else 0
        return row == 0 or row == h - 1 or col == 0 or col == w - 1
    return predicate


def has_neighbor_color(color: int) -> CellPredicate:
    """Create a predicate that checks if a cell has a neighbor with
    a specific color.

    Neighbors are the 4-adjacent cells (up, down, left, right).

    Args:
        color: The color to look for in neighbors

    Returns:
        A predicate function that returns True if any 4-adjacent neighbor
        has the specified color
    """
    def predicate(grid: Grid, row: int, col: int) -> bool:
        h = len(grid)
        w = len(grid[0]) if h > 0 else 0

        # Check all 4 neighbors
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = row + dr, col + dc
            if 0 <= nr < h and 0 <= nc < w:
                if grid[nr][nc] == color:
                    return True
        return False
    return predicate


def count_neighbors_of_color(
    color: int,
    exactly: Optional[int] = None,
    at_least: Optional[int] = None,
    at_most: Optional[int] = None,
) -> CellPredicate:
    """Create a predicate that checks neighbor count of a specific color.

    Args:
        color: The color to count in neighbors
        exactly: If set, predicate returns True if exactly this many neighbors
        at_least: If set, predicate returns True if at least this many neighbors
        at_most: If set, predicate returns True if at most this many neighbors

    Returns:
        A predicate function that counts 4-adjacent neighbors with the color
    """
    def predicate(grid: Grid, row: int, col: int) -> bool:
        h = len(grid)
        w = len(grid[0]) if h > 0 else 0
        count = 0

        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = row + dr, col + dc
            if 0 <= nr < h and 0 <= nc < w and grid[nr][nc] == color:
                count += 1

        if exactly is not None:
            return count == exactly
        if at_least is not None and count < at_least:
            return False
        if at_most is not None and count > at_most:
            return False
        return True
    return predicate


def is_adjacent_to_nonzero() -> CellPredicate:
    """Create a predicate that checks if a cell has any nonzero neighbors.

    Neighbors are the 4-adjacent cells (up, down, left, right).

    Returns:
        A predicate function that returns True if any 4-adjacent neighbor
        is nonzero
    """
    def predicate(grid: Grid, row: int, col: int) -> bool:
        h = len(grid)
        w = len(grid[0]) if h > 0 else 0

        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = row + dr, col + dc
            if 0 <= nr < h and 0 <= nc < w and grid[nr][nc] != 0:
                return True
        return False
    return predicate


# ============================================================
# CELL ACTIONS: Operations on individual cells
# ============================================================

def set_color(color: int) -> CellAction:
    """Create an action that sets a cell to a specific color.

    Args:
        color: The color value to set

    Returns:
        An action function that takes (grid, row, col) and sets
        grid[row][col] = color
    """
    def action(grid: Grid, row: int, col: int) -> None:
        grid[row][col] = color
    return action


def copy_neighbor_color(direction: str) -> CellAction:
    """Create an action that copies a color from a specific neighbor.

    Args:
        direction: One of "up", "down", "left", "right"

    Returns:
        An action function that copies the neighbor's color if the
        neighbor exists, otherwise does nothing
    """
    direction_map = {
        "up": (-1, 0),
        "down": (1, 0),
        "left": (0, -1),
        "right": (0, 1),
    }

    if direction not in direction_map:
        raise ValueError(f"Invalid direction: {direction}")

    dr, dc = direction_map[direction]

    def action(grid: Grid, row: int, col: int) -> None:
        h = len(grid)
        w = len(grid[0]) if h > 0 else 0
        nr, nc = row + dr, col + dc

        if 0 <= nr < h and 0 <= nc < w:
            grid[row][col] = grid[nr][nc]
    return action


def copy_neighbor_matching(color: int) -> CellAction:
    """Create an action that copies a specific color from any neighbor.

    Searches all 4-adjacent neighbors and copies the first one with
    the specified color.

    Args:
        color: The color to search for and copy

    Returns:
        An action function that copies the color if found in any neighbor
    """
    def action(grid: Grid, row: int, col: int) -> None:
        h = len(grid)
        w = len(grid[0]) if h > 0 else 0

        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = row + dr, col + dc
            if 0 <= nr < h and 0 <= nc < w and grid[nr][nc] == color:
                grid[row][col] = color
                return
    return action


# ============================================================
# CELL RULE: Data structure pairing predicate + action
# ============================================================

@dataclass
class CellRule:
    """A rule that applies an action to cells matching a predicate.

    Attributes:
        predicate: CellPredicate that tests a condition
        action: CellAction to apply if predicate is True
        name: Optional custom name for the rule
    """
    predicate: CellPredicate
    action: CellAction
    name: str = ""

    def __post_init__(self):
        """Auto-generate name if not provided."""
        if not self.name:
            pred_name = getattr(self.predicate, '__name__', 'pred')
            action_name = getattr(self.action, '__name__', 'action')
            self.name = f"rule_{pred_name}→{action_name}"


# ============================================================
# CELL RULE CONCEPT: Applies rules to entire grids
# ============================================================

class CellRuleConcept(Concept):
    """A Concept that applies a sequence of cell rules to a grid.

    Each rule is applied to every cell in the grid. Multiple rules
    can be composed in sequence, allowing sophisticated multi-step
    transformations.

    Example:
        # Rule 1: Change all 0s to 5s
        rule1 = CellRule(is_color(0), set_color(5))
        # Rule 2: Copy color from neighbors for remaining 0s
        rule2 = CellRule(is_color(0), copy_neighbor_matching(2))
        # Apply both rules
        concept = CellRuleConcept([rule1, rule2])
        result = concept.apply(grid)
    """

    def __init__(self, rules: list[CellRule], name: str = ""):
        """Initialize a CellRuleConcept.

        Args:
            rules: List of CellRule objects to apply in sequence
            name: Optional custom name (auto-generated if not provided)
        """
        self.rules = rules

        if not name:
            rule_names = [r.name for r in rules]
            name = f"cell_rules({', '.join(rule_names)})"

        super().__init__(
            kind="cell_rule",
            name=name,
            implementation=self._apply_rules,
        )

    def _apply_rules(self, grid: Grid) -> Grid:
        """Apply all rules to the grid.

        For each rule, iterate through all cells and apply the action
        if the predicate returns True.

        Args:
            grid: The input grid

        Returns:
            The transformed grid
        """
        # Deep copy to avoid modifying input
        result = [row[:] for row in grid]
        h = len(result)
        w = len(result[0]) if h > 0 else 0

        # Apply each rule
        for rule in self.rules:
            for row in range(h):
                for col in range(w):
                    # Test predicate on this cell
                    if rule.predicate(result, row, col):
                        # Apply action
                        rule.action(result, row, col)

        return result

    def apply(self, grid: Grid) -> Optional[Grid]:
        """Apply this concept to a grid.

        Overrides parent to track usage and call implementation.

        Args:
            grid: The input grid

        Returns:
            The transformed grid, or None on failure
        """
        try:
            self.usage_count += 1
            result = self._apply_rules(grid)
            if result is not None:
                return result
        except Exception:
            pass
        return None


# ============================================================
# CELL RULE FACTORY FUNCTIONS: For common patterns
# ============================================================

def make_border_color_rule(color: int) -> CellRule:
    """Factory: create a rule that colors all border cells.

    Args:
        color: The color to apply to borders

    Returns:
        A CellRule that sets all border cells to the specified color
    """
    return CellRule(
        is_border(),
        set_color(color),
        name=f"color_border_{color}"
    )


def make_swap_rule(from_color: int, to_color: int) -> CellRule:
    """Factory: create a rule that swaps one color for another.

    Args:
        from_color: The original color
        to_color: The replacement color

    Returns:
        A CellRule that changes from_color to to_color
    """
    return CellRule(
        is_color(from_color),
        set_color(to_color),
        name=f"swap_{from_color}→{to_color}"
    )


def make_fill_from_neighbors_rule(
    target_color: int,
    source_color: int,
) -> CellRule:
    """Factory: create a rule that fills target color from neighbors.

    Useful for "flood fill" or "spread from neighbors" patterns.

    Args:
        target_color: The color to fill (usually 0 for empty)
        source_color: The color to copy from neighbors

    Returns:
        A CellRule that changes target_color to source_color if
        a neighbor has source_color
    """
    return CellRule(
        is_color(target_color),
        copy_neighbor_matching(source_color),
        name=f"fill_{target_color}_from_{source_color}"
    )
