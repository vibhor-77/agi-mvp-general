"""Domain-Specific Language for Grid Transformation Synthesis

A typed DSL with ~20 atomic operations and combinators for expressing
grid transformations as composable expression trees. Programs are
represented as trees (not strings), enabling structural mutation and
bottom-up enumeration.

This is the "assembly language" underneath our 287 macro primitives.
Instead of searching over fixed primitive sequences, the synthesis
engine searches over DSL expression trees to construct novel transforms.

Architecture:
  DSLType    — Enum of types (GRID, COLOR, INT, BOOL, COLOR_MAP)
  DSLExpr    — Expression tree node (op, args, return_type, value)
  DSLInterpreter — Recursive descent evaluator for expression trees

Design principles:
  - Pure functions: no side effects, no mutation of input grids
  - Type-checked at construction: catch invalid compositions early
  - Minimal but sufficient: ~20 ops cover most ARC patterns
  - Composable: output of any Grid→Grid expr can feed into another
"""
from __future__ import annotations

from collections import Counter
from enum import Enum, auto
from typing import Any, Optional

from .concepts import Grid


class DSLType(Enum):
    """Types in the DSL type system."""
    GRID = auto()
    COLOR = auto()       # int 0-9
    INT = auto()
    BOOL = auto()
    COLOR_MAP = auto()   # dict[int, int]
    LAMBDA = auto()      # a callable transform (for map_objects)


class DSLExpr:
    """An expression tree node in the DSL.

    Each node has:
      op          — operation name (str)
      args        — list of child DSLExpr nodes
      return_type — DSLType of this expression's result
      value       — literal value (only for op="literal" or op="input_grid")

    Expression trees are immutable once constructed. The interpreter
    evaluates them recursively, substituting the input grid for
    "input_grid" nodes.
    """

    __slots__ = ("op", "args", "return_type", "value")

    def __init__(self, op: str, args: list[DSLExpr],
                 return_type: DSLType, value: Any = None):
        self.op = op
        self.args = args
        self.return_type = return_type
        self.value = value

    # ---- Factory methods ----

    @staticmethod
    def literal(value: Any, typ: DSLType) -> DSLExpr:
        """Create a literal constant node."""
        return DSLExpr("literal", [], typ, value=value)

    @staticmethod
    def input_grid() -> DSLExpr:
        """Create the input grid reference node."""
        return DSLExpr("input_grid", [], DSLType.GRID)

    @staticmethod
    def make_op(name: str, args: list[DSLExpr], return_type: DSLType) -> DSLExpr:
        """Create an operation node."""
        return DSLExpr(name, args, return_type)

    @staticmethod
    def lambda_expr(name: str, args: list[DSLExpr],
                    return_type: DSLType) -> DSLExpr:
        """Create a lambda (deferred transform) node for map_objects."""
        return DSLExpr(name, args, DSLType.LAMBDA, value=return_type)

    # ---- Tree metrics ----

    @property
    def depth(self) -> int:
        """Maximum depth of this expression tree."""
        if not self.args:
            return 0
        return 1 + max(a.depth for a in self.args)

    @property
    def size(self) -> int:
        """Total number of nodes in this expression tree."""
        return 1 + sum(a.size for a in self.args)

    def __repr__(self) -> str:
        if self.op == "literal":
            return f"Lit({self.value})"
        if self.op == "input_grid":
            return "Input"
        arg_strs = ", ".join(repr(a) for a in self.args)
        return f"{self.op}({arg_strs})"


class DSLInterpreter:
    """Recursive descent interpreter for DSL expression trees.

    Evaluates a DSLExpr against a concrete input grid, returning the
    result value (Grid, int, bool, etc.) or None on failure.

    Thread-safe: no mutable state between evaluations.
    """

    def evaluate(self, expr: DSLExpr, input_grid: Grid) -> Any:
        """Evaluate an expression tree against an input grid.

        Args:
            expr: The DSL expression to evaluate.
            input_grid: The concrete input grid.

        Returns:
            The result value, or None if evaluation fails.
        """
        try:
            return self._eval(expr, input_grid)
        except Exception:
            return None

    def _eval(self, expr: DSLExpr, grid: Grid) -> Any:
        """Internal recursive evaluation."""
        op = expr.op

        # --- Leaf nodes ---
        if op == "literal":
            return expr.value
        if op == "input_grid":
            return grid

        # --- Evaluate arguments first (skip lambdas) ---
        args = []
        for a in expr.args:
            if a.return_type == DSLType.LAMBDA:
                args.append(a)  # defer lambda evaluation
            else:
                val = self._eval(a, grid)
                if val is None and a.op != "literal":
                    return None
                args.append(val)

        # --- Dispatch to operation implementations ---

        # Cell-level operations
        if op == "grid_height":
            return len(args[0]) if args[0] else 0
        if op == "grid_width":
            return len(args[0][0]) if args[0] and args[0][0] else 0
        if op == "get_cell":
            g, r, c = args
            if 0 <= r < len(g) and 0 <= c < len(g[0]):
                return g[r][c]
            return 0
        if op == "set_cell":
            g, r, c, color = args
            if 0 <= r < len(g) and 0 <= c < len(g[0]):
                result = [row[:] for row in g]
                result[r][c] = color
                return result
            return g

        # Query operations
        if op == "count_color":
            g, color = args
            return sum(1 for row in g for cell in row if cell == color)
        if op == "most_common_color":
            g = args[0]
            if not g or not g[0]:
                return 0
            counts = Counter(cell for row in g for cell in row)
            return counts.most_common(1)[0][0]
        if op == "least_common_color":
            g = args[0]
            if not g or not g[0]:
                return 0
            counts = Counter(cell for row in g for cell in row)
            return counts.most_common()[-1][0]
        if op == "unique_colors":
            g = args[0]
            return set(cell for row in g for cell in row)

        # Color operations
        if op == "replace_color":
            g, old, new = args
            return [[new if cell == old else cell for cell in row]
                    for row in g]
        if op == "apply_color_map":
            g, cmap = args
            return [[cmap.get(cell, cell) for cell in row] for row in g]

        # Geometric operations
        if op == "transpose":
            g = args[0]
            if not g or not g[0]:
                return g
            return [[g[r][c] for r in range(len(g))]
                    for c in range(len(g[0]))]
        if op == "flip_h":
            return [row[::-1] for row in args[0]]
        if op == "flip_v":
            return list(reversed(args[0]))
        if op == "rotate_90":
            g = args[0]
            if not g or not g[0]:
                return g
            h, w = len(g), len(g[0])
            return [[g[h - 1 - j][i] for j in range(h)] for i in range(w)]

        # Crop
        if op == "crop":
            g, r0, c0, r1, c1 = args
            return [row[c0:c1 + 1] for row in g[r0:r1 + 1]]

        # Map objects
        if op == "map_objects":
            # args[0] is the evaluated grid, args[1] is the unevaluated lambda
            return self._eval_map_objects(args[0], args[1], grid)

        # Unknown operation
        return None

    def _eval_map_objects(self, g: Grid, lambda_expr: DSLExpr,
                         original_grid: Grid) -> Optional[Grid]:
        """Apply a lambda transform to each foreground object."""
        from .objects import find_foreground_shapes, place_subgrid

        shapes = find_foreground_shapes(g)
        if not shapes:
            return g

        # Determine background color
        counts = Counter(cell for row in g for cell in row)
        bg = counts.most_common(1)[0][0]

        # Build result canvas
        result = [row[:] for row in g]

        for shape in shapes:
            subgrid = shape["subgrid"]
            pos = shape["position"]

            # Apply the lambda transform to the object subgrid
            if lambda_expr.op == "replace_all_fg_with":
                # Special lambda: replace all non-zero pixels with a color
                new_color = self._eval(lambda_expr.args[0], original_grid)
                if new_color is None:
                    return None
                transformed = [
                    [new_color if cell != 0 else 0 for cell in row]
                    for row in subgrid
                ]
            else:
                # General case: evaluate the expression with subgrid as input
                transformed = self._eval(lambda_expr, subgrid)
                if transformed is None:
                    return None

            # Place transformed subgrid back
            for r in range(len(subgrid)):
                for c in range(len(subgrid[0])):
                    if subgrid[r][c] != 0:
                        gr = pos[0] + r
                        gc = pos[1] + c
                        if 0 <= gr < len(result) and 0 <= gc < len(result[0]):
                            if isinstance(transformed, list) and \
                               r < len(transformed) and c < len(transformed[0]):
                                result[gr][gc] = transformed[r][c]
                            elif isinstance(new_color, int):
                                result[gr][gc] = new_color

        return result


# ============================================================
# DSL Operation Registry (for synthesis enumeration)
# ============================================================

# Maps operation name → (arg_types, return_type)
# Used by the synthesis engine to enumerate valid expressions.
DSL_OPS: dict[str, tuple[list[DSLType], DSLType]] = {
    # Grid → Int
    "grid_height": ([DSLType.GRID], DSLType.INT),
    "grid_width": ([DSLType.GRID], DSLType.INT),
    # Grid → Color
    "most_common_color": ([DSLType.GRID], DSLType.COLOR),
    "least_common_color": ([DSLType.GRID], DSLType.COLOR),
    # Grid, Color → Int
    "count_color": ([DSLType.GRID, DSLType.COLOR], DSLType.INT),
    # Grid, Color, Color → Grid
    "replace_color": ([DSLType.GRID, DSLType.COLOR, DSLType.COLOR],
                      DSLType.GRID),
    # Grid, ColorMap → Grid
    "apply_color_map": ([DSLType.GRID, DSLType.COLOR_MAP], DSLType.GRID),
    # Grid → Grid (geometric)
    "transpose": ([DSLType.GRID], DSLType.GRID),
    "flip_h": ([DSLType.GRID], DSLType.GRID),
    "flip_v": ([DSLType.GRID], DSLType.GRID),
    "rotate_90": ([DSLType.GRID], DSLType.GRID),
}
