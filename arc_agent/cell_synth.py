"""Cell Synthesis: Enumeration-based DSL for per-cell transformations.

Discovers task-specific cell-level transformation rules by enumerating small
DSL programs and scoring them against training examples. This enables discovering
context-dependent color mappings and conditional rules automatically.

Architecture:
  CellExpr: Abstract base for cell-level expressions (AST nodes)
    ├─ Const(color): Always output this color
    ├─ Self: Keep current cell value
    ├─ NeighborMajority: Majority color of 4-neighbors
    ├─ NeighborAt(dir): Color of neighbor in direction
    ├─ MapColor(from, to): if cell==from, output to, else self
    ├─ IfColor(c, then, else): if cell==c, then expr, else expr
    └─ IfNeighborHas(c, then, else): if any neighbor==c, then/else

Enumeration generates all CellExpr up to depth N, pruning for sanity.
Scoring: for each training example, apply CellExpr to every cell, compare to output.
Integration: wrap best CellExpr as a Concept reusable in solver.

Key design:
  - CellExpr nodes are dataclasses for clarity and introspection
  - Evaluation is recursive: evaluate_cell_expr(expr, grid, row, col)
  - Enumeration uses BFS with size/depth limits for efficiency
  - Scoring averages across all training examples
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Callable
from abc import ABC, abstractmethod
from .concepts import Concept, Grid


# ============================================================
# DSL: Cell Expression Nodes
# ============================================================

@dataclass(frozen=True)
class CellExpr(ABC):
    """Base class for cell-level expressions.

    Frozen dataclass ensures immutability and hashability (for dedup in enumeration).
    """

    @abstractmethod
    def __repr__(self) -> str:
        """String representation for debugging and dedup."""
        pass


@dataclass(frozen=True)
class Const(CellExpr):
    """Always output this color."""
    color: int

    def __repr__(self) -> str:
        return f"Const({self.color})"


@dataclass(frozen=True)
class Self(CellExpr):
    """Keep the current cell's value unchanged."""

    def __repr__(self) -> str:
        return "Self"


@dataclass(frozen=True)
class NeighborMajority(CellExpr):
    """Output the most common color among 4-neighbors (up, down, left, right).

    If tie, pick the minimum color. If no neighbors exist, return 0.
    """

    def __repr__(self) -> str:
        return "NeighborMajority"


@dataclass(frozen=True)
class NeighborAt(CellExpr):
    """Output the value of the neighbor in direction (up/down/left/right).

    If neighbor doesn't exist, return 0.
    """
    direction: str  # 'up', 'down', 'left', 'right'

    def __repr__(self) -> str:
        return f"NeighborAt({self.direction})"


@dataclass(frozen=True)
class MapColor(CellExpr):
    """If cell == from_color, output to_color; else output self."""
    from_color: int
    to_color: int

    def __repr__(self) -> str:
        return f"MapColor({self.from_color}→{self.to_color})"


@dataclass(frozen=True)
class IfColor(CellExpr):
    """If cell == color, then branch; else else branch."""
    color: int
    then_expr: CellExpr
    else_expr: CellExpr

    def __repr__(self) -> str:
        return f"IfColor({self.color},{self.then_expr},{self.else_expr})"


@dataclass(frozen=True)
class IfNeighborHas(CellExpr):
    """If any 4-neighbor == color, then branch; else else branch."""
    color: int
    then_expr: CellExpr
    else_expr: CellExpr

    def __repr__(self) -> str:
        return f"IfNeighborHas({self.color},{self.then_expr},{self.else_expr})"


# ============================================================
# Evaluation: execute a CellExpr at a specific cell
# ============================================================

def evaluate_cell_expr(expr: CellExpr, grid: Grid, row: int, col: int) -> int:
    """Evaluate a cell expression at grid[row][col].

    Args:
        expr: The CellExpr to evaluate
        grid: The grid (list of lists of ints)
        row, col: The cell position

    Returns:
        The computed value (0-9 for ARC colors)
    """
    if isinstance(expr, Const):
        return expr.color

    elif isinstance(expr, Self):
        return grid[row][col]

    elif isinstance(expr, NeighborMajority):
        # Count colors of 4-neighbors
        neighbors = []
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = row + dr, col + dc
            if 0 <= nr < len(grid) and 0 <= nc < len(grid[0]):
                neighbors.append(grid[nr][nc])

        if not neighbors:
            return 0

        # Find majority (most common color)
        from collections import Counter
        counts = Counter(neighbors)
        return min(counts, key=lambda c: (-counts[c], c))  # Most common, then min

    elif isinstance(expr, NeighborAt):
        # Get neighbor in specified direction
        dirs = {
            'up': (-1, 0),
            'down': (1, 0),
            'left': (0, -1),
            'right': (0, 1),
        }
        if expr.direction not in dirs:
            return 0
        dr, dc = dirs[expr.direction]
        nr, nc = row + dr, col + dc
        if 0 <= nr < len(grid) and 0 <= nc < len(grid[0]):
            return grid[nr][nc]
        return 0

    elif isinstance(expr, MapColor):
        current = grid[row][col]
        if current == expr.from_color:
            return expr.to_color
        return current

    elif isinstance(expr, IfColor):
        if grid[row][col] == expr.color:
            return evaluate_cell_expr(expr.then_expr, grid, row, col)
        else:
            return evaluate_cell_expr(expr.else_expr, grid, row, col)

    elif isinstance(expr, IfNeighborHas):
        # Check if any neighbor has the target color
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = row + dr, col + dc
            if 0 <= nr < len(grid) and 0 <= nc < len(grid[0]):
                if grid[nr][nc] == expr.color:
                    return evaluate_cell_expr(expr.then_expr, grid, row, col)
        return evaluate_cell_expr(expr.else_expr, grid, row, col)

    else:
        return 0


# ============================================================
# Scoring: evaluate a cell expr against training examples
# ============================================================

def score_cell_expr(expr: CellExpr, input_grid: Grid, output_grid: Grid) -> float:
    """Score a cell expression against an input/output pair.

    Applies expr to every cell of input_grid and compares to output_grid.
    Returns fraction of cells that match (0.0 to 1.0).

    Args:
        expr: The CellExpr to score
        input_grid: The input grid
        output_grid: The expected output grid

    Returns:
        Fraction of cells matching (0.0 to 1.0)
    """
    if len(input_grid) != len(output_grid):
        return 0.0

    height = len(input_grid)
    width = len(input_grid[0]) if height > 0 else 0

    if any(len(row) != width for row in input_grid):
        return 0.0
    if any(len(row) != width for row in output_grid):
        return 0.0

    matches = 0
    total = height * width

    for r in range(height):
        for c in range(width):
            predicted = evaluate_cell_expr(expr, input_grid, r, c)
            expected = output_grid[r][c]
            if predicted == expected:
                matches += 1

    return matches / total if total > 0 else 0.0


# ============================================================
# Enumeration: generate all cell programs up to depth N
# ============================================================

def enumerate_cell_exprs(
    colors: set[int],
    max_depth: int = 2,
    max_count: int = 10000,
) -> list[CellExpr]:
    """Enumerate all cell expressions up to a given depth.

    Uses BFS to generate programs, pruning by size and depth.
    Ensures no duplicates via set tracking.

    Args:
        colors: Set of colors that appear in the task
        max_depth: Maximum nesting depth (0=constants only, 1=primitives, etc.)
        max_count: Safety limit on enumeration count

    Returns:
        List of unique CellExpr sorted by (depth, repr) for consistency
    """
    seen = set()
    programs_by_depth = [[] for _ in range(max_depth + 1)]

    # Depth 0: Constants and basic primitives (no arguments)
    depth_0 = [Const(c) for c in sorted(colors)]
    depth_0.append(Self())
    depth_0.append(NeighborMajority())

    for expr in depth_0:
        key = str(expr)
        if key not in seen:
            seen.add(key)
            programs_by_depth[0].append(expr)

    # Depth 0: Neighbors in all 4 directions
    for direction in ['up', 'down', 'left', 'right']:
        expr = NeighborAt(direction)
        key = str(expr)
        if key not in seen:
            seen.add(key)
            programs_by_depth[0].append(expr)

    # Depth 1+: Compositions of depth-0 and depth-(d-1) programs
    for depth in range(1, max_depth + 1):
        if len(seen) >= max_count:
            break

        candidates = []

        # MapColor(from_color, to_color)
        for from_c in sorted(colors):
            for to_c in sorted(colors):
                if from_c != to_c:
                    expr = MapColor(from_c, to_c)
                    key = str(expr)
                    if key not in seen and len(seen) < max_count:
                        seen.add(key)
                        candidates.append(expr)

        # IfColor(color, then, else) - combine with previous depths
        for color in sorted(colors):
            for then_expr in programs_by_depth[depth - 1]:
                for else_expr in programs_by_depth[depth - 1]:
                    expr = IfColor(color, then_expr, else_expr)
                    key = str(expr)
                    if key not in seen and len(seen) < max_count:
                        seen.add(key)
                        candidates.append(expr)
                        if len(seen) >= max_count:
                            break
                if len(seen) >= max_count:
                    break

        # IfNeighborHas(color, then, else)
        for color in sorted(colors):
            for then_expr in programs_by_depth[depth - 1]:
                for else_expr in programs_by_depth[depth - 1]:
                    expr = IfNeighborHas(color, then_expr, else_expr)
                    key = str(expr)
                    if key not in seen and len(seen) < max_count:
                        seen.add(key)
                        candidates.append(expr)
                        if len(seen) >= max_count:
                            break
                if len(seen) >= max_count:
                    break

        programs_by_depth[depth] = candidates

    # Flatten and return all programs
    all_programs = []
    for depth_programs in programs_by_depth:
        all_programs.extend(depth_programs)

    return all_programs


# ============================================================
# Top-level synthesis: enumerate, score, return best
# ============================================================

def synthesize_cell_program(
    task: dict,
    max_depth: int = 2,
    verbose: bool = False,
) -> tuple[Optional[CellExpr], float]:
    """Synthesize a cell program for the given task.

    Enumerates cell programs, scores each on training examples,
    returns the best (highest average score).

    Args:
        task: Dict with 'train' key containing [{'input': grid, 'output': grid}, ...]
        max_depth: Maximum nesting depth for enumeration
        verbose: Print progress

    Returns:
        (best_expr, avg_score) where best_expr is the CellExpr with highest score,
        or (None, 0.0) if no suitable program found.
    """
    train_examples = task.get('train', [])
    if not train_examples:
        return None, 0.0

    # Extract colors from training inputs/outputs
    colors = set()
    for example in train_examples:
        for row in example['input']:
            colors.update(row)
        for row in example['output']:
            colors.update(row)
    colors.discard(0)  # Remove background
    if not colors:
        colors = {0}

    # Enumerate all candidate programs
    programs = enumerate_cell_exprs(colors=colors, max_depth=max_depth)
    if verbose:
        print(f"  Enumerated {len(programs)} cell programs")

    best_expr = None
    best_score = 0.0

    for i, expr in enumerate(programs):
        # Score on first example to enable early termination
        score = score_cell_expr(
            expr,
            train_examples[0]['input'],
            train_examples[0]['output'],
        )

        # Early termination: if first example scores < 0.5, skip remaining examples
        if score < 0.5:
            continue

        # Score on all remaining examples
        total_score = score
        example_count = 1
        for example in train_examples[1:]:
            example_count += 1
            score = score_cell_expr(
                expr,
                example['input'],
                example['output'],
            )
            total_score += score

        avg_score = total_score / example_count

        if avg_score > best_score:
            best_score = avg_score
            best_expr = expr
            if verbose and avg_score >= 0.95:
                print(f"    Found strong program: {expr} (score={avg_score:.3f})")

    if verbose:
        if best_expr:
            print(f"  Best cell program: {best_expr} (score={best_score:.3f})")
        else:
            print(f"  No suitable cell program found")

    return best_expr, best_score


# ============================================================
# Concept Wrapping: make cell expr reusable in solver
# ============================================================

def wrap_cell_expr_as_concept(
    expr: CellExpr,
    name: str = "",
) -> Concept:
    """Wrap a synthesized cell expression as a reusable Concept.

    The resulting Concept applies the cell expression to every cell
    of the input grid and returns the transformed grid.

    Args:
        expr: The CellExpr to wrap
        name: Optional custom name (auto-generated if not provided)

    Returns:
        A Concept that applies the cell expr to all cells
    """
    if not name:
        name = f"cell_synth_{expr}"

    def cell_expr_implementation(grid: Grid) -> Optional[Grid]:
        """Apply cell expr to every cell of the grid."""
        try:
            if not grid or not grid[0]:
                return grid

            height = len(grid)
            width = len(grid[0])
            result = []

            for r in range(height):
                row = []
                for c in range(width):
                    value = evaluate_cell_expr(expr, grid, r, c)
                    row.append(value)
                result.append(row)

            return result
        except Exception:
            return None

    return Concept(
        kind="operator",
        name=name,
        implementation=cell_expr_implementation,
    )
