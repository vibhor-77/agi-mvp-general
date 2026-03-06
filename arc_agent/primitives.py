"""
DSL Primitives for ARC-AGI Grid Transformations

These are the atomic building blocks — the "base concepts" from Vibhor's
framework. Each primitive is a Concept that can be composed with others
to form higher-level abstractions.

Categories (from Vibhor's concept grammar):
  - Constants: Colors, shapes
  - Operators: Geometric transforms, fill operations
  - Relationships: Spatial and color predicates
"""
from __future__ import annotations
import copy
from typing import Optional
from .concepts import Concept, Grid, Toolkit


def _deep_copy_grid(grid: Grid) -> Grid:
    return [row[:] for row in grid]


def _grid_dims(grid: Grid) -> tuple[int, int]:
    """Return (height, width) of grid."""
    if not grid:
        return (0, 0)
    return (len(grid), len(grid[0]))


# ============================================================
# OPERATORS: Geometric Transformations
# ============================================================

def rotate_90_cw(grid: Grid) -> Grid:
    """Rotate grid 90 degrees clockwise."""
    h, w = _grid_dims(grid)
    if h == 0:
        return grid
    return [[grid[h - 1 - j][i] for j in range(h)] for i in range(w)]


def rotate_90_ccw(grid: Grid) -> Grid:
    """Rotate grid 90 degrees counter-clockwise."""
    h, w = _grid_dims(grid)
    if h == 0:
        return grid
    return [[grid[j][w - 1 - i] for j in range(h)] for i in range(w)]


def rotate_180(grid: Grid) -> Grid:
    """Rotate grid 180 degrees."""
    return [row[::-1] for row in reversed(grid)]


def mirror_horizontal(grid: Grid) -> Grid:
    """Mirror grid horizontally (left-right flip)."""
    return [row[::-1] for row in grid]


def mirror_vertical(grid: Grid) -> Grid:
    """Mirror grid vertically (top-bottom flip)."""
    return list(reversed([row[:] for row in grid]))


def transpose(grid: Grid) -> Grid:
    """Transpose grid (swap rows and columns)."""
    h, w = _grid_dims(grid)
    if h == 0:
        return grid
    return [[grid[j][i] for j in range(h)] for i in range(w)]


def identity(grid: Grid) -> Grid:
    """Return grid unchanged (useful as a no-op in compositions)."""
    return _deep_copy_grid(grid)


# ============================================================
# OPERATORS: Color Transformations
# ============================================================

def _make_color_swap(from_color: int, to_color: int):
    """Factory: create a function that swaps one color for another."""
    def swap(grid: Grid) -> Grid:
        result = _deep_copy_grid(grid)
        for r in range(len(result)):
            for c in range(len(result[0])):
                if result[r][c] == from_color:
                    result[r][c] = to_color
        return result
    return swap


def _make_recolor_nonzero(to_color: int):
    """Factory: recolor all non-zero cells to a specific color."""
    def recolor(grid: Grid) -> Grid:
        result = _deep_copy_grid(grid)
        for r in range(len(result)):
            for c in range(len(result[0])):
                if result[r][c] != 0:
                    result[r][c] = to_color
        return result
    return recolor


def invert_colors(grid: Grid) -> Grid:
    """Swap 0 and non-0: background becomes foreground and vice versa."""
    result = _deep_copy_grid(grid)
    for r in range(len(result)):
        for c in range(len(result[0])):
            if result[r][c] == 0:
                result[r][c] = 1
            else:
                result[r][c] = 0
    return result


def most_common_color_fill(grid: Grid) -> Grid:
    """Fill the entire grid with the most common non-zero color."""
    h, w = _grid_dims(grid)
    counts = {}
    for r in range(h):
        for c in range(w):
            v = grid[r][c]
            if v != 0:
                counts[v] = counts.get(v, 0) + 1
    if not counts:
        return _deep_copy_grid(grid)
    dominant = max(counts, key=counts.get)
    return [[dominant] * w for _ in range(h)]


# ============================================================
# OPERATORS: Spatial / Structural Transformations
# ============================================================

def crop_to_nonzero(grid: Grid) -> Grid:
    """Crop grid to the bounding box of non-zero cells."""
    h, w = _grid_dims(grid)
    if h == 0:
        return grid
    min_r, max_r, min_c, max_c = h, 0, w, 0
    found = False
    for r in range(h):
        for c in range(w):
            if grid[r][c] != 0:
                found = True
                min_r = min(min_r, r)
                max_r = max(max_r, r)
                min_c = min(min_c, c)
                max_c = max(max_c, c)
    if not found:
        return [[0]]
    return [grid[r][min_c:max_c + 1] for r in range(min_r, max_r + 1)]


def tile_2x2(grid: Grid) -> Grid:
    """Tile the grid in a 2x2 pattern."""
    h, w = _grid_dims(grid)
    result = []
    for _ in range(2):
        for r in range(h):
            result.append(grid[r][:] + grid[r][:])
    return result


def tile_3x3(grid: Grid) -> Grid:
    """Tile the grid in a 3x3 pattern."""
    h, w = _grid_dims(grid)
    result = []
    for _ in range(3):
        for r in range(h):
            result.append(grid[r][:] * 3)
    return result


def scale_2x(grid: Grid) -> Grid:
    """Scale each cell to a 2x2 block."""
    result = []
    for row in grid:
        new_row = []
        for cell in row:
            new_row.extend([cell, cell])
        result.append(new_row[:])
        result.append(new_row[:])
    return result


def scale_3x(grid: Grid) -> Grid:
    """Scale each cell to a 3x3 block."""
    result = []
    for row in grid:
        new_row = []
        for cell in row:
            new_row.extend([cell, cell, cell])
        for _ in range(3):
            result.append(new_row[:])
    return result


def gravity_down(grid: Grid) -> Grid:
    """Move all non-zero cells to the bottom of their column."""
    h, w = _grid_dims(grid)
    result = [[0] * w for _ in range(h)]
    for c in range(w):
        non_zero = [grid[r][c] for r in range(h) if grid[r][c] != 0]
        for i, v in enumerate(reversed(non_zero)):
            result[h - 1 - i][c] = v
    return result


def gravity_up(grid: Grid) -> Grid:
    """Move all non-zero cells to the top of their column."""
    h, w = _grid_dims(grid)
    result = [[0] * w for _ in range(h)]
    for c in range(w):
        non_zero = [grid[r][c] for r in range(h) if grid[r][c] != 0]
        for i, v in enumerate(non_zero):
            result[i][c] = v
    return result


def gravity_left(grid: Grid) -> Grid:
    """Move all non-zero cells to the left of their row."""
    h, w = _grid_dims(grid)
    result = [[0] * w for _ in range(h)]
    for r in range(h):
        non_zero = [grid[r][c] for c in range(w) if grid[r][c] != 0]
        for i, v in enumerate(non_zero):
            result[r][i] = v
    return result


def gravity_right(grid: Grid) -> Grid:
    """Move all non-zero cells to the right of their row."""
    h, w = _grid_dims(grid)
    result = [[0] * w for _ in range(h)]
    for r in range(h):
        non_zero = [grid[r][c] for c in range(w) if grid[r][c] != 0]
        for i, v in enumerate(reversed(non_zero)):
            result[r][w - 1 - i] = v
    return result


def flood_fill_background(grid: Grid) -> Grid:
    """Fill all background (0) connected to top-left with most common non-zero color."""
    h, w = _grid_dims(grid)
    counts = {}
    for r in range(h):
        for c in range(w):
            v = grid[r][c]
            if v != 0:
                counts[v] = counts.get(v, 0) + 1
    if not counts:
        return _deep_copy_grid(grid)
    fill_color = max(counts, key=counts.get)

    result = _deep_copy_grid(grid)
    visited = set()
    stack = [(0, 0)]
    while stack:
        r, c = stack.pop()
        if (r, c) in visited or r < 0 or r >= h or c < 0 or c >= w:
            continue
        if result[r][c] != 0:
            continue
        visited.add((r, c))
        result[r][c] = fill_color
        stack.extend([(r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)])
    return result


def extract_unique_colors(grid: Grid) -> Grid:
    """Create a 1-row grid of unique non-zero colors found in input."""
    colors = set()
    for row in grid:
        for cell in row:
            if cell != 0:
                colors.add(cell)
    sorted_colors = sorted(colors)
    if not sorted_colors:
        return [[0]]
    return [sorted_colors]


def count_nonzero_per_row(grid: Grid) -> Grid:
    """Create a column grid where each cell = count of non-zero in that row."""
    return [[sum(1 for c in row if c != 0)] for row in grid]


def outline(grid: Grid) -> Grid:
    """Keep only the border cells of non-zero regions."""
    h, w = _grid_dims(grid)
    result = [[0] * w for _ in range(h)]
    for r in range(h):
        for c in range(w):
            if grid[r][c] != 0:
                # Check if any neighbor is 0 or out of bounds
                is_border = False
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = r + dr, c + dc
                    if nr < 0 or nr >= h or nc < 0 or nc >= w or grid[nr][nc] == 0:
                        is_border = True
                        break
                if is_border:
                    result[r][c] = grid[r][c]
    return result


def fill_enclosed(grid: Grid) -> Grid:
    """Fill enclosed (fully surrounded) zero-regions with the surrounding color."""
    h, w = _grid_dims(grid)
    result = _deep_copy_grid(grid)

    # Find all 0-cells reachable from border (these are NOT enclosed)
    border_connected = set()
    stack = []
    for r in range(h):
        for c in range(w):
            if (r == 0 or r == h - 1 or c == 0 or c == w - 1) and grid[r][c] == 0:
                stack.append((r, c))

    while stack:
        r, c = stack.pop()
        if (r, c) in border_connected or r < 0 or r >= h or c < 0 or c >= w:
            continue
        if grid[r][c] != 0:
            continue
        border_connected.add((r, c))
        stack.extend([(r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)])

    # Fill enclosed zeros with surrounding non-zero color
    for r in range(h):
        for c in range(w):
            if grid[r][c] == 0 and (r, c) not in border_connected:
                # Find surrounding color
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w and grid[nr][nc] != 0:
                        result[r][c] = grid[nr][nc]
                        break
    return result


# ============================================================
# RELATIONSHIPS: Predicates for conditional logic
# ============================================================

def is_symmetric_h(grid: Grid) -> bool:
    """Check if grid has horizontal symmetry."""
    return all(row == row[::-1] for row in grid)


def is_symmetric_v(grid: Grid) -> bool:
    """Check if grid has vertical symmetry."""
    h = len(grid)
    for i in range(h // 2):
        if grid[i] != grid[h - 1 - i]:
            return False
    return True


def is_square(grid: Grid) -> bool:
    """Check if grid is square."""
    h, w = _grid_dims(grid)
    return h == w and h > 0


def has_single_color(grid: Grid) -> bool:
    """Check if grid has only one non-zero color."""
    colors = set()
    for row in grid:
        for cell in row:
            if cell != 0:
                colors.add(cell)
    return len(colors) <= 1


def is_tall(grid: Grid) -> bool:
    """Check if grid is taller than it is wide (height > width)."""
    h, w = _grid_dims(grid)
    return h > w


def is_wide(grid: Grid) -> bool:
    """Check if grid is wider than it is tall (width > height)."""
    h, w = _grid_dims(grid)
    return w > h


def has_many_colors(grid: Grid) -> bool:
    """Check if grid has more than 3 non-zero colors."""
    colors = set()
    for row in grid:
        for cell in row:
            if cell != 0:
                colors.add(cell)
    return len(colors) > 3


def is_small(grid: Grid) -> bool:
    """Check if grid is small (total cells < 50)."""
    h, w = _grid_dims(grid)
    return h * w < 50


def is_large(grid: Grid) -> bool:
    """Check if grid is large (total cells > 200)."""
    h, w = _grid_dims(grid)
    return h * w > 200


def has_background_majority(grid: Grid) -> bool:
    """Check if more than 50% of cells are background (0)."""
    h, w = _grid_dims(grid)
    if h == 0 or w == 0:
        return True
    total_cells = h * w
    zero_count = sum(1 for row in grid for cell in row if cell == 0)
    return zero_count > total_cells / 2


# ============================================================
# GRID PARTITIONING AND PATTERN OPERATIONS
# ============================================================

def get_top_half(grid: Grid) -> Grid:
    """Extract the top half of the grid."""
    h = len(grid)
    return _deep_copy_grid(grid[:h // 2])


def get_bottom_half(grid: Grid) -> Grid:
    """Extract the bottom half of the grid."""
    h = len(grid)
    return _deep_copy_grid(grid[h // 2:])


def get_left_half(grid: Grid) -> Grid:
    """Extract the left half of the grid."""
    w = len(grid[0]) if grid else 0
    return [row[:w // 2] for row in grid]


def get_right_half(grid: Grid) -> Grid:
    """Extract the right half of the grid."""
    w = len(grid[0]) if grid else 0
    return [row[w // 2:] for row in grid]


def get_border(grid: Grid) -> Grid:
    """Extract only the border cells, zero out the interior."""
    if not grid or not grid[0]:
        return _deep_copy_grid(grid)
    h, w = len(grid), len(grid[0])
    result = [[0] * w for _ in range(h)]
    for r in range(h):
        for c in range(w):
            if r == 0 or r == h - 1 or c == 0 or c == w - 1:
                result[r][c] = grid[r][c]
    return result


def get_interior(grid: Grid) -> Grid:
    """Extract only the interior cells (remove 1-cell border)."""
    if not grid or len(grid) < 3 or len(grid[0]) < 3:
        return _deep_copy_grid(grid)
    return [row[1:-1] for row in grid[1:-1]]


def replace_color(grid: Grid, old: int, new: int) -> Grid:
    """Replace all occurrences of old color with new color."""
    return [[new if c == old else c for c in row] for row in grid]


def _make_replace_bg(new_color: int):
    """Factory: replace background (0) with a specific color."""
    def _replace(grid: Grid) -> Grid:
        return replace_color(grid, 0, new_color)
    return _replace


def _make_replace_with_bg(old_color: int):
    """Factory: replace a specific color with background (0)."""
    def _replace(grid: Grid) -> Grid:
        return replace_color(grid, old_color, 0)
    return _replace


def most_common_color(grid: Grid) -> int:
    """Find the most common non-zero color in the grid."""
    counts: dict[int, int] = {}
    for row in grid:
        for c in row:
            if c != 0:
                counts[c] = counts.get(c, 0) + 1
    if not counts:
        return 0
    return max(counts, key=lambda k: counts[k])


def least_common_color(grid: Grid) -> int:
    """Find the least common non-zero color in the grid."""
    counts: dict[int, int] = {}
    for row in grid:
        for c in row:
            if c != 0:
                counts[c] = counts.get(c, 0) + 1
    if not counts:
        return 0
    return min(counts, key=lambda k: counts[k])


def recolor_to_most_common(grid: Grid) -> Grid:
    """Recolor all non-zero cells to the most common color."""
    mc = most_common_color(grid)
    if mc == 0:
        return _deep_copy_grid(grid)
    return [[mc if c != 0 else 0 for c in row] for row in grid]


def deduplicate_rows(grid: Grid) -> Grid:
    """Remove duplicate consecutive rows."""
    if not grid:
        return []
    result = [grid[0][:]]
    for row in grid[1:]:
        if row != result[-1]:
            result.append(row[:])
    return result


def deduplicate_cols(grid: Grid) -> Grid:
    """Remove duplicate consecutive columns."""
    if not grid or not grid[0]:
        return _deep_copy_grid(grid)
    t = transpose(grid)
    deduped = deduplicate_rows(t)
    return transpose(deduped)


def upscale_to_max(grid: Grid) -> Grid:
    """Upscale grid by 2x or 3x based on size (targets ~10x10)."""
    h, w = len(grid), len(grid[0]) if grid else 0
    if h <= 3 and w <= 3:
        return scale_3x(grid)
    elif h <= 5 and w <= 5:
        return scale_2x(grid)
    return _deep_copy_grid(grid)


def sort_rows_by_color_count(grid: Grid) -> Grid:
    """Sort rows by number of non-zero cells (ascending)."""
    rows = [row[:] for row in grid]
    rows.sort(key=lambda r: sum(1 for c in r if c != 0))
    return rows


def reverse_rows(grid: Grid) -> Grid:
    """Reverse the order of rows (flip vertically)."""
    return _deep_copy_grid(grid[::-1])


def reverse_cols(grid: Grid) -> Grid:
    """Reverse each row (flip horizontally). Same as mirror_h."""
    return [row[::-1] for row in grid]


# ============================================================
# TOOLKIT INITIALIZATION
# ============================================================

def build_initial_toolkit(include_objects: bool = True) -> Toolkit:
    """Build the initial toolkit with all primitive concepts.

    This is the 'seed' — the equivalent of the basic biological
    machinery that evolution starts with. From Vibhor's framework,
    these are the base concepts that will be composed into
    higher-level abstractions through the learning loop.

    Args:
        include_objects: If True, include object-level primitives
            (connected components, extraction, etc). Default True.
    """
    toolkit = Toolkit()

    # Geometric operators
    operators = [
        ("rotate_90_cw", rotate_90_cw),
        ("rotate_90_ccw", rotate_90_ccw),
        ("rotate_180", rotate_180),
        ("mirror_h", mirror_horizontal),
        ("mirror_v", mirror_vertical),
        ("transpose", transpose),
        ("identity", identity),
        ("crop_nonzero", crop_to_nonzero),
        ("tile_2x2", tile_2x2),
        ("tile_3x3", tile_3x3),
        ("scale_2x", scale_2x),
        ("scale_3x", scale_3x),
        ("gravity_down", gravity_down),
        ("gravity_up", gravity_up),
        ("gravity_left", gravity_left),
        ("gravity_right", gravity_right),
        ("flood_fill_bg", flood_fill_background),
        ("outline", outline),
        ("fill_enclosed", fill_enclosed),
        ("invert_colors", invert_colors),
        ("extract_colors", extract_unique_colors),
        ("count_per_row", count_nonzero_per_row),
    ]

    for name, impl in operators:
        toolkit.add_concept(Concept(
            kind="operator",
            name=name,
            implementation=impl,
        ))

    # Color swap operators (a selection of common swaps)
    for from_c in range(1, 5):
        for to_c in range(1, 5):
            if from_c != to_c:
                name = f"swap_{from_c}_to_{to_c}"
                toolkit.add_concept(Concept(
                    kind="operator",
                    name=name,
                    implementation=_make_color_swap(from_c, to_c),
                ))

    # Recolor operators
    for color in range(1, 10):
        name = f"recolor_to_{color}"
        toolkit.add_concept(Concept(
            kind="operator",
            name=name,
            implementation=_make_recolor_nonzero(color),
        ))

    # Grid partitioning and pattern operators
    partitioning_ops = [
        ("get_top_half", get_top_half),
        ("get_bottom_half", get_bottom_half),
        ("get_left_half", get_left_half),
        ("get_right_half", get_right_half),
        ("get_border", get_border),
        ("get_interior", get_interior),
        ("recolor_to_most_common", recolor_to_most_common),
        ("deduplicate_rows", deduplicate_rows),
        ("deduplicate_cols", deduplicate_cols),
        ("upscale_to_max", upscale_to_max),
        ("sort_rows_by_color_count", sort_rows_by_color_count),
        ("reverse_rows", reverse_rows),
        ("reverse_cols", reverse_cols),
    ]

    for name, impl in partitioning_ops:
        toolkit.add_concept(Concept(
            kind="operator",
            name=name,
            implementation=impl,
        ))

    # Background replacement operators (fill bg with color)
    for color in range(1, 10):
        name = f"fill_bg_{color}"
        toolkit.add_concept(Concept(
            kind="operator",
            name=name,
            implementation=_make_replace_bg(color),
        ))

    # Erase-color operators (replace color with background)
    for color in range(1, 10):
        name = f"erase_{color}"
        toolkit.add_concept(Concept(
            kind="operator",
            name=name,
            implementation=_make_replace_with_bg(color),
        ))

    # Add predicates (for conditional logic)
    predicates = [
        ("is_symmetric_h", is_symmetric_h),
        ("is_symmetric_v", is_symmetric_v),
        ("is_square", is_square),
        ("has_single_color", has_single_color),
        ("is_tall", is_tall),
        ("is_wide", is_wide),
        ("has_many_colors", has_many_colors),
        ("is_small", is_small),
        ("is_large", is_large),
        ("has_background_majority", has_background_majority),
    ]

    for name, predicate_fn in predicates:
        toolkit.add_concept(Concept(
            kind="predicate",
            name=name,
            implementation=predicate_fn,
        ))

    # Add object-level primitives
    if include_objects:
        from .objects import add_object_concepts
        add_object_concepts(toolkit)

    return toolkit
