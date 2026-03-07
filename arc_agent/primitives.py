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
# SYMMETRY COMPLETION
# ============================================================

def complete_symmetry_h(grid: Grid) -> Grid:
    """Enforce horizontal (left-right) symmetry in each row.

    For each row, the half with more non-zero content is mirrored
    onto the other half. If both halves have equal content, the
    left half is mirrored onto the right.
    """
    h, w = _grid_dims(grid)
    if h == 0 or w < 2:
        return _deep_copy_grid(grid)
    result = _deep_copy_grid(grid)
    mid = w // 2
    for r in range(h):
        left_nz = sum(1 for c in range(mid) if grid[r][c] != 0)
        right_nz = sum(1 for c in range(mid, w) if grid[r][c] != 0)
        if left_nz >= right_nz:
            # Mirror left onto right
            for c in range(w):
                result[r][w - 1 - c] = result[r][c]
        else:
            # Mirror right onto left
            for c in range(w):
                result[r][c] = result[r][w - 1 - c]
    return result


def complete_symmetry_v(grid: Grid) -> Grid:
    """Enforce vertical (top-bottom) symmetry.

    The half (top or bottom) with more non-zero content is mirrored
    onto the other half.
    """
    h, w = _grid_dims(grid)
    if h < 2:
        return _deep_copy_grid(grid)
    result = _deep_copy_grid(grid)
    mid = h // 2
    top_nz = sum(1 for r in range(mid) for c in range(w) if grid[r][c] != 0)
    bot_nz = sum(1 for r in range(mid, h) for c in range(w) if grid[r][c] != 0)
    if top_nz >= bot_nz:
        for r in range(h):
            result[h - 1 - r] = result[r][:]
    else:
        for r in range(h):
            result[r] = result[h - 1 - r][:]
    return result


def complete_symmetry_4(grid: Grid) -> Grid:
    """Enforce 4-fold symmetry (horizontal + vertical)."""
    return complete_symmetry_v(complete_symmetry_h(grid))


# ============================================================
# MAJORITY VOTING / DENOISE
# ============================================================

def _majority_vote(grid: Grid, radius: int) -> Grid:
    """Replace each cell with the majority color in its neighborhood.

    Only replaces a cell if it disagrees with the strict majority of
    its (2*radius+1)×(2*radius+1) neighborhood. Ties preserve the
    original value.
    """
    h, w = _grid_dims(grid)
    if h == 0:
        return _deep_copy_grid(grid)
    result = _deep_copy_grid(grid)
    for r in range(h):
        for c in range(w):
            counts: dict[int, int] = {}
            for dr in range(-radius, radius + 1):
                for dc in range(-radius, radius + 1):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w:
                        v = grid[nr][nc]
                        counts[v] = counts.get(v, 0) + 1
            majority = max(counts, key=lambda k: counts[k])
            # Only replace if majority is strict (> half of neighborhood)
            total = sum(counts.values())
            if counts[majority] > total // 2:
                result[r][c] = majority
    return result


def denoise_3x3(grid: Grid) -> Grid:
    """Replace each cell with 3×3 neighborhood majority."""
    return _majority_vote(grid, 1)


def denoise_5x5(grid: Grid) -> Grid:
    """Replace each cell with 5×5 neighborhood majority."""
    return _majority_vote(grid, 2)


# ============================================================
# GRID OVERLAY / BOOLEAN OPERATIONS
# ============================================================

def _combine_halves_v(grid: Grid, op: str) -> Grid:
    """Combine top and bottom halves with a boolean operation.

    op: 'xor', 'or', 'and'. Non-zero values are treated as True.
    The result grid has the same width but half the height.
    """
    h, w = _grid_dims(grid)
    if h < 2:
        return _deep_copy_grid(grid)
    mid = h // 2
    result = [[0] * w for _ in range(mid)]
    for r in range(mid):
        for c in range(w):
            a = grid[r][c]
            b = grid[mid + r][c]
            if op == "xor":
                # XOR: keep whichever is non-zero, but zero if both non-zero
                result[r][c] = (a if b == 0 else (b if a == 0 else 0))
            elif op == "or":
                result[r][c] = a if a != 0 else b
            elif op == "and":
                result[r][c] = a if (a != 0 and b != 0) else 0
    return result


def _combine_halves_h(grid: Grid, op: str) -> Grid:
    """Combine left and right halves with a boolean operation.

    Result grid has the same height but half the width.
    """
    h, w = _grid_dims(grid)
    if w < 2:
        return _deep_copy_grid(grid)
    mid = w // 2
    result = [[0] * mid for _ in range(h)]
    for r in range(h):
        for c in range(mid):
            a = grid[r][c]
            b = grid[r][mid + c]
            if op == "xor":
                result[r][c] = (a if b == 0 else (b if a == 0 else 0))
            elif op == "or":
                result[r][c] = a if a != 0 else b
            elif op == "and":
                result[r][c] = a if (a != 0 and b != 0) else 0
    return result


def xor_halves_v(grid: Grid) -> Grid:
    """XOR top and bottom halves (half-height output)."""
    return _combine_halves_v(grid, "xor")

def or_halves_v(grid: Grid) -> Grid:
    """OR top and bottom halves (half-height output)."""
    return _combine_halves_v(grid, "or")

def and_halves_v(grid: Grid) -> Grid:
    """AND top and bottom halves (half-height output)."""
    return _combine_halves_v(grid, "and")

def xor_halves_h(grid: Grid) -> Grid:
    """XOR left and right halves (half-width output)."""
    return _combine_halves_h(grid, "xor")

def or_halves_h(grid: Grid) -> Grid:
    """OR left and right halves (half-width output)."""
    return _combine_halves_h(grid, "or")

def and_halves_h(grid: Grid) -> Grid:
    """AND left and right halves (half-width output)."""
    return _combine_halves_h(grid, "and")


# ============================================================
# COLOR MAPPING BY FREQUENCY
# ============================================================

def swap_most_least(grid: Grid) -> Grid:
    """Swap the most-common and least-common non-zero colors."""
    mc = most_common_color(grid)
    lc = least_common_color(grid)
    if mc == 0 or lc == 0 or mc == lc:
        return _deep_copy_grid(grid)
    return [
        [lc if v == mc else (mc if v == lc else v) for v in row]
        for row in grid
    ]


def recolor_least_common(grid: Grid) -> Grid:
    """Replace least-common non-zero color with most-common."""
    mc = most_common_color(grid)
    lc = least_common_color(grid)
    if mc == 0 or lc == 0 or mc == lc:
        return _deep_copy_grid(grid)
    return [
        [mc if v == lc else v for v in row]
        for row in grid
    ]


# ============================================================
# PATTERN STACKING / REPETITION
# ============================================================

def repeat_rows_2x(grid: Grid) -> Grid:
    """Repeat the row sequence vertically (double height)."""
    return _deep_copy_grid(grid) + _deep_copy_grid(grid)


def repeat_cols_2x(grid: Grid) -> Grid:
    """Repeat the column sequence horizontally (double width)."""
    return [row[:] + row[:] for row in grid]


def stack_with_mirror_v(grid: Grid) -> Grid:
    """Stack grid with its vertical mirror below (double height).

    Original on top, vertically-mirrored copy below.
    """
    top = _deep_copy_grid(grid)
    bottom = list(reversed(_deep_copy_grid(grid)))
    return top + bottom


def stack_with_mirror_h(grid: Grid) -> Grid:
    """Stack grid with its horizontal mirror to the right (double width).

    Original on left, horizontally-mirrored copy on right.
    """
    return [row[:] + row[::-1] for row in grid]


# ============================================================
# DIAGONAL OPERATIONS
# ============================================================

def mirror_diagonal_main(grid: Grid) -> Grid:
    """Mirror grid along the main diagonal (top-left to bottom-right).

    Same as transpose for square grids. For non-square grids, pads with 0.
    """
    h, w = _grid_dims(grid)
    if h == 0:
        return _deep_copy_grid(grid)
    n = max(h, w)
    result = [[0] * n for _ in range(n)]
    for r in range(h):
        for c in range(w):
            result[c][r] = grid[r][c]
    # Crop to content
    return [row[:w] for row in result[:h]] if h == w else result


def mirror_diagonal_anti(grid: Grid) -> Grid:
    """Mirror grid along the anti-diagonal (top-right to bottom-left)."""
    h, w = _grid_dims(grid)
    if h == 0:
        return _deep_copy_grid(grid)
    n = max(h, w)
    result = [[0] * n for _ in range(n)]
    for r in range(h):
        for c in range(w):
            result[n - 1 - c][n - 1 - r] = grid[r][c]
    return [row[:h] for row in result[:w]]


# ============================================================
# FILL OPERATIONS
# ============================================================

def fill_holes_per_color(grid: Grid) -> Grid:
    """Fill enclosed zero-regions for each color independently.

    For each non-zero color, find enclosed zero cells (surrounded by
    that color) and fill them. More precise than fill_enclosed which
    uses any adjacent color.
    """
    h, w = _grid_dims(grid)
    if h == 0:
        return _deep_copy_grid(grid)

    result = _deep_copy_grid(grid)

    # Get all colors
    colors = set()
    for row in grid:
        for c in row:
            if c != 0:
                colors.add(c)

    for color in colors:
        # Create binary mask: 1 where color, 0 elsewhere
        # Find zeros NOT reachable from border through zeros/other-colors
        # (i.e., zeros enclosed by this color)
        border_reachable = set()
        stack = []

        # Start from border cells that are 0 or not this color
        for r in range(h):
            for c in range(w):
                if (r == 0 or r == h - 1 or c == 0 or c == w - 1):
                    if grid[r][c] != color:
                        stack.append((r, c))

        while stack:
            r, c = stack.pop()
            if (r, c) in border_reachable:
                continue
            if r < 0 or r >= h or c < 0 or c >= w:
                continue
            if grid[r][c] == color:
                continue
            border_reachable.add((r, c))
            stack.extend([(r+1, c), (r-1, c), (r, c+1), (r, c-1)])

        # Fill enclosed zeros with this color
        for r in range(h):
            for c in range(w):
                if grid[r][c] == 0 and (r, c) not in border_reachable:
                    result[r][c] = color

    return result


def fill_rectangles(grid: Grid) -> Grid:
    """Complete partial rectangles by filling their bounding boxes.

    For each connected component, fill its entire bounding box with
    its color. Useful for tasks where partial shapes need completing.
    """
    from .objects import find_objects
    objects = find_objects(grid)
    if not objects:
        return _deep_copy_grid(grid)

    result = _deep_copy_grid(grid)
    for obj in objects:
        min_r, min_c, max_r, max_c = obj.bbox
        for r in range(min_r, max_r + 1):
            for c in range(min_c, max_c + 1):
                if result[r][c] == 0:
                    result[r][c] = obj.color
    return result


# ============================================================
# SORT OPERATIONS
# ============================================================

def sort_cols_by_color_count(grid: Grid) -> Grid:
    """Sort columns by number of non-zero cells (ascending)."""
    h, w = _grid_dims(grid)
    if h == 0 or w == 0:
        return _deep_copy_grid(grid)

    # Get column indices sorted by non-zero count
    col_counts = []
    for c in range(w):
        count = sum(1 for r in range(h) if grid[r][c] != 0)
        col_counts.append((count, c))
    col_counts.sort()

    # Build result with sorted columns
    result = [[0] * w for _ in range(h)]
    for new_c, (_, old_c) in enumerate(col_counts):
        for r in range(h):
            result[r][new_c] = grid[r][old_c]
    return result


# ============================================================
# GRID ARITHMETIC
# ============================================================

def grid_difference(grid: Grid) -> Grid:
    """Subtract bottom half from top half (keep non-zero in top but zero in bottom).

    Useful for finding what's unique to the top half vs bottom half.
    Result has half-height.
    """
    h, w = _grid_dims(grid)
    if h < 2:
        return _deep_copy_grid(grid)
    mid = h // 2
    result = [[0] * w for _ in range(mid)]
    for r in range(mid):
        for c in range(w):
            a = grid[r][c]
            b = grid[mid + r][c]
            # Keep a only if b is zero (a is unique to top)
            result[r][c] = a if (a != 0 and b == 0) else 0
    return result


def grid_difference_h(grid: Grid) -> Grid:
    """Subtract right half from left half (keep unique-to-left cells).

    Result has half-width.
    """
    h, w = _grid_dims(grid)
    if w < 2:
        return _deep_copy_grid(grid)
    mid = w // 2
    result = [[0] * mid for _ in range(h)]
    for r in range(h):
        for c in range(mid):
            a = grid[r][c]
            b = grid[r][mid + c]
            result[r][c] = a if (a != 0 and b == 0) else 0
    return result


# ============================================================
# PIXEL CONNECTIVITY / FLOOD FILL VARIANTS
# ============================================================

def spread_colors(grid: Grid) -> Grid:
    """Spread each non-zero cell to fill its 4-connected zero neighbors.

    One step of cellular automaton: each zero cell adjacent to a non-zero
    cell takes that color. If multiple colors compete, the smallest
    color value wins (deterministic).
    """
    h, w = _grid_dims(grid)
    if h == 0:
        return _deep_copy_grid(grid)
    result = _deep_copy_grid(grid)
    for r in range(h):
        for c in range(w):
            if grid[r][c] == 0:
                candidates = []
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w and grid[nr][nc] != 0:
                        candidates.append(grid[nr][nc])
                if candidates:
                    result[r][c] = min(candidates)
    return result


def erode(grid: Grid) -> Grid:
    """Erode non-zero regions: remove cells that border any zero cell.

    Opposite of spread_colors — shrinks objects by one pixel.
    """
    h, w = _grid_dims(grid)
    if h == 0:
        return _deep_copy_grid(grid)
    result = _deep_copy_grid(grid)
    for r in range(h):
        for c in range(w):
            if grid[r][c] != 0:
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = r + dr, c + dc
                    if nr < 0 or nr >= h or nc < 0 or nc >= w or grid[nr][nc] == 0:
                        result[r][c] = 0
                        break
    return result


# ============================================================
# COLOR MASK OPERATIONS
# ============================================================

def keep_only_largest_color(grid: Grid) -> Grid:
    """Keep only the most common non-zero color, zero everything else."""
    mc = most_common_color(grid)
    if mc == 0:
        return _deep_copy_grid(grid)
    return [[v if v == mc else 0 for v in row] for row in grid]


def keep_only_smallest_color(grid: Grid) -> Grid:
    """Keep only the least common non-zero color, zero everything else."""
    lc = least_common_color(grid)
    if lc == 0:
        return _deep_copy_grid(grid)
    return [[v if v == lc else 0 for v in row] for row in grid]


# ============================================================
# TILE / REPEATING PATTERN EXTRACTION
# ============================================================

def extract_repeating_tile(grid: Grid) -> Grid:
    """Find the smallest tile that, when repeated, reconstructs the grid.

    Tries all (h_tile, w_tile) divisors of (H, W) from smallest to largest.
    Returns the tile if tiling it reproduces the grid exactly, otherwise
    returns the grid unchanged.
    """
    h, w = _grid_dims(grid)
    if h == 0:
        return _deep_copy_grid(grid)

    for th in range(1, h + 1):
        if h % th != 0:
            continue
        for tw in range(1, w + 1):
            if w % tw != 0:
                continue
            if th == h and tw == w:
                continue  # Skip the trivial "tile is the whole grid"
            # Check if (th × tw) tile repeats to fill grid
            match = True
            for r in range(h):
                if not match:
                    break
                for c in range(w):
                    if grid[r][c] != grid[r % th][c % tw]:
                        match = False
                        break
            if match:
                return [grid[r][:tw] for r in range(th)]

    return _deep_copy_grid(grid)


def extract_top_left_block(grid: Grid) -> Grid:
    """Extract the top-left block delimited by a separator line.

    Looks for a full row or column of a single color that acts as
    a separator, then returns the block above/left of it.
    """
    h, w = _grid_dims(grid)
    if h < 2 or w < 2:
        return _deep_copy_grid(grid)

    # Find horizontal separator (row where all cells are the same non-zero color)
    for r in range(1, h):
        if len(set(grid[r])) == 1 and grid[r][0] != 0:
            return _deep_copy_grid(grid[:r])

    # Find vertical separator (column where all cells are the same non-zero color)
    for c in range(1, w):
        col_vals = set(grid[r][c] for r in range(h))
        if len(col_vals) == 1 and grid[0][c] != 0:
            return [row[:c] for row in grid]

    return _deep_copy_grid(grid)


def extract_bottom_right_block(grid: Grid) -> Grid:
    """Extract the bottom-right block delimited by a separator line."""
    h, w = _grid_dims(grid)
    if h < 2 or w < 2:
        return _deep_copy_grid(grid)

    # Find last horizontal separator
    for r in range(h - 1, 0, -1):
        if len(set(grid[r])) == 1 and grid[r][0] != 0:
            return _deep_copy_grid(grid[r + 1:]) if r + 1 < h else _deep_copy_grid(grid)

    # Find last vertical separator
    for c in range(w - 1, 0, -1):
        col_vals = set(grid[r][c] for r in range(h))
        if len(col_vals) == 1 and grid[0][c] != 0:
            return [row[c + 1:] for row in grid] if c + 1 < w else _deep_copy_grid(grid)

    return _deep_copy_grid(grid)


def split_by_separator_and_overlay(grid: Grid) -> Grid:
    """Split grid by separator line, overlay the sub-grids using OR logic.

    Finds a row/column of uniform non-zero color, splits the grid,
    and merges the pieces using OR (keep any non-zero cell).
    """
    h, w = _grid_dims(grid)
    if h < 3:
        return _deep_copy_grid(grid)

    # Try horizontal split
    for r in range(1, h - 1):
        if len(set(grid[r])) == 1 and grid[r][0] != 0:
            top = grid[:r]
            bottom = grid[r + 1:]
            th, bh = len(top), len(bottom)
            if th == bh and th > 0:
                result = [[0] * w for _ in range(th)]
                for i in range(th):
                    for j in range(w):
                        result[i][j] = top[i][j] if top[i][j] != 0 else bottom[i][j]
                return result

    # Try vertical split
    for c in range(1, w - 1):
        col_vals = set(grid[r][c] for r in range(h))
        if len(col_vals) == 1 and grid[0][c] != 0:
            left = [row[:c] for row in grid]
            right = [row[c + 1:] for row in grid]
            lw, rw = len(left[0]), len(right[0]) if right else 0
            if lw == rw and lw > 0:
                result = [[0] * lw for _ in range(h)]
                for i in range(h):
                    for j in range(lw):
                        result[i][j] = left[i][j] if left[i][j] != 0 else right[i][j]
                return result

    return _deep_copy_grid(grid)


def split_by_separator_and_xor(grid: Grid) -> Grid:
    """Split grid by separator line, XOR the sub-grids.

    Returns cells that are non-zero in exactly one half.
    """
    h, w = _grid_dims(grid)
    if h < 3:
        return _deep_copy_grid(grid)

    # Try horizontal split
    for r in range(1, h - 1):
        if len(set(grid[r])) == 1 and grid[r][0] != 0:
            top = grid[:r]
            bottom = grid[r + 1:]
            th, bh = len(top), len(bottom)
            if th == bh and th > 0:
                result = [[0] * w for _ in range(th)]
                for i in range(th):
                    for j in range(w):
                        a, b = top[i][j], bottom[i][j]
                        result[i][j] = a if b == 0 else (b if a == 0 else 0)
                return result

    # Try vertical split
    for c in range(1, w - 1):
        col_vals = set(grid[r][c] for r in range(h))
        if len(col_vals) == 1 and grid[0][c] != 0:
            left = [row[:c] for row in grid]
            right = [row[c + 1:] for row in grid]
            lw, rw = len(left[0]), len(right[0]) if right else 0
            if lw == rw and lw > 0:
                result = [[0] * lw for _ in range(h)]
                for i in range(h):
                    for j in range(lw):
                        a, b = left[i][j], right[i][j]
                        result[i][j] = a if b == 0 else (b if a == 0 else 0)
                return result

    return _deep_copy_grid(grid)


def compress_rows(grid: Grid) -> Grid:
    """Compress grid by removing duplicate rows (keep first occurrence)."""
    if not grid:
        return []
    seen = []
    result = []
    for row in grid:
        t = tuple(row)
        if t not in seen:
            seen.append(t)
            result.append(row[:])
    return result


def compress_cols(grid: Grid) -> Grid:
    """Compress grid by removing duplicate columns (keep first occurrence)."""
    if not grid or not grid[0]:
        return _deep_copy_grid(grid)
    t = transpose(grid)
    compressed = compress_rows(t)
    return transpose(compressed)


def max_color_per_cell(grid: Grid) -> Grid:
    """For each cell position, take the max color across NxN sub-blocks.

    Treats the grid as being composed of identical-sized blocks separated
    by single-color rows/columns. Overlays blocks with max-wins logic.
    Falls back to identity if no separator found.
    """
    h, w = _grid_dims(grid)
    if h < 3:
        return _deep_copy_grid(grid)

    # Find separator rows
    sep_rows = []
    for r in range(h):
        vals = set(grid[r])
        if len(vals) == 1 and grid[r][0] != 0:
            sep_rows.append(r)

    if not sep_rows:
        return _deep_copy_grid(grid)

    # Extract blocks between separators
    boundaries = [-1] + sep_rows + [h]
    blocks = []
    for i in range(len(boundaries) - 1):
        start = boundaries[i] + 1
        end = boundaries[i + 1]
        if start < end:
            blocks.append([grid[r][:] for r in range(start, end)])

    if len(blocks) < 2:
        return _deep_copy_grid(grid)

    # All blocks should be same dimensions
    bh = len(blocks[0])
    bw = len(blocks[0][0]) if blocks[0] else 0
    if any(len(b) != bh for b in blocks) or any(len(b[0]) != bw for b in blocks if b):
        return _deep_copy_grid(grid)

    # Overlay with max
    result = [[0] * bw for _ in range(bh)]
    for block in blocks:
        for r in range(bh):
            for c in range(bw):
                if block[r][c] != 0:
                    result[r][c] = max(result[r][c], block[r][c])
    return result


def min_color_per_cell(grid: Grid) -> Grid:
    """Like max_color_per_cell but keeps the minimum non-zero color.

    For overlapping blocks separated by colored lines: keep the non-zero
    color that is smallest at each position. Zero stays zero.
    """
    h, w = _grid_dims(grid)
    if h < 3:
        return _deep_copy_grid(grid)

    sep_rows = []
    for r in range(h):
        vals = set(grid[r])
        if len(vals) == 1 and grid[r][0] != 0:
            sep_rows.append(r)

    if not sep_rows:
        return _deep_copy_grid(grid)

    boundaries = [-1] + sep_rows + [h]
    blocks = []
    for i in range(len(boundaries) - 1):
        start = boundaries[i] + 1
        end = boundaries[i + 1]
        if start < end:
            blocks.append([grid[r][:] for r in range(start, end)])

    if len(blocks) < 2:
        return _deep_copy_grid(grid)

    bh = len(blocks[0])
    bw = len(blocks[0][0]) if blocks[0] else 0
    if any(len(b) != bh for b in blocks) or any(len(b[0]) != bw for b in blocks if b):
        return _deep_copy_grid(grid)

    result = [[0] * bw for _ in range(bh)]
    for block in blocks:
        for r in range(bh):
            for c in range(bw):
                v = block[r][c]
                if v != 0:
                    if result[r][c] == 0:
                        result[r][c] = v
                    else:
                        result[r][c] = min(result[r][c], v)
    return result


def extract_unique_block(grid: Grid) -> Grid:
    """Find a sub-block that differs from the others.

    Splits the grid into NxN blocks (using separator lines or equal
    division) and returns the one that is unique. If all blocks are
    identical or no separator found, returns the grid unchanged.
    """
    h, w = _grid_dims(grid)
    if h < 2:
        return _deep_copy_grid(grid)

    # Try finding separator rows first
    sep_rows = []
    for r in range(h):
        vals = set(grid[r])
        if len(vals) == 1 and grid[r][0] != 0:
            sep_rows.append(r)

    if sep_rows:
        boundaries = [-1] + sep_rows + [h]
        blocks = []
        for i in range(len(boundaries) - 1):
            start = boundaries[i] + 1
            end = boundaries[i + 1]
            if start < end:
                blocks.append(tuple(tuple(grid[r]) for r in range(start, end)))
    else:
        # Try equal division into 2, 3, or 4 blocks vertically
        blocks = []
        for n in [2, 3, 4]:
            if h % n == 0:
                bh = h // n
                blocks = [tuple(tuple(grid[r]) for r in range(i*bh, (i+1)*bh)) for i in range(n)]
                break
        if not blocks:
            return _deep_copy_grid(grid)

    if len(blocks) < 2:
        return _deep_copy_grid(grid)

    # Find the unique block (one that differs from majority)
    from collections import Counter
    block_counts = Counter(blocks)
    if len(block_counts) == 1:
        return _deep_copy_grid(grid)  # All identical

    # Return the least common block
    least_common = block_counts.most_common()[-1][0]
    return [list(row) for row in least_common]


def flatten_to_row(grid: Grid) -> Grid:
    """Flatten unique non-zero colors into a single row, sorted ascending."""
    colors = sorted(set(c for row in grid for c in row if c != 0))
    if not colors:
        return [[0]]
    return [colors]


def flatten_to_column(grid: Grid) -> Grid:
    """Flatten unique non-zero colors into a single column, sorted ascending."""
    colors = sorted(set(c for row in grid for c in row if c != 0))
    if not colors:
        return [[0]]
    return [[c] for c in colors]


def count_objects_as_grid(grid: Grid) -> Grid:
    """Return a 1×1 grid containing the number of connected objects."""
    from .objects import find_objects
    n = len(find_objects(grid))
    return [[n]]


def mode_color_per_row(grid: Grid) -> Grid:
    """Replace each row with its most common non-zero color (or 0)."""
    h, w = _grid_dims(grid)
    result = [[0] * w for _ in range(h)]
    for r in range(h):
        counts: dict[int, int] = {}
        for c in range(w):
            v = grid[r][c]
            if v != 0:
                counts[v] = counts.get(v, 0) + 1
        if counts:
            dominant = max(counts, key=lambda k: counts[k])
            result[r] = [dominant] * w
    return result


def mode_color_per_col(grid: Grid) -> Grid:
    """Replace each column with its most common non-zero color (or 0)."""
    h, w = _grid_dims(grid)
    result = [[0] * w for _ in range(h)]
    for c in range(w):
        counts: dict[int, int] = {}
        for r in range(h):
            v = grid[r][c]
            if v != 0:
                counts[v] = counts.get(v, 0) + 1
        if counts:
            dominant = max(counts, key=lambda k: counts[k])
            for r in range(h):
                result[r][c] = dominant
    return result


# ============================================================
# TILE COMPLETION
# ============================================================

def fill_tile_pattern(grid: Grid) -> Grid:
    """Infer a repeating tile from visible cells and fill zeros with it.

    For grids where some cells have been zeroed out but the underlying
    pattern is a repeating tile: find the tile period and restore all
    zero cells from the corresponding tile position.

    Strategy: try all tile sizes (th × tw) that divide (H, W). For
    each candidate, compute the tile by taking the most-common non-zero
    value at each position across all repetitions. Accept the smallest
    tile where at least 50% of positions have an unambiguous value.
    """
    h, w = _grid_dims(grid)
    if h == 0:
        return _deep_copy_grid(grid)

    from collections import Counter

    for th in range(1, h + 1):
        if h % th != 0:
            continue
        for tw in range(1, w + 1):
            if w % tw != 0:
                continue
            if th == h and tw == w:
                continue  # Skip trivial whole-grid tile

            # Collect votes for each tile position
            votes: list[list[Counter]] = [
                [Counter() for _ in range(tw)] for _ in range(th)
            ]
            for r in range(h):
                for c in range(w):
                    v = grid[r][c]
                    if v != 0:
                        votes[r % th][c % tw][v] += 1

            # Build tile: each position gets the plurality non-zero value
            tile: list[list[int]] = [[0] * tw for _ in range(th)]
            n_resolved = 0
            for tr in range(th):
                for tc in range(tw):
                    if votes[tr][tc]:
                        tile[tr][tc] = votes[tr][tc].most_common(1)[0][0]
                        n_resolved += 1

            # Require >= 50% of tile positions have data
            if n_resolved < th * tw * 0.5:
                continue

            # Consistency check: >= 90% of non-zero cells must agree with tile
            n_agree = 0
            n_nonzero = 0
            for r in range(h):
                for c in range(w):
                    v = grid[r][c]
                    if v != 0:
                        n_nonzero += 1
                        if tile[r % th][c % tw] == v:
                            n_agree += 1
            if n_nonzero > 0 and n_agree / n_nonzero < 0.90:
                continue

            # Fill grid using tile
            result = [[0] * w for _ in range(h)]
            for r in range(h):
                for c in range(w):
                    result[r][c] = tile[r % th][c % tw]
            return result

    return _deep_copy_grid(grid)


def fill_by_symmetry(grid: Grid) -> Grid:
    """Fill a rectangular masked region using the grid's symmetry.

    When a rectangular block of identical cells (the "mask color") covers
    part of the grid, and the unmasked region is rotationally or reflectionally
    symmetric, recover the masked cells from their symmetric counterparts.

    Handles: 180° rotational symmetry (most common in ARC) and H/V mirror.
    Falls back to identity if no symmetric counterpart available.
    """
    h, w = _grid_dims(grid)
    if h == 0:
        return _deep_copy_grid(grid)

    from collections import Counter

    # Find the most common color in rows/cols that have many repeated cells
    # (that's the mask color)
    flat = [v for row in grid for v in row]
    color_counts = Counter(flat)

    # Mask color candidates: colors that appear in rectangular patches
    # Heuristic: a color that appears in a contiguous rectangle
    def find_mask_rect(mask_c):
        """Find bounding box of mask_c cells. Return (r0,c0,r1,c1) or None."""
        cells = [(r, c) for r in range(h) for c in range(w)
                 if grid[r][c] == mask_c]
        if not cells:
            return None
        rs = [r for r, _ in cells]
        cs = [c for _, c in cells]
        r0, r1, c0, c1 = min(rs), max(rs), min(cs), max(cs)
        # Check that ALL cells in the bounding box are the mask color
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                if grid[r][c] != mask_c:
                    return None
        return r0, c0, r1, c1

    result = _deep_copy_grid(grid)

    # Try each non-zero color as a potential mask
    for mask_c in range(1, 10):
        if color_counts.get(mask_c, 0) < 2:
            continue
        rect = find_mask_rect(mask_c)
        if rect is None:
            continue
        r0, c0, r1, c1 = rect

        # Try 180° rotational symmetry
        filled = True
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                sym_r = h - 1 - r
                sym_c = w - 1 - c
                if grid[sym_r][sym_c] != mask_c:
                    result[r][c] = grid[sym_r][sym_c]
                else:
                    filled = False
        if filled:
            return result

        # Try horizontal mirror symmetry
        result = _deep_copy_grid(grid)
        filled = True
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                sym_r = h - 1 - r
                sym_c = c
                if grid[sym_r][sym_c] != mask_c:
                    result[r][c] = grid[sym_r][sym_c]
                else:
                    filled = False
        if filled:
            return result

        # Try vertical mirror symmetry
        result = _deep_copy_grid(grid)
        filled = True
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                sym_r = r
                sym_c = w - 1 - c
                if grid[sym_r][sym_c] != mask_c:
                    result[r][c] = grid[sym_r][sym_c]
                else:
                    filled = False
        if filled:
            return result

        # Restore to original before trying next color
        result = _deep_copy_grid(grid)

    return result


def recolor_by_nearest_border(grid: Grid) -> Grid:
    """Recolor isolated pixels using the nearest border stripe color.

    For grids with colored border stripes (rows or columns fully filled
    with a single non-zero color) and isolated "noise" pixels of a
    different color in the interior: replace each noise pixel with
    the color of the nearest border stripe.

    This handles patterns like:
      - Vertical border cols at left/right, horizontal border rows at top/bottom
      - Interior pixels get assigned the color of their closest border
    """
    h, w = _grid_dims(grid)
    if h == 0 or w < 2:
        return _deep_copy_grid(grid)

    from collections import Counter

    # Find border stripes: rows or columns uniformly filled with one non-zero color
    border_rows: dict[int, int] = {}  # row_idx -> color
    border_cols: dict[int, int] = {}  # col_idx -> color

    for r in range(h):
        vals = set(grid[r])
        if len(vals) == 1 and grid[r][0] != 0:
            border_rows[r] = grid[r][0]

    for c in range(w):
        vals = set(grid[r][c] for r in range(h))
        if len(vals) == 1 and grid[0][c] != 0:
            border_cols[c] = grid[0][c]

    if not border_rows and not border_cols:
        return _deep_copy_grid(grid)

    # Find the "noise" color: the least common non-zero non-border color
    border_colors = set(border_rows.values()) | set(border_cols.values())
    interior_colors = Counter()
    for r in range(h):
        for c in range(w):
            v = grid[r][c]
            if v != 0 and v not in border_colors:
                interior_colors[v] += 1

    if not interior_colors:
        return _deep_copy_grid(grid)

    noise_color = min(interior_colors, key=lambda k: interior_colors[k])

    result = _deep_copy_grid(grid)
    for r in range(h):
        for c in range(w):
            if grid[r][c] == noise_color:
                # Find nearest border (by Manhattan distance to stripe)
                best_dist = h + w
                best_color = noise_color

                for br, bc in border_rows.items():
                    d = abs(r - br)
                    if d < best_dist:
                        best_dist = d
                        best_color = bc

                for bc_idx, bc_color in border_cols.items():
                    d = abs(c - bc_idx)
                    if d < best_dist:
                        best_dist = d
                        best_color = bc_color

                result[r][c] = best_color

    return result


def extend_to_border_h(grid: Grid) -> Grid:
    """Extend each non-zero cell horizontally to fill its entire row.

    Every non-zero value propagates left and right within its row,
    stopping at the row boundary. If a row has multiple colors,
    each cell keeps its own color (no overwrite).
    """
    h, w = _grid_dims(grid)
    if h == 0:
        return _deep_copy_grid(grid)

    result = [[0] * w for _ in range(h)]
    for r in range(h):
        # Collect non-zero values in row
        nz = [(c, grid[r][c]) for c in range(w) if grid[r][c] != 0]
        if not nz:
            continue
        if len(set(v for _, v in nz)) == 1:
            # Single color in row — fill entire row
            color = nz[0][1]
            result[r] = [color] * w
        else:
            # Multiple colors: copy as-is, extend to nearest neighbor
            # Fill left-to-right, then right-to-left (nearest wins)
            row = [0] * w
            last = 0
            for c in range(w):
                if grid[r][c] != 0:
                    last = grid[r][c]
                row[c] = last
            # Right-to-left pass to fill zeros at start
            last = 0
            for c in range(w - 1, -1, -1):
                if row[c] == 0:
                    if last != 0:
                        row[c] = last
                elif grid[r][c] != 0:
                    last = row[c]
            result[r] = row

    return result


def extend_to_border_v(grid: Grid) -> Grid:
    """Extend each non-zero cell vertically to fill its entire column."""
    return transpose(extend_to_border_h(transpose(grid)))


def spread_in_lanes_h(grid: Grid) -> Grid:
    """Spread non-separator colors horizontally within their row lanes.

    Identifies separator rows (rows where all cells are the same non-zero
    value). Divides grid into horizontal lanes between separators. Within
    each lane, if any row-cell has a non-bg, non-separator color, that color
    fills ALL the non-separator cells in that full row of the lane.

    Handles the common ARC pattern: grid divided by separator lines, and
    colored cells must propagate along their row.
    """
    h, w = _grid_dims(grid)
    if h == 0:
        return _deep_copy_grid(grid)

    from collections import Counter

    # Identify separator rows (all same non-zero value)
    sep_color_counts: Counter = Counter()
    sep_rows: set[int] = set()
    for r in range(h):
        vals = set(grid[r])
        if len(vals) == 1 and grid[r][0] != 0:
            sep_rows.add(r)
            sep_color_counts[grid[r][0]] += 1

    # Separator columns
    sep_cols: set[int] = set()
    for c in range(w):
        vals = set(grid[r][c] for r in range(h))
        if len(vals) == 1 and grid[0][c] != 0:
            sep_cols.add(c)
            sep_color_counts[grid[0][c]] += 1

    # The separator color is the most common across sep rows/cols
    if not sep_color_counts:
        return _deep_copy_grid(grid)
    sep_color = sep_color_counts.most_common(1)[0][0]

    result = _deep_copy_grid(grid)

    # For each non-separator row, find its non-separator, non-bg color
    for r in range(h):
        if r in sep_rows:
            continue
        # Collect colors in this row (excluding sep cells and zero)
        row_colors = [grid[r][c] for c in range(w)
                      if c not in sep_cols and grid[r][c] != 0 and grid[r][c] != sep_color]
        if not row_colors:
            continue
        # Dominant color in the lane row
        fill_color = Counter(row_colors).most_common(1)[0][0]
        # Fill all non-separator cells in this row
        for c in range(w):
            if c not in sep_cols and grid[r][c] != sep_color:
                result[r][c] = fill_color

    return result


def spread_in_lanes_v(grid: Grid) -> Grid:
    """Spread non-separator colors vertically within their column lanes.

    Mirror of spread_in_lanes_h but operates on columns.
    """
    return transpose(spread_in_lanes_h(transpose(grid)))


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
        # Symmetry completion
        ("complete_symmetry_h", complete_symmetry_h),
        ("complete_symmetry_v", complete_symmetry_v),
        ("complete_symmetry_4", complete_symmetry_4),
        # Denoise / majority voting
        ("denoise_3x3", denoise_3x3),
        ("denoise_5x5", denoise_5x5),
        # Grid overlay (boolean ops on halves)
        ("xor_halves_v", xor_halves_v),
        ("or_halves_v", or_halves_v),
        ("and_halves_v", and_halves_v),
        ("xor_halves_h", xor_halves_h),
        ("or_halves_h", or_halves_h),
        ("and_halves_h", and_halves_h),
        # Color frequency ops
        ("swap_most_least", swap_most_least),
        ("recolor_least_common", recolor_least_common),
        # Pattern stacking
        ("repeat_rows_2x", repeat_rows_2x),
        ("repeat_cols_2x", repeat_cols_2x),
        ("stack_with_mirror_v", stack_with_mirror_v),
        ("stack_with_mirror_h", stack_with_mirror_h),
        # Diagonal operations
        ("mirror_diagonal_main", mirror_diagonal_main),
        ("mirror_diagonal_anti", mirror_diagonal_anti),
        # Fill operations
        ("fill_holes_per_color", fill_holes_per_color),
        ("fill_rectangles", fill_rectangles),
        # Sort operations
        ("sort_cols_by_color_count", sort_cols_by_color_count),
        # Grid arithmetic
        ("grid_difference", grid_difference),
        ("grid_difference_h", grid_difference_h),
        # Spread / erode (morphological)
        ("spread_colors", spread_colors),
        ("erode", erode),
        # Color mask
        ("keep_only_largest_color", keep_only_largest_color),
        ("keep_only_smallest_color", keep_only_smallest_color),
        # Tile / pattern extraction
        ("extract_repeating_tile", extract_repeating_tile),
        ("extract_top_left_block", extract_top_left_block),
        ("extract_bottom_right_block", extract_bottom_right_block),
        ("split_sep_overlay", split_by_separator_and_overlay),
        ("split_sep_xor", split_by_separator_and_xor),
        ("compress_rows", compress_rows),
        ("compress_cols", compress_cols),
        ("max_color_per_cell", max_color_per_cell),
        ("min_color_per_cell", min_color_per_cell),
        ("extract_unique_block", extract_unique_block),
        ("flatten_to_row", flatten_to_row),
        ("flatten_to_column", flatten_to_column),
        ("count_objects_grid", count_objects_as_grid),
        ("mode_color_per_row", mode_color_per_row),
        ("mode_color_per_col", mode_color_per_col),
        # Tile completion and masking
        ("fill_tile_pattern", fill_tile_pattern),
        ("fill_by_symmetry", fill_by_symmetry),
        ("recolor_by_nearest_border", recolor_by_nearest_border),
        ("extend_to_border_h", extend_to_border_h),
        ("extend_to_border_v", extend_to_border_v),
        # Lane spreading (colors spread within separator-defined lanes)
        ("spread_in_lanes_h", spread_in_lanes_h),
        ("spread_in_lanes_v", spread_in_lanes_v),
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
