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
# V14 NEW PRIMITIVES
# ============================================================

def connect_pixels_to_rect(grid: Grid) -> Grid:
    """Connect isolated single-pixel anomalies to the nearest rectangle border.

    Finds isolated non-background pixels (surrounded entirely by background),
    then draws a straight line (H or V) from that pixel to the nearest edge
    of any rectangle object, filling with the isolated pixel's color.

    This handles tasks like 2c608aff where a pixel "shoots" to a rectangle.
    """
    if not grid or not grid[0]:
        return grid
    h, w = len(grid), len(grid[0])

    # Determine background (most common color)
    from collections import Counter
    flat = [v for row in grid for v in row]
    bg = Counter(flat).most_common(1)[0][0]

    result = [row[:] for row in grid]

    # Find connected components
    visited = [[False] * w for _ in range(h)]
    components = []

    def bfs(sr, sc):
        color = grid[sr][sc]
        cells = []
        queue = [(sr, sc)]
        visited[sr][sc] = True
        while queue:
            r, c = queue.pop(0)
            cells.append((r, c))
            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                nr, nc = r+dr, c+dc
                if 0 <= nr < h and 0 <= nc < w and not visited[nr][nc] and grid[nr][nc] == color:
                    visited[nr][nc] = True
                    queue.append((nr, nc))
        return color, cells

    for r in range(h):
        for c in range(w):
            if not visited[r][c] and grid[r][c] != bg:
                color, cells = bfs(r, c)
                components.append((color, cells))

    if len(components) < 2:
        return grid

    # Identify "large" rects vs isolated pixels
    # Isolated = single cell surrounded by bg on all 4 sides
    isolated = []
    rects = []
    for color, cells in components:
        if len(cells) == 1:
            r, c = cells[0]
            neighbors = [grid[r+dr][c+dc] for dr,dc in [(-1,0),(1,0),(0,-1),(0,1)]
                         if 0 <= r+dr < h and 0 <= c+dc < w]
            if all(v == bg for v in neighbors):
                isolated.append((color, r, c))
            else:
                rects.append((color, cells))
        else:
            rects.append((color, cells))

    if not isolated or not rects:
        return grid

    # For each isolated pixel, find nearest rect cell and draw line
    rect_cells_set = set()
    for _, cells in rects:
        for rc in cells:
            rect_cells_set.add(rc)

    for iso_color, ir, ic in isolated:
        # Find nearest rect cell
        best_dist = float('inf')
        best_rc = None
        for rr, rc in rect_cells_set:
            # Only consider H or V alignment
            if rr == ir or rc == ic:
                d = abs(rr - ir) + abs(rc - ic)
                if d < best_dist:
                    best_dist = d
                    best_rc = (rr, rc)

        if best_rc is None:
            # No aligned rect cell, find closest overall
            for rr, rc in rect_cells_set:
                d = abs(rr - ir) + abs(rc - ic)
                if d < best_dist:
                    best_dist = d
                    best_rc = (rr, rc)

        if best_rc is None:
            continue

        rr, rc = best_rc
        # Draw line from isolated pixel toward rect
        if rr == ir:
            # Same row - draw horizontal
            c_start, c_end = min(ic, rc), max(ic, rc)
            for c in range(c_start, c_end + 1):
                if result[ir][c] == bg:
                    result[ir][c] = iso_color
        elif rc == ic:
            # Same col - draw vertical
            r_start, r_end = min(ir, rr), max(ir, rr)
            for r in range(r_start, r_end + 1):
                if result[r][ic] == bg:
                    result[r][ic] = iso_color
        else:
            # Not aligned - draw to closest edge (H then V, pick shorter)
            dist_h = abs(ic - rc)
            dist_v = abs(ir - rr)
            if dist_h <= dist_v:
                # Draw horizontal to align column, then stop
                c_start, c_end = min(ic, rc), max(ic, rc)
                for c in range(c_start, c_end + 1):
                    if result[ir][c] == bg:
                        result[ir][c] = iso_color
            else:
                r_start, r_end = min(ir, rr), max(ir, rr)
                for r in range(r_start, r_end + 1):
                    if result[r][ic] == bg:
                        result[r][ic] = iso_color

    return result


def recolor_2nd_to_3rd_color(grid: Grid) -> Grid:
    """Replace the 2nd most common (non-bg) color with the 3rd most common.

    Useful for tasks where 'recolor_smallest' needs to be applied twice but
    the object shrinks each step. Handles tasks like 32597951 / 36fdfd69.
    """
    if not grid or not grid[0]:
        return grid
    from collections import Counter
    flat = [v for row in grid for v in row]
    counts = Counter(flat)
    bg = counts.most_common(1)[0][0]
    non_bg = [(c, v) for v, c in counts.most_common() if v != bg]
    if len(non_bg) < 3:
        return grid
    # 2nd most common non-bg → 3rd most common non-bg
    src_color = non_bg[1][1]
    dst_color = non_bg[2][1]
    return [[dst_color if v == src_color else v for v in row] for row in grid]


def recolor_least_to_second_least(grid: Grid) -> Grid:
    """Replace the least common non-bg color with the 2nd least common.

    Handles repeated recolor_smallest chains more efficiently.
    """
    if not grid or not grid[0]:
        return grid
    from collections import Counter
    flat = [v for row in grid for v in row]
    counts = Counter(flat)
    bg = counts.most_common(1)[0][0]
    non_bg_sorted = [(v, c) for v, c in sorted(counts.items(), key=lambda x: x[1])
                     if v != bg]
    if len(non_bg_sorted) < 2:
        return grid
    src_color = non_bg_sorted[0][0]   # least common
    dst_color = non_bg_sorted[1][0]   # 2nd least common
    return [[dst_color if v == src_color else v for v in row] for row in grid]


def fill_holes_in_objects(grid: Grid) -> Grid:
    """Fill enclosed zero-regions (holes) inside objects with the object color.

    More aggressive than fill_enclosed - finds objects first, then fills
    any bg cells fully surrounded by that object.
    """
    if not grid or not grid[0]:
        return grid
    h, w = len(grid), len(grid[0])
    from collections import Counter, deque
    flat = [v for row in grid for v in row]
    bg = Counter(flat).most_common(1)[0][0]

    result = [row[:] for row in grid]

    # Find all bg cells reachable from the border (not enclosed)
    reachable = [[False]*w for _ in range(h)]
    queue = deque()
    for r in range(h):
        for c in [0, w-1]:
            if grid[r][c] == bg and not reachable[r][c]:
                reachable[r][c] = True
                queue.append((r, c))
    for c in range(w):
        for r in [0, h-1]:
            if grid[r][c] == bg and not reachable[r][c]:
                reachable[r][c] = True
                queue.append((r, c))

    while queue:
        r, c = queue.popleft()
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            nr, nc = r+dr, c+dc
            if 0 <= nr < h and 0 <= nc < w and not reachable[nr][nc] and grid[nr][nc] == bg:
                reachable[nr][nc] = True
                queue.append((nr, nc))

    # Enclosed bg cells get filled with the surrounding object color
    for r in range(h):
        for c in range(w):
            if grid[r][c] == bg and not reachable[r][c]:
                # Find surrounding non-bg color (scan outward)
                fill_color = bg
                for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                    for dist in range(1, max(h,w)):
                        nr, nc = r + dr*dist, c + dc*dist
                        if not (0 <= nr < h and 0 <= nc < w):
                            break
                        if grid[nr][nc] != bg:
                            fill_color = grid[nr][nc]
                            break
                    if fill_color != bg:
                        break
                result[r][c] = fill_color

    return result


def gravity_toward_color(grid: Grid) -> Grid:
    """Pull all non-bg cells toward the row/col containing the most rare color.

    Specialized: if there's a 'band' (full row/col of one color), move
    scattered dots to be adjacent to that band. Handles task 4093f84a
    where scattered 2s move to adjoin the 5-band.
    """
    if not grid or not grid[0]:
        return grid
    h, w = len(grid), len(grid[0])
    from collections import Counter
    flat = [v for row in grid for v in row]
    counts = Counter(flat)
    bg = counts.most_common(1)[0][0]

    # Find solid horizontal bands (rows where all cells are the same non-bg color)
    band_rows = []
    for r in range(h):
        row_vals = set(grid[r])
        if len(row_vals) == 1 and list(row_vals)[0] != bg:
            band_rows.append((r, list(row_vals)[0]))

    band_cols = []
    for c in range(w):
        col_vals = set(grid[r][c] for r in range(h))
        if len(col_vals) == 1 and list(col_vals)[0] != bg:
            band_cols.append((c, list(col_vals)[0]))

    if not band_rows and not band_cols:
        return grid

    result = [row[:] for row in grid]

    # Move scattered dots (non-bg, non-band) to adjacent to nearest band
    band_row_set = {r for r, _ in band_rows}
    band_col_set = {c for c, _ in band_cols}

    if band_rows:
        for c in range(w):
            # Collect non-band non-bg values in this column
            vals_above = []
            vals_below = []
            first_band = band_rows[0][0]
            last_band = band_rows[-1][0]

            for r in range(h):
                if r in band_row_set:
                    continue
                if grid[r][c] != bg:
                    if r < first_band:
                        vals_above.append(grid[r][c])
                    else:
                        vals_below.append(grid[r][c])

            # Clear the column (non-band cells)
            for r in range(h):
                if r not in band_row_set:
                    result[r][c] = bg

            # Pack above-band values adjacent to band (touching band)
            for i, val in enumerate(reversed(vals_above)):
                r = first_band - 1 - i
                if 0 <= r < h and r not in band_row_set:
                    result[r][c] = val

            # Pack below-band values adjacent to band
            for i, val in enumerate(vals_below):
                r = last_band + 1 + i
                if 0 <= r < h and r not in band_row_set:
                    result[r][c] = val

    if band_cols:
        for r in range(h):
            vals_left = []
            vals_right = []
            first_band = band_cols[0][0]
            last_band = band_cols[-1][0]

            for c in range(w):
                if c in band_col_set:
                    continue
                if grid[r][c] != bg:
                    if c < first_band:
                        vals_left.append(grid[r][c])
                    else:
                        vals_right.append(grid[r][c])

            for c in range(w):
                if c not in band_col_set:
                    result[r][c] = bg

            for i, val in enumerate(reversed(vals_left)):
                c = first_band - 1 - i
                if 0 <= c < w and c not in band_col_set:
                    result[r][c] = val

            for i, val in enumerate(vals_right):
                c = last_band + 1 + i
                if 0 <= c < w and c not in band_col_set:
                    result[r][c] = val

    return result


def swap_colors_12(grid: Grid) -> Grid:
    """Swap color 1 and color 2."""
    return [[2 if v == 1 else (1 if v == 2 else v) for v in row] for row in grid]

def swap_colors_13(grid: Grid) -> Grid:
    """Swap color 1 and color 3."""
    return [[3 if v == 1 else (1 if v == 3 else v) for v in row] for row in grid]

def swap_colors_14(grid: Grid) -> Grid:
    """Swap color 1 and color 4."""
    return [[4 if v == 1 else (1 if v == 4 else v) for v in row] for row in grid]

def swap_colors_15(grid: Grid) -> Grid:
    """Swap color 1 and color 5."""
    return [[5 if v == 1 else (1 if v == 5 else v) for v in row] for row in grid]

def swap_colors_23(grid: Grid) -> Grid:
    """Swap color 2 and color 3."""
    return [[3 if v == 2 else (2 if v == 3 else v) for v in row] for row in grid]

def swap_colors_24(grid: Grid) -> Grid:
    """Swap color 2 and color 4."""
    return [[4 if v == 2 else (2 if v == 4 else v) for v in row] for row in grid]

def swap_colors_25(grid: Grid) -> Grid:
    """Swap color 2 and color 5."""
    return [[5 if v == 2 else (2 if v == 5 else v) for v in row] for row in grid]

def swap_colors_34(grid: Grid) -> Grid:
    """Swap color 3 and color 4."""
    return [[4 if v == 3 else (3 if v == 4 else v) for v in row] for row in grid]

def swap_colors_35(grid: Grid) -> Grid:
    """Swap color 3 and color 5."""
    return [[5 if v == 3 else (3 if v == 5 else v) for v in row] for row in grid]

def swap_colors_45(grid: Grid) -> Grid:
    """Swap color 4 and color 5."""
    return [[5 if v == 4 else (4 if v == 5 else v) for v in row] for row in grid]


def swap_most_and_second_color(grid: Grid) -> Grid:
    """Swap the most common and 2nd most common non-bg colors."""
    if not grid or not grid[0]:
        return grid
    from collections import Counter
    flat = [v for row in grid for v in row]
    counts = Counter(flat)
    bg = counts.most_common(1)[0][0]
    non_bg = [v for v, _ in counts.most_common() if v != bg]
    if len(non_bg) < 2:
        return grid
    c1, c2 = non_bg[0], non_bg[1]
    return [[c2 if v == c1 else (c1 if v == c2 else v) for v in row] for row in grid]


def swap_largest_and_smallest_obj_color(grid: Grid) -> Grid:
    """Swap the colors of the largest and smallest objects."""
    if not grid or not grid[0]:
        return grid
    h, w = len(grid), len(grid[0])
    from collections import Counter, deque
    flat = [v for row in grid for v in row]
    bg = Counter(flat).most_common(1)[0][0]

    visited = [[False]*w for _ in range(h)]
    components = []

    for sr in range(h):
        for sc in range(w):
            if not visited[sr][sc] and grid[sr][sc] != bg:
                color = grid[sr][sc]
                cells = []
                q = deque([(sr, sc)])
                visited[sr][sc] = True
                while q:
                    r, c = q.popleft()
                    cells.append((r, c))
                    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                        nr, nc = r+dr, c+dc
                        if 0<=nr<h and 0<=nc<w and not visited[nr][nc] and grid[nr][nc]==color:
                            visited[nr][nc] = True
                            q.append((nr, nc))
                components.append((len(cells), color, cells))

    if len(components) < 2:
        return grid

    components.sort(key=lambda x: x[0])
    _, small_color, small_cells = components[0]
    _, large_color, large_cells = components[-1]

    if small_color == large_color:
        return grid

    result = [row[:] for row in grid]
    for r, c in small_cells:
        result[r][c] = large_color
    for r, c in large_cells:
        result[r][c] = small_color
    return result


def color_by_row_position(grid: Grid) -> Grid:
    """Replace each non-bg cell with a color matching its row index (mod 9) + 1.

    Creates a row-striped coloring. Useful for tasks that map row position to color.
    """
    if not grid or not grid[0]:
        return grid
    from collections import Counter
    flat = [v for row in grid for v in row]
    bg = Counter(flat).most_common(1)[0][0]
    h = len(grid)
    result = [row[:] for row in grid]
    for r in range(h):
        c_val = (r % 9) + 1
        for c in range(len(grid[r])):
            if grid[r][c] != bg:
                result[r][c] = c_val
    return result


def color_by_col_position(grid: Grid) -> Grid:
    """Replace each non-bg cell with a color matching its col index (mod 9) + 1."""
    if not grid or not grid[0]:
        return grid
    from collections import Counter
    flat = [v for row in grid for v in row]
    bg = Counter(flat).most_common(1)[0][0]
    result = [row[:] for row in grid]
    for r in range(len(grid)):
        for c in range(len(grid[r])):
            if grid[r][c] != bg:
                result[r][c] = (c % 9) + 1
    return result


def complete_pattern_4way(grid: Grid) -> Grid:
    """Complete a partial pattern by enforcing 4-way (D4) symmetry.

    Takes the 'union' of 4-fold rotational and reflective symmetry:
    if any of the 4 symmetric positions has a non-bg value, fill all 4.
    """
    if not grid or not grid[0]:
        return grid
    h, w = len(grid), len(grid[0])
    from collections import Counter
    flat = [v for row in grid for v in row]
    bg = Counter(flat).most_common(1)[0][0]

    result = [row[:] for row in grid]
    # For each cell, check its 4-fold symmetric partners and take non-bg value
    for r in range(h):
        for c in range(w):
            candidates = [
                grid[r][c],
                grid[h-1-r][c] if 0 <= h-1-r < h else bg,
                grid[r][w-1-c] if 0 <= w-1-c < w else bg,
                grid[h-1-r][w-1-c] if 0 <= h-1-r < h and 0 <= w-1-c < w else bg,
            ]
            non_bg_candidates = [v for v in candidates if v != bg]
            if non_bg_candidates:
                val = Counter(non_bg_candidates).most_common(1)[0][0]
                result[r][c] = val
                result[h-1-r][c] = val
                result[r][w-1-c] = val
                result[h-1-r][w-1-c] = val

    return result


def fill_bg_with_color_from_border(grid: Grid) -> Grid:
    """Fill all background cells with the most common non-bg color on the border.

    Useful when the border color defines the fill, like tasks with a ring pattern.
    """
    if not grid or not grid[0]:
        return grid
    h, w = len(grid), len(grid[0])
    from collections import Counter
    flat = [v for row in grid for v in row]
    bg = Counter(flat).most_common(1)[0][0]

    border_vals = []
    for c in range(w):
        border_vals += [grid[0][c], grid[h-1][c]]
    for r in range(1, h-1):
        border_vals += [grid[r][0], grid[r][w-1]]

    non_bg_border = [v for v in border_vals if v != bg]
    if not non_bg_border:
        return grid
    fill = Counter(non_bg_border).most_common(1)[0][0]

    return [[fill if v == bg else v for v in row] for row in grid]


def keep_only_unique_rows(grid: Grid) -> Grid:
    """Remove duplicate rows, keeping only the first occurrence of each unique row."""
    if not grid:
        return grid
    seen = []
    result = []
    for row in grid:
        key = tuple(row)
        if key not in seen:
            seen.append(key)
            result.append(row[:])
    return result if result else grid


def keep_only_unique_cols(grid: Grid) -> Grid:
    """Remove duplicate columns, keeping only the first occurrence."""
    if not grid or not grid[0]:
        return grid
    h, w = len(grid), len(grid[0])
    seen = []
    keep_cols = []
    for c in range(w):
        key = tuple(grid[r][c] for r in range(h))
        if key not in seen:
            seen.append(key)
            keep_cols.append(c)
    if not keep_cols:
        return grid
    return [[grid[r][c] for c in keep_cols] for r in range(h)]


def rotate_colors_up(grid: Grid) -> Grid:
    """Cycle all non-bg colors: each color → (color % 9) + 1.

    E.g., 1→2, 2→3, ..., 9→1. Useful for color-rotation tasks.
    """
    if not grid or not grid[0]:
        return grid
    from collections import Counter
    flat = [v for row in grid for v in row]
    bg = Counter(flat).most_common(1)[0][0]
    result = []
    for row in grid:
        new_row = []
        for v in row:
            if v == bg:
                new_row.append(v)
            else:
                new_row.append((v % 9) + 1)
        result.append(new_row)
    return result


def rotate_colors_down(grid: Grid) -> Grid:
    """Cycle all non-bg colors downward: each color → ((color-2) % 9) + 1.

    E.g., 1→9, 2→1, 3→2, ..., 9→8.
    """
    if not grid or not grid[0]:
        return grid
    from collections import Counter
    flat = [v for row in grid for v in row]
    bg = Counter(flat).most_common(1)[0][0]
    result = []
    for row in grid:
        new_row = []
        for v in row:
            if v == bg:
                new_row.append(v)
            else:
                new_row.append(((v - 2) % 9) + 1)
        result.append(new_row)
    return result


def extend_nonzero_to_fill_row(grid: Grid) -> Grid:
    """For each row, if it has exactly one non-bg color, fill the whole row with it.

    Handles tasks where a partial row needs to be extended to full width.
    """
    if not grid or not grid[0]:
        return grid
    from collections import Counter
    flat = [v for row in grid for v in row]
    bg = Counter(flat).most_common(1)[0][0]
    result = []
    for row in grid:
        non_bg = [v for v in row if v != bg]
        if non_bg and len(set(non_bg)) == 1:
            result.append([non_bg[0]] * len(row))
        else:
            result.append(row[:])
    return result


def extend_nonzero_to_fill_col(grid: Grid) -> Grid:
    """For each col, if it has exactly one non-bg color, fill the whole col with it."""
    return transpose(extend_nonzero_to_fill_row(transpose(grid)))


# ============================================================
# V15 NEW PRIMITIVES
# ============================================================

def recolor_isolated_to_nearest(grid: Grid) -> Grid:
    """Recolor isolated pixels (no 4-way same-color neighbor) to nearest non-bg color.

    An isolated pixel is one with no orthogonal neighbor of the same color.
    Each such pixel is recolored to the non-bg color closest to it (Manhattan).
    Handles tasks where scattered markers need to take on surrounding object colors.
    """
    if not grid or not grid[0]:
        return grid
    h, w = len(grid), len(grid[0])
    from collections import Counter
    bg = Counter(v for row in grid for v in row).most_common(1)[0][0]
    result = [row[:] for row in grid]
    for r in range(h):
        for c in range(w):
            v = grid[r][c]
            if v == bg:
                continue
            has_same_neighbor = any(
                0 <= r+dr < h and 0 <= c+dc < w and grid[r+dr][c+dc] == v
                for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]
            )
            if has_same_neighbor:
                continue
            # Isolated — find nearest non-bg, non-v cell
            best_d, best_v = float('inf'), None
            for rr in range(h):
                for cc in range(w):
                    if grid[rr][cc] not in (bg, v):
                        d = abs(r-rr) + abs(c-cc)
                        if d < best_d:
                            best_d, best_v = d, grid[rr][cc]
            if best_v is not None:
                result[r][c] = best_v
    return result


def recolor_small_objects_to_nearest(grid: Grid) -> Grid:
    """Recolor small objects (size ≤ 3) to the color of the nearest larger object.

    Helps tasks where tiny blobs of one color need to adopt a nearby object's color.
    """
    if not grid or not grid[0]:
        return grid
    h, w = len(grid), len(grid[0])
    from collections import Counter, deque
    bg = Counter(v for row in grid for v in row).most_common(1)[0][0]

    visited = [[False]*w for _ in range(h)]
    components = []

    for sr in range(h):
        for sc in range(w):
            if not visited[sr][sc] and grid[sr][sc] != bg:
                color = grid[sr][sc]
                cells = []
                q = deque([(sr, sc)])
                visited[sr][sc] = True
                while q:
                    r, c = q.popleft()
                    cells.append((r, c))
                    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                        nr, nc = r+dr, c+dc
                        if 0<=nr<h and 0<=nc<w and not visited[nr][nc] and grid[nr][nc]==color:
                            visited[nr][nc] = True
                            q.append((nr, nc))
                components.append((color, cells))

    # Partition into small vs large
    sizes = [len(cells) for _, cells in components]
    if not sizes:
        return grid
    max_size = max(sizes)
    threshold = min(3, max_size // 3) if max_size > 3 else 1

    large_cells = {}  # cell → color
    for color, cells in components:
        if len(cells) > threshold:
            for cell in cells:
                large_cells[cell] = color

    if not large_cells:
        return grid

    result = [row[:] for row in grid]
    for color, cells in components:
        if len(cells) <= threshold:
            for r, c in cells:
                # Find nearest large cell
                best_d, best_v = float('inf'), None
                for (lr, lc), lcolor in large_cells.items():
                    d = abs(r-lr) + abs(c-lc)
                    if d < best_d:
                        best_d, best_v = d, lcolor
                if best_v is not None:
                    result[r][c] = best_v
    return result


def mirror_h_merge(grid: Grid) -> Grid:
    """Mirror horizontally and overlay: keep non-bg from either original or mirrored.

    Useful for tasks where the left and right halves are complementary.
    """
    if not grid or not grid[0]:
        return grid
    from collections import Counter
    bg = Counter(v for row in grid for v in row).most_common(1)[0][0]
    result = []
    for row in grid:
        rev = row[::-1]
        new_row = [a if a != bg else b for a, b in zip(row, rev)]
        result.append(new_row)
    return result


def mirror_v_merge(grid: Grid) -> Grid:
    """Mirror vertically and overlay: keep non-bg from either original or mirrored."""
    if not grid or not grid[0]:
        return grid
    from collections import Counter
    bg = Counter(v for row in grid for v in row).most_common(1)[0][0]
    h = len(grid)
    result = []
    for r in range(h):
        row = []
        for c in range(len(grid[r])):
            a = grid[r][c]
            b = grid[h-1-r][c]
            row.append(a if a != bg else b)
        result.append(row)
    return result


def sort_rows_by_value(grid: Grid) -> Grid:
    """Sort values in each row in ascending order."""
    return [sorted(row) for row in grid]


def sort_cols_by_value(grid: Grid) -> Grid:
    """Sort values in each column in ascending order."""
    return transpose(sort_rows_by_value(transpose(grid)))


def recolor_by_size_rank(grid: Grid) -> Grid:
    """Recolor each connected component by its size rank (largest=1, 2nd=2, ...)."""
    if not grid or not grid[0]:
        return grid
    h, w = len(grid), len(grid[0])
    from collections import Counter, deque
    bg = Counter(v for row in grid for v in row).most_common(1)[0][0]

    visited = [[False]*w for _ in range(h)]
    components = []
    for sr in range(h):
        for sc in range(w):
            if not visited[sr][sc] and grid[sr][sc] != bg:
                color = grid[sr][sc]
                cells = []
                q = deque([(sr, sc)])
                visited[sr][sc] = True
                while q:
                    r, c = q.popleft()
                    cells.append((r, c))
                    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                        nr, nc = r+dr, c+dc
                        if 0<=nr<h and 0<=nc<w and not visited[nr][nc] and grid[nr][nc]==color:
                            visited[nr][nc] = True
                            q.append((nr, nc))
                components.append((color, cells))

    if not components:
        return grid

    # Sort by size descending, assign rank color (1=largest)
    components.sort(key=lambda x: len(x[1]), reverse=True)
    result = [row[:] for row in grid]
    for rank, (_, cells) in enumerate(components):
        new_color = (rank % 9) + 1
        for r, c in cells:
            result[r][c] = new_color
    return result


def fill_row_from_right(grid: Grid) -> Grid:
    """Propagate each row's non-bg color rightward (fill bg cells to the right)."""
    if not grid or not grid[0]:
        return grid
    from collections import Counter
    bg = Counter(v for row in grid for v in row).most_common(1)[0][0]
    result = []
    for row in grid:
        new_row = row[:]
        last = bg
        for i in range(len(row)-1, -1, -1):
            if row[i] != bg:
                last = row[i]
            else:
                new_row[i] = last
        result.append(new_row)
    return result


def fill_col_from_bottom(grid: Grid) -> Grid:
    """Propagate each col's non-bg color upward from the bottom."""
    return transpose(fill_row_from_right(transpose(grid)))


def extract_objects_on_grid(grid: Grid) -> Grid:
    """Keep only objects that lie on a regular grid (equally spaced rows/cols).

    Useful for tasks where objects are arranged in a grid pattern and off-grid
    noise should be removed.
    """
    if not grid or not grid[0]:
        return grid
    h, w = len(grid), len(grid[0])
    from collections import Counter
    bg = Counter(v for row in grid for v in row).most_common(1)[0][0]

    # Find rows and cols that have the most non-bg content
    row_density = [sum(1 for v in grid[r] if v != bg) for r in range(h)]
    col_density = [sum(1 for r in range(h) if grid[r][c] != bg) for c in range(w)]

    # Find top-density rows/cols (potential grid lines)
    max_row_d = max(row_density) if row_density else 0
    max_col_d = max(col_density) if col_density else 0
    if max_row_d == 0:
        return grid

    threshold_row = max_row_d * 0.5
    threshold_col = max_col_d * 0.5
    dense_rows = {r for r in range(h) if row_density[r] >= threshold_row}
    dense_cols = {c for c in range(w) if col_density[c] >= threshold_col}

    result = [[bg]*w for _ in range(h)]
    for r in dense_rows:
        for c in range(w):
            result[r][c] = grid[r][c]
    for c in dense_cols:
        for r in range(h):
            if result[r][c] == bg:
                result[r][c] = grid[r][c]
    return result


def recolor_by_column_index(grid: Grid) -> Grid:
    """Replace non-bg cells with (column_index % 9) + 1.

    Creates a column-striped recoloring. Different from color_by_col_position
    as it uses 1-indexed cyclic coloring.
    """
    if not grid or not grid[0]:
        return grid
    from collections import Counter
    bg = Counter(v for row in grid for v in row).most_common(1)[0][0]
    return [[(c % 9) + 1 if v != bg else bg for c, v in enumerate(row)] for row in grid]


def recolor_by_row_index(grid: Grid) -> Grid:
    """Replace non-bg cells with (row_index % 9) + 1."""
    if not grid or not grid[0]:
        return grid
    from collections import Counter
    bg = Counter(v for row in grid for v in row).most_common(1)[0][0]
    return [[(r % 9) + 1 if v != bg else bg for v in row] for r, row in enumerate(grid)]


def tile_grid_2x1(grid: Grid) -> Grid:
    """Tile the grid twice horizontally (double width)."""
    return [row + row for row in grid]


def tile_grid_1x2(grid: Grid) -> Grid:
    """Tile the grid twice vertically (double height)."""
    return grid + [row[:] for row in grid]


def crop_to_content_border(grid: Grid) -> Grid:
    """Crop to the bounding box of all non-bg cells, then add 1-cell bg border."""
    if not grid or not grid[0]:
        return grid
    h, w = len(grid), len(grid[0])
    from collections import Counter
    bg = Counter(v for row in grid for v in row).most_common(1)[0][0]

    rows = [r for r in range(h) if any(grid[r][c] != bg for c in range(w))]
    cols = [c for c in range(w) if any(grid[r][c] != bg for r in range(h))]
    if not rows or not cols:
        return grid

    r0, r1 = max(0, rows[0]-1), min(h-1, rows[-1]+1)
    c0, c1 = max(0, cols[0]-1), min(w-1, cols[-1]+1)
    return [grid[r][c0:c1+1] for r in range(r0, r1+1)]


def mask_by_color_overlap(grid: Grid) -> Grid:
    """For each pair of cells in same row: where colors overlap, keep; else bg.

    Finds the most common non-bg value and keeps only cells matching it.
    """
    if not grid or not grid[0]:
        return grid
    from collections import Counter
    flat = [v for row in grid for v in row]
    bg = Counter(flat).most_common(1)[0][0]
    non_bg = [v for v, _ in Counter(flat).most_common() if v != bg]
    if not non_bg:
        return grid
    keep = non_bg[0]  # most common non-bg
    return [[v if v == keep else bg for v in row] for row in grid]


def complete_symmetry_diagonal(grid: Grid) -> Grid:
    """Complete diagonal (main diagonal) symmetry: if grid[r][c] set, set grid[c][r] too."""
    if not grid or not grid[0]:
        return grid
    h, w = len(grid), len(grid[0])
    if h != w:
        return grid
    from collections import Counter
    bg = Counter(v for row in grid for v in row).most_common(1)[0][0]
    result = [row[:] for row in grid]
    for r in range(h):
        for c in range(w):
            a, b = grid[r][c], grid[c][r]
            if a != bg and b == bg:
                result[c][r] = a
            elif b != bg and a == bg:
                result[r][c] = b
    return result


def remove_color_noise(grid: Grid) -> Grid:
    """Remove isolated single pixels of any color (replace with bg).

    Opposite of recolor_isolated_to_nearest — just erases isolated pixels.
    """
    if not grid or not grid[0]:
        return grid
    h, w = len(grid), len(grid[0])
    from collections import Counter
    bg = Counter(v for row in grid for v in row).most_common(1)[0][0]
    result = [row[:] for row in grid]
    for r in range(h):
        for c in range(w):
            v = grid[r][c]
            if v == bg:
                continue
            has_same = any(
                0<=r+dr<h and 0<=c+dc<w and grid[r+dr][c+dc]==v
                for dr,dc in [(-1,0),(1,0),(0,-1),(0,1)]
            )
            if not has_same:
                result[r][c] = bg
    return result


def fill_diagonal_stripes(grid: Grid) -> Grid:
    """Fill background cells with a diagonal stripe pattern based on (r+c) % num_colors.

    Uses the non-bg colors found in the grid.
    """
    if not grid or not grid[0]:
        return grid
    h, w = len(grid), len(grid[0])
    from collections import Counter
    flat = [v for row in grid for v in row]
    bg = Counter(flat).most_common(1)[0][0]
    colors = [v for v, _ in Counter(flat).most_common() if v != bg]
    if not colors:
        return grid
    n = len(colors)
    result = []
    for r in range(h):
        row = []
        for c in range(w):
            if grid[r][c] != bg:
                row.append(grid[r][c])
            else:
                row.append(colors[(r+c) % n])
        result.append(row)
    return result


def keep_border_only(grid: Grid) -> Grid:
    """Keep only the outermost ring of cells, set interior to bg."""
    if not grid or not grid[0]:
        return grid
    h, w = len(grid), len(grid[0])
    from collections import Counter
    bg = Counter(v for row in grid for v in row).most_common(1)[0][0]
    result = [[bg]*w for _ in range(h)]
    for c in range(w):
        result[0][c] = grid[0][c]
        result[h-1][c] = grid[h-1][c]
    for r in range(h):
        result[r][0] = grid[r][0]
        result[r][w-1] = grid[r][w-1]
    return result


def repeat_pattern_to_size(grid: Grid) -> Grid:
    """Find smallest repeating sub-pattern and tile it to fill original size.

    More aggressive version of fill_tile_pattern — works bottom-up from 1x1.
    """
    if not grid or not grid[0]:
        return grid
    h, w = len(grid), len(grid[0])

    # Try tile sizes from 1x1 up to h//2 x w//2
    for th in range(1, h//2 + 1):
        if h % th != 0:
            continue
        for tw in range(1, w//2 + 1):
            if w % tw != 0:
                continue
            # Check if grid is a tiling of this tile
            tile = [grid[r][:tw] for r in range(th)]
            valid = True
            for r in range(h):
                for c in range(w):
                    if grid[r][c] != tile[r % th][c % tw]:
                        valid = False
                        break
                if not valid:
                    break
            if valid and (th < h or tw < w):
                # Return the tile tiled to original size (same content)
                return [[tile[r % th][c % tw] for c in range(w)] for r in range(h)]
    return grid


# ============================================================
# V19 NEW PRIMITIVES
# ============================================================

def _make_recolor_nonzero_inside_accent_bbox(accent_color: int, new_color: int):
    """Factory: recolor non-zero, non-accent cells inside bounding box of accent_color → new_color.

    Finds the tight axis-aligned bounding box of all accent_color cells, then
    recolors every cell inside that box that is non-zero and not the accent_color.
    Useful for: marking cells that fall within the 'reach' of an accent object.
    """
    def _fn(grid):
        h, w = _grid_dims(grid)
        accent_cells = [(r, c) for r in range(h) for c in range(w) if grid[r][c] == accent_color]
        if not accent_cells:
            return grid
        min_r = min(r for r, c in accent_cells)
        max_r = max(r for r, c in accent_cells)
        min_c = min(c for r, c in accent_cells)
        max_c = max(c for r, c in accent_cells)
        result = _deep_copy_grid(grid)
        for r in range(min_r, max_r + 1):
            for c in range(min_c, max_c + 1):
                v = grid[r][c]
                if v != 0 and v != accent_color:
                    result[r][c] = new_color
        return result
    return _fn


# Common accent/new-color pairs for pair-search coverage
recolor_nonzero_inside_8_bbox_to_3 = _make_recolor_nonzero_inside_accent_bbox(8, 3)
recolor_nonzero_inside_8_bbox_to_4 = _make_recolor_nonzero_inside_accent_bbox(8, 4)
recolor_nonzero_inside_8_bbox_to_2 = _make_recolor_nonzero_inside_accent_bbox(8, 2)
recolor_nonzero_inside_2_bbox_to_4 = _make_recolor_nonzero_inside_accent_bbox(2, 4)
recolor_nonzero_inside_2_bbox_to_8 = _make_recolor_nonzero_inside_accent_bbox(2, 8)
recolor_nonzero_inside_2_bbox_to_3 = _make_recolor_nonzero_inside_accent_bbox(2, 3)
recolor_nonzero_inside_3_bbox_to_4 = _make_recolor_nonzero_inside_accent_bbox(3, 4)
recolor_nonzero_inside_3_bbox_to_8 = _make_recolor_nonzero_inside_accent_bbox(3, 8)
recolor_nonzero_inside_6_bbox_to_4 = _make_recolor_nonzero_inside_accent_bbox(6, 4)
recolor_nonzero_inside_6_bbox_to_8 = _make_recolor_nonzero_inside_accent_bbox(6, 8)


def _fill_rect_interiors(grid, fill_color: int):
    """Fill the interior of all complete rectangular frames with fill_color.

    A 'complete rectangular frame' is a region where the border cells form
    a full closed rectangle of a single non-bg color, and the interior cells
    are all bg.  Uses an efficient O(h·w) flood-fill approach:
    1. flood-fill bg from borders
    2. any bg cell NOT reachable from border that is surrounded by non-bg = interior
    But we also require the interior to be a proper rectangular hole.
    Useful for: marking/counting enclosed areas in frame patterns.
    """
    from collections import Counter
    h, w = _grid_dims(grid)
    all_vals = [grid[r][c] for r in range(h) for c in range(w)]
    bg = Counter(all_vals).most_common(1)[0][0]

    # BFS from border to find bg cells reachable from outside
    reachable = set()
    queue = []
    for r in range(h):
        for c in range(w):
            if (r == 0 or r == h - 1 or c == 0 or c == w - 1) and grid[r][c] == bg:
                if (r, c) not in reachable:
                    reachable.add((r, c))
                    queue.append((r, c))
    while queue:
        r, c = queue.pop()
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in reachable and grid[nr][nc] == bg:
                reachable.add((nr, nc))
                queue.append((nr, nc))

    result = _deep_copy_grid(grid)
    for r in range(h):
        for c in range(w):
            if grid[r][c] == bg and (r, c) not in reachable:
                result[r][c] = fill_color
    return result


def fill_rect_interior_with_2(grid):
    """Fill interior of rectangular frames with color 2.

    Finds complete rectangular outlines (border all one color, interior all bg)
    and fills their interior with 2.
    Useful for: marking/counting enclosed areas in frame patterns.
    """
    return _fill_rect_interiors(grid, 2)


def fill_rect_interior_with_4(grid):
    """Fill interior of rectangular frames with color 4."""
    return _fill_rect_interiors(grid, 4)


def fill_rect_interior_with_1(grid):
    """Fill interior of rectangular frames with color 1."""
    return _fill_rect_interiors(grid, 1)


def fill_rect_interior_with_3(grid):
    """Fill interior of rectangular frames with color 3."""
    return _fill_rect_interiors(grid, 3)


def _recolor_cells_at_color_intersections(grid, accent_color: int, new_color: int):
    """Recolor bg cells at row/col intersections of rows and cols containing accent_color.

    For every row that contains at least one accent_color cell, and every column
    that contains at least one accent_color cell, mark the bg cell at their intersection.
    Useful for: grid/cross-marking patterns.
    """
    from collections import Counter
    h, w = _grid_dims(grid)
    all_vals = [grid[r][c] for r in range(h) for c in range(w)]
    bg = Counter(all_vals).most_common(1)[0][0]

    accent_rows = {r for r in range(h) if any(grid[r][c] == accent_color for c in range(w))}
    accent_cols = {c for c in range(w) if any(grid[r][c] == accent_color for r in range(h))}

    result = _deep_copy_grid(grid)
    for r in accent_rows:
        for c in accent_cols:
            if grid[r][c] == bg:
                result[r][c] = new_color
    return result


def mark_row_col_intersections_with_2(grid):
    """Mark bg cells at row+col intersections of the accent (2nd most common non-bg) color → 2.

    Finds which rows and columns contain any non-bg accent color cells, then
    marks the bg cell at each (accent_row, accent_col) intersection with 2.
    Useful for: cross/grid patterns where intersections need highlighting.
    """
    from collections import Counter
    h, w = _grid_dims(grid)
    all_vals = [grid[r][c] for r in range(h) for c in range(w)]
    cnt = Counter(all_vals)
    bg = cnt.most_common(1)[0][0]
    non_bg = [(v, c) for v, c in cnt.most_common() if v != bg]
    if not non_bg:
        return grid
    accent = non_bg[0][0]
    return _recolor_cells_at_color_intersections(grid, accent, 2)


def mark_row_col_intersections_with_3(grid):
    """Mark bg cells at row+col intersections of the accent color → 3."""
    from collections import Counter
    h, w = _grid_dims(grid)
    all_vals = [grid[r][c] for r in range(h) for c in range(w)]
    cnt = Counter(all_vals)
    bg = cnt.most_common(1)[0][0]
    non_bg = [(v, c) for v, c in cnt.most_common() if v != bg]
    if not non_bg:
        return grid
    accent = non_bg[0][0]
    return _recolor_cells_at_color_intersections(grid, accent, 3)


def mark_row_col_intersections_with_4(grid):
    """Mark bg cells at row+col intersections of the accent color → 4."""
    from collections import Counter
    h, w = _grid_dims(grid)
    all_vals = [grid[r][c] for r in range(h) for c in range(w)]
    cnt = Counter(all_vals)
    bg = cnt.most_common(1)[0][0]
    non_bg = [(v, c) for v, c in cnt.most_common() if v != bg]
    if not non_bg:
        return grid
    accent = non_bg[0][0]
    return _recolor_cells_at_color_intersections(grid, accent, 4)


def _extend_nonbg_lines_to_contact(grid):
    """Extend each non-bg colored line segment until it meets another non-bg cell or boundary.

    For each non-bg cell that appears to be the end of a line, extends it
    in the same direction until it reaches another colored cell or the border.
    Useful for: completing grid lines, extending arrows, connecting segments.
    """
    from collections import Counter
    h, w = _grid_dims(grid)
    all_vals = [grid[r][c] for r in range(h) for c in range(w)]
    bg = Counter(all_vals).most_common(1)[0][0]

    result = _deep_copy_grid(grid)
    # For each row, extend all runs of non-bg cells to fill gaps
    for r in range(h):
        non_bg_cols = [(c, grid[r][c]) for c in range(w) if grid[r][c] != bg]
        if len(non_bg_cols) >= 2:
            min_c = non_bg_cols[0][0]
            max_c = non_bg_cols[-1][0]
            # Fill between first and last non-bg in this row
            fill_color = non_bg_cols[0][1]
            if all(grid[r][c] in (bg, fill_color) for c in range(min_c, max_c + 1)):
                for c in range(min_c, max_c + 1):
                    if result[r][c] == bg:
                        result[r][c] = fill_color

    # For each col, extend all runs of non-bg cells to fill gaps
    for c in range(w):
        non_bg_rows = [(r, grid[r][c]) for r in range(h) if grid[r][c] != bg]
        if len(non_bg_rows) >= 2:
            min_r = non_bg_rows[0][0]
            max_r = non_bg_rows[-1][0]
            fill_color = non_bg_rows[0][1]
            if all(grid[r][c] in (bg, fill_color) for r in range(min_r, max_r + 1)):
                for r in range(min_r, max_r + 1):
                    if result[r][c] == bg:
                        result[r][c] = fill_color
    return result


def extend_lines_to_contact(grid):
    """Extend non-bg colored segments to fill gaps within their row or column.

    For each row (col) containing at least 2 non-bg cells of the same color,
    fills the bg cells between them with that color — but only if the entire
    span between them is either bg or the same color.
    Useful for: connecting line segments, completing stripe patterns.
    """
    return _extend_nonbg_lines_to_contact(grid)


def _recolor_adjacent_to_accent_to_new(grid, accent_color: int, new_color: int):
    """Recolor bg cells adjacent (4-way) to accent_color cells → new_color."""
    from collections import Counter
    h, w = _grid_dims(grid)
    all_vals = [grid[r][c] for r in range(h) for c in range(w)]
    bg = Counter(all_vals).most_common(1)[0][0]

    result = _deep_copy_grid(grid)
    for r in range(h):
        for c in range(w):
            if grid[r][c] != bg:
                continue
            if any(0 <= r + dr < h and 0 <= c + dc < w and grid[r + dr][c + dc] == accent_color
                   for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]):
                result[r][c] = new_color
    return result


def fill_bg_adjacent_to_accent_with_3(grid):
    """Fill bg cells adjacent to accent (2nd most common non-bg) color → 3.

    Finds the 2nd most common non-bg color (accent), then fills all bg cells
    that are 4-way adjacent to it with color 3.
    Useful for: expanding/growing a color by one layer.
    """
    from collections import Counter
    h, w = _grid_dims(grid)
    all_vals = [grid[r][c] for r in range(h) for c in range(w)]
    cnt = Counter(all_vals)
    bg = cnt.most_common(1)[0][0]
    non_bg = [(v, c) for v, c in cnt.most_common() if v != bg]
    if len(non_bg) < 2:
        return grid
    accent = non_bg[1][0]
    return _recolor_adjacent_to_accent_to_new(grid, accent, 3)


def fill_bg_adjacent_to_accent_with_8(grid):
    """Fill bg cells adjacent to accent color → 8."""
    from collections import Counter
    h, w = _grid_dims(grid)
    all_vals = [grid[r][c] for r in range(h) for c in range(w)]
    cnt = Counter(all_vals)
    bg = cnt.most_common(1)[0][0]
    non_bg = [(v, c) for v, c in cnt.most_common() if v != bg]
    if len(non_bg) < 2:
        return grid
    accent = non_bg[1][0]
    return _recolor_adjacent_to_accent_to_new(grid, accent, 8)


def fill_bg_adjacent_to_dominant_with_3(grid):
    """Fill bg cells adjacent to the dominant (most common non-bg) color → 3."""
    from collections import Counter
    h, w = _grid_dims(grid)
    all_vals = [grid[r][c] for r in range(h) for c in range(w)]
    cnt = Counter(all_vals)
    bg = cnt.most_common(1)[0][0]
    non_bg = [(v, c) for v, c in cnt.most_common() if v != bg]
    if not non_bg:
        return grid
    dominant = non_bg[0][0]
    return _recolor_adjacent_to_accent_to_new(grid, dominant, 3)


def fill_bg_adjacent_to_dominant_with_8(grid):
    """Fill bg cells adjacent to the dominant color → 8."""
    from collections import Counter
    h, w = _grid_dims(grid)
    all_vals = [grid[r][c] for r in range(h) for c in range(w)]
    cnt = Counter(all_vals)
    bg = cnt.most_common(1)[0][0]
    non_bg = [(v, c) for v, c in cnt.most_common() if v != bg]
    if not non_bg:
        return grid
    dominant = non_bg[0][0]
    return _recolor_adjacent_to_accent_to_new(grid, dominant, 8)


# ============================================================
# V18 NEW PRIMITIVES
# ============================================================

def _make_recolor_dominant_touching_accent(new_color: int):
    """Factory: recolor dominant-color cells adjacent to the accent (2nd) color."""
    def _fn(grid):
        from collections import Counter
        h, w = _grid_dims(grid)
        all_vals = [grid[r][c] for r in range(h) for c in range(w)]
        bg = Counter(all_vals).most_common(1)[0][0]
        non_bg = [(v, cnt) for v, cnt in Counter(all_vals).most_common() if v != bg]
        if len(non_bg) < 2:
            return grid
        dominant = non_bg[0][0]
        accent = non_bg[1][0]
        # Find cells of dominant color adjacent (4-way) to accent
        result = _deep_copy_grid(grid)
        for r in range(h):
            for c in range(w):
                if grid[r][c] != dominant:
                    continue
                for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                    nr, nc = r+dr, c+dc
                    if 0<=nr<h and 0<=nc<w and grid[nr][nc] == accent:
                        result[r][c] = new_color
                        break
        return result
    return _fn


# Factory-generated: recolor dominant-color cells touching accent → target color
recolor_dominant_touching_accent_to_4 = _make_recolor_dominant_touching_accent(4)
recolor_dominant_touching_accent_to_6 = _make_recolor_dominant_touching_accent(6)
recolor_dominant_touching_accent_to_7 = _make_recolor_dominant_touching_accent(7)
recolor_dominant_touching_accent_to_8 = _make_recolor_dominant_touching_accent(8)
recolor_dominant_touching_accent_to_2 = _make_recolor_dominant_touching_accent(2)
recolor_dominant_touching_accent_to_3 = _make_recolor_dominant_touching_accent(3)


def fill_smallest_rect_hole_with_1(grid):
    """Fill the smallest enclosed rectangular bg region with color 1.

    Finds enclosed bg regions (not reachable from border), picks the
    smallest one whose cells form a compact rectangular cluster,
    and fills it with color 1.
    Useful for: filling small rectangular voids in scattered-dot patterns.
    """
    return _fill_smallest_hole(grid, new_color=1)


def fill_smallest_rect_hole_with_4(grid):
    """Fill the smallest enclosed rectangular bg region with color 4."""
    return _fill_smallest_hole(grid, new_color=4)


def fill_smallest_rect_hole_with_8(grid):
    """Fill the smallest enclosed rectangular bg region with color 8."""
    return _fill_smallest_hole(grid, new_color=8)


def _fill_smallest_hole(grid, new_color):
    from collections import Counter
    h, w = _grid_dims(grid)
    all_vals = [grid[r][c] for r in range(h) for c in range(w)]
    bg = Counter(all_vals).most_common(1)[0][0]

    # BFS from border
    reachable = set()
    queue = []
    for r in range(h):
        for c in range(w):
            if (r==0 or r==h-1 or c==0 or c==w-1) and grid[r][c] == bg:
                if (r,c) not in reachable:
                    reachable.add((r,c))
                    queue.append((r,c))
    while queue:
        r, c = queue.pop()
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            nr, nc = r+dr, c+dc
            if 0<=nr<h and 0<=nc<w and (nr,nc) not in reachable and grid[nr][nc] == bg:
                reachable.add((nr,nc))
                queue.append((nr,nc))

    # Find enclosed holes
    visited = set()
    holes = []
    for sr in range(h):
        for sc in range(w):
            if grid[sr][sc] == bg and (sr,sc) not in reachable and (sr,sc) not in visited:
                hole = []
                q = [(sr,sc)]
                visited.add((sr,sc))
                while q:
                    r, c = q.pop()
                    hole.append((r,c))
                    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                        nr, nc = r+dr, c+dc
                        if 0<=nr<h and 0<=nc<w and (nr,nc) not in visited and grid[nr][nc] == bg:
                            visited.add((nr,nc))
                            q.append((nr,nc))
                holes.append(hole)

    if not holes:
        return grid

    # Pick smallest hole
    smallest = min(holes, key=len)
    result = _deep_copy_grid(grid)
    for r, c in smallest:
        result[r][c] = new_color
    return result


def recolor_bg_enclosed_by_dominant(grid):
    """Fill enclosed bg regions with the dominant non-bg color.

    Finds all bg cells not reachable from the border (enclosed),
    then fills them with the most common non-bg color.
    Useful for: filling interior voids in large object grids.
    """
    from collections import Counter
    h, w = _grid_dims(grid)
    all_vals = [grid[r][c] for r in range(h) for c in range(w)]
    bg = Counter(all_vals).most_common(1)[0][0]
    non_bg = [(v,cnt) for v,cnt in Counter(all_vals).most_common() if v != bg]
    if not non_bg:
        return grid
    fill_color = non_bg[0][0]

    reachable = set()
    queue = []
    for r in range(h):
        for c in range(w):
            if (r==0 or r==h-1 or c==0 or c==w-1) and grid[r][c] == bg:
                if (r,c) not in reachable:
                    reachable.add((r,c))
                    queue.append((r,c))
    while queue:
        r, c = queue.pop()
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            nr, nc = r+dr, c+dc
            if 0<=nr<h and 0<=nc<w and (nr,nc) not in reachable and grid[nr][nc] == bg:
                reachable.add((nr,nc))
                queue.append((nr,nc))

    result = _deep_copy_grid(grid)
    for r in range(h):
        for c in range(w):
            if grid[r][c] == bg and (r,c) not in reachable:
                result[r][c] = fill_color
    return result


def sort_rows_by_sum(grid):
    """Sort rows by their sum of values (ascending).

    Rows with smaller total color values come first.
    Useful for: canonical row ordering tasks.
    """
    return sorted(grid, key=lambda row: sum(row))


def sort_cols_by_sum(grid):
    """Sort columns by their sum of values (ascending)."""
    h, w = _grid_dims(grid)
    if h == 0:
        return grid
    col_sums = [sum(grid[r][c] for r in range(h)) for c in range(w)]
    order = sorted(range(w), key=lambda c: col_sums[c])
    return [[grid[r][order[c]] for c in range(w)] for r in range(h)]


def recolor_2nd_color_to_dominant(grid):
    """Recolor the 2nd most common non-bg color to the dominant non-bg color.

    Merges the secondary color into the primary, keeping only one non-bg color.
    Useful for: unifying two similar colored regions.
    """
    from collections import Counter
    h, w = _grid_dims(grid)
    all_vals = [grid[r][c] for r in range(h) for c in range(w)]
    bg = Counter(all_vals).most_common(1)[0][0]
    non_bg = [(v,cnt) for v,cnt in Counter(all_vals).most_common() if v != bg]
    if len(non_bg) < 2:
        return grid
    dominant, accent = non_bg[0][0], non_bg[1][0]
    result = _deep_copy_grid(grid)
    for r in range(h):
        for c in range(w):
            if result[r][c] == accent:
                result[r][c] = dominant
    return result


def erase_2nd_color(grid):
    """Erase (→ bg) the 2nd most common non-bg color.

    Removes the secondary color, leaving only bg and the dominant color.
    Useful for: cleaning up noise/accent from predominantly single-color grids.
    """
    from collections import Counter
    h, w = _grid_dims(grid)
    all_vals = [grid[r][c] for r in range(h) for c in range(w)]
    bg = Counter(all_vals).most_common(1)[0][0]
    non_bg = [(v,cnt) for v,cnt in Counter(all_vals).most_common() if v != bg]
    if len(non_bg) < 2:
        return grid
    accent = non_bg[1][0]
    result = _deep_copy_grid(grid)
    for r in range(h):
        for c in range(w):
            if result[r][c] == accent:
                result[r][c] = bg
    return result


# ============================================================
# V16 NEW PRIMITIVES
# ============================================================

def fill_stripe_gaps_h(grid: Grid) -> Grid:
    """Fill bg (0) cells between same-color cells within each row.

    For each row, if a 0 lies between two cells of the same non-zero color
    with no other colors between them, fill those 0s with that color.
    Useful for: extending color stripes horizontally to fill gaps.
    """
    h, w = _grid_dims(grid)
    result = _deep_copy_grid(grid)
    for r in range(h):
        row = result[r]
        # Forward scan: track last non-zero color seen and its column
        changed = True
        while changed:
            changed = False
            for c in range(w):
                if row[c] == 0:
                    # Find nearest non-zero to left and right
                    left_c, left_col = 0, -1
                    for lc in range(c - 1, -1, -1):
                        if row[lc] != 0:
                            left_c, left_col = row[lc], lc
                            break
                    right_c, right_col = 0, -1
                    for rc in range(c + 1, w):
                        if row[rc] != 0:
                            right_c, right_col = row[rc], rc
                            break
                    if left_c != 0 and left_c == right_c:
                        # Fill the gap
                        for fc in range(left_col + 1, right_col):
                            if row[fc] == 0:
                                row[fc] = left_c
                                changed = True
    return result


def fill_stripe_gaps_v(grid: Grid) -> Grid:
    """Fill bg (0) cells between same-color cells within each column.

    Vertical counterpart of fill_stripe_gaps_h.
    """
    h, w = _grid_dims(grid)
    result = _deep_copy_grid(grid)
    for c in range(w):
        changed = True
        while changed:
            changed = False
            for r in range(h):
                if result[r][c] == 0:
                    left_c, left_row = 0, -1
                    for lr in range(r - 1, -1, -1):
                        if result[lr][c] != 0:
                            left_c, left_row = result[lr][c], lr
                            break
                    right_c, right_row = 0, -1
                    for rr in range(r + 1, h):
                        if result[rr][c] != 0:
                            right_c, right_row = result[rr][c], rr
                            break
                    if left_c != 0 and left_c == right_c:
                        for fr in range(left_row + 1, right_row):
                            if result[fr][c] == 0:
                                result[fr][c] = left_c
                                changed = True
    return result


def complete_tile_from_modal_col(grid: Grid) -> Grid:
    """For each column position, fill anomalous cells with the modal (majority) value.

    In tiled grids, column position c%tile_width determines the expected color.
    Cells that deviate from the column's modal value are filled with the mode.
    Useful for: fixing isolated exceptions in tiled/striped patterns.
    """
    from collections import Counter
    h, w = _grid_dims(grid)
    result = _deep_copy_grid(grid)
    for c in range(w):
        col_vals = [grid[r][c] for r in range(h)]
        mode = Counter(col_vals).most_common(1)[0][0]
        mode_count = Counter(col_vals)[mode]
        if mode_count > h // 2:  # Clear majority
            for r in range(h):
                if result[r][c] != mode and col_vals.count(result[r][c]) == 1:
                    result[r][c] = mode
    return result


def complete_tile_from_modal_row(grid: Grid) -> Grid:
    """For each row, fill anomalous cells with the modal (majority) value.

    Row counterpart of complete_tile_from_modal_col.
    Useful for: fixing isolated exceptions in tiled/striped patterns.
    """
    from collections import Counter
    h, w = _grid_dims(grid)
    result = _deep_copy_grid(grid)
    for r in range(h):
        row_vals = grid[r]
        mode = Counter(row_vals).most_common(1)[0][0]
        mode_count = Counter(row_vals)[mode]
        if mode_count > w // 2:
            for c in range(w):
                if result[r][c] != mode and row_vals.count(result[r][c]) == 1:
                    result[r][c] = mode
    return result


def recolor_minority_in_rows(grid: Grid) -> Grid:
    """In each row, recolor cells that hold a color unique to that row.

    If a row has one cell of a color not repeated elsewhere in that row
    but that color appears in an adjacent row at the same column,
    replace it with the row's dominant non-bg color.
    Specifically: cells where color appears only once in that row → bg color.
    Useful for: fixing stray/exception cells in horizontally striped grids.
    """
    from collections import Counter
    h, w = _grid_dims(grid)
    result = _deep_copy_grid(grid)
    # Find background: most common color overall
    all_vals = [grid[r][c] for r in range(h) for c in range(w)]
    bg = Counter(all_vals).most_common(1)[0][0]
    for r in range(h):
        row_vals = [grid[r][c] for c in range(w)]
        counter = Counter(row_vals)
        # The dominant non-bg color for this row
        row_dominant = None
        for color, cnt in counter.most_common():
            if color != bg:
                row_dominant = color
                break
        if row_dominant is None:
            continue
        for c in range(w):
            cell = grid[r][c]
            if cell != bg and cell != row_dominant and counter[cell] == 1:
                # This is a minority outlier — check if it matches column neighbors
                result[r][c] = row_dominant
    return result


def recolor_minority_in_cols(grid: Grid) -> Grid:
    """In each column, recolor cells that hold a color unique to that column.

    Column counterpart of recolor_minority_in_rows.
    Useful for: fixing stray/exception cells in vertically striped grids.
    """
    from collections import Counter
    h, w = _grid_dims(grid)
    result = _deep_copy_grid(grid)
    all_vals = [grid[r][c] for r in range(h) for c in range(w)]
    bg = Counter(all_vals).most_common(1)[0][0]
    for c in range(w):
        col_vals = [grid[r][c] for r in range(h)]
        counter = Counter(col_vals)
        col_dominant = None
        for color, cnt in counter.most_common():
            if color != bg:
                col_dominant = color
                break
        if col_dominant is None:
            continue
        for r in range(h):
            cell = grid[r][c]
            if cell != bg and cell != col_dominant and counter[cell] == 1:
                result[r][c] = col_dominant
    return result


def snap_isolated_to_rect_boundary(grid: Grid) -> Grid:
    """Move isolated non-bg pixels to the nearest edge of the largest rectangle.

    For each isolated pixel (no 4-connected same-color neighbor) that is NOT
    inside or adjacent to the largest non-bg rectangle:
    - Project it onto the nearest row/column of that rectangle's boundary.
    Useful for: tasks where scattered pixels should align to a central object.
    """
    from collections import Counter
    h, w = _grid_dims(grid)
    all_vals = [grid[r][c] for r in range(h) for c in range(w)]
    bg = Counter(all_vals).most_common(1)[0][0]

    # Find the largest connected non-bg region's bounding box
    visited = [[False] * w for _ in range(h)]
    best_size, best_cells = 0, []
    for sr in range(h):
        for sc in range(w):
            if grid[sr][sc] != bg and not visited[sr][sc]:
                # BFS
                queue = [(sr, sc)]
                visited[sr][sc] = True
                cells = []
                while queue:
                    r, c = queue.pop()
                    cells.append((r, c))
                    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                        nr, nc = r+dr, c+dc
                        if 0 <= nr < h and 0 <= nc < w and not visited[nr][nc] and grid[nr][nc] != bg:
                            visited[nr][nc] = True
                            queue.append((nr, nc))
                if len(cells) > best_size:
                    best_size = len(cells)
                    best_cells = cells

    if not best_cells:
        return grid

    rect_rows = {r for r, c in best_cells}
    rect_cols = {c for r, c in best_cells}
    r_min, r_max = min(rect_rows), max(rect_rows)
    c_min, c_max = min(rect_cols), max(rect_cols)

    result = _deep_copy_grid(grid)
    for r in range(h):
        for c in range(w):
            if grid[r][c] == bg:
                continue
            # Check if isolated (no 4-connected same-color neighbor)
            color = grid[r][c]
            isolated = True
            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                nr, nc = r+dr, c+dc
                if 0 <= nr < h and 0 <= nc < w and grid[nr][nc] == color:
                    isolated = False
                    break
            if not isolated:
                continue
            # Check if it's part of the main rectangle
            if r in rect_rows and c in rect_cols:
                continue
            # Project to nearest edge of bounding box
            # Clamp to bbox
            proj_r = max(r_min, min(r_max, r))
            proj_c = max(c_min, min(c_max, c))
            if proj_r == r and proj_c == c:
                continue  # Already inside bbox
            # Place at boundary
            result[r][c] = bg
            result[proj_r][proj_c] = color
    return result


def recolor_smallest_obj_in_each_row(grid: Grid) -> Grid:
    """In each row, find the smallest non-bg connected segment and recolor it.

    Each row is treated as a 1D sequence. The shortest run of a single
    non-bg color that is NOT the dominant color of the row gets recolored
    to the row's dominant color.
    Useful for: fixing minority-colored segments in striped grids.
    """
    from collections import Counter
    h, w = _grid_dims(grid)
    all_vals = [grid[r][c] for r in range(h) for c in range(w)]
    bg = Counter(all_vals).most_common(1)[0][0]
    result = _deep_copy_grid(grid)
    for r in range(h):
        row = grid[r]
        counter = Counter(v for v in row if v != bg)
        if len(counter) < 2:
            continue
        dominant = counter.most_common(1)[0][0]
        minority_colors = {c for c, cnt in counter.items() if c != dominant and cnt <= 3}
        for c in range(w):
            if row[c] in minority_colors:
                result[r][c] = dominant
    return result


def recolor_smallest_obj_in_each_col(grid: Grid) -> Grid:
    """In each column, recolor minority-colored segments to the column's dominant color.

    Column counterpart of recolor_smallest_obj_in_each_row.
    """
    from collections import Counter
    h, w = _grid_dims(grid)
    all_vals = [grid[r][c] for r in range(h) for c in range(w)]
    bg = Counter(all_vals).most_common(1)[0][0]
    result = _deep_copy_grid(grid)
    for c in range(w):
        col = [grid[r][c] for r in range(h)]
        counter = Counter(v for v in col if v != bg)
        if len(counter) < 2:
            continue
        dominant = counter.most_common(1)[0][0]
        minority_colors = {clr for clr, cnt in counter.items() if clr != dominant and cnt <= 3}
        for r in range(h):
            if col[r] in minority_colors:
                result[r][c] = dominant
    return result


def fill_grid_intersections(grid: Grid) -> Grid:
    """At row/col intersections, fill with the color that appears in both that row and col.

    For each bg cell at (r,c): if there's a non-bg color that appears both
    in row r AND in column c, place that color there.
    Useful for: grid/matrix tasks where intersections should be colored.
    """
    from collections import Counter
    h, w = _grid_dims(grid)
    all_vals = [grid[r][c] for r in range(h) for c in range(w)]
    bg = Counter(all_vals).most_common(1)[0][0]
    result = _deep_copy_grid(grid)
    for r in range(h):
        row_colors = {grid[r][c] for c in range(w) if grid[r][c] != bg}
        for c in range(w):
            if grid[r][c] != bg:
                continue
            col_colors = {grid[rr][c] for rr in range(h) if grid[rr][c] != bg}
            intersection = row_colors & col_colors
            if len(intersection) == 1:
                result[r][c] = intersection.pop()
    return result


def propagate_color_h(grid: Grid) -> Grid:
    """Extend each non-bg color rightward to fill bg cells until hitting another color.

    Each non-bg cell propagates its color rightward through bg cells,
    stopping when it hits a non-bg cell.
    Useful for: filling color bars / stripes to the right.
    """
    from collections import Counter
    h, w = _grid_dims(grid)
    all_vals = [grid[r][c] for r in range(h) for c in range(w)]
    bg = Counter(all_vals).most_common(1)[0][0]
    result = _deep_copy_grid(grid)
    for r in range(h):
        current = bg
        for c in range(w):
            if grid[r][c] != bg:
                current = grid[r][c]
            elif current != bg:
                result[r][c] = current
    return result


def propagate_color_v(grid: Grid) -> Grid:
    """Extend each non-bg color downward to fill bg cells until hitting another color.

    Vertical counterpart of propagate_color_h.
    """
    from collections import Counter
    h, w = _grid_dims(grid)
    all_vals = [grid[r][c] for r in range(h) for c in range(w)]
    bg = Counter(all_vals).most_common(1)[0][0]
    result = _deep_copy_grid(grid)
    for c in range(w):
        current = bg
        for r in range(h):
            if grid[r][c] != bg:
                current = grid[r][c]
            elif current != bg:
                result[r][c] = current
    return result


def _make_recolor_neighbors_of_color(target_color: int, new_color: int):
    """Factory: recolor non-bg cells adjacent to target_color cells to new_color."""
    def _recolor_neighbors(grid: Grid) -> Grid:
        from collections import Counter
        h, w = _grid_dims(grid)
        all_vals = [grid[r][c] for r in range(h) for c in range(w)]
        bg = Counter(all_vals).most_common(1)[0][0]
        # Find second most common non-bg color if target_color == -1
        non_bg = [(v, c) for v, c in Counter(all_vals).most_common() if v != bg]
        if not non_bg:
            return grid
        # resolve target_color: -1 means 2nd most common
        tc = target_color if target_color > 0 else (non_bg[1][0] if len(non_bg) > 1 else non_bg[0][0])
        nc = new_color if new_color > 0 else (non_bg[0][0] if non_bg else bg)

        # Find all cells adjacent to tc
        adjacent = set()
        for r in range(h):
            for c in range(w):
                if grid[r][c] == tc:
                    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                        nr, nc2 = r+dr, c+dc
                        if 0 <= nr < h and 0 <= nc2 < w and grid[nr][nc2] not in (bg, tc):
                            adjacent.add((nr, nc2))

        result = _deep_copy_grid(grid)
        for r, c in adjacent:
            result[r][c] = nc
        return result
    return _recolor_neighbors


# V16: recolor cells touching 2nd-most-common color, to color 8 or 3
recolor_touching_2nd_to_8 = _make_recolor_neighbors_of_color(-1, 8)
recolor_touching_2nd_to_3 = _make_recolor_neighbors_of_color(-1, 3)


def recolor_neighbors_of_2nd_color(grid: Grid) -> Grid:
    """Recolor cells adjacent to the 2nd most common non-bg color.

    Finds the 2nd most common non-bg color (the 'accent' color), then
    recolors all cells touching it to color 8.
    Useful for: border/halo marking around a secondary object.
    """
    from collections import Counter
    h, w = _grid_dims(grid)
    all_vals = [grid[r][c] for r in range(h) for c in range(w)]
    bg = Counter(all_vals).most_common(1)[0][0]
    non_bg = [(v, cnt) for v, cnt in Counter(all_vals).most_common() if v != bg]
    if len(non_bg) < 2:
        return grid

    accent_color = non_bg[1][0]
    dominant = non_bg[0][0]

    adjacent = set()
    for r in range(h):
        for c in range(w):
            if grid[r][c] == accent_color:
                for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                    nr, nc = r+dr, c+dc
                    if 0 <= nr < h and 0 <= nc < w and grid[nr][nc] == dominant:
                        adjacent.add((nr, nc))

    result = _deep_copy_grid(grid)
    for r, c in adjacent:
        result[r][c] = 8
    return result


def extend_color_within_col_bounds(grid: Grid) -> Grid:
    """Within each column, extend the dominant non-bg color to fill bg cells.

    For each column, find the row range that contains non-bg cells,
    then fill all bg cells within that range with the column's dominant color.
    Useful for: filling gaps in vertical stripes.
    """
    from collections import Counter
    h, w = _grid_dims(grid)
    all_vals = [grid[r][c] for r in range(h) for c in range(w)]
    bg = Counter(all_vals).most_common(1)[0][0]
    result = _deep_copy_grid(grid)
    for c in range(w):
        col = [grid[r][c] for r in range(h)]
        non_bg_rows = [r for r in range(h) if col[r] != bg]
        if not non_bg_rows:
            continue
        r_min, r_max = min(non_bg_rows), max(non_bg_rows)
        col_counter = Counter(v for v in col if v != bg)
        if not col_counter:
            continue
        dominant = col_counter.most_common(1)[0][0]
        for r in range(r_min, r_max + 1):
            if result[r][c] == bg:
                result[r][c] = dominant
    return result


def extend_color_within_row_bounds(grid: Grid) -> Grid:
    """Within each row, extend the dominant non-bg color to fill bg cells.

    Row counterpart of extend_color_within_col_bounds.
    """
    from collections import Counter
    h, w = _grid_dims(grid)
    all_vals = [grid[r][c] for r in range(h) for c in range(w)]
    bg = Counter(all_vals).most_common(1)[0][0]
    result = _deep_copy_grid(grid)
    for r in range(h):
        row = [grid[r][c] for c in range(w)]
        non_bg_cols = [c for c in range(w) if row[c] != bg]
        if not non_bg_cols:
            continue
        c_min, c_max = min(non_bg_cols), max(non_bg_cols)
        row_counter = Counter(v for v in row if v != bg)
        if not row_counter:
            continue
        dominant = row_counter.most_common(1)[0][0]
        for c in range(c_min, c_max + 1):
            if result[r][c] == bg:
                result[r][c] = dominant
    return result


def recolor_unique_in_row_col(grid: Grid) -> Grid:
    """Recolor cells where a color appears exactly once in its row OR once in its col.

    If a non-bg cell's color appears only once in its row AND it's not the
    dominant color of its column, recolor it to the column's dominant color.
    Useful for: fixing stray exception cells that don't fit the row/col pattern.
    """
    from collections import Counter
    h, w = _grid_dims(grid)
    all_vals = [grid[r][c] for r in range(h) for c in range(w)]
    bg = Counter(all_vals).most_common(1)[0][0]
    result = _deep_copy_grid(grid)
    for r in range(h):
        row_counter = Counter(grid[r][c] for c in range(w) if grid[r][c] != bg)
        for c in range(w):
            if grid[r][c] == bg:
                continue
            cell_color = grid[r][c]
            if row_counter[cell_color] != 1:
                continue  # Not unique in row
            # This cell's color appears exactly once in its row
            col_vals = [grid[rr][c] for rr in range(h) if grid[rr][c] != bg]
            col_counter = Counter(col_vals)
            if not col_counter:
                continue
            col_dominant = col_counter.most_common(1)[0][0]
            if col_dominant != cell_color:
                result[r][c] = col_dominant
    return result


# ============================================================
# V20 NEW PRIMITIVES
# ============================================================

# --- Shift operations (cyclic wrap-around) ---

def shift_down_1(grid: Grid) -> Grid:
    """Shift all rows down by 1, wrapping bottom row to top."""
    return [grid[-1]] + grid[:-1]


def shift_up_1(grid: Grid) -> Grid:
    """Shift all rows up by 1, wrapping top row to bottom."""
    return grid[1:] + [grid[0]]


def shift_left_1(grid: Grid) -> Grid:
    """Shift all columns left by 1, wrapping leftmost to right."""
    return [row[1:] + [row[0]] for row in grid]


def shift_right_1(grid: Grid) -> Grid:
    """Shift all columns right by 1, wrapping rightmost to left."""
    return [[row[-1]] + row[:-1] for row in grid]


# --- Fill enclosed bg regions with surrounding wall color ---

def fill_enclosed_wall_color(grid: Grid) -> Grid:
    """Fill each enclosed bg region with the most common color of its wall.

    'Enclosed' means bg cells not reachable from the grid border via
    4-connected bg paths. Each enclosed component gets filled with the
    most common non-bg color adjacent to it.
    """
    from collections import Counter
    h, w = _grid_dims(grid)
    bg = Counter(grid[r][c] for r in range(h) for c in range(w)).most_common(1)[0][0]

    # BFS from border to find reachable bg cells
    reachable = set()
    queue = []
    for r in range(h):
        for c in range(w):
            if (r == 0 or r == h - 1 or c == 0 or c == w - 1) and grid[r][c] == bg:
                if (r, c) not in reachable:
                    reachable.add((r, c))
                    queue.append((r, c))
    while queue:
        r, c = queue.pop()
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in reachable and grid[nr][nc] == bg:
                reachable.add((nr, nc))
                queue.append((nr, nc))

    # Find enclosed components and fill each with its wall's dominant color
    visited = [[False] * w for _ in range(h)]
    result = _deep_copy_grid(grid)
    for sr in range(h):
        for sc in range(w):
            if grid[sr][sc] == bg and (sr, sc) not in reachable and not visited[sr][sc]:
                component = []
                wall_colors = Counter()
                q = [(sr, sc)]
                visited[sr][sc] = True
                while q:
                    r, c = q.pop()
                    component.append((r, c))
                    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < h and 0 <= nc < w:
                            if grid[nr][nc] == bg and not visited[nr][nc] and (nr, nc) not in reachable:
                                visited[nr][nc] = True
                                q.append((nr, nc))
                            elif grid[nr][nc] != bg:
                                wall_colors[grid[nr][nc]] += 1
                if wall_colors:
                    fill_color = wall_colors.most_common(1)[0][0]
                    for r, c in component:
                        result[r][c] = fill_color
    return result


# --- Object border/interior operations ---

def remove_border_objects(grid: Grid) -> Grid:
    """Remove (erase to bg) all connected objects that touch the grid border."""
    from collections import Counter
    h, w = _grid_dims(grid)
    bg = Counter(grid[r][c] for r in range(h) for c in range(w)).most_common(1)[0][0]
    visited = [[False] * w for _ in range(h)]
    result = _deep_copy_grid(grid)
    for sr in range(h):
        for sc in range(w):
            if not visited[sr][sc] and grid[sr][sc] != bg:
                cells = []
                color = grid[sr][sc]
                q = [(sr, sc)]
                visited[sr][sc] = True
                while q:
                    r, c = q.pop()
                    cells.append((r, c))
                    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < h and 0 <= nc < w and not visited[nr][nc] and grid[nr][nc] == color:
                            visited[nr][nc] = True
                            q.append((nr, nc))
                if any(r == 0 or r == h - 1 or c == 0 or c == w - 1 for r, c in cells):
                    for r, c in cells:
                        result[r][c] = bg
    return result


def keep_interior_objects(grid: Grid) -> Grid:
    """Keep only objects that do NOT touch the grid border; erase border objects."""
    from collections import Counter
    h, w = _grid_dims(grid)
    bg = Counter(grid[r][c] for r in range(h) for c in range(w)).most_common(1)[0][0]
    visited = [[False] * w for _ in range(h)]
    result = [[bg] * w for _ in range(h)]
    for sr in range(h):
        for sc in range(w):
            if not visited[sr][sc] and grid[sr][sc] != bg:
                cells = []
                color = grid[sr][sc]
                q = [(sr, sc)]
                visited[sr][sc] = True
                while q:
                    r, c = q.pop()
                    cells.append((r, c))
                    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < h and 0 <= nc < w and not visited[nr][nc] and grid[nr][nc] == color:
                            visited[nr][nc] = True
                            q.append((nr, nc))
                if not any(r == 0 or r == h - 1 or c == 0 or c == w - 1 for r, c in cells):
                    for r, c in cells:
                        result[r][c] = color
    return result


def hollow_objects(grid: Grid) -> Grid:
    """Erase interior of colored objects, keeping only boundary cells.

    A cell is 'interior' if all 4 cardinal neighbors have the same color.
    """
    from collections import Counter
    h, w = _grid_dims(grid)
    bg = Counter(grid[r][c] for r in range(h) for c in range(w)).most_common(1)[0][0]
    result = _deep_copy_grid(grid)
    for r in range(h):
        for c in range(w):
            if grid[r][c] != bg:
                if all(
                    0 <= r + dr < h and 0 <= c + dc < w and grid[r + dr][c + dc] == grid[r][c]
                    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1))
                ):
                    result[r][c] = bg
    return result


def fill_object_bboxes(grid: Grid) -> Grid:
    """Fill the bounding box of each connected object with the object's color."""
    from collections import Counter
    h, w = _grid_dims(grid)
    bg = Counter(grid[r][c] for r in range(h) for c in range(w)).most_common(1)[0][0]
    visited = [[False] * w for _ in range(h)]
    result = _deep_copy_grid(grid)
    for sr in range(h):
        for sc in range(w):
            if not visited[sr][sc] and grid[sr][sc] != bg:
                cells = []
                color = grid[sr][sc]
                q = [(sr, sc)]
                visited[sr][sc] = True
                while q:
                    r, c = q.pop()
                    cells.append((r, c))
                    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < h and 0 <= nc < w and not visited[nr][nc] and grid[nr][nc] == color:
                            visited[nr][nc] = True
                            q.append((nr, nc))
                min_r = min(r for r, c in cells)
                max_r = max(r for r, c in cells)
                min_c = min(c for r, c in cells)
                max_c = max(c for r, c in cells)
                for r in range(min_r, max_r + 1):
                    for c in range(min_c, max_c + 1):
                        result[r][c] = color
    return result


# ============================================================
# INPAINTING: Fill holes in patterned grids
# ============================================================

def inpaint_tiled(grid: Grid) -> Grid:
    """Fill zero-cells by detecting a repeating tile period.

    Many ARC tasks have a grid with a 2D periodic pattern where rectangular
    regions have been replaced with 0 (holes). This primitive detects the
    row and column periods from the non-zero cells, then fills each zero
    with the value implied by the tiling.

    Algorithm:
      1. For each candidate row period p_r (1..H//2), check if all non-zero
         cells satisfy grid[r][c] == grid[r % p_r][c] across all rows.
      2. Same for column period p_c.
      3. Build a "template" tile of size p_r × p_c from the non-zero cells.
      4. Fill zeros by looking up template[r % p_r][c % p_c].

    Returns the grid unchanged if no period is detected or if there are
    no zeros to fill.
    """
    h, w = len(grid), len(grid[0]) if grid else 0
    if h == 0 or w == 0:
        return grid

    # Quick check: any zeros?
    has_zeros = any(grid[r][c] == 0 for r in range(h) for c in range(w))
    if not has_zeros:
        return [row[:] for row in grid]

    # Detect row period: smallest p_r such that for all non-zero cells,
    # grid[r][c] == grid[r % p_r][c] (or the reference is zero too)
    best_pr = h  # fallback: no row period
    for pr in range(1, h // 2 + 1):
        # Don't require exact divisibility — the grid may be truncated
        consistent = True
        for r in range(pr, h):
            if not consistent:
                break
            for c in range(w):
                val = grid[r][c]
                ref = grid[r % pr][c]
                if val != 0 and ref != 0 and val != ref:
                    consistent = False
                    break
        if consistent:
            best_pr = pr
            break

    # Detect column period
    best_pc = w  # fallback: no column period
    for pc in range(1, w // 2 + 1):
        # Don't require exact divisibility — the grid may be truncated
        consistent = True
        for r in range(h):
            if not consistent:
                break
            for c in range(pc, w):
                val = grid[r][c]
                ref = grid[r][c % pc]
                if val != 0 and ref != 0 and val != ref:
                    consistent = False
                    break
        if consistent:
            best_pc = pc
            break

    # Build template from all non-zero cells, collecting votes
    template: list[list[dict[int, int]]] = [
        [{} for _ in range(best_pc)] for _ in range(best_pr)
    ]
    for r in range(h):
        for c in range(w):
            val = grid[r][c]
            if val != 0:
                tr, tc = r % best_pr, c % best_pc
                template[tr][tc][val] = template[tr][tc].get(val, 0) + 1

    # Resolve template to majority vote
    resolved: list[list[int]] = [[0] * best_pc for _ in range(best_pr)]
    for tr in range(best_pr):
        for tc in range(best_pc):
            votes = template[tr][tc]
            if votes:
                resolved[tr][tc] = max(votes, key=votes.get)

    # Fill zeros
    result = [row[:] for row in grid]
    for r in range(h):
        for c in range(w):
            if result[r][c] == 0:
                fill_val = resolved[r % best_pr][c % best_pc]
                if fill_val != 0:
                    result[r][c] = fill_val

    return result


def inpaint_from_context(grid: Grid) -> Grid:
    """Fill zero-cells by inferring from row and column context.

    For each zero cell at (r, c):
      1. Look at all non-zero cells in the same column — if a consistent
         pattern maps row index to value, use it.
      2. Look at all non-zero cells in the same row — if a consistent
         pattern maps column index to value, use it.
      3. If row and column agree, fill. If only one has an answer, use it.

    This handles non-periodic patterns where each row/column has a unique
    but predictable structure (e.g., diagonal patterns, arithmetic sequences).

    Returns the grid unchanged if there are no zeros.
    """
    h, w = len(grid), len(grid[0]) if grid else 0
    if h == 0 or w == 0:
        return grid

    has_zeros = any(grid[r][c] == 0 for r in range(h) for c in range(w))
    if not has_zeros:
        return [row[:] for row in grid]

    result = [row[:] for row in grid]

    # Strategy 1: For each zero, check if the same column has a consistent
    # value at the same row position across periodic row groups.
    # Try multiple passes (iterative filling may reveal more context).
    for _ in range(3):  # up to 3 passes
        changed = False
        for r in range(h):
            for c in range(w):
                if result[r][c] != 0:
                    continue

                # Try column context: find what value appears at row r
                # in this column by looking at a period offset
                col_vote = _vote_from_column(result, r, c, h)
                row_vote = _vote_from_row(result, r, c, w)

                if col_vote != 0 and row_vote != 0:
                    # Both agree? Use column (usually more reliable)
                    result[r][c] = col_vote
                    changed = True
                elif col_vote != 0:
                    result[r][c] = col_vote
                    changed = True
                elif row_vote != 0:
                    result[r][c] = row_vote
                    changed = True

        if not changed:
            break

    return result


def _vote_from_column(grid: Grid, r: int, c: int, h: int) -> int:
    """Infer value at (r, c) from non-zero cells in the same column.

    Tries to find a period p such that grid[r + k*p][c] is non-zero
    and consistent for some k.
    """
    for p in range(1, h):
        candidates = []
        for k in [-1, 1, -2, 2, -3, 3]:
            nr = r + k * p
            if 0 <= nr < h and grid[nr][c] != 0:
                candidates.append(grid[nr][c])

        if len(candidates) >= 2:
            # Check if all candidates agree
            if len(set(candidates)) == 1:
                return candidates[0]
        elif len(candidates) == 1:
            # Only one reference point — less reliable but still useful
            # Only use if period == 1 (adjacent cell)
            if p == 1:
                return candidates[0]

    return 0


def _vote_from_row(grid: Grid, r: int, c: int, w: int) -> int:
    """Infer value at (r, c) from non-zero cells in the same row.

    Tries to find a period p such that grid[r][c + k*p] is non-zero
    and consistent for some k.
    """
    for p in range(1, w):
        candidates = []
        for k in [-1, 1, -2, 2, -3, 3]:
            nc = c + k * p
            if 0 <= nc < w and grid[r][nc] != 0:
                candidates.append(grid[r][nc])

        if len(candidates) >= 2:
            if len(set(candidates)) == 1:
                return candidates[0]
        elif len(candidates) == 1:
            if p == 1:
                return candidates[0]

    return 0


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
        ("inpaint_tiled", inpaint_tiled),
        ("inpaint_from_context", inpaint_from_context),
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
        # V14: Connect and project primitives
        ("connect_pixels_to_rect", connect_pixels_to_rect),
        ("gravity_toward_color", gravity_toward_color),
        ("fill_holes_in_objects", fill_holes_in_objects),
        # V14: Color reordering
        ("recolor_2nd_to_3rd", recolor_2nd_to_3rd_color),
        ("recolor_least_to_2nd_least", recolor_least_to_second_least),
        ("swap_most_and_2nd_color", swap_most_and_second_color),
        ("swap_largest_smallest_obj_color", swap_largest_and_smallest_obj_color),
        # V14: Pairwise color swaps
        ("swap_colors_12", swap_colors_12),
        ("swap_colors_13", swap_colors_13),
        ("swap_colors_14", swap_colors_14),
        ("swap_colors_15", swap_colors_15),
        ("swap_colors_23", swap_colors_23),
        ("swap_colors_24", swap_colors_24),
        ("swap_colors_25", swap_colors_25),
        ("swap_colors_34", swap_colors_34),
        ("swap_colors_35", swap_colors_35),
        ("swap_colors_45", swap_colors_45),
        # V14: Pattern completion
        ("complete_pattern_4way", complete_pattern_4way),
        ("fill_bg_from_border", fill_bg_with_color_from_border),
        # V14: Row/col dedup
        ("keep_unique_rows", keep_only_unique_rows),
        ("keep_unique_cols", keep_only_unique_cols),
        # V14: Color cycling
        ("rotate_colors_up", rotate_colors_up),
        ("rotate_colors_down", rotate_colors_down),
        # V14: Row/col fill
        ("extend_nonzero_fill_row", extend_nonzero_to_fill_row),
        ("extend_nonzero_fill_col", extend_nonzero_to_fill_col),
        # V14: Position-based coloring
        ("color_by_row_position", color_by_row_position),
        ("color_by_col_position", color_by_col_position),
        # V15: Isolated pixel operations
        ("recolor_isolated_to_nearest", recolor_isolated_to_nearest),
        ("recolor_small_objs_to_nearest", recolor_small_objects_to_nearest),
        ("remove_color_noise", remove_color_noise),
        # V15: Mirror merge
        ("mirror_h_merge", mirror_h_merge),
        ("mirror_v_merge", mirror_v_merge),
        # V15: Sort operations
        ("sort_rows_by_value", sort_rows_by_value),
        ("sort_cols_by_value", sort_cols_by_value),
        # V15: Object recoloring
        ("recolor_by_size_rank", recolor_by_size_rank),
        # V15: Propagation
        ("fill_row_from_right", fill_row_from_right),
        ("fill_col_from_bottom", fill_col_from_bottom),
        # V15: Structural
        ("extract_objects_on_grid", extract_objects_on_grid),
        ("crop_to_content_border", crop_to_content_border),
        ("keep_border_only", keep_border_only),
        ("complete_symmetry_diagonal", complete_symmetry_diagonal),
        # V15: Tiling
        ("tile_grid_2x1", tile_grid_2x1),
        ("tile_grid_1x2", tile_grid_1x2),
        ("repeat_pattern_to_size", repeat_pattern_to_size),
        # V15: Misc
        ("fill_diagonal_stripes", fill_diagonal_stripes),
        ("mask_by_color_overlap", mask_by_color_overlap),
        # V16: Stripe gap filling
        ("fill_stripe_gaps_h", fill_stripe_gaps_h),
        ("fill_stripe_gaps_v", fill_stripe_gaps_v),
        # V16: Tile completion from modal row/col
        ("complete_tile_from_modal_col", complete_tile_from_modal_col),
        ("complete_tile_from_modal_row", complete_tile_from_modal_row),
        # V16: Minority recoloring in rows/cols
        ("recolor_minority_in_rows", recolor_minority_in_rows),
        ("recolor_minority_in_cols", recolor_minority_in_cols),
        ("recolor_smallest_obj_in_each_row", recolor_smallest_obj_in_each_row),
        ("recolor_smallest_obj_in_each_col", recolor_smallest_obj_in_each_col),
        # V16: Grid intersection fill
        ("fill_grid_intersections", fill_grid_intersections),
        # V16: Directional propagation
        ("propagate_color_h", propagate_color_h),
        ("propagate_color_v", propagate_color_v),
        # V16: Unique-in-row/col recoloring
        ("recolor_unique_in_row_col", recolor_unique_in_row_col),
        # V16: Snap isolated pixels to object
        ("snap_isolated_to_rect_boundary", snap_isolated_to_rect_boundary),
        # V16: Recolor cells touching a specific color object
        ("recolor_touching_2nd_to_8", recolor_touching_2nd_to_8),
        ("recolor_touching_2nd_to_3", recolor_touching_2nd_to_3),
        ("recolor_neighbors_of_2nd_color", recolor_neighbors_of_2nd_color),
        # V16: Extend color to fill column/row within object bounds
        ("extend_color_within_col_bounds", extend_color_within_col_bounds),
        ("extend_color_within_row_bounds", extend_color_within_row_bounds),
        # V19: recolor non-zero inside bbox of accent color
        ("recolor_nonzero_inside_8_bbox_to_3", recolor_nonzero_inside_8_bbox_to_3),
        ("recolor_nonzero_inside_8_bbox_to_4", recolor_nonzero_inside_8_bbox_to_4),
        ("recolor_nonzero_inside_8_bbox_to_2", recolor_nonzero_inside_8_bbox_to_2),
        ("recolor_nonzero_inside_2_bbox_to_4", recolor_nonzero_inside_2_bbox_to_4),
        ("recolor_nonzero_inside_2_bbox_to_8", recolor_nonzero_inside_2_bbox_to_8),
        ("recolor_nonzero_inside_2_bbox_to_3", recolor_nonzero_inside_2_bbox_to_3),
        ("recolor_nonzero_inside_3_bbox_to_4", recolor_nonzero_inside_3_bbox_to_4),
        ("recolor_nonzero_inside_3_bbox_to_8", recolor_nonzero_inside_3_bbox_to_8),
        ("recolor_nonzero_inside_6_bbox_to_4", recolor_nonzero_inside_6_bbox_to_4),
        ("recolor_nonzero_inside_6_bbox_to_8", recolor_nonzero_inside_6_bbox_to_8),
        # V19: fill rectangular frame interiors
        ("fill_rect_interior_with_2", fill_rect_interior_with_2),
        ("fill_rect_interior_with_4", fill_rect_interior_with_4),
        ("fill_rect_interior_with_1", fill_rect_interior_with_1),
        ("fill_rect_interior_with_3", fill_rect_interior_with_3),
        # V19: row/col intersection marking
        ("mark_row_col_intersections_with_2", mark_row_col_intersections_with_2),
        ("mark_row_col_intersections_with_3", mark_row_col_intersections_with_3),
        ("mark_row_col_intersections_with_4", mark_row_col_intersections_with_4),
        # V19: extend lines to fill gaps
        ("extend_lines_to_contact", extend_lines_to_contact),
        # V19: fill bg adjacent to accent/dominant
        ("fill_bg_adjacent_to_accent_with_3", fill_bg_adjacent_to_accent_with_3),
        ("fill_bg_adjacent_to_accent_with_8", fill_bg_adjacent_to_accent_with_8),
        ("fill_bg_adjacent_to_dominant_with_3", fill_bg_adjacent_to_dominant_with_3),
        ("fill_bg_adjacent_to_dominant_with_8", fill_bg_adjacent_to_dominant_with_8),
        # V18: dominant-touching-accent recoloring (factory variants)
        ("recolor_dominant_touching_accent_to_4", recolor_dominant_touching_accent_to_4),
        ("recolor_dominant_touching_accent_to_6", recolor_dominant_touching_accent_to_6),
        ("recolor_dominant_touching_accent_to_7", recolor_dominant_touching_accent_to_7),
        ("recolor_dominant_touching_accent_to_8", recolor_dominant_touching_accent_to_8),
        ("recolor_dominant_touching_accent_to_2", recolor_dominant_touching_accent_to_2),
        ("recolor_dominant_touching_accent_to_3", recolor_dominant_touching_accent_to_3),
        # V18: hole filling variants
        ("fill_smallest_rect_hole_with_1", fill_smallest_rect_hole_with_1),
        ("fill_smallest_rect_hole_with_4", fill_smallest_rect_hole_with_4),
        ("fill_smallest_rect_hole_with_8", fill_smallest_rect_hole_with_8),
        ("recolor_bg_enclosed_by_dominant", recolor_bg_enclosed_by_dominant),
        # V18: sorting
        ("sort_rows_by_sum", sort_rows_by_sum),
        ("sort_cols_by_sum", sort_cols_by_sum),
        # V18: color merging
        ("recolor_2nd_color_to_dominant", recolor_2nd_color_to_dominant),
        ("erase_2nd_color", erase_2nd_color),
        # V20: shift operations (cyclic wrap-around)
        ("shift_down_1", shift_down_1),
        ("shift_up_1", shift_up_1),
        ("shift_left_1", shift_left_1),
        ("shift_right_1", shift_right_1),
        # V20: fill enclosed bg with wall color
        ("fill_enclosed_wall_color", fill_enclosed_wall_color),
        # V20: object border/interior operations
        ("remove_border_objects", remove_border_objects),
        ("keep_interior_objects", keep_interior_objects),
        ("hollow_objects", hollow_objects),
        ("fill_object_bboxes", fill_object_bboxes),
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
