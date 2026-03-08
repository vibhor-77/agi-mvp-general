"""Focused structural primitives for spatial/geometric patterns.

These primitives target specific visual patterns found in near-miss ARC tasks:
- extend_lines: Complete partial lines to grid boundaries
- fill_rooms_with_new_color: Fill enclosed rectangular regions with a new color
- mirror_pattern_across_axis: Complete symmetric patterns
- gravity_drop: Drop cells in a direction until hitting obstacles

Each is a simple Grid->Grid function that encodes a geometric concept.
"""
from __future__ import annotations
from typing import Literal
from arc_agent.concepts import Grid


def _deep_copy_grid(grid: Grid) -> Grid:
    """Create a deep copy of a grid."""
    return [row[:] for row in grid]


def _grid_dims(grid: Grid) -> tuple[int, int]:
    """Return (height, width) of grid."""
    if not grid:
        return (0, 0)
    return (len(grid), len(grid[0]))


def extend_lines(grid: Grid) -> Grid:
    """Find partial lines of non-background color and extend to grid boundary.

    Detects straight lines (horizontal or vertical) of a non-zero color and
    extends them in the directions they already extend until hitting another
    non-zero cell or reaching the edge. Lines must have at least 2 consecutive cells.

    Args:
        grid: Input grid

    Returns:
        Grid with extended lines
    """
    h, w = _grid_dims(grid)
    if h == 0:
        return grid

    result = _deep_copy_grid(grid)

    # Find and extend horizontal lines
    for r in range(h):
        c = 0
        while c < w:
            if result[r][c] != 0:
                color = result[r][c]
                start = c
                # Find extent of this run
                while c < w and result[r][c] == color:
                    c += 1
                end = c - 1

                length = end - start + 1
                if length >= 2:
                    # Extend left to boundary or obstacle
                    for left_c in range(start - 1, -1, -1):
                        if result[r][left_c] != 0:
                            break
                        result[r][left_c] = color

                    # Extend right to boundary or obstacle
                    for right_c in range(end + 1, w):
                        if result[r][right_c] != 0:
                            break
                        result[r][right_c] = color
            else:
                c += 1

    # Find and extend vertical lines
    for c in range(w):
        r = 0
        while r < h:
            if result[r][c] != 0:
                color = result[r][c]
                start = r
                # Find extent of this run
                while r < h and result[r][c] == color:
                    r += 1
                end = r - 1

                length = end - start + 1
                if length >= 2:
                    # Extend up to boundary or obstacle
                    for up_r in range(start - 1, -1, -1):
                        if result[up_r][c] != 0:
                            break
                        result[up_r][c] = color

                    # Extend down to boundary or obstacle
                    for down_r in range(end + 1, h):
                        if result[down_r][c] != 0:
                            break
                        result[down_r][c] = color
            else:
                r += 1

    return result


def fill_rooms_with_new_color(grid: Grid) -> Grid:
    """Fill enclosed rectangular regions with a new color.

    Detects rooms (connected regions of 0s) that are completely enclosed by
    non-zero walls, and fills them with a new color not present in the input.
    Tries to use color 4 first, then other integers 1-9.

    Args:
        grid: Input grid where walls are non-zero and rooms are 0

    Returns:
        Grid with rooms filled
    """
    h, w = _grid_dims(grid)
    if h == 0:
        return grid

    result = _deep_copy_grid(grid)

    # Find all colors in input
    colors_in_input = set()
    for row in grid:
        for cell in row:
            colors_in_input.add(cell)

    # Find a new color not in input, prefer 4 (typical new color in ARC)
    new_color = None
    if 4 not in colors_in_input:
        new_color = 4
    else:
        for c in range(1, 10):
            if c not in colors_in_input:
                new_color = c
                break

    if new_color is None:
        return result

    # Find all 0-regions connected to border (NOT enclosed)
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

    # Fill enclosed 0-regions with new_color
    for r in range(h):
        for c in range(w):
            if grid[r][c] == 0 and (r, c) not in border_connected:
                result[r][c] = new_color

    return result


def mirror_pattern_across_axis(grid: Grid) -> Grid:
    """Detect a partial pattern and complete it by mirroring across an axis.

    Analyzes the grid to detect if there's an asymmetric pattern that should be
    symmetric, then mirrors the existing pattern to complete it. Detects both
    horizontal and vertical symmetry axes.

    Args:
        grid: Input grid with partial pattern

    Returns:
        Grid with completed symmetric pattern
    """
    h, w = _grid_dims(grid)
    if h == 0 or w == 0:
        return grid

    result = _deep_copy_grid(grid)

    # Try horizontal symmetry (mirror across vertical center axis)
    # Count non-zero cells in left vs right half
    left_nonzero = 0
    right_nonzero = 0

    for r in range(h):
        for c in range(w // 2):
            if result[r][c] != 0:
                left_nonzero += 1
        for c in range((w + 1) // 2, w):
            if result[r][c] != 0:
                right_nonzero += 1

    # If left has pattern and right is mostly empty, mirror left to right
    if left_nonzero > 0 and right_nonzero < left_nonzero // 2:
        for r in range(h):
            for c in range(w // 2):
                mirror_c = w - 1 - c
                if result[r][c] != 0 and result[r][mirror_c] == 0:
                    result[r][mirror_c] = result[r][c]
        return result

    # If right has pattern and left is mostly empty, mirror right to left
    if right_nonzero > 0 and left_nonzero < right_nonzero // 2:
        for r in range(h):
            for c in range(w // 2):
                mirror_c = w - 1 - c
                if result[r][mirror_c] != 0 and result[r][c] == 0:
                    result[r][c] = result[r][mirror_c]
        return result

    # Try vertical symmetry (mirror across horizontal center axis)
    top_nonzero = 0
    bottom_nonzero = 0

    for r in range(h // 2):
        for c in range(w):
            if result[r][c] != 0:
                top_nonzero += 1
    for r in range((h + 1) // 2, h):
        for c in range(w):
            if result[r][c] != 0:
                bottom_nonzero += 1

    # If top has pattern and bottom is mostly empty, mirror top to bottom
    if top_nonzero > 0 and bottom_nonzero < top_nonzero // 2:
        for r in range(h // 2):
            mirror_r = h - 1 - r
            for c in range(w):
                if result[r][c] != 0 and result[mirror_r][c] == 0:
                    result[mirror_r][c] = result[r][c]
        return result

    # If bottom has pattern and top is mostly empty, mirror bottom to top
    if bottom_nonzero > 0 and top_nonzero < bottom_nonzero // 2:
        for r in range(h // 2):
            mirror_r = h - 1 - r
            for c in range(w):
                if result[mirror_r][c] != 0 and result[r][c] == 0:
                    result[r][c] = result[mirror_r][c]
        return result

    return result


def gravity_drop(
    grid: Grid,
    direction: Literal['down', 'up', 'left', 'right'] = 'down',
    stop_color: int = 0,
) -> Grid:
    """Drop non-zero cells in a direction until hitting stop_color or boundary.

    Moves all non-zero, non-stop_color cells in the specified direction.
    Cells move until they hit a stop_color cell or the grid boundary.
    Stop_color cells remain fixed.

    Args:
        grid: Input grid
        direction: Direction to move ('down', 'up', 'left', 'right')
        stop_color: Color that acts as a barrier (cells don't move through it)

    Returns:
        Grid with cells moved in the specified direction
    """
    h, w = _grid_dims(grid)
    if h == 0:
        return grid

    result = _deep_copy_grid(grid)

    if direction == 'down':
        for c in range(w):
            # Process each cell from bottom to top, moving it down until blocked
            for r in range(h - 2, -1, -1):
                if result[r][c] != 0 and result[r][c] != stop_color:
                    cell = result[r][c]
                    result[r][c] = 0
                    # Move down until hitting obstacle
                    new_r = r
                    while new_r + 1 < h and result[new_r + 1][c] == 0:
                        new_r += 1
                    result[new_r][c] = cell

    elif direction == 'up':
        for c in range(w):
            # Process each cell from top to bottom, moving it up until blocked
            for r in range(1, h):
                if result[r][c] != 0 and result[r][c] != stop_color:
                    cell = result[r][c]
                    result[r][c] = 0
                    # Move up until hitting obstacle
                    new_r = r
                    while new_r - 1 >= 0 and result[new_r - 1][c] == 0:
                        new_r -= 1
                    result[new_r][c] = cell

    elif direction == 'left':
        for r in range(h):
            # Process each cell from right to left, moving it left until blocked
            for c in range(w - 1, -1, -1):
                if result[r][c] != 0 and result[r][c] != stop_color:
                    cell = result[r][c]
                    result[r][c] = 0
                    # Move left until hitting obstacle
                    new_c = c
                    while new_c - 1 >= 0 and result[r][new_c - 1] == 0:
                        new_c -= 1
                    result[r][new_c] = cell

    elif direction == 'right':
        for r in range(h):
            # Process each cell from left to right, moving it right until blocked
            for c in range(w):
                if result[r][c] != 0 and result[r][c] != stop_color:
                    cell = result[r][c]
                    result[r][c] = 0
                    # Move right until hitting obstacle
                    new_c = c
                    while new_c + 1 < w and result[r][new_c + 1] == 0:
                        new_c += 1
                    result[r][new_c] = cell

    return result


__all__ = [
    'extend_lines',
    'fill_rooms_with_new_color',
    'mirror_pattern_across_axis',
    'gravity_drop',
]
