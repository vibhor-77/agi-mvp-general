"""
Object-Level Primitives for ARC-AGI

Many ARC tasks require reasoning about discrete objects within grids,
not just whole-grid transformations. This module provides:

1. Connected component extraction (finding objects)
2. Object property detection (size, color, bounding box, center)
3. Object-level transformations (extract, recolor, mirror)

These become new Concepts in the Toolkit, composable with existing
primitives via Pillar 3 (Abstraction & Composability).

Performance: find_objects uses Numba JIT when available (install with
`pip install numba`). The JIT version runs the flood-fill loop natively
with no Python overhead — ~5-20x faster than pure Python on large grids.
On first import, Numba compiles the kernel (a few seconds). Subsequent
calls use the cached compiled code. If Numba is not installed, the
pure-Python implementation is used transparently.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import numpy as np
from .concepts import Grid, Concept, Toolkit

# ---------------------------------------------------------------------------
# Numba JIT — optional, graceful degradation if not installed
# ---------------------------------------------------------------------------

try:
    import numba as _nb

    @_nb.njit(cache=True)
    def _flood_fill_labels(grid: np.ndarray) -> tuple:
        """Label connected components with 4-connectivity via flood fill.

        JIT-compiled with Numba: runs at native speed with no Python overhead.
        ARC grids are at most 30×30 so the stack buffer (900 cells) is safe.

        Returns:
            labels — int32 array same shape as grid, each component gets a
                     unique positive integer label (0 = background).
            colors — int32 array of length n_objects: color of each component.
            n      — number of components found.
        """
        h, w = grid.shape
        labels = np.zeros((h, w), dtype=np.int32)
        colors = np.zeros(h * w, dtype=np.int32)   # max possible objects
        n = 0

        # Stack arrays — fixed max size (h*w is the absolute upper bound)
        max_stack = h * w
        stack_r = np.empty(max_stack, dtype=np.int32)
        stack_c = np.empty(max_stack, dtype=np.int32)

        for r in range(h):
            for c in range(w):
                if grid[r, c] != 0 and labels[r, c] == 0:
                    n += 1
                    label = n
                    color = grid[r, c]
                    colors[label - 1] = color

                    sp = 0
                    stack_r[sp] = r
                    stack_c[sp] = c
                    sp += 1

                    while sp > 0:
                        sp -= 1
                        cr = stack_r[sp]
                        cc = stack_c[sp]
                        if cr < 0 or cr >= h or cc < 0 or cc >= w:
                            continue
                        if labels[cr, cc] != 0 or grid[cr, cc] != color:
                            continue
                        labels[cr, cc] = label
                        if sp + 4 < max_stack:
                            stack_r[sp] = cr - 1; stack_c[sp] = cc; sp += 1
                            stack_r[sp] = cr + 1; stack_c[sp] = cc; sp += 1
                            stack_r[sp] = cr; stack_c[sp] = cc - 1; sp += 1
                            stack_r[sp] = cr; stack_c[sp] = cc + 1; sp += 1

        return labels, colors, n

    def _find_objects_numba(grid: Grid) -> list:
        """find_objects implementation using Numba-JIT flood fill."""
        arr = np.array(grid, dtype=np.int32)
        labels, colors, n = _flood_fill_labels(arr)
        objects = []
        for label in range(1, n + 1):
            rr, cc = np.where(labels == label)
            pixels = set(zip(rr.tolist(), cc.tolist()))
            objects.append(GridObject(color=int(colors[label - 1]), pixels=pixels))
        return objects

    _USE_NUMBA = True

except ImportError:
    _USE_NUMBA = False


@dataclass
class GridObject:
    """A connected component (object) within a grid.

    Represents a contiguous region of same-colored non-zero cells,
    along with its properties. Uses 4-connectivity (not diagonal).
    """
    color: int
    pixels: set[tuple[int, int]]  # set of (row, col) coordinates

    @property
    def size(self) -> int:
        """Number of cells in this object."""
        return len(self.pixels)

    @property
    def bbox(self) -> tuple[int, int, int, int]:
        """Bounding box as (min_row, min_col, max_row, max_col)."""
        rows = [r for r, _ in self.pixels]
        cols = [c for _, c in self.pixels]
        return (min(rows), min(cols), max(rows), max(cols))

    @property
    def center(self) -> tuple[int, int]:
        """Center of mass (integer-rounded)."""
        rows = [r for r, _ in self.pixels]
        cols = [c for _, c in self.pixels]
        return (sum(rows) // len(rows), sum(cols) // len(cols))

    def to_grid(self) -> Grid:
        """Extract this object as a minimal sub-grid (cropped to bbox).

        Background cells within the bounding box that aren't part of
        the object are set to 0.
        """
        min_r, min_c, max_r, max_c = self.bbox
        height = max_r - min_r + 1
        width = max_c - min_c + 1
        result = [[0] * width for _ in range(height)]
        for r, c in self.pixels:
            result[r - min_r][c - min_c] = self.color
        return result


def find_objects(grid: Grid) -> list[GridObject]:
    """Find all connected components (objects) in a grid.

    Uses 4-connectivity flood fill. Each contiguous region of same-colored
    non-zero cells becomes a GridObject.

    Dispatches to the Numba-JIT implementation when Numba is available
    (install with `pip install numba`). Falls back to pure Python otherwise.

    Args:
        grid: Input grid (list of lists of ints, values 0-9).

    Returns:
        List of GridObject, one per connected component, in scan order.
    """
    if not grid or not grid[0]:
        return []

    if _USE_NUMBA:
        return _find_objects_numba(grid)

    # Pure-Python fallback (used when Numba is not installed).
    # Python's built-in set is fastest for small integer-tuple lookups.
    # On ARC grids (≤ 30×30) this is ~150-200µs; Numba JIT is ~5-10µs.
    height = len(grid)
    width  = len(grid[0])
    visited: set[tuple[int, int]] = set()
    objects: list[GridObject] = []

    for r in range(height):
        for c in range(width):
            if grid[r][c] != 0 and (r, c) not in visited:
                color = grid[r][c]
                pixels: set[tuple[int, int]] = set()
                stack = [(r, c)]

                while stack:
                    cr, cc = stack.pop()
                    if (cr, cc) in visited:
                        continue
                    if cr < 0 or cr >= height or cc < 0 or cc >= width:
                        continue
                    if grid[cr][cc] != color:
                        continue
                    visited.add((cr, cc))
                    pixels.add((cr, cc))
                    stack.extend([
                        (cr - 1, cc), (cr + 1, cc),
                        (cr, cc - 1), (cr, cc + 1),
                    ])

                objects.append(GridObject(color=color, pixels=pixels))

    return objects


def count_objects(grid: Grid) -> int:
    """Count the number of distinct objects in a grid."""
    return len(find_objects(grid))


def extract_largest_object(grid: Grid) -> Grid:
    """Extract the largest object as a cropped sub-grid.

    Grid → Grid transform suitable for use as a Concept.
    """
    objects = find_objects(grid)
    if not objects:
        return [[0]]
    largest = max(objects, key=lambda o: o.size)
    return largest.to_grid()


def extract_smallest_object(grid: Grid) -> Grid:
    """Extract the smallest object as a cropped sub-grid."""
    objects = find_objects(grid)
    if not objects:
        return [[0]]
    smallest = min(objects, key=lambda o: o.size)
    return smallest.to_grid()


def remove_color(grid: Grid, color: int) -> Grid:
    """Remove all cells of a given color (set to 0)."""
    return [
        [0 if cell == color else cell for cell in row]
        for row in grid
    ]


def isolate_color(grid: Grid, color: int) -> Grid:
    """Keep only cells of a given color, zero everything else."""
    return [
        [cell if cell == color else 0 for cell in row]
        for row in grid
    ]


def recolor_largest_object(grid: Grid, new_color: int) -> Grid:
    """Recolor the largest object to a new color."""
    objects = find_objects(grid)
    if not objects:
        return [row[:] for row in grid]
    largest = max(objects, key=lambda o: o.size)
    result = [row[:] for row in grid]
    for r, c in largest.pixels:
        result[r][c] = new_color
    return result


def recolor_smallest_object(grid: Grid, new_color: int) -> Grid:
    """Recolor the smallest object to a new color."""
    objects = find_objects(grid)
    if not objects:
        return [row[:] for row in grid]
    smallest = min(objects, key=lambda o: o.size)
    result = [row[:] for row in grid]
    for r, c in smallest.pixels:
        result[r][c] = new_color
    return result


def recolor_all_to_most_common(grid: Grid) -> Grid:
    """Recolor all objects to the most frequent object color.

    Finds the color that appears in the most objects (not most pixels),
    then recolors every non-zero cell to that color.
    """
    objects = find_objects(grid)
    if not objects:
        return [row[:] for row in grid]
    # Count how many objects have each color
    color_counts: dict[int, int] = {}
    for obj in objects:
        color_counts[obj.color] = color_counts.get(obj.color, 0) + 1
    dominant = max(color_counts, key=lambda k: color_counts[k])
    return [
        [dominant if cell != 0 else 0 for cell in row]
        for row in grid
    ]


def mirror_objects_horizontal(grid: Grid) -> Grid:
    """Mirror each object horizontally within its bounding box.

    Finds each object, mirrors it left-right within its bbox,
    and places it back in the grid.
    """
    objects = find_objects(grid)
    if not objects:
        return [row[:] for row in grid]

    result = [[0] * len(row) for row in grid]

    # First, copy background (cells not part of any object)
    all_object_pixels: set[tuple[int, int]] = set()
    for obj in objects:
        all_object_pixels.update(obj.pixels)
    for r in range(len(grid)):
        for c in range(len(grid[0])):
            if (r, c) not in all_object_pixels:
                result[r][c] = grid[r][c]

    # Mirror each object within its bounding box
    for obj in objects:
        min_r, min_c, max_r, max_c = obj.bbox
        for r, c in obj.pixels:
            # Mirror column within bbox
            mirrored_c = max_c - (c - min_c)
            result[r][mirrored_c] = obj.color

    return result


# ============================================================
# Factory functions for creating Concept-wrapped object ops
# ============================================================

def _make_remove_color(color: int):
    """Factory: create a remove_color function for a specific color."""
    def _remove(grid: Grid) -> Grid:
        return remove_color(grid, color)
    return _remove


def _make_isolate_color(color: int):
    """Factory: create an isolate_color function for a specific color."""
    def _isolate(grid: Grid) -> Grid:
        return isolate_color(grid, color)
    return _isolate


def _make_recolor_largest(color: int):
    """Factory: create a recolor_largest_object function for a specific color."""
    def _recolor(grid: Grid) -> Grid:
        return recolor_largest_object(grid, color)
    return _recolor


def _make_recolor_smallest(color: int):
    """Factory: create a recolor_smallest_object function for a specific color."""
    def _recolor(grid: Grid) -> Grid:
        return recolor_smallest_object(grid, color)
    return _recolor


def add_object_concepts(toolkit: Toolkit) -> None:
    """Add object-level concepts to an existing toolkit.

    This extends the primitive set with object-aware operations,
    expanding the space of programs the synthesizer can explore.
    """
    # Object extraction
    toolkit.add_concept(Concept(
        kind="operator", name="extract_largest",
        implementation=extract_largest_object,
    ))
    toolkit.add_concept(Concept(
        kind="operator", name="extract_smallest",
        implementation=extract_smallest_object,
    ))
    toolkit.add_concept(Concept(
        kind="operator", name="mirror_objects_h",
        implementation=mirror_objects_horizontal,
    ))

    toolkit.add_concept(Concept(
        kind="operator", name="recolor_all_to_most_common_obj",
        implementation=recolor_all_to_most_common,
    ))

    # Color-specific object operations
    for color in range(1, 10):
        toolkit.add_concept(Concept(
            kind="operator", name=f"remove_color_{color}",
            implementation=_make_remove_color(color),
        ))
        toolkit.add_concept(Concept(
            kind="operator", name=f"isolate_color_{color}",
            implementation=_make_isolate_color(color),
        ))
        toolkit.add_concept(Concept(
            kind="operator", name=f"recolor_largest_to_{color}",
            implementation=_make_recolor_largest(color),
        ))
        toolkit.add_concept(Concept(
            kind="operator", name=f"recolor_smallest_to_{color}",
            implementation=_make_recolor_smallest(color),
        ))
