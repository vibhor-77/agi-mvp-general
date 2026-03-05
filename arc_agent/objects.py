"""
Object-Level Primitives for ARC-AGI

Many ARC tasks require reasoning about discrete objects within grids,
not just whole-grid transformations. This module provides:

1. Connected component extraction (finding objects)
2. Object property detection (size, color, bounding box, center)
3. Object-level transformations (extract, recolor, mirror)

These become new Concepts in the Toolkit, composable with existing
primitives via Pillar 3 (Abstraction & Composability).
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
from .concepts import Grid, Concept, Toolkit


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

    Uses 4-connectivity flood fill. Each contiguous region of
    same-colored non-zero cells becomes a GridObject.

    Args:
        grid: Input grid (list of lists of ints).

    Returns:
        List of GridObject instances, one per connected component.
    """
    if not grid or not grid[0]:
        return []

    height = len(grid)
    width = len(grid[0])
    visited: set[tuple[int, int]] = set()
    objects: list[GridObject] = []

    for r in range(height):
        for c in range(width):
            if grid[r][c] != 0 and (r, c) not in visited:
                # Flood fill to find all connected cells of this color
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
                    # 4-connectivity: up, down, left, right
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
