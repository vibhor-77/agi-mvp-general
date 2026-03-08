"""Object Decomposition Solver for ARC-AGI

Implements the perceive → decompose → transform-per-object → reassemble
architecture. For tasks where the solution applies the same transform to
each discrete object independently.

Pipeline:
  1. Perceive: find_foreground_shapes(input) → list of object subgrids
  2. For each candidate transform in the toolkit:
     a. Apply transform to each object's subgrid
     b. Place transformed subgrids back onto background canvas
     c. Score against expected output
  3. Return the best consistent transform (if pixel-perfect on all examples)

This module is deliberately simple — it reuses existing primitives as
per-object transforms rather than inventing new ones.
"""
from __future__ import annotations
from typing import Optional, Callable
from collections import Counter

import numpy as np

from .concepts import Grid, Concept, Program, Toolkit
from .objects import find_foreground_shapes, place_subgrid
from .scorer import TaskCache


def _get_background_color(grid: Grid) -> int:
    """Determine background color (most frequent value in the grid)."""
    counts: dict[int, int] = Counter()
    for row in grid:
        for val in row:
            counts[val] += 1
    return max(counts, key=lambda k: counts[k])


def _make_background_canvas(grid: Grid, bg_color: int) -> Grid:
    """Create a canvas filled with the background color, same dims as grid."""
    h, w = len(grid), len(grid[0]) if grid else 0
    return [[bg_color] * w for _ in range(h)]


def _apply_transform_per_object(
    grid: Grid,
    transform: Callable[[Grid], Grid],
    bg_color: int = 0,
) -> Optional[Grid]:
    """Apply a transform to each object's subgrid and reassemble.

    Steps:
      1. Extract foreground shapes from the grid.
      2. Create a blank canvas with the background color.
      3. For each shape: apply transform to its subgrid, place result
         back at the same position.

    Args:
        grid: Input grid.
        transform: Grid→Grid function to apply to each object subgrid.
        bg_color: Background color for the canvas.

    Returns:
        Reassembled grid, or None if transform fails on any object.
    """
    shapes = find_foreground_shapes(grid)
    if not shapes:
        return None

    canvas = _make_background_canvas(grid, bg_color)

    for shape in shapes:
        subgrid = shape["subgrid"]
        position = shape["position"]

        try:
            transformed = transform(subgrid)
            if transformed is None:
                return None
        except Exception:
            return None

        # Place the transformed subgrid back at the original position.
        # Use bg_color as transparent so we don't overwrite other objects
        # or the background with the object's internal background cells.
        canvas = place_subgrid(canvas, transformed, position,
                               transparent_color=bg_color)

    return canvas


def solve_by_object_decomposition(
    task: dict,
    toolkit: Toolkit,
    cache: TaskCache,
) -> Optional[Program]:
    """Try to solve a task by applying the same transform to each object.

    Iterates over all operator concepts in the toolkit and tests whether
    applying any single one to each object's subgrid produces the correct
    output across all training examples.

    Args:
        task: ARC task dict with 'train' key.
        toolkit: The solver's toolkit containing all available concepts.
        cache: Pre-computed scoring cache for this task.

    Returns:
        A Program wrapping the per-object transform if pixel-perfect on
        all training examples, or None if no consistent transform found.
    """
    train = task.get("train", [])
    if not train:
        return None

    # Determine background color from first example's input
    bg_color = _get_background_color(train[0]["input"])

    # Only try same-dims tasks (input and output have same shape)
    for ex in train:
        inp, out = ex["input"], ex["output"]
        if len(inp) != len(out) or len(inp[0]) != len(out[0]):
            return None

    # Collect all operator concepts to try as per-object transforms
    operators = [
        c for c in toolkit.concepts.values()
        if c.kind == "operator"
    ]

    best_program = None
    best_score = 0.0

    for concept in operators:
        transform = concept.implementation

        # Build the composite transform: apply transform per-object
        def make_per_object_fn(t=transform, bg=bg_color):
            def fn(grid: Grid) -> Grid:
                result = _apply_transform_per_object(grid, t, bg)
                return result if result is not None else grid
            return fn

        per_object_fn = make_per_object_fn()

        # Test on all training examples
        all_match = True
        for ex in train:
            inp = ex["input"]
            try:
                result = per_object_fn(inp)
            except Exception:
                all_match = False
                break

            # Quick check: does result match expected output?
            expected = ex["output"]
            if result != expected:
                all_match = False
                break

        if all_match:
            # Found a consistent per-object transform!
            program = Program(
                steps=[Concept(
                    kind="composed",
                    name=f"per_object({concept.name})",
                    implementation=per_object_fn,
                )],
            )
            program.fitness = 1.0
            # Score it properly through the cache to get structural_similarity
            score = cache.score_program(program)
            if score > best_score:
                best_score = score
                best_program = program

            # If pixel-perfect, no need to keep searching
            if cache.is_pixel_perfect(program):
                return program

    return best_program
