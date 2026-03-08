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

    # Strategy 2: Conditional per-object recolor by property.
    # Many ARC tasks recolor objects based on their size, position, or shape.
    # Learn a property→color mapping from training examples.
    conditional = _try_conditional_recolor(task, cache)
    if conditional is not None:
        cond_score = cache.score_program(conditional)
        if cond_score > best_score:
            best_score = cond_score
            best_program = conditional

    return best_program


# ============================================================
# Conditional per-object recolor by property
# ============================================================

def _try_conditional_recolor(
    task: dict,
    cache: TaskCache,
) -> Optional[Program]:
    """Learn a per-object recolor rule based on object properties.

    Matches input objects to output objects by position overlap, then
    checks if the color mapping is determined by a single property:
      - size (number of pixels)
      - shape (normalized pixel set)
      - is_singleton (size == 1 vs size > 1)

    Returns a Program if a consistent, generalizable rule is found.
    """
    train = task.get("train", [])
    if not train:
        return None

    # Only same-dims tasks
    for ex in train:
        inp, out = ex["input"], ex["output"]
        if len(inp) != len(out) or len(inp[0]) != len(out[0]):
            return None

    # Try each property-based strategy
    for strategy_name, strategy_fn in [
        ("by_size", _learn_recolor_by_size),
        ("by_singleton", _learn_recolor_by_singleton),
    ]:
        rule = strategy_fn(train)
        if rule is None:
            continue

        # Build the transform function
        transform_fn = _make_conditional_recolor_fn(rule, strategy_name)
        program = Program(
            steps=[Concept(
                kind="composed",
                name=f"per_object_recolor({strategy_name})",
                implementation=transform_fn,
            )],
        )

        # Score on training
        score = cache.score_program(program)
        program.fitness = score

        if cache.is_pixel_perfect(program):
            return program

    return None


def _match_objects_by_position(inp: Grid, out: Grid) -> list[tuple[dict, dict]] | None:
    """Match input objects to output objects by position overlap.

    Returns list of (input_shape, output_shape) pairs, or None if
    matching fails (different number of objects, ambiguous matches).
    """
    from .objects import find_foreground_shapes

    shapes_in = find_foreground_shapes(inp)
    shapes_out = find_foreground_shapes(out)

    if len(shapes_in) == 0:
        return None

    matches = []
    used_out = set()

    for si in shapes_in:
        # Build pixel set for input shape
        si_pixels = set()
        for r in range(len(si["subgrid"])):
            for c in range(len(si["subgrid"][0])):
                if si["subgrid"][r][c] != 0:
                    si_pixels.add((si["position"][0] + r, si["position"][1] + c))

        # Find best matching output shape by pixel overlap
        best_idx = -1
        best_overlap = 0
        for j, so in enumerate(shapes_out):
            if j in used_out:
                continue
            so_pixels = set()
            for r in range(len(so["subgrid"])):
                for c in range(len(so["subgrid"][0])):
                    if so["subgrid"][r][c] != 0:
                        so_pixels.add((so["position"][0] + r, so["position"][1] + c))
            overlap = len(si_pixels & so_pixels)
            if overlap > best_overlap:
                best_overlap = overlap
                best_idx = j

        if best_idx >= 0:
            used_out.add(best_idx)
            matches.append((si, shapes_out[best_idx]))
        else:
            # No matching output — object was removed or entirely new pixels
            # Try matching by position proximity
            best_idx = -1
            best_dist = float("inf")
            for j, so in enumerate(shapes_out):
                if j in used_out:
                    continue
                dist = abs(si["position"][0] - so["position"][0]) + \
                       abs(si["position"][1] - so["position"][1])
                if dist < best_dist:
                    best_dist = dist
                    best_idx = j
            if best_idx >= 0:
                used_out.add(best_idx)
                matches.append((si, shapes_out[best_idx]))
            else:
                return None

    return matches


def _learn_recolor_by_size(train: list[dict]) -> dict | None:
    """Learn a size→color mapping from training examples.

    If all objects of the same size consistently map to the same output color
    across all training examples, returns {size: output_color}.
    """
    size_to_color: dict[int, int] = {}

    for ex in train:
        matches = _match_objects_by_position(ex["input"], ex["output"])
        if matches is None:
            return None

        for si, so in matches:
            size = si["size"]
            out_color = so["color"]
            if size in size_to_color:
                if size_to_color[size] != out_color:
                    return None  # inconsistent
            size_to_color[size] = out_color

    # Validate: the mapping must actually change SOMETHING
    # (not all objects map to the same color they already had)
    has_change = False
    for ex in train:
        matches = _match_objects_by_position(ex["input"], ex["output"])
        if matches is None:
            return None
        for si, so in matches:
            if si["color"] != so["color"]:
                has_change = True
                break
        if has_change:
            break

    if not has_change:
        return None

    return size_to_color


def _learn_recolor_by_singleton(train: list[dict]) -> dict | None:
    """Learn a singleton-vs-multi recolor rule.

    If objects with size==1 always map to one color and objects with size>1
    always map to another color (consistently across examples), return the rule.
    Returns {True: color_for_multi, False: color_for_singleton} or None.
    """
    singleton_colors: set[int] = set()
    multi_colors: set[int] = set()

    for ex in train:
        matches = _match_objects_by_position(ex["input"], ex["output"])
        if matches is None:
            return None

        for si, so in matches:
            if si["size"] == 1:
                singleton_colors.add(so["color"])
            else:
                multi_colors.add(so["color"])

    # Each class must map to exactly one color
    if len(singleton_colors) != 1 or len(multi_colors) != 1:
        return None

    singleton_color = next(iter(singleton_colors))
    multi_color = next(iter(multi_colors))

    # Must actually change something
    if singleton_color == multi_color:
        return None

    return {True: multi_color, False: singleton_color}


def _make_conditional_recolor_fn(
    rule: dict,
    strategy: str,
) -> Callable[[Grid], Grid]:
    """Build a Grid→Grid function from a learned conditional recolor rule."""
    from .objects import find_foreground_shapes, place_subgrid

    def transform(grid: Grid) -> Grid:
        shapes = find_foreground_shapes(grid)
        if not shapes:
            return grid

        # Start with a copy of the grid (preserve background)
        result = [row[:] for row in grid]

        for shape in shapes:
            if strategy == "by_size":
                size = shape["size"]
                if size in rule:
                    new_color = rule[size]
                else:
                    # Unknown size — keep original color
                    new_color = shape["color"]
            elif strategy == "by_singleton":
                is_multi = shape["size"] > 1
                new_color = rule[is_multi]
            else:
                new_color = shape["color"]

            # Recolor the object's pixels in-place
            pos = shape["position"]
            for r in range(len(shape["subgrid"])):
                for c in range(len(shape["subgrid"][0])):
                    if shape["subgrid"][r][c] != 0:
                        result[pos[0] + r][pos[1] + c] = new_color

        return result

    return transform
