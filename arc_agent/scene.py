"""
Object-Centric Scene Reasoning for ARC-AGI

Implements a perceive → compare → infer → apply pipeline:
  1. build_scene: segment grid into objects with properties
  2. diff_scenes: structured diff of input vs output scenes
  3. find_consistent_rules: find patterns consistent across all examples
  4. apply_rules: apply inferred rules to a new grid
  5. solve_with_object_rules: end-to-end pipeline for a task

This module sits alongside the existing primitive-search pipeline.
It targets same-dims tasks where objects interact in-place (68% of ARC tasks).
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Optional, Callable

from .concepts import Grid
from .objects import find_objects, GridObject


# ── Scene representation ──────────────────────────────────────────────────

@dataclass
class SceneObject:
    """An object within a scene, with identity and shape properties."""
    id: int
    color: int
    pixels: frozenset[tuple[int, int]]
    size: int
    bbox: tuple[int, int, int, int]  # (min_r, min_c, max_r, max_c)
    center: tuple[float, float]

    @property
    def shape_signature(self) -> frozenset[tuple[int, int]]:
        """Relative pixel positions normalized to bbox top-left.

        Two objects with the same shape_signature have the same shape,
        regardless of position or color.
        """
        min_r, min_c = self.bbox[0], self.bbox[1]
        return frozenset((r - min_r, c - min_c) for r, c in self.pixels)

    @property
    def compactness(self) -> float:
        """Ratio of actual pixels to bounding box area (0-1)."""
        min_r, min_c, max_r, max_c = self.bbox
        bbox_area = (max_r - min_r + 1) * (max_c - min_c + 1)
        return self.size / bbox_area if bbox_area > 0 else 1.0


@dataclass
class SceneGraph:
    """Structured representation of a grid as background + objects."""
    grid_shape: tuple[int, int]  # (height, width)
    bg_color: int
    objects: list[SceneObject]


def _make_scene_object(grid_obj: GridObject, obj_id: int) -> SceneObject:
    """Wrap a GridObject into a SceneObject with computed properties."""
    pixels = frozenset(grid_obj.pixels)
    rows = [r for r, _ in pixels]
    cols = [c for _, c in pixels]
    return SceneObject(
        id=obj_id,
        color=grid_obj.color,
        pixels=pixels,
        size=grid_obj.size,
        bbox=grid_obj.bbox,
        center=(sum(rows) / len(rows), sum(cols) / len(cols)),
    )


def build_scene(grid: Grid) -> SceneGraph:
    """Segment a grid into background + objects.

    Background is the most frequent color value. All connected components
    of non-background colors become SceneObjects.
    """
    if not grid or not grid[0]:
        return SceneGraph(grid_shape=(0, 0), bg_color=0, objects=[])

    h, w = len(grid), len(grid[0])

    # Background = most frequent color
    flat = [cell for row in grid for cell in row]
    bg_color = Counter(flat).most_common(1)[0][0]

    # Find objects: connected components of non-bg color
    # find_objects() uses color 0 as background. If bg != 0, we need to
    # remap the grid so that bg becomes 0 and original 0 becomes something else.
    if bg_color == 0:
        raw_objects = find_objects(grid)
    else:
        # Remap: bg_color → 0, 0 → bg_color (swap)
        remapped = []
        for row in grid:
            new_row = []
            for cell in row:
                if cell == bg_color:
                    new_row.append(0)
                elif cell == 0:
                    new_row.append(bg_color)
                else:
                    new_row.append(cell)
            remapped.append(new_row)
        raw_objects = find_objects(remapped)
        # Fix colors back: objects that had color bg_color in remapped
        # were originally color 0 in the real grid
        for obj in raw_objects:
            if obj.color == bg_color:
                obj.color = 0  # was originally 0 before swap

    scene_objects = [_make_scene_object(obj, i) for i, obj in enumerate(raw_objects)]
    return SceneGraph(grid_shape=(h, w), bg_color=bg_color, objects=scene_objects)


# ── Scene diffing ─────────────────────────────────────────────────────────

@dataclass
class ObjectDiff:
    """How one input object maps to one output object."""
    src: SceneObject
    dst: SceneObject
    new_color: Optional[int]  # None if color unchanged
    movement: tuple[int, int]  # (delta_row, delta_col) of center
    shape_preserved: bool
    size_delta: int  # dst.size - src.size


@dataclass
class SceneDiff:
    """Structured diff between an input scene and an output scene."""
    matched: list[ObjectDiff]  # Objects present in both
    removed: list[SceneObject]  # In input but not output
    added: list[SceneObject]  # In output but not input
    bg_changed: bool


def diff_scenes(src: SceneGraph, dst: SceneGraph) -> SceneDiff:
    """Compare input and output scenes of one training example.

    Matching strategy (greedy, multi-pass):
      1. Match by shape_signature + same color (strongest signal)
      2. Match by shape_signature only (color changed)
      3. Match by color + similar size (shape changed)
      4. Match remaining by closest size
    """
    src_unmatched = list(range(len(src.objects)))
    dst_unmatched = list(range(len(dst.objects)))
    matches: list[tuple[int, int]] = []  # (src_idx, dst_idx)

    # Pass 1: exact shape + same color
    for si in list(src_unmatched):
        s = src.objects[si]
        for di in list(dst_unmatched):
            d = dst.objects[di]
            if s.shape_signature == d.shape_signature and s.color == d.color:
                matches.append((si, di))
                src_unmatched.remove(si)
                dst_unmatched.remove(di)
                break

    # Pass 2: exact shape, different color
    for si in list(src_unmatched):
        s = src.objects[si]
        for di in list(dst_unmatched):
            d = dst.objects[di]
            if s.shape_signature == d.shape_signature:
                matches.append((si, di))
                src_unmatched.remove(si)
                dst_unmatched.remove(di)
                break

    # Pass 3: same color, closest size
    for si in list(src_unmatched):
        s = src.objects[si]
        candidates = [(di, dst.objects[di]) for di in dst_unmatched
                       if dst.objects[di].color == s.color]
        if candidates:
            best_di, best_d = min(candidates, key=lambda x: abs(x[1].size - s.size))
            matches.append((si, best_di))
            src_unmatched.remove(si)
            dst_unmatched.remove(best_di)

    # Pass 4: any remaining, match by closest size
    for si in list(src_unmatched):
        s = src.objects[si]
        if not dst_unmatched:
            break
        candidates = [(di, dst.objects[di]) for di in dst_unmatched]
        best_di, best_d = min(candidates, key=lambda x: abs(x[1].size - s.size))
        matches.append((si, best_di))
        src_unmatched.remove(si)
        dst_unmatched.remove(best_di)

    # Build ObjectDiffs
    matched_diffs = []
    for si, di in matches:
        s = src.objects[si]
        d = dst.objects[di]
        # Movement: difference in centers (rounded to int)
        dr = round(d.center[0] - s.center[0])
        dc = round(d.center[1] - s.center[1])
        matched_diffs.append(ObjectDiff(
            src=s,
            dst=d,
            new_color=d.color if d.color != s.color else None,
            movement=(dr, dc),
            shape_preserved=(s.shape_signature == d.shape_signature),
            size_delta=d.size - s.size,
        ))

    return SceneDiff(
        matched=matched_diffs,
        removed=[src.objects[i] for i in src_unmatched],
        added=[dst.objects[i] for i in dst_unmatched],
        bg_changed=(src.bg_color != dst.bg_color),
    )


# ── Rule inference ────────────────────────────────────────────────────────

@dataclass
class ObjectRule:
    """A simple object-level transformation rule."""
    kind: str  # "recolor", "remove", "move"
    src_color: int = 0
    dst_color: int = 0
    movement: tuple[int, int] = (0, 0)


def find_consistent_rules(diffs: list[SceneDiff]) -> list[ObjectRule]:
    """Find transformation rules consistent across ALL training examples.

    Strategy: collect per-color observations across all diffs,
    then keep only rules that hold in every example where that color appears.
    """
    if not diffs:
        return []

    rules: list[ObjectRule] = []

    # --- Recolor rules: color X always becomes color Y ---
    # Collect: for each src_color, what dst_colors do we see?
    recolor_map: dict[int, set[int]] = {}
    recolor_counts: dict[int, int] = {}  # how many examples have this src_color
    for diff in diffs:
        seen_colors: dict[int, set[int]] = {}
        for m in diff.matched:
            if m.new_color is not None:
                seen_colors.setdefault(m.src.color, set()).add(m.new_color)
        for color, dsts in seen_colors.items():
            recolor_map.setdefault(color, set()).update(dsts)
            recolor_counts[color] = recolor_counts.get(color, 0) + 1

    for src_c, dst_set in recolor_map.items():
        if len(dst_set) == 1:
            # Consistent: every time src_c appears, it becomes the same dst
            # Also check: in examples where src_c appears, it ALWAYS changes
            dst_c = next(iter(dst_set))
            # Verify no example has this color unchanged
            consistent = True
            for diff in diffs:
                for m in diff.matched:
                    if m.src.color == src_c and m.new_color is None:
                        consistent = False
                        break
                if not consistent:
                    break
            if consistent and recolor_counts.get(src_c, 0) >= 1:
                rules.append(ObjectRule(kind="recolor", src_color=src_c,
                                        dst_color=dst_c))

    # --- Removal rules: color X always removed ---
    removal_candidates: dict[int, int] = {}  # color → count of examples
    removal_presence: dict[int, int] = {}  # color → examples where it exists in input
    for diff in diffs:
        # Colors of removed objects
        removed_colors = {o.color for o in diff.removed}
        # Colors present in input
        input_colors = {m.src.color for m in diff.matched} | removed_colors
        for c in removed_colors:
            removal_candidates[c] = removal_candidates.get(c, 0) + 1
        for c in input_colors:
            removal_presence[c] = removal_presence.get(c, 0) + 1

    for color, remove_count in removal_candidates.items():
        presence = removal_presence.get(color, 0)
        if remove_count == presence and presence >= 1:
            # Every time this color appears, it gets removed
            # But check it's not also matched (partially removed)
            fully_removed = True
            for diff in diffs:
                for m in diff.matched:
                    if m.src.color == color and m.new_color is None:
                        fully_removed = False
                        break
                if not fully_removed:
                    break
            if fully_removed:
                rules.append(ObjectRule(kind="remove", src_color=color))

    return rules


# ── Rule application ──────────────────────────────────────────────────────

def apply_rules(grid: Grid, rules: list[ObjectRule], bg_color: int = 0) -> Grid:
    """Apply a set of object rules to produce an output grid."""
    result = [row[:] for row in grid]

    for rule in rules:
        if rule.kind == "recolor":
            for r in range(len(result)):
                for c in range(len(result[0])):
                    if result[r][c] == rule.src_color:
                        result[r][c] = rule.dst_color

        elif rule.kind == "remove":
            for r in range(len(result)):
                for c in range(len(result[0])):
                    if result[r][c] == rule.src_color:
                        result[r][c] = bg_color

    return result


# ── End-to-end pipeline ───────────────────────────────────────────────────

def solve_with_object_rules(task: dict) -> Optional[Callable[[Grid], Grid]]:
    """Try to solve an ARC task using object-level rule inference.

    Returns a callable (grid → grid) if a consistent rule set is found
    that is pixel-perfect on all training examples. Returns None otherwise.
    """
    train = task.get("train", [])
    if len(train) < 1:
        return None

    # Only same-dims tasks for now
    same_dims = all(
        len(ex["input"]) == len(ex["output"])
        and len(ex["input"][0]) == len(ex["output"][0])
        for ex in train
    )
    if not same_dims:
        return None

    # Build scenes and compute diffs
    diffs = []
    bg_colors = []
    for ex in train:
        src_scene = build_scene(ex["input"])
        dst_scene = build_scene(ex["output"])
        bg_colors.append(src_scene.bg_color)
        diffs.append(diff_scenes(src_scene, dst_scene))

    if not diffs:
        return None

    # Use the most common background color
    bg_color = Counter(bg_colors).most_common(1)[0][0]

    # Find consistent rules
    rules = find_consistent_rules(diffs)
    if not rules:
        return None

    # Validate: rules must produce pixel-perfect output on ALL training examples
    for ex in train:
        predicted = apply_rules(ex["input"], rules, bg_color)
        if predicted != ex["output"]:
            return None

    # Success: return a callable that applies the rules
    def transform(grid: Grid) -> Grid:
        return apply_rules(grid, rules, bg_color)

    return transform
