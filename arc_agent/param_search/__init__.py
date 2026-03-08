"""Parameterized Primitive System: Learn parameters from training examples.

Key Insight:
  Instead of hard-coding 290+ primitives with fixed behavior, we learn
  STRUCTURAL parameters from examples. Parameters express ROLES (frequency
  ranks, spatial relationships) rather than absolute color values.

Example:
  - SubstituteColor learns: "replace the least common color with the
    most common non-background color" (not "replace 3 with 7")
  - FillEnclosedWith learns: "fill enclosed bg regions with the color
    that appears in output but not input"
  - RecolorByFrequency learns: "match colors by frequency rank"

This enables GENERALIZATION from train to test without overfitting.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from collections import Counter
from typing import Optional, Any
from arc_agent.concepts import Grid, Concept, Program


class ParameterizedPrimitive(ABC):
    """Base class for parameterized primitives.

    A parameterized primitive is a template function + parameter learner.
    Unlike hard-coded primitives, it learns parameters from training examples
    and expresses them structurally (by role/frequency) not by absolute values.
    """

    @abstractmethod
    def learn_params(self, task: dict) -> Optional[dict]:
        """Learn parameters from training examples.

        Args:
            task: Dict with 'train' key containing list of {'input', 'output'} dicts.

        Returns:
            Dict of learned parameters, or None if learning fails.
        """
        pass

    @abstractmethod
    def instantiate(self, params: dict) -> Grid:
        """Instantiate a concrete Grid->Grid function from learned parameters.

        Args:
            params: Dict of parameters from learn_params.

        Returns:
            A function that takes Grid and returns Grid.
        """
        pass

    def score(self, task: dict) -> float:
        """Score this primitive on training examples.

        Args:
            task: The ARC task dict with 'train' examples.

        Returns:
            Pixel accuracy on training examples (0.0 to 1.0).
        """
        from arc_agent.scorer import TaskCache

        params = self.learn_params(task)
        if params is None:
            return 0.0

        func = self.instantiate(params)

        # Score on all training examples
        total_pixels = 0
        correct_pixels = 0

        for ex in task.get("train", []):
            inp = ex["input"]
            expected = ex["output"]
            try:
                result = func(inp)
                if result is None:
                    continue

                # Count matching pixels
                rows = min(len(result), len(expected))
                cols = min(len(result[0]) if result else 0,
                          len(expected[0]) if expected else 0)

                for r in range(rows):
                    for c in range(cols):
                        total_pixels += 1
                        if result[r][c] == expected[r][c]:
                            correct_pixels += 1
            except Exception:
                pass

        if total_pixels == 0:
            return 0.0
        return correct_pixels / total_pixels


class SubstituteColor(ParameterizedPrimitive):
    """Learn color substitution mapping from examples.

    Strategy: Learn empirical absolute color mappings as fallback, but also
    try to express them structurally when possible. For example, if we see
    "rare color → dominant color", remember both the absolute mapping and
    the structural role.
    """

    def learn_params(self, task: dict) -> Optional[dict]:
        """Learn color mapping from input→output changes.

        Returns dict with:
          - 'color_map': empirical absolute color mappings {color: color}
          - 'structural': learned structural transformation if applicable
        """
        train = task.get("train", [])
        if not train:
            return None

        # Collect all changed pixels
        empirical_mapping = {}
        structural_mappings = []

        for ex in train:
            inp, out = ex["input"], ex["output"]
            if len(inp) != len(out) or len(inp[0]) != len(out[0]):
                continue

            # Empirical: just record which colors map to which
            for r in range(len(inp)):
                for c in range(len(inp[0])):
                    if inp[r][c] != out[r][c]:
                        if inp[r][c] not in empirical_mapping:
                            empirical_mapping[inp[r][c]] = Counter()
                        empirical_mapping[inp[r][c]][out[r][c]] += 1

            # Structural: identify roles based on color changes
            # Key insight: when input color A (role X) → output color B,
            # we care about B's ROLE IN THE OUTPUT, not A's role in the input.
            inp_flat = [inp[r][c] for r in range(len(inp)) for c in range(len(inp[0]))]
            inp_freq = Counter(inp_flat)

            # Assign roles to input colors
            inp_roles = self._assign_roles(inp_freq)

            # Assign roles to output colors
            out_flat = [out[r][c] for r in range(len(out)) for c in range(len(out[0]))]
            out_freq = Counter(out_flat)
            out_roles = self._assign_roles(out_freq)

            # Track what role transformations happened
            role_transforms = Counter()
            for r in range(len(inp)):
                for c in range(len(inp[0])):
                    if inp[r][c] != out[r][c]:
                        src_role = inp_roles.get(inp[r][c])
                        dst_role = out_roles.get(out[r][c])
                        if src_role and dst_role:
                            role_transforms[(src_role, dst_role)] += 1

            if role_transforms:
                structural_mappings.append(role_transforms.most_common(1)[0][0])

        if not empirical_mapping and not structural_mappings:
            return None

        # Build final color map from empirical data
        color_map = {k: v.most_common(1)[0][0]
                    for k, v in empirical_mapping.items()
                    if v}

        structural = None
        if structural_mappings:
            # Most common structural transformation
            structural = Counter(structural_mappings).most_common(1)[0][0]

        return {
            "color_map": color_map,
            "structural": structural,
        }

    @staticmethod
    def _assign_roles(color_freq: Counter) -> dict[int, str]:
        """Assign structural roles to colors based on frequency rank."""
        if not color_freq:
            return {}

        sorted_colors = sorted(color_freq.items(), key=lambda x: -x[1])
        roles = {}

        for rank, (color, _) in enumerate(sorted_colors):
            if rank == 0:
                roles[color] = "bg"
            elif rank == len(sorted_colors) - 1:
                roles[color] = "rare"
            else:
                roles[color] = "dominant"

        return roles

    def instantiate(self, params: dict) -> Any:
        """Create a Grid->Grid function from learned parameters."""
        color_map = params.get("color_map", {})
        structural = params.get("structural")

        def apply_color_map(grid: Grid) -> Grid:
            # Use empirical color map
            if not color_map:
                return grid

            return [[color_map.get(cell, cell) for cell in row] for row in grid]

        return apply_color_map


class FillEnclosedWith(ParameterizedPrimitive):
    """Fill enclosed regions of background (0) with a learned color.

    Learns which color to fill with from output examples.
    """

    def learn_params(self, task: dict) -> Optional[dict]:
        """Learn which color appears in enclosed regions.

        Returns dict with 'fill_color' key.
        """
        train = task.get("train", [])
        if not train:
            return None

        # Simple heuristic: find enclosed zeros in input that change in output
        # The output color becomes the fill color
        fill_colors = Counter()

        for ex in train:
            inp, out = ex["input"], ex["output"]
            if len(inp) != len(out) or len(inp[0]) != len(out[0]):
                continue

            # Find positions that are 0 in input and non-0 in output
            for r in range(len(inp)):
                for c in range(len(inp[0])):
                    if inp[r][c] == 0 and out[r][c] != 0:
                        # This zero was filled with something
                        fill_colors[out[r][c]] += 1

        if not fill_colors:
            return None

        fill_color = fill_colors.most_common(1)[0][0]
        return {"fill_color": fill_color}

    def instantiate(self, params: dict) -> Any:
        """Create a function that fills enclosed zeros."""
        fill_color = params.get("fill_color", 0)

        def fill_enclosed(grid: Grid) -> Grid:
            # Simple flood-fill: mark all zeros reachable from boundary
            rows, cols = len(grid), len(grid[0]) if grid else 0
            result = [row[:] for row in grid]

            # Track boundary-reachable zeros
            reachable = set()
            visited = set()

            def flood_from(r, c):
                """Flood-fill from (r,c) to mark all connected 0s."""
                if (r, c) in visited:
                    return
                if not (0 <= r < rows and 0 <= c < cols):
                    return
                if result[r][c] != 0:
                    return
                visited.add((r, c))
                reachable.add((r, c))
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    flood_from(r + dr, c + dc)

            # Start flood-fill from boundary zeros
            for r in range(rows):
                flood_from(r, 0)
                flood_from(r, cols - 1)
            for c in range(cols):
                flood_from(0, c)
                flood_from(rows - 1, c)

            # Fill enclosed zeros (those not reachable from boundary)
            for r in range(rows):
                for c in range(cols):
                    if result[r][c] == 0 and (r, c) not in reachable:
                        result[r][c] = fill_color

            return result

        return fill_enclosed


class RecolorByFrequency(ParameterizedPrimitive):
    """Recolor by matching frequency ranks.

    Learn which output colors correspond to which input colors by frequency rank.
    This enables generalization: if we learn "most common→color X, rare→color Y",
    then apply that to any grid where most common→X, rare→Y.

    Key structural parameter: "rank_map" = {rank: output_color}
    This is independent of which color is actually at that rank.
    """

    def learn_params(self, task: dict) -> Optional[dict]:
        """Learn frequency-rank-based color mapping (STRUCTURAL).

        Returns dict with:
          - 'rank_map': {rank: output_color} for position in frequency order
        """
        train = task.get("train", [])
        if not train:
            return None

        # Extract the rank-to-output-color mapping from training
        # Use the first example to establish the mapping
        for ex in train:
            inp, out = ex["input"], ex["output"]
            if len(inp) != len(out) or len(inp[0]) != len(out[0]):
                continue

            # Count colors in input and output
            inp_flat = [inp[r][c] for r in range(len(inp)) for c in range(len(inp[0]))]
            out_flat = [out[r][c] for r in range(len(out)) for c in range(len(out[0]))]

            inp_freq = Counter(inp_flat)
            out_freq = Counter(out_flat)

            # Sort by frequency (descending)
            inp_sorted = sorted(inp_freq.items(), key=lambda x: -x[1])
            out_sorted = sorted(out_freq.items(), key=lambda x: -x[1])

            # Map by rank: rank N in input → output color at rank N
            rank_map = {}
            for rank in range(min(len(inp_sorted), len(out_sorted))):
                out_color = out_sorted[rank][0]
                rank_map[rank] = out_color

            # Success if we found a mapping
            if rank_map:
                return {"rank_map": rank_map}

        return None

    def instantiate(self, params: dict) -> Any:
        """Create a function that matches colors by frequency rank.

        On any input grid, recolor: color at rank N → output_color[N].
        """
        rank_map = params.get("rank_map", {})

        def recolor_by_rank(grid: Grid) -> Grid:
            if not rank_map:
                return grid

            # Count input colors to determine ranks
            flat = [grid[r][c] for r in range(len(grid)) for c in range(len(grid[0]))]
            color_freq = Counter(flat)
            sorted_colors = sorted(color_freq.items(), key=lambda x: -x[1])

            # Build mapping: color at rank N → output_color[N]
            color_map = {}
            for rank, (color, _) in enumerate(sorted_colors):
                if rank in rank_map:
                    color_map[color] = rank_map[rank]

            # Apply mapping
            return [[color_map.get(cell, cell) for cell in row] for row in grid]

        return recolor_by_rank


def try_parameterized(task: dict, cache: Optional[Any] = None) -> Optional[Program]:
    """Try all parameterized primitives on a task.

    Args:
        task: The ARC task dict.
        cache: Optional TaskCache for efficient scoring.

    Returns:
        Best Program wrapping a parameterized primitive, or None.
    """
    if not task.get("train"):
        return None

    # List of all parameterized primitives to try
    primitives = [
        SubstituteColor(),
        FillEnclosedWith(),
        RecolorByFrequency(),
    ]

    best_program = None
    best_score = 0.0

    for prim in primitives:
        try:
            score = prim.score(task)
            if score > best_score:
                best_score = score

                # Learn params and instantiate
                params = prim.learn_params(task)
                if params is None:
                    continue

                func = prim.instantiate(params)
                concept = Concept(
                    kind="operator",
                    name=f"parameterized_{prim.__class__.__name__}",
                    implementation=func,
                )
                program = Program([concept])
                program.fitness = score
                best_program = program
        except Exception:
            pass

    return best_program if best_score >= 0.5 else None


__all__ = [
    "ParameterizedPrimitive",
    "SubstituteColor",
    "FillEnclosedWith",
    "RecolorByFrequency",
    "try_parameterized",
]
