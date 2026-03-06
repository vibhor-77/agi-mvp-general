"""
Feature 2: Task Decomposition

DecompositionEngine attempts to solve hard tasks by breaking them down into
smaller, more manageable subproblems. Three strategies:

1. Color-channel decomposition: Solve each color independently, merge results
2. Spatial quadrant decomposition: Divide grid into quadrants, solve each
3. Input-output diff focus: Focus on cells that changed between in/out
"""
from __future__ import annotations
from typing import Optional, Callable
from .concepts import Program, Grid, Concept
from .scorer import score_program_on_task


def _extract_color_channel(grid: Grid, color: int) -> Grid:
    """Extract a single color channel as a binary mask (color vs 0)."""
    return [[1 if cell == color else 0 for cell in row] for row in grid]


def _merge_color_channels(channels: dict[int, Grid]) -> Grid:
    """Merge multiple color channels back into a single grid."""
    if not channels:
        return [[0]]

    # Get dimensions from first channel
    first_grid = next(iter(channels.values()))
    h = len(first_grid)
    w = len(first_grid[0]) if first_grid else 0

    result = [[0] * w for _ in range(h)]

    # Overlay each color channel
    for color, grid in channels.items():
        for r in range(h):
            for c in range(w):
                if grid[r][c] != 0:
                    result[r][c] = color

    return result


def _split_into_quadrants(grid: Grid) -> dict[str, Grid]:
    """Divide grid into four quadrants: 'TL', 'TR', 'BL', 'BR'."""
    h, w = len(grid), len(grid[0]) if grid else 0
    mid_h = h // 2
    mid_w = w // 2

    quadrants = {}

    # Top-left
    quadrants['TL'] = [row[:mid_w] for row in grid[:mid_h]]

    # Top-right
    quadrants['TR'] = [row[mid_w:] for row in grid[:mid_h]]

    # Bottom-left
    quadrants['BL'] = [row[:mid_w] for row in grid[mid_h:]]

    # Bottom-right
    quadrants['BR'] = [row[mid_w:] for row in grid[mid_h:]]

    return quadrants


def _merge_quadrants(quadrants: dict[str, Grid]) -> Grid:
    """Merge four quadrants back into a single grid."""
    tl = quadrants.get('TL', [[0]])
    tr = quadrants.get('TR', [[0]])
    bl = quadrants.get('BL', [[0]])
    br = quadrants.get('BR', [[0]])

    # Merge rows in top half
    top_rows = []
    for tl_row, tr_row in zip(tl, tr):
        top_rows.append(tl_row + tr_row)

    # Merge rows in bottom half
    bottom_rows = []
    for bl_row, br_row in zip(bl, br):
        bottom_rows.append(bl_row + br_row)

    return top_rows + bottom_rows


def _find_changed_cells(input_grid: Grid, output_grid: Grid) -> set[tuple[int, int]]:
    """Find all cells that differ between input and output."""
    changed = set()
    h_in, w_in = len(input_grid), len(input_grid[0]) if input_grid else 0
    h_out, w_out = len(output_grid), len(output_grid[0]) if output_grid else 0

    # Check all cells that exist in either grid
    for r in range(max(h_in, h_out)):
        for c in range(max(w_in, w_out)):
            in_val = input_grid[r][c] if r < h_in and c < w_in else 0
            out_val = output_grid[r][c] if r < h_out and c < w_out else 0
            if in_val != out_val:
                changed.add((r, c))

    return changed


def _extract_region_around_changes(grid: Grid, changed_cells: set[tuple[int, int]],
                                    padding: int = 1) -> Grid:
    """Extract a rectangular region containing all changed cells with padding."""
    if not changed_cells:
        return grid

    rows, cols = zip(*changed_cells)
    min_r, max_r = min(rows), max(rows)
    min_c, max_c = min(cols), max(cols)

    # Apply padding
    min_r = max(0, min_r - padding)
    max_r = min(len(grid) - 1, max_r + padding)
    min_c = max(0, min_c - padding)
    max_c = min(len(grid[0]) - 1, max_c + padding) if grid else 0

    return [grid[r][min_c:max_c + 1] for r in range(min_r, max_r + 1)]


class DecompositionEngine:
    """Attempts to solve hard tasks via problem decomposition.

    When evolutionary synthesis achieves a score < 0.99, this engine tries
    three decomposition strategies:
      1. Color-channel: Solve each color independently
      2. Spatial quadrants: Solve quadrants separately
      3. Diff focus: Focus on changed regions
    """

    def __init__(self, toolkit=None):
        """Initialize the decomposition engine.

        Args:
            toolkit: Optional toolkit to find concepts (unused for now,
                    but useful for future semantic decomposition).
        """
        self.toolkit = toolkit

    def try_color_channel_decomposition(
        self,
        task: dict,
        synthesize_fn: Callable[[dict], tuple[Optional[Program], list[dict]]]
    ) -> Optional[Program]:
        """Try color-channel decomposition strategy.

        For each non-zero color in the training examples, create a binary
        (single-color) task and synthesize a solution for that channel.
        Then merge the solutions.

        Args:
            task: ARC task with 'train' examples
            synthesize_fn: Function (task_dict) -> (program, history)

        Returns:
            A program that solves the task via color decomposition, or None.
        """
        # Extract all non-zero colors from training inputs
        colors_seen = set()
        for example in task.get('train', []):
            for row in example.get('input', []):
                for cell in row:
                    if cell != 0:
                        colors_seen.add(cell)

        if len(colors_seen) <= 1:
            # Not useful for single-color tasks
            return None

        # For each color, create a binary task and synthesize
        color_programs = {}
        for color in colors_seen:
            binary_task = {
                'train': [
                    {
                        'input': _extract_color_channel(ex['input'], color),
                        'output': _extract_color_channel(ex['output'], color),
                    }
                    for ex in task.get('train', [])
                ]
            }

            program, _ = synthesize_fn(binary_task)
            if program is None:
                return None
            color_programs[color] = program

        # Create composite program that solves via color decomposition
        def composite(grid: Grid) -> Optional[Grid]:
            """Apply per-color synthesis, then merge."""
            # Extract channels
            channels = {
                color: _extract_color_channel(grid, color)
                for color in color_programs.keys()
            }

            # Apply each program to its channel
            solved_channels = {}
            for color, program in color_programs.items():
                channel_grid = channels[color]
                result = program.execute(channel_grid)
                if result is None:
                    return None
                solved_channels[color] = result

            # Merge back
            return _merge_color_channels(solved_channels)

        program = Program([Concept(
            kind="composed",
            name="color_channel_decomp",
            implementation=composite,
        )])
        program.fitness = score_program_on_task(program, task)
        return program

    def try_spatial_decomposition(
        self,
        task: dict,
        synthesize_fn: Callable[[dict], tuple[Optional[Program], list[dict]]]
    ) -> Optional[Program]:
        """Try spatial quadrant decomposition strategy.

        Divide training grids into quadrants, synthesize a solution for each,
        then compose them.

        Args:
            task: ARC task with 'train' examples
            synthesize_fn: Function (task_dict) -> (program, history)

        Returns:
            A program that solves via spatial decomposition, or None.
        """
        # Check if input grids are large enough for meaningful quadrant split
        examples = task.get('train', [])
        if not examples:
            return None

        min_dim = min(len(ex['input']) for ex in examples)
        if min_dim < 4:
            # Too small to decompose meaningfully
            return None

        # Create quadrant tasks
        quadrant_programs = {}
        for quad_name in ['TL', 'TR', 'BL', 'BR']:
            quad_task = {
                'train': []
            }

            for example in examples:
                quads_in = _split_into_quadrants(example['input'])
                quads_out = _split_into_quadrants(example['output'])

                # Check if this quadrant has meaningful content
                quad_in = quads_in.get(quad_name, [[0]])
                quad_out = quads_out.get(quad_name, [[0]])

                if not quad_in or not quad_out:
                    return None

                quad_task['train'].append({
                    'input': quad_in,
                    'output': quad_out,
                })

            program, _ = synthesize_fn(quad_task)
            if program is None:
                return None
            quadrant_programs[quad_name] = program

        # Create composite program
        def composite(grid: Grid) -> Optional[Grid]:
            """Apply per-quadrant synthesis, then merge."""
            quadrants = _split_into_quadrants(grid)
            solved_quads = {}

            for quad_name, program in quadrant_programs.items():
                quad_grid = quadrants.get(quad_name, [[0]])
                result = program.execute(quad_grid)
                if result is None:
                    return None
                solved_quads[quad_name] = result

            return _merge_quadrants(solved_quads)

        program = Program([Concept(
            kind="composed",
            name="spatial_quadrant_decomp",
            implementation=composite,
        )])
        program.fitness = score_program_on_task(program, task)
        return program

    def try_diff_focus_decomposition(
        self,
        task: dict,
        synthesize_fn: Callable[[dict], tuple[Optional[Program], list[dict]]]
    ) -> Optional[Program]:
        """Try diff-focus decomposition strategy.

        Find which cells changed between input and output, extract that region,
        synthesize a solution on the reduced problem.

        Args:
            task: ARC task with 'train' examples
            synthesize_fn: Function (task_dict) -> (program, history)

        Returns:
            A program that focuses on changed regions, or None.
        """
        examples = task.get('train', [])
        if not examples:
            return None

        # Find the changed region across all examples
        all_changed = set()
        for example in examples:
            changed = _find_changed_cells(example['input'], example['output'])
            all_changed.update(changed)

        if not all_changed:
            # No changes detected (shouldn't happen for a non-trivial task)
            return None

        # Extract the region containing changes
        first_input = examples[0]['input']
        change_region = _extract_region_around_changes(first_input, all_changed, padding=1)

        if not change_region:
            return None

        # Create a focused task on the change region
        focused_task = {
            'train': [
                {
                    'input': _extract_region_around_changes(
                        ex['input'],
                        _find_changed_cells(ex['input'], ex['output']),
                        padding=1
                    ),
                    'output': _extract_region_around_changes(
                        ex['output'],
                        _find_changed_cells(ex['input'], ex['output']),
                        padding=1
                    ),
                }
                for ex in examples
            ]
        }

        # Synthesize on focused task
        program, _ = synthesize_fn(focused_task)
        if program is None:
            return None

        program.fitness = score_program_on_task(program, task)
        return program

    def decompose_if_needed(
        self,
        task: dict,
        best_score: float,
        synthesize_fn: Callable[[dict], tuple[Optional[Program], list[dict]]]
    ) -> Optional[Program]:
        """Try decomposition strategies if best_score is below threshold.

        Args:
            task: ARC task
            best_score: Current best score from standard synthesis
            synthesize_fn: Function to synthesize on a task

        Returns:
            A decomposed program if found and better than best_score, else None.
        """
        if best_score >= 0.99:
            # Already solved well enough
            return None

        # Try strategies in order
        strategies = [
            self.try_color_channel_decomposition,
            self.try_spatial_decomposition,
            self.try_diff_focus_decomposition,
        ]

        for strategy in strategies:
            try:
                program = strategy(task, synthesize_fn)
                if program is not None and program.fitness > best_score:
                    return program
            except Exception:
                # Silently skip strategies that fail
                pass

        return None
