"""
Feature 2: Task Decomposition

DecompositionEngine attempts to solve hard tasks by breaking them down into
smaller, more manageable subproblems. Six strategies:

1. Color-channel decomposition: Solve each color independently, merge results
2. Spatial quadrant decomposition: Divide grid into quadrants, solve each
3. Input-output diff focus: Focus on cells that changed between in/out
4. Pattern decomposition: Detect repeating sub-patterns and solve the tile
5. Input-output size ratio: If output is 2x/3x input, try tiling/scaling
6. Masking decomposition: Separate foreground and background, solve each

This follows the principle of composability (Pillar 3): breaking problems
into simpler sub-problems that can be solved independently and then merged.
"""
from __future__ import annotations
from typing import Optional, Callable
from collections import Counter
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


def _detect_repeating_pattern(grid: Grid) -> Optional[tuple[int, int, Grid]]:
    """Detect if a grid is a repeating pattern of a smaller tile.

    Returns (tile_h, tile_w, tile_grid) if a repeating pattern is found,
    where the grid can be divided into tile_h x tile_w identical tiles.
    Returns None if no repeating pattern is detected.
    """
    if not grid or not grid[0]:
        return None

    h, w = len(grid), len(grid[0])

    # Try common tile sizes: 2x2, 3x3, 2x3, 3x2, etc.
    # (only if grid dimensions are divisible)
    candidates = []
    for th in range(1, h // 2 + 1):
        if h % th == 0:
            for tw in range(1, w // 2 + 1):
                if w % tw == 0:
                    candidates.append((th, tw))

    # Try candidates, preferring larger tiles (simpler patterns)
    candidates.sort(key=lambda x: -(x[0] * x[1]))

    for tile_h, tile_w in candidates:
        # Extract first tile
        first_tile = [grid[r][:tile_w] for r in range(tile_h)]

        # Check if all tiles match the first one
        valid = True
        for tile_r in range(h // tile_h):
            for tile_c in range(w // tile_w):
                start_r = tile_r * tile_h
                start_c = tile_c * tile_w
                tile = [grid[r][start_c:start_c + tile_w]
                       for r in range(start_r, start_r + tile_h)]
                if tile != first_tile:
                    valid = False
                    break
            if not valid:
                break

        if valid:
            return (tile_h, tile_w, first_tile)

    return None


def _separate_foreground_background(grid: Grid) -> tuple[Grid, int]:
    """Separate grid into foreground (non-zero) and background color.

    Returns (foreground_mask, background_color) where foreground_mask is
    a binary grid (1 for foreground, 0 for background).
    """
    if not grid or not grid[0]:
        return [[]], 0

    # Background is the most common color
    flat = [cell for row in grid for cell in row]
    bg = Counter(flat).most_common(1)[0][0]

    # Create foreground mask (1 = non-bg, 0 = bg)
    mask = [[1 if cell != bg else 0 for cell in row] for row in grid]

    return mask, bg


def _merge_foreground_background(fg_grid: Grid, bg_color: int) -> Grid:
    """Merge a foreground grid with background color.

    The input fg_grid is expected to have colors for foreground cells.
    Background cells are colored with bg_color.
    """
    if not fg_grid or not fg_grid[0]:
        return [[bg_color]]

    h, w = len(fg_grid), len(fg_grid[0])
    result = [[bg_color] * w for _ in range(h)]

    for r in range(h):
        for c in range(w):
            if fg_grid[r][c] != 0:  # Foreground cell
                result[r][c] = fg_grid[r][c]

    return result


def _get_bounding_box(mask: Grid) -> Optional[tuple[int, int, int, int]]:
    """Get bounding box of non-zero cells in a grid.

    Returns (min_r, min_c, max_r, max_c) or None if all zero.
    """
    if not mask or not mask[0]:
        return None

    h, w = len(mask), len(mask[0])
    min_r, min_c = h, w
    max_r, max_c = -1, -1

    for r in range(h):
        for c in range(w):
            if mask[r][c] != 0:
                min_r = min(min_r, r)
                min_c = min(min_c, c)
                max_r = max(max_r, r)
                max_c = max(max_c, c)

    if max_r < 0:
        return None

    return (min_r, min_c, max_r, max_c)


def _extract_subgrid(grid: Grid, r0: int, c0: int, r1: int, c1: int) -> Grid:
    """Extract a rectangular subgrid from (r0,c0) to (r1,c1) inclusive."""
    return [grid[r][c0:c1 + 1] for r in range(r0, r1 + 1)]


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

    def try_pattern_decomposition(
        self,
        task: dict,
        synthesize_fn: Callable[[dict], tuple[Optional[Program], list[dict]]]
    ) -> Optional[Program]:
        """Try pattern decomposition: detect repeating tiles and solve the tile.

        For grids that consist of a repeating pattern (e.g., 3x3 tile repeated
        in a 9x9 grid), extract and solve just the tile, then apply the
        solution to all tiles.

        Args:
            task: ARC task with 'train' examples
            synthesize_fn: Function (task_dict) -> (program, history)

        Returns:
            A program that solves via pattern decomposition, or None.
        """
        examples = task.get('train', [])
        if not examples:
            return None

        # Check if all input examples have the same repeating pattern
        pattern_info = None
        for example in examples:
            info = _detect_repeating_pattern(example['input'])
            if info is None:
                return None  # Not all inputs have repeating patterns
            if pattern_info is None:
                pattern_info = info
            else:
                # Verify all inputs have same tile dimensions
                if (info[0] != pattern_info[0] or info[1] != pattern_info[1]):
                    return None

        if pattern_info is None:
            return None

        tile_h, tile_w, _ = pattern_info

        # Create a tile-based task: just the single tile and its expected output
        tile_task = {'train': []}
        for example in examples:
            # Extract first tile from input
            input_tile = _extract_subgrid(example['input'], 0, 0, tile_h - 1, tile_w - 1)
            # Extract first tile from output (same dimensions)
            output_tile = _extract_subgrid(example['output'], 0, 0, tile_h - 1, tile_w - 1)
            tile_task['train'].append({
                'input': input_tile,
                'output': output_tile,
            })

        # Synthesize on the tile task
        tile_program, _ = synthesize_fn(tile_task)
        if tile_program is None:
            return None

        # Create composite program that applies tile solution to all tiles
        def composite(grid: Grid) -> Optional[Grid]:
            """Apply the tile solution to each tile and reassemble."""
            h, w = len(grid), len(grid[0]) if grid else 0

            # Verify grid dimensions are compatible with tile size
            if h % tile_h != 0 or w % tile_w != 0:
                return None

            result = [[0] * w for _ in range(h)]

            # Process each tile
            for tile_r in range(h // tile_h):
                for tile_c in range(w // tile_w):
                    start_r = tile_r * tile_h
                    start_c = tile_c * tile_w
                    tile = _extract_subgrid(grid, start_r, start_c,
                                           start_r + tile_h - 1, start_c + tile_w - 1)

                    # Apply tile program
                    solved_tile = tile_program.execute(tile)
                    if solved_tile is None:
                        return None

                    # Place back in result
                    for r in range(tile_h):
                        for c in range(tile_w):
                            if r < len(solved_tile) and c < len(solved_tile[0]):
                                result[start_r + r][start_c + c] = solved_tile[r][c]

            return result

        program = Program([Concept(
            kind="composed",
            name="pattern_decomp",
            implementation=composite,
        )])
        program.fitness = score_program_on_task(program, task)
        return program

    def try_size_ratio_decomposition(
        self,
        task: dict,
        synthesize_fn: Callable[[dict], tuple[Optional[Program], list[dict]]]
    ) -> Optional[Program]:
        """Try size-ratio decomposition: if output is 2x/3x input, try scaling/tiling.

        If output dimensions are exactly 2x or 3x the input dimensions,
        decompose as: solution on scaled/tiled input, then apply appropriate
        dimension transform to get back to original size.

        Args:
            task: ARC task with 'train' examples
            synthesize_fn: Function (task_dict) -> (program, history)

        Returns:
            A program that uses scaling/tiling decomposition, or None.
        """
        examples = task.get('train', [])
        if not examples:
            return None

        # Check dimension ratios across all examples
        ratios = set()
        for ex in examples:
            h_in, w_in = len(ex['input']), len(ex['input'][0])
            h_out, w_out = len(ex['output']), len(ex['output'][0])
            if h_in == 0 or w_in == 0:
                return None
            h_ratio = h_out / h_in
            w_ratio = w_out / w_in
            ratios.add((h_ratio, w_ratio))

        if len(ratios) != 1:
            return None  # Inconsistent ratios

        h_ratio, w_ratio = ratios.pop()

        # Check if ratio matches 2x or 3x expansion
        scale_factor = None
        if h_ratio == 2.0 and w_ratio == 2.0:
            scale_factor = 2
        elif h_ratio == 3.0 and w_ratio == 3.0:
            scale_factor = 3
        else:
            return None

        # Create downscaled task: work with input size, then upscale to output size
        downscaled_task = {'train': []}
        for ex in examples:
            # For downscale: just use the input as-is; scaling is implicit
            input_grid = ex['input']

            # For output: downscale to match input dimensions
            h_out, w_out = len(ex['output']), len(ex['output'][0])
            h_target = h_out // scale_factor
            w_target = w_out // scale_factor

            # Simple downscale: take every scale_factor-th pixel
            downscaled_out = [
                [ex['output'][r][c]
                 for c in range(0, w_out, scale_factor)]
                for r in range(0, h_out, scale_factor)
            ]

            downscaled_task['train'].append({
                'input': input_grid,
                'output': downscaled_out,
            })

        # Synthesize on downscaled task
        program, _ = synthesize_fn(downscaled_task)
        if program is None:
            return None

        # Create composite that applies program then upscales
        def composite(grid: Grid) -> Optional[Grid]:
            """Apply program, then upscale by scale_factor."""
            # Apply the program
            result = program.execute(grid)
            if result is None:
                return None

            # Upscale: repeat each pixel scale_factor times
            h, w = len(result), len(result[0]) if result else 0
            upscaled = []
            for row in result:
                # Repeat row elements
                expanded_row = []
                for cell in row:
                    expanded_row.extend([cell] * scale_factor)
                # Add this row scale_factor times
                for _ in range(scale_factor):
                    upscaled.append(expanded_row[:])

            return upscaled

        program = Program([Concept(
            kind="composed",
            name="size_ratio_decomp",
            implementation=composite,
        )])
        program.fitness = score_program_on_task(program, task)
        return program

    def try_masking_decomposition(
        self,
        task: dict,
        synthesize_fn: Callable[[dict], tuple[Optional[Program], list[dict]]]
    ) -> Optional[Program]:
        """Try masking decomposition: separate foreground and background, solve each.

        For tasks where foreground and background can be modified independently,
        extract each, synthesize separate solutions, then merge.

        Args:
            task: ARC task with 'train' examples
            synthesize_fn: Function (task_dict) -> (program, history)

        Returns:
            A program that solves via foreground/background decomposition, or None.
        """
        examples = task.get('train', [])
        if not examples:
            return None

        # Check if all examples have same-dimensions (required for masking)
        for ex in examples:
            if len(ex['input']) != len(ex['output']):
                return None
            if len(ex['input'][0]) != len(ex['output'][0]):
                return None

        # Extract foreground masks and background colors from all inputs
        bg_color = None
        for ex in examples:
            mask, bg = _separate_foreground_background(ex['input'])
            if bg_color is None:
                bg_color = bg
            elif bg_color != bg:
                return None  # Inconsistent background color

        if bg_color is None:
            return None

        # Create foreground and background tasks
        fg_task = {'train': []}
        bg_task = {'train': []}

        for ex in examples:
            inp = ex['input']
            out = ex['output']

            # Extract foreground grids (only non-bg cells)
            fg_input = [row[:] for row in inp]
            fg_output = [row[:] for row in out]

            # Create background grids (only background cells)
            bg_input = [
                [bg_color if inp[r][c] == bg_color else 0 for c in range(len(inp[0]))]
                for r in range(len(inp))
            ]
            bg_output = [
                [bg_color if out[r][c] == bg_color else 0 for c in range(len(out[0]))]
                for r in range(len(out))
            ]

            fg_task['train'].append({'input': fg_input, 'output': fg_output})
            bg_task['train'].append({'input': bg_input, 'output': bg_output})

        # Try synthesizing both tasks
        fg_program, _ = synthesize_fn(fg_task)
        if fg_program is None:
            return None

        bg_program, _ = synthesize_fn(bg_task)
        if bg_program is None:
            return None

        # Create composite that applies both and merges
        def composite(grid: Grid) -> Optional[Grid]:
            """Apply foreground and background solutions and merge."""
            # Apply foreground solution
            fg_result = fg_program.execute(grid)
            if fg_result is None:
                return None

            # Create background grid
            h, w = len(grid), len(grid[0])
            bg_input = [
                [bg_color if grid[r][c] == bg_color else 0 for c in range(w)]
                for r in range(h)
            ]

            # Apply background solution
            bg_result = bg_program.execute(bg_input)
            if bg_result is None:
                return None

            # Merge: take foreground cells from fg_result, bg cells from bg_result
            result = [[0] * w for _ in range(h)]
            for r in range(h):
                for c in range(w):
                    fg_cell = fg_result[r][c] if r < len(fg_result) and c < len(fg_result[0]) else 0
                    bg_cell = bg_result[r][c] if r < len(bg_result) and c < len(bg_result[0]) else 0

                    # If bg_cell is the background color, use fg_cell
                    if bg_cell == bg_color:
                        result[r][c] = fg_cell
                    else:
                        result[r][c] = bg_cell

            return result

        program = Program([Concept(
            kind="composed",
            name="masking_decomp",
            implementation=composite,
        )])
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

        # Try strategies in order: simpler/faster first
        strategies = [
            self.try_color_channel_decomposition,
            self.try_spatial_decomposition,
            self.try_diff_focus_decomposition,
            self.try_pattern_decomposition,
            self.try_size_ratio_decomposition,
            self.try_masking_decomposition,
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
