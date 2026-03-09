# Decomposition Engine Improvements

## Overview

The DecompositionEngine has been enhanced with three new decomposition strategies to handle complex ARC tasks by breaking them into simpler subproblems. This aligns with **Pillar 3 (Composability)**: breaking hard problems into manageable pieces that can be solved independently and merged.

## Current Strategies

The DecompositionEngine now implements six strategies (tried in order of speed):

### 1. Color-Channel Decomposition (Original)

**Idea**: Solve each color independently, then merge results.

**When to use**: Tasks where different colored regions transform independently.

**Algorithm**:
1. Extract binary masks for each non-zero color
2. Synthesize a solution for each color channel
3. Merge all color channels back together

**Example**: A grid with red (1) and blue (2) regions where red stays red and blue becomes green.

---

### 2. Spatial Quadrant Decomposition (Original)

**Idea**: Divide grid into 4 quadrants, solve each separately.

**When to use**: Large grids (≥4x4) that transform in a quadrant-local manner.

**Algorithm**:
1. Split input/output examples into 4 quadrants (TL, TR, BL, BR)
2. Synthesize a solution for each quadrant
3. Apply each solution and merge quadrants back

**Example**: A 4x4 grid where top-left rotates, top-right flips, etc.

---

### 3. Diff-Focus Decomposition (Original)

**Idea**: Focus synthesis on cells that actually change.

**When to use**: Tasks where most cells stay the same, but a small region changes.

**Algorithm**:
1. Find all cells that differ between input and output
2. Extract bounding box around changed cells (with padding)
3. Synthesize on the reduced problem
4. Apply to full grid

**Example**: A large grid mostly background with one small object that transforms.

---

### 4. Pattern Decomposition (NEW)

**Idea**: Detect repeating tile patterns and solve just one tile.

**When to use**: Grids that consist of identical repeating tiles (e.g., 3x3 tile repeated 3x3 times).

**Algorithm**:
1. Detect if grid is a perfect repetition of a smaller tile
2. Try tile sizes from largest to smallest
3. Extract just the first tile from input/output
4. Synthesize on the tile task
5. Apply tile solution to all tiles

**Example**: A 9x9 grid is actually a 3x3 tile repeated 3x3 times.

```
Input: 3x3 tile repeated 3x3
┌─────────────────────────┐
│ A B C │ A B C │ A B C   │
│ D E F │ D E F │ D E F   │
│ G H I │ G H I │ G H I   │
├───────┼───────┼─────────┤
│ A B C │ A B C │ A B C   │
│ D E F │ D E F │ D E F   │
│ G H I │ G H I │ G H I   │
├───────┼───────┼─────────┤
│ A B C │ A B C │ A B C   │
│ D E F │ D E F │ D E F   │
│ G H I │ G H I │ G H I   │
└─────────────────────────┘

Solve just one tile, apply to all 9 tiles.
```

**Implementation**:
- `_detect_repeating_pattern(grid)`: Returns (tile_h, tile_w, tile_grid) or None
- `try_pattern_decomposition()`: Main strategy method
- `_extract_subgrid()`: Extract rectangular regions

---

### 5. Input-Output Size Ratio Decomposition (NEW)

**Idea**: If output is 2x or 3x the input dimensions, try upscaling/tiling.

**When to use**: Tasks with clear dimension scaling (output is 2x or 3x input).

**Algorithm**:
1. Detect size ratio: output dimensions / input dimensions
2. If ratio is 2x2 or 3x3:
   - Downscale output to match input dimensions
   - Synthesize on downscaled task
   - Wrap solution with upscaling operation

**Example**: Input 3x3 → Output 6x6 (2x scaling).

```
Input (3x3):          Output (6x6):
┌───┬───┬───┐        ┌───────┬───────┬───────┐
│ 1 │ 2 │ 3 │        │ 1 1 │ 2 2 │ 3 3 │
├───┼───┼───┤        │ 1 1 │───┼───┼───┤
│ 4 │ 5 │ 6 │   →    │ 4 4 │ 5 5 │ 6 6 │
├───┼───┼───┤        │ 4 4 │───┼───┼───┤
│ 7 │ 8 │ 9 │        │ 7 7 │ 8 8 │ 9 9 │
└───┴───┴───┘        │ 7 7 │───┴───┴───┘
```

**Ratios supported**:
- 2.0x2.0 (2x scale)
- 3.0x3.0 (3x scale)
- 0.5x0.5 (halving)
- 2.0x1.0, 1.0x2.0, etc. (asymmetric)

**Implementation**:
- `try_size_ratio_decomposition()`: Main strategy method
- Upscaling via pixel replication
- Downscaling via subsampling

---

### 6. Masking Decomposition (NEW)

**Idea**: Separate foreground (non-zero) and background, solve each independently.

**When to use**: Tasks where foreground and background transform independently.

**Algorithm**:
1. Identify background color (most common)
2. Create separate foreground and background tasks
3. Synthesize solutions for each independently
4. Merge: apply foreground solution where needed, background where needed

**Example**: Black background (0) with colored objects that transform.

```
Input:                Foreground task:       Background task:
┌─────┬─────┐        ┌─────┬─────┐         ┌─────┬─────┐
│ 0 1 │ 0 2 │        │ - 1 │ - 2 │         │ 0 - │ 0 - │
├─────┼─────┤   →    ├─────┼─────┤   +     ├─────┼─────┤
│ 0 0 │ 3 0 │        │ - - │ 3 - │         │ 0 0 │ - 0 │
└─────┴─────┘        └─────┴─────┘         └─────┴─────┘

Then merge with priority: use background solution
for background cells, foreground for others.
```

**Implementation**:
- `_separate_foreground_background()`: Extract mask and identify background
- `_merge_foreground_background()`: Reconstruct grid
- `try_masking_decomposition()`: Main strategy method

---

## Implementation Details

### Helper Functions

All new decomposition strategies use helper functions for modularity:

```python
# Pattern detection
_detect_repeating_pattern(grid: Grid) -> Optional[(tile_h, tile_w, tile)]

# Foreground/background separation
_separate_foreground_background(grid) -> (mask, bg_color)
_merge_foreground_background(fg_grid, bg_color) -> grid

# Bounding box extraction
_get_bounding_box(mask) -> Optional[(min_r, min_c, max_r, max_c)]

# Subgrid operations
_extract_subgrid(grid, r0, c0, r1, c1) -> grid
```

### Integration with DecompositionEngine

The engine tries strategies in order:
1. Cheapest/fastest first (pattern, size-ratio)
2. Then more expensive (masking, etc.)
3. Falls back gracefully if a strategy fails

Each strategy returns:
- A `Program` if successful (with fitness score)
- `None` if not applicable or failed

The `decompose_if_needed()` method only triggers decomposition if `best_score < 0.99`.

---

## Test Coverage

All new functionality has comprehensive unit tests:

### Helper Function Tests
- `TestPatternDetection`: 5 tests
  - 2x2 and 3x3 repeating patterns
  - Non-repeating and empty grids

- `TestForegroundBackground`: 4 tests
  - Foreground/background separation
  - Different background colors
  - Roundtrip merging

- `TestBoundingBox`: 5 tests
  - Single and multiple cells
  - Empty grids
  - Subgrid extraction

### Strategy Tests
- `TestPatternDecompositionStrategy`: 2 tests
- `TestSizeRatioDecomposition`: 3 tests
- `TestMaskingDecomposition`: 2 tests

**All 41 decomposition tests pass** (plus 604 other tests, totaling 645).

---

## Design Principles

1. **Modularity**: Each strategy is self-contained and independent
2. **Graceful Failure**: Strategies silently skip if not applicable
3. **Correct Merging**: Results must perfectly reconstruct when merged
4. **Type Safety**: Grid dimensions are validated before operations
5. **Performance**: Simpler patterns tried first; expensive operations last

---

## Usage Examples

### Pattern Decomposition
```python
# Task: 3x3 tile repeated 3x3 times
task = {
    'train': [
        {
            'input': [[1,2,3,1,2,3,1,2,3],
                      [4,5,6,4,5,6,4,5,6],
                      [7,8,9,7,8,9,7,8,9],
                      [1,2,3,1,2,3,1,2,3],
                      ...],  # 9x9 grid
            'output': [...],
        }
    ]
}

engine = DecompositionEngine()
result = engine.try_pattern_decomposition(task, synthesize_fn)
# Detects 3x3 pattern, solves just one tile
```

### Size-Ratio Decomposition
```python
# Task: 3x3 → 6x6 (2x scaling)
task = {
    'train': [
        {
            'input': [[1,2,3], [4,5,6], [7,8,9]],
            'output': [[1,1,2,2,3,3],
                      [1,1,2,2,3,3],
                      [4,4,5,5,6,6],
                      [4,4,5,5,6,6],
                      [7,7,8,8,9,9],
                      [7,7,8,8,9,9]],
        }
    ]
}

result = engine.try_size_ratio_decomposition(task, synthesize_fn)
# Detects 2x scaling, learns transform on original size
```

### Masking Decomposition
```python
# Task: Foreground objects transform, background is static
task = {
    'train': [
        {
            'input': [[0,1,0], [1,1,1], [0,1,0]],
            'output': [[0,2,0], [2,2,2], [0,2,0]],
        }
    ]
}

result = engine.try_masking_decomposition(task, synthesize_fn)
# Separates color change (1→2) from background (0)
```

---

## Performance Notes

- **Pattern detection**: O(n²) grid scans for each candidate tile size
- **Foreground/background**: O(n) for single pass identification
- **Size-ratio**: O(n) downscaling via subsampling
- **Masking**: Creates two separate tasks but search is often faster on simpler tasks

All strategies are only invoked when `best_score < 0.99`, so they don't impact already-solved tasks.

---

## Future Enhancements

Potential improvements for future iterations:

1. **Hierarchical patterns**: Detect patterns within patterns
2. **Adaptive downscaling**: Use smarter downscaling than subsampling
3. **Object tracking**: Track individual objects across transformations
4. **Symmetry detection**: Exploit rotational/reflective symmetry
5. **Color quantization**: Reduce color palette before solving

---

## References

- **Pillar 3 (Composability)**: Breaking complex problems into simpler subproblems
- **MDL Principle**: Prefer simpler explanations (fewer decomposition layers)
- **Observational Equivalence**: Only keep distinct output patterns, prune duplicates
