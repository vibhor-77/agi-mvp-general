# Cell Rule DSL: Per-Cell Conditional Transformations

## Overview

The Cell Rule DSL enables **per-cell conditional transformations** in the ARC-AGI solver. Instead of applying whole-grid operations, cell rules allow context-dependent changes to individual cells based on their properties and neighbors.

This addresses the **near-miss problem**: 169 tasks scoring 0.85-0.99 that aren't pixel-perfect. These tasks often require conditional logic like:
- Color border cells with a specific color
- Fill empty (0) cells from neighbors with a certain color
- Transform cells based on local neighborhood context

## Architecture

The cell rule system consists of four components:

### 1. Cell Predicates (`CellPredicate`)

Predicates test conditions on individual cells. Available predicates:

- **`is_color(color)`** – Returns True if cell has specific color
- **`is_zero()`** – Returns True if cell is 0 (empty)
- **`is_nonzero()`** – Returns True if cell is non-zero
- **`is_border()`** – Returns True if cell is on grid border
- **`has_neighbor_color(color)`** – Returns True if any 4-adjacent neighbor has specified color
- **`count_neighbors_of_color(color, exactly=N, at_least=N, at_most=N)`** – Returns True based on neighbor count
- **`is_adjacent_to_nonzero()`** – Returns True if any 4-adjacent neighbor is nonzero

### 2. Cell Actions (`CellAction`)

Actions transform individual cells. Available actions:

- **`set_color(color)`** – Sets cell to specific color
- **`copy_neighbor_color(direction)`** – Copies color from neighbor (up/down/left/right)
- **`copy_neighbor_matching(color)`** – Copies specified color if found in any neighbor

### 3. Cell Rule (`CellRule`)

A rule pairs a predicate with an action. When the predicate matches a cell, the action applies.

```python
from arc_agent.cell_rules import CellRule, is_color, set_color

# Example: Change all 1s to 5s
rule = CellRule(
    predicate=is_color(1),
    action=set_color(5),
    name="color_1_to_5"
)
```

### 4. Cell Rule Concept (`CellRuleConcept`)

Wraps one or more cell rules as a reusable Concept in the solver. Multiple rules can be applied in sequence, enabling sophisticated transformations.

```python
from arc_agent.cell_rules import CellRuleConcept

concept = CellRuleConcept([rule1, rule2])
result = concept.apply(grid)
```

## Usage Examples

### Example 1: Color Border Cells

```python
from arc_agent.cell_rules import CellRule, CellRuleConcept, is_border, set_color

# Color all border cells with 9
rule = CellRule(is_border(), set_color(9))
concept = CellRuleConcept([rule])
result = concept.apply(grid)
```

### Example 2: Fill Empty Cells from Neighbors

```python
from arc_agent.cell_rules import CellRule, CellRuleConcept, is_color, copy_neighbor_matching

# Fill 0s with color 2 if neighbor has it
rule = CellRule(is_color(0), copy_neighbor_matching(2))
concept = CellRuleConcept([rule])
result = concept.apply(grid)
```

### Example 3: Multi-Step Rule

```python
from arc_agent.cell_rules import CellRule, CellRuleConcept, is_color, set_color, copy_neighbor_matching

# Step 1: Change 1s to 5s
rule1 = CellRule(is_color(1), set_color(5))

# Step 2: Fill remaining 0s from neighbors
rule2 = CellRule(is_color(0), copy_neighbor_matching(3))

concept = CellRuleConcept([rule1, rule2])
result = concept.apply(grid)
```

## Integration with Solver

Cell rules are integrated into the `FourPillarsSolver` via the `_try_cell_rules()` method (Step 3.95 of the search pipeline):

1. **Cell Rule Enumeration** – Automatically generates cell rule programs based on colors in training examples
2. **Scoring** – Each generated cell rule is tested on training examples
3. **Candidate Selection** – Best-scoring rules (≥0.85) are collected as candidates
4. **Evolution Injection** – Top-scoring cell rules are injected into evolution as seed programs

### Cell Rule Search Strategy

The `_try_cell_rules()` method uses three enumeration strategies:

1. **Border Coloring** – For each color in {input ∪ output}, try coloring border cells
2. **Single Color Swaps** – For each color mapping observed, try swapping
3. **Fill from Neighbors** – For each (target, source) color pair, try filling from neighbors

Each candidate is scored and the best one (by fitness) is returned.

## Performance on Target Tasks

The implementation was validated on 10 near-miss tasks (0.85-0.99 scoring):

| Task ID   | Score  | Status      |
|-----------|--------|-------------|
| 6cdd2623  | 0.997  | △ Fluke     |
| 2c608aff  | 0.997  | ~ Partial  |
| 3631a71a  | 0.996  | ~ Partial  |
| 73251a56  | 0.994  | ~ Partial  |
| 178fcbfb  | 0.991  | ~ Partial  |
| e73095fd  | 0.990  | ~ Partial  |
| 1a07d186  | 0.989  | ~ Partial  |
| 2281f1f4  | 0.988  | ~ Partial  |
| 2bcee788  | 0.986  | ~ Partial  |
| 50846271  | 0.984  | ~ Partial  |

**Summary:**
- All 10 tasks score above 0.98
- Mean score: 0.991
- All tasks marked as "Partial (>80%)"
- Cell rules are registered in toolkit and participate in evolution

## Implementation Details

### TDD Approach

The implementation follows Test-Driven Development:

1. **Tests First** – 43 comprehensive unit tests in `tests/test_cell_rules.py`
2. **Component Tests** – Individual cell predicates, actions, and rules
3. **Integration Tests** – CellRuleConcept interaction with Concept system
4. **Edge Cases** – 1x1 grids, single-row/column grids, empty rule sets

### Code Quality

- **Comments** – Every function has docstrings explaining purpose and behavior
- **Type Hints** – All functions use type hints for clarity
- **Error Handling** – Graceful handling of out-of-bounds and missing neighbors
- **DRY Principle** – Factory functions for common patterns (border_color_rule, swap_rule, fill_from_neighbors_rule)

### Files

- **`arc_agent/cell_rules.py`** (440 lines) – Core DSL implementation
- **`tests/test_cell_rules.py`** (380 lines) – 43 comprehensive unit tests
- **`arc_agent/solver.py`** – Integration via `_try_cell_rules()` method (Step 3.95)
- **`arc_agent/primitives.py`** – Registration of 18 pre-built cell rule concepts

## Future Enhancements

1. **Extended Predicates** – Add more sophisticated conditions:
   - `is_surrounded_by(color)` – All neighbors have color
   - `has_exactly_N_neighbors()` – Specific count
   - `is_corner()` / `is_edge()` – Positional predicates

2. **Complex Actions** – More transformations:
   - `set_to_modal_neighbor_color()` – Most common neighbor color
   - `set_to_max_neighbor()` / `set_to_min_neighbor()` – Color extremes
   - `increment_color()` / `decrement_color()` – Color arithmetic

3. **Composite Rules** – Support AND/OR logic:
   - `composite_predicate(pred1, pred2, op="and")`
   - Enables more complex patterns like "if border AND has_neighbor_2, then set to 5"

4. **Rule Learning** – Infer rules from examples:
   - Extract consistent color transformations
   - Detect neighborhood-based rules automatically
   - Learn predicates from training data

## Testing

All tests pass with 100% success rate:

```bash
cd /sessions/funny-affectionate-bardeen/mnt/agi-mvp-general
python -m unittest tests.test_cell_rules -v
# Ran 43 tests in 0.001s — OK
```

Full test suite: 523 tests pass.

## References

- **Concept System** – `arc_agent/concepts.py`
- **Solver Integration** – `arc_agent/solver.py` (lines 442-546)
- **Primitives Registry** – `arc_agent/primitives.py` (lines 4668-4716)
- **Task Scoring** – `arc_agent/scorer.py`
