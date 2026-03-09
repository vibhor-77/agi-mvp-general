"""Bottom-Up Program Synthesis over the Grid DSL

Enumerates DSL expression trees bottom-up, pruned by execution on
training examples. The key idea: instead of searching over sequences
of 287 fixed primitives, we search over compositions of ~20 atomic
operations to synthesize novel transforms.

Algorithm:
  1. Start with leaf expressions (input_grid, color literals 0-9)
  2. Apply each DSL operation to compatible sub-expressions → depth+1
  3. Execute each candidate on training inputs
  4. Prune by observational equivalence (same outputs = keep simpler)
  5. Check for pixel-perfect match on all training examples
  6. If found, wrap as a Program and return

Optimizations:
  - Type-driven: only compose expressions where arg types match
  - Output hashing: detect observational equivalence via hash
  - Time budget: hard cap prevents runaway enumeration
  - Learned color map: extract from training examples as a shortcut

This module is deliberately simple — it prioritizes correctness and
clarity over exhaustive coverage. The DSL can be extended incrementally
as we identify new patterns that need coverage.
"""
from __future__ import annotations

import hashlib
import time
from typing import Optional

from .concepts import Grid, Concept, Program
from .dsl import DSLExpr, DSLType, DSLInterpreter, DSL_OPS
from .scorer import TaskCache


def synthesize_dsl_program(
    task: dict,
    cache: TaskCache,
    time_budget: float = 5.0,
    max_depth: int = 2,
) -> Optional[Program]:
    """Synthesize a Grid→Grid DSL program from training examples.

    Uses bottom-up enumeration with execution-guided pruning:
      1. Generate expressions of increasing depth
      2. Execute on training inputs to check correctness
      3. Prune observationally equivalent expressions
      4. Return first pixel-perfect program found

    Args:
        task: ARC task dict with 'train' key.
        cache: Pre-computed scoring cache.
        time_budget: Maximum seconds to spend searching.
        max_depth: Maximum expression tree depth to enumerate.

    Returns:
        A Program wrapping the DSL expression, or None if not found.
    """
    train = task.get("train", [])
    if not train:
        return None

    interp = DSLInterpreter()
    inputs = [ex["input"] for ex in train]
    outputs = [ex["output"] for ex in train]

    deadline = time.time() + time_budget

    # Phase 0a: Try learned color map shortcut
    result = _try_color_map_shortcut(inputs, outputs, interp, cache)
    if result is not None:
        return result

    # Phase 0b: Try learned neighbor rule shortcut
    result = _try_neighbor_rule_shortcut(inputs, outputs, interp, cache)
    if result is not None:
        return result

    # Phase 0c: Try dimension-change shortcuts (tiling, scaling, halving)
    result = _try_dimension_shortcuts(inputs, outputs, interp, cache)
    if result is not None:
        return result

    # Phase 1: Generate depth-0 expressions (leaves)
    leaves = _generate_leaves(inputs, outputs)

    # Phase 2: Bottom-up enumeration
    # For each depth, generate new expressions by applying ops to
    # existing sub-expressions, execute, and check.

    # Bank: maps output_hash → (expr, outputs_list)
    # Only keep the simplest expression per distinct output signature.
    bank: dict[str, tuple[DSLExpr, list]] = {}

    # Add leaves to bank
    for expr in leaves:
        if time.time() > deadline:
            break
        results = _execute_on_all(expr, inputs, interp)
        if results is not None:
            h = _hash_outputs(results)
            if h not in bank or expr.size < bank[h][0].size:
                bank[h] = (expr, results)

    # Check leaves for pixel-perfect match
    for h, (expr, results) in bank.items():
        if expr.return_type == DSLType.GRID:
            prog = _check_match(expr, results, outputs, interp, cache)
            if prog is not None:
                return prog

    # Enumerate depth 1, 2, ...
    for depth in range(1, max_depth + 1):
        if time.time() > deadline:
            break

        new_exprs = _enumerate_depth(bank, inputs, outputs, interp, depth)

        for expr, results in new_exprs:
            if time.time() > deadline:
                break

            h = _hash_outputs(results)
            if h not in bank or expr.size < bank[h][0].size:
                bank[h] = (expr, results)

            # Check for match
            if expr.return_type == DSLType.GRID:
                prog = _check_match(expr, results, outputs, interp, cache)
                if prog is not None:
                    return prog

    return None


def _generate_leaves(inputs: list[Grid],
                     outputs: list[Grid]) -> list[DSLExpr]:
    """Generate all depth-0 (leaf) expressions.

    Includes:
      - input_grid
      - Color literals (0-9) that appear in training examples
      - Learned color map literal (if consistent mapping exists)
    """
    leaves = [DSLExpr.input_grid()]

    # Collect all colors that appear in training I/O
    colors = set()
    for inp in inputs:
        for row in inp:
            for cell in row:
                colors.add(cell)
    for out in outputs:
        for row in out:
            for cell in row:
                colors.add(cell)

    for c in sorted(colors):
        leaves.append(DSLExpr.literal(c, DSLType.COLOR))

    # Try to learn a consistent color mapping from I/O pairs
    cmap = _learn_color_map(inputs, outputs)
    if cmap is not None:
        leaves.append(DSLExpr.literal(cmap, DSLType.COLOR_MAP))

    return leaves


def _learn_color_map(inputs: list[Grid],
                     outputs: list[Grid]) -> Optional[dict[int, int]]:
    """Learn a consistent pixel-level color mapping from training examples.

    If every pixel at position (r,c) maps from input[r][c] → output[r][c]
    consistently across all examples, returns the mapping dict.
    """
    # Only works for same-dims tasks
    for inp, out in zip(inputs, outputs):
        if len(inp) != len(out):
            return None
        for ri, ro in zip(inp, out):
            if len(ri) != len(ro):
                return None

    color_map: dict[int, int] = {}
    for inp, out in zip(inputs, outputs):
        for ri, ro in zip(inp, out):
            for ci, co in zip(ri, ro):
                if ci in color_map:
                    if color_map[ci] != co:
                        return None  # inconsistent
                color_map[ci] = co

    # Must actually change something
    if all(k == v for k, v in color_map.items()):
        return None

    return color_map


def _try_color_map_shortcut(
    inputs: list[Grid],
    outputs: list[Grid],
    interp: DSLInterpreter,
    cache: TaskCache,
) -> Optional[Program]:
    """Quick check: does a simple color map solve the task?"""
    cmap = _learn_color_map(inputs, outputs)
    if cmap is None:
        return None

    expr = DSLExpr.make_op(
        "apply_color_map",
        [DSLExpr.input_grid(), DSLExpr.literal(cmap, DSLType.COLOR_MAP)],
        DSLType.GRID,
    )

    results = _execute_on_all(expr, inputs, interp)
    if results is None:
        return None

    return _check_match(expr, results, outputs, interp, cache)


def _try_neighbor_rule_shortcut(
    inputs: list[Grid],
    outputs: list[Grid],
    interp: DSLInterpreter,
    cache: TaskCache,
) -> Optional[Program]:
    """Quick check: does a learned neighbor rule solve the task?

    Learns a mapping from (cell_color, n_nonbg_4_neighbors) → output_color
    from I/O pairs. If consistent across all examples, builds a DSL program.

    Includes LOOCV: for each training example, learn the rule from the
    other N-1 examples and verify it predicts the held-out example.
    This prevents overfitting to training-specific patterns.
    """
    # Only same-dims tasks
    for inp, out in zip(inputs, outputs):
        if len(inp) != len(out):
            return None
        for ri, ro in zip(inp, out):
            if len(ri) != len(ro):
                return None

    rule = _learn_neighbor_rule(inputs, outputs)
    if rule is None:
        return None

    # LOOCV generalization check: learn from N-1, verify on held-out
    if not _loocv_neighbor_rule(inputs, outputs, interp):
        return None

    expr = DSLExpr.make_op(
        "apply_neighbor_rule",
        [DSLExpr.input_grid(), DSLExpr.literal(rule, DSLType.COLOR_MAP)],
        DSLType.GRID,
    )

    results = _execute_on_all(expr, inputs, interp)
    if results is None:
        return None

    return _check_match(expr, results, outputs, interp, cache)


def _loocv_neighbor_rule(
    inputs: list[Grid],
    outputs: list[Grid],
    interp: DSLInterpreter,
) -> bool:
    """Leave-one-out cross-validation for neighbor rules.

    For each training example i:
      1. Learn a neighbor rule from all examples EXCEPT i
      2. Apply it to input[i]
      3. Check if it produces output[i]

    Returns True only if the rule generalizes across all held-out examples.
    For n=1 tasks, always returns True (can't cross-validate).
    """
    n = len(inputs)
    if n <= 1:
        return True

    for hold_out in range(n):
        # Learn from all except hold_out
        train_in = inputs[:hold_out] + inputs[hold_out + 1:]
        train_out = outputs[:hold_out] + outputs[hold_out + 1:]

        rule = _learn_neighbor_rule(train_in, train_out)
        if rule is None:
            return False  # Can't even learn a consistent rule from N-1

        # Apply to held-out input
        expr = DSLExpr.make_op(
            "apply_neighbor_rule",
            [DSLExpr.input_grid(),
             DSLExpr.literal(rule, DSLType.COLOR_MAP)],
            DSLType.GRID,
        )
        result = interp.evaluate(expr, inputs[hold_out])
        if result is None or result != outputs[hold_out]:
            return False

    return True


def _learn_neighbor_rule(
    inputs: list[Grid],
    outputs: list[Grid],
) -> Optional[dict[tuple[int, int], int]]:
    """Learn a (cell_color, n_nonbg_4) → output_color rule from I/O pairs.

    For each cell position across all training examples, extract:
      - The cell's current color
      - The number of non-background 4-connected neighbors
      - The expected output color

    If this mapping is consistent (same key → same value), return it.
    Only return rules that actually change something.
    """
    from collections import Counter

    rule: dict[tuple[int, int], int] = {}

    for inp, out in zip(inputs, outputs):
        h, w = len(inp), len(inp[0])
        # Background = most common color in input
        counts = Counter(cell for row in inp for cell in row)
        bg = counts.most_common(1)[0][0]

        for r in range(h):
            for c in range(w):
                cell = inp[r][c]
                expected = out[r][c]

                # Count non-bg 4-neighbors
                n4 = 0
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w and inp[nr][nc] != bg:
                        n4 += 1

                key = (cell, n4)
                if key in rule:
                    if rule[key] != expected:
                        return None  # inconsistent
                rule[key] = expected

    # Must actually change something
    has_change = any(key[0] != val for key, val in rule.items())
    if not has_change:
        return None

    # Filter to only include entries that change the color
    # (reduces rule size, prevents unnecessary rewrites)
    filtered = {k: v for k, v in rule.items() if k[0] != v}
    if not filtered:
        return None

    return filtered


def _try_dimension_shortcuts(
    inputs: list[Grid],
    outputs: list[Grid],
    interp: DSLInterpreter,
    cache: TaskCache,
) -> Optional[Program]:
    """Quick checks for common dimension-changing patterns.

    Tests tile_2x2, tile_3x3, scale_2x, scale_3x, grid halving,
    and border extraction without full enumeration.
    """
    if not inputs or not outputs:
        return None

    # Detect dimension ratios across all training examples
    ratios = set()
    for inp, out in zip(inputs, outputs):
        h_in, w_in = len(inp), len(inp[0])
        h_out, w_out = len(out), len(out[0])
        if h_in == 0 or w_in == 0:
            return None
        ratios.add((h_out / h_in, w_out / w_in))

    if len(ratios) != 1:
        return None  # inconsistent ratios across examples
    h_ratio, w_ratio = ratios.pop()

    # Try single atomic ops that match the dimension ratio
    candidates: list[str] = []

    if h_ratio == 2.0 and w_ratio == 2.0:
        candidates.extend(["tile_2x2", "scale_2x"])
    elif h_ratio == 3.0 and w_ratio == 3.0:
        candidates.extend(["tile_3x3", "scale_3x"])
    elif h_ratio == 0.5 and w_ratio == 1.0:
        # Top/bottom half, or boolean overlay of top+bottom halves
        candidates.extend(["get_top_half", "get_bottom_half",
                           "xor_halves_v", "or_halves_v", "and_halves_v"])
    elif h_ratio == 1.0 and w_ratio == 0.5:
        # Left/right half, or boolean overlay of left+right halves
        candidates.extend(["get_left_half", "get_right_half",
                           "xor_halves_h", "or_halves_h", "and_halves_h"])
    elif h_ratio == 0.5 and w_ratio == 0.5:
        # Both dimensions halved — try boolean overlays
        candidates.extend(["xor_halves_v", "or_halves_v", "and_halves_v",
                           "xor_halves_h", "or_halves_h", "and_halves_h"])

    # Also try crop_to_content for any dimension reduction
    if h_ratio <= 1.0 and w_ratio <= 1.0 and (h_ratio < 1.0 or w_ratio < 1.0):
        candidates.append("crop_to_content")

    for op_name in candidates:
        if op_name not in DSL_OPS:
            continue
        expr = DSLExpr.make_op(op_name, [DSLExpr.input_grid()], DSLType.GRID)
        results = _execute_on_all(expr, inputs, interp)
        if results is None:
            continue
        prog = _check_match(expr, results, outputs, interp, cache)
        if prog is not None:
            return prog

    # Try single op + color map: e.g., tile_2x2 then apply_color_map
    cmap = _learn_color_map(inputs, outputs)
    # For tiling/scaling with color adjustment, learn color map
    # on the transformed grid rather than the raw input
    for op_name in candidates:
        if op_name not in DSL_OPS:
            continue
        expr_step1 = DSLExpr.make_op(
            op_name, [DSLExpr.input_grid()], DSLType.GRID)
        step1_results = _execute_on_all(expr_step1, inputs, interp)
        if step1_results is None:
            continue

        # Learn color map from step1 output → expected output
        step_cmap = _learn_color_map(step1_results, outputs)
        if step_cmap is not None:
            expr = DSLExpr.make_op(
                "apply_color_map",
                [expr_step1, DSLExpr.literal(step_cmap, DSLType.COLOR_MAP)],
                DSLType.GRID,
            )
            results = _execute_on_all(expr, inputs, interp)
            if results is not None:
                prog = _check_match(expr, results, outputs, interp, cache)
                if prog is not None:
                    return prog

    return None


def _enumerate_depth(
    bank: dict[str, tuple[DSLExpr, list]],
    inputs: list[Grid],
    outputs: list[Grid],
    interp: DSLInterpreter,
    target_depth: int,
) -> list[tuple[DSLExpr, list]]:
    """Generate new expressions at the given depth.

    For each DSL operation, try all combinations of sub-expressions
    from the bank where types match and at least one arg is at
    depth == target_depth - 1.
    """
    new_exprs: list[tuple[DSLExpr, list]] = []

    # Collect bank expressions by type
    by_type: dict[DSLType, list[tuple[DSLExpr, list]]] = {}
    for h, (expr, results) in bank.items():
        t = expr.return_type
        if t not in by_type:
            by_type[t] = []
        by_type[t].append((expr, results))

    for op_name, (arg_types, ret_type) in DSL_OPS.items():
        if len(arg_types) == 1:
            # Unary operation
            t = arg_types[0]
            for expr, _ in by_type.get(t, []):
                if expr.depth != target_depth - 1:
                    continue
                new_expr = DSLExpr.make_op(op_name, [expr], ret_type)
                results = _execute_on_all(new_expr, inputs, interp)
                if results is not None:
                    new_exprs.append((new_expr, results))

        elif len(arg_types) == 2:
            # Binary operation
            t0, t1 = arg_types
            for e0, _ in by_type.get(t0, []):
                for e1, _ in by_type.get(t1, []):
                    # At least one arg should be at target depth - 1
                    if max(e0.depth, e1.depth) != target_depth - 1:
                        continue
                    new_expr = DSLExpr.make_op(op_name, [e0, e1], ret_type)
                    results = _execute_on_all(new_expr, inputs, interp)
                    if results is not None:
                        new_exprs.append((new_expr, results))

        elif len(arg_types) == 3:
            # Ternary operation (e.g., replace_color)
            t0, t1, t2 = arg_types
            for e0, _ in by_type.get(t0, []):
                for e1, _ in by_type.get(t1, []):
                    for e2, _ in by_type.get(t2, []):
                        if max(e0.depth, e1.depth, e2.depth) \
                                != target_depth - 1:
                            continue
                        new_expr = DSLExpr.make_op(
                            op_name, [e0, e1, e2], ret_type)
                        results = _execute_on_all(
                            new_expr, inputs, interp)
                        if results is not None:
                            new_exprs.append((new_expr, results))

    return new_exprs


def _execute_on_all(
    expr: DSLExpr,
    inputs: list[Grid],
    interp: DSLInterpreter,
) -> Optional[list]:
    """Execute an expression on all training inputs.

    Returns list of results, or None if any execution fails.
    """
    results = []
    for inp in inputs:
        result = interp.evaluate(expr, inp)
        if result is None:
            return None
        results.append(result)
    return results


def _check_match(
    expr: DSLExpr,
    results: list,
    outputs: list[Grid],
    interp: DSLInterpreter,
    cache: TaskCache,
) -> Optional[Program]:
    """Check if results match expected outputs pixel-perfectly.

    If they do, wrap the expression as a Program and return it.
    """
    # Quick check: do results match outputs exactly?
    if len(results) != len(outputs):
        return None

    for result, expected in zip(results, outputs):
        if not isinstance(result, list):
            return None
        if result != expected:
            return None

    # Wrap as Program
    def make_fn(e=expr, i=interp):
        def fn(grid: Grid) -> Grid:
            result = i.evaluate(e, grid)
            return result if result is not None else grid
        return fn

    program = Program(
        steps=[Concept(
            kind="dsl",
            name=f"dsl({expr})",
            implementation=make_fn(),
        )],
    )

    # Verify through the cache (belt-and-suspenders)
    score = cache.score_program(program)
    program.fitness = score

    if cache.is_pixel_perfect(program):
        return program

    return None


def _hash_outputs(results: list) -> str:
    """Hash a list of execution results for observational equivalence."""
    # Convert to a canonical string representation and hash it
    parts = []
    for r in results:
        if isinstance(r, list):
            parts.append(str(r))
        elif isinstance(r, (int, float, bool)):
            parts.append(str(r))
        elif isinstance(r, set):
            parts.append(str(sorted(r)))
        else:
            parts.append(str(r))
    combined = "|".join(parts)
    return hashlib.md5(combined.encode()).hexdigest()
