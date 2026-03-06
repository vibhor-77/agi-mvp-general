"""
Pillar 1: Feedback Loops — The Scoring Engine

The scorer provides the tight feedback loop that drives learning.
Instead of binary correct/incorrect, it provides continuous feedback
(Pillar 2: Approximability) — partial credit for getting some pixels right.

From Vibhor's framework: "Interaction is the only source of truth."
The environment (expected output grid) tells us how close we are.

Performance: NumPy is used for vectorized grid comparison when available,
giving a significant speedup on the M1 Max (no Python loops over pixels).
Pure-Python fallback is retained for portability.
"""
from __future__ import annotations
from typing import Optional
from .concepts import Grid

# NumPy is optional but strongly recommended for performance.
# On M1 Max with numpy, scoring is ~10-20x faster than pure Python.
try:
    import numpy as np
    _NUMPY_AVAILABLE = True
except ImportError:
    _NUMPY_AVAILABLE = False


def _to_array(grid: Grid):
    """Convert a list-of-lists grid to a uint8 NumPy array.

    Returns None if NumPy is not available.
    """
    if not _NUMPY_AVAILABLE:
        return None
    return np.array(grid, dtype=np.uint8)


def pixel_accuracy(predicted: Grid, expected: Grid) -> float:
    """Calculate fraction of pixels that match.

    This is the core approximability metric — it makes the fitness
    landscape smooth rather than binary, enabling gradient-free
    optimization to work.

    Returns:
        Float in [0, 1]. 1.0 = perfect match.
    """
    if not predicted or not expected:
        return 0.0

    pred_h = len(predicted)
    pred_w = len(predicted[0]) if predicted else 0
    exp_h = len(expected)
    exp_w = len(expected[0]) if expected else 0

    # Dimension mismatch — penalize but don't zero out
    if pred_h != exp_h or pred_w != exp_w:
        dim_score = 0.0
        if pred_h == exp_h:
            dim_score += 0.1
        if pred_w == exp_w:
            dim_score += 0.1
        return dim_score

    total = exp_h * exp_w
    if total == 0:
        return 1.0 if not predicted else 0.0

    if _NUMPY_AVAILABLE:
        # Vectorized comparison — avoids Python-level pixel loops
        p = np.array(predicted, dtype=np.uint8)
        e = np.array(expected, dtype=np.uint8)
        return float(np.sum(p == e)) / total

    # Pure-Python fallback
    matching = sum(
        1 for r in range(exp_h) for c in range(exp_w)
        if predicted[r][c] == expected[r][c]
    )
    return matching / total


def structural_similarity(predicted: Grid, expected: Grid) -> float:
    """A richer similarity score that captures structural features.

    Weighted composite of:
      0.60 — pixel accuracy (most informative signal)
      0.15 — dimension match
      0.15 — color palette overlap (Jaccard)
      0.10 — non-zero count similarity

    NumPy is used for all inner-loop work when available.
    """
    if not predicted or not expected:
        return 0.0

    pred_h = len(predicted)
    pred_w = len(predicted[0])
    exp_h = len(expected)
    exp_w = len(expected[0])

    if _NUMPY_AVAILABLE:
        # Convert once, reuse for all sub-scores.
        # dtype=np.int8 would fail for color 9; uint8 is safe (0-9).
        p = np.array(predicted, dtype=np.uint8)
        e = np.array(expected, dtype=np.uint8)

        # 1. Pixel accuracy
        if pred_h == exp_h and pred_w == exp_w:
            total = pred_h * pred_w
            pa = float(np.sum(p == e)) / total if total > 0 else 1.0
        else:
            dim_score = 0.0
            if pred_h == exp_h:
                dim_score += 0.1
            if pred_w == exp_w:
                dim_score += 0.1
            pa = dim_score

        # 2. Dimension match
        dim_match = 1.0 if (pred_h == exp_h and pred_w == exp_w) else 0.0

        # 3. Color palette similarity (Jaccard).
        # ARC uses colors 0-9 only → use a 10-element presence bitmask,
        # computed via np.bincount, which is faster than np.unique + set ops.
        p_flat = p.ravel()
        e_flat = e.ravel()
        p_counts = np.bincount(p_flat, minlength=10)
        e_counts = np.bincount(e_flat, minlength=10)
        # Colors 1-9 only (skip background=0)
        p_present = p_counts[1:] > 0   # bool array length 9
        e_present = e_counts[1:] > 0
        inter = int(np.sum(p_present & e_present))
        union = int(np.sum(p_present | e_present))
        if union > 0:
            color_overlap = inter / union
        else:
            color_overlap = 1.0 if not np.any(p_present) else 0.0

        # 4. Non-zero count similarity
        pred_nz = int(np.count_nonzero(p_flat))
        exp_nz  = int(np.count_nonzero(e_flat))
        max_nz  = max(pred_nz, exp_nz, 1)
        nz_sim  = 1.0 - abs(pred_nz - exp_nz) / max_nz

    else:
        # Pure-Python fallback
        pa = pixel_accuracy(predicted, expected)
        dim_match = 1.0 if (pred_h == exp_h and pred_w == exp_w) else 0.0

        pred_colors: set = set()
        exp_colors: set = set()
        for row in predicted:
            pred_colors.update(row)
        for row in expected:
            exp_colors.update(row)
        pred_colors.discard(0)
        exp_colors.discard(0)
        if exp_colors:
            union = len(pred_colors | exp_colors)
            color_overlap = len(pred_colors & exp_colors) / union if union > 0 else 1.0
        else:
            color_overlap = 1.0 if not pred_colors else 0.0

        pred_nz = sum(1 for row in predicted for c in row if c != 0)
        exp_nz  = sum(1 for row in expected  for c in row if c != 0)
        max_nz  = max(pred_nz, exp_nz, 1)
        nz_sim  = 1.0 - abs(pred_nz - exp_nz) / max_nz

    return 0.6 * pa + 0.15 * dim_match + 0.15 * color_overlap + 0.1 * nz_sim


def score_program_on_task(program, task: dict) -> float:
    """Score a program on an ARC task using training examples.

    This is the feedback loop: we apply the program to each training
    input and compare with expected output. The score tells us how
    close we are (approximability).

    Args:
        program: A Program instance with .execute(grid) method
        task: Dict with 'train' list of {'input': grid, 'output': grid}

    Returns:
        Average structural similarity across all training examples.
    """
    train_examples = task.get("train", [])
    if not train_examples:
        return 0.0

    total_score = 0.0
    for example in train_examples:
        predicted = program.execute(example["input"])
        if predicted is None:
            continue
        total_score += structural_similarity(predicted, example["output"])

    return total_score / len(train_examples)


def score_population_on_task(programs: list, task: dict) -> list[float]:
    """Score an entire population of programs on a task.

    Processes all programs and returns their scores. This avoids
    per-program overhead by amortizing the train-example iteration
    across the population.

    Args:
        programs: List of Program instances.
        task: ARC task dict.

    Returns:
        List of float scores, one per program (same order).
    """
    train_examples = task.get("train", [])
    if not train_examples:
        return [0.0] * len(programs)

    scores = []
    for program in programs:
        total = 0.0
        for example in train_examples:
            predicted = program.execute(example["input"])
            if predicted is not None:
                total += structural_similarity(predicted, example["output"])
        scores.append(total / len(train_examples))

    return scores


def validate_on_test(program, task: dict) -> tuple[bool, float]:
    """Validate a program on the test examples (held out).

    Returns:
        (exact_match, score) — whether all test outputs match exactly,
        and the average score.
    """
    test_examples = task.get("test", [])
    if not test_examples:
        return False, 0.0

    all_exact = True
    total_score = 0.0

    for example in test_examples:
        predicted = program.execute(example["input"])
        if predicted is None:
            all_exact = False
        else:
            score = pixel_accuracy(predicted, example["output"])
            total_score += score
            if score < 1.0:
                all_exact = False

    avg_score = total_score / len(test_examples)
    return all_exact, avg_score


def extract_task_features(task: dict) -> dict:
    """Extract structural features from a task for similarity matching.

    Used by Archive to find similar tasks for cross-task transfer.
    """
    features = {}
    train = task.get("train", [])
    if not train:
        return features

    # Input/output dimensions
    in_dims  = [(len(e["input"]),  len(e["input"][0]))  for e in train]
    out_dims = [(len(e["output"]), len(e["output"][0])) for e in train]

    features["same_dims"] = all(id_ == od for id_, od in zip(in_dims, out_dims))
    features["in_square"]  = all(h == w for h, w in in_dims)
    features["out_square"] = all(h == w for h, w in out_dims)

    if in_dims and out_dims:
        in_h,  in_w  = in_dims[0]
        out_h, out_w = out_dims[0]
        features["grows"]   = out_h > in_h or out_w > in_w
        features["shrinks"] = out_h < in_h or out_w < in_w
        features["h_ratio"] = out_h / max(in_h, 1)
        features["w_ratio"] = out_w / max(in_w, 1)

    # Color analysis
    in_colors: set  = set()
    out_colors: set = set()
    for e in train:
        for row in e["input"]:
            in_colors.update(row)
        for row in e["output"]:
            out_colors.update(row)

    features["in_colors"]   = len(in_colors)
    features["out_colors"]  = len(out_colors)
    features["new_colors"]  = len(out_colors - in_colors) > 0
    features["lost_colors"] = len(in_colors - out_colors) > 0
    features["num_examples"] = len(train)

    return features
