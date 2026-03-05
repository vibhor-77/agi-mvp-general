"""
Pillar 1: Feedback Loops — The Scoring Engine

The scorer provides the tight feedback loop that drives learning.
Instead of binary correct/incorrect, it provides continuous feedback
(Pillar 2: Approximability) — partial credit for getting some pixels right.

From Vibhor's framework: "Interaction is the only source of truth."
The environment (expected output grid) tells us how close we are.
"""
from __future__ import annotations
from typing import Optional
from .concepts import Grid


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

    pred_h, pred_w = len(predicted), len(predicted[0]) if predicted else 0
    exp_h, exp_w = len(expected), len(expected[0]) if expected else 0

    # Dimension mismatch is a strong signal — penalize but don't zero out
    if pred_h != exp_h or pred_w != exp_w:
        # Give partial credit for getting dimensions right on one axis
        dim_score = 0.0
        if pred_h == exp_h:
            dim_score += 0.1
        if pred_w == exp_w:
            dim_score += 0.1
        return dim_score

    total = exp_h * exp_w
    if total == 0:
        return 1.0 if not predicted else 0.0

    matching = sum(
        1 for r in range(exp_h) for c in range(exp_w)
        if predicted[r][c] == expected[r][c]
    )
    return matching / total


def structural_similarity(predicted: Grid, expected: Grid) -> float:
    """A richer similarity score that captures structural features.

    Beyond pixel matching, this checks:
    - Color palette overlap
    - Shape dimensions
    - Non-zero cell count similarity
    """
    if not predicted or not expected:
        return 0.0

    scores = []

    # 1. Pixel accuracy (weighted heavily)
    pa = pixel_accuracy(predicted, expected)
    scores.append(("pixel", pa, 0.6))

    # 2. Dimension match
    pred_h, pred_w = len(predicted), len(predicted[0])
    exp_h, exp_w = len(expected), len(expected[0])
    dim_match = 1.0 if (pred_h == exp_h and pred_w == exp_w) else 0.0
    scores.append(("dims", dim_match, 0.15))

    # 3. Color palette similarity
    pred_colors = set()
    exp_colors = set()
    for row in predicted:
        pred_colors.update(row)
    for row in expected:
        exp_colors.update(row)
    if exp_colors:
        color_overlap = len(pred_colors & exp_colors) / len(pred_colors | exp_colors)
    else:
        color_overlap = 1.0 if not pred_colors else 0.0
    scores.append(("colors", color_overlap, 0.15))

    # 4. Non-zero count similarity
    pred_nz = sum(1 for row in predicted for c in row if c != 0)
    exp_nz = sum(1 for row in expected for c in row if c != 0)
    max_nz = max(pred_nz, exp_nz, 1)
    nz_sim = 1.0 - abs(pred_nz - exp_nz) / max_nz
    scores.append(("nonzero", nz_sim, 0.1))

    # Weighted sum
    total = sum(score * weight for _, score, weight in scores)
    return total


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
        input_grid = example["input"]
        expected_output = example["output"]

        predicted = program.execute(input_grid)
        if predicted is None:
            total_score += 0.0
        else:
            total_score += structural_similarity(predicted, expected_output)

    return total_score / len(train_examples)


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
        input_grid = example["input"]
        expected_output = example["output"]

        predicted = program.execute(input_grid)
        if predicted is None:
            all_exact = False
            total_score += 0.0
        else:
            score = pixel_accuracy(predicted, expected_output)
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
    in_dims = [(len(e["input"]), len(e["input"][0])) for e in train]
    out_dims = [(len(e["output"]), len(e["output"][0])) for e in train]

    features["same_dims"] = all(id == od for id, od in zip(in_dims, out_dims))
    features["in_square"] = all(h == w for h, w in in_dims)
    features["out_square"] = all(h == w for h, w in out_dims)

    # Size relationships
    if in_dims and out_dims:
        in_h, in_w = in_dims[0]
        out_h, out_w = out_dims[0]
        features["grows"] = out_h > in_h or out_w > in_w
        features["shrinks"] = out_h < in_h or out_w < in_w
        features["h_ratio"] = out_h / max(in_h, 1)
        features["w_ratio"] = out_w / max(in_w, 1)

    # Color analysis
    in_colors = set()
    out_colors = set()
    for e in train:
        for row in e["input"]:
            in_colors.update(row)
        for row in e["output"]:
            out_colors.update(row)

    features["in_colors"] = len(in_colors)
    features["out_colors"] = len(out_colors)
    features["new_colors"] = len(out_colors - in_colors) > 0
    features["lost_colors"] = len(in_colors - out_colors) > 0
    features["num_examples"] = len(train)

    return features
