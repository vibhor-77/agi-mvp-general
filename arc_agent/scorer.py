"""
Pillar 1: Feedback Loops — The Scoring Engine

The scorer provides the tight feedback loop that drives learning.
Instead of binary correct/incorrect, it provides continuous feedback
(Pillar 2: Approximability) — partial credit for getting some pixels right.

From Vibhor's framework: "Interaction is the only source of truth."
The environment (expected output grid) tells us how close we are.

Implementation note: NumPy is a required dependency (see requirements.txt).
All scoring is vectorized: no Python-level pixel loops.
"""
from __future__ import annotations
import numpy as np
from .concepts import Grid


def pixel_accuracy(predicted: Grid, expected: Grid) -> float:
    """Fraction of pixels that match exactly.

    This is the core approximability metric — it makes the fitness
    landscape smooth rather than binary, enabling gradient-free
    optimization to work.

    Returns float in [0, 1]. 1.0 = perfect match.
    Dimension mismatch gives a small partial score (0.1 per matching axis).
    """
    if not predicted or not expected:
        return 0.0

    pred_h, pred_w = len(predicted), len(predicted[0])
    exp_h,  exp_w  = len(expected),  len(expected[0])

    if pred_h != exp_h or pred_w != exp_w:
        # Penalize dimension mismatch, but don't zero out
        return (0.1 if pred_h == exp_h else 0.0) + (0.1 if pred_w == exp_w else 0.0)

    total = exp_h * exp_w
    if total == 0:
        return 1.0

    p = np.array(predicted, dtype=np.uint8)
    e = np.array(expected,   dtype=np.uint8)
    return float(np.sum(p == e)) / total


def structural_similarity(predicted: Grid, expected: Grid) -> float:
    """Richer similarity that captures shape, color palette, and density.

    Weighted composite:
      0.60 — pixel accuracy (dominant signal)
      0.15 — dimension match (binary reward)
      0.15 — non-zero color palette overlap (Jaccard, colors 1-9)
      0.10 — non-zero pixel count similarity

    ARC uses only 10 colors (0-9), so color analysis uses np.bincount
    with minlength=10 rather than np.unique — O(n) with minimal overhead.
    """
    if not predicted or not expected:
        return 0.0

    pred_h, pred_w = len(predicted), len(predicted[0])
    exp_h,  exp_w  = len(expected),  len(expected[0])

    p = np.array(predicted, dtype=np.uint8)
    e = np.array(expected,   dtype=np.uint8)

    # 1. Pixel accuracy
    if pred_h == exp_h and pred_w == exp_w:
        total = pred_h * pred_w
        pa = float(np.sum(p == e)) / total if total > 0 else 1.0
    else:
        pa = (0.1 if pred_h == exp_h else 0.0) + (0.1 if pred_w == exp_w else 0.0)

    # 2. Dimension match
    dim_match = 1.0 if (pred_h == exp_h and pred_w == exp_w) else 0.0

    # 3. Color palette overlap (Jaccard, non-background colors only)
    p_counts = np.bincount(p.ravel(), minlength=10)
    e_counts = np.bincount(e.ravel(), minlength=10)
    p_present = p_counts[1:] > 0   # bool[9], colors 1-9
    e_present = e_counts[1:] > 0
    inter = int(np.sum(p_present & e_present))
    union = int(np.sum(p_present | e_present))
    color_overlap = inter / union if union > 0 else (1.0 if not np.any(p_present) else 0.0)

    # 4. Non-zero count similarity
    pred_nz = int(np.count_nonzero(p))
    exp_nz  = int(np.count_nonzero(e))
    max_nz  = max(pred_nz, exp_nz, 1)
    nz_sim  = 1.0 - abs(pred_nz - exp_nz) / max_nz

    return 0.6 * pa + 0.15 * dim_match + 0.15 * color_overlap + 0.1 * nz_sim


def score_program_on_task(program, task: dict) -> float:
    """Score a program on all training examples of an ARC task.

    This is the feedback loop: we apply the program to each training
    input and compare with expected output. The score tells us how
    close we are (approximability).

    Returns average structural similarity in [0, 1] across training examples.
    """
    train_examples = task.get("train", [])
    if not train_examples:
        return 0.0

    total_score = 0.0
    for example in train_examples:
        predicted = program.execute(example["input"])
        if predicted is not None:
            total_score += structural_similarity(predicted, example["output"])

    return total_score / len(train_examples)


def score_population_on_task(programs: list, task: dict) -> list[float]:
    """Score an entire evolutionary population on a task in one pass.

    Amortizes the train-example loop across the whole population,
    which is more cache-friendly than calling score_program_on_task
    individually for each program.

    Returns list of float scores in the same order as `programs`.
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
    """Validate a program on held-out test examples.

    Returns:
        (all_exact, avg_score) — whether every test output matched exactly,
        and the average pixel accuracy across test examples.
    """
    test_examples = task.get("test", [])
    if not test_examples:
        return False, 0.0

    all_exact   = True
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

    return all_exact, total_score / len(test_examples)


def extract_task_features(task: dict) -> dict:
    """Extract structural features from a task for similarity matching.

    Used by Archive to find similar past tasks for cross-task transfer
    (Pillar 4: Exploration — exploit what we've learned before).
    """
    features: dict = {}
    train = task.get("train", [])
    if not train:
        return features

    in_dims  = [(len(e["input"]),  len(e["input"][0]))  for e in train]
    out_dims = [(len(e["output"]), len(e["output"][0])) for e in train]

    features["same_dims"]  = all(id_ == od for id_, od in zip(in_dims, out_dims))
    features["in_square"]  = all(h == w for h, w in in_dims)
    features["out_square"] = all(h == w for h, w in out_dims)

    if in_dims and out_dims:
        in_h,  in_w  = in_dims[0]
        out_h, out_w = out_dims[0]
        features["grows"]   = out_h > in_h or out_w > in_w
        features["shrinks"] = out_h < in_h or out_w < in_w
        features["h_ratio"] = out_h / max(in_h, 1)
        features["w_ratio"] = out_w / max(in_w, 1)

    in_colors:  set = set()
    out_colors: set = set()
    for ex in train:
        for row in ex["input"]:
            in_colors.update(row)
        for row in ex["output"]:
            out_colors.update(row)

    features["in_colors"]    = len(in_colors)
    features["out_colors"]   = len(out_colors)
    features["new_colors"]   = len(out_colors - in_colors) > 0
    features["lost_colors"]  = len(in_colors - out_colors) > 0
    features["num_examples"] = len(train)

    return features
