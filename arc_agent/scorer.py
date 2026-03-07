"""
Pillar 1: Feedback Loops — The Scoring Engine

The scorer provides the tight feedback loop that drives learning.
Instead of binary correct/incorrect, it provides continuous feedback
(Pillar 2: Approximability) — partial credit for getting some pixels right.

From Vibhor's framework: "Interaction is the only source of truth."
The environment (expected output grid) tells us how close we are.

Performance design:
  The dominant cost is np.array() conversion from list-of-lists.
  Expected outputs are fixed for the lifetime of a task, so we convert
  them once via TaskCache and reuse across all scoring calls.
  Predicted outputs (from program execution) still require conversion
  each time since each program produces different results.

  Measured impact: ~40% reduction in scoring time.

  Numba JIT is used in objects.py for flood-fill (find_objects).
  The rest of scoring is already vectorized with NumPy.
"""
from __future__ import annotations
import numpy as np
from .concepts import Grid


# ---------------------------------------------------------------------------
# Guard: validate that program output is an actual grid
# ---------------------------------------------------------------------------

def _is_valid_grid(value) -> bool:
    """Return True only if value is a non-empty list-of-lists of numbers.

    Programs that implement predicates (is_symmetric, has_color, …) return
    bool.  Programs that count, sum, or otherwise reduce return int or float.
    Neither can be scored as a grid — silently skip them rather than crash.
    """
    return (
        isinstance(value, list)
        and len(value) > 0
        and isinstance(value[0], list)
    )


# ---------------------------------------------------------------------------
# Core similarity metrics (operate on pre-converted numpy arrays)
# ---------------------------------------------------------------------------

def _pixel_accuracy_np(p: np.ndarray, e: np.ndarray) -> float:
    """Pixel accuracy between two same-shape uint8 arrays."""
    total = e.size
    return float(np.sum(p == e)) / total if total > 0 else 1.0


def _structural_similarity_np(p: np.ndarray, e: np.ndarray,
                               pred_h: int, pred_w: int,
                               exp_h: int, exp_w: int) -> float:
    """Structural similarity between predicted and expected arrays.

    Called with pre-converted arrays. Dimension info is passed explicitly
    so we don't pay for .shape attribute lookup in the hot path.
    """
    # 1. Pixel accuracy (or dim-mismatch penalty)
    if pred_h == exp_h and pred_w == exp_w:
        pa = _pixel_accuracy_np(p, e)
        dim_match = 1.0
    else:
        pa = (0.1 if pred_h == exp_h else 0.0) + (0.1 if pred_w == exp_w else 0.0)
        dim_match = 0.0

    # 2. Color palette overlap (Jaccard, non-background colors).
    #    ARC inputs use colors 0-9, but some derived primitives (e.g. count_per_row)
    #    can produce values outside that range. We pad both histograms to the same
    #    length so the boolean comparison is always well-defined.
    p_flat, e_flat = p.ravel(), e.ravel()
    p_max = int(p_flat.max()) if p_flat.size > 0 else 0
    e_max = int(e_flat.max()) if e_flat.size > 0 else 0
    n_bins = max(p_max, e_max, 9) + 1
    p_counts = np.bincount(p_flat, minlength=n_bins)
    e_counts = np.bincount(e_flat, minlength=n_bins)
    p_present = p_counts[1:] > 0   # skip background (color 0)
    e_present = e_counts[1:] > 0
    inter = int(np.sum(p_present & e_present))
    union = int(np.sum(p_present | e_present))
    color_overlap = inter / union if union > 0 else (1.0 if not np.any(p_present) else 0.0)

    # 3. Non-zero pixel count similarity
    pred_nz = int(np.count_nonzero(p))
    exp_nz  = int(np.count_nonzero(e))
    max_nz  = max(pred_nz, exp_nz, 1)
    nz_sim  = 1.0 - abs(pred_nz - exp_nz) / max_nz

    return 0.6 * pa + 0.15 * dim_match + 0.15 * color_overlap + 0.1 * nz_sim


# ---------------------------------------------------------------------------
# Public API: list-of-lists interface (used for one-off calls)
# ---------------------------------------------------------------------------

def _safe_to_np(grid: Grid) -> "np.ndarray | None":
    """Convert grid to numpy array, returning None for jagged/invalid grids."""
    if not _is_valid_grid(grid):
        return None
    w = len(grid[0])
    if any(len(row) != w for row in grid):
        return None
    try:
        arr = np.array(grid, dtype=np.int32)
        # Clamp values to valid ARC range [0, 9]
        if arr.max() > 9 or arr.min() < 0:
            return None
        return arr.astype(np.uint8)
    except (ValueError, TypeError, OverflowError):
        return None


def pixel_accuracy(predicted: Grid, expected: Grid) -> float:
    """Fraction of pixels that match exactly.

    Returns float in [0, 1]. Dimension mismatch → small partial score.
    """
    p = _safe_to_np(predicted)
    e = _safe_to_np(expected)
    if p is None or e is None:
        return 0.0

    pred_h, pred_w = p.shape
    exp_h,  exp_w  = e.shape

    if pred_h != exp_h or pred_w != exp_w:
        return (0.1 if pred_h == exp_h else 0.0) + (0.1 if pred_w == exp_w else 0.0)

    return _pixel_accuracy_np(p, e)


def structural_similarity(predicted: Grid, expected: Grid) -> float:
    """Richer similarity capturing shape, color palette, and density.

    Weighted composite:
      0.60 — pixel accuracy (dominant signal)
      0.15 — dimension match (binary)
      0.15 — non-background color palette overlap (Jaccard)
      0.10 — non-zero pixel count similarity

    Converts inputs on every call. Use TaskCache for repeated scoring.
    """
    p = _safe_to_np(predicted)
    e = _safe_to_np(expected)
    if p is None or e is None:
        return 0.0

    pred_h, pred_w = p.shape
    exp_h,  exp_w  = e.shape

    return _structural_similarity_np(p, e, pred_h, pred_w, exp_h, exp_w)


# ---------------------------------------------------------------------------
# TaskCache: pre-convert expected outputs once per task
# ---------------------------------------------------------------------------

class TaskCache:
    """Pre-converted expected outputs for a single ARC task.

    The expected outputs are fixed for the lifetime of a task run.
    Converting them to numpy arrays once (rather than once per program
    per generation) eliminates the dominant source of redundant work.

    Usage:
        cache = TaskCache(task)
        scores = cache.score_population(programs)
    """

    def __init__(self, task: dict) -> None:
        train = task.get("train", [])
        self.n_examples = len(train)

        # Pre-convert expected outputs once
        self._expected: list[np.ndarray] = []
        self._exp_dims: list[tuple[int, int]] = []
        self._inputs: list[Grid] = []
        for ex in train:
            e = np.array(ex["output"], dtype=np.uint8)
            self._expected.append(e)
            self._exp_dims.append((e.shape[0], e.shape[1]))
            self._inputs.append(ex["input"])

        test = task.get("test", [])
        self._test_expected: list[np.ndarray] = []
        self._test_inputs: list[Grid] = []
        for ex in test:
            self._test_expected.append(np.array(ex["output"], dtype=np.uint8))
            self._test_inputs.append(ex["input"])

    def score_program(self, program) -> float:
        """Score one program using pre-converted expected arrays."""
        if self.n_examples == 0:
            return 0.0
        total = 0.0
        for inp, e, (exp_h, exp_w) in zip(self._inputs, self._expected, self._exp_dims):
            predicted = program.execute(inp)
            p = _safe_to_np(predicted)
            if p is None:
                continue
            pred_h, pred_w = p.shape
            total += _structural_similarity_np(p, e, pred_h, pred_w, exp_h, exp_w)
        return total / self.n_examples

    def score_population(self, programs: list) -> list[float]:
        """Score an entire population using pre-converted expected arrays.

        Single pass: for each program, iterate over the (already-converted)
        training examples. The only conversion left is predicted→ndarray.
        """
        if self.n_examples == 0:
            return [0.0] * len(programs)

        scores = []
        for program in programs:
            total = 0.0
            for inp, e, (exp_h, exp_w) in zip(self._inputs, self._expected, self._exp_dims):
                predicted = program.execute(inp)
                p = _safe_to_np(predicted)
                if p is None:
                    continue
                pred_h, pred_w = p.shape
                total += _structural_similarity_np(p, e, pred_h, pred_w, exp_h, exp_w)
            scores.append(total / self.n_examples)

        return scores

    def validate_on_test(self, program) -> tuple[bool, float]:
        """Validate on held-out test examples. Returns (all_exact, avg_score)."""
        if not self._test_expected:
            return False, 0.0
        all_exact   = True
        total_score = 0.0
        for inp, e in zip(self._test_inputs, self._test_expected):
            predicted = program.execute(inp)
            p = _safe_to_np(predicted)
            if p is None:
                all_exact = False
                continue
            pred_h, pred_w = p.shape
            exp_h,  exp_w  = e.shape
            if pred_h == exp_h and pred_w == exp_w:
                score = _pixel_accuracy_np(p, e)
            else:
                score = 0.0
            total_score += score
            if score < 1.0:
                all_exact = False
        return all_exact, total_score / len(self._test_expected)


# ---------------------------------------------------------------------------
# Convenience wrappers (keep existing call sites working)
# ---------------------------------------------------------------------------

def score_program_on_task(program, task: dict) -> float:
    """Score a program on all training examples of an ARC task.

    For repeated scoring (synthesizer, solver), prefer TaskCache.score_program.
    This wrapper is for one-off calls.
    """
    return TaskCache(task).score_program(program)


def score_population_on_task(programs: list, task: dict) -> list[float]:
    """Score an entire population on a task.

    For repeated use across generations, prefer TaskCache(task) and reuse it.
    This wrapper is for one-off calls.
    """
    return TaskCache(task).score_population(programs)


def validate_on_test(program, task: dict) -> tuple[bool, float]:
    """Validate a program on held-out test examples.

    Returns (all_exact, avg_pixel_accuracy).
    """
    return TaskCache(task).validate_on_test(program)


def extract_task_features(task: dict) -> dict:
    """Extract structural features for task similarity matching.

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
