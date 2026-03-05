"""Unit tests for the Scoring Engine (Pillar 1: Feedback Loops)."""
import pytest
from arc_agent.scorer import (
    pixel_accuracy, structural_similarity,
    score_program_on_task, validate_on_test,
    extract_task_features,
)
from arc_agent.concepts import Concept, Program


class TestPixelAccuracy:
    def test_perfect_match(self):
        grid = [[1, 2], [3, 4]]
        assert pixel_accuracy(grid, grid) == 1.0

    def test_no_match(self):
        pred = [[1, 1], [1, 1]]
        exp = [[2, 2], [2, 2]]
        assert pixel_accuracy(pred, exp) == 0.0

    def test_partial_match(self):
        pred = [[1, 2], [3, 4]]
        exp = [[1, 2], [3, 5]]
        assert pixel_accuracy(pred, exp) == 0.75

    def test_dimension_mismatch(self):
        pred = [[1, 2, 3]]
        exp = [[1, 2]]
        score = pixel_accuracy(pred, exp)
        # Should be low but not zero (partial credit for h match)
        assert 0.0 <= score <= 0.2

    def test_empty_grids(self):
        assert pixel_accuracy([], []) == 0.0
        assert pixel_accuracy([[1]], []) == 0.0

    def test_single_cell(self):
        assert pixel_accuracy([[5]], [[5]]) == 1.0
        assert pixel_accuracy([[5]], [[3]]) == 0.0


class TestStructuralSimilarity:
    def test_perfect_match(self):
        grid = [[1, 2], [3, 4]]
        score = structural_similarity(grid, grid)
        assert score >= 0.99

    def test_completely_wrong(self):
        pred = [[9, 9, 9]]
        exp = [[1, 2]]
        score = structural_similarity(pred, exp)
        assert score < 0.3

    def test_right_dims_wrong_colors(self):
        pred = [[9, 9], [9, 9]]
        exp = [[1, 2], [3, 4]]
        score = structural_similarity(pred, exp)
        # Should get some credit for right dimensions
        assert score > 0.0
        assert score < 0.5

    def test_score_in_range(self):
        pred = [[1, 0], [0, 1]]
        exp = [[1, 1], [1, 1]]
        score = structural_similarity(pred, exp)
        assert 0.0 <= score <= 1.0


class TestScoreProgramOnTask:
    def test_identity_on_identity_task(self):
        """Identity program should score 1.0 on a task where output == input."""
        task = {
            "train": [
                {"input": [[1, 2], [3, 4]], "output": [[1, 2], [3, 4]]},
                {"input": [[5, 6]], "output": [[5, 6]]},
            ]
        }
        identity = Concept(kind="operator", name="id", implementation=lambda g: g)
        program = Program([identity])
        score = score_program_on_task(program, task)
        assert score >= 0.99

    def test_wrong_program_low_score(self):
        task = {
            "train": [
                {"input": [[1, 2], [3, 4]], "output": [[4, 3], [2, 1]]},
            ]
        }
        identity = Concept(kind="operator", name="id", implementation=lambda g: g)
        program = Program([identity])
        score = score_program_on_task(program, task)
        assert score < 0.5

    def test_empty_task(self):
        task = {"train": []}
        identity = Concept(kind="operator", name="id", implementation=lambda g: g)
        program = Program([identity])
        score = score_program_on_task(program, task)
        assert score == 0.0


class TestValidateOnTest:
    def test_correct_validation(self):
        task = {
            "test": [
                {"input": [[1, 2]], "output": [[1, 2]]},
            ]
        }
        identity = Concept(kind="operator", name="id", implementation=lambda g: g)
        program = Program([identity])
        exact, score = validate_on_test(program, task)
        assert exact is True
        assert score == 1.0

    def test_incorrect_validation(self):
        task = {
            "test": [
                {"input": [[1, 2]], "output": [[3, 4]]},
            ]
        }
        identity = Concept(kind="operator", name="id", implementation=lambda g: g)
        program = Program([identity])
        exact, score = validate_on_test(program, task)
        assert exact is False


class TestExtractTaskFeatures:
    def test_same_dims_detection(self):
        task = {
            "train": [
                {"input": [[1, 2], [3, 4]], "output": [[5, 6], [7, 8]]},
            ]
        }
        features = extract_task_features(task)
        assert features["same_dims"] is True
        assert features["grows"] is False
        assert features["shrinks"] is False

    def test_growing_detection(self):
        task = {
            "train": [
                {"input": [[1]], "output": [[1, 1], [1, 1]]},
            ]
        }
        features = extract_task_features(task)
        assert features["grows"] is True
        assert features["shrinks"] is False

    def test_shrinking_detection(self):
        task = {
            "train": [
                {"input": [[1, 2, 3], [4, 5, 6]], "output": [[1]]},
            ]
        }
        features = extract_task_features(task)
        assert features["shrinks"] is True

    def test_color_analysis(self):
        task = {
            "train": [
                {"input": [[1, 2, 0]], "output": [[3, 4, 0]]},
            ]
        }
        features = extract_task_features(task)
        assert features["new_colors"] is True  # 3, 4 not in input
        assert features["lost_colors"] is True  # 1, 2 not in output

    def test_empty_task(self):
        features = extract_task_features({"train": []})
        assert features == {}
