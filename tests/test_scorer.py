"""Unit tests for the Scoring Engine (Pillar 1: Feedback Loops)."""
import unittest
from arc_agent.scorer import (
    pixel_accuracy, structural_similarity,
    score_program_on_task, validate_on_test,
    extract_task_features,
)
from arc_agent.concepts import Concept, Program


class TestPixelAccuracy(unittest.TestCase):
    def test_perfect_match(self):
        grid = [[1, 2], [3, 4]]
        self.assertEqual(pixel_accuracy(grid, grid), 1.0)

    def test_no_match(self):
        pred = [[1, 1], [1, 1]]
        exp = [[2, 2], [2, 2]]
        self.assertEqual(pixel_accuracy(pred, exp), 0.0)

    def test_partial_match(self):
        pred = [[1, 2], [3, 4]]
        exp = [[1, 2], [3, 5]]
        self.assertEqual(pixel_accuracy(pred, exp), 0.75)

    def test_dimension_mismatch(self):
        pred = [[1, 2, 3]]
        exp = [[1, 2]]
        score = pixel_accuracy(pred, exp)
        # Should be low but not zero (partial credit for h match)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 0.2)

    def test_empty_grids(self):
        self.assertEqual(pixel_accuracy([], []), 0.0)
        self.assertEqual(pixel_accuracy([[1]], []), 0.0)

    def test_single_cell(self):
        self.assertEqual(pixel_accuracy([[5]], [[5]]), 1.0)
        self.assertEqual(pixel_accuracy([[5]], [[3]]), 0.0)


class TestStructuralSimilarity(unittest.TestCase):
    def test_perfect_match(self):
        grid = [[1, 2], [3, 4]]
        score = structural_similarity(grid, grid)
        self.assertGreaterEqual(score, 0.99)

    def test_completely_wrong(self):
        pred = [[9, 9, 9]]
        exp = [[1, 2]]
        score = structural_similarity(pred, exp)
        self.assertLess(score, 0.3)

    def test_right_dims_wrong_colors(self):
        pred = [[9, 9], [9, 9]]
        exp = [[1, 2], [3, 4]]
        score = structural_similarity(pred, exp)
        # Should get some credit for right dimensions
        self.assertGreater(score, 0.0)
        self.assertLess(score, 0.5)

    def test_score_in_range(self):
        pred = [[1, 0], [0, 1]]
        exp = [[1, 1], [1, 1]]
        score = structural_similarity(pred, exp)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)


class TestScoreProgramOnTask(unittest.TestCase):
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
        self.assertGreaterEqual(score, 0.99)

    def test_wrong_program_low_score(self):
        task = {
            "train": [
                {"input": [[1, 2], [3, 4]], "output": [[4, 3], [2, 1]]},
            ]
        }
        identity = Concept(kind="operator", name="id", implementation=lambda g: g)
        program = Program([identity])
        score = score_program_on_task(program, task)
        self.assertLess(score, 0.5)

    def test_empty_task(self):
        task = {"train": []}
        identity = Concept(kind="operator", name="id", implementation=lambda g: g)
        program = Program([identity])
        score = score_program_on_task(program, task)
        self.assertEqual(score, 0.0)


class TestValidateOnTest(unittest.TestCase):
    def test_correct_validation(self):
        task = {
            "test": [
                {"input": [[1, 2]], "output": [[1, 2]]},
            ]
        }
        identity = Concept(kind="operator", name="id", implementation=lambda g: g)
        program = Program([identity])
        exact, score = validate_on_test(program, task)
        self.assertTrue(exact)
        self.assertEqual(score, 1.0)

    def test_incorrect_validation(self):
        task = {
            "test": [
                {"input": [[1, 2]], "output": [[3, 4]]},
            ]
        }
        identity = Concept(kind="operator", name="id", implementation=lambda g: g)
        program = Program([identity])
        exact, score = validate_on_test(program, task)
        self.assertFalse(exact)


class TestExtractTaskFeatures(unittest.TestCase):
    def test_same_dims_detection(self):
        task = {
            "train": [
                {"input": [[1, 2], [3, 4]], "output": [[5, 6], [7, 8]]},
            ]
        }
        features = extract_task_features(task)
        self.assertTrue(features["same_dims"])
        self.assertFalse(features["grows"])
        self.assertFalse(features["shrinks"])

    def test_growing_detection(self):
        task = {
            "train": [
                {"input": [[1]], "output": [[1, 1], [1, 1]]},
            ]
        }
        features = extract_task_features(task)
        self.assertTrue(features["grows"])
        self.assertFalse(features["shrinks"])

    def test_shrinking_detection(self):
        task = {
            "train": [
                {"input": [[1, 2, 3], [4, 5, 6]], "output": [[1]]},
            ]
        }
        features = extract_task_features(task)
        self.assertTrue(features["shrinks"])

    def test_color_analysis(self):
        task = {
            "train": [
                {"input": [[1, 2, 0]], "output": [[3, 4, 0]]},
            ]
        }
        features = extract_task_features(task)
        self.assertTrue(features["new_colors"])  # 3, 4 not in input
        self.assertTrue(features["lost_colors"])  # 1, 2 not in output

    def test_empty_task(self):
        features = extract_task_features({"train": []})
        self.assertEqual(features, {})


if __name__ == '__main__':
    unittest.main()
