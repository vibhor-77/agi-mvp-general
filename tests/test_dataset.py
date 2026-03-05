"""
Tests for ARC-AGI dataset loading and evaluation harness (TDD).

The loader reads the official ARC-AGI JSON format:
  {
    "train": [{"input": [[...]], "output": [[...]]}],
    "test":  [{"input": [[...]], "output": [[...]]}]
  }

Each task is a JSON file. Tasks are organized into directories:
  data/training/  — 400 training tasks (ARC-AGI-1)
  data/evaluation/ — 400 evaluation tasks (ARC-AGI-1)
"""
import unittest
import tempfile
import os
import json
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestLoadTask(unittest.TestCase):
    """Test loading a single ARC-AGI task from JSON."""

    def setUp(self):
        """Create a temporary task JSON file."""
        self.task_data = {
            "train": [
                {"input": [[0, 1], [1, 0]], "output": [[1, 0], [0, 1]]},
                {"input": [[0, 2], [2, 0]], "output": [[2, 0], [0, 2]]},
            ],
            "test": [
                {"input": [[0, 3], [3, 0]], "output": [[3, 0], [0, 3]]},
            ],
        }
        self.tmpdir = tempfile.mkdtemp()
        self.task_path = os.path.join(self.tmpdir, "abc123.json")
        with open(self.task_path, "w") as f:
            json.dump(self.task_data, f)

    def tearDown(self):
        os.unlink(self.task_path)
        os.rmdir(self.tmpdir)

    def test_load_single_task(self):
        from arc_agent.dataset import load_task
        task = load_task(self.task_path)
        self.assertIn("train", task)
        self.assertIn("test", task)
        self.assertEqual(len(task["train"]), 2)
        self.assertEqual(len(task["test"]), 1)

    def test_train_examples_have_input_output(self):
        from arc_agent.dataset import load_task
        task = load_task(self.task_path)
        for example in task["train"]:
            self.assertIn("input", example)
            self.assertIn("output", example)

    def test_grids_are_lists_of_lists(self):
        from arc_agent.dataset import load_task
        task = load_task(self.task_path)
        grid = task["train"][0]["input"]
        self.assertIsInstance(grid, list)
        self.assertIsInstance(grid[0], list)


class TestLoadDataset(unittest.TestCase):
    """Test loading an entire dataset directory."""

    def setUp(self):
        """Create a temporary directory with multiple task files."""
        self.tmpdir = tempfile.mkdtemp()
        self.task_ids = []
        for i in range(3):
            task_id = f"task_{i:04d}"
            self.task_ids.append(task_id)
            task_data = {
                "train": [{"input": [[i]], "output": [[i + 1]]}],
                "test": [{"input": [[i + 2]], "output": [[i + 3]]}],
            }
            with open(os.path.join(self.tmpdir, f"{task_id}.json"), "w") as f:
                json.dump(task_data, f)

    def tearDown(self):
        for f in os.listdir(self.tmpdir):
            os.unlink(os.path.join(self.tmpdir, f))
        os.rmdir(self.tmpdir)

    def test_loads_all_tasks(self):
        from arc_agent.dataset import load_dataset
        tasks = load_dataset(self.tmpdir)
        self.assertEqual(len(tasks), 3)

    def test_task_ids_are_filenames(self):
        from arc_agent.dataset import load_dataset
        tasks = load_dataset(self.tmpdir)
        for task_id in self.task_ids:
            self.assertIn(task_id, tasks)

    def test_ignores_non_json_files(self):
        from arc_agent.dataset import load_dataset
        # Add a non-JSON file
        with open(os.path.join(self.tmpdir, "README.md"), "w") as f:
            f.write("not a task")
        tasks = load_dataset(self.tmpdir)
        self.assertEqual(len(tasks), 3)

    def test_empty_directory(self):
        from arc_agent.dataset import load_dataset
        empty = tempfile.mkdtemp()
        tasks = load_dataset(empty)
        self.assertEqual(len(tasks), 0)
        os.rmdir(empty)


class TestEvaluationHarness(unittest.TestCase):
    """Test the evaluation harness that runs the solver on a dataset."""

    def setUp(self):
        """Create a small dataset with tasks our solver can handle."""
        self.tmpdir = tempfile.mkdtemp()
        # A mirror_h task — our solver can definitely solve this
        task_data = {
            "train": [
                {"input": [[1, 2, 3]], "output": [[3, 2, 1]]},
                {"input": [[4, 5, 6]], "output": [[6, 5, 4]]},
            ],
            "test": [
                {"input": [[7, 8, 9]], "output": [[9, 8, 7]]},
            ],
        }
        with open(os.path.join(self.tmpdir, "mirror_task.json"), "w") as f:
            json.dump(task_data, f)

        # An identity task
        task_data2 = {
            "train": [
                {"input": [[1, 2], [3, 4]], "output": [[1, 2], [3, 4]]},
            ],
            "test": [
                {"input": [[5, 6], [7, 8]], "output": [[5, 6], [7, 8]]},
            ],
        }
        with open(os.path.join(self.tmpdir, "identity_task.json"), "w") as f:
            json.dump(task_data2, f)

    def tearDown(self):
        for f in os.listdir(self.tmpdir):
            os.unlink(os.path.join(self.tmpdir, f))
        os.rmdir(self.tmpdir)

    def test_evaluate_returns_results(self):
        from arc_agent.dataset import load_dataset, evaluate_dataset
        tasks = load_dataset(self.tmpdir)
        results = evaluate_dataset(tasks, verbose=False)
        self.assertIsInstance(results, dict)
        self.assertIn("task_results", results)
        self.assertIn("summary", results)

    def test_summary_has_key_metrics(self):
        from arc_agent.dataset import load_dataset, evaluate_dataset
        tasks = load_dataset(self.tmpdir)
        results = evaluate_dataset(tasks, verbose=False)
        summary = results["summary"]
        self.assertIn("total_tasks", summary)
        self.assertIn("solved_exact", summary)
        self.assertIn("solve_rate", summary)
        self.assertIn("test_correct", summary)
        self.assertIn("test_rate", summary)

    def test_solves_simple_tasks(self):
        from arc_agent.dataset import load_dataset, evaluate_dataset
        tasks = load_dataset(self.tmpdir)
        results = evaluate_dataset(tasks, verbose=False)
        # Should solve at least the identity task
        self.assertGreater(results["summary"]["solved_exact"], 0)

    def test_results_saved_to_json(self):
        from arc_agent.dataset import load_dataset, evaluate_dataset
        tasks = load_dataset(self.tmpdir)
        output_path = os.path.join(self.tmpdir, "results.json")
        results = evaluate_dataset(tasks, verbose=False, output_path=output_path)
        self.assertTrue(os.path.exists(output_path))
        with open(output_path) as f:
            saved = json.load(f)
        self.assertEqual(saved["summary"]["total_tasks"], 2)


if __name__ == "__main__":
    unittest.main()
