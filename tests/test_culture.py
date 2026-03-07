"""Unit tests for culture persistence (save/load learned knowledge)."""
import json
import os
import tempfile
import unittest

from arc_agent.concepts import Concept, Program, Toolkit, Archive, Grid
from arc_agent.primitives import build_initial_toolkit
from arc_agent.culture import save_culture, load_culture, _extract_step_names


class TestExtractStepNames(unittest.TestCase):
    """Test step name extraction from concepts."""

    def test_simple_concept(self):
        c = Concept(kind="operator", name="mirror_h", implementation=lambda g: g)
        self.assertEqual(_extract_step_names(c), ["mirror_h"])

    def test_composed_concept(self):
        a = Concept(kind="operator", name="mirror_h", implementation=lambda g: g)
        b = Concept(kind="operator", name="crop_nonzero", implementation=lambda g: g)
        composed = Concept(
            kind="composed", name="learned_test",
            implementation=lambda g: g, children=[a, b],
        )
        self.assertEqual(_extract_step_names(composed), ["mirror_h", "crop_nonzero"])


class TestSaveCulture(unittest.TestCase):
    """Test saving culture to JSON."""

    def test_save_empty_culture(self):
        toolkit = Toolkit()
        archive = Archive()
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            culture = save_culture(toolkit, archive, path)
            self.assertEqual(len(culture["learned_concepts"]), 0)
            self.assertEqual(len(culture["successful_programs"]), 0)
            # Verify file is valid JSON
            with open(path) as fh:
                loaded = json.load(fh)
            self.assertEqual(loaded["version"], "0.9")
        finally:
            os.unlink(path)

    def test_save_with_learned_concepts(self):
        toolkit = build_initial_toolkit()
        archive = Archive()

        # Add a learned concept
        a = toolkit.concepts["mirror_h"]
        b = toolkit.concepts["crop_nonzero"]
        learned = Concept(
            kind="composed", name="learned_test_123",
            implementation=lambda g: g, children=[a, b],
        )
        learned.usage_count = 5
        learned.success_count = 3
        toolkit.add_concept(learned)

        # Add a program to archive
        prog = Program([a, b])
        prog.fitness = 0.95
        archive.record_solution("task_001", prog, 0.95)
        archive.record_features("task_001", {"same_dims": True, "shrinks": False})

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            culture = save_culture(toolkit, archive, path)
            self.assertEqual(len(culture["learned_concepts"]), 1)
            self.assertEqual(culture["learned_concepts"][0]["name"], "learned_test_123")
            self.assertEqual(culture["learned_concepts"][0]["steps"], ["mirror_h", "crop_nonzero"])
            self.assertEqual(len(culture["successful_programs"]), 1)
            self.assertEqual(culture["task_features"]["task_001"]["same_dims"], True)
        finally:
            os.unlink(path)


class TestLoadCulture(unittest.TestCase):
    """Test loading culture from JSON."""

    def test_roundtrip(self):
        """Save then load culture, verify concepts are reconstructed."""
        toolkit = build_initial_toolkit()
        archive = Archive()
        initial_size = toolkit.size

        # Add learned concept
        a = toolkit.concepts["mirror_h"]
        b = toolkit.concepts["crop_nonzero"]
        learned = Concept(
            kind="composed", name="learned_roundtrip",
            implementation=lambda g: g, children=[a, b],
        )
        toolkit.add_concept(learned)

        # Add program
        prog = Program([a, b])
        prog.fitness = 0.99
        archive.record_solution("task_rt", prog, 0.99)

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            save_culture(toolkit, archive, path)

            # Load into a fresh toolkit
            fresh_toolkit = build_initial_toolkit()
            fresh_archive = Archive()
            self.assertEqual(fresh_toolkit.size, initial_size)

            stats = load_culture(fresh_toolkit, path, fresh_archive)
            self.assertEqual(stats["concepts_loaded"], 1)
            self.assertEqual(stats["programs_loaded"], 1)
            self.assertIn("learned_roundtrip", fresh_toolkit.concepts)

            # Verify the loaded concept actually works
            loaded_concept = fresh_toolkit.concepts["learned_roundtrip"]
            test_grid = [[1, 2], [3, 4]]
            result = loaded_concept.apply(test_grid)
            self.assertIsNotNone(result)
        finally:
            os.unlink(path)

    def test_load_skips_missing_steps(self):
        """Concepts referencing non-existent primitives are skipped."""
        culture = {
            "version": "0.9",
            "learned_concepts": [{
                "name": "learned_bad",
                "steps": ["nonexistent_primitive_xyz"],
                "kind": "composed",
                "usage_count": 0,
                "success_count": 0,
            }],
            "successful_programs": [],
            "task_features": {},
        }
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            json.dump(culture, f)
            path = f.name
        try:
            toolkit = build_initial_toolkit()
            stats = load_culture(toolkit, path)
            self.assertEqual(stats["concepts_skipped"], 1)
            self.assertNotIn("learned_bad", toolkit.concepts)
        finally:
            os.unlink(path)

    def test_load_features_into_archive(self):
        """Task features are loaded into the archive for cross-task transfer."""
        culture = {
            "version": "0.9",
            "learned_concepts": [],
            "successful_programs": [],
            "task_features": {
                "task_abc": {"same_dims": True, "shrinks": False, "grows": True},
            },
        }
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            json.dump(culture, f)
            path = f.name
        try:
            toolkit = build_initial_toolkit()
            archive = Archive()
            stats = load_culture(toolkit, path, archive)
            self.assertEqual(stats["features_loaded"], 1)
            self.assertIn("task_abc", archive.task_features)
            self.assertEqual(archive.task_features["task_abc"]["same_dims"], True)
        finally:
            os.unlink(path)


if __name__ == "__main__":
    unittest.main()
