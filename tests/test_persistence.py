"""
Tests for persistent toolkit serialization (TDD).

Solves the "Reset Button Problem": the Toolkit must survive across runs.
This means we need to serialize/deserialize the concept library, including
learned composed concepts with their implementations.
"""
import unittest
import tempfile
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestToolkitSerialization(unittest.TestCase):
    """Test saving and loading the Toolkit."""

    def test_save_creates_file(self):
        from arc_agent.persistence import save_toolkit
        from arc_agent.primitives import build_initial_toolkit
        toolkit = build_initial_toolkit()
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            save_toolkit(toolkit, path)
            self.assertTrue(os.path.exists(path))
            self.assertGreater(os.path.getsize(path), 0)
        finally:
            os.unlink(path)

    def test_roundtrip_preserves_concept_count(self):
        from arc_agent.persistence import save_toolkit, load_toolkit
        from arc_agent.primitives import build_initial_toolkit
        toolkit = build_initial_toolkit()
        original_size = toolkit.size
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            save_toolkit(toolkit, path)
            loaded = load_toolkit(path)
            self.assertEqual(loaded.size, original_size)
        finally:
            os.unlink(path)

    def test_roundtrip_preserves_concept_names(self):
        from arc_agent.persistence import save_toolkit, load_toolkit
        from arc_agent.primitives import build_initial_toolkit
        toolkit = build_initial_toolkit()
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            save_toolkit(toolkit, path)
            loaded = load_toolkit(path)
            original_names = set(toolkit.concepts.keys())
            loaded_names = set(loaded.concepts.keys())
            self.assertEqual(original_names, loaded_names)
        finally:
            os.unlink(path)

    def test_loaded_concepts_are_functional(self):
        """Loaded concepts must actually transform grids correctly."""
        from arc_agent.persistence import save_toolkit, load_toolkit
        from arc_agent.primitives import build_initial_toolkit
        toolkit = build_initial_toolkit()
        grid = [[1, 2], [3, 4]]
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            save_toolkit(toolkit, path)
            loaded = load_toolkit(path)
            # Test that mirror_h still works
            original_result = toolkit.concepts["mirror_h"].apply(grid)
            loaded_result = loaded.concepts["mirror_h"].apply(grid)
            self.assertEqual(original_result, loaded_result)
        finally:
            os.unlink(path)

    def test_preserves_usage_stats(self):
        from arc_agent.persistence import save_toolkit, load_toolkit
        from arc_agent.primitives import build_initial_toolkit
        toolkit = build_initial_toolkit()
        # Simulate some usage
        toolkit.concepts["mirror_h"].usage_count = 10
        toolkit.concepts["mirror_h"].success_count = 7
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            save_toolkit(toolkit, path)
            loaded = load_toolkit(path)
            self.assertEqual(loaded.concepts["mirror_h"].usage_count, 10)
            self.assertEqual(loaded.concepts["mirror_h"].success_count, 7)
        finally:
            os.unlink(path)

    def test_preserves_learned_concepts(self):
        """Composed concepts must survive serialization."""
        from arc_agent.persistence import save_toolkit, load_toolkit
        from arc_agent.primitives import build_initial_toolkit
        from arc_agent.concepts import Concept
        toolkit = build_initial_toolkit()

        # Simulate learning a composed concept
        mirror = toolkit.concepts["mirror_h"]
        rotate = toolkit.concepts["rotate_90_cw"]
        composed = toolkit.compose(mirror, rotate)
        composed.name = "learned_mirror_rotate"
        composed.kind = "composed"
        toolkit.add_concept(composed)

        grid = [[1, 2], [3, 4]]
        original_result = composed.apply(grid)

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            save_toolkit(toolkit, path)
            loaded = load_toolkit(path)
            self.assertIn("learned_mirror_rotate", loaded.concepts)
            loaded_concept = loaded.concepts["learned_mirror_rotate"]
            self.assertEqual(loaded_concept.kind, "composed")
            loaded_result = loaded_concept.apply(grid)
            self.assertEqual(loaded_result, original_result)
        finally:
            os.unlink(path)


class TestArchiveSerialization(unittest.TestCase):
    """Test saving and loading the Archive."""

    def test_save_and_load_archive(self):
        from arc_agent.persistence import save_archive, load_archive
        from arc_agent.concepts import Archive, Program, Concept

        archive = Archive()
        # Record a solution
        concept = Concept(
            kind="operator", name="test_op",
            implementation=lambda g: g,
        )
        program = Program([concept], name="test_program")
        archive.record_solution("task1", program, 0.95)
        archive.record_features("task1", {"same_dims": True, "grows": False})

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            save_archive(archive, path)
            loaded = load_archive(path)
            self.assertIn("task1", loaded.task_features)
            self.assertEqual(loaded.task_features["task1"]["same_dims"], True)
            self.assertEqual(len(loaded.history), 1)
            self.assertEqual(loaded.history[0]["task_id"], "task1")
        finally:
            os.unlink(path)


if __name__ == "__main__":
    unittest.main()
