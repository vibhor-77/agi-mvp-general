"""Unit tests for Exploration Engine (Pillar 4: Explore/Exploit)."""
import unittest
import random
from arc_agent.explorer import ExplorationEngine
from arc_agent.primitives import build_initial_toolkit
from arc_agent.concepts import Toolkit, Archive, Program, Concept


class TestUCBSelection(unittest.TestCase):
    def setUp(self):
        self.toolkit = build_initial_toolkit()
        self.archive = Archive()
        self.explorer = ExplorationEngine(self.toolkit, self.archive, epsilon=0.3)

    def test_ucb_returns_concept(self):
        concept = self.explorer.select_concept_ucb()
        self.assertIsInstance(concept, Concept)

    def test_ucb_prefers_untried(self):
        """UCB should prefer concepts that haven't been tried yet."""
        # Mark all but one concept as tried
        for c in list(self.explorer.toolkit.concepts.values())[:-1]:
            c.usage_count = 10
            c.success_count = 5
        self.explorer.total_selections = 100

        selected = self.explorer.select_concept_ucb()
        self.assertEqual(selected.usage_count, 0)

    def test_ucb_exploits_successful(self):
        """After enough exploration, UCB should prefer high-success concepts."""
        random.seed(42)
        # Set up one concept as clearly the best
        concepts = list(self.explorer.toolkit.concepts.values())
        for c in concepts:
            c.usage_count = 50
            c.success_count = 5  # 10% success

        star = concepts[0]
        star.usage_count = 50
        star.success_count = 45  # 90% success

        self.explorer.total_selections = 1000
        # Over many selections, the star should be picked most often
        selections = [self.explorer.select_concept_ucb().name for _ in range(100)]
        self.assertGreater(selections.count(star.name), 20)


class TestNovelProgramGeneration(unittest.TestCase):
    def setUp(self):
        self.toolkit = build_initial_toolkit()
        self.archive = Archive()
        self.explorer = ExplorationEngine(self.toolkit, self.archive, epsilon=0.3)

    def test_generates_programs(self):
        random.seed(42)
        programs = self.explorer.generate_novel_programs(10)
        self.assertGreater(len(programs), 0)
        self.assertTrue(all(isinstance(p, Program) for p in programs))

    def test_programs_have_steps(self):
        random.seed(42)
        programs = self.explorer.generate_novel_programs(5)
        for p in programs:
            self.assertGreaterEqual(len(p), 1)


class TestSeedPrograms(unittest.TestCase):
    def setUp(self):
        self.toolkit = build_initial_toolkit()
        self.archive = Archive()
        self.explorer = ExplorationEngine(self.toolkit, self.archive, epsilon=0.3)

    def test_generates_seeds_for_same_dims(self):
        features = {"same_dims": True, "grows": False, "shrinks": False}
        seeds = self.explorer.generate_seed_programs(features)
        self.assertGreater(len(seeds), 0)
        # Should include geometric transforms for same-dims tasks
        names = [s.name for s in seeds]
        has_geometric = any(
            n in str(names) for n in ["rotate", "mirror", "transpose"]
        )
        self.assertTrue(has_geometric)

    def test_generates_seeds_for_growing(self):
        features = {"same_dims": False, "grows": True, "shrinks": False}
        seeds = self.explorer.generate_seed_programs(features)
        names = " ".join(s.name for s in seeds)
        has_scaling = "scale" in names or "tile" in names
        self.assertTrue(has_scaling)

    def test_generates_seeds_for_shrinking(self):
        features = {"same_dims": False, "grows": False, "shrinks": True}
        seeds = self.explorer.generate_seed_programs(features)
        names = " ".join(s.name for s in seeds)
        self.assertIn("crop", names)

    def test_includes_cross_task_transfer(self):
        """Seeds should include programs from similar past tasks."""
        c = Concept(kind="operator", name="past_solution",
                   implementation=lambda g: g)
        p = Program([c])
        # Use the explorer's own archive so the transfer lookup works
        self.explorer.archive.record_features("old_task", {"same_dims": True, "grows": False})
        self.explorer.archive.record_solution("old_task", p, 1.0)

        seeds = self.explorer.generate_seed_programs({"same_dims": True, "grows": False})
        seed_names = [s.name for s in seeds]
        self.assertIn("past_solution", seed_names)


class TestConceptDiscovery(unittest.TestCase):
    def setUp(self):
        self.toolkit = build_initial_toolkit()
        self.archive = Archive()
        self.explorer = ExplorationEngine(self.toolkit, self.archive, epsilon=0.3)

    def test_discover_promotes_multi_step(self):
        c1 = Concept(kind="operator", name="step1", implementation=lambda g: g)
        c2 = Concept(kind="operator", name="step2", implementation=lambda g: g)
        program = Program([c1, c2])

        new_concept = self.explorer.discover_new_concept(program, "task_x")
        self.assertIsNotNone(new_concept)
        self.assertEqual(new_concept.kind, "composed")
        self.assertIn("learned", new_concept.name)

    def test_discover_skips_single_step(self):
        c = Concept(kind="operator", name="single", implementation=lambda g: g)
        program = Program([c])

        new_concept = self.explorer.discover_new_concept(program, "task_x")
        self.assertIsNone(new_concept)


class TestEpsilonDecay(unittest.TestCase):
    def setUp(self):
        self.toolkit = build_initial_toolkit()
        self.archive = Archive()
        self.explorer = ExplorationEngine(self.toolkit, self.archive, epsilon=0.3)

    def test_decay_reduces_epsilon(self):
        initial = self.explorer.epsilon
        self.explorer.decay_epsilon()
        self.assertLess(self.explorer.epsilon, initial)

    def test_decay_has_minimum(self):
        for _ in range(1000):
            self.explorer.decay_epsilon()
        self.assertGreaterEqual(self.explorer.epsilon, 0.05)

    def test_should_explore_respects_epsilon(self):
        random.seed(42)
        self.explorer.epsilon = 1.0
        self.assertTrue(self.explorer.should_explore())

        self.explorer.epsilon = 0.0
        self.assertFalse(self.explorer.should_explore())
