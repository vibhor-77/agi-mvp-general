"""Unit tests for Exploration Engine (Pillar 4: Explore/Exploit)."""

import random
from arc_agent.explorer import ExplorationEngine
from arc_agent.primitives import build_initial_toolkit
from arc_agent.concepts import Toolkit, Archive, Program, Concept


# fixture
def toolkit():
    return build_initial_toolkit()


# fixture
def archive():
    return Archive()


# fixture
def explorer(toolkit, archive):
    return ExplorationEngine(toolkit, archive, epsilon=0.3)


class TestUCBSelection:
    def test_ucb_returns_concept(self, explorer):
        concept = explorer.select_concept_ucb()
        assert isinstance(concept, Concept)

    def test_ucb_prefers_untried(self, explorer):
        """UCB should prefer concepts that haven't been tried yet."""
        # Mark all but one concept as tried
        for c in list(explorer.toolkit.concepts.values())[:-1]:
            c.usage_count = 10
            c.success_count = 5
        explorer.total_selections = 100

        selected = explorer.select_concept_ucb()
        assert selected.usage_count == 0

    def test_ucb_exploits_successful(self, explorer):
        """After enough exploration, UCB should prefer high-success concepts."""
        random.seed(42)
        # Set up one concept as clearly the best
        concepts = list(explorer.toolkit.concepts.values())
        for c in concepts:
            c.usage_count = 50
            c.success_count = 5  # 10% success

        star = concepts[0]
        star.usage_count = 50
        star.success_count = 45  # 90% success

        explorer.total_selections = 1000
        # Over many selections, the star should be picked most often
        selections = [explorer.select_concept_ucb().name for _ in range(100)]
        assert selections.count(star.name) > 20


class TestNovelProgramGeneration:
    def test_generates_programs(self, explorer):
        random.seed(42)
        programs = explorer.generate_novel_programs(10)
        assert len(programs) > 0
        assert all(isinstance(p, Program) for p in programs)

    def test_programs_have_steps(self, explorer):
        random.seed(42)
        programs = explorer.generate_novel_programs(5)
        for p in programs:
            assert len(p) >= 1


class TestSeedPrograms:
    def test_generates_seeds_for_same_dims(self, explorer):
        features = {"same_dims": True, "grows": False, "shrinks": False}
        seeds = explorer.generate_seed_programs(features)
        assert len(seeds) > 0
        # Should include geometric transforms for same-dims tasks
        names = [s.name for s in seeds]
        has_geometric = any(
            n in str(names) for n in ["rotate", "mirror", "transpose"]
        )
        assert has_geometric

    def test_generates_seeds_for_growing(self, explorer):
        features = {"same_dims": False, "grows": True, "shrinks": False}
        seeds = explorer.generate_seed_programs(features)
        names = " ".join(s.name for s in seeds)
        has_scaling = "scale" in names or "tile" in names
        assert has_scaling

    def test_generates_seeds_for_shrinking(self, explorer):
        features = {"same_dims": False, "grows": False, "shrinks": True}
        seeds = explorer.generate_seed_programs(features)
        names = " ".join(s.name for s in seeds)
        assert "crop" in names

    def test_includes_cross_task_transfer(self, explorer):
        """Seeds should include programs from similar past tasks."""
        c = Concept(kind="operator", name="past_solution",
                   implementation=lambda g: g)
        p = Program([c])
        # Use the explorer's own archive so the transfer lookup works
        explorer.archive.record_features("old_task", {"same_dims": True, "grows": False})
        explorer.archive.record_solution("old_task", p, 1.0)

        seeds = explorer.generate_seed_programs({"same_dims": True, "grows": False})
        seed_names = [s.name for s in seeds]
        assert "past_solution" in seed_names


class TestConceptDiscovery:
    def test_discover_promotes_multi_step(self, explorer):
        c1 = Concept(kind="operator", name="step1", implementation=lambda g: g)
        c2 = Concept(kind="operator", name="step2", implementation=lambda g: g)
        program = Program([c1, c2])

        new_concept = explorer.discover_new_concept(program, "task_x")
        assert new_concept is not None
        assert new_concept.kind == "composed"
        assert "learned" in new_concept.name

    def test_discover_skips_single_step(self, explorer):
        c = Concept(kind="operator", name="single", implementation=lambda g: g)
        program = Program([c])

        new_concept = explorer.discover_new_concept(program, "task_x")
        assert new_concept is None


class TestEpsilonDecay:
    def test_decay_reduces_epsilon(self, explorer):
        initial = explorer.epsilon
        explorer.decay_epsilon()
        assert explorer.epsilon < initial

    def test_decay_has_minimum(self, explorer):
        for _ in range(1000):
            explorer.decay_epsilon()
        assert explorer.epsilon >= 0.05

    def test_should_explore_respects_epsilon(self, explorer):
        random.seed(42)
        explorer.epsilon = 1.0
        assert explorer.should_explore() is True

        explorer.epsilon = 0.0
        assert explorer.should_explore() is False
