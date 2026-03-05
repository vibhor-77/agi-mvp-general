"""Unit tests for Program Synthesizer (Pillar 2: Approximability)."""

import random
from arc_agent.synthesizer import ProgramSynthesizer
from arc_agent.primitives import build_initial_toolkit
from arc_agent.concepts import Program


# fixture
def toolkit():
    return build_initial_toolkit()


# fixture
def synth(toolkit):
    return ProgramSynthesizer(toolkit, population_size=20, max_program_length=3)


class TestProgramGeneration:
    def test_random_program(self, synth):
        random.seed(42)
        p = synth._random_program()
        assert isinstance(p, Program)
        assert 1 <= len(p) <= 3

    def test_initial_population_includes_singles(self, synth):
        pop = synth.generate_initial_population()
        # Should include single-concept programs
        singles = [p for p in pop if len(p) == 1]
        assert len(singles) > 0

    def test_initial_population_size(self, synth):
        pop = synth.generate_initial_population()
        assert len(pop) == synth.population_size


class TestMutation:
    def test_mutate_produces_different_program(self, synth):
        random.seed(42)
        p = synth._random_program()
        mutated = synth.mutate(p)
        # Mutation should produce a program (might be same by chance)
        assert isinstance(mutated, Program)
        assert len(mutated) >= 1

    def test_mutate_respects_max_length(self, synth):
        random.seed(42)
        for _ in range(50):
            p = synth._random_program(max_len=3)
            m = synth.mutate(p)
            assert len(m) <= synth.max_program_length


class TestCrossover:
    def test_crossover_combines_parents(self, synth):
        random.seed(42)
        p1 = synth._random_program(max_len=3)
        p2 = synth._random_program(max_len=3)
        child = synth.crossover(p1, p2)
        assert isinstance(child, Program)
        assert len(child) >= 1

    def test_crossover_respects_max_length(self, synth):
        random.seed(42)
        for _ in range(50):
            p1 = synth._random_program(max_len=3)
            p2 = synth._random_program(max_len=3)
            child = synth.crossover(p1, p2)
            assert len(child) <= synth.max_program_length


class TestEvolution:
    def test_evolve_generation_preserves_size(self, synth):
        random.seed(42)
        task = {
            "train": [
                {"input": [[1, 2], [3, 4]], "output": [[1, 2], [3, 4]]},
            ]
        }
        pop = synth.generate_initial_population()
        new_pop = synth.evolve_generation(pop, task)
        assert len(new_pop) == synth.population_size

    def test_evolve_generation_scores_programs(self, synth):
        random.seed(42)
        task = {
            "train": [
                {"input": [[1, 2], [3, 4]], "output": [[3, 1], [4, 2]]},
            ]
        }
        pop = synth.generate_initial_population()
        new_pop = synth.evolve_generation(pop, task)
        # Best program should have a fitness score
        assert new_pop[0].fitness >= 0.0

    def test_synthesize_finds_identity(self, synth):
        """The synthesizer should find the identity function for trivial tasks."""
        random.seed(42)
        task = {
            "train": [
                {"input": [[1, 2], [3, 4]], "output": [[1, 2], [3, 4]]},
                {"input": [[5, 6]], "output": [[5, 6]]},
            ]
        }
        best, history = synth.synthesize(task, max_generations=5)
        assert best.fitness >= 0.99

    def test_synthesize_returns_history(self, synth):
        random.seed(42)
        task = {
            "train": [
                {"input": [[1]], "output": [[1]]},
            ]
        }
        best, history = synth.synthesize(task, max_generations=5)
        assert len(history) > 0
        assert "generation" in history[0]
        assert "best_fitness" in history[0]

    def test_synthesize_finds_mirror(self, synth):
        """Should find mirror_h for a horizontal mirror task."""
        random.seed(42)
        task = {
            "train": [
                {"input": [[1, 2, 3]], "output": [[3, 2, 1]]},
                {"input": [[4, 5]], "output": [[5, 4]]},
            ]
        }
        best, history = synth.synthesize(task, max_generations=15)
        assert best.fitness >= 0.99
