"""Unit tests for Program Synthesizer (Pillar 2: Approximability)."""
import unittest
import random
from arc_agent.synthesizer import ProgramSynthesizer
from arc_agent.primitives import build_initial_toolkit
from arc_agent.concepts import Program


class TestProgramGeneration(unittest.TestCase):
    def setUp(self):
        self.toolkit = build_initial_toolkit()
        self.synth = ProgramSynthesizer(self.toolkit, population_size=20, max_program_length=3)

    def test_random_program(self):
        random.seed(42)
        p = self.synth._random_program()
        self.assertIsInstance(p, Program)
        self.assertGreaterEqual(len(p), 1)
        self.assertLessEqual(len(p), 3)

    def test_initial_population_includes_singles(self):
        pop = self.synth.generate_initial_population()
        # Should include single-concept programs
        singles = [p for p in pop if len(p) == 1]
        self.assertGreater(len(singles), 0)

    def test_initial_population_size(self):
        pop = self.synth.generate_initial_population()
        self.assertEqual(len(pop), self.synth.population_size)


class TestMutation(unittest.TestCase):
    def setUp(self):
        self.toolkit = build_initial_toolkit()
        self.synth = ProgramSynthesizer(self.toolkit, population_size=20, max_program_length=3)

    def test_mutate_produces_different_program(self):
        random.seed(42)
        p = self.synth._random_program()
        mutated = self.synth.mutate(p)
        # Mutation should produce a program (might be same by chance)
        self.assertIsInstance(mutated, Program)
        self.assertGreaterEqual(len(mutated), 1)

    def test_mutate_respects_max_length(self):
        random.seed(42)
        for _ in range(50):
            p = self.synth._random_program(max_len=3)
            m = self.synth.mutate(p)
            self.assertLessEqual(len(m), self.synth.max_program_length)


class TestCrossover(unittest.TestCase):
    def setUp(self):
        self.toolkit = build_initial_toolkit()
        self.synth = ProgramSynthesizer(self.toolkit, population_size=20, max_program_length=3)

    def test_crossover_combines_parents(self):
        random.seed(42)
        p1 = self.synth._random_program(max_len=3)
        p2 = self.synth._random_program(max_len=3)
        child = self.synth.crossover(p1, p2)
        self.assertIsInstance(child, Program)
        self.assertGreaterEqual(len(child), 1)

    def test_crossover_respects_max_length(self):
        random.seed(42)
        for _ in range(50):
            p1 = self.synth._random_program(max_len=3)
            p2 = self.synth._random_program(max_len=3)
            child = self.synth.crossover(p1, p2)
            self.assertLessEqual(len(child), self.synth.max_program_length)


class TestEvolution(unittest.TestCase):
    def setUp(self):
        self.toolkit = build_initial_toolkit()
        self.synth = ProgramSynthesizer(self.toolkit, population_size=20, max_program_length=3)

    def test_evolve_generation_preserves_size(self):
        random.seed(42)
        task = {
            "train": [
                {"input": [[1, 2], [3, 4]], "output": [[1, 2], [3, 4]]},
            ]
        }
        pop = self.synth.generate_initial_population()
        new_pop = self.synth.evolve_generation(pop, task)
        self.assertEqual(len(new_pop), self.synth.population_size)

    def test_evolve_generation_scores_programs(self):
        random.seed(42)
        task = {
            "train": [
                {"input": [[1, 2], [3, 4]], "output": [[3, 1], [4, 2]]},
            ]
        }
        pop = self.synth.generate_initial_population()
        new_pop = self.synth.evolve_generation(pop, task)
        # Best program should have a fitness score
        self.assertGreaterEqual(new_pop[0].fitness, 0.0)

    def test_synthesize_finds_identity(self):
        """The synthesizer should find the identity function for trivial tasks."""
        random.seed(42)
        task = {
            "train": [
                {"input": [[1, 2], [3, 4]], "output": [[1, 2], [3, 4]]},
                {"input": [[5, 6]], "output": [[5, 6]]},
            ]
        }
        best, history = self.synth.synthesize(task, max_generations=5)
        self.assertGreaterEqual(best.fitness, 0.99)

    def test_synthesize_returns_history(self):
        random.seed(42)
        task = {
            "train": [
                {"input": [[1]], "output": [[1]]},
            ]
        }
        best, history = self.synth.synthesize(task, max_generations=5)
        self.assertGreater(len(history), 0)
        self.assertIn("generation", history[0])
        self.assertIn("best_fitness", history[0])

    def test_synthesize_finds_mirror(self):
        """Should find mirror_h for a horizontal mirror task."""
        random.seed(42)
        task = {
            "train": [
                {"input": [[1, 2, 3]], "output": [[3, 2, 1]]},
                {"input": [[4, 5]], "output": [[5, 4]]},
            ]
        }
        best, history = self.synth.synthesize(task, max_generations=15)
        self.assertGreaterEqual(best.fitness, 0.99)


if __name__ == '__main__':
    unittest.main()
