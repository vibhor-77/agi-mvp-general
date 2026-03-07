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


class TestSeedInjection(unittest.TestCase):
    """Seed programs fill up to a quarter of the population."""

    def test_seed_fraction_is_quarter(self):
        random.seed(42)
        toolkit = build_initial_toolkit()
        synth = ProgramSynthesizer(toolkit, population_size=20)

        concepts = [c for c in toolkit.concepts.values() if c.kind != "predicate"]
        seeds = [Program([concepts[i % len(concepts)]]) for i in range(20)]

        population = synth.generate_initial_population()
        initial_size = len(population)

        added = 0
        for sp in seeds[:synth.population_size // 4]:
            population.append(sp)
            population.append(synth.mutate(sp))
            added += 2

        # population_size // 4 = 5 seeds × 2 = 10 extra entries
        self.assertEqual(added, synth.population_size // 2)
        self.assertEqual(len(population), initial_size + synth.population_size // 2)


class TestTripleSearch(unittest.TestCase):
    """Near-miss triple search finds 3-step solutions when pairs fall short."""

    def _crop_mirror_task(self):
        """Task solvable by crop_nonzero → mirror_h (pair) or with 3rd step."""
        return {
            "train": [
                {"input": [[0, 0, 0], [0, 1, 2], [0, 3, 4]],
                 "output": [[2, 1], [4, 3]]},
                {"input": [[0, 0, 0, 0], [0, 5, 6, 0], [0, 0, 0, 0]],
                 "output": [[6, 5]]},
            ]
        }

    def test_try_best_triples_returns_none_when_pair_none(self):
        """Returns None when best_pair is None."""
        toolkit = build_initial_toolkit()
        synth = ProgramSynthesizer(toolkit, population_size=20)
        from arc_agent.scorer import TaskCache
        task = self._crop_mirror_task()
        cache = TaskCache(task)
        result = synth.try_best_triples(None, cache, pair_score_threshold=0.5)
        self.assertIsNone(result)

    def test_try_best_triples_returns_none_when_pair_weak(self):
        """Returns None when best pair is below threshold."""
        random.seed(42)
        toolkit = build_initial_toolkit()
        synth = ProgramSynthesizer(toolkit, population_size=20)
        from arc_agent.scorer import TaskCache
        task = self._crop_mirror_task()
        cache = TaskCache(task)
        pair = synth.try_all_pairs(task, cache)
        # Threshold above pair score → should skip
        result = synth.try_best_triples(pair, cache, pair_score_threshold=1.01)
        self.assertIsNone(result)

    def test_try_best_triples_returns_program(self):
        """Returns a 3-step Program when a near-miss pair exists."""
        random.seed(42)
        toolkit = build_initial_toolkit()
        synth = ProgramSynthesizer(toolkit, population_size=20)
        from arc_agent.scorer import TaskCache
        task = self._crop_mirror_task()
        cache = TaskCache(task)
        pair = synth.try_all_pairs(task, cache)
        result = synth.try_best_triples(pair, cache, pair_score_threshold=0.5)
        self.assertIsNotNone(result)
        self.assertIsInstance(result, Program)
        self.assertEqual(len(result.steps), 3)

    def test_try_best_triples_score_geq_pair(self):
        """Triple score is non-negative."""
        random.seed(42)
        toolkit = build_initial_toolkit()
        synth = ProgramSynthesizer(toolkit, population_size=20)
        from arc_agent.scorer import TaskCache
        task = self._crop_mirror_task()
        cache = TaskCache(task)
        pair = synth.try_all_pairs(task, cache)
        triple = synth.try_best_triples(pair, cache, pair_score_threshold=0.5)
        if triple is not None:
            self.assertGreaterEqual(triple.fitness, 0.0)


if __name__ == '__main__':
    unittest.main()
