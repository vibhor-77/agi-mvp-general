"""Unit tests for the Concept system (Pillar 3: Abstraction & Composability)."""
import unittest
from arc_agent.concepts import Concept, Program, Toolkit, Archive, Grid


# ── Concept Tests ──────────────────────────────────────────────

class TestConcept(unittest.TestCase):
    def test_constant_concept(self):
        c = Concept(kind="constant", name="zero", implementation=lambda g: g)
        self.assertEqual(c.kind, "constant")
        self.assertEqual(c.name, "zero")
        self.assertEqual(c.usage_count, 0)
        self.assertEqual(c.success_count, 0)

    def test_apply_increments_usage(self):
        c = Concept(kind="operator", name="id", implementation=lambda g: g)
        grid = [[1, 2], [3, 4]]
        c.apply(grid)
        self.assertEqual(c.usage_count, 1)

    def test_apply_returns_result(self):
        def double_first(grid):
            return [[cell * 2 for cell in row] for row in grid]
        c = Concept(kind="operator", name="double", implementation=double_first)
        result = c.apply([[1, 2], [3, 4]])
        self.assertEqual(result, [[2, 4], [6, 8]])

    def test_apply_returns_none_on_exception(self):
        def bad_fn(grid):
            raise ValueError("broken")
        c = Concept(kind="operator", name="bad", implementation=bad_fn)
        result = c.apply([[1]])
        self.assertIsNone(result)
        self.assertEqual(c.usage_count, 1)

    def test_reinforce_success(self):
        c = Concept(kind="operator", name="test", implementation=lambda g: g)
        c.usage_count = 5
        c.reinforce(True)
        self.assertEqual(c.success_count, 1)
        c.reinforce(True)
        self.assertEqual(c.success_count, 2)
        c.reinforce(False)
        self.assertEqual(c.success_count, 2)

    def test_success_rate(self):
        c = Concept(kind="operator", name="test", implementation=lambda g: g)
        self.assertEqual(c.success_rate, 0.0)
        c.usage_count = 10
        c.success_count = 7
        self.assertEqual(c.success_rate, 0.7)

    def test_repr_simple(self):
        c = Concept(kind="operator", name="rotate", implementation=lambda g: g)
        self.assertIn("operator", repr(c))
        self.assertIn("rotate", repr(c))

    def test_repr_composed(self):
        a = Concept(kind="operator", name="step1", implementation=lambda g: g)
        b = Concept(kind="operator", name="step2", implementation=lambda g: g)
        composed = Concept(kind="composed", name="step1→step2",
                          implementation=lambda g: g, children=[a, b])
        self.assertIn("step1", repr(composed))
        self.assertIn("step2", repr(composed))


# ── Program Tests ──────────────────────────────────────────────

class TestProgram(unittest.TestCase):
    def test_program_creation(self):
        c1 = Concept(kind="operator", name="a", implementation=lambda g: g)
        c2 = Concept(kind="operator", name="b", implementation=lambda g: g)
        p = Program([c1, c2])
        self.assertEqual(len(p), 2)
        self.assertIn("a", p.name)
        self.assertIn("b", p.name)

    def test_program_execute_chains_steps(self):
        def add_one(grid):
            return [[cell + 1 for cell in row] for row in grid]
        c1 = Concept(kind="operator", name="add1", implementation=add_one)
        c2 = Concept(kind="operator", name="add1b", implementation=add_one)
        p = Program([c1, c2])
        result = p.execute([[0, 0], [0, 0]])
        self.assertEqual(result, [[2, 2], [2, 2]])

    def test_program_execute_stops_on_none(self):
        c1 = Concept(kind="operator", name="ok", implementation=lambda g: g)
        c2 = Concept(kind="operator", name="fail",
                    implementation=lambda g: None)
        c3 = Concept(kind="operator", name="never",
                    implementation=lambda g: [[99]])
        p = Program([c1, c2, c3])
        result = p.execute([[1]])
        self.assertIsNone(result)

    def test_program_custom_name(self):
        c = Concept(kind="operator", name="x", implementation=lambda g: g)
        p = Program([c], name="my_program")
        self.assertEqual(p.name, "my_program")


# ── Toolkit Tests ──────────────────────────────────────────────

class TestToolkit(unittest.TestCase):
    def setUp(self):
        self.tk = Toolkit()

    def test_add_and_retrieve_concept(self):
        c = Concept(kind="operator", name="test", implementation=lambda g: g)
        self.tk.add_concept(c)
        self.assertEqual(self.tk.size, 1)
        self.assertIn("test", self.tk.concepts)

    def test_get_best_concepts(self):
        for i in range(5):
            c = Concept(kind="operator", name=f"c{i}", implementation=lambda g: g)
            c.usage_count = 10
            c.success_count = i * 2  # Varying success rates
            self.tk.add_concept(c)
        best = self.tk.get_best_concepts(3)
        self.assertEqual(len(best), 3)
        # Should be sorted by success rate descending
        self.assertGreaterEqual(best[0].success_rate, best[1].success_rate)

    def test_compose_two_concepts(self):
        def double(grid):
            return [[c * 2 for c in row] for row in grid]
        def add_one(grid):
            return [[c + 1 for c in row] for row in grid]

        c1 = Concept(kind="operator", name="double", implementation=double)
        c2 = Concept(kind="operator", name="add1", implementation=add_one)
        composed = self.tk.compose(c1, c2)

        self.assertEqual(composed.kind, "composed")
        self.assertEqual(len(composed.children), 2)
        # double then add1: 3 → 6 → 7
        result = composed.apply([[3]])
        self.assertEqual(result, [[7]])

    def test_get_concepts_by_kind(self):
        self.tk.add_concept(Concept(kind="operator", name="op1", implementation=lambda g: g))
        self.tk.add_concept(Concept(kind="constant", name="const1", implementation=lambda g: g))
        self.tk.add_concept(Concept(kind="operator", name="op2", implementation=lambda g: g))
        ops = self.tk.get_concepts_by_kind("operator")
        self.assertEqual(len(ops), 2)

    def test_add_program(self):
        c = Concept(kind="operator", name="x", implementation=lambda g: g)
        p = Program([c])
        self.tk.add_program(p)
        self.assertEqual(len(self.tk.programs), 1)


# ── Archive Tests ──────────────────────────────────────────────

class TestArchive(unittest.TestCase):
    def setUp(self):
        self.archive = Archive()

    def test_record_solution(self):
        c = Concept(kind="operator", name="x", implementation=lambda g: g)
        p = Program([c])
        self.archive.record_solution("task1", p, 0.95)
        self.assertIn("task1", self.archive.task_solutions)
        self.assertEqual(len(self.archive.history), 1)

    def test_record_features(self):
        features = {"same_dims": True, "grows": False}
        self.archive.record_features("task1", features)
        self.assertEqual(self.archive.task_features["task1"], features)

    def test_find_similar_tasks(self):
        self.archive.record_features("task1", {"same_dims": True, "grows": False, "in_colors": 3})
        self.archive.record_features("task2", {"same_dims": False, "grows": True, "in_colors": 5})
        self.archive.record_features("task3", {"same_dims": True, "grows": False, "in_colors": 3})

        similar = self.archive.find_similar_tasks({"same_dims": True, "grows": False, "in_colors": 3})
        # task1 and task3 should be most similar
        self.assertIn("task1", similar)
        self.assertIn("task3", similar)

    def test_get_programs_for_similar_tasks(self):
        c = Concept(kind="operator", name="x", implementation=lambda g: g)
        p = Program([c])

        self.archive.record_features("task1", {"same_dims": True})
        self.archive.record_solution("task1", p, 1.0)

        programs = self.archive.get_programs_for_similar_tasks({"same_dims": True})
        self.assertGreaterEqual(len(programs), 1)


if __name__ == '__main__':
    unittest.main()
