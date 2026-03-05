"""
Pillar 3: Abstraction & Composability

The recursive concept grammar from Vibhor's framework:
  Concept → Constant | Relationship | Operator | Concept Op Concept

Dual memory system:
  Toolkit (Type 1): Reusable, timeless patterns
  Archive (Type 2): Task-specific, time-bound context
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Optional, Any
import json
import copy


# Type alias for ARC grids
Grid = list[list[int]]


@dataclass
class Concept:
    """A composable building block of intelligence.

    Follows Vibhor's recursive grammar:
      Concept → Constant | Relationship | Operator | Composed

    A Concept wraps a grid transformation function and can be composed
    with other Concepts to form higher-level abstractions.
    """
    kind: str  # "constant", "relationship", "operator", "composed"
    name: str
    implementation: Callable[[Grid], Grid]
    children: list['Concept'] = field(default_factory=list)
    # Track utility for reinforcement (Pillar 2: Approximability)
    usage_count: int = 0
    success_count: int = 0

    @property
    def success_rate(self) -> float:
        if self.usage_count == 0:
            return 0.0
        return self.success_count / self.usage_count

    def apply(self, grid: Grid) -> Optional[Grid]:
        """Apply this concept to a grid, returning None on failure."""
        try:
            self.usage_count += 1
            result = self.implementation(grid)
            if result is not None:
                return result
        except Exception:
            pass
        return None

    def reinforce(self, success: bool):
        """Update success tracking (Pillar 1: Feedback Loop)."""
        if success:
            self.success_count += 1

    def __repr__(self):
        if self.kind == "composed":
            child_names = [c.name for c in self.children]
            return f"Concept({self.name}: {' → '.join(child_names)})"
        return f"Concept({self.kind}:{self.name})"


class Program:
    """A sequence of Concepts applied in order (a composed transformation).

    This is the unit of program synthesis — the thing we evolve.
    Programs are themselves composable (Pillar 3: fractal composition).
    """
    def __init__(self, steps: list[Concept], name: str = ""):
        self.steps = steps
        self.name = name or " → ".join(s.name for s in steps)
        self.fitness: float = 0.0
        self.task_scores: dict[str, float] = {}

    def execute(self, grid: Grid) -> Optional[Grid]:
        """Execute the program: apply each step in sequence."""
        current = grid
        for step in self.steps:
            result = step.apply(current)
            if result is None:
                return None
            current = result
        return current

    def __len__(self):
        return len(self.steps)

    def __repr__(self):
        return f"Program({self.name}, fitness={self.fitness:.3f})"


class Toolkit:
    """Type 1 Memory: Reusable patterns (timeless, context-independent).

    The Toolkit stores all learned concepts and grows over time.
    This is the 'cumulative culture' mechanism from Vibhor's framework —
    knowledge compounds rather than resetting.
    """
    def __init__(self):
        self.concepts: dict[str, Concept] = {}
        self.programs: list[Program] = []  # Successful programs
        self._concept_counter = 0

    def add_concept(self, concept: Concept):
        """Add a concept to the toolkit."""
        self.concepts[concept.name] = concept

    def add_program(self, program: Program):
        """Add a successful program to the toolkit for reuse."""
        self.programs.append(program)

    def get_best_concepts(self, n: int = 10) -> list[Concept]:
        """Get the most successful concepts (exploit known patterns)."""
        ranked = sorted(
            self.concepts.values(),
            key=lambda c: c.success_rate,
            reverse=True
        )
        return ranked[:n]

    def compose(self, concept_a: Concept, concept_b: Concept) -> Concept:
        """Compose two concepts into a higher-level concept.

        This is the core of Pillar 3: old outputs become new inputs.
        """
        self._concept_counter += 1
        name = f"{concept_a.name}→{concept_b.name}"

        def composed_impl(grid: Grid) -> Optional[Grid]:
            intermediate = concept_a.apply(grid)
            if intermediate is None:
                return None
            return concept_b.apply(intermediate)

        composed = Concept(
            kind="composed",
            name=name,
            implementation=composed_impl,
            children=[concept_a, concept_b]
        )
        return composed

    def get_concepts_by_kind(self, kind: str) -> list[Concept]:
        return [c for c in self.concepts.values() if c.kind == kind]

    @property
    def size(self) -> int:
        return len(self.concepts)


class Archive:
    """Type 2 Memory: Task-specific context (time-bound).

    Records which programs worked for which tasks, enabling
    cross-task transfer learning.
    """
    def __init__(self):
        self.task_solutions: dict[str, list[Program]] = {}
        self.task_features: dict[str, dict] = {}
        self.history: list[dict] = []

    def record_solution(self, task_id: str, program: Program, score: float):
        """Record that a program achieved a score on a task."""
        if task_id not in self.task_solutions:
            self.task_solutions[task_id] = []
        self.task_solutions[task_id].append(program)
        self.history.append({
            "task_id": task_id,
            "program": program.name,
            "score": score,
        })

    def record_features(self, task_id: str, features: dict):
        """Record extracted features of a task for similarity matching."""
        self.task_features[task_id] = features

    def find_similar_tasks(self, features: dict, n: int = 5) -> list[str]:
        """Find tasks with similar features (for cross-task transfer)."""
        if not self.task_features:
            return []

        similarities = []
        for tid, tf in self.task_features.items():
            # Simple feature overlap score
            common_keys = set(features.keys()) & set(tf.keys())
            if not common_keys:
                continue
            score = sum(
                1.0 for k in common_keys
                if features.get(k) == tf.get(k)
            ) / max(len(features), len(tf))
            similarities.append((tid, score))

        similarities.sort(key=lambda x: x[1], reverse=True)
        return [tid for tid, _ in similarities[:n]]

    def get_programs_for_similar_tasks(self, features: dict) -> list[Program]:
        """Get programs that worked on similar tasks (cross-task transfer)."""
        similar = self.find_similar_tasks(features)
        programs = []
        for tid in similar:
            programs.extend(self.task_solutions.get(tid, []))
        return programs
