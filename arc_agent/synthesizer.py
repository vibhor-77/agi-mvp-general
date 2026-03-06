"""
Pillar 2: Approximability — The Program Synthesizer

Iteratively refine candidate programs to converge toward the correct
transformation. Uses evolutionary search (mutation + crossover + selection)
which directly maps to Vibhor's biological evolution analogy.

The key insight from Vibhor's framework: the search landscape must have
approximability — partial solutions guide us toward full solutions.
"""
from __future__ import annotations
import random
import copy
from typing import Optional
from .concepts import Concept, ConditionalConcept, Program, Toolkit, Grid, Predicate
from .scorer import score_program_on_task, score_population_on_task


class ProgramSynthesizer:
    """Evolutionary program synthesizer.

    Generates candidate programs by composing concepts from the Toolkit,
    then iteratively refines them using feedback (Pillar 1) to converge
    toward the correct transformation (Pillar 2: Approximability).
    """

    def __init__(
        self,
        toolkit: Toolkit,
        population_size: int = 50,
        max_program_length: int = 4,
        mutation_rate: float = 0.3,
        crossover_rate: float = 0.2,
        elite_fraction: float = 0.1,
        conditional_rate: float = 0.1,
    ):
        self.toolkit = toolkit
        self.population_size = population_size
        self.max_program_length = max_program_length
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elite_fraction = elite_fraction
        self.conditional_rate = conditional_rate
        # Cache available predicates from toolkit
        self._predicates: Optional[list[Predicate]] = None

    def _get_predicates(self) -> list[Predicate]:
        """Get all available predicates from the toolkit.

        Predicates are stored as Concept objects with kind="predicate".
        """
        if self._predicates is None:
            self._predicates = []
            for concept in self.toolkit.concepts.values():
                if concept.kind == "predicate":
                    # Extract the actual predicate function
                    # For predicates, implementation is the predicate function
                    if callable(concept.implementation):
                        self._predicates.append(concept.implementation)
        return self._predicates

    def _random_concept(self) -> Concept:
        """Pick a random concept from the toolkit.

        Excludes predicates (which are only for conditionals).
        """
        concepts = [c for c in self.toolkit.concepts.values() if c.kind != "predicate"]
        return random.choice(concepts) if concepts else list(self.toolkit.concepts.values())[0]

    def _random_conditional(self) -> Optional[ConditionalConcept]:
        """Create a random conditional concept from random predicate and branches.

        Returns None if no predicates are available.
        """
        predicates = self._get_predicates()
        if not predicates:
            return None

        predicate = random.choice(predicates)
        then_concept = self._random_concept()
        else_concept = self._random_concept()

        return ConditionalConcept(predicate, then_concept, else_concept)

    def _random_program(self, max_len: Optional[int] = None) -> Program:
        """Generate a random program (sequence of concepts)."""
        length = random.randint(1, max_len or self.max_program_length)
        steps = [self._random_concept() for _ in range(length)]
        return Program(steps)

    def generate_initial_population(self) -> list[Program]:
        """Generate the initial population of random programs.

        Includes single-concept programs (to test each primitive alone),
        random compositions, and some conditional programs.
        Excludes predicates (which are only for conditionals).
        """
        population = []

        # First: try every single primitive alone (skip predicates)
        for concept in self.toolkit.concepts.values():
            if concept.kind == "predicate":
                continue
            population.append(Program([concept]))

        # Then: generate random compositions and conditionals to fill population
        while len(population) < self.population_size:
            if random.random() < self.conditional_rate:
                # Try to create a conditional program
                cond = self._random_conditional()
                if cond is not None:
                    population.append(Program([cond]))
                    continue
            population.append(self._random_program())

        return population[:self.population_size]

    def mutate(self, program: Program) -> Program:
        """Mutate a program by changing, adding, or removing a step.

        Mutation types (Pillar 4: Exploration):
        1. Replace a random step with a different concept
        2. Insert a new step at a random position
        3. Remove a random step (if length > 1)
        4. Swap two adjacent steps
        5. Replace a step with a conditional (if predicates available)
        """
        steps = [s for s in program.steps]  # shallow copy of list

        # Include "conditional" as an option if predicates are available
        mutation_types = ["replace", "insert", "remove", "swap"]
        if self._get_predicates():
            mutation_types.append("conditional")

        mutation_type = random.choice(mutation_types)

        if mutation_type == "replace" and steps:
            idx = random.randint(0, len(steps) - 1)
            steps[idx] = self._random_concept()

        elif mutation_type == "insert" and len(steps) < self.max_program_length:
            idx = random.randint(0, len(steps))
            steps.insert(idx, self._random_concept())

        elif mutation_type == "remove" and len(steps) > 1:
            idx = random.randint(0, len(steps) - 1)
            steps.pop(idx)

        elif mutation_type == "swap" and len(steps) >= 2:
            idx = random.randint(0, len(steps) - 2)
            steps[idx], steps[idx + 1] = steps[idx + 1], steps[idx]

        elif mutation_type == "conditional" and steps:
            idx = random.randint(0, len(steps) - 1)
            cond = self._random_conditional()
            if cond is not None:
                steps[idx] = cond

        return Program(steps)

    def crossover(self, parent_a: Program, parent_b: Program) -> Program:
        """Combine two programs via crossover.

        Takes the first half of parent_a and second half of parent_b.
        This is the composability principle in action — combining
        successful sub-sequences.
        """
        mid_a = max(1, len(parent_a) // 2)
        mid_b = max(1, len(parent_b) // 2)

        child_steps = parent_a.steps[:mid_a] + parent_b.steps[mid_b:]

        # Enforce max length
        if len(child_steps) > self.max_program_length:
            child_steps = child_steps[:self.max_program_length]

        return Program(child_steps)

    def evolve_generation(
        self,
        population: list[Program],
        task: dict,
    ) -> list[Program]:
        """Evolve one generation: score → select → mutate → crossover.

        This is the core learning loop:
        1. FEEDBACK: Score each program on the task (Pillar 1)
        2. SELECT: Keep the best (Pillar 2: exploit what's close)
        3. MUTATE: Vary the survivors (Pillar 4: explore)
        4. CROSSOVER: Combine successful programs (Pillar 3: compose)
        """
        # Score all programs in one pass (FEEDBACK LOOP)
        # score_population_on_task iterates training examples once per program,
        # amortizing Python overhead vs calling score_program_on_task individually.
        scores = score_population_on_task(population, task)
        for program, score in zip(population, scores):
            program.fitness = score

        # Sort by fitness (APPROXIMABILITY — better programs survive)
        population.sort(key=lambda p: p.fitness, reverse=True)

        # Elite selection
        n_elite = max(1, int(len(population) * self.elite_fraction))
        elite = population[:n_elite]
        new_population = list(elite)

        # Fill rest of population
        while len(new_population) < self.population_size:
            # Tournament selection
            candidates = random.sample(
                population[:len(population) // 2],
                min(3, len(population) // 2)
            )
            parent = max(candidates, key=lambda p: p.fitness)

            if random.random() < self.crossover_rate and len(elite) >= 2:
                other = random.choice(elite)
                child = self.crossover(parent, other)
            elif random.random() < self.mutation_rate:
                child = self.mutate(parent)
            else:
                child = self.mutate(parent)

            new_population.append(child)

        return new_population[:self.population_size]

    def synthesize(
        self,
        task: dict,
        max_generations: int = 30,
        target_score: float = 1.0,
        seed_programs: Optional[list[Program]] = None,
        verbose: bool = False,
    ) -> tuple[Program, list[dict]]:
        """Run the full evolutionary synthesis loop.

        This is the complete learning loop from Vibhor's framework:
        tight feedback → approximability → composition → exploration

        Args:
            task: ARC task dict with 'train' examples
            max_generations: Maximum evolution cycles
            target_score: Stop if we reach this score
            seed_programs: Pre-existing programs to seed population with
            verbose: Print progress

        Returns:
            (best_program, history) — the best found and evolution log
        """
        # Initialize population
        population = self.generate_initial_population()

        # Seed with known programs if available (cross-task transfer)
        if seed_programs:
            for sp in seed_programs[:self.population_size // 4]:
                population.append(sp)
                # Also add mutations of seed programs
                population.append(self.mutate(sp))

        history = []
        best_ever = None
        best_ever_score = 0.0

        for gen in range(max_generations):
            population = self.evolve_generation(population, task)

            best = population[0]
            avg_fitness = sum(p.fitness for p in population) / len(population)

            history.append({
                "generation": gen,
                "best_fitness": best.fitness,
                "avg_fitness": avg_fitness,
                "best_program": best.name,
                "population_size": len(population),
            })

            if best.fitness > best_ever_score:
                best_ever = best
                best_ever_score = best.fitness

            if verbose and gen % 5 == 0:
                print(f"  Gen {gen:3d}: best={best.fitness:.3f} "
                      f"avg={avg_fitness:.3f} prog={best.name[:50]}")

            # Early stopping if we found a perfect solution
            if best.fitness >= target_score:
                if verbose:
                    print(f"  ✓ Perfect solution found at generation {gen}!")
                break

        return best_ever, history
