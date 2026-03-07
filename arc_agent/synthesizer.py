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
from .scorer import TaskCache, score_program_on_task


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
        max_program_length: int = 5,
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
        cache: "TaskCache | None" = None,
    ) -> list[Program]:
        """Evolve one generation: score → select → mutate → crossover.

        This is the core learning loop:
        1. FEEDBACK: Score each program on the task (Pillar 1)
        2. SELECT: Keep the best (Pillar 2: exploit what's close)
        3. MUTATE: Vary the survivors (Pillar 4: explore)
        4. CROSSOVER: Combine successful programs (Pillar 3: compose)

        Args:
            cache: Pre-converted TaskCache. If supplied, expected outputs are
                   not re-converted from list-of-lists this generation (~40%
                   faster). Pass None to fall back to on-the-fly conversion.
        """
        # Score all programs in one pass (FEEDBACK LOOP).
        # If a TaskCache is provided, expected outputs were already converted
        # to numpy arrays once for the whole task — no redundant work.
        if cache is not None:
            scores = cache.score_population(population)
        else:
            scores = TaskCache(task).score_population(population)
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

    # Essential structural primitives that are always included in pair search.
    # These are common "second steps" that score poorly alone but are critical
    # in compositions (e.g. crop_nonzero after a color filter).
    ESSENTIAL_PAIR_CONCEPTS = frozenset([
        "identity", "crop_nonzero", "mirror_h", "mirror_v",
        "rotate_90_cw", "rotate_90_ccw", "rotate_180", "transpose",
        "fill_enclosed", "outline", "invert_colors",
        "scale_2x", "scale_3x", "tile_2x2",
        "get_top_half", "get_bottom_half", "get_left_half", "get_right_half",
        "get_interior", "get_border",
        "extract_largest", "extract_smallest",
        "gravity_down", "gravity_up", "gravity_left", "gravity_right",
        # Tile / pattern extraction (v0.9)
        "extract_repeating_tile", "extract_unique_block",
        "split_sep_overlay", "split_sep_xor",
        "compress_rows", "compress_cols",
        "max_color_per_cell", "min_color_per_cell",
    ])

    def try_all_pairs(
        self,
        task: dict,
        cache: "TaskCache | None" = None,
        top_k: int = 20,
    ) -> Optional[Program]:
        """Exhaustively try all pairs of top-scoring + essential primitives.

        Many ARC tasks are solved by exactly two steps (e.g. crop→mirror,
        fill→outline). Combines the top-k by individual score with a fixed
        set of essential structural primitives that are common second steps
        but may rank low individually.

        Args:
            task: ARC task dict
            cache: Pre-converted TaskCache
            top_k: Number of top single primitives to consider for pairing

        Returns:
            Best 2-step program found, or None if nothing scored well.
        """
        if cache is None:
            cache = TaskCache(task)

        # Score all single primitives
        singles: list[tuple[float, Concept]] = []
        for concept in self.toolkit.concepts.values():
            if concept.kind == "predicate":
                continue
            prog = Program([concept])
            score = cache.score_program(prog)
            singles.append((score, concept))

        # Keep top-k by score
        singles.sort(key=lambda x: x[0], reverse=True)
        top_concepts = [c for _, c in singles[:top_k]]

        # Add essential structural primitives that may not be in top-k
        top_names = {c.name for c in top_concepts}
        for name in self.ESSENTIAL_PAIR_CONCEPTS:
            if name not in top_names and name in self.toolkit.concepts:
                top_concepts.append(self.toolkit.concepts[name])

        best_prog = None
        best_score = 0.0

        for a in top_concepts:
            for b in top_concepts:
                prog = Program([a, b])
                score = cache.score_program(prog)
                if score > best_score:
                    best_score = score
                    best_prog = prog
                    best_prog.fitness = score
                    if score >= 0.99:
                        return best_prog

        return best_prog

    def try_best_triples(
        self,
        best_pair: Optional[Program],
        cache: "TaskCache",
        pair_score_threshold: float = 0.90,
    ) -> Optional[Program]:
        """Targeted triple search: extend a near-miss pair with a third step.

        When pair exhaustion finds a pair that scores ≥ pair_score_threshold
        but < 0.99, it means the program is close but missing one final
        transformation. We try every concept in the toolkit as that third
        step — O(N) scoring calls, only paid when concrete evidence exists.

        Accepts the already-computed best_pair to avoid re-running pair search
        (which would double the O(N²) cost on every task).

        Args:
            best_pair:            Best pair from try_all_pairs (may be None)
            cache:                Pre-converted TaskCache
            pair_score_threshold: Only extend pairs scoring at least this high

        Returns:
            Best 3-step program found, or None if pair too weak / no pair.
        """
        if best_pair is None or best_pair.fitness < pair_score_threshold:
            return None

        # Extend with every non-predicate concept as a third step (~171 calls)
        best_triple = None
        best_score  = 0.0
        for concept in self.toolkit.concepts.values():
            if concept.kind == "predicate":
                continue
            prog  = Program(list(best_pair.steps) + [concept])
            score = cache.score_program(prog)
            if score > best_score:
                best_score  = score
                best_triple = prog
                best_triple.fitness = score
                if score >= 0.99:
                    return best_triple

        return best_triple

    def hill_climb(
        self,
        program: Program,
        cache: "TaskCache",
        max_steps: int = 50,
    ) -> Program:
        """Stochastic hill climbing to refine a near-miss program.

        Makes small mutations and keeps improvements. More focused than
        full evolution — useful when we're close (>0.90) but not perfect.
        """
        current = program
        current_score = cache.score_program(current)

        for _ in range(max_steps):
            candidate = self.mutate(current)
            score = cache.score_program(candidate)
            if score > current_score:
                current = candidate
                current.fitness = score
                current_score = score
                if score >= 0.99:
                    break

        return current

    def synthesize(
        self,
        task: dict,
        max_generations: int = 30,
        target_score: float = 1.0,
        seed_programs: Optional[list[Program]] = None,
        verbose: bool = False,
        cache: "TaskCache | None" = None,
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
            cache: Pre-converted TaskCache (created once by solver, reused
                   across all generations). Pass None to auto-create.

        Returns:
            (best_program, history) — the best found and evolution log
        """
        # Pre-convert expected outputs once for all generations.
        # This is the key optimisation: np.array() is called once per
        # training example, not once per (program × generation × example).
        if cache is None:
            cache = TaskCache(task)

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
            population = self.evolve_generation(population, task, cache=cache)

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

        # Phase 3: Hill climbing refinement for near-misses
        if best_ever and 0.85 <= best_ever_score < 0.99:
            refined = self.hill_climb(best_ever, cache, max_steps=60)
            if refined.fitness > best_ever_score:
                best_ever = refined
                best_ever_score = refined.fitness
                if verbose:
                    print(f"  ↑ Hill climb improved to {best_ever_score:.3f}")

        return best_ever, history
