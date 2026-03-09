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
        """Generate a random program (sequence of concepts).

        Each step has a chance of being a conditional (if predicates are
        available), controlled by self.conditional_rate.
        """
        length = random.randint(1, max_len or self.max_program_length)
        steps = []
        for _ in range(length):
            if random.random() < self.conditional_rate and self._get_predicates():
                cond = self._random_conditional()
                if cond is not None:
                    steps.append(cond)
                    continue
            steps.append(self._random_concept())
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
    # ESSENTIAL_PAIR_CONCEPTS: concepts that score LOW individually (won't appear in
    # top_k=20 naturally) but are highly valuable as a first or second step in pairs.
    # Keep this list SMALL (target ≤25) to control O(N²) pair-search cost.
    # Rule: only include if avg single-prim rank > 40 AND demonstrated pair-solve value.
    ESSENTIAL_PAIR_CONCEPTS = frozenset([
        # Structural transformations — low individual score, strong as second steps
        "identity",             # catch-all; needed when first step is the transform
        "fill_enclosed",        # flood-fill enclosed regions — rarely top-k alone
        "split_sep_overlay",    # grid-separator overlay
        "split_sep_xor",        # grid-separator XOR
        "compress_rows",        # deduplicate rows
        "compress_cols",        # deduplicate cols
        "max_color_per_cell",   # cell-wise max across examples
        "min_color_per_cell",   # cell-wise min across examples
        # Tile/pattern (v0.13)
        "fill_by_symmetry",
        "fill_tile_pattern",
        "spread_in_lanes_h",
        "spread_in_lanes_v",
        # V14: demonstrated pair-solve value
        "gravity_toward_color",
        "fill_holes_in_objects",
        "connect_pixels_to_rect",
        "recolor_2nd_to_3rd",
        "extend_nonzero_fill_row",
        "complete_pattern_4way",
        # V15: demonstrated pair-solve value
        "recolor_isolated_to_nearest",
        "mirror_h_merge",
        "mirror_v_merge",
        "complete_symmetry_diagonal",
        "sort_rows_by_value",
        "remove_color_noise",
        # V15 speed-fix: re-added after confirmed lost solve (0b148d64)
        "crop_nonzero",             # low solo rank but critical as 2nd step
        # V16: demonstrated pair-solve value
        "fill_stripe_gaps_h",               # 22168020 solved; 40853293 as 1st step
        "fill_stripe_gaps_v",               # complement of fill_stripe_gaps_h
        "propagate_color_v",                # d037b0a7 solved
        "complete_tile_from_modal_row",     # 7f4411dc solved as 2nd step
        "recolor_smallest_obj_in_each_row", # ba97ae07 solved
        "snap_isolated_to_rect_boundary",   # d89b689b solved
        # V17: only add to ESSENTIAL after confirmed pair-solve value
        # (keeping slim — evolution phase will find these when needed)
        # V19: confirmed solves (22eb0ac0, ded97339, 32597951 all test-confirmed)
        "extend_lines_to_contact",              # 22eb0ac0, ded97339 solved; 40853293 improved
        "recolor_nonzero_inside_8_bbox_to_3",   # 32597951 solved
        # V20: confirmed pair-solve value (a79310a0 test-confirmed)
        "shift_down_1",                         # 25ff71a9 solved; a79310a0 pair-solved
        # V21: confirmed pair-solve value
        "swap_most_least",                      # b94a9452 pair-solved (tc=True)
    ])

    def try_all_pairs(
        self,
        task: dict,
        cache: "TaskCache | None" = None,
        top_k: int = 40,
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
                   (default 40 — wider search catches more solutions)

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
        pair_score_threshold: float = 0.80,
    ) -> Optional[Program]:
        """Targeted triple search: extend a near-miss pair with a third step.

        When pair exhaustion finds a pair that scores ≥ pair_score_threshold
        but < 0.99, we try every concept as:
          1. Appended third step:  pair_a → pair_b → concept
          2. Prepended first step: concept → pair_a → pair_b

        This costs 2×N scoring calls, only paid when concrete evidence exists.

        Args:
            best_pair:            Best pair from try_all_pairs (may be None)
            cache:                Pre-converted TaskCache
            pair_score_threshold: Only extend pairs scoring at least this high

        Returns:
            Best 3-step program found, or None if pair too weak / no pair.
        """
        if best_pair is None or best_pair.fitness < pair_score_threshold:
            return None

        best_triple = None
        best_score  = 0.0
        pair_steps  = list(best_pair.steps)

        for concept in self.toolkit.concepts.values():
            if concept.kind == "predicate":
                continue

            # Try appending: pair → concept
            prog  = Program(pair_steps + [concept])
            score = cache.score_program(prog)
            if score > best_score:
                best_score  = score
                best_triple = prog
                best_triple.fitness = score
                if score >= 0.99:
                    return best_triple

            # Try prepending: concept → pair
            prog  = Program([concept] + pair_steps)
            score = cache.score_program(prog)
            if score > best_score:
                best_score  = score
                best_triple = prog
                best_triple.fitness = score
                if score >= 0.99:
                    return best_triple

        return best_triple

    def try_all_triples(
        self,
        task: dict,
        cache: "TaskCache | None" = None,
        top_k: int = 15,
    ) -> Optional[Program]:
        """Exhaustively try all triples of top-scoring singles.

        This is the key unlocking insight: ~15% of ARC tasks need exactly
        3 steps, and 20³ = 8,000 combinations is cheap to exhaust.
        Combined with the essential pair concepts, this covers the most
        common 3-step compositions deterministically.

        Args:
            task: ARC task dict
            cache: Pre-computed TaskCache
            top_k: Number of top singles to use in triple combinations

        Returns:
            Best 3-step program found, or None if nothing scored well.
        """
        if cache is None:
            cache = TaskCache(task)

        # Score all single primitives (reuses same logic as try_all_pairs)
        singles: list[tuple[float, Concept]] = []
        for concept in self.toolkit.concepts.values():
            if concept.kind == "predicate":
                continue
            prog = Program([concept])
            score = cache.score_program(prog)
            singles.append((score, concept))

        singles.sort(key=lambda x: x[0], reverse=True)
        top_concepts = [c for _, c in singles[:top_k]]

        # Add essential structural primitives
        top_names = {c.name for c in top_concepts}
        for name in self.ESSENTIAL_PAIR_CONCEPTS:
            if name not in top_names and name in self.toolkit.concepts:
                top_concepts.append(self.toolkit.concepts[name])

        best_prog = None
        best_score = 0.0

        for a in top_concepts:
            for b in top_concepts:
                for c in top_concepts:
                    # Skip degenerate A→A→A triples: equivalent to single A
                    # (already tested). Saves ~N evaluations per loop.
                    if a.name == b.name == c.name:
                        continue
                    prog = Program([a, b, c])
                    score = cache.score_program(prog)
                    if score > best_score:
                        best_score = score
                        best_prog = prog
                        best_prog.fitness = score
                        if score >= 0.99:
                            return best_prog

        return best_prog

    def try_near_miss_refinement(
        self,
        candidates: list[tuple["Program", str]],
        cache: "TaskCache",
        score_threshold: float = 0.80,
    ) -> Optional[Program]:
        """Refine near-miss programs by trying single-step fixes.

        For each candidate scoring >= threshold but < 0.99, try:
          1. Appending every single concept
          2. Prepending every single concept
          3. Replacing each step with every other concept

        This is the highest-ROI search: programs at 0.8-0.99 are almost
        right — they often need just one color swap, crop, or flip.

        Args:
            candidates: List of (program, method) tuples from earlier search
            cache: Pre-computed TaskCache
            score_threshold: Only refine programs scoring at least this high

        Returns:
            Best refined program if pixel-perfect, else None.
        """
        # Collect near-misses (high score but not pixel-perfect)
        near_misses = []
        for prog, method in candidates:
            if prog.fitness >= score_threshold and not cache.is_pixel_perfect(prog):
                near_misses.append(prog)

        if not near_misses:
            return None

        # Get all non-predicate concepts
        all_concepts = [c for c in self.toolkit.concepts.values()
                        if c.kind != "predicate"]

        best_prog = None
        best_score = 0.0

        for near_miss in near_misses:
            steps = list(near_miss.steps)

            # Try appending
            if len(steps) < self.max_program_length:
                for concept in all_concepts:
                    prog = Program(steps + [concept])
                    score = cache.score_program(prog)
                    if score > best_score:
                        best_score = score
                        best_prog = prog
                        best_prog.fitness = score
                        if score >= 0.99 and cache.is_pixel_perfect(best_prog):
                            return best_prog

            # Try prepending
            if len(steps) < self.max_program_length:
                for concept in all_concepts:
                    prog = Program([concept] + steps)
                    score = cache.score_program(prog)
                    if score > best_score:
                        best_score = score
                        best_prog = prog
                        best_prog.fitness = score
                        if score >= 0.99 and cache.is_pixel_perfect(best_prog):
                            return best_prog

            # Try replacing each step
            for i in range(len(steps)):
                for concept in all_concepts:
                    if concept.name == steps[i].name:
                        continue  # skip identity replacement
                    new_steps = steps[:i] + [concept] + steps[i + 1:]
                    prog = Program(new_steps)
                    score = cache.score_program(prog)
                    if score > best_score:
                        best_score = score
                        best_prog = prog
                        best_prog.fitness = score
                        if score >= 0.99 and cache.is_pixel_perfect(best_prog):
                            return best_prog

        return best_prog

    def try_color_fix(
        self,
        program: "Program",
        cache: "TaskCache",
    ) -> Optional[Program]:
        """Try to fix a near-miss by applying a color remapping to its output.

        Many ARC near-misses differ from the target by a consistent color
        substitution (e.g., all 3s should be 5s). This method:
          1. Runs the program on each training input
          2. Compares output vs expected pixel-by-pixel
          3. Infers a color remapping from mismatches
          4. Wraps program + remap into a new Program

        Returns a pixel-perfect program if the color fix works, else None.
        """
        import numpy as np

        if cache.n_examples == 0:
            return None

        # Collect mismatch color pairs across all training examples
        from collections import Counter
        color_map_votes: Counter = Counter()
        has_structural_mismatch = False

        for i in range(cache.n_examples):
            inp = cache._inputs[i]
            expected = cache._expected[i]
            try:
                result = program.execute(inp)
                if result is None:
                    return None
                result = np.array(result)
            except Exception:
                return None

            if result.shape != expected.shape:
                has_structural_mismatch = True
                break

            # Find mismatched positions
            diff_mask = result != expected
            if not diff_mask.any():
                continue  # this example already matches

            # Collect (got -> want) pairs
            got_colors = result[diff_mask]
            want_colors = expected[diff_mask]
            for g, w in zip(got_colors.flat, want_colors.flat):
                if g != w:
                    color_map_votes[(int(g), int(w))] += 1

        if has_structural_mismatch or not color_map_votes:
            return None

        # Build a consistent color remapping: for each "got" color,
        # pick the most common "want" color
        remap: dict[int, int] = {}
        by_got: dict[int, Counter] = {}
        for (g, w), count in color_map_votes.items():
            if g not in by_got:
                by_got[g] = Counter()
            by_got[g][w] += count
        for g, votes in by_got.items():
            best_w, _ = votes.most_common(1)[0]
            remap[g] = best_w

        if not remap:
            return None

        # Check that the remap is consistent (no conflicts)
        # A conflict is when the same "got" color needs to map to different
        # "want" colors in different positions
        for g, votes in by_got.items():
            if len(votes) > 1:
                top_count = votes.most_common(1)[0][1]
                total = sum(votes.values())
                # If the top vote has <80% of cases, the remap is ambiguous
                if top_count / total < 0.80:
                    return None

        # Find or create a concept that does this specific remap
        from arc_agent.concepts import Concept
        def make_remap_fn(mapping):
            def remap_fn(grid):
                import numpy as np
                g = np.array(grid)
                result = g.copy()
                for old_c, new_c in mapping.items():
                    result[g == old_c] = new_c
                return result
            return remap_fn

        remap_concept = Concept(
            kind="derived",
            name=f"color_remap_{'_'.join(f'{k}to{v}' for k, v in sorted(remap.items()))}",
            implementation=make_remap_fn(remap),
        )

        # Build fixed program: original steps + remap
        fixed = Program(list(program.steps) + [remap_concept])
        score = cache.score_program(fixed)
        fixed.fitness = score

        if score >= 0.99 and cache.is_pixel_perfect(fixed):
            return fixed

        return None

    # ────────────────────────────────────────────────────────────────
    # CONDITIONAL SEARCH (deterministic, no evolution)
    # ────────────────────────────────────────────────────────────────
    # Many ARC tasks require branching: "if condition then transform_A
    # else transform_B". The random evolution discovers these slowly;
    # exhaustive search finds them in seconds.
    # ────────────────────────────────────────────────────────────────

    def try_conditional_singles(
        self,
        task: dict,
        cache: "TaskCache | None" = None,
        top_k: int = 15,
    ) -> Optional[Program]:
        """Exhaustively try single-step conditional programs with optimizations.

        For each predicate P and each pair of top primitives (A, B):
            if P(input) → A(input)  else → B(input)

        Optimizations:
        1. PREDICATE PRE-FILTERING: Skip predicates that return the same value
           for all inputs (non-branching). These are redundant with single
           concepts already tested.
        2. BRANCH GROUPING: Partition inputs by predicate outcome. Score each
           concept independently on each group, then combine the best scores.
           This allows early pruning of bad concept combinations.
        3. EARLY EXIT: If best possible score (perfect on remaining examples)
           can't beat current best, skip this predicate.

        Original complexity: P × top_k² where P = # predicates (17 × 225 = 3,825).
        Optimized complexity: P' × top_k² where P' = non-trivial predicates (<17),
        plus O(P × N × top_k) for per-group scoring.

        Args:
            task: ARC task dict
            cache: Pre-converted TaskCache
            top_k: Top primitives to try as branches

        Returns:
            Best single-conditional program, or None.
        """
        if cache is None:
            cache = TaskCache(task)

        predicates = self._get_predicates()
        if not predicates:
            return None

        # Score all single primitives, keep top-k
        singles: list[tuple[float, Concept]] = []
        for concept in self.toolkit.concepts.values():
            if concept.kind == "predicate":
                continue
            prog = Program([concept])
            score = cache.score_program(prog)
            singles.append((score, concept))

        singles.sort(key=lambda x: x[0], reverse=True)
        top_concepts = [c for _, c in singles[:top_k]]

        # Pre-compute input groups for each predicate (reused below)
        # predicate_groups[i] = (true_indices, false_indices, is_trivial)
        predicate_groups = []
        for pred in predicates:
            true_indices = []
            false_indices = []
            for idx, inp in enumerate(cache._inputs):
                try:
                    result = pred(inp)
                    if result:
                        true_indices.append(idx)
                    else:
                        false_indices.append(idx)
                except Exception:
                    # Predicate failed; treat as False
                    false_indices.append(idx)

            # OPTIMIZATION 1: Skip predicates that always return same value
            # (no branching = already tested as single concepts)
            is_trivial = len(true_indices) == 0 or len(false_indices) == 0
            predicate_groups.append((true_indices, false_indices, is_trivial))

        best_prog = None
        best_score = 0.0

        for pred_idx, pred in enumerate(predicates):
            true_indices, false_indices, is_trivial = predicate_groups[pred_idx]

            # Skip non-branching predicates (OPTIMIZATION 1)
            if is_trivial:
                continue

            # OPTIMIZATION 3: Early exit check
            # Best possible score = 1.0 if we had perfect concepts on each branch
            # If this can't beat best_score, skip entire predicate
            if best_score >= 0.99:
                return best_prog  # Already found near-perfect solution

            # OPTIMIZATION 2: Pre-score each concept on each group
            # Use concept indices as keys since Concept objects are not hashable
            concept_scores = []  # list of (concept, true_score, false_score)
            n_true = len(true_indices) if true_indices else 1  # avoid div by zero
            n_false = len(false_indices) if false_indices else 1

            for c_idx, c in enumerate(top_concepts):
                true_score = 0.0
                false_score = 0.0

                # Score on true branch
                for idx in true_indices:
                    inp = cache._inputs[idx]
                    exp = cache._expected[idx]
                    exp_h, exp_w = cache._exp_dims[idx]
                    predicted = c.apply(inp)
                    from .scorer import _safe_to_np, _structural_similarity_np
                    p = _safe_to_np(predicted)
                    if p is not None:
                        pred_h, pred_w = p.shape
                        true_score += _structural_similarity_np(p, exp, pred_h, pred_w, exp_h, exp_w)

                # Score on false branch
                for idx in false_indices:
                    inp = cache._inputs[idx]
                    exp = cache._expected[idx]
                    exp_h, exp_w = cache._exp_dims[idx]
                    predicted = c.apply(inp)
                    from .scorer import _safe_to_np, _structural_similarity_np
                    p = _safe_to_np(predicted)
                    if p is not None:
                        pred_h, pred_w = p.shape
                        false_score += _structural_similarity_np(p, exp, pred_h, pred_w, exp_h, exp_w)

                concept_scores.append((c, true_score / n_true, false_score / n_false))

            # Try best combinations per group (OPTIMIZATION 2 continued)
            # For each branch, rank concepts by their per-group score
            true_ranked = sorted(concept_scores, key=lambda x: x[1], reverse=True)
            false_ranked = sorted(concept_scores, key=lambda x: x[2], reverse=True)

            # Prune to top candidates per branch to reduce branching
            best_true = [x[0] for x in true_ranked[:min(5, len(true_ranked))]]
            best_false = [x[0] for x in false_ranked[:min(5, len(false_ranked))]]

            for then_c in best_true:
                for else_c in best_false:
                    if then_c is else_c:
                        continue
                    cond = ConditionalConcept(pred, then_c, else_c)
                    prog = Program([cond])
                    score = cache.score_program(prog)
                    if score > best_score:
                        best_score = score
                        best_prog = prog
                        best_prog.fitness = score
                        if score >= 0.99:
                            return best_prog

        return best_prog

    def try_conditional_pairs(
        self,
        task: dict,
        cache: "TaskCache | None" = None,
        top_k: int = 10,
    ) -> Optional[Program]:
        """Try 2-step programs involving conditionals with optimizations.

        Combines top single conditionals with top primitives:
            conditional → primitive  (post-process)
            primitive → conditional  (pre-process)

        Optimizations:
        1. PREDICATE PRE-FILTERING: Skip trivial predicates (same value for all inputs)
        2. BRANCH GROUPING: Score concepts per input group (like try_conditional_singles)
        3. AGGRESSIVE PRUNING: Keep only top 5 conditional + 5 primitive candidates
           for pairing (reduces O(P × top_k²) to O(5 × 5) per predicate)

        Args:
            task: ARC task dict
            cache: Pre-converted TaskCache
            top_k: Top primitives/conditionals to combine

        Returns:
            Best conditional pair, or None.
        """
        if cache is None:
            cache = TaskCache(task)

        predicates = self._get_predicates()
        if not predicates:
            return None

        # Get top primitives
        singles: list[tuple[float, Concept]] = []
        for concept in self.toolkit.concepts.values():
            if concept.kind == "predicate":
                continue
            prog = Program([concept])
            score = cache.score_program(prog)
            singles.append((score, concept))
        singles.sort(key=lambda x: x[0], reverse=True)
        top_concepts = [c for _, c in singles[:top_k]]

        # Pre-compute input groups for each predicate (reused below)
        # predicate_groups[i] = (true_indices, false_indices, is_trivial)
        predicate_groups = []
        for pred in predicates:
            true_indices = []
            false_indices = []
            for idx, inp in enumerate(cache._inputs):
                try:
                    result = pred(inp)
                    if result:
                        true_indices.append(idx)
                    else:
                        false_indices.append(idx)
                except Exception:
                    false_indices.append(idx)

            is_trivial = len(true_indices) == 0 or len(false_indices) == 0
            predicate_groups.append((true_indices, false_indices, is_trivial))

        # Build best conditional per predicate (OPTIMIZATION 2: branch grouping)
        best_conds: list[tuple[float, ConditionalConcept]] = []

        for pred_idx, pred in enumerate(predicates):
            true_indices, false_indices, is_trivial = predicate_groups[pred_idx]

            # Skip trivial predicates (OPTIMIZATION 1)
            if is_trivial:
                continue

            # OPTIMIZATION 2: Score concepts per branch
            n_true = len(true_indices) if true_indices else 1
            n_false = len(false_indices) if false_indices else 1
            concept_scores = []  # list of (concept, true_score, false_score)

            for c_idx, c in enumerate(top_concepts):
                true_score = 0.0
                false_score = 0.0

                for idx in true_indices:
                    inp = cache._inputs[idx]
                    exp = cache._expected[idx]
                    exp_h, exp_w = cache._exp_dims[idx]
                    predicted = c.apply(inp)
                    from .scorer import _safe_to_np, _structural_similarity_np
                    p = _safe_to_np(predicted)
                    if p is not None:
                        pred_h, pred_w = p.shape
                        true_score += _structural_similarity_np(p, exp, pred_h, pred_w, exp_h, exp_w)

                for idx in false_indices:
                    inp = cache._inputs[idx]
                    exp = cache._expected[idx]
                    exp_h, exp_w = cache._exp_dims[idx]
                    predicted = c.apply(inp)
                    from .scorer import _safe_to_np, _structural_similarity_np
                    p = _safe_to_np(predicted)
                    if p is not None:
                        pred_h, pred_w = p.shape
                        false_score += _structural_similarity_np(p, exp, pred_h, pred_w, exp_h, exp_w)

                concept_scores.append((c, true_score / n_true, false_score / n_false))

            # Rank by per-group scores
            true_ranked = sorted(concept_scores, key=lambda x: x[1], reverse=True)
            false_ranked = sorted(concept_scores, key=lambda x: x[2], reverse=True)

            best_true = [x[0] for x in true_ranked[:5]]
            best_false = [x[0] for x in false_ranked[:5]]

            # Try all combinations within pruned sets
            for then_c in best_true:
                for else_c in best_false:
                    if then_c is else_c:
                        continue
                    cond = ConditionalConcept(pred, then_c, else_c)
                    prog = Program([cond])
                    score = cache.score_program(prog)
                    if score > 0.3:  # Only keep above threshold
                        best_conds.append((score, cond))

        # Sort and keep top-5 (OPTIMIZATION 3: aggressive pruning)
        best_conds.sort(key=lambda x: x[0], reverse=True)
        best_conds = best_conds[:5]

        best_prog = None
        best_score = 0.0

        # Pair with top-5 primitives only (OPTIMIZATION 3)
        pair_concepts = top_concepts[:5]

        for score, cond in best_conds:
            for prim in pair_concepts:
                # cond → prim
                prog = Program([cond, prim])
                score = cache.score_program(prog)
                if score > best_score:
                    best_score = score
                    best_prog = prog
                    best_prog.fitness = score
                    if score >= 0.99:
                        return best_prog

                # prim → cond
                prog = Program([prim, cond])
                score = cache.score_program(prog)
                if score > best_score:
                    best_score = score
                    best_prog = prog
                    best_prog.fitness = score
                    if score >= 0.99:
                        return best_prog

        return best_prog

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
