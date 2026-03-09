"""
The Main Solver — Integration of All Four Pillars

This is the complete learning loop from Vibhor's framework:
  1. FEEDBACK LOOPS: Test programs against examples, get scores
  2. APPROXIMABILITY: Iteratively refine using evolutionary search
  3. COMPOSABILITY: Build complex programs from simple concepts
  4. EXPLORATION: Balance exploit (known good) vs explore (novel)

The solver also implements the "cumulative culture" principle:
  - Successful programs become reusable concepts
  - Knowledge compounds across tasks (no "reset button")
"""
from __future__ import annotations
import time
from collections import defaultdict
from typing import Optional
from .concepts import Toolkit, Archive, Program, Concept, Grid
from .primitives import build_initial_toolkit
from .synthesizer import ProgramSynthesizer
from .explorer import ExplorationEngine
from .decompose import DecompositionEngine
from .scorer import (
    TaskCache,
    score_program_on_task,
    validate_on_test,
    extract_task_features,
)


class FourPillarsSolver:
    """The main AGI agent implementing Vibhor's 4 Pillars framework.

    Architecture:
      Toolkit (Pillar 3) ←→ Synthesizer (Pillar 2) ←→ Explorer (Pillar 4)
                              ↕
                          Scorer (Pillar 1)
    """

    def __init__(
        self,
        population_size: int = 60,
        max_generations: int = 30,
        max_program_length: int = 6,
        verbose: bool = True,
    ):
        # Initialize the dual memory system
        self.toolkit = build_initial_toolkit()
        self.archive = Archive()

        # Initialize the engines
        self.synthesizer = ProgramSynthesizer(
            toolkit=self.toolkit,
            population_size=population_size,
            max_program_length=max_program_length,
        )
        self.explorer = ExplorationEngine(
            toolkit=self.toolkit,
            archive=self.archive,
        )
        self.decomposer = DecompositionEngine(toolkit=self.toolkit)

        self.max_generations = max_generations
        self.verbose = verbose

        # Tracking metrics
        self.tasks_attempted = 0
        self.tasks_solved = 0
        self.tasks_partially_solved = 0  # >80% pixel accuracy
        self.solve_times: list[float] = []
        self.concept_growth: list[int] = []

    def solve_task(self, task: dict, task_id: str = "unknown",
                   mode: str = "train") -> dict:
        """Solve a single ARC task using all 4 pillars.

        Runs all search strategies exhaustively and collects all pixel-perfect
        candidates. No early exits — we always want the full picture.

        Args:
            task: The ARC task dict with 'train' and optionally 'test'.
            task_id: Identifier for this task.
            mode: Passed through for metadata only. The solver runs
                  identically in all modes.

        Returns:
            Dict with solve status, best program, score, and metadata.
        """
        start_time = time.time()
        self.tasks_attempted += 1

        if self.verbose:
            print(f"\n{'='*60}")
            print(f"Task: {task_id}")
            print(f"Training examples: {len(task.get('train', []))}")
            print(f"Toolkit size: {self.toolkit.size}")
            print(f"{'='*60}")

        # Pre-convert expected outputs once for all scoring in this task.
        # The cache is passed to _try_single_primitives and synthesizer.synthesize
        # so np.array() is called once per training example, not thousands of times.
        cache = TaskCache(task)

        # Step 1: Extract task features (for cross-task transfer)
        features = extract_task_features(task)
        self.archive.record_features(task_id, features)

        if self.verbose:
            print(f"Features: dims_same={features.get('same_dims')}, "
                  f"grows={features.get('grows')}, "
                  f"shrinks={features.get('shrinks')}, "
                  f"colors_in={features.get('in_colors')}, "
                  f"colors_out={features.get('out_colors')}")

        # Step 2: Generate seed programs (EXPLOIT past knowledge)
        seed_programs = self.explorer.generate_seed_programs(features)
        if self.verbose:
            print(f"Seed programs from transfer: {len(seed_programs)}")

        # Step 2b: Learn task-specific concepts from examples (MDL principle)
        # These are temporarily injected into the toolkit for this task's search.
        learned_concepts = self._learn_task_concepts(task)
        for lc in learned_concepts:
            self.toolkit.add_concept(lc)
        if self.verbose and learned_concepts:
            print(f"Learned concepts: {[lc.name for lc in learned_concepts]}")

        # ── Candidate collection ──────────────────────────────────────
        # Instead of returning on the first high-scoring program, we collect
        # all pixel-perfect candidates across search phases and pick the
        # simplest one (MDL principle: shortest program that explains the data).
        # Programs that score high on structural_similarity but aren't
        # pixel-perfect are kept as "best effort" but NOT declared solved.
        candidates: list[tuple[Program, str]] = []  # (program, method)

        # Near-miss pool: collect high-scoring non-perfect programs from
        # all search phases for later refinement (append/prepend/replace).
        near_miss_pool: list[tuple[Program, str]] = []

        # Step 3: Quick check — does any single primitive solve it?
        best_single = self._try_single_primitives(task, cache)
        if best_single and best_single.fitness >= 0.99:
            if cache.is_pixel_perfect(best_single):
                candidates.append((best_single, "single_primitive"))

        # Step 3a: Try all culture programs directly (cross-run transfer).
        culture_result: Optional[Program] = None
        if self.toolkit.programs:
            culture_result = self._try_culture_programs(task, cache)
            if culture_result and culture_result.fitness >= 0.99:
                if cache.is_pixel_perfect(culture_result):
                    candidates.append((culture_result, "culture_transfer"))

        # Step 3b: Try parameterized primitives (learn from examples).
        # These are templates that learn parameters from training examples,
        # enabling structural generalization (e.g., "map by frequency rank" not absolute colors).
        param_result = self._try_parameterized(task, cache)
        if param_result and param_result.fitness >= 0.99:
            if cache.is_pixel_perfect(param_result):
                candidates.append((param_result, "parameterized"))

        # No early exit — always try all search strategies to find
        # the best and most diverse set of candidates.

        # Step 3c: Conditional search — single conditionals (if-then-else)
        # Exhaustive: try every predicate × top primitive pairs as branches.
        cond_single = self.synthesizer.try_conditional_singles(task, cache, top_k=15)
        if cond_single and cond_single.fitness >= 0.99:
            if cache.is_pixel_perfect(cond_single):
                candidates.append((cond_single, "conditional_single"))

        # Step 3d: Try all pairs of top primitives (wider search: top-40)
        pair_result = self.synthesizer.try_all_pairs(task, cache, top_k=40)
        if pair_result and pair_result.fitness >= 0.99:
            if cache.is_pixel_perfect(pair_result):
                candidates.append((pair_result, "pair_exhaustion"))
        if pair_result and pair_result.fitness >= 0.80 and not cache.is_pixel_perfect(pair_result):
            near_miss_pool.append((pair_result, "pair_near_miss"))

        # Step 3e: Exhaustive triple search (top-15³ = 3,375 combinations)
        # This is the key unlocking step: ~15% of tasks need exactly 3 steps.
        # Exhaustive search guarantees finding all L=3 solutions in top-15.
        triple_result = self.synthesizer.try_all_triples(task, cache, top_k=15)
        if triple_result and triple_result.fitness >= 0.99:
            if cache.is_pixel_perfect(triple_result):
                candidates.append((triple_result, "triple_exhaustion"))
        if triple_result and triple_result.fitness >= 0.80 and not cache.is_pixel_perfect(triple_result):
            near_miss_pool.append((triple_result, "triple_near_miss"))

        # Step 3f: Near-miss triple extension (extend best pair with a third step)
        # Cheaper than exhaustive: only ~600 evals, catches cases where the
        # near-miss pair is just missing one step.
        if not triple_result or triple_result.fitness < 0.99:
            triple_ext = self.synthesizer.try_best_triples(
                pair_result, cache, pair_score_threshold=0.80
            )
            if triple_ext and triple_ext.fitness >= 0.99:
                if cache.is_pixel_perfect(triple_ext):
                    candidates.append((triple_ext, "triple_extension"))

        # Step 3g: Conditional pairs
        cond_pair = self.synthesizer.try_conditional_pairs(task, cache, top_k=10)
        if cond_pair and cond_pair.fitness >= 0.99:
            if cache.is_pixel_perfect(cond_pair):
                candidates.append((cond_pair, "conditional_pair"))

        # Step 3h: Object-centric reasoning
        # Only for same-dims tasks. Runs even if we have pair/triple candidates
        # since object rules may provide a better (more generalizable) solution.
        object_result = self._try_object_rules(task, cache)
        if object_result and object_result.fitness >= 0.99:
            if cache.is_pixel_perfect(object_result):
                candidates.append((object_result, "object_rules"))

        # Step 3i: Object decomposition (per-object transform search)
        # Apply each toolkit primitive to each object independently, reassemble.
        obj_decomp_result = self._try_object_decomposition(task, cache)
        if obj_decomp_result and obj_decomp_result.fitness >= 0.99:
            if cache.is_pixel_perfect(obj_decomp_result):
                candidates.append((obj_decomp_result, "object_decompose"))

        # Step 3j: DSL synthesis (novel transform search)
        # Synthesize Grid→Grid transforms from sub-primitive operations.
        dsl_result = self._try_dsl_synthesis(task, cache)
        if dsl_result and dsl_result.fitness >= 0.99:
            if cache.is_pixel_perfect(dsl_result):
                candidates.append((dsl_result, "dsl_synthesis"))
        if dsl_result and dsl_result.fitness >= 0.80 and not cache.is_pixel_perfect(dsl_result):
            near_miss_pool.append((dsl_result, "dsl_near_miss"))

        # Step 3k: Near-miss refinement (fix-up pass)
        # For programs scoring 0.8+ but not pixel-perfect, try single-step
        # fixes: append, prepend, or replace one step. Collects all high-
        # scoring programs from every search strategy above, deduplicates,
        # and tries each as a refinement base.
        near_miss_inputs: list[tuple[Program, str]] = list(candidates)
        all_near_miss_sources = [
            (pair_result, "pair"), (triple_result, "triple"),
            (dsl_result, "dsl"), (cond_single, "cond"),
            (cond_pair, "cond_pair"), (best_single, "single"),
            (param_result, "param"), (culture_result, "culture"),
            (object_result, "object_rules"),
            (obj_decomp_result, "object_decompose"),
        ]
        seen_keys: set[tuple[str, ...]] = set()
        for prog, label in all_near_miss_sources:
            if prog and prog.fitness >= 0.80 and not cache.is_pixel_perfect(prog):
                key = tuple(s.name for s in prog.steps)
                if key not in seen_keys:
                    seen_keys.add(key)
                    near_miss_inputs.append((prog, label))
        # Sort by fitness (highest first) for better early-exit
        near_miss_inputs.sort(key=lambda x: x[0].fitness, reverse=True)
        refined = self.synthesizer.try_near_miss_refinement(
            near_miss_inputs, cache, score_threshold=0.80
        )
        if refined and refined.fitness >= 0.99:
            if cache.is_pixel_perfect(refined):
                candidates.append((refined, "near_miss_refine"))

        # Inject best candidates into seeds for evolution, best first.
        seed_programs = list(seed_programs) if seed_programs else []
        if dsl_result and dsl_result.fitness >= 0.70 and \
                not cache.is_pixel_perfect(dsl_result):
            seed_programs.insert(0, dsl_result)
        if culture_result and culture_result.fitness > 0.85:
            seed_programs.insert(0, culture_result)
        if triple_result and triple_result.fitness > 0.85:
            seed_programs.insert(0, triple_result)
        if pair_result and pair_result.fitness > 0.85:
            seed_programs.insert(0, pair_result)

        # Step 4: Evolutionary synthesis (FEEDBACK + APPROXIMABILITY + EXPLORATION)
        # Reduced budget: deterministic search now covers L≤3 exhaustively,
        # so evolution only needs to find L≥4 solutions. Single restart with
        # reduced generations (15 instead of 30) saves ~80% of evolution budget.
        n_restarts = 1
        best_program = None
        best_score = 0.0
        all_history = []
        evo_generations = max(15, self.max_generations // 2)

        if candidates:
            # Already have at least one pixel-perfect candidate from deterministic
            # search. Skip evolution to save ~15-30 seconds per task.
            n_restarts = 0
            if self.verbose:
                print(f"  ⚡ Skipping evolution — {len(candidates)} candidate(s) already found")

        for restart_idx in range(n_restarts):
            if restart_idx > 0:
                import random
                random.seed(random.randint(0, 2**31) + restart_idx * 7919)

            program, history = self.synthesizer.synthesize(
                task=task,
                max_generations=evo_generations,
                target_score=2.0,  # unreachable → always run full budget
                seed_programs=seed_programs,
                verbose=self.verbose and restart_idx == 0,
                cache=cache,
            )
            all_history.extend(history)

            if program and program.fitness > best_score:
                best_program = program
                best_score = program.fitness

            # If we found a pixel-perfect solution, add it as candidate
            if program and program.fitness >= 0.99 and cache.is_pixel_perfect(program):
                candidates.append((program, f"evolved_r{restart_idx}"))

        elapsed = time.time() - start_time

        # Step 4a: Try decomposition as fallback (COMPOSABILITY)
        # Skip if we already have candidates from deterministic search.
        if best_score < 0.99 and not candidates:
            decomposed = self.decomposer.decompose_if_needed(
                task,
                best_score,
                lambda t: self.synthesizer.synthesize(
                    task=t,
                    max_generations=self.max_generations // 2,
                    target_score=0.99,
                    verbose=False,
                )
            )
            if decomposed is not None and decomposed.fitness > best_score:
                best_program = decomposed
                best_score = decomposed.fitness
                if self.verbose:
                    print(f"  ◆ Decomposition improved score to {best_score:.3f}")

        # Step 4b: Add evolved/decomposed result to candidates if pixel-perfect.
        if best_score >= 0.99 and best_program and cache.is_pixel_perfect(best_program):
            candidates.append((best_program, "evolved"))

        # Step 5: Post-evolution near-miss refinement.
        # Try single-step fixes (append/prepend/replace) on the best program
        # AND all near-miss programs collected from search phases.
        if not candidates:
            # Build refinement pool: best program + top near-miss candidates
            # Sort by fitness and keep top 3 to limit refinement cost
            near_miss_pool.sort(key=lambda x: x[0].fitness, reverse=True)
            refine_pool: list[tuple[Program, str]] = near_miss_pool[:3]
            if best_program and best_program.fitness >= 0.80:
                refine_pool.append((best_program, "evolved_best"))

            if refine_pool:
                refined_post = self.synthesizer.try_near_miss_refinement(
                    refine_pool, cache, score_threshold=0.80
                )
                if refined_post and refined_post.fitness >= 0.99:
                    if cache.is_pixel_perfect(refined_post):
                        candidates.append((refined_post, "near_miss_refine"))

        # Step 5a: Check if best single primitive beats evolved
        if best_single and best_single.fitness >= best_score:
            if cache.is_pixel_perfect(best_single) and (
                not best_program or len(best_single) <= len(best_program)
            ):
                candidates.append((best_single, "single_primitive"))

        # Step 6: Deduplicate candidates by step sequence.
        seen_steps: set[tuple[str, ...]] = set()
        unique_candidates: list[tuple[Program, str]] = []
        for prog, meth in candidates:
            key = tuple(s.name for s in prog.steps)
            if key not in seen_steps:
                seen_steps.add(key)
                unique_candidates.append((prog, meth))
        candidates = unique_candidates

        # Step 7: Pick the winner from ALL candidates.
        # Strategy: prefer candidates that PASS TEST (if test available),
        # then use MDL (shortest program) to break ties.
        # Pure MDL sometimes picks a program that overfits training.
        solved = False
        method = "evolved"
        if candidates:
            # Validate all candidates on test (if available)
            if task.get("test"):
                test_passers = []
                for prog, meth in candidates:
                    t_exact, _ = validate_on_test(prog, task)
                    if t_exact:
                        test_passers.append((prog, meth))
                if test_passers:
                    # Among test-passers, pick shortest (MDL)
                    winner, method = min(test_passers, key=lambda x: len(x[0].steps))
                else:
                    # No test-passers: fall back to MDL on all candidates
                    winner, method = min(candidates, key=lambda x: len(x[0].steps))
            else:
                # No test data: pure MDL
                winner, method = min(candidates, key=lambda x: len(x[0].steps))
            best_program = winner
            best_score = winner.fitness
            solved = True
            self._record_success(task_id, best_program, best_score, elapsed)
        elif best_score >= 0.95:
            # Near-miss: promote as concept but don't count as "solved"
            self.tasks_partially_solved += 1
            self.archive.record_solution(task_id, best_program, best_score)
            new_concept = self.explorer.discover_new_concept(best_program, task_id)
            if new_concept:
                self.toolkit.add_concept(new_concept)
                if self.verbose:
                    print(f"  ◆ Near-miss concept promoted: {new_concept.name} "
                          f"(score={best_score:.3f})")
        elif best_score > 0.8:
            self.tasks_partially_solved += 1
            self.archive.record_solution(task_id, best_program, best_score)

        # Also check if best_single was the best (even if not solved)
        if not solved and best_single and best_single.fitness >= (best_score or 0):
            best_program = best_single
            best_score = best_single.fitness
            method = "single_primitive"

        # Clean up task-specific learned concepts (they're task-bound, not reusable)
        for lc in learned_concepts:
            self.toolkit.concepts.pop(lc.name, None)

        # Record concept library growth
        self.concept_growth.append(self.toolkit.size)
        self.solve_times.append(elapsed)

        # Decay exploration over time
        self.explorer.decay_epsilon()

        # Validate best program on test examples (if available)
        test_exact = False
        test_score = 0.0
        if best_program and task.get("test"):
            test_exact, test_score = validate_on_test(best_program, task)

        # Validate ALL candidates on test (not just the winner)
        # This lets us see which alternatives also pass/fail test
        candidate_test_results = []
        if task.get("test"):
            for prog, meth in candidates:
                c_exact, c_score = validate_on_test(prog, task)
                candidate_test_results.append((c_exact, c_score))

        # For fluke diagnostics: per-example train accuracy
        train_example_exact = []
        n_train = len(task.get("train", []))
        if best_program and not solved:
            train_example_exact = cache.per_example_exact(best_program)

        return self._make_result(task_id, best_program, best_score, elapsed, method,
                                  pixel_perfect=solved,
                                  test_exact=test_exact,
                                  test_score=test_score,
                                  n_candidates=len(candidates),
                                  candidates=candidates,
                                  candidate_test_results=candidate_test_results,
                                  n_evals=cache.n_evals,
                                  n_train=n_train,
                                  train_example_exact=train_example_exact)

    def _try_culture_programs(self, task: dict,
                               cache: "TaskCache | None" = None) -> Optional[Program]:
        """Try all programs stored in the toolkit (loaded from a culture file).

        This is the deterministic culture-transfer step — every program that
        solved a training task gets a direct shot at each eval task before we
        fall through to pair exhaustion and evolution.

        Returns the best-scoring program found (or None if toolkit is empty).
        """
        if cache is None:
            cache = TaskCache(task)

        best_program = None
        best_score   = 0.0

        for prog in self.toolkit.programs:
            score = cache.score_program(prog)
            if score > best_score:
                best_score   = score
                best_program = prog
                best_program.fitness = score
                if score >= 0.99:
                    return best_program

        return best_program

    def _try_single_primitives(self, task: dict,
                                cache: "TaskCache | None" = None) -> Optional[Program]:
        """Quick pass: try every single primitive individually.

        This is a fast "exploit" check before committing to evolution.
        Skips predicates (which don't transform grids).

        Uses the shared TaskCache so expected outputs are not re-converted.
        """
        if cache is None:
            cache = TaskCache(task)

        best_program = None
        best_score   = 0.0

        for concept in self.toolkit.concepts.values():
            # Skip predicates — they test conditions, not transform grids
            if concept.kind == "predicate":
                continue

            program = Program([concept])
            score = cache.score_program(program)
            if score > best_score:
                best_score   = score
                best_program = program
                best_program.fitness = score

        return best_program

    def _record_success(self, task_id: str, program: Program, score: float,
                         elapsed: float):
        """Record a successful solve and compound knowledge."""
        self.tasks_solved += 1
        self.archive.record_solution(task_id, program, score)

        # Reinforce successful concepts
        for step in program.steps:
            step.reinforce(True)

        # Promote to a reusable concept (COMPOSABILITY / NO RESET)
        new_concept = self.explorer.discover_new_concept(program, task_id)
        if new_concept:
            self.toolkit.add_concept(new_concept)
            if self.verbose:
                print(f"  ★ New concept added: {new_concept.name}")

        # Add successful program to toolkit for future seeding
        self.toolkit.add_program(program)

        if self.verbose:
            print(f"  ✓ SOLVED in {elapsed:.2f}s (score={score:.3f})")

    # ----------------------------------------------------------------
    # Parameterized primitives: learn structure from examples
    # ----------------------------------------------------------------

    def _try_parameterized(self, task: dict,
                          cache: "TaskCache | None" = None) -> Optional[Program]:
        """Try parameterized primitives that learn from training examples.

        These are template functions that learn parameters structurally
        (by frequency rank, spatial role) rather than absolute values,
        enabling generalization to unseen colors.

        Returns the best Program wrapping a parameterized primitive, or None.
        """
        from .param_search import try_parameterized

        if cache is None:
            cache = TaskCache(task)

        result = try_parameterized(task, cache=cache)

        if result:
            score = cache.score_program(result)
            result.fitness = score
            if self.verbose:
                print(f"  ◆ Parameterized primitive found (score={score:.3f})")

        return result

    # ----------------------------------------------------------------
    # Object-centric reasoning: perceive → compare → infer → apply
    # ----------------------------------------------------------------

    def _try_object_rules(self, task: dict,
                           cache: "TaskCache") -> Optional[Program]:
        """Try to solve the task using object-level rule inference.

        Builds scene graphs from each training example, computes structured
        diffs (what changed about each object), finds consistent rules
        across all examples, and validates them.

        Returns a Program wrapping the inferred rules, or None.
        """
        from .scene import solve_with_object_rules, ObjectRule
        from .concepts import Concept

        transform = solve_with_object_rules(task)
        if transform is None:
            return None

        # Wrap the transform as a Concept → Program
        concept = Concept(
            kind="operator",
            name="object_rule_inferred",
            implementation=transform,
        )
        program = Program([concept])

        # Score it to set fitness
        score = cache.score_program(program)
        program.fitness = score

        if self.verbose:
            print(f"  ◆ Object rules inferred (score={score:.3f})")

        return program

    # ----------------------------------------------------------------
    # Object decomposition: per-object transform search
    # ----------------------------------------------------------------

    def _try_object_decomposition(self, task: dict,
                                   cache: "TaskCache") -> Optional[Program]:
        """Try to solve the task by applying a single transform per object.

        For each primitive in the toolkit, applies it to each object's
        subgrid independently and reassembles. If any primitive produces
        pixel-perfect results on all training examples, returns it.
        """
        from .object_decompose import solve_by_object_decomposition

        result = solve_by_object_decomposition(task, self.toolkit, cache)

        if result and self.verbose:
            print(f"  ◆ Object decomposition (score={result.fitness:.3f})")

        return result

    def _try_dsl_synthesis(self, task: dict,
                           cache: "TaskCache",
                           time_budget: float = 5.0) -> Optional[Program]:
        """Synthesize a novel Grid→Grid transform from DSL sub-primitives.

        Uses bottom-up enumeration over a typed DSL to construct transforms
        that may not exist in the primitive library. This enables solving
        tasks that require novel compositions of basic operations.
        """
        from .dsl_synth import synthesize_dsl_program

        result = synthesize_dsl_program(task, cache, time_budget=time_budget)

        if result and self.verbose:
            print(f"  ◇ DSL synthesis (score={result.fitness:.3f})")

        return result

    # ----------------------------------------------------------------
    # Example-parameterized concepts: learn transforms FROM the task
    # ----------------------------------------------------------------

    def _learn_task_concepts(self, task: dict) -> list[Concept]:
        """Learn example-parameterized concepts from training examples.

        Instead of using only hard-coded primitives, we extract simple
        transforms directly from the input→output examples. This follows
        the MDL / Kolmogorov principle: the simplest explanation wins.

        Currently learns:
          - Color mapping: if color A→B consistently, apply that mapping.
          - Neighbor rules: local neighborhood features → output color.

        Returns a list of Concept objects that can be temporarily injected
        into the toolkit for this task's search.
        """
        concepts: list[Concept] = []
        train = task.get("train", [])
        if not train:
            return concepts

        # Only learn same-dims concepts (input and output grids must match)
        same_dims = all(
            len(ex["input"]) == len(ex["output"])
            and len(ex["input"][0]) == len(ex["output"][0])
            for ex in train
        )
        if not same_dims:
            return concepts

        # 1. COLOR MAPPING: learn consistent pixel-by-pixel color changes
        color_map = self._learn_color_mapping(train)
        if color_map:
            mapping = dict(color_map)  # capture for closure

            def apply_color_map(grid: Grid, _m=mapping) -> Grid:
                return [[_m.get(cell, cell) for cell in row] for row in grid]

            concepts.append(Concept(
                kind="operator",
                name="learned_color_map",
                implementation=apply_color_map,
            ))

        # 2. NEIGHBOR RULES: learn local neighborhood → output color mappings
        # Try multiple feature extractors; each successful one becomes a concept
        neighbor_concepts = self._learn_neighbor_rules(train)
        concepts.extend(neighbor_concepts)

        return concepts

    # ----------------------------------------------------------------
    # Neighbor-rule learning (MDL: local rules from examples)
    # ----------------------------------------------------------------

    @staticmethod
    def _get_bg(grid: Grid) -> int:
        """Get background color (most common) for a grid."""
        from collections import Counter
        flat = [grid[r][c] for r in range(len(grid)) for c in range(len(grid[0]))]
        return Counter(flat).most_common(1)[0][0]

    @staticmethod
    def _neighbor_info(grid: Grid, r: int, c: int, bg: int):
        """Get 4-connected and 8-connected neighbor info."""
        from collections import Counter
        rows, cols = len(grid), len(grid[0])

        # 4-connected
        n4 = []
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                n4.append(grid[nr][nc])

        # 8-connected (all 8 neighbors)
        n8 = []
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    n8.append(grid[nr][nc])

        non_bg_4 = [n for n in n4 if n != bg]
        non_bg_8 = [n for n in n8 if n != bg]
        n_non_bg_4 = len(non_bg_4)
        n_non_bg_8 = len(non_bg_8)
        dom_4 = Counter(non_bg_4).most_common(1)[0][0] if non_bg_4 else bg
        dom_8 = Counter(non_bg_8).most_common(1)[0][0] if non_bg_8 else bg

        return n_non_bg_4, n_non_bg_8, dom_4, dom_8

    @staticmethod
    def _extract_features_basic(grid: Grid, r: int, c: int) -> tuple:
        """Extract basic neighborhood features for a cell.

        Features: (is_bg, n_non_bg_4, n_non_bg_8, dominant_8)
        """
        bg = FourPillarsSolver._get_bg(grid)
        cell = grid[r][c]
        is_bg = cell == bg
        n4, n8, _dom4, dom8 = FourPillarsSolver._neighbor_info(grid, r, c, bg)
        return (is_bg, n4, n8, dom8)

    @staticmethod
    def _extract_features_with_center(grid: Grid, r: int, c: int) -> tuple:
        """Features: (center_color, n_non_bg_4, n_non_bg_8, dominant_8)"""
        bg = FourPillarsSolver._get_bg(grid)
        cell = grid[r][c]
        n4, n8, _dom4, dom8 = FourPillarsSolver._neighbor_info(grid, r, c, bg)
        return (cell, n4, n8, dom8)

    @staticmethod
    def _extract_features_with_row(grid: Grid, r: int, c: int) -> tuple:
        """Features: (is_bg, n_non_bg_4, n_non_bg_8, dom_8, row_dominant_color)"""
        from collections import Counter
        bg = FourPillarsSolver._get_bg(grid)
        cell = grid[r][c]
        is_bg = cell == bg
        n4, n8, _dom4, dom8 = FourPillarsSolver._neighbor_info(grid, r, c, bg)

        # Row dominant non-bg color
        row_non_bg = [v for v in grid[r] if v != bg]
        row_dom = Counter(row_non_bg).most_common(1)[0][0] if row_non_bg else bg

        return (is_bg, n4, n8, dom8, row_dom)

    @staticmethod
    def _extract_features_with_col(grid: Grid, r: int, c: int) -> tuple:
        """Features: (is_bg, n_non_bg_4, n_non_bg_8, dom_8, col_dominant_color)"""
        from collections import Counter
        rows = len(grid)
        bg = FourPillarsSolver._get_bg(grid)
        cell = grid[r][c]
        is_bg = cell == bg
        n4, n8, _dom4, dom8 = FourPillarsSolver._neighbor_info(grid, r, c, bg)

        # Column dominant non-bg color
        col_non_bg = [grid[rr][c] for rr in range(rows) if grid[rr][c] != bg]
        col_dom = Counter(col_non_bg).most_common(1)[0][0] if col_non_bg else bg

        return (is_bg, n4, n8, dom8, col_dom)

    @staticmethod
    def _extract_features_with_position(grid: Grid, r: int, c: int) -> tuple:
        """Features: (center_color, is_border, n_non_bg_4, n_non_bg_8, dom_8)

        Adds positional awareness (is_border) and center color.
        Helps with tasks that change border vs interior cells differently.
        """
        rows = len(grid)
        cols = len(grid[0]) if grid else 0
        bg = FourPillarsSolver._get_bg(grid)
        cell = grid[r][c]
        is_border = r == 0 or c == 0 or r == rows - 1 or c == cols - 1
        n4, n8, _dom4, dom8 = FourPillarsSolver._neighbor_info(grid, r, c, bg)
        return (cell, is_border, n4, n8, dom8)

    def _learn_neighbor_rules(self, train: list[dict]) -> list[Concept]:
        """Learn neighbor-rule concepts from training examples.

        For each feature extractor, collect (features → output_color) across
        all changed cells in all training examples. If a consistent mapping
        exists (every feature tuple maps to exactly one output color) and it
        correctly predicts all changes while NOT changing cells that should
        stay the same, it becomes a Concept.

        Returns a list of valid neighbor-rule Concepts.
        """
        from collections import Counter

        feature_extractors = [
            ("basic", self._extract_features_basic),
            ("with_center", self._extract_features_with_center),
            ("with_row", self._extract_features_with_row),
            ("with_col", self._extract_features_with_col),
            ("with_position", self._extract_features_with_position),
        ]

        concepts = []

        for feat_name, extractor in feature_extractors:
            rules: dict[tuple, dict[int, int]] = {}  # feat → {out_color: count}
            valid = True

            # Phase 1: Collect rules from all changed cells
            for ex in train:
                inp, out = ex["input"], ex["output"]
                rows, cols = len(inp), len(inp[0])
                for r in range(rows):
                    for c_idx in range(cols):
                        if inp[r][c_idx] != out[r][c_idx]:
                            feat = extractor(inp, r, c_idx)
                            if feat not in rules:
                                rules[feat] = {}
                            out_color = out[r][c_idx]
                            rules[feat][out_color] = rules[feat].get(out_color, 0) + 1

            if not rules:
                continue

            # Phase 2: Check consistency — each feature tuple must map
            # to exactly one output color
            rule_map: dict[tuple, int] = {}
            for feat, out_counts in rules.items():
                if len(out_counts) != 1:
                    valid = False
                    break
                rule_map[feat] = next(iter(out_counts))

            if not valid or not rule_map:
                continue

            # Phase 3: Validate — apply rules to training inputs and check
            # that (a) all changed cells are correctly predicted and
            # (b) no unchanged cells are incorrectly modified.
            all_correct = True
            for ex in train:
                inp, out = ex["input"], ex["output"]
                rows, cols = len(inp), len(inp[0])

                # Compute background for this grid
                flat = [inp[r][c_idx] for r in range(rows) for c_idx in range(cols)]
                bg = Counter(flat).most_common(1)[0][0]

                for r in range(rows):
                    for c_idx in range(cols):
                        feat = extractor(inp, r, c_idx)
                        if feat in rule_map:
                            predicted = rule_map[feat]
                            if predicted != out[r][c_idx]:
                                all_correct = False
                                break
                        else:
                            # No rule for this feature → cell stays unchanged
                            if inp[r][c_idx] != out[r][c_idx]:
                                all_correct = False
                                break
                    if not all_correct:
                        break
                if not all_correct:
                    break

            if not all_correct:
                continue

            # Phase 3.5: Generalization check — Leave-One-Out Cross-Validation.
            # For each training example, learn the rule from the other N-1
            # examples and test on the held-out one. If any held-out example
            # fails, the rule is likely to overfit on unseen test data.
            # Only run LOOCV when we have enough examples (≥3).
            if len(train) >= 3:
                loocv_pass = True
                for hold_idx in range(len(train)):
                    # Learn rule from all examples except hold_idx
                    loo_rules: dict[tuple, dict[int, int]] = {}
                    for ei, ex in enumerate(train):
                        if ei == hold_idx:
                            continue
                        inp, out = ex["input"], ex["output"]
                        rows, cols = len(inp), len(inp[0])
                        for r in range(rows):
                            for c_idx in range(cols):
                                if inp[r][c_idx] != out[r][c_idx]:
                                    feat = extractor(inp, r, c_idx)
                                    if feat not in loo_rules:
                                        loo_rules[feat] = {}
                                    oc = out[r][c_idx]
                                    loo_rules[feat][oc] = \
                                        loo_rules[feat].get(oc, 0) + 1

                    # Build LOO rule map (skip inconsistent)
                    loo_map: dict[tuple, int] = {}
                    loo_valid = True
                    for feat, out_counts in loo_rules.items():
                        if len(out_counts) != 1:
                            loo_valid = False
                            break
                        loo_map[feat] = next(iter(out_counts))

                    if not loo_valid:
                        loocv_pass = False
                        break

                    # Test on held-out example
                    held = train[hold_idx]
                    h_inp, h_out = held["input"], held["output"]
                    h_rows, h_cols = len(h_inp), len(h_inp[0])
                    for r in range(h_rows):
                        for c_idx in range(h_cols):
                            feat = extractor(h_inp, r, c_idx)
                            if feat in loo_map:
                                if loo_map[feat] != h_out[r][c_idx]:
                                    loocv_pass = False
                                    break
                            else:
                                if h_inp[r][c_idx] != h_out[r][c_idx]:
                                    loocv_pass = False
                                    break
                        if not loocv_pass:
                            break
                    if not loocv_pass:
                        break

                if not loocv_pass:
                    continue

            # Phase 3.6: Complexity check — reject rules with too many
            # unique features relative to training size. A rule that maps
            # 50 unique feature tuples from only 3 examples is likely
            # memorizing rather than learning a generalizable pattern.
            # Allow at most 8 rules per training example.
            n_rules = len(rule_map)
            max_rules = max(8, len(train) * 8)
            if n_rules > max_rules:
                continue

            # Phase 3.7: Also require LOOCV for n_train=2 (stricter).
            # For 2-example tasks, train on example 0 and test on 1,
            # then vice versa. Both must pass.
            if len(train) == 2:
                loocv_pass_2 = True
                for hold_idx in range(2):
                    other_idx = 1 - hold_idx
                    loo_rules_2: dict[tuple, dict[int, int]] = {}
                    oi, oo = train[other_idx]["input"], train[other_idx]["output"]
                    for r in range(len(oi)):
                        for c_idx in range(len(oi[0])):
                            if oi[r][c_idx] != oo[r][c_idx]:
                                feat = extractor(oi, r, c_idx)
                                if feat not in loo_rules_2:
                                    loo_rules_2[feat] = {}
                                oc = oo[r][c_idx]
                                loo_rules_2[feat][oc] = \
                                    loo_rules_2[feat].get(oc, 0) + 1
                    # Build LOO map
                    loo_map_2: dict[tuple, int] = {}
                    loo_valid_2 = True
                    for feat, out_counts in loo_rules_2.items():
                        if len(out_counts) != 1:
                            loo_valid_2 = False
                            break
                        loo_map_2[feat] = next(iter(out_counts))
                    if not loo_valid_2:
                        loocv_pass_2 = False
                        break
                    # Test on held-out
                    hi, ho = train[hold_idx]["input"], train[hold_idx]["output"]
                    for r in range(len(hi)):
                        for c_idx in range(len(hi[0])):
                            feat = extractor(hi, r, c_idx)
                            if feat in loo_map_2:
                                if loo_map_2[feat] != ho[r][c_idx]:
                                    loocv_pass_2 = False
                                    break
                            else:
                                if hi[r][c_idx] != ho[r][c_idx]:
                                    loocv_pass_2 = False
                                    break
                        if not loocv_pass_2:
                            break
                    if not loocv_pass_2:
                        break
                if not loocv_pass_2:
                    continue

            # Phase 4: Build the concept closure
            frozen_rules = dict(rule_map)  # capture
            frozen_name = feat_name  # capture

            def make_applier(rules_dict, ext_func):
                """Create a closure with its own copies of rules and extractor."""
                def apply_neighbor_rule(grid: Grid) -> Grid:
                    result = [row[:] for row in grid]
                    rows_g, cols_g = len(grid), len(grid[0])
                    for r in range(rows_g):
                        for c_idx in range(cols_g):
                            feat_val = ext_func(grid, r, c_idx)
                            if feat_val in rules_dict:
                                result[r][c_idx] = rules_dict[feat_val]
                    return result
                return apply_neighbor_rule

            applier = make_applier(frozen_rules, extractor)

            concepts.append(Concept(
                kind="operator",
                name=f"learned_neighbor_{frozen_name}",
                implementation=applier,
            ))

        return concepts

    @staticmethod
    def _learn_color_mapping(train: list[dict]) -> Optional[dict[int, int]]:
        """Learn a consistent color mapping from training examples.

        Returns {in_color: out_color} if every changed pixel follows the
        same mapping across ALL examples. Returns None otherwise.
        """
        mappings: dict[int, dict[int, int]] = defaultdict(lambda: defaultdict(int))
        for ex in train:
            inp, out = ex["input"], ex["output"]
            for r in range(len(inp)):
                for c in range(len(inp[0])):
                    if inp[r][c] != out[r][c]:
                        mappings[inp[r][c]][out[r][c]] += 1

        if not mappings:
            return None

        # Each input color must map to exactly one output color
        result: dict[int, int] = {}
        for in_color, out_counts in mappings.items():
            best_out = max(out_counts, key=out_counts.get)
            # Check consistency: best_out must dominate
            total = sum(out_counts.values())
            if out_counts[best_out] < total * 0.9:
                return None  # ambiguous mapping
            result[in_color] = best_out

        return result if result else None

    def _make_result(self, task_id, program, score, elapsed, method,
                      pixel_perfect: bool = False,
                      test_exact: bool = False,
                      test_score: float = 0.0,
                      n_candidates: int = 0,
                      candidates: list | None = None,
                      candidate_test_results: list | None = None,
                      n_evals: int = 0,
                      n_train: int = 0,
                      train_example_exact: list | None = None):
        """Build the per-task result dict.

        Args:
            pixel_perfect: True if program is pixel-perfect on all TRAINING examples.
            test_exact: True if program is pixel-perfect on all TEST examples.
            test_score: Average pixel accuracy on test examples.
            candidates: List of (Program, method_str) tuples.
            candidate_test_results: Per-candidate (test_exact, test_score) tuples.
            n_train: Number of training examples for this task.
            train_example_exact: Per-example pixel-perfect booleans (for flukes).
        """
        # Serialize candidates to dicts (Programs aren't JSON-serializable)
        cand_dicts = []
        test_results = candidate_test_results or []
        if candidates:
            for i, (prog, meth) in enumerate(candidates):
                c_test_exact, c_test_score = test_results[i] if i < len(test_results) else (False, 0.0)
                cand_dicts.append({
                    "program": prog.name,
                    "method": meth,
                    "steps": [s.name for s in prog.steps],
                    "test_exact": c_test_exact,
                    "test_score": round(c_test_score, 4),
                })

        # Classify solve status
        # test_confirmed = pixel-perfect on BOTH train AND test (the golden metric)
        # fluke = passed test but NOT train-perfect (lucky generalization)
        # overfit = train-perfect but NOT test-exact
        test_confirmed = pixel_perfect and test_exact
        fluke = (not pixel_perfect) and test_exact

        # Program steps as a list of primitive names (the "program tree")
        program_steps = [s.name for s in program.steps] if program else []

        return {
            "task_id": task_id,
            "solved": pixel_perfect,
            "test_exact": test_exact,
            "test_confirmed": test_confirmed,
            "fluke": fluke,
            "score": score,
            "test_score": test_score,
            "program": program.name if program else "none",
            "program_steps": program_steps,
            "program_length": len(program) if program else 0,
            "time_seconds": elapsed,
            "method": method,
            "toolkit_size": self.toolkit.size,
            "n_candidates": n_candidates,
            "candidates": cand_dicts,
            "n_evals": n_evals,
            "n_train": n_train,
            "train_example_exact": train_example_exact or [],
        }

    def solve_batch(self, tasks: dict[str, dict]) -> dict:
        """Solve a batch of tasks, accumulating knowledge across them.

        This is where the 'cumulative culture' principle is tested:
        later tasks should benefit from concepts learned on earlier ones.
        """
        results = {}

        for i, (task_id, task) in enumerate(tasks.items()):
            if self.verbose:
                print(f"\n[{i+1}/{len(tasks)}] ", end="")
            result = self.solve_task(task, task_id)
            results[task_id] = result

        # Print summary
        if self.verbose:
            self._print_summary(results)

        return results

    def _print_summary(self, results: dict):
        """Print evaluation summary."""
        print(f"\n{'='*60}")
        print("EVALUATION SUMMARY")
        print(f"{'='*60}")
        print(f"Tasks attempted: {self.tasks_attempted}")
        print(f"Tasks solved (exact): {self.tasks_solved} "
              f"({100*self.tasks_solved/max(1,self.tasks_attempted):.1f}%)")
        print(f"Tasks partially solved (>80%): {self.tasks_partially_solved}")

        scores = [r["score"] for r in results.values()]
        print(f"Average score: {sum(scores)/len(scores):.3f}")
        print(f"Median score: {sorted(scores)[len(scores)//2]:.3f}")

        times = [r["time_seconds"] for r in results.values()]
        print(f"Average solve time: {sum(times)/len(times):.2f}s")

        print(f"\nToolkit growth: {self.concept_growth[0] if self.concept_growth else 0} → {self.toolkit.size}")
        print(f"Learned concepts: {len(self.toolkit.programs)}")

        # Show if later tasks solve faster (cumulative culture test)
        if len(self.solve_times) >= 4:
            first_half = self.solve_times[:len(self.solve_times)//2]
            second_half = self.solve_times[len(self.solve_times)//2:]
            print(f"\nCumulative Culture Test:")
            print(f"  First half avg time: {sum(first_half)/len(first_half):.2f}s")
            print(f"  Second half avg time: {sum(second_half)/len(second_half):.2f}s")

        print(f"{'='*60}")
