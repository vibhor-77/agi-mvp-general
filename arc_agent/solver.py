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
from typing import Optional
from .concepts import Toolkit, Archive, Program
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
        max_program_length: int = 4,
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

    def solve_task(self, task: dict, task_id: str = "unknown") -> dict:
        """Solve a single ARC task using all 4 pillars.

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

        # Step 3: Quick check — does any single primitive solve it?
        best_single = self._try_single_primitives(task, cache)
        if best_single and best_single.fitness >= 0.99:
            elapsed = time.time() - start_time
            self._record_success(task_id, best_single, best_single.fitness, elapsed)
            return self._make_result(task_id, best_single, best_single.fitness,
                                      elapsed, "single_primitive")

        # Step 3.5: Try all pairs of top primitives (fast, high-yield)
        pair_result = self.synthesizer.try_all_pairs(task, cache, top_k=15)
        if pair_result and pair_result.fitness >= 0.99:
            elapsed = time.time() - start_time
            self._record_success(task_id, pair_result, pair_result.fitness, elapsed)
            return self._make_result(task_id, pair_result, pair_result.fitness,
                                      elapsed, "pair_exhaustion")

        # Inject best pair into seed programs for evolution
        if pair_result and pair_result.fitness > 0.5:
            seed_programs = list(seed_programs) if seed_programs else []
            seed_programs.insert(0, pair_result)

        # Step 4: Evolutionary synthesis (FEEDBACK + APPROXIMABILITY + EXPLORATION)
        best_program, history = self.synthesizer.synthesize(
            task=task,
            max_generations=self.max_generations,
            target_score=0.99,
            seed_programs=seed_programs,
            verbose=self.verbose,
            cache=cache,
        )

        elapsed = time.time() - start_time
        best_score = best_program.fitness if best_program else 0.0

        # Step 4.5: Try decomposition as fallback (COMPOSABILITY)
        # If the evolved solution is weak, attempt problem decomposition
        if best_score < 0.99:
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

        # Step 5: Promote solutions to reusable concepts (COMPOSABILITY)
        # Lower threshold (0.95) allows near-miss knowledge to compound.
        # This is critical for cumulative culture — even imperfect solutions
        # contain useful sub-patterns that should be available for future tasks.
        if best_score >= 0.99:
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
            # Record for transfer but don't promote to toolkit
            self.archive.record_solution(task_id, best_program, best_score)

        # Record concept library growth
        self.concept_growth.append(self.toolkit.size)
        self.solve_times.append(elapsed)

        # Decay exploration over time
        self.explorer.decay_epsilon()

        method = "evolved"
        if best_single and best_single.fitness >= best_score:
            best_program = best_single
            best_score = best_single.fitness
            method = "single_primitive"

        return self._make_result(task_id, best_program, best_score, elapsed, method)

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

    def _make_result(self, task_id, program, score, elapsed, method):
        return {
            "task_id": task_id,
            "solved": score >= 0.99,
            "score": score,
            "program": program.name if program else "none",
            "program_length": len(program) if program else 0,
            "time_seconds": elapsed,
            "method": method,
            "toolkit_size": self.toolkit.size,
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
