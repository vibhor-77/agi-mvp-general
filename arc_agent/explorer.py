"""
Pillar 4: Exploration — The Explore/Exploit Engine

From Vibhor's framework: "True intelligence balances the Exploit/Explore
trade-off. It uses known patterns to survive but constantly attempts to
combine patterns in novel ways."

This module implements:
1. Epsilon-greedy exploration strategy
2. Novel concept generation (composing existing primitives)
3. Curiosity-driven exploration (prefer under-explored concepts)
4. Cross-task transfer (reuse what worked elsewhere)
"""
from __future__ import annotations
import random
import math
from typing import Optional
from .concepts import Concept, Program, Toolkit, Archive, Grid


class ExplorationEngine:
    """Manages the explore/exploit tradeoff.

    Uses UCB1 (Upper Confidence Bound) for concept selection,
    which naturally balances exploitation of known-good concepts
    with exploration of under-tested ones.
    """

    def __init__(
        self,
        toolkit: Toolkit,
        archive: Archive,
        epsilon: float = 0.3,
        curiosity_weight: float = 1.0,
        composition_depth: int = 3,
    ):
        self.toolkit = toolkit
        self.archive = archive
        self.epsilon = epsilon
        self.curiosity_weight = curiosity_weight
        self.composition_depth = composition_depth
        self.total_selections = 0

    def select_concept_ucb(self) -> Concept:
        """Select a concept using UCB1 (Upper Confidence Bound).

        UCB1 = success_rate + curiosity_weight * sqrt(ln(N) / n_i)

        This naturally explores under-used concepts while exploiting
        known-good ones. The curiosity term encourages exploration
        of concepts that haven't been tried much.
        """
        self.total_selections += 1
        concepts = list(self.toolkit.concepts.values())

        if self.total_selections < len(concepts):
            # Initial phase: try each concept at least once
            untried = [c for c in concepts if c.usage_count == 0]
            if untried:
                return random.choice(untried)

        best_score = -1
        best_concept = None

        for concept in concepts:
            if concept.usage_count == 0:
                # Infinite UCB for untried concepts
                return concept

            exploitation = concept.success_rate
            exploration = self.curiosity_weight * math.sqrt(
                math.log(self.total_selections) / concept.usage_count
            )
            ucb_score = exploitation + exploration

            if ucb_score > best_score:
                best_score = ucb_score
                best_concept = concept

        return best_concept or random.choice(concepts)

    def generate_novel_programs(self, n: int = 10) -> list[Program]:
        """Generate novel programs by composing concepts in new ways.

        This is the creative engine — Vibhor's "Engine of Novelty."
        It builds patterns that don't exist in the dataset by
        computationally experimenting with new combinations.
        """
        programs = []

        for _ in range(n):
            strategy = random.choice([
                "compose_best",
                "random_chain",
                "mix_categories",
                "extend_successful",
            ])

            if strategy == "compose_best":
                # Compose two high-performing concepts
                best = self.toolkit.get_best_concepts(10)
                if len(best) >= 2:
                    a, b = random.sample(best, 2)
                    programs.append(Program([a, b]))

            elif strategy == "random_chain":
                # Random chain of 2-3 concepts using UCB selection
                length = random.randint(2, min(3, self.composition_depth))
                steps = [self.select_concept_ucb() for _ in range(length)]
                programs.append(Program(steps))

            elif strategy == "mix_categories":
                # Combine concepts from different categories
                ops = self.toolkit.get_concepts_by_kind("operator")
                if ops:
                    steps = [random.choice(ops) for _ in range(random.randint(1, 3))]
                    programs.append(Program(steps))

            elif strategy == "extend_successful":
                # Take a successful program and extend it
                if self.toolkit.programs:
                    base = random.choice(self.toolkit.programs)
                    new_step = self.select_concept_ucb()
                    # Add step at beginning or end
                    if random.random() < 0.5:
                        steps = [new_step] + base.steps
                    else:
                        steps = base.steps + [new_step]
                    if len(steps) <= self.composition_depth + 1:
                        programs.append(Program(steps))

        return programs

    def generate_seed_programs(
        self,
        task_features: dict,
    ) -> list[Program]:
        """Generate seed programs informed by task features and past experience.

        This implements cross-task transfer — Vibhor's "cumulative culture"
        principle. Programs that worked on similar tasks are good starting
        points for new tasks.
        """
        seeds = []

        # 1. Get programs from similar tasks (EXPLOIT past learning)
        similar_programs = self.archive.get_programs_for_similar_tasks(task_features)
        seeds.extend(similar_programs[:10])

        # 2. Feature-guided heuristic programs
        if task_features.get("same_dims"):
            # Input/output same size → color or local transformation
            for name in ["identity", "invert_colors", "fill_enclosed", "outline",
                         "recolor_to_most_common", "reverse_rows", "reverse_cols",
                         "denoise_3x3", "denoise_5x5", "fill_holes_per_color",
                         "fill_rectangles", "spread_colors", "erode",
                         "complete_symmetry_h", "complete_symmetry_v",
                         "complete_symmetry_4", "swap_most_least",
                         "recolor_least_common", "keep_only_largest_color",
                         "keep_only_smallest_color"]:
                if name in self.toolkit.concepts:
                    seeds.append(Program([self.toolkit.concepts[name]]))
            # Try gravity ops (same dims, rearrange cells)
            for name in ["gravity_down", "gravity_up", "gravity_left", "gravity_right",
                         "sort_rows_by_color_count", "sort_cols_by_color_count"]:
                if name in self.toolkit.concepts:
                    seeds.append(Program([self.toolkit.concepts[name]]))
            # Try erase + fill combos for same-dims tasks
            for color in range(1, 5):
                erase = f"erase_{color}"
                fill = f"fill_bg_{color}"
                if erase in self.toolkit.concepts:
                    seeds.append(Program([self.toolkit.concepts[erase]]))
                if fill in self.toolkit.concepts:
                    seeds.append(Program([self.toolkit.concepts[fill]]))

        if task_features.get("grows"):
            # Output bigger than input → scaling or tiling
            for name in ["scale_2x", "scale_3x", "tile_2x2", "tile_3x3",
                         "upscale_to_max"]:
                if name in self.toolkit.concepts:
                    seeds.append(Program([self.toolkit.concepts[name]]))

        if task_features.get("shrinks"):
            # Output smaller → cropping, partitioning, dedup
            for name in ["crop_nonzero", "get_top_half", "get_bottom_half",
                         "get_left_half", "get_right_half", "get_interior",
                         "deduplicate_rows", "deduplicate_cols"]:
                if name in self.toolkit.concepts:
                    seeds.append(Program([self.toolkit.concepts[name]]))

        h_ratio = task_features.get("h_ratio", 1.0)
        w_ratio = task_features.get("w_ratio", 1.0)
        if abs(h_ratio - 1.0) < 0.01 and abs(w_ratio - 1.0) < 0.01:
            # Same size → try geometric transforms
            for name in ["rotate_90_cw", "rotate_90_ccw", "rotate_180",
                         "mirror_h", "mirror_v", "transpose",
                         "get_border", "flood_fill_bg"]:
                if name in self.toolkit.concepts:
                    seeds.append(Program([self.toolkit.concepts[name]]))

        # Half-size output → likely a partitioning task
        if abs(h_ratio - 0.5) < 0.1 or abs(w_ratio - 0.5) < 0.1:
            for name in ["get_top_half", "get_bottom_half",
                         "get_left_half", "get_right_half"]:
                if name in self.toolkit.concepts:
                    seeds.append(Program([self.toolkit.concepts[name]]))

        if task_features.get("shrinks"):
            for name in ["xor_halves_v", "or_halves_v", "and_halves_v",
                         "xor_halves_h", "or_halves_h", "and_halves_h",
                         "grid_difference", "grid_difference_h"]:
                if name in self.toolkit.concepts:
                    seeds.append(Program([self.toolkit.concepts[name]]))

        # 3. Two-step combo seeds for common patterns
        combos = [
            ("crop_nonzero", "mirror_h"),
            ("crop_nonzero", "mirror_v"),
            ("crop_nonzero", "rotate_90_cw"),
            ("crop_nonzero", "scale_2x"),
            ("fill_enclosed", "outline"),
            ("outline", "fill_enclosed"),
            ("get_interior", "crop_nonzero"),
            ("invert_colors", "crop_nonzero"),
            # New combos with v0.7 primitives
            ("fill_holes_per_color", "crop_nonzero"),
            ("fill_rectangles", "crop_nonzero"),
            ("spread_colors", "crop_nonzero"),
            ("erode", "crop_nonzero"),
            ("keep_only_largest_color", "crop_nonzero"),
            ("keep_only_smallest_color", "crop_nonzero"),
            ("denoise_3x3", "crop_nonzero"),
            ("fill_enclosed", "denoise_3x3"),
            ("spread_colors", "erode"),
            ("erode", "spread_colors"),
            ("complete_symmetry_h", "fill_enclosed"),
            ("complete_symmetry_v", "fill_enclosed"),
        ]
        for a_name, b_name in combos:
            if a_name in self.toolkit.concepts and b_name in self.toolkit.concepts:
                seeds.append(Program([
                    self.toolkit.concepts[a_name],
                    self.toolkit.concepts[b_name],
                ]))

        # 4. Three-step combo seeds for common multi-step patterns
        triples = [
            ("crop_nonzero", "get_top_half", "get_left_half"),
            ("crop_nonzero", "get_bottom_half", "get_right_half"),
            ("crop_nonzero", "mirror_h", "mirror_v"),
            ("crop_nonzero", "rotate_90_cw", "mirror_h"),
            ("keep_only_largest_color", "crop_nonzero", "scale_2x"),
            ("keep_only_smallest_color", "crop_nonzero", "scale_2x"),
            ("keep_only_largest_color", "crop_nonzero", "mirror_h"),
            ("keep_only_smallest_color", "crop_nonzero", "mirror_h"),
            ("fill_enclosed", "crop_nonzero", "mirror_h"),
            ("invert_colors", "crop_nonzero", "mirror_h"),
            ("remove_largest_obj", "crop_nonzero", "scale_2x"),
            ("remove_smallest_obj", "crop_nonzero", "mirror_v"),
        ]
        for names in triples:
            if all(n in self.toolkit.concepts for n in names):
                seeds.append(Program([self.toolkit.concepts[n] for n in names]))

        # 5. Add novel explorations (EXPLORE new territory)
        seeds.extend(self.generate_novel_programs(10))

        return seeds

    def decay_epsilon(self, factor: float = 0.995):
        """Gradually shift from exploration to exploitation.

        Over time, as the toolkit grows and we've explored more,
        we should exploit more and explore less.
        """
        self.epsilon = max(0.05, self.epsilon * factor)

    def should_explore(self) -> bool:
        """Epsilon-greedy decision: explore or exploit?"""
        return random.random() < self.epsilon

    def discover_new_concept(self, program: Program, task_id: str) -> Optional[Concept]:
        """If a program solves a task, promote it to a reusable concept.

        This is the core of knowledge compounding — successful programs
        become first-class concepts in the toolkit, available for future
        composition. This is how the system avoids 'resetting'.
        """
        if len(program.steps) <= 1:
            return None  # Single primitives are already in the toolkit

        # Create a composed concept from the successful program
        name = f"learned_{task_id}_{len(self.toolkit.concepts)}"

        def composed_impl(grid: Grid) -> Optional[Grid]:
            current = grid
            for step in program.steps:
                result = step.apply(current)
                if result is None:
                    return None
                current = result
            return current

        new_concept = Concept(
            kind="composed",
            name=name,
            implementation=composed_impl,
            children=list(program.steps),
        )

        return new_concept
