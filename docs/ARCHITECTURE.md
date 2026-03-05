# Architecture Guide

This document provides a detailed technical walkthrough of the Four Pillars AGI Agent architecture.

---

## Overview

The agent implements Vibhor Jain's four pillars of general intelligence as a program synthesis system that learns to solve ARC-AGI grid transformation tasks. It starts from a minimal set of hand-coded primitives and builds up composed concepts through interaction with the environment.

```
┌─────────────────────────────────────────────────────────────┐
│                     ENVIRONMENT                              │
│  (ARC-AGI grids / Zork / any interactive environment)        │
└──────────────────────────┬──────────────────────────────────┘
                           │ Observation + Feedback
                           ▼
┌──────────────────────────────────────────────────────────────┐
│                    FOUR PILLARS AGENT                         │
│                                                              │
│  ┌────────────────┐  ┌─────────────────┐  ┌──────────────┐  │
│  │   TOOLKIT       │  │  SYNTHESIZER    │  │  EXPLORER    │  │
│  │  (Pillar 3)     │←→│  (Pillar 2)     │←→│  (Pillar 4)  │  │
│  │                 │  │                 │  │              │  │
│  │  Concept Library│  │  Evolutionary   │  │  UCB1 Select │  │
│  │  Dual Memory    │  │  Search + Refine│  │  ε-Greedy    │  │
│  └────────┬───────┘  └───────┬─────────┘  └──────────────┘  │
│           │          ┌───────▼─────────┐                     │
│           └─────────→│    SCORER       │                     │
│                      │   (Pillar 1)    │                     │
│                      └─────────────────┘                     │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │  ARCHIVE — cross-task transfer & episodic memory         │ │
│  └─────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────┘
```

## Module Responsibilities

### `concepts.py` — Pillar 3: Abstraction & Composability

The foundational data model. Contains:

- **`Concept`**: A composable building block wrapping a grid transformation function. Tracks usage/success for UCB1. Kinds: `constant`, `relationship`, `operator`, `composed`.
- **`Program`**: An ordered sequence of Concepts applied in chain. The unit of evolutionary synthesis. Tracks fitness scores.
- **`Toolkit`** (Type 1 Memory): The growing concept library. Stores reusable, context-independent patterns. Supports composition of two concepts into a higher-level concept. This is where cumulative culture lives.
- **`Archive`** (Type 2 Memory): Episodic memory. Records which programs solved which tasks, enabling cross-task transfer via feature-based similarity matching.

### `primitives.py` — Base Grid Concepts (43 primitives)

The seed knowledge — analogous to biological building blocks that evolution starts with. Organized into:

- **Geometric operators** (7): rotate_90_cw/ccw, rotate_180, mirror_h/v, transpose, identity
- **Color operators** (22): color swaps (12), recolor (9), invert
- **Spatial operators** (12): crop, tile, scale, gravity (4 directions), flood_fill, outline, fill_enclosed, extract_colors, count_per_row
- **Predicates** (4): is_symmetric_h/v, is_square, has_single_color

Factory function `build_initial_toolkit(include_objects=True)` assembles all primitives into a fresh Toolkit, optionally including object-level concepts.

### `objects.py` — Object-Level Primitives (30 concepts)

Added in v0.2. Many ARC tasks require reasoning about discrete objects within grids. This module provides:

- **`GridObject`** dataclass: Represents a connected component with properties (color, pixels, bbox, center, size) and `to_grid()` extraction.
- **`find_objects()`**: Flood-fill connected component extraction using 4-connectivity.
- **Grid → Grid transforms**: `extract_largest_object`, `extract_smallest_object`, `mirror_objects_horizontal`, `remove_color`, `isolate_color`, `recolor_largest_object`.
- **Factory-generated per-color concepts** (27): `remove_color_N`, `isolate_color_N`, `recolor_largest_to_N` for colors 1–9.
- **`add_object_concepts()`**: Registers all object concepts into an existing Toolkit.

### `persistence.py` — Toolkit & Archive Serialization

Added in v0.2. Solves the "Reset Button Problem" by enabling the Toolkit and Archive to survive across process runs.

- **`save_toolkit()` / `load_toolkit()`**: JSON serialization of the entire concept library, including composed concepts. Composed concepts are stored as recipes (list of child names) and re-composed on load by chaining their children. Topological resolution handles concepts that depend on other composed concepts.
- **`save_archive()` / `load_archive()`**: JSON serialization of task features, history, and solution metadata.
- Design constraint: primitive concepts are restored by name from `build_initial_toolkit()`, so the serialized format is stable across code changes to primitives.

### `scorer.py` — Pillar 1: Feedback Loops

The environment feedback mechanism:

- **`pixel_accuracy()`**: Fraction of matching pixels. Handles dimension mismatches with partial credit.
- **`structural_similarity()`**: Weighted composite score: 0.6×pixel + 0.15×dimensions + 0.15×colors + 0.1×nonzero. Creates smooth fitness landscape.
- **`score_program_on_task()`**: Averages structural similarity across all training examples.
- **`validate_on_test()`**: Held-out test validation (exact match + score).
- **`extract_task_features()`**: Extracts structural features for cross-task similarity matching.

### `synthesizer.py` — Pillar 2: Approximability

Evolutionary program synthesis:

- **Population management**: Initial population includes all single primitives + random compositions.
- **Mutation**: Replace step, insert step, remove step, swap adjacent steps.
- **Crossover**: First half of parent A + second half of parent B.
- **Selection**: Elite preservation + tournament selection.
- **`synthesize()`**: Full evolutionary loop with early stopping on perfect solutions.

### `explorer.py` — Pillar 4: Exploration

Explore/exploit balance:

- **UCB1 selection**: `success_rate + C × sqrt(ln(N) / n_i)` — mathematically optimal regret bounds.
- **Novel program generation**: 4 strategies (compose_best, random_chain, mix_categories, extend_successful).
- **Seed program generation**: Feature-guided heuristics + cross-task transfer from Archive.
- **Concept discovery**: Promotes successful multi-step programs to first-class Toolkit concepts.
- **Epsilon decay**: Gradually shifts from exploration to exploitation.

### `solver.py` — Integration of All Four Pillars

The main learning loop:

1. **Extract features** → record in Archive
2. **Generate seeds** → exploit past knowledge via Explorer
3. **Quick check** → try each primitive alone (fast exploit)
4. **Evolutionary synthesis** → Synthesizer runs full evolution
5. **Record & promote** → Archive records solution, Explorer promotes to Toolkit

Also implements batch solving with cumulative culture metrics.

### `main.py` — Entry Point

Runs full evaluation, test validation, and prints cumulative culture metrics.

## Data Flow

```
Task (train examples) ──→ Feature Extraction ──→ Archive (record features)
                                                      │
                                                      ▼
                                              Seed Generation (Explorer)
                                                      │
                                                      ▼
                                              Quick Primitive Check
                                              (try each alone)
                                                      │
                                                      ▼ (if not solved)
                                              Evolutionary Synthesis
                                              (population → score → select
                                               → mutate → crossover → repeat)
                                                      │
                                                      ▼
                                              Best Program Found
                                                      │
                                              ┌───────┴───────┐
                                              ▼               ▼
                                        Score ≥ 0.99    Score < 0.99
                                              │               │
                                              ▼               ▼
                                        Promote to      Record partial
                                        Toolkit         (still useful
                                        (cumulative     for transfer)
                                        culture)
```

## Key Invariants

1. **Toolkit grows monotonically**: Concepts are only added, never removed (cumulative culture).
2. **Programs are pure functions**: `Grid → Grid` with no side effects.
3. **Scoring is deterministic**: Same program + same task = same score.
4. **Zero runtime dependencies**: Everything uses Python stdlib only.
5. **Composability is fractal**: A composed concept can be composed with other composed concepts to arbitrary depth.
6. **Persistence is lossless**: `save_toolkit` → `load_toolkit` roundtrip preserves all concept names, kinds, usage stats, and composed concept behavior.
