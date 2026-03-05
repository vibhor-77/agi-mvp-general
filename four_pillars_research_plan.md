# Four Pillars of General Learning: Research Plan

**Author:** Vibhor Jain
**Date:** March 2026
**Status:** Active Research — Prototype v0.1 Complete

---

## 1. Thesis

General intelligence is not a single algorithm. It is an emergent property that arises when four necessary and sufficient conditions operate together in a system that interacts with an environment:

1. **Feedback Loops** — The environment provides continuous, automatic signals about the correctness of actions. Tighter, faster loops produce faster learning.
2. **Approximability** — The search landscape is structured such that iterative refinement converges toward truth. Learning is not binary; it is the mathematical confidence that error is decreasing.
3. **Abstraction & Composability** — Learned patterns become reusable building blocks that compose recursively to form higher-level concepts. Complexity emerges from the composition of simpler parts.
4. **Exploration** — The system balances exploiting known patterns with exploring novel combinations, generating creativity and handling the unknown.

These four pillars are substrate-independent: they apply whether the system runs on neurons, silicon, or any other computational substrate. Intelligence scales fractally — the same principles operate at the level of a neuron, a brain, an individual, and a society.

**Key Critique of Current AI:** Large language models lack continuous learning loops. Each training run "resets" the system. They memorize patterns from training data rather than learning from interaction with an environment. This is the "Reset Button Problem" — knowledge does not compound.

---

## 2. Formal Definitions

### 2.1 Concept Grammar

```
Concept → Constant | Relationship | Operator | Concept Op Concept
```

Where:
- **Constant** = an atomic value (a color, a number, a shape)
- **Relationship** = a predicate over concepts (same_color, adjacent, contains)
- **Operator** = a transformation function (rotate, scale, fill)
- **Composed** = a recursive combination of concepts via operators or relationships

This grammar is the foundation of Pillar 3. It ensures that the system's knowledge is inherently composable — every learned concept can serve as a building block for more complex concepts.

### 2.2 Dual Memory System

- **Toolkit (Type 1 Memory):** Stores timeless, context-independent patterns. These are the reusable concepts that transfer across tasks. Analogous to long-term procedural memory.
- **Archive (Type 2 Memory):** Stores time-bound, task-specific context. Records which programs solved which tasks, enabling cross-task transfer. Analogous to episodic memory.

### 2.3 The Learning Loop

```
For each interaction with the environment:
  1. PERCEIVE:  Extract features from the current state
  2. MATCH:     Search Toolkit for applicable patterns (exploit)
  3. SYNTHESIZE: If no match, compose new programs from existing concepts
  4. TEST:      Apply candidate to environment, receive feedback
  5. SCORE:     Measure proximity to desired outcome (approximability)
  6. REFINE:    Iteratively improve using evolutionary or gradient-free search
  7. INTEGRATE: Promote successful programs to Toolkit (knowledge compounding)
  8. EXPLORE:   With probability ε, try novel combinations instead
```

### 2.4 Scoring Function (Approximability)

For ARC-AGI grid tasks:

```
S(predicted, expected) = 0.6 × pixel_accuracy + 0.15 × dim_match + 0.15 × color_overlap + 0.1 × nonzero_similarity
```

This weighted score creates a smooth fitness landscape where partial solutions provide gradient information toward full solutions. This is the mathematical mechanism of Pillar 2.

---

## 3. Benchmarks

### 3.1 ARC-AGI

**Why ARC-AGI:** Designed specifically to test fluid intelligence and abstract reasoning. Tasks require few-shot generalization (2-5 examples), symbolic rule inference, and compositional reasoning — exactly what the 4 Pillars framework claims to enable.

**Versions:**
- ARC-AGI-1: 400 training + 400 evaluation tasks (SOTA: 97.9% by Confluence Labs)
- ARC-AGI-2: 1000 training + 240 evaluation tasks (harder, human baseline ~75%)
- ARC-AGI-3: Interactive environments launching March 25, 2026

**Mapping to 4 Pillars:**

| Pillar | ARC-AGI Implementation |
|--------|----------------------|
| Feedback Loops | Grid comparison score after each candidate program |
| Approximability | Pixel-level partial credit enables evolutionary convergence |
| Composability | Programs composed from DSL primitives; successful programs become new concepts |
| Exploration | UCB1-based concept selection; evolutionary mutation/crossover |

**Current Prototype Results (v0.1):**
- 10/10 sample tasks solved on training examples
- 9/10 on held-out test examples (90% test accuracy)
- Key demonstration: Task 6 evolved a 3-step composition (crop → mirror → rotate) in 5 generations
- Task 10 reused the concept learned from Task 6 — demonstrating knowledge compounding

### 3.2 Zork (Text Adventure)

**Why Zork:** Tests fundamentally different capabilities than ARC-AGI — world modeling from natural language, long-horizon planning, and state management. If the 4 Pillars framework is truly general, it should work on both visual grids and text worlds.

**Current SOTA:** ~20% completion (Claude Opus 4.5). Notably, "thinking modes" don't help, confirming that reasoning about text is not the same as learning from interaction.

**Mapping to 4 Pillars:**

| Pillar | Zork Implementation |
|--------|-------------------|
| Feedback Loops | Game score changes + text descriptions as environment feedback |
| Approximability | Score trajectory and exploration coverage as convergence metrics |
| Composability | Action primitives (go, take, use) compose into strategies (explore_room, solve_puzzle) |
| Exploration | Epsilon-greedy room/action exploration with knowledge graph state |

**Implementation plan:** Use Jericho (Microsoft Research) for programmatic Zork access. Build a knowledge graph of the game world that compounds as the agent explores.

---

## 4. Success Metrics

### 4.1 Primary Metrics

| Metric | ARC-AGI Target | Zork Target |
|--------|---------------|-------------|
| Solve rate (exact match) | >30% on ARC-AGI-1 eval | >30% completion |
| Partial solve rate (>80%) | >50% | N/A |
| No LLM in the loop | Yes | Yes |

### 4.2 Cumulative Culture Metrics (unique to this framework)

These metrics test the core thesis — that the system compounds knowledge rather than resetting:

- **Concept Library Growth:** Does the toolkit grow over time? (measured: concepts added per task)
- **Transfer Acceleration:** Do later tasks solve faster than earlier ones? (measured: solve time trend)
- **Cross-Task Reuse:** Are learned concepts reused on new tasks? (measured: reuse count)
- **Composition Depth:** Do composed concepts increase in complexity? (measured: average program length of solutions)

### 4.3 Ablation Studies

To validate that each pillar is necessary:

| Ablation | Expected Effect |
|----------|----------------|
| Remove feedback loops (random scoring) | System cannot converge; performance = random |
| Remove approximability (binary scoring only) | Evolutionary search loses gradient; much slower convergence |
| Remove composability (single primitives only) | Cannot solve multi-step tasks |
| Remove exploration (exploit only) | Gets stuck in local optima; cannot discover novel solutions |

---

## 5. Experimental Protocol

### Phase 1: ARC-AGI-1 (Current)
1. **Expand DSL primitives** to cover more ARC transformation types (object extraction, pattern detection, symmetry operations)
2. **Download full ARC-AGI-1 dataset** (400 training + 400 eval tasks)
3. **Run full evaluation** and measure all metrics from Section 4
4. **Run ablation studies** to validate each pillar's contribution
5. **Compare** against random baseline and single-primitive baseline

### Phase 2: ARC-AGI-2 (Next)
1. **Extend DSL** with multi-step contextual primitives needed for harder tasks
2. **Implement task decomposition** — break hard tasks into sub-problems (fractal problem-solving)
3. **Add object-level reasoning** — extract objects from grids, reason about object properties
4. **Run on ARC-AGI-2** semi-public evaluation set

### Phase 3: ARC-AGI-3 (Interactive, launching March 25, 2026)
1. **Extend architecture** to handle interactive environments (state, actions, observations)
2. **Implement persistent world model** that compounds knowledge across episodes
3. **Add planning module** for goal-directed behavior
4. This version directly tests the framework's core claim: learning through continuous interaction

### Phase 4: Zork
1. **Implement text parsing** and world state extraction
2. **Build knowledge graph** that persists across game sessions
3. **Implement action composition** — primitive actions → composed strategies
4. **Run evaluation** and compare with SOTA

---

## 6. Relation to Existing Work

### 6.1 Program Synthesis
The prototype's evolutionary program synthesis is related to:
- **DreamCoder** (Ellis et al., 2021): Learns a library of programs and uses them to solve new tasks. Our approach differs in using evolutionary search rather than neural-guided search, and in the explicit dual memory system.
- **PCFG-based synthesis** (Rule & Tenenbaum): Our concept grammar is a form of PCFG. The key difference is that our grammar grows through learning.

### 6.2 Evolutionary Computation
- **Genetic Programming** (Koza): Direct ancestor of our approach. We extend it with the UCB-based exploration strategy and the concept library that compounds across tasks.
- **MAP-Elites**: Our Archive serves a similar function to the MAP-Elites behavioral archive — maintaining diversity of solutions.

### 6.3 ARC-AGI SOTA Approaches
- **Confluence Labs (97.9%):** Uses LLMs to generate Python code, then tests against examples. Our approach deliberately avoids LLMs to isolate the 4 Pillars contribution.
- **Imbue Evolution (95.1%):** Closest to our approach — evolutionary code mutation. Key difference: they use Gemini as a base model, while we use no pre-trained model.
- **ARChitects:** Test-time-trained diffusion models. Different paradigm — neural rather than symbolic.

### 6.4 Cognitive Science
- **Cumulative Cultural Evolution** (Tomasello, 2009): Our framework operationalizes this for AI — each generation of concepts builds on the last.
- **Constructivism** (Piaget): Children build schemas through interaction, accommodating and assimilating new experiences. Our Toolkit/Archive maps to schemas/experiences.
- **Universal Darwinism** (Campbell, Dawkins): The evolutionary search mechanism applies the Darwinian algorithm (variation + selection + retention) to program space.

---

## 7. Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                     ENVIRONMENT                              │
│  (ARC-AGI grid tasks / Zork text world / any interactive)    │
└──────────────────────────┬──────────────────────────────────┘
                           │ Observation + Feedback Signal
                           ▼
┌──────────────────────────────────────────────────────────────┐
│                    FOUR PILLARS AGENT                         │
│                                                              │
│  ┌────────────────┐  ┌─────────────────┐  ┌──────────────┐  │
│  │   TOOLKIT       │  │  SYNTHESIZER    │  │  EXPLORER    │  │
│  │  (Pillar 3)     │←→│  (Pillar 2)     │←→│  (Pillar 4)  │  │
│  │                 │  │                 │  │              │  │
│  │  Concept Library│  │  Evolutionary   │  │  UCB1 Select │  │
│  │  Composed Progs │  │  Search + Refine│  │  Novel Gen   │  │
│  │  Dual Memory    │  │  Beam Search    │  │  ε-Greedy    │  │
│  └────────┬───────┘  └───────┬─────────┘  └──────────────┘  │
│           │                  │                                │
│           │          ┌───────▼─────────┐                     │
│           │          │    SCORER       │                     │
│           └─────────→│   (Pillar 1)    │                     │
│                      │                 │                     │
│                      │  Pixel Accuracy │                     │
│                      │  Structural Sim │                     │
│                      │  Feature Extract│                     │
│                      └─────────────────┘                     │
│                                                              │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │                    ARCHIVE                               │ │
│  │  Task solutions, features, cross-task transfer           │ │
│  └─────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────┘
```

---

## 8. Prototype File Structure

```
arc_agent/
├── __init__.py          # Package definition
├── concepts.py          # Concept, Program, Toolkit, Archive (Pillar 3)
├── primitives.py        # DSL primitives — 43 grid transformations
├── synthesizer.py       # Evolutionary program synthesis (Pillar 2)
├── solver.py            # Main learning loop (integrates all 4 pillars)
├── scorer.py            # Feedback scoring engine (Pillar 1)
├── explorer.py          # Explore/exploit engine (Pillar 4)
├── sample_tasks.py      # 10 sample ARC-AGI tasks
└── main.py              # Entry point and evaluation runner
```

---

## 9. Next Steps (Immediate)

1. **Download full ARC-AGI-1 dataset** and run evaluation at scale
2. **Add object-level primitives** — extract connected components, detect patterns, find symmetries
3. **Implement task decomposition** — when a task is too hard, break it into sub-problems
4. **Add persistent concept serialization** — save/load the Toolkit so it survives across runs (solving the "Reset Button" problem)
5. **Run ablation studies** to validate each pillar is necessary
6. **Prepare ARC-AGI-3 integration** — interactive environment support
7. **Begin Zork agent** implementation using Jericho framework

---

## 10. Long-Term Vision

If the 4 Pillars framework is validated on ARC-AGI and Zork, the next frontier is:

- **Multi-domain transfer:** Can concepts learned on ARC-AGI transfer to Zork? This would demonstrate substrate-independent composability.
- **Self-improving architecture:** The agent should be able to discover new primitives, not just compose existing ones.
- **Swarm learning:** Multiple specialized agents with different strengths, trading their discoveries — implementing the "Society/Swarm" level of the fractal intelligence hierarchy.
- **Embodied applications:** Robotics (continuous skill acquisition), cybersecurity (component-based threat recognition), space exploration (real-time terrain adaptation).

The ultimate test: a system that, starting from minimal primitives, builds up its own understanding of a domain through interaction alone — no pre-training, no human labels, just the four pillars operating on raw experience.
