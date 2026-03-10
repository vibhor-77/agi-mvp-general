# Design Notes & Implementation Thoughts

## Why This Approach?

### No LLM in the Loop (Deliberate Choice)

The current ARC-AGI SOTA (Confluence Labs at 97.9%) uses LLMs to generate Python code. While effective, this conflates two questions:

1. Are the 4 Pillars sufficient for general intelligence?
2. Can a pre-trained LLM solve ARC tasks?

By building a system with **no pre-trained model**, we isolate the 4 Pillars contribution. If the system learns to solve tasks using only feedback loops, approximability, composability, and exploration — starting from a minimal set of hand-coded primitives — that's a much stronger validation of the framework than wrapping an LLM in a feedback loop.

This is directly from Vibhor's framework: the argument is that intelligence emerges from the *principles*, not from the *substrate* or *pre-existing knowledge*.

### Program Synthesis, Not Neural Networks

We chose explicit, inspectable programs over neural networks for several reasons:

- **Composability is first-class**: Programs are literally sequences of composable functions. You can see exactly what the system learned and how it composes knowledge.
- **No catastrophic forgetting**: Adding a new concept to the Toolkit doesn't degrade existing concepts. Knowledge genuinely compounds.
- **Interpretability**: Every solution is a readable chain of transformations. This is crucial for validating the framework — we need to *see* that composability is happening.
- **Matches the concept grammar**: Vibhor's recursive grammar (`Concept → Constant | Op | Concept Op Concept`) maps directly to program composition.

### Evolutionary Search (Not Gradient Descent)

Evolutionary search was chosen because:

- **It works on discrete program spaces** where gradients don't exist.
- **It directly models biological evolution** — Vibhor's framework draws heavily from evolutionary biology.
- **Mutation = Exploration**: Random mutations are the mechanism for discovering novel programs.
- **Selection = Feedback + Approximability**: Fitness scoring drives convergence.
- **Crossover = Composability**: Combining successful sub-programs is literal composition.

## Key Design Decisions

### 1. Dual Memory (Toolkit + Archive)

The Toolkit/Archive split maps to Vibhor's Type 1 / Type 2 memory:

- **Toolkit**: Concepts that work *in general*. A rotation works regardless of which task you're solving. These are timeless, context-free.
- **Archive**: Records of *what worked where*. "Task X had these features and was solved by Program Y." This is episodic, contextual.

Cross-task transfer happens when the Archive notices that a new task has similar features to a previously solved one, and seeds the synthesizer with programs that worked before.

### 2. Partial Credit Scoring (Approximability)

The single most important design decision for making evolutionary search work. Binary scoring (correct/incorrect) creates a flat fitness landscape — the synthesizer can't tell if it's getting closer. Partial credit creates gradients:

```
Score = 0.6 × pixel_accuracy + 0.15 × dim_match + 0.15 × color_overlap + 0.1 × nonzero_similarity
```

This means a program that gets the right dimensions but wrong colors scores higher than one that gets everything wrong. Evolution can then incrementally fix color issues while preserving the dimensional structure.

### 3. UCB1 for Exploration

Why UCB1 instead of simple epsilon-greedy?

UCB1 naturally balances exploration and exploitation *per concept*. A concept that's been tried 1000 times and failed gets a lower exploration bonus than one that's been tried 3 times. This means the system preferentially explores under-tested concepts while exploiting known-good ones.

The formula: `UCB = success_rate + C × sqrt(ln(N) / n_i)`

This is mathematically principled (optimal regret bounds) rather than an arbitrary epsilon parameter.

### 4. Concept Promotion (Knowledge Compounding)

When a multi-step program solves a task, it gets promoted to a first-class Concept in the Toolkit. This is the key mechanism for avoiding the "Reset Button":

1. Task 6 evolves `crop_nonzero → mirror_v → rotate_180`
2. This becomes `learned_crop_then_mirror_43` in the Toolkit
3. Task 10 can now use this as a single building block
4. Future tasks can compose *on top of it*

Over time, the Toolkit grows from simple primitives to increasingly complex composed concepts. This is fractal: the system builds hierarchies of abstraction automatically.

## What's Missing (Known Limitations)

### 1. Object-Level Reasoning — *Partially Addressed in v0.2*

~~The current DSL operates on whole grids.~~ v0.2 added `objects.py` with connected component extraction, object property detection, and basic object transforms (extract, recolor, mirror). Still missing:

- Object-to-object relationships (same color, touching, aligned, contains)
- Object movement (translate object to position)
- Object duplication and pattern repetition
- Sub-pattern detection within objects

### 2. Conditional Logic — *Partially Addressed in v0.25*

~~The current programs are linear chains: Step1 → Step2 → Step3.~~ v0.25 added neighbor-rule learning with LOOCV generalization: the solver learns cell-level conditional transforms based on local neighborhood features (color counts, directional neighbors). Still missing:

- General if-then-else branching in programs
- Predicate evaluation on arbitrary grid regions
- For-each-object iteration with conditionals

### 3. Task Decomposition — *Addressed in v0.20+*

~~Hard ARC tasks often require solving sub-problems.~~ The DecompositionEngine (v0.20) implements six strategies: color-channel, spatial quadrant, diff-focus, pattern detection, size-ratio, and masking decomposition. See `docs/DECOMPOSITION_IMPROVEMENTS.md` for details. In v0.28, decomposition was further improved with deterministic sub-synthesis (replacing slow evolutionary search for sub-problems) and a 30-second time budget. Still potential improvements:

- Hierarchical decomposition (decompose sub-problems further)
- Object-tracking decomposition (track individual objects across transforms)
- Symmetry-based decomposition

### 4. Persistent State — *Addressed in v0.2*

~~Currently, the Toolkit lives in memory and dies when the process ends.~~ v0.2 added `persistence.py` with full JSON serialization of Toolkit and Archive. Composed concepts survive via recipe-based reconstruction. Still needed:

- Version management (concept library versions)
- Selective forgetting (prune low-utility concepts to prevent bloat)
- Migration support when primitives change between code versions

### 5. Scale

The prototype works on 10 hand-crafted tasks. Real validation requires the full ARC-AGI dataset (400+ tasks). The evolutionary search parameters (population size, generations, program length) may need tuning at scale.

## Connections to Vibhor's Framework

| Framework Concept | Implementation |
|------------------|----------------|
| "Interaction is the only source of truth" | Programs are only scored by running them against examples |
| "The Reset Button Problem" | Concept promotion + persistent serialization (v0.2) |
| "Cumulative culture" | Toolkit grows monotonically; persists across runs via save/load |
| "Fractal intelligence" | Concepts compose recursively; programs of any depth |
| "Substrate independence" | Pure Python with no ML dependencies |
| "Concept grammar" | Dataclass hierarchy: Constant \| Operator \| Composed |
| "Dual memory" | Toolkit (Type 1) + Archive (Type 2) |
| "Explore/exploit tradeoff" | UCB1 + epsilon-greedy + evolutionary mutation |
| "Approximability" | Partial-credit scoring with weighted similarity metrics |

## Future Directions

### ARC-AGI-3 (Interactive)

ARC-AGI-3 (launching March 25, 2026) introduces interactive environments — the agent takes actions and observes results over time. This is *exactly* what the 4 Pillars framework is designed for: continuous interaction with an environment, building knowledge through feedback loops. The current architecture extends naturally:

- **Feedback Loop**: Action → observation → score (real-time)
- **Approximability**: Score trajectory over time
- **Composability**: Action sequences → composed strategies
- **Exploration**: Try new actions in new environments

### Zork

Zork requires fundamentally different primitives (text parsing, knowledge graph, action planning) but the *architecture* is identical. The same Toolkit/Archive/Synthesizer/Explorer pattern applies — just with different base concepts. This is the substrate independence claim in practice.

### Multi-Agent Swarm

Vibhor's framework describes intelligence scaling from neurons → brains → societies. The natural extension is multiple specialized agents:

- Agent A excels at spatial reasoning
- Agent B excels at color transformations
- Agent C excels at pattern detection
- They share their Toolkits, and the collective solves tasks none could solve alone

This maps directly to the "Specialist Swarm" slide in Vibhor's presentation — a society of diverse agents outperforms a single generalist.
