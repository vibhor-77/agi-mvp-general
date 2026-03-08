# Four Pillars AGI Agent

**A framework for general intelligence built on four fundamental principles, validated on ARC-AGI.**

Based on the research and principles proposed by [Vibhor Jain](https://github.com/vibhor-77).

---

## The Thesis

General intelligence is not a single algorithm. It is an emergent property that arises when four necessary conditions operate together in a system that interacts with an environment:

| Pillar | Principle | Role |
|--------|-----------|------|
| **1. Feedback Loops** | The environment provides continuous, automatic signals about correctness | Tighter loops = faster learning |
| **2. Approximability** | The search landscape allows iterative refinement toward truth | Partial solutions guide convergence |
| **3. Abstraction & Composability** | Learned patterns become reusable blocks that compose recursively | Complexity from simplicity |
| **4. Exploration** | Balance exploiting known patterns with exploring novel combinations | Creativity and novelty |

These pillars are substrate-independent and scale fractally — from neurons to brains to societies.

## Key Innovation: No Reset Button

Current AI systems "reset" with each training run — knowledge doesn't compound. This agent implements **cumulative culture**: successful programs become first-class concepts in the Toolkit, available for future composition. Later tasks benefit from earlier learning. The Toolkit can be saved to disk and loaded across runs, solving the Reset Button Problem completely.

## Quick Start

```bash
# Clone and install (NumPy is the only runtime dependency)
git clone https://github.com/vibhor-77/agi-mvp-general.git
cd agi-mvp-general
pip install numpy           # or: conda install numpy

# Run the test suite to verify everything works
python -m unittest discover -s tests -p "*.py"

# Clone the ARC-AGI dataset
git clone https://github.com/fchollet/ARC-AGI.git

# Full benchmark with culture transfer (RECOMMENDED):
# Step 1: Train and save learned culture
python -m arc_agent.evaluate train --data-dir ARC-AGI/data/training \
    --culture-file culture.json --output results_train.json

# Step 2: Infer — generate candidates (never peeks at test output)
python -m arc_agent.evaluate infer --data-dir ARC-AGI/data/evaluation \
    --culture-file culture.json --output predictions_eval.json

# Step 3: Eval — score against expected test output
python -m arc_agent.evaluate eval --data-dir ARC-AGI/data/evaluation \
    --culture-file culture.json --output results_eval.json
```

**Requirements:** Python 3.9+, NumPy 1.24+. See [INSTALL.md](INSTALL.md) for conda/venv setup and all CLI options.

## Results

### ARC-AGI-1 v0.22 (current)

| Metric | Training (400) | Evaluation (400) |
|--------|---------------|-----------------|
| **Solved (exact)** | 79/400 (19.8%) | 18/400 (4.5%) |
| Test confirmed | 82/400 (20.5%) | 23/400 (5.8%) |
| Flukes | 3 | 5 |
| Overfits | 20 | 8 |
| Mean score | 0.851 | — |
| LLM used | None | None |

Solved (exact) = pixel-perfect on train AND test (our golden metric).
287 hand-crafted primitives, neighbor-rule learning, example-parameterized color mapping, exhaustive pair + triple search, evolutionary synthesis. Pure Four Pillars — no LLMs.

## Running the Full ARC-AGI Benchmark

```bash
# 1. Clone the official ARC-AGI dataset
git clone https://github.com/fchollet/ARC-AGI.git

# 2. RECOMMENDED: Three-phase pipeline with culture transfer
#    Phase 1 — Train: learn culture from training set
python -m arc_agent.evaluate train --data-dir ARC-AGI/data/training \
    --culture-file culture.json --output results_train.json

#    Phase 2 — Infer: generate candidates (clean, no test peeking)
python -m arc_agent.evaluate infer --data-dir ARC-AGI/data/evaluation \
    --culture-file culture.json --output predictions_eval.json

#    Phase 3 — Eval: score predictions against answers
python -m arc_agent.evaluate eval --data-dir ARC-AGI/data/evaluation \
    --culture-file culture.json --output results_eval.json

# 3. Quick test on first 20 tasks (single process for easy debugging)
python -m arc_agent.evaluate train --data-dir ARC-AGI/data/training --limit 20 --workers 1

# 4. Run with explicit worker count (e.g. 10 cores on M1 Max)
python -m arc_agent.evaluate train --data-dir ARC-AGI/data/training --workers 10

# 5. Save results JSON with culture
python -m arc_agent.evaluate train --data-dir ARC-AGI/data/training \
    --output results.json --culture-file culture.json
```

## Project Structure

```
agi-mvp-general/
├── README.md                        # This file — project overview
├── INSTALL.md                       # Setup instructions (conda/venv/pip)
├── requirements.txt                 # Dependencies: numpy (runtime) + pytest (dev)
├── pyproject.toml                   # Python project configuration (v0.5.0)
├── run_tests.py                     # Test runner with built-in coverage
├── docs/                            # Documentation
│   ├── ARCHITECTURE.md              # Detailed technical architecture guide
│   ├── DESIGN_NOTES.md              # Design decisions and rationale
│   ├── RESEARCH_PLAN.md             # Formal research plan with metrics
│   └── PROMPT_LOG.md                # Full session history, prompts & results
├── arc_agent/                       # Core agent implementation
│   ├── __init__.py                  # Package definition (v0.5.0)
│   ├── concepts.py                  # Concept, ConditionalConcept, Program, Toolkit, Archive
│   ├── primitives.py                # 66 DSL grid transforms + 10 predicates
│   ├── objects.py                   # Object-level primitives (30 concepts)
│   ├── decompose.py                 # Task decomposition engine (3 strategies)
│   ├── persistence.py               # Toolkit/Archive save/load (JSON)
│   ├── synthesizer.py               # Evolutionary synthesis with conditional support
│   ├── scene.py                     # Object-centric reasoning (perceive→infer→apply)
│   ├── solver.py                    # Main learning loop (all 4 pillars)
│   ├── scorer.py                    # Feedback scoring engine (Pillar 1, NumPy)
│   ├── explorer.py                  # Explore/exploit engine (Pillar 4)
│   ├── cpu_utils.py                 # CPU topology detection (performance vs efficiency cores)
│   ├── dataset.py                   # ARC-AGI dataset loader + parallel evaluation
│   ├── evaluate.py                  # Full benchmark evaluation CLI
│   ├── sample_tasks.py              # 10 sample ARC-AGI tasks
│   └── main.py                      # CLI entry point with persistence flags
└── tests/                           # Test suite (242 tests)
    ├── test_concepts.py             # Concept system (21 tests)
    ├── test_primitives.py           # DSL grid transforms (55 tests)
    ├── test_objects.py              # Object-level primitives (19 tests)
    ├── test_scorer.py               # Scoring engine (18 tests)
    ├── test_synthesizer.py          # Program synthesis (12 tests)
    ├── test_explorer.py             # Exploration engine (14 tests)
    ├── test_persistence.py          # Serialization (7 tests)
    ├── test_dataset.py              # Dataset loader (11 tests)
    ├── test_conditionals.py         # Conditional logic (16 tests)
    ├── test_decompose.py            # Task decomposition (21 tests)
    ├── test_performance.py          # Scorer correctness + parallel eval (26 tests)
    └── test_integration.py          # Full pipeline integration (22 tests)
```

## Architecture

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

For a detailed architecture walkthrough, see [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## The Four Pillars in Code

**Pillar 1: Feedback Loops** (`scorer.py`) — Every candidate program is tested against training examples. The scorer provides continuous feedback — not just "right or wrong" but *how close* — enabling gradient-free optimization. Fully vectorized with NumPy: `np.bincount` for the ARC 10-color palette, no Python-level pixel loops.

**Pillar 2: Approximability** (`synthesizer.py`) — Evolutionary search (mutation + crossover + selection) iteratively refines programs. Partial-credit scoring creates a smooth fitness landscape where better programs survive and reproduce.

**Pillar 3: Abstraction & Composability** (`concepts.py`, `primitives.py`, `objects.py`, `decompose.py`) — Programs are sequences of composable Concepts following a recursive grammar: `Concept → Constant | Operator | If(Predicate, Concept, Concept) | Concept Op Concept`. Successful multi-step programs are promoted to first-class Concepts in the Toolkit. Object-level primitives enable object reasoning; conditional logic enables branching; task decomposition enables fractal problem-solving.

**Pillar 4: Exploration** (`explorer.py`) — UCB1 (Upper Confidence Bound) balances exploitation of known-good concepts with curiosity-driven exploration of under-tested ones. Novel programs are generated by composing concepts in untested ways.

## Documentation

| Document | Description |
|----------|-------------|
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | Detailed technical architecture, module responsibilities, data flow |
| [docs/DESIGN_NOTES.md](docs/DESIGN_NOTES.md) | Design decisions, rationale, known limitations, future directions |
| [docs/RESEARCH_PLAN.md](docs/RESEARCH_PLAN.md) | Formal research plan with metrics, protocols, and benchmarks |
| [docs/PROMPT_LOG.md](docs/PROMPT_LOG.md) | Full session history: prompts given, reasoning, results obtained |

## Roadmap

- [x] **Core 4 Pillars prototype** (feedback, approximability, composability, exploration)
- [x] **Object-level primitives** (connected components, extraction, recoloring)
- [x] **Persistent Toolkit serialization** (save/load across runs)
- [x] **Test suite with coverage** (231 tests, built-in coverage measurement)
- [x] **ARC-AGI-1 evaluation harness** (dataset loader + benchmark CLI)
- [x] **Run full ARC-AGI-1 evaluation** — 20/400 (5.0%) exact, 185/400 (46.3%) partial
- [x] **Expanded toolkit** (104 concepts with partitioning, border, color ops)
- [x] **Knowledge compounding** (near-miss concept promotion at 0.95 threshold)
- [x] **Conditional logic in programs** (`ConditionalConcept`: if-then-else branching with 10 predicates)
- [x] **Task decomposition** (`DecompositionEngine`: color-channel, spatial quadrant, diff-focus strategies)
- [x] **NumPy-accelerated scoring** (vectorized pixel accuracy + color palette scoring via `np.bincount`)
- [x] **Multiprocessing parallel evaluation** (`--workers` flag, all CPU cores, ~10× wall-clock speedup)
- [ ] **Ablation studies** (validate each pillar is necessary)
- [ ] **ARC-AGI-2** evaluation
- [ ] **ARC-AGI-3** interactive environment support (launching March 25, 2026)
- [ ] **Zork agent** (text adventure world modeling)
- [ ] **Cross-domain transfer** (can ARC concepts help in Zork?)

## Research Context

This work relates to several areas of AI research:

- **Program Synthesis**: DreamCoder (Ellis et al., 2021) — learns program libraries. We differ in using evolutionary rather than neural-guided search.
- **Evolutionary Computation**: Genetic Programming (Koza) — direct ancestor. We add UCB exploration and cross-task knowledge compounding.
- **ARC-AGI SOTA**: Confluence Labs (97.9%) uses LLMs + code generation. We deliberately avoid LLMs to isolate the 4 Pillars contribution.
- **Cognitive Science**: Cumulative Cultural Evolution (Tomasello), Constructivism (Piaget), Universal Darwinism (Campbell/Dawkins).

## License

MIT

## Citation

If you use this work, please cite:

```
@misc{jain2026fourpillars,
  author = {Jain, Vibhor},
  title = {Four Pillars of General Learning: A Framework for Artificial General Intelligence},
  year = {2026},
  url = {https://github.com/vibhor-77/agi-mvp-general}
}
```
