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
# Clone the repository
git clone https://github.com/vibhor-77/agi-mvp-general.git
cd agi-mvp-general

# Install development dependencies (optional — agent has zero runtime deps)
pip install -r requirements.txt

# Run the evaluation
python -m arc_agent.main

# Save learned knowledge for later reuse
python -m arc_agent.main --save-toolkit toolkit.json --save-archive archive.json

# Resume from saved knowledge (cumulative culture across runs)
python -m arc_agent.main --load-toolkit toolkit.json

# Run tests with coverage
python run_tests.py

# Run a single task for debugging
python -m arc_agent.main --task mirror_h
```

**Requirements:** Python 3.9+. Zero runtime dependencies (stdlib only). `pytest` and `pytest-cov` are optional dev dependencies for enhanced testing.

## Results

### ARC-AGI-1 Full Benchmark (400 training tasks)

| Metric | v0.2 | v0.3 | Change |
|--------|------|------|--------|
| **Exact solve rate** | 20/400 (5.0%) | **24/400 (6.0%)** | **+20%** |
| Test correct (held-out) | 19/400 (4.8%) | 23/400 (5.8%) | +21% |
| Partial solve rate (>80%) | 185/400 | 200/400 | +15 tasks |
| Concepts learned | 3 | 24 | **8× more compounding** |
| Toolkit growth | 73 → 76 | 104 → 118 | 14 concepts compounded |
| Total time | 159.7s (0.40s/task) | 107.0s (0.27s/task) | 33% faster |
| LLM used | None | None | Pure 4 Pillars |

### v0.3 Key Changes

| Change | Details |
|--------|---------|
| Toolkit expanded | 73 → 104 concepts (+13 partitioning, +9 fill bg, +9 erase color) |
| Concept promotion threshold | Lowered from 0.99 → 0.95 (near-miss knowledge compounds) |
| Seed generation | Richer feature-guided heuristics + 2-step combo seeds |
| New primitive categories | Grid partitioning, border extraction, color replacement, dedup, sorting |
| Test suite | 155 → 180 tests |

## Running the Full ARC-AGI Benchmark

```bash
# 1. Clone the official ARC-AGI dataset
git clone https://github.com/fchollet/ARC-AGI.git

# 2. Run on training set (400 tasks)
python -m arc_agent.evaluate --data-dir ARC-AGI/data/training

# 3. Run on evaluation set (400 tasks, held-out)
python -m arc_agent.evaluate --data-dir ARC-AGI/data/evaluation

# 4. Quick test on first 20 tasks
python -m arc_agent.evaluate --data-dir ARC-AGI/data/training --limit 20

# 5. Save results and learned toolkit
python -m arc_agent.evaluate --data-dir ARC-AGI/data/training \
    --output results.json --save-toolkit learned_toolkit.json
```

## Project Structure

```
agi-mvp-general/
├── README.md                        # This file — project overview
├── requirements.txt                 # Dev dependencies (pytest, pytest-cov)
├── pyproject.toml                   # Python project configuration
├── run_tests.py                     # Test runner with built-in coverage
├── docs/                            # Documentation
│   ├── ARCHITECTURE.md              # Detailed technical architecture guide
│   ├── DESIGN_NOTES.md              # Design decisions and rationale
│   ├── RESEARCH_PLAN.md             # Formal research plan with metrics
│   └── PROMPT_LOG.md                # Full session history, prompts & results
├── arc_agent/                       # Core agent implementation
│   ├── __init__.py                  # Package definition (v0.4.0)
│   ├── concepts.py                  # Concept, ConditionalConcept, Program, Toolkit, Archive
│   ├── primitives.py                # 66 DSL grid transforms + 10 predicates
│   ├── objects.py                   # Object-level primitives (30 concepts)
│   ├── decompose.py                 # Task decomposition engine (3 strategies)
│   ├── persistence.py               # Toolkit/Archive save/load (JSON)
│   ├── synthesizer.py               # Evolutionary synthesis with conditional support
│   ├── solver.py                    # Main learning loop (all 4 pillars)
│   ├── scorer.py                    # Feedback scoring engine (Pillar 1)
│   ├── explorer.py                  # Explore/exploit engine (Pillar 4)
│   ├── dataset.py                   # ARC-AGI dataset loader
│   ├── evaluate.py                  # Full benchmark evaluation CLI
│   ├── sample_tasks.py              # 10 sample ARC-AGI tasks
│   └── main.py                      # CLI entry point with persistence flags
└── tests/                           # Test suite (216 tests)
    ├── test_concepts.py             # Unit tests for concept system (21 tests)
    ├── test_primitives.py           # Unit tests for DSL primitives (55 tests)
    ├── test_objects.py              # Unit tests for object primitives (19 tests)
    ├── test_scorer.py               # Unit tests for scoring engine (18 tests)
    ├── test_synthesizer.py          # Unit tests for program synthesis (12 tests)
    ├── test_explorer.py             # Unit tests for exploration engine (14 tests)
    ├── test_persistence.py          # Unit tests for serialization (7 tests)
    ├── test_dataset.py              # Unit tests for dataset loader (11 tests)
    ├── test_conditionals.py         # Unit tests for conditional logic (16 tests)
    ├── test_decompose.py            # Unit tests for task decomposition (21 tests)
    └── test_integration.py          # Integration tests (full pipeline, 22 tests)
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

**Pillar 1: Feedback Loops** (`scorer.py`) — Every candidate program is tested against training examples. The scorer provides continuous feedback — not just "right or wrong" but *how close* — enabling gradient-free optimization.

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
- [x] **Test suite with coverage** (180 tests, built-in coverage measurement)
- [x] **ARC-AGI-1 evaluation harness** (dataset loader + benchmark CLI)
- [x] **Run full ARC-AGI-1 evaluation** — 20/400 (5.0%) exact, 185/400 (46.3%) partial
- [x] **Expanded toolkit** (104 concepts with partitioning, border, color ops)
- [x] **Knowledge compounding** (near-miss concept promotion at 0.95 threshold)
- [x] **Conditional logic in programs** (`ConditionalConcept`: if-then-else branching with 10 predicates)
- [x] **Task decomposition** (`DecompositionEngine`: color-channel, spatial quadrant, diff-focus strategies)
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
