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
pip install numpy

# Run the test suite (480 tests)
python -m unittest discover -s tests -p "*.py"

# Clone the ARC-AGI dataset
git clone https://github.com/fchollet/ARC-AGI.git

# Run the full pipeline (see CLI Modes below)
python -m arc_agent.evaluate train --data-dir ARC-AGI/data/training \
    --culture-file culture.json --output results_train.json
python -m arc_agent.evaluate eval --data-dir ARC-AGI/data/evaluation \
    --culture-file culture.json --output results_eval.json
```

**Requirements:** Python 3.9+, NumPy 1.24+. See [INSTALL.md](INSTALL.md) for conda/venv setup.

## CLI Modes

The benchmark has three modes with clean separation of concerns:

### `train` — Learn from training data

Runs the solver with full access to training answers. Discovers concepts and programs, saves them to a culture file for reuse. Uses exhaustive search with multiple restarts.

```bash
python -m arc_agent.evaluate train \
    --data-dir ARC-AGI/data/training \
    --culture-file culture.json \
    --output results_train.json
```

Culture file is **saved** after the run. This is the most expensive mode — it invests compute to build up knowledge.

### `eval` — Score against held-out data

Runs the solver on evaluation tasks and scores results against expected test output. Loads culture from training. Produces the full scoreboard with solved(exact), flukes, overfits.

```bash
python -m arc_agent.evaluate eval \
    --data-dir ARC-AGI/data/evaluation \
    --culture-file culture.json \
    --output results_eval.json
```

Culture file is **loaded** before the run.

### `infer` — Generate predictions (no test peeking)

Same as eval but never looks at test output. Outputs ranked candidates per task with their programs. Use this for private eval submission — clean separation guarantees no data leakage.

```bash
python -m arc_agent.evaluate infer \
    --data-dir ARC-AGI/data/evaluation \
    --culture-file culture.json \
    --output predictions_eval.json
```

### Recommended workflow

```bash
# Step 1: Train (learn culture from training set)
python -m arc_agent.evaluate train \
    --data-dir ARC-AGI/data/training \
    --culture-file culture.json \
    --output results_train.json

# Step 2: Eval (score on held-out evaluation set)
python -m arc_agent.evaluate eval \
    --data-dir ARC-AGI/data/evaluation \
    --culture-file culture.json \
    --output results_eval.json
```

For private eval submission, replace `eval` with `infer` in Step 2.

### Common options

```
--limit N        Only run first N tasks (sorted by ID)
--tasks ID ...   Run only specific task IDs (space-separated)
--workers N      Parallel workers (0=auto, 1=debug)
--population N   Evolutionary population size (default: 60)
--generations N  Max evolution generations (default: 30)
--seed N         Random seed for reproducibility (default: 42)
--top-k N        Candidates to submit per task (default: 3)
--quiet          Suppress per-task output
```

### Quick debugging run

```bash
# Run a small subset
python -m arc_agent.evaluate train \
    --data-dir ARC-AGI/data/training \
    --limit 20 --workers 1

# Run specific tasks by ID
python -m arc_agent.evaluate train \
    --data-dir ARC-AGI/data/training \
    --tasks 0b148d64 2204b7a8 3c9b0459
```

## Results

### ARC-AGI-1 v0.25 (current)

| Metric | Training (400) | Evaluation (400) |
|--------|---------------|-----------------|
| **Solved (exact)** | 92/400 (23.0%) | 25/400 (6.2%) |
| Test confirmed | 96/400 (24.0%) | 30/400 (7.5%) |
| Flukes | 4 | 5 |
| Overfits | 26 | 9 |
| Mean score | 0.868 | 0.834 |
| Median score | 0.936 | 0.907 |
| LLM used | None | None |

**Metrics explained:**
- **Solved (exact)** = pixel-perfect on train AND test (our golden metric, best estimate of private eval)
- **Test confirmed** = solved exact + flukes (passed test regardless of train accuracy)
- **Flukes** = passed test but NOT pixel-perfect on train (likely luck)
- **Overfits** = pixel-perfect on train but FAILED test (memorized, doesn't generalize)

287 hand-crafted primitives, object-centric scene reasoning, neighbor-rule learning, color mapping, per-object decomposition with conditional recolor, exhaustive pair + triple search, evolutionary synthesis with multiple restarts. Pure Four Pillars — no LLMs.

### Version history

| Version | Train (exact) | Eval (exact) | Key change |
|---------|--------------|-------------|------------|
| v0.15 | 68/400 (17.0%) | 26/400 (6.5%) | Baseline with speed fix |
| v0.16 | 77/400 (19.2%) | 30/400 (7.5%) | Expanded primitives |
| v0.17 | 78/400 (19.5%) | 31/400 (7.8%) | Near-miss promotion |
| v0.22 | 79/400 (19.8%) | 18/400 (4.5%) | Fixed metrics (exact = train AND test) |
| v0.23 | 81/400 (20.2%) | 19/400 (4.8%) | Object-centric reasoning, no early exits |
| v0.25 | 92/400 (23.0%) | 25/400 (6.2%) | Object decomposition, conditional recolor, Numba fix |

Note: v0.22 appears lower than v0.17 because the metric definition changed. Earlier versions counted "solved" as pixel-perfect on train only; v0.22+ requires pixel-perfect on BOTH train AND test.

## Project Structure

```
agi-mvp-general/
├── README.md                        # This file
├── INSTALL.md                       # Setup instructions (conda/venv/pip)
├── requirements.txt                 # Dependencies: numpy (runtime) + pytest (dev)
├── pyproject.toml                   # Python project configuration
├── arc_agent/                       # Core agent (18 modules, ~6,400 LOC)
│   ├── evaluate.py                  # CLI entry point (train/infer/eval modes)
│   ├── dataset.py                   # Dataset loader + parallel evaluation harness
│   ├── solver.py                    # Main learning loop (all 4 pillars)
│   ├── synthesizer.py               # Evolutionary synthesis (mutation, crossover, selection)
│   ├── scorer.py                    # Feedback scoring engine (NumPy-vectorized)
│   ├── explorer.py                  # Explore/exploit engine (UCB1 + ε-greedy)
│   ├── concepts.py                  # Concept, Program, Toolkit, Archive data structures
│   ├── primitives.py                # 287 DSL grid transforms (largest module)
│   ├── objects.py                   # Object-level primitives (connected components)
│   ├── scene.py                     # Object-centric reasoning (perceive→infer→apply)
│   ├── decompose.py                 # Task decomposition (color-channel, spatial, diff)
│   ├── object_decompose.py          # Per-object decomposition (perceive→transform→reassemble)
│   ├── culture.py                   # Culture save/load (cumulative knowledge transfer)
│   ├── persistence.py               # Toolkit/Archive serialization (JSON)
│   ├── cpu_utils.py                 # CPU topology detection (P-cores vs E-cores)
│   └── main.py                      # Legacy CLI entry point
├── tests/                           # 461 tests (~5,500 LOC)
│   ├── test_primitives.py           # Grid transforms (55 tests)
│   ├── test_scene.py                # Object-centric reasoning (26 tests)
│   ├── test_concepts.py             # Concept system (21 tests)
│   ├── test_decompose.py            # Task decomposition (21 tests)
│   ├── test_integration.py          # Full pipeline (22 tests)
│   ├── test_performance.py          # Parallel eval + scorer correctness (26 tests)
│   └── ...                          # 14 test files total
├── scripts/                         # Diagnostic/analysis scripts
│   ├── analyze_object_rules.py      # Pattern analysis across tasks
│   └── debug_recolor.py             # Deep dive into recolor failures
└── docs/                            # Documentation
    ├── ARCHITECTURE.md              # Technical architecture guide
    ├── DESIGN_NOTES.md              # Design decisions and rationale
    ├── RESEARCH_PLAN.md             # Research plan with metrics
    └── PROMPT_LOG.md                # Full session history and results
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     ENVIRONMENT                              │
│  (ARC-AGI grids / any interactive environment)               │
└──────────────────────────┬──────────────────────────────────┘
                           │ Observation + Feedback
                           ▼
┌──────────────────────────────────────────────────────────────┐
│                    FOUR PILLARS AGENT                         │
│                                                              │
│  ┌────────────────┐  ┌─────────────────┐  ┌──────────────┐  │
│  │   TOOLKIT       │  │  SYNTHESIZER    │  │  EXPLORER    │  │
│  │  (Pillar 3)     │←→│  (Pillar 2)     │←→│  (Pillar 4)  │  │
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
│  ┌─────────────────────────────────────────────────────────┐ │
│  │  SCENE — object-centric reasoning (perceive→infer→apply) │ │
│  └─────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────┘
```

For a detailed architecture walkthrough, see [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## The Four Pillars in Code

**Pillar 1: Feedback Loops** (`scorer.py`) — Every candidate program is tested against training examples. The scorer provides continuous feedback — not just "right or wrong" but *how close* — enabling gradient-free optimization. Fully vectorized with NumPy.

**Pillar 2: Approximability** (`synthesizer.py`) — Evolutionary search (mutation + crossover + selection) iteratively refines programs. Partial-credit scoring creates a smooth fitness landscape where better programs survive and reproduce.

**Pillar 3: Abstraction & Composability** (`concepts.py`, `primitives.py`, `objects.py`, `scene.py`, `decompose.py`) — Programs are sequences of composable Concepts. Successful programs are promoted to first-class Concepts. Object-centric reasoning enables perception-level abstraction; task decomposition enables fractal problem-solving.

**Pillar 4: Exploration** (`explorer.py`) — UCB1 (Upper Confidence Bound) balances exploitation of known-good concepts with curiosity-driven exploration. Novel programs are generated by composing concepts in untested ways.

## Documentation

| Document | Description |
|----------|-------------|
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | Technical architecture, module responsibilities, data flow |
| [docs/DESIGN_NOTES.md](docs/DESIGN_NOTES.md) | Design decisions, rationale, known limitations |
| [docs/RESEARCH_PLAN.md](docs/RESEARCH_PLAN.md) | Research plan with metrics and protocols |
| [docs/PROMPT_LOG.md](docs/PROMPT_LOG.md) | Full session history: prompts, reasoning, results |
| [docs/CELL_RULES.md](docs/CELL_RULES.md) | Cell Rule DSL for per-cell conditional transformations |

## Roadmap

- [x] Core 4 Pillars prototype (feedback, approximability, composability, exploration)
- [x] Object-level primitives (connected components, extraction, recoloring)
- [x] Object-centric scene reasoning (perceive → compare → infer → apply)
- [x] Persistent Toolkit serialization (save/load across runs)
- [x] Test suite (480 tests)
- [x] ARC-AGI-1 evaluation harness with train/infer/eval modes
- [x] Exhaustive pair + triple search
- [x] Conditional logic in programs (if-then-else branching)
- [x] Task decomposition (color-channel, spatial, diff-focus strategies)
- [x] NumPy-accelerated scoring
- [x] Multiprocessing parallel evaluation
- [x] Multiple evolution restarts (3x during training)
- [x] Consistent metric definitions (solved exact = train AND test)
- [x] Cell-level rule synthesis (per-cell conditional transformations)
- [x] Object decomposition (perceive → transform-per-object → reassemble)
- [x] Parameterized primitives (structural parameter learning)
- [x] Spatial primitives (line extension, room filling, mirror, gravity)
- [ ] Richer object rules (movement, conditional, relational)
- [x] Multiple candidate submission (top-k diverse predictions per task)
- [ ] ARC-AGI-2 evaluation
- [ ] Cross-domain transfer experiments

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
