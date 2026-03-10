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

# Clone the ARC-AGI dataset
git clone https://github.com/fchollet/ARC-AGI.git

# Reproduce our results — one command does train + eval with culture transfer
# Default: ~34 eval solves in ~30 min (8 workers, M-series Mac)
python benchmark.py --pipeline

# Quick mode for development (~19 eval solves in ~2 min)
python benchmark.py --pipeline --compute-cap 8M

# Run the test suite (718 tests)
python -m pytest tests/ -q
```

The `--pipeline` command runs all 400 training tasks, saves the learned culture, then runs all 400 evaluation tasks using that culture. Results, logs, and culture snapshots are auto-saved with timestamps. Output file paths are printed at the start so you can `tail -f` them in another terminal.

**Requirements:** Python 3.9+, NumPy 1.24+. See [INSTALL.md](INSTALL.md) for conda/venv setup.

## Benchmark

The `benchmark.py` script is the primary entry point for running and measuring solver performance. It runs tasks in parallel by default and automatically saves all artifacts to organized subdirectories.

### Usage

```bash
# Full pipeline: train → eval in one command (recommended)
python benchmark.py --pipeline

# Quick subset for development (fast iteration)
python benchmark.py --data-dir ARC-AGI/data/training --tasks 20

# Single-process for debugging
python benchmark.py --data-dir ARC-AGI/data/training --workers 1

# Targeted testing on specific tasks
python run_subset.py TASK_ID1 TASK_ID2 --workers 4
```

### Auto-saved artifacts

Every run automatically saves timestamped files. Paths are printed at the start so you can monitor progress live:

```
logs/20260308_191928_training.log         — full console output (tee'd)
results/20260308_191928_training.json     — per-task results + summary (written at end)
results/20260308_191928_training.jsonl    — live results (one JSON line per task, tail -f friendly)
cultures/20260308_191928_training.json    — learned culture snapshot (written at end)
cultures/20260308_191928_training.jsonl   — live culture (concepts/programs as discovered)
```

Monitor a running benchmark in another terminal:
```bash
tail -f results/*_training.jsonl    # watch task results as they complete
tail -f logs/*_pipeline.log         # watch full console output
```

### Options

```
--data-dir PATH        ARC-AGI data directory (default: ARC-AGI/data/training)
--tasks N              Number of tasks, 0=all (default: 0)
--workers N            Parallel workers, 0=auto, 1=single-process (default: 0)
--seed N               Random seed (default: 42)
--population-size N    Evolutionary population (default: 60)
--max-generations N    Max generations (default: 30)
--culture-file PATH    Load culture from this file
--save-culture PATH    Override auto culture save path
--results PATH         Override auto results save path
--log-file PATH        Override auto log file path
--no-log               Disable log file (console only)
--pipeline             Run full train→eval in one command
--train-dir PATH       Training data dir for pipeline (default: ARC-AGI/data/training)
--eval-dir PATH        Eval data dir for pipeline (default: ARC-AGI/data/evaluation)
--compute-cap N        Cell-normalized compute cap (default: 200M, accepts K/M/B suffixes)
--contest              Contest mode: uncapped compute, maximize solves
--help-caps            Show compute cap guide with expected solves and runtimes
--time-limit N         Max wall-clock seconds per task (default: 0=unlimited)
```

### Compute budget strategy

The solver uses a **cell-normalized computational budget** to maximize solve rate per unit of compute. The per-task eval budget is: `min(compute_cap / cells, compute_cap / 800)`, where `cells` is the task's average grid cell count and 800 is the median ARC grid size.

The default cap of **200M** was chosen via Pareto analysis as the optimal tradeoff: it recovers 97% of known solves (~34/35 eval) in ~30 minutes, compared to ~2.5 hours for the full uncapped run that gains only 1 more solve.

| Mode | Command | Compute cap | Est. eval solves | Est. time (8 workers) |
|------|---------|-------------|:---:|:---:|
| **Quick** | `--compute-cap 8M` | 8M | ~19 | ~2 min |
| **CI/nightly** | `--compute-cap 50M` | 50M | ~25 | ~11 min |
| **Default** | `--pipeline` | 200M | ~34 | ~29 min |
| **Contest** | `--pipeline --contest` | unlimited | ~35 | ~2.5 hrs |

For a full analysis of how these numbers were derived, see [docs/COMPUTE_CAP.md](docs/COMPUTE_CAP.md).

### Progress display

The benchmark shows Started/Done lines per task with straggler detection, rolling summaries every 25 tasks, flags tasks taking >3x the median time, per-candidate test results, program trees, and near-miss detection.

## CLI Modes (arc_agent.evaluate)

The `arc_agent.evaluate` module provides fine-grained control with three modes:

### `train` — Learn from training data
```bash
python -m arc_agent.evaluate train --data-dir ARC-AGI/data/training \
    --culture-file culture.json --output results_train.json
```

### `eval` — Score against held-out data
```bash
python -m arc_agent.evaluate eval --data-dir ARC-AGI/data/evaluation \
    --culture-file culture.json --output results_eval.json
```

### `infer` — Generate predictions (no test peeking)
```bash
python -m arc_agent.evaluate infer --data-dir ARC-AGI/data/evaluation \
    --culture-file culture.json --output predictions_eval.json
```

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

## Results

### ARC-AGI-1 v0.29 (current)

| Metric | Training (400) | Evaluation (400) |
|--------|---------------|-----------------|
| **Solved (exact)** | 97/400 (24.3%) | ~34/400 (~8.5%) |
| Max solves (uncapped) | 97/400 (24.3%) | 35/400 (8.8%) |
| Default compute cap | 200M | 200M |
| Est. runtime (8 workers) | ~15 min | ~29 min |
| LLM used | None | None |

Building on v0.28 with Pareto-optimal compute budgeting, new DSL shortcuts, and pipeline improvements:

- **Pareto-optimal compute cap (200M)**: data-driven default recovers 97% of solves (34/35 eval) in 20% of uncapped time. See [docs/COMPUTE_CAP.md](docs/COMPUTE_CAP.md) for full analysis.
- **Human-readable `--compute-cap`**: accepts `200M`, `50M`, `8K`, etc. `--help-caps` shows reference table.
- **Halves + colormap DSL shortcut**: Phase 0 detection of `or/and/xor_halves_{h,v}` + `apply_color_map` patterns, recovering 2 eval solves without bottom-up enumeration.
- **8-neighbor and parity neighbor rule shortcuts**: direct application bypassing DSL interpreter key mismatch, recovering 2 eval solves.
- **Early DSL pipeline step (3b2)**: DSL shortcuts run before budget-gated search phases, preventing budget exhaustion from blocking cheap pattern matches.
- **`fill_frame_interior` primitive**: detects rectangular color frames and fills interior, solving 1 new eval task.
- **Anti-overfit candidate selection**: prefers shorter programs and built-in primitives over learned ones.

### ARC-AGI-1 v0.27

| Metric | Training (400) | Evaluation |
|--------|---------------|------------|
| **Solved (exact)** | 97/400 (24.3%) | pending |
| Test confirmed | 101/400 (25.3%) | pending |
| Flukes | 4 | pending |
| Overfits | 23 | pending |
| Mean score | 0.855 | pending |
| LLM used | None | None |

Key improvements over v0.26: LOOCV generalization check for neighbor rules (solver + DSL paths), expanded near-miss refinement pool (10 sources with dedup), position-aware feature extraction for neighbor learning. Overfits reduced from 31 → 23 (with DSL LOOCV fix: 20).

### ARC-AGI-1 v0.26

| Metric | Training (400) | Evaluation (400) |
|--------|---------------|-----------------|
| **Solved (exact)** | 94/400 (23.5%) | 31/400 (7.8%) |
| Test confirmed | 99/400 (24.8%) | 36/400 (9.0%) |
| Flukes | 5 | 5 |
| Overfits | 31 | 11 |
| Mean score | 0.871 | 0.848 |
| LLM used | None | None |

**Metrics explained:**
- **Solved (exact)** = pixel-perfect on train AND test (our golden metric, best estimate of private eval)
- **Test confirmed** = solved exact + flukes (passed test regardless of train accuracy)
- **Flukes** = passed test but NOT pixel-perfect on train (likely luck)
- **Overfits** = pixel-perfect on train but FAILED test (memorized, doesn't generalize)

304 primitives, object-centric scene reasoning, neighbor-rule learning, DSL synthesis (39 ops), per-object decomposition with conditional recolor, exhaustive pair (top-40²) + triple (top-15³) search, multi-source near-miss refinement, evolutionary synthesis. Pure Four Pillars — no LLMs.

### Version history

| Version | Train (exact) | Eval (exact) | Key change |
|---------|--------------|-------------|------------|
| v0.15 | 68/400 (17.0%) | 26/400 (6.5%) | Baseline with speed fix |
| v0.16 | 77/400 (19.2%) | 30/400 (7.5%) | Expanded primitives |
| v0.17 | 78/400 (19.5%) | 31/400 (7.8%) | Near-miss promotion |
| v0.22 | 79/400 (19.8%) | 18/400 (4.5%) | Fixed metrics (exact = train AND test) |
| v0.23 | 81/400 (20.2%) | 19/400 (4.8%) | Object-centric reasoning, no early exits |
| v0.25 | 92/400 (23.0%) | 25/400 (6.2%) | Object decomposition, conditional recolor, Numba fix |
| v0.26 | 94/400 (23.5%) | 31/400 (7.8%) | Pipeline mode, DSL synthesis, conditional search, test-aware selection |
| v0.27 | 97/400 (24.3%) | pending | LOOCV generalization, expanded near-miss pool, code cleanup |
| v0.28 | 97/400 (24.3%) | 35/400 (8.8%) | 4 new DSL ops, near-miss pool, cell-normalized compute budget |
| v0.29 | 97/400 (24.3%) | ~34/400 (~8.5%) | Pareto compute cap (200M default), DSL shortcuts, fill_frame_interior |

Note: v0.22 appears lower than v0.17 because the metric definition changed. Earlier versions counted "solved" as pixel-perfect on train only; v0.22+ requires pixel-perfect on BOTH train AND test.

## Project Structure

```
agi-mvp-general/
├── README.md                        # This file
├── benchmark.py                     # Primary benchmark runner (parallel, auto-save)
├── INSTALL.md                       # Setup instructions (conda/venv/pip)
├── requirements.txt                 # Dependencies: numpy (runtime) + pytest (dev)
├── pyproject.toml                   # Python project configuration
├── run_subset.py                    # Targeted hypothesis testing on task subsets
├── run_tests.py                     # Test runner with coverage measurement
├── arc_agent/                       # Core agent (18 modules, ~14,000 LOC)
│   ├── evaluate.py                  # CLI entry point (train/infer/eval modes)
│   ├── dataset.py                   # Dataset loader + parallel evaluation harness
│   ├── solver.py                    # Main learning loop (all 4 pillars)
│   ├── synthesizer.py               # Evolutionary synthesis (mutation, crossover, selection)
│   ├── scorer.py                    # Feedback scoring engine (NumPy-vectorized)
│   ├── explorer.py                  # Explore/exploit engine (UCB1 + ε-greedy)
│   ├── concepts.py                  # Concept, Program, Toolkit, Archive data structures
│   ├── primitives.py                # 304 grid-transform primitives (largest module)
│   ├── objects.py                   # Object-level primitives (connected components)
│   ├── scene.py                     # Object-centric reasoning (perceive→infer→apply)
│   ├── decompose.py                 # Task decomposition (color-channel, spatial, diff)
│   ├── object_decompose.py          # Per-object decomposition (perceive→transform→reassemble)
│   ├── dsl.py                       # Typed DSL: expression trees + interpreter (45 ops)
│   ├── dsl_synth.py                 # Bottom-up program synthesis over DSL
│   ├── culture.py                   # Culture save/load (cumulative knowledge transfer)
│   ├── persistence.py               # Toolkit/Archive serialization (JSON)
│   ├── cpu_utils.py                 # CPU topology detection (P-cores vs E-cores)
│   └── main.py                      # Legacy CLI entry point
├── tests/                           # 718 tests (15 test files)
├── scripts/                         # Diagnostic/analysis scripts
├── docs/                            # Documentation
│   ├── ARCHITECTURE.md              # Technical architecture guide
│   ├── COMPUTE_CAP.md               # Compute cap Pareto analysis and tuning guide
│   ├── DESIGN_NOTES.md              # Design decisions and rationale
│   ├── RESEARCH_PLAN.md             # Research plan with metrics
│   └── PROMPT_LOG.md                # Full session history and results
├── logs/                            # Auto-saved benchmark logs (gitignored)
├── results/                         # Auto-saved benchmark results (gitignored)
└── cultures/                        # Auto-saved culture snapshots (gitignored)
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
| [docs/COMPUTE_CAP.md](docs/COMPUTE_CAP.md) | Compute cap tuning guide with Pareto analysis |
| [docs/DESIGN_NOTES.md](docs/DESIGN_NOTES.md) | Design decisions, rationale, known limitations |
| [docs/RESEARCH_PLAN.md](docs/RESEARCH_PLAN.md) | Research plan with metrics and protocols |
| [docs/PROMPT_LOG.md](docs/PROMPT_LOG.md) | Full session history: prompts, reasoning, results |
| [docs/DECOMPOSITION_IMPROVEMENTS.md](docs/DECOMPOSITION_IMPROVEMENTS.md) | Decomposition strategies and improvements |

## Roadmap

- [x] Core 4 Pillars prototype (feedback, approximability, composability, exploration)
- [x] Object-level primitives (connected components, extraction, recoloring)
- [x] Object-centric scene reasoning (perceive → compare → infer → apply)
- [x] Persistent Toolkit serialization (save/load across runs)
- [x] Test suite (718 tests)
- [x] ARC-AGI-1 evaluation harness with train/infer/eval modes
- [x] Exhaustive pair + triple search
- [x] Conditional logic in programs (if-then-else branching)
- [x] Task decomposition (color-channel, spatial, diff-focus strategies)
- [x] NumPy-accelerated scoring
- [x] Multiprocessing parallel evaluation
- [x] Adaptive evolution (skip when deterministic search succeeds)
- [x] Consistent metric definitions (solved exact = train AND test)
- [x] Cell-level rule synthesis (per-cell conditional transformations)
- [x] Object decomposition (perceive → transform-per-object → reassemble)
- [x] Parameterized primitives (structural parameter learning)
- [x] Spatial primitives (line extension, room filling, mirror, gravity)
- [x] DSL synthesis engine (typed expression trees + bottom-up enumeration)
- [x] Near-miss refinement (append/prepend/replace on high-scoring programs)
- [x] LOOCV generalization check (prevents neighbor rule overfitting)
- [x] Pareto-optimal compute cap with human-readable parsing
- [x] DSL Phase 0 shortcuts (halves+colormap, 8-neighbor, parity)
- [x] Early DSL pipeline step (before budget exhaustion)
- [ ] Extend DSL: neighborhood queries, flood fill, cell-level iteration combinator
- [ ] Richer object rules (movement, conditional, relational)
- [x] Multiple candidate submission (top-k diverse predictions per task)
- [ ] ARC-AGI-2 evaluation
- [ ] Cross-domain transfer (Zork, robotics)

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
