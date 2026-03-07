# Prompt Log & Session History

This document records the prompts given to Claude, the reasoning behind each step, and the results obtained. It serves as a living record of the project's evolution.

---

## Session 1 — Project Genesis (March 2026)

### Prompt

> I would like to make progress towards AGI using a 4 pillars model:
> 1. Feedback Loops
> 2. Approximability
> 3. Abstraction & Composability
> 4. Exploration
>
> I would like to prove this on a widely accepted benchmark, e.g. I am thinking of starting with ARC-AGI and also Zork.

**Uploaded documents:**
- `Principles_of_General_Learning.pdf` — Core framework definitions
- `First_Principles_of_Learning.pdf` — Formal foundations
- `Abstraction_and_Composition.pdf` — Pillar 3 deep dive
- Video presentation on the 4 Pillars

**User clarifications:**
- Wanted both a research plan document AND a working prototype
- Requested document review before implementation

### Claude's Reasoning

After reviewing all uploaded documents, the key insights extracted were:

1. **Recursive Concept Grammar**: `Concept → Constant | Relationship | Operator | Concept Op Concept` — this maps directly to composable program synthesis.
2. **Dual Memory System**: Toolkit (Type 1, timeless) + Archive (Type 2, episodic) — this solves the "Reset Button Problem" where knowledge doesn't compound.
3. **Cumulative Culture**: The system must grow smarter over time, not reset. Successful compositions become first-class concepts.
4. **Substrate Independence**: The 4 pillars should work regardless of implementation substrate (neural, symbolic, evolutionary).
5. **No LLM in the loop**: Deliberate choice to isolate the 4 Pillars contribution from pre-trained knowledge.

**Architecture decisions:**
- **Program synthesis over neural networks**: Programs are inherently composable and inspectable, directly implementing Pillar 3.
- **Evolutionary search over gradient descent**: Works on discrete program spaces, models biological evolution (core metaphor in Vibhor's framework).
- **UCB1 for exploration**: Mathematically principled (optimal regret bounds) rather than arbitrary epsilon.
- **Partial-credit scoring**: Creates smooth fitness landscape enabling gradient-free convergence.

### Results — v0.1 Prototype

| Metric | Result |
|--------|--------|
| Training solve rate | 10/10 (100%) |
| Test solve rate (held-out) | 9/10 (90%) |
| Multi-step compositions evolved | 2 |
| Knowledge compounding demonstrated | Yes (Task 10 reused Task 6's concept) |
| Runtime dependencies | Zero (stdlib only) |

**Key demonstration:** Task 6 evolved `crop_nonzero → mirror_v → rotate_180` in 5 generations. Task 10 then reused this learned concept (`learned_crop_then_mirror_43`) combined with `invert_colors → swap_4_to_1`, solving a composition problem by building on prior knowledge. This is cumulative culture in action.

### ARC-AGI Research Context

- ARC-AGI-1 SOTA: 97.9% (Confluence Labs, using LLMs)
- ARC-AGI-2: Harder, human baseline ~75%
- ARC-AGI-3: Interactive environments, launching March 25, 2026
- Zork SOTA: ~20% (Claude Opus 4.5)

---

## Session 2 — Documentation, Tests & GitHub (March 2026)

### Prompt

> Upload to GitHub including documentation, README, prompt, the plan, your thoughts and unit and integration tests.

**Global instructions added:**
1. Keep repo consistent — all tests and code working at all times
2. Upload to GitHub every time — give push commands after every change
3. Verify claims — actually run and confirm things work before returning control
4. TDD — unit test and integration test everything

### Work Done

- Created README.md, DESIGN_NOTES.md, pyproject.toml, requirements.txt, .gitignore
- Wrote 6 test files (test_concepts.py, test_primitives.py, test_scorer.py, test_synthesizer.py, test_explorer.py, test_integration.py)
- Fixed pytest fixture issues (sed had stripped `@pytest.fixture` decorators)
- Created run_tests.py fallback runner for environments without pytest
- Removed unused `import numpy as np` from concepts.py
- Fixed README with correct repo URL (`vibhor-77/agi-mvp-general`)
- Verified: 38/38 fallback tests pass, 10/10 training, 9/10 test validation

### Issues Encountered & Resolved

1. **Sandbox network restrictions**: Could not install pytest or clone ARC dataset. Solution: embedded sample tasks, created unittest fallback runner.
2. **Fixture stripping**: `sed` cleanup of pytest imports also removed `@pytest.fixture` decorators. Solution: rewrote all test files.
3. **False numpy dependency**: `import numpy as np` existed but `np.` was never used. Solution: removed the import.
4. **Wrong repo URL**: README had `vibhor-jain/four-pillars-agi`. Solution: corrected to `vibhor-77/agi-mvp-general`.

---

## Session 3 — Next Steps from Roadmap (March 2026)

### Prompt (Updated Global Instructions)

> Continue next steps from the roadmap. Updated global instructions:
> - Code should be world class: minimal, elegant, comprehensive, well commented, documented
> - Working and well tested with high code/branch coverage
> - TDD where possible and sensible
> - Upload to GitHub every time
> - Verify all claims
> - Keep all documentation up to date, remove obsolete stuff
> - Use docs folder, tie together in README
> - Always keep documentation listing the entire prompt, thoughts, results, plan, and next steps

### Claude's Reasoning

The roadmap Phase 1 priorities are:
1. **Object-level primitives** — The current DSL operates on whole grids. Real ARC tasks need connected component extraction, object property detection, and object-level transforms.
2. **Conditional logic** — Programs are linear chains; many ARC tasks need if-then-else branching.
3. **Persistent toolkit serialization** — Save/load the Toolkit across runs to truly solve the "Reset Button Problem".
4. **Coverage measurement** — Need to track and report test coverage.

### Results — v0.2

**New modules implemented (TDD):**
1. **`objects.py`** — Object-level primitives. Tests written first (`test_objects.py`, 19 tests), then implementation. Includes `GridObject` dataclass, `find_objects()` flood-fill, `extract_largest/smallest_object`, `remove_color`, `isolate_color`, `recolor_largest_object`, `mirror_objects_horizontal`, and 27 factory-generated per-color concepts.

2. **`persistence.py`** — Toolkit & Archive serialization. Tests written first (`test_persistence.py`, 7 tests), then implementation. Composed concepts are stored as recipes (child name lists) and re-composed on load via topological resolution. Full roundtrip verified: concepts, usage stats, and composed behavior all survive.

3. **Updated `primitives.py`** — `build_initial_toolkit(include_objects=True)` now includes 30 object concepts by default, bringing the initial toolkit from 43 to 73 concepts.

4. **Updated `main.py`** — CLI with `argparse`: `--save-toolkit`, `--load-toolkit`, `--save-archive`, `--task` flags.

5. **Updated `run_tests.py`** — Custom coverage measurement using Python's `trace` module. Reports per-module line coverage without needing `pytest-cov`.

6. **Converted all existing tests** from pytest to pure unittest (no pytest dependency required).

7. **Documentation restructure** — Moved DESIGN_NOTES.md and RESEARCH_PLAN.md into `docs/`. Created `docs/ARCHITECTURE.md` and `docs/PROMPT_LOG.md`.

| Metric | v0.1 | v0.2 |
|--------|------|------|
| Initial toolkit | 43 concepts | 73 concepts |
| Training solve rate | 10/10 | 10/10 |
| Test solve rate | 9/10 | 9/10 |
| Total tests | 38 (fallback) | 144 |
| Line coverage | unmeasured | 47.6% |
| Persistence | None | Full JSON roundtrip |
| Object reasoning | None | Connected components, extraction, recoloring |

---

## Session 3 (continued) — ARC-AGI Benchmark Harness

### Prompt

> Continue from where you left off. It is ok to pip install some utility and lightweight dependencies like pytest, pytest-cov etc. Also, did you run the full ARC-AGI benchmark, or is it part of the instructions?

### Claude's Reasoning

The full ARC-AGI benchmark has NOT been run yet. The 10/10 results were on 10 hand-crafted sample tasks. The sandbox blocks both pip installs and git clones (network access restricted), so we cannot download the ARC-AGI dataset or install pytest here. However, we can build all the infrastructure so running the real benchmark is a single command locally.

### Work Done

1. **`dataset.py`** — ARC-AGI dataset loader with `load_task()`, `load_dataset()`, and `evaluate_dataset()`. Handles the official JSON format, progress tracking, per-task metrics, test validation, and result export.

2. **`evaluate.py`** — Full benchmark CLI with `--data-dir`, `--output`, `--limit`, `--save-toolkit`, `--save-archive`, `--population`, `--generations`, `--seed`, `--quiet` flags.

3. **`test_dataset.py`** — 11 tests (TDD): task loading, dataset loading, evaluation harness, result export, metric validation.

4. Updated README with benchmark instructions, updated project structure and test count.

| Metric | Value |
|--------|-------|
| Total tests | 155 |
| All passing | Yes |
| Evaluation harness ready | Yes |
| Full benchmark run | Not yet (requires dataset download) |

### How to Run the Full Benchmark (User Action Required)

```bash
git clone https://github.com/fchollet/ARC-AGI.git
python -m arc_agent.evaluate --data-dir ARC-AGI/data/training
```

---

## Session 4 — ARC-AGI-1 Benchmark Results & v0.3 Improvements (March 2026)

### Prompt

> [User ran the full ARC-AGI-1 benchmark locally and pasted 400-task output]

### ARC-AGI-1 Benchmark Results (v0.2)

The user ran `python -m arc_agent.evaluate --data-dir ARC-AGI/data/training` on all 400 training tasks:

| Metric | Result |
|--------|--------|
| **Solved (exact)** | **20/400 (5.0%)** |
| Partial (>80%) | 185/400 (46.3%) |
| Test correct | 19/400 (4.8%) |
| Total time | 159.7s (avg 0.40s/task) |
| Toolkit growth | 73 → 76 (only 3 concepts learned) |

### Analysis

1. **185 tasks scored >80% but only 20 exact** — the evolutionary search gets close but can't finish. Huge improvement opportunity.
2. **Only 3 new concepts learned** — concept promotion threshold of 0.99 is too strict. Almost no knowledge compounding happened.
3. **All 20 solves are single-primitive** — multi-step composition never fired for real tasks.
4. **Missing primitives** — grid partitioning, border detection, color replacement patterns absent from toolkit.

### v0.3 Changes

1. **Expanded toolkit**: 73 → 104 concepts
   - 13 new primitives: `get_top/bottom/left/right_half`, `get_border`, `get_interior`, `recolor_to_most_common`, `deduplicate_rows/cols`, `upscale_to_max`, `sort_rows_by_color_count`, `reverse_rows/cols`
   - 9 `fill_bg_N` operators (replace background with color N)
   - 9 `erase_N` operators (replace color N with background)

2. **Lowered concept promotion threshold**: 0.99 → 0.95
   - Near-miss solutions (≥0.95 score) are now promoted to reusable concepts
   - Critical for cumulative culture — imperfect solutions contain useful sub-patterns

3. **Richer seed generation** in explorer.py:
   - Feature-guided seeds for shrinking tasks (partitioning, dedup)
   - Half-size output detection → partition seeds
   - 2-step combo seeds (crop+mirror, fill+outline, etc.)
   - More novel programs per task (5 → 10)
   - More transfer programs per task (5 → 10)

4. **Tests expanded**: 155 → 180 tests (25 new tests for partitioning, pattern, and toolkit registration)

### Verification

| Metric | Value |
|--------|-------|
| Total tests | 180 |
| All passing | Yes |
| Sample task training | 9/10 (90%) |
| Sample task test | 8/10 (80%) |
| Toolkit size | 104 concepts |

### ARC-AGI-1 Benchmark Results (v0.3)

User ran the full benchmark with v0.3 changes:

| Metric | v0.2 | v0.3 | Change |
|--------|------|------|--------|
| **Exact solve** | 20/400 (5.0%) | **24/400 (6.0%)** | **+20%** |
| Test correct | 19/400 (4.8%) | 23/400 (5.8%) | +21% |
| Partial (>80%) | 185 | 200 | +15 tasks |
| Concepts learned | 3 | 24 | **8× more** |
| Toolkit growth | 73→76 | 104→118 | 14 concepts compounded |
| Speed | 0.40s/task | 0.27s/task | 33% faster |

**Key insight:** The lower concept promotion threshold (0.95 vs 0.99) unlocked real knowledge compounding. 24 concepts learned vs 3 — the cumulative culture mechanism is now working as intended.

---

## Session 5 — Conditional Logic & Task Decomposition (v0.4, March 2026)

### Prompt

> "both, but systematically, methodically and incrementally (but parallel progress is also good). Can you do as much work yourself as possible without asking me to run scripts that you can run yourself and verify?"

### Claude's Reasoning

Two independent features developed in parallel, both integrated and verified without user involvement.

### Work Done

**Feature 1: Conditional Logic (`concepts.py`, `primitives.py`, `synthesizer.py`)**

Extended the concept grammar to support if-then-else branching:
`Concept → ... | If(Predicate, ThenConcept, ElseConcept)`

- `ConditionalConcept` class added to `concepts.py` — a proper Concept subclass that evaluates a `Grid → bool` predicate and dispatches to the appropriate branch. Fits seamlessly into Program.execute() via the existing Concept.apply() interface.
- 6 new predicates added to `primitives.py`: `is_tall`, `is_wide`, `has_many_colors`, `is_small`, `is_large`, `has_background_majority` (plus existing 4: `is_symmetric_h/v`, `is_square`, `has_single_color`)
- All 10 predicates registered in `build_initial_toolkit()` as kind="predicate"
- `ProgramSynthesizer` updated with `conditional_rate=0.1` parameter, `_random_conditional()` method, conditional mutation type, and conditional programs in initial population generation
- Predicates excluded from single-primitive sweep in solver (they return bool, not Grid)

**Feature 2: Task Decomposition (`decompose.py`, `solver.py`)**

`DecompositionEngine` with 3 strategies, each taking a `synthesize_fn` callback:

1. **Color-channel decomposition** — Extracts binary mask per color, synthesizes independently per channel, merges results. Useful for multi-color tasks with independent channel transformations.
2. **Spatial quadrant decomposition** — Splits grid into TL/TR/BL/BR quadrants, synthesizes per quadrant, merges. Useful for tasks with spatially local patterns.
3. **Diff-focus decomposition** — Finds changed cells between in/out, focuses synthesis on the minimal bounding region. Useful for tasks with sparse changes.

Integrated into solver.py as a fallback after evolutionary synthesis if score < 0.99.

### Verification Results

| Metric | v0.3 | v0.4 |
|--------|------|------|
| Sample task training | 9/10 | **10/10** |
| Sample task test | 8/10 | **9/10** |
| Total tests | 180 | **216** |
| All passing | Yes | Yes |
| Toolkit size (initial) | 104 | **114** (10 predicates added) |
| Concepts learned (sample run) | 1 | **2** |

`invert_then_crop` now solved again (regression in v0.3 fixed by richer program space).

### Next Steps

1. **Run full ARC-AGI-1 benchmark** with v0.4 — measure impact of conditionals + decomposition
2. **Ablation studies** — Validate each pillar is necessary
3. **Improve code coverage** — Target 70%+ line coverage
4. **Analyze partial solves** — Find patterns in the 200 >80% tasks to guide next primitives
5. **Zork agent** — Text adventure environment modeling

---

## Session 6 — NumPy Acceleration & Multiprocessing (v0.5, March 2026)

### Prompt

> "I have an M1 Max Macbook Pro with 64GB RAM. Take full advantage of it, but don't crash it. I have installed homebrew and conda. I have installed numpy and numba. You should take advantage of them to avoid the python tax/overhead and run heavy computation natively to get enormous speedup. Feel free to take advantage of pip utilities and libraries, such standard dependencies are ok, but let's not depend on external services like LLMs, or other dependencies which cause us to deviate from first principles."

### Claude's Reasoning

The M1 Max has 10 CPU cores and 64GB RAM. The bottlenecks identified were:

1. **Scoring**: `structural_similarity()` was called millions of times per benchmark run (population × generations × examples × tasks). Pure-Python per-pixel loops were the hot path.
2. **Task serialisation**: Each ARC task was evaluated sequentially. With 400 tasks, 10 idle cores were wasted.
3. **Population batch overhead**: Scoring each program individually re-iterated over train examples per program, rather than amortizing the iteration.

**Strategy:**
- **NumPy vectorization** for `pixel_accuracy` and `structural_similarity`: replace Python loops with `np.array(dtype=uint8)` operations. Use `np.bincount` (O(10) lookup table) instead of `np.unique` + Python set ops for the ARC color palette (colors 0-9). Pure-Python fallback retained for portability.
- **`multiprocessing.Pool`** for parallel task evaluation: each worker gets an independent `FourPillarsSolver` instance (avoids shared-state race conditions). Tasks distributed round-robin. Per-worker random seeds ensure reproducibility.
- **`score_population_on_task()`** batch scorer: amortizes train-example iteration across the entire population per generation.
- **`--workers` CLI flag**: defaults to 0 (use all CPU cores). `workers=1` gives single-process path for debugging.

### Work Done

**`arc_agent/scorer.py`** — Full NumPy rewrite:
- `pixel_accuracy()`: uses `np.sum(p == e)` for vectorized comparison
- `structural_similarity()`: uses `np.bincount(ravel, minlength=10)` for color palette scoring; `np.count_nonzero()` for non-zero count; all sub-scores computed without Python loops
- Pure-Python fallback retained (importable without NumPy)
- New `score_population_on_task(programs, task)`: batch scores entire population

**`arc_agent/dataset.py`** — Full multiprocessing rewrite:
- `_solve_chunk(args)`: worker function, runs in a subprocess; creates independent solver, processes its chunk of tasks, returns results dict
- `evaluate_dataset(..., workers=0)`: round-robin distributes tasks across N workers; per-worker seeds = seed + i×1000; uses `Pool.map()` for parallel execution; merges results and prints summary with `workers_used`
- `workers=1` path: calls `_solve_chunk` directly in-process (no subprocess overhead, good for debugging)

**`arc_agent/evaluate.py`** — Added `--workers` CLI flag:
```
--workers: Parallel worker processes (0 = use all CPU cores)
```

**`arc_agent/synthesizer.py`** — Updated `evolve_generation()`:
- Now calls `score_population_on_task(population, task)` for batch scoring
- Removes the per-program loop in favour of the amortized batch call

**`arc_agent/__init__.py`** — Bumped to `v0.5.0`

**`tests/test_performance.py`** — 15 new tests:
- `TestPixelAccuracyNumPy`: NumPy path produces identical results to pure-Python reference
- `TestStructuralSimilarityNumPy`: same equivalence guarantee for composite scorer
- `TestScorePopulationOnTask`: batch scores match per-program scores exactly
- `TestParallelEvaluation`: `workers=1` and `workers=2` produce identical solve counts on trivial tasks; `--workers` flag accepted by CLI
- `TestNumPyAvailability`: sanity check that NumPy is importable

### Verification Results

| Metric | v0.4 | v0.5 |
|--------|------|------|
| Total tests | 216 | **231** |
| All passing | Yes | Yes |
| NumPy accelerated | No | **Yes** |
| Parallel evaluation | No | **Yes (all cores)** |
| Batch population scoring | No | **Yes** |
| `--workers` CLI flag | No | **Yes** |
| Expected speedup (scoring) | baseline | **~10-20× (NumPy vectorisation)** |
| Expected speedup (eval) | baseline | **~10× (10 cores, M1 Max)** |

### Next Steps

1. **Run full v0.4+v0.5 ARC-AGI-1 benchmark** to measure actual speedup:
   ```bash
   python -m arc_agent.evaluate --data-dir ARC-AGI/data/training --workers 10 --output results_v05.json
   ```
2. **Numba JIT** on flood-fill and object-extraction hot loops in `objects.py` for further speedup
3. **Ablation studies** — Validate each pillar is necessary
4. **ARC-AGI-2** — Test on second benchmark set
5. **Analyze partial solves** — 200 tasks with >80% accuracy — find patterns to guide new primitives
6. **Zork agent** — Text adventure environment modeling

---

## Session 7 — Cleanup, CPU Auto-detection & Reproducibility (v0.5.1, March 2026)

### Prompt

> I can do the commits and github pushes, but are you able to run and validate the code yourself?
> My macbook has 8 performance cores and 2 efficiency cores, not 10 performance cores. See if you can make the code automatically figure such stuff out. Use good defaults, but also keep everything configurable.
> Make the code work on any machine with the correct instructions for python, virtual environment, conda, pip packages etc.
> However, keep code simple. NumPy is standard everywhere, right? So is there any need to make 2 versions, with and without NumPy?
> Also, keep the code and tests deterministic and reproducible even when it runs in parallel. Ensure the entire repository is consistent and coherent, minimal but comprehensive, and most importantly understandable and navigable.

### Claude's Reasoning

Four issues to address:

1. **CPU topology auto-detection** — `multiprocessing.cpu_count()` returns *all* logical CPUs (P + E cores on Apple Silicon). On an M1 Max with 8P + 2E cores, using all 10 is suboptimal: the 2 efficiency cores are ~3× slower and become bottlenecks when tasks don't finish at the same time. The right default is performance cores only.
   - macOS: `sysctl hw.perflevel0.logicalcpu` gives the P-core count directly.
   - Linux ARM big.LITTLE: `/sys/devices/system/cpu/cpu*/cpu_capacity` — cores with max capacity (1024) are big cores.
   - x86/homogeneous: all CPUs are equal, use `os.cpu_count()`.

2. **NumPy fallback removal** — the dual-path code in scorer.py was correct but unnecessary complexity. NumPy is standard and should be a declared dependency. The pure-Python fallback doubled the code surface and the tests that verified their equivalence were testing code that should simply not exist. Removed both.

3. **Determinism in parallel mode** — results were already sorted by task_id before printing, and worker seeds were already derived as `seed + i*1000`. Verified that `evaluate_dataset` with the same (seed, workers) pair always produces the same output. Added `seed` to the summary dict so it's recorded in JSON output.

4. **Repository coherence** — `pyproject.toml` version was still `0.1.0` (not `0.5.0`). requirements.txt described NumPy as optional when it's required. No INSTALL.md existed. The `--workers` CLI help didn't show the machine-specific auto value inline.

### Work Done

**`arc_agent/cpu_utils.py`** — New module: runtime CPU topology detection
- `_detect_performance_cores()`: tries macOS `sysctl`, then Linux `/sys/.../cpu_capacity`, then `os.cpu_count()` fallback
- `default_workers()`: returns performance core count, cached at import time
- `describe_cpu()`: human-readable string ("8 performance + 2 efficiency cores (macOS Apple Silicon)")

**`arc_agent/scorer.py`** — Removed pure-Python fallback entirely
- `_NUMPY_AVAILABLE` flag gone
- `_to_array()` helper gone
- Both pure-Python branches in `pixel_accuracy` and `structural_similarity` gone
- Code is ~40% shorter with identical behaviour (NumPy was always used anyway)

**`arc_agent/dataset.py`** — Updated to use `cpu_utils`
- `default_workers()` replaces `multiprocessing.cpu_count()` for the default
- `describe_cpu()` printed in verbose header
- `seed` added to summary dict
- Docstring updated

**`arc_agent/evaluate.py`** — Updated CLI
- `--workers` help text shows machine-specific auto value inline: e.g. `Auto on this machine = 8 (8 performance + 2 efficiency cores (macOS Apple Silicon))`
- `sys.exit(main())` instead of `exit(main())`
- Cleaner docstring with all usage examples

**`pyproject.toml`** — Fixed: version `0.1.0` → `0.5.0`; NumPy declared as `dependencies` (required, not optional)

**`requirements.txt`** — NumPy moved from comment to required entry

**`INSTALL.md`** — New file: complete setup instructions for conda, venv, and pip; all CLI flags documented; reproducibility section explaining seeding strategy

**`tests/test_performance.py`** — Rewritten (was 15 tests, now 26 tests)
- Removed: `_NUMPY_AVAILABLE` import (gone), pure-Python equivalence tests (testing deleted code)
- Added: `TestPixelAccuracy` (9 tests covering edge cases), `TestStructuralSimilarity` (6 tests including randomized range check), `TestCpuUtils` (3 tests: positive, reasonable ceiling, string type)
- Kept: `TestScorePopulationOnTask` (4 tests), `TestParallelEvaluation` (4 tests, added seed-in-summary and sorted-keys checks)

### Verification

| Metric | Before | After |
|--------|--------|-------|
| Total tests | 231 | **242** |
| All passing | Yes | Yes |
| NumPy fallback code | ~80 lines | **0 lines** |
| `pyproject.toml` version | 0.1.0 | **0.5.0** |
| CPU auto-detection | `cpu_count()` total | **P-cores only** |
| INSTALL.md | missing | **added** |

### Next Steps

1. **Run full ARC-AGI-1 benchmark** with v0.5.1 — measure actual solve rate:
   ```bash
   python -m arc_agent.evaluate --data-dir ARC-AGI/data/training --output results_v051.json
   ```
2. **Numba JIT** on hot loops in `objects.py` (flood-fill) for further speedup
3. **Ablation studies** — validate each pillar is necessary
4. **ARC-AGI-2** evaluation
5. **Zork agent** — text adventure environment modeling

---

## Session 8 — New Primitives & Failure Mode Analysis (v0.9, March 2026)

### Prompt

> Let's continue making improvements.

### Claude's Reasoning

Ran systematic failure mode analysis on training set:
- 45 near-miss tasks with `identity` as best primitive — tasks need local edits, not global transforms
- 67% of hard tasks involve shrinking — output smaller than input
- Triple extension of pair near-misses: only 2 improved, 0 solved — not a useful direction

**Root cause**: The toolkit had 156 concepts covering geometric transforms and object recoloring well, but was missing primitives for shrink tasks (tile extraction, deduplication, pattern combination).

**Strategy**: Add 15 new primitives targeting the 67% shrink-task gap.

### Work Done

**`arc_agent/primitives.py`** — 15 new operators registered in `build_initial_toolkit()`:

| Primitive | Purpose |
|-----------|---------|
| `extract_repeating_tile` | Find smallest tile that tiles to reconstruct grid |
| `extract_top_left_block` / `extract_bottom_right_block` | Split by separator lines |
| `split_by_separator_and_overlay` / `split_by_separator_and_xor` | Split & merge blocks |
| `compress_rows` / `compress_cols` | Deduplicate non-consecutive duplicate rows/cols |
| `max_color_per_cell` / `min_color_per_cell` | Combine two stacked block grids |
| `extract_unique_block` | Find non-repeating sub-block in tiled grid |
| `flatten_to_row` / `flatten_to_column` | Unique colors as 1D grid |
| `count_objects_as_grid` | Count distinct objects as 1×1 grid |
| `mode_color_per_row` / `mode_color_per_col` | Per-row/col dominant color |

**`arc_agent/scorer.py`** — Fixed `_safe_to_np()` OverflowError:
- `count_objects_as_grid` can return integers >255
- Changed from `dtype=np.uint8` directly to `dtype=np.int32` first, then reject values outside [0,9]

**`arc_agent/synthesizer.py`** — Updated `ESSENTIAL_PAIR_CONCEPTS` to include all 8 new tile-extraction ops for pair exhaustion coverage.

**`arc_agent/explorer.py`** — Updated `generate_seed_programs()`:
- Added all new v0.9 primitives to heuristic seed lists
- Added 10 new 2-step combo seeds for tile extraction patterns

**`tests/test_primitives.py`** — Added `TestTileExtraction` (14 tests) and `TestNewToolkitSizeV09` (2 tests).

### Bugs Fixed

1. **`OverflowError: Python integer 274 out of bounds for uint8`** — `count_objects_as_grid` returned values >255, crashing `_safe_to_np()`. Fixed by using `np.int32` intermediate array.
2. **`test_split_separator_xor` wrong expected value** — Fixed test case to use non-overlapping colors.

### Results

| Metric | Before (v0.8) | After (v0.9) |
|--------|---------------|--------------|
| Toolkit size | 156 | **171** |
| Deterministic solves | 44 | **50** |
| Training solve rate | 55/400 (13.8%) | **66/400 (16.5%)** |
| Tests passing | 280 | **301** |

New single-primitive solves: `compress_cols` (2dee498d), `extract_repeating_tile` (7b7f7511), `split_sep_overlay` (e98196ab).

---

## Session 9 — Culture Persistence: Cross-Run Transfer (v0.10, March 2026)

### Prompt

> I am running the jobs, but shouldn't the evaluation run be using some model or something from the output of the training run?
> We should be learning more primitives and higher level structures from the training set and be applying them to the evaluation set. One way to do that is to store the cumulative culture from the training set as a configurable 'model' file or a 'learning' file or something, and reuse that by passing the file as a flag to the evaluation run.

### Claude's Reasoning

The training and evaluation runs were completely independent — zero cross-run transfer. Training discovered ~66 programs that solve tasks; these were discarded. The evaluation run started from scratch each time.

**Solution**: Serialize the culture (learned concepts + solved programs + task features) as a JSON file after training. Load it into each worker's solver before solving eval tasks.

**Key challenge**: Python closures (compositions) can't be serialized. Solution: save step-name sequences (e.g., `["mirror_h", "crop_nonzero"]`) and reconstruct the composition chain at load time by looking up each name in the base toolkit.

### Work Done

**`arc_agent/culture.py`** — New module (full culture persistence):
- `save_culture(toolkit, archive, path)` — serializes `learned_*` concepts as step-name lists and all solved programs to JSON
- `load_culture(toolkit, path, archive)` — rebuilds composed concepts from step names, seeds `toolkit.programs`
- `_extract_step_names(concept)` — recursively flattens concept children to primitive names
- `_rebuild_concept(step_names, toolkit, name)` — reconstructs chained closure from step names

**`arc_agent/dataset.py`** — Updated worker and evaluation harness:
- `_solve_one` accepts `culture_path` as 6th arg, loads culture into each fresh solver
- Workers return `_learned_concepts`, `_solved_programs`, `_task_features` for aggregation
- `_aggregate_and_save_culture()` merges results from all parallel workers, deduplicates
- `evaluate_dataset()` accepts `load_culture_path` and `save_culture_path` params

**`arc_agent/evaluate.py`** — Added CLI flags:
```
--save-culture <path>   # Save culture after training run
--load-culture <path>   # Load culture before evaluation run
```

**`arc_agent/explorer.py`** — Updated `generate_seed_programs()`:
- `toolkit.programs` (loaded culture programs) are now seeded first in evolution

**`tests/test_culture.py`** — New file with 7 tests covering: step extraction, save/load roundtrip, missing primitive handling, archive feature loading.

### Two-Phase Pipeline

```bash
# Phase 1: Training run — discover programs, save culture
python -m arc_agent.evaluate --data-dir ARC-AGI/data/training --save-culture culture.json

# Phase 2: Evaluation run — load culture, solve with transfer
python -m arc_agent.evaluate --data-dir ARC-AGI/data/evaluation --load-culture culture.json
```

### Results

Training: 66/400 (16.5%), Evaluation: 23/400 (5.8%)

Culture infrastructure works correctly (saves/loads without errors). Eval score unchanged from before — culture programs were only used as evolutionary seeds, not tested deterministically. The gap to fix: loaded programs needed a direct test against each eval task before falling through to evolution.

---

## Session 10 — Deterministic Culture Transfer & Within-Run Compounding (v0.11, March 2026)

### Prompt

> Just checking, I hope during evaluation, you are still using compounding by applying lessons from earlier tasks to the later tasks.
> And same during training as well. i.e. leverage compounding as much as possible while being deterministic and reproducible.

### Claude's Reasoning

Two critical gaps identified:

1. **No deterministic culture transfer**: Loaded programs were only added as evolutionary seeds (probabilistic), not tested exhaustively against each task before evolution. This meant training programs with score=1.0 on training tasks could be rejected by evolution in favor of random mutations.

2. **No within-run compounding**: Each task in `evaluate_dataset` got a fresh `FourPillarsSolver` instance. Concepts discovered on task #1 were invisible to task #2. The `solve_batch()` method in `solver.py` did compound, but `evaluate_dataset()` never used it.

**Fixes**:
- Add `_try_culture_programs()` step in solver — try ALL loaded culture programs deterministically before pair exhaustion
- Add `_solve_sequential_compounding()` in dataset.py — single-process path uses ONE shared solver so knowledge compounds across tasks
- Refactor `_solve_one` to use shared `_collect_result()` helper (DRY)

### Work Done

**`arc_agent/solver.py`** — Two additions:

```python
# Step 3.2: Deterministic culture transfer
def _try_culture_programs(self, task, cache=None):
    """Try all programs in toolkit.programs directly. Returns best or None."""
    for prog in self.toolkit.programs:
        score = cache.score_program(prog)
        if score >= 0.99:
            return prog  # early exit
    return best_program  # best found

# New method label in solve_task: "culture_transfer"
```

Order in `solve_task()`:
1. Single primitives
2. **Culture programs (NEW: deterministic, all loaded programs tested)**
3. Pair exhaustion
4. Evolution (seeded with best culture + pair results)

**`arc_agent/dataset.py`** — Compounding path:

```python
def _solve_sequential_compounding(worker_args, tracker, ...):
    """One shared solver for all tasks — knowledge compounds in sorted order."""
    solver = FourPillarsSolver(...)
    load_culture(solver.toolkit, culture_path, solver.archive)
    for task_id, task, ... in worker_args:
        random.seed(seed)  # Per-task seed for reproducibility
        result = solver.solve_task(task, task_id)
        ...
```

The `n_workers == 1` path now calls `_solve_sequential_compounding` instead of fresh-solver-per-task `_solve_one`.

**`tests/test_culture.py`** — Added 3 new tests:
- `test_try_culture_programs_empty` — returns None when no programs loaded
- `test_try_culture_programs_finds_solver` — scores and returns matching program
- `test_culture_transfer_method_used` — result method is "culture_transfer"

### Design Decisions

- **Parallel workers still get independent solvers**: True cross-worker compounding within a parallel run requires shared memory or message passing between processes — too complex and would break determinism. The parallel path retains independent workers (each loading the pre-trained culture file).
- **Sequential path has full compounding**: `--workers 1` gives maximum within-run knowledge reuse. Use this when you want the richest compounding.
- **Per-task seed preserved**: Even in the shared-solver path, each task's evolution is seeded with `seed + i * 1000` for full reproducibility.

### Results

| Metric | Before (v0.10) | After (v0.11) |
|--------|----------------|---------------|
| Tests passing | 301 | **304** |
| Culture transfer method | seeding only | **deterministic + seeding** |
| Within-run compounding | none (parallel) | **full (--workers 1)** |
| `_collect_result` helper | duplicated in worker | **shared helper** |

**To run with maximum compounding** (single process, full knowledge reuse):
```bash
# Training with compounding:
python -m arc_agent.evaluate --data-dir ARC-AGI/data/training --workers 1 --save-culture culture.json

# Evaluation with culture transfer + compounding:
python -m arc_agent.evaluate --data-dir ARC-AGI/data/evaluation --workers 1 --load-culture culture.json
```

**Full speed on M1 Max** (parallel, still with culture transfer):
```bash
python -m arc_agent.evaluate --data-dir ARC-AGI/data/training --save-culture culture.json
python -m arc_agent.evaluate --data-dir ARC-AGI/data/evaluation --load-culture culture.json
```

---

## Session 11 — New Primitives: Tile Completion, Symmetry Masking, Lane Spreading (v0.13, March 2026)

### Prompt

> [Context continuation — user said "yes" to implementing Step 3: object-conditional primitive layer from the prior analysis]

### Claude's Reasoning

From the systematic analysis in the previous session (see Session 10's findings), 130/400 tasks require local/conditional edits that no global chain can express. The near-miss analysis (30 tasks scoring 0.90–0.99 with pair exhaustion) identified specific recurring ARC patterns not covered by the existing 171-concept toolkit:

1. **Tile completion** (task 29ec7d0e): A repeating tile pattern with some cells zeroed out — reconstruct the full tile and fill.
2. **Symmetry masking** (task 3631a71a): Symmetric grid with a rectangular block of a "mask" color covering cells — recover from symmetry counterparts.
3. **Nearest-border recoloring** (task 2204b7a8): Isolated pixels get assigned the color of their nearest border stripe.
4. **Extend-to-border** (task 2c608aff): Non-zero cells extend in one direction to the grid boundary.
5. **Lane spreading** (task 06df4c85, 1f642eb9, 36d67576): Grid divided by separator lines; colored cells spread within their lane row/column.

### Work Done

**`arc_agent/primitives.py`** — Added 7 new operator functions + registered in `build_initial_toolkit()`:

```python
# Tile Completion
def fill_tile_pattern(grid):
    """Infer repeating tile from visible non-zero cells (with 90% consistency check), fill all."""

# Symmetry Masking
def fill_by_symmetry(grid):
    """Find rectangular block of mask color, recover from 180°/H/V symmetry counterpart."""

# Nearest Border Recoloring
def recolor_by_nearest_border(grid):
    """Find border stripes (uniform rows/cols), recolor isolated noise pixels to nearest border color."""

# Extend to Border
def extend_to_border_h(grid):
    """Each non-zero cell extends left+right to fill entire row."""
def extend_to_border_v(grid):
    """Each non-zero cell extends up+down to fill entire column."""

# Lane Spreading
def spread_in_lanes_h(grid):
    """Within separator-defined lanes, spread colored cells to fill entire lane row."""
def spread_in_lanes_v(grid):
    """Transpose of spread_in_lanes_h."""
```

**`arc_agent/synthesizer.py`** — Added `fill_by_symmetry`, `fill_tile_pattern`, `spread_in_lanes_h`, `spread_in_lanes_v` to `ESSENTIAL_PAIR_CONCEPTS` (they score well as second steps).

**`tests/test_primitives.py`** — Added 20 new unit tests across 6 test classes:
- `TestFillTilePattern` (4 tests)
- `TestFillBySymmetry` (3 tests)
- `TestRecolorByNearestBorder` (3 tests)
- `TestExtendToBorder` (4 tests)
- `TestSpreadInLanes` (4 tests)
- `TestV13ToolkitContents` (2 tests)

### Design Decisions

- **`fill_tile_pattern` 90% consistency gate**: First version accepted any plurality tile (≥50% positions resolved) — this caused a degenerate 1×1 tile (just "most common value") to always win. Added a consistency check: ≥90% of non-zero cells must agree with the inferred tile. This raised the score for task 29ec7d0e from 0.50 → 0.94.

- **ESSENTIAL_PAIR_CONCEPTS discipline**: Only added primitives that score LOW individually but are useful as SECOND steps (fill_by_symmetry, fill_tile_pattern, spread_in_lanes_h/v). High-scoring new primitives (recolor_by_nearest_border, extend_to_border_h/v) are already in top_k=20 for their target tasks and don't need to be in ESSENTIAL.

- **Speed regression avoidance**: Tested that wall-clock rate stays ≥1.5 tasks/s at 4 workers. Adding too many ESSENTIAL concepts explodes pair search: (top_k + |essential|)² pairs × n_train examples.

### Results

| Metric | Before (v0.12) | After (v0.13) |
|--------|----------------|---------------|
| Tests passing | 309 | **329** |
| Toolkit size | 171 | **178** |
| Pair-solve rate (100 tasks) | 11/100 | **13/100** |
| Exact solves (40 tasks, seed 42) | 5/40 (12.5%) | **6/40 (15.0%)** |
| Mean score (40 tasks) | 0.827 | **0.809** (mean slightly down but median up) |
| Wall-clock rate | ~1.1 tasks/s | **~1.58 tasks/s** |
| New pair-solves | — | 2204b7a8 (recolor_by_nearest_border), 3c9b0459 (rotate_180 → fill_tile_pattern) |

The 7 new primitives added 2 deterministic pair-solves, improve the evolution seed quality for ~10 near-miss tasks, and maintain throughput.

---

## Session 12 — v0.14: 27 New Primitives from Near-Miss Analysis

**Date:** 2026-03-07

**Starting state:** v0.13 committed (178 concepts, 329 tests). Evaluation at ~16.5% on training set.

**Analysis performed:**
- Fixed near-miss analysis scripts (`.execute()` not `.apply()`)
- Scanned first 200 tasks: found 87 near-misses (single-prim score 0.85-0.99)
- Categorized residual errors: **54/87 need zeros→color fill**, 28 need recoloring, 5 need erasing
- Ran pair-search on first 100 tasks: 8 solves previously → checking all pairs
- Identified highest-ROI candidates by testing new functions against 400 tasks

**New primitives added (27 total):**
- `connect_pixels_to_rect`: BFS connect isolated pixels to nearest rectangle with a line segment (45 near-misses as single)
- `gravity_toward_color`: Pack scattered dots adjacent to solid-color bands (61 near-misses, 9 pair-solves)
- `fill_holes_in_objects`: Fill enclosed background regions with surrounding object color (50 near-misses, 8 pair-solves)
- `recolor_2nd_to_3rd`: Replace 2nd most common color with 3rd most common (43 near-misses)
- `recolor_least_to_2nd_least`: Collapse least-common into 2nd-least
- `swap_most_and_2nd_color`, `swap_largest_smallest_obj_color`: Color swap ops
- 10x pairwise color swaps: `swap_colors_12` through `swap_colors_45`
- `complete_pattern_4way`: Enforce D4 symmetry (fill all 4 symmetric positions)
- `fill_bg_from_border`: Fill bg with most common border color
- `keep_unique_rows`, `keep_unique_cols`: Remove duplicate rows/cols
- `rotate_colors_up`, `rotate_colors_down`: Cycle all non-bg colors
- `extend_nonzero_fill_row`, `extend_nonzero_fill_col`: If row has exactly one non-bg color, fill whole row
- `color_by_row_position`, `color_by_col_position`: Map position to color

**ESSENTIAL_PAIR_CONCEPTS additions (v14):** `gravity_toward_color`, `fill_holes_in_objects`, `connect_pixels_to_rect`, `recolor_2nd_to_3rd`, `extend_nonzero_fill_row`, `extend_nonzero_fill_col`, `complete_pattern_4way`

**Results:**
- Tests: 349 passing (was 329)
- Toolkit size: 205 concepts (was 178)
- 80-task benchmark: **12/80 (15.0%)** exact solves (was 11/80 = 13.8%)
- 200-task benchmark: **35/200 (17.5%)** exact solves
- New solves confirmed: 22eb0ac0 (extend_nonzero_fill_row → x2), 4093f84a (gravity_toward_color)

**Primitives dropped (scored 0 on all 400 tasks):** `rotate_colors_up`, `rotate_colors_down`, `color_by_row_position`, `color_by_col_position` (kept in code but removed from ESSENTIAL)
