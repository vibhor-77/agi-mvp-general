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

### Next Steps

1. **Re-run full ARC-AGI-1 benchmark** with v0.3 changes to measure improvement
2. **Conditional logic** — If-then-else branching in programs
3. **Task decomposition** — Fractal problem-solving for hard tasks
4. **Improve coverage** — Target 70%+ line coverage
5. **Ablation studies** — Validate each pillar is necessary
