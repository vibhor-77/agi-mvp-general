# Compute Cap Tuning Guide

The `--compute-cap` flag controls how many evaluations each task gets. This is the single most impactful parameter for controlling the tradeoff between solve rate and runtime.

## How It Works

Each task gets a per-task eval budget calculated as:

```
budget = min(compute_cap / cells, compute_cap / 800)
```

where `cells` is the task's average grid cell count and 800 is the median ARC grid size. This normalizes compute: large grids (expensive per eval) get fewer evals, while small grids get more. The floor is 500 evals regardless of cap.

## Pareto Analysis

We ran a simulation using the best uncapped benchmark (35 exact eval solves, 45.8M total evals, ~2.5 hours). For each cap level, we calculated which of the 35 solved tasks would still have enough budget to solve, plus tasks recovered by DSL Phase 0 shortcuts (which run before the budget check).

| `--compute-cap` | Eval solves | Est. wall-time (8 workers) | Est. time (4 workers) | Ceiling evals/task | Total evals |
|:---:|:---:|:---:|:---:|:---:|:---:|
| `8M` | ~19 | ~3 min | ~6 min | ~10K | 2.4M |
| `50M` | ~25 | ~18 min | ~35 min | ~62K | 14.8M |
| `100M` | ~29 | ~35 min | ~70 min | ~125K | 28.9M |
| **`200M`** | **~33** | **~48 min** | **~95 min** | **~250K** | **39.0M** |
| `400M` | ~35 | ~90 min | ~3 hrs | ~500K | 44.1M |
| `0` (unlimited) | ~35 | ~2.5 hrs | ~5 hrs | unlimited | 45.8M |

**Validated on Apple M3 Pro (8 workers, March 2026):** The 200M default produced 33 eval solves in 48 minutes wall-clock, using 39.0M total evals. Budget was exceeded on 145/400 tasks. Solve count varies by ±1-2 across runs due to search order nondeterminism and culture quality.

**Why 200M is the default:** It recovers ~94-97% of known solves (33-34/35) while using only 85% of the total compute. The remaining 1-2 tasks require >300M compute each due to deep near-miss refinement on large grids. Going from 200M to unlimited gains only 1-2 more solves but costs ~3x the time.

The curve shows a classic diminishing-returns Pareto front: most tasks are cheap to solve (the first 19 solves cost only 2.4M evals total), but a long tail of expensive tasks drives the total compute. 200M sits at the "knee" where the marginal cost per solve spikes.

## DSL Shortcuts: Free Solves

Four eval tasks are solved by DSL Phase 0 shortcuts that run before the budget is checked:

| Task | Method | Why it's "free" |
|------|--------|-----------------|
| `84f2aca1` | 8-neighbor rule | Learned in O(cells) time by `_try_neighbor_rule_8_shortcut` |
| `e0fb7511` | 4-neighbor rule | Learned in O(cells) time by `_try_neighbor_rule_shortcut` |
| `66f2d22f` | halves + colormap | `or_halves_h` + `apply_color_map` detected by `_try_halves_colormap_shortcut` |
| `e345f17b` | halves + colormap | Same shortcut, different color map |

These tasks appear in the "+DSL shorts" column and are counted in the solve totals above. They were previously solvable only with full bottom-up DSL enumeration (requiring >50M compute each), but the shortcuts detect the pattern in milliseconds.

## Methodology

The Pareto table was derived in two steps:

1. **Simulation:** Using actual per-task eval counts and grid sizes from the best uncapped benchmark (`results/20260309_062523_evaluation.jsonl`, 400 tasks, 35 exact solves), we compute the effective budget per task for each hypothetical cap using the formula `min(compute_cap / cells, compute_cap / 800)`, then check whether each solved task's actual eval count fits within that budget.

2. **Validation:** The 200M default was validated with a real benchmark run on an Apple M3 Pro (`results/20260310_101307_evaluation.jsonl`): 33 exact solves, 39.0M total evals, 48 min wall-clock, 5h38m CPU time, 145/400 tasks budget-exceeded. The 1-2 solve difference from the simulation (which predicted ~34) is expected due to culture quality variance and search nondeterminism.

Wall-clock times scale approximately linearly with total evals but include a constant overhead for task setup, culture loading, and result serialization. Larger-budget tasks also tend to have higher per-eval cost (bigger grids), so time does not scale perfectly linearly. The estimates above are calibrated against the validated 200M run and should be treated as ±30% depending on hardware and system load.

## Quick Reference

```bash
# Default: Pareto-optimal balance (~33 solves, ~48 min)
python benchmark.py --pipeline

# Quick iteration during development (~19 solves, ~3 min)
python benchmark.py --pipeline --compute-cap 8M

# Medium: good for CI/nightly (~25 solves, ~18 min)
python benchmark.py --pipeline --compute-cap 50M

# Maximum solves, no time limit (~35 solves, ~2.5 hrs)
python benchmark.py --pipeline --contest

# Show the reference table
python benchmark.py --help-caps
```

## Choosing a Cap for Your Use Case

**Development/debugging:** `8M` — fast feedback, catches regressions in deterministic search.

**CI/nightly builds:** `50M` — good balance, runs in ~11 minutes, catches most regressions.

**Pre-release validation:** `200M` (default) — near-maximum solves, reasonable time.

**Competition/publication:** `--contest` (unlimited) — squeeze out every last solve regardless of time.

**Hardware notes:** Times above assume a modern machine (Apple M1/M2/M3, 8-core Intel/AMD, or similar). The solver is CPU-bound and scales well with worker count up to the number of physical cores. Using more workers than physical cores provides no benefit due to GIL-free multiprocessing.
