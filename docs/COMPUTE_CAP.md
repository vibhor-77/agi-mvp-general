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
| `8M` | ~19 | ~2 min | ~4 min | ~10K | 2.4M |
| `50M` | ~25 | ~11 min | ~22 min | ~62K | 14.8M |
| `100M` | ~29 | ~22 min | ~43 min | ~125K | 28.9M |
| **`200M`** | **~34** | **~29 min** | **~58 min** | **~250K** | **38.8M** |
| `400M` | ~35 | ~55 min | ~110 min | ~500K | 44.1M |
| `0` (unlimited) | ~35 | ~2.5 hrs | ~5 hrs | unlimited | 45.8M |

**Why 200M is the default:** It recovers 97% of known solves (34/35) while using only 85% of the total compute and running in ~30 minutes on a modern machine. The single lost task (`903d1b4a`) requires 322M compute due to deep near-miss refinement on a 2560-cell grid. Going from 200M to unlimited gains just 1 more solve but costs 4x the time.

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

The simulation uses actual per-task eval counts and grid sizes from the uncapped benchmark run (`results/20260309_062523_evaluation.jsonl`, 400 tasks, 35 exact solves). For each hypothetical cap, we compute the effective budget per task using the same formula the solver uses, then check whether each solved task's actual eval count fits within that budget.

Wall-clock times are estimated by scaling from the measured 8M run (2.4M evals in 217 seconds, 4 workers) proportionally to total evals, with a 2x speedup factor for 8 workers (validated against the uncapped 8-worker run at 8,927 seconds). Actual times depend on hardware, system load, and task mix.

## Quick Reference

```bash
# Default: Pareto-optimal balance (~34 solves, ~30 min)
python benchmark.py --pipeline

# Quick iteration during development (~19 solves, ~2 min)
python benchmark.py --pipeline --compute-cap 8M

# Medium: good for CI/nightly (~25 solves, ~11 min)
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
