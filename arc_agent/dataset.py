"""
ARC-AGI Dataset Loader and Parallel Evaluation Harness

Loads the official ARC-AGI dataset (JSON files) and runs the Four Pillars
solver on them, collecting full metrics.

Dataset structure (ARC-AGI-1):
  data/training/    — 400 tasks for development
  data/evaluation/  — 400 tasks for held-out evaluation

Each task is a JSON file:
  {"train": [{"input": [[int]], "output": [[int]]}],
   "test":  [{"input": [[int]], "output": [[int]]}]}

Grids are rectangular matrices of ints 0-9, sizes 1×1 to 30×30.

Parallel evaluation:
  Tasks are distributed across worker processes using imap_unordered,
  so results stream back as workers finish — you see live output and can
  Ctrl-C to abort at any time with a clean summary of work done so far.

  Worker seeds are derived deterministically from (global_seed + worker_index
  * 1000) so results are reproducible for any fixed (seed, workers) pair.
  Results are sorted by task_id in the final JSON output.

  Default worker count = performance core count (see cpu_utils.py).
  On Apple Silicon M-series this is the P-core count, not the total.
"""
from __future__ import annotations

import json
import os
import random
import statistics
import time
from typing import Iterator

import multiprocessing

from .solver import FourPillarsSolver
from .scorer import validate_on_test, validate_candidates_on_test
from .cpu_utils import default_workers, describe_cpu
from .culture import save_culture, load_culture


# ── Dataset loading ────────────────────────────────────────────────────────

def load_task(path: str) -> dict:
    """Load a single ARC-AGI task from a JSON file."""
    with open(path) as f:
        return json.load(f)


def load_dataset(directory: str) -> dict[str, dict]:
    """Load all ARC-AGI tasks from a directory.

    Returns a dict mapping task_id → task, sorted alphabetically by task_id
    so that --limit N always picks the same N tasks regardless of OS.
    """
    tasks: dict[str, dict] = {}
    if not os.path.isdir(directory):
        return tasks

    for filename in sorted(os.listdir(directory)):
        if not filename.endswith(".json"):
            continue
        task_id = filename[:-5]
        try:
            tasks[task_id] = load_task(os.path.join(directory, filename))
        except (json.JSONDecodeError, OSError) as exc:
            print(f"  Warning: skipping {filename}: {exc}")

    return tasks


# ── Worker ─────────────────────────────────────────────────────────────────

def _collect_result(solver: "FourPillarsSolver", result: dict, task_id: str,
                     task: dict, seed: int, mode: str = "train",
                     top_k: int = 3) -> dict:
    """Package a per-task result with culture data for aggregation.

    Shared by the single-process (shared-solver) path and the per-process
    worker path. Both compute the same result dict so that _aggregate_and_save_culture
    always sees the same format.

    When multiple candidates are available, validates ALL of them (up to top_k)
    against the test output. If ANY candidate passes, test_passed = True.
    This is the core of the multiple candidate submission feature.
    """
    test_passed = False
    test_score  = 0.0

    # In infer mode, we NEVER look at test output — that's the whole point.
    # In train/eval modes, validate against test for honest reporting.
    if mode != "infer":
        # Rebuild candidate programs from step names for multi-candidate validation
        candidate_programs = _rebuild_candidate_programs(
            result.get("candidates", []), solver.toolkit
        )

        if candidate_programs:
            # Test ALL candidates — if any passes, the task is test-confirmed
            test_passed, test_score = validate_candidates_on_test(
                candidate_programs, task, top_k=top_k
            )
        else:
            # Fallback: no candidates, try the single best program
            programs = solver.archive.task_solutions.get(task_id, [])
            if programs:
                test_passed, test_score = validate_on_test(programs[0], task)

    # Collect learned concepts and programs for culture saving
    learned_concepts = []
    for name, concept in solver.toolkit.concepts.items():
        if name.startswith("learned_"):
            from .culture import _extract_step_names
            step_names = _extract_step_names(concept)
            learned_concepts.append({
                "name": name,
                "steps": step_names,
                "kind": concept.kind,
                "usage_count": concept.usage_count,
                "success_count": concept.success_count,
            })

    solved_programs = []
    for tid, programs in solver.archive.task_solutions.items():
        for prog in programs:
            step_names = [s.name for s in prog.steps]
            all_reconstructable = all(
                sn in solver.toolkit.concepts for sn in step_names
            )
            if all_reconstructable and step_names:
                solved_programs.append({
                    "task_id": tid,
                    "steps": step_names,
                    "fitness": prog.fitness,
                    "name": prog.name,
                })

    return {
        "task_id":        task_id,
        "solved":         result["solved"],
        "score":          result["score"],
        "test_passed":    test_passed,
        "test_score":     test_score,
        "program":        result["program"],
        "program_length": result["program_length"],
        "method":         result["method"],
        "time_seconds":   result["time_seconds"],
        "toolkit_size":   result["toolkit_size"],
        "n_candidates":   result.get("n_candidates", 0),
        "candidates":     result.get("candidates", []),
        "worker_seed":    seed,
        # Culture data for cross-worker aggregation
        "_learned_concepts": learned_concepts,
        "_solved_programs":  solved_programs,
        "_task_features":    solver.archive.task_features,
    }


def _rebuild_candidate_programs(
    candidate_dicts: list[dict],
    toolkit: "Toolkit",
) -> list:
    """Rebuild Program objects from serialized candidate dicts.

    Each candidate dict has 'steps': list of step names. We look up each
    step in the toolkit's concepts and reconstruct the Program.
    Candidates with missing steps (e.g., task-specific learned concepts
    that were cleaned up) are silently skipped.

    Returns a list of Program objects in the same order as candidates.
    """
    from .concepts import Program

    programs = []
    for cand in candidate_dicts:
        steps = cand.get("steps", [])
        if not steps:
            continue
        concepts = []
        all_found = True
        for step_name in steps:
            if step_name in toolkit.concepts:
                concepts.append(toolkit.concepts[step_name])
            else:
                all_found = False
                break
        if all_found and concepts:
            programs.append(Program(concepts))
    return programs


def _avg_cells(task: dict) -> int:
    """Average cell count across all training input grids."""
    grids = [ex["input"] for ex in task.get("train", [])]
    if not grids:
        return 1
    total = sum(len(g) * len(g[0]) for g in grids if g and g[0])
    return max(1, total // len(grids))


def _solve_one(args: tuple) -> dict:
    """Worker function: solve a single task in a subprocess.

    Using one-task-at-a-time (via imap_unordered) instead of chunked
    pool.map lets results stream back as soon as each task finishes,
    enabling live progress display and clean Ctrl-C abort.

    Args:
        args: (task_id, task_dict, population_size, max_generations, seed,
               culture_path, mode)
              culture_path can be "" if no pre-trained culture to load.
              mode is "train" or "eval".
              The solver is created fresh per worker call so there is no
              shared mutable state between workers.

    Returns:
        Dict with task_id and all per-task metrics, plus a worker_seed
        for reproducibility bookkeeping and culture data for aggregation.
    """
    # Support variable-length args for backward compatibility.
    # Current format: (task_id, task, pop, gens, seed, culture, mode,
    #                  top_k, compute_cap, time_limit)
    task_id = args[0]
    task = args[1]
    population_size = args[2]
    max_generations = args[3]
    seed = args[4]
    culture_path = args[5]
    mode = args[6]
    top_k = args[7] if len(args) > 7 else 3
    compute_cap = args[8] if len(args) > 8 else 8_000_000
    time_limit = args[9] if len(args) > 9 else 0.0

    # Cell-normalized compute cap with proportional per-task ceiling.
    #
    # Formula: min(compute_cap / cells, max_evals)
    #   where max_evals = compute_cap / DEFAULT_CELLS
    #   DEFAULT_CELLS = 800 (median grid size in ARC training)
    #
    # At the default 8M cap: max_evals = 8M/800 = 10K — the natural
    # saturation point where deterministic search (~1-3K) + evolution
    # (~7-9K) exhaust useful work.  This prevents small-grid tasks
    # from burning time in low-ROI triples search.
    #
    # At higher caps (e.g. 400M for contest mode), the ceiling scales
    # proportionally: 400M/800 = 500K, allowing deep search when the
    # user explicitly requests more compute.
    DEFAULT_CELLS = 800
    if compute_cap > 0:
        cells = _avg_cells(task)
        max_evals = compute_cap // DEFAULT_CELLS
        evals_budget = min(max(compute_cap // cells, 500), max(max_evals, 500))
    else:
        evals_budget = 10_000_000  # Effectively unlimited

    random.seed(seed)

    solver = FourPillarsSolver(
        population_size=population_size,
        max_generations=max_generations,
        verbose=False,
    )

    # Load pre-trained culture if provided
    if culture_path:
        try:
            load_culture(solver.toolkit, culture_path, solver.archive)
        except Exception:
            pass  # Gracefully degrade — run without culture

    result = solver.solve_task(task, task_id, mode=mode,
                               evals_budget=evals_budget,
                               time_limit=time_limit)
    return _collect_result(solver, result, task_id, task, seed, mode=mode,
                           top_k=top_k)


# ── Progress display ────────────────────────────────────────────────────────

class _ProgressTracker:
    """Accumulates per-task results and prints a live status line.

    Designed to be called from the main process as results stream in
    from imap_unordered, so it sees completed tasks one at a time.

    Output columns:
      [idx/total] symbol task_id   score=X.XXX  Xs  method  |  running stats
    """

    def __init__(self, n_tasks: int, n_workers: int, start_time: float,
                 mode: str = "train"):
        self.n_tasks    = n_tasks
        self.n_workers  = n_workers
        self.start_time = start_time
        self.mode       = mode

        # Accumulators
        self.done       = 0
        self.solved_exact = 0  # pixel-perfect on train AND test (golden metric)
        self.pp_train   = 0    # pixel-perfect on train only
        self.partial    = 0    # score > 0.80
        self.test_ok    = 0
        self.flukes     = 0
        self.overfits   = 0
        self.scores:    list[float] = []
        self.times:     list[float] = []

    def update(self, r: dict) -> None:
        """Record a completed task result and print one progress line."""
        self.done += 1
        self.scores.append(r["score"])
        self.times.append(r["time_seconds"])

        if r["solved"] and r.get("test_passed"):
            self.solved_exact += 1
        if r["solved"]:
            self.pp_train += 1
        if r["solved"] and not r.get("test_passed"):
            self.overfits += 1
        if not r["solved"] and r.get("test_passed"):
            self.flukes += 1
        if not r["solved"] and r["score"] > 0.80:
            self.partial += 1
        if r.get("test_passed"):
            self.test_ok += 1

        self._print_task_line(r)

        # Print a rolling summary every 10 tasks (or at the very end).
        if self.done % 10 == 0 or self.done == self.n_tasks:
            self._print_rolling_summary()

    def _print_task_line(self, r: dict) -> None:
        """Print one line per completed task with all relevant metrics."""
        if self.mode == "infer":
            # No test info — show train-only status
            status = "✓" if r["solved"] else ("~" if r["score"] > 0.80 else "✗")
        else:
            # ✓=solved exact, ◇=overfit, △=fluke, ~=partial, ✗=low
            if r["solved"] and r.get("test_passed"):
                status = "✓"  # solved exact
            elif r["solved"] and not r.get("test_passed"):
                status = "◇"  # overfit
            elif not r["solved"] and r.get("test_passed"):
                status = "△"  # fluke
            elif r["score"] > 0.80:
                status = "~"  # partial
            else:
                status = "✗"  # low score
        elapsed  = time.time() - self.start_time
        # Projected finish time
        rate     = self.done / max(elapsed, 0.001)          # tasks/s wall-clock
        remaining_tasks = self.n_tasks - self.done
        eta_sec  = remaining_tasks / rate if rate > 0 else 0
        eta_str  = _fmt_duration(eta_sec)

        # Show test status for solved and near-miss programs alike
        # In infer mode, test was not checked, so skip test_tag
        if self.mode != "infer" and (r["solved"] or r.get("test_score", 0) > 0):
            test_tag = f" test={'✓' if r['test_passed'] else '✗'}"
        else:
            test_tag = ""
        method   = r.get("method", "")[:12]

        print(
            f"  [{self.done:3d}/{self.n_tasks}] {status} "
            f"{r['task_id'][:16]:16s} "
            f"score={r['score']:.3f}{test_tag:7s} "
            f"{r['time_seconds']:5.2f}s  "
            f"{method:14s} "
            f"tk={r['toolkit_size']:3d}  "
            f"ETA {eta_str}"
        )

    def _print_rolling_summary(self) -> None:
        """Print a compact running-total statistics block."""
        elapsed    = time.time() - self.start_time
        mean_score = statistics.mean(self.scores)
        med_score  = statistics.median(self.scores)
        mean_time  = statistics.mean(self.times)
        solve_pct  = 100 * self.solved_exact / self.done
        above80    = self.pp_train + self.partial

        # Rolling score distribution buckets
        buckets = {"≥0.99": 0, "0.80-0.99": 0, "0.50-0.80": 0, "<0.50": 0}
        for s in self.scores:
            if   s >= 0.99: buckets["≥0.99"]    += 1
            elif s >= 0.80: buckets["0.80-0.99"] += 1
            elif s >= 0.50: buckets["0.50-0.80"] += 1
            else:           buckets["<0.50"]      += 1

        print(f"\n  ┌─ [{self.done}/{self.n_tasks} done  {_fmt_duration(elapsed)} elapsed] ─────────────────")
        if self.mode == "infer":
            print(f"  │  PP train:           {self.pp_train:3d}  ({100*self.pp_train/self.done:.1f}%)")
            print(f"  │  Partial (>80%):     {self.partial:3d}   "
                  f"above 80%: {above80} ({100*above80/self.done:.0f}%)")
        else:
            print(f"  │  ✓ Solved (exact):   {self.solved_exact:3d}  ({solve_pct:.1f}%)")
            print(f"  │  ◇ Overfits:         {self.overfits:3d}  "
                  f"(PP train: {self.pp_train})")
            print(f"  │  △ Flukes:           {self.flukes:3d}  "
                  f"(TC: {self.test_ok})")
            print(f"  │  ~ Partial (>80%):   {self.partial:3d}   "
                  f"above 80%: {above80} ({100*above80/self.done:.0f}%)")
        print(f"  │  Mean / median score:  {mean_score:.3f} / {med_score:.3f}")
        print(f"  │  Score dist:  "
              f"✓{buckets['≥0.99']}  "
              f"~{buckets['0.80-0.99']}  "
              f"△{buckets['0.50-0.80']}  "
              f"✗{buckets['<0.50']}")
        print(f"  │  Avg task time:  {mean_time:.2f}s  "
              f"(min {min(self.times):.2f}s  max {max(self.times):.2f}s)")
        print(f"  │  Wall-clock rate:  {self.done/max(elapsed,0.001):.2f} tasks/s  "
              f"({self.n_workers} workers)")
        print(f"  └──────────────────────────────────────────────────────────\n")


def _fmt_duration(seconds: float) -> str:
    """Format seconds as '1h23m', '4m32s', or '45s'."""
    s = int(seconds)
    if s >= 3600:
        return f"{s // 3600}h{(s % 3600) // 60:02d}m"
    if s >= 60:
        return f"{s // 60}m{s % 60:02d}s"
    return f"{s}s"


# ── Culture aggregation ────────────────────────────────────────────────────

def _aggregate_and_save_culture(
    task_results: dict[str, dict],
    save_path: str,
    verbose: bool = True,
) -> None:
    """Aggregate learned culture from all workers and save to a JSON file.

    Each worker independently discovers concepts and programs. This function
    merges them all into a single culture file, deduplicating by name.
    """
    all_concepts: dict[str, dict] = {}
    all_programs: list[dict] = []
    all_features: dict[str, dict] = {}

    for r in task_results.values():
        # Merge learned concepts (deduplicate by name)
        for concept in r.get("_learned_concepts", []):
            name = concept["name"]
            if name not in all_concepts:
                all_concepts[name] = concept

        # Collect all solved programs
        all_programs.extend(r.get("_solved_programs", []))

        # Merge task features
        for tid, features in r.get("_task_features", {}).items():
            if tid not in all_features:
                # Only keep serializable values
                serializable = {}
                for k, v in features.items():
                    if isinstance(v, (bool, int, float, str)):
                        serializable[k] = v
                all_features[tid] = serializable

    # Deduplicate programs by (task_id, name) pair
    seen_progs = set()
    unique_programs = []
    for prog in all_programs:
        key = (prog.get("task_id", ""), prog.get("name", ""))
        if key not in seen_progs:
            seen_progs.add(key)
            unique_programs.append(prog)

    culture = {
        "version": "0.9",
        "learned_concepts": list(all_concepts.values()),
        "successful_programs": unique_programs,
        "task_features": all_features,
    }

    with open(save_path, "w") as f:
        json.dump(culture, f, indent=2)

    if verbose:
        print(f"\nCulture saved to: {save_path}")
        print(f"  Learned concepts: {len(all_concepts)}")
        print(f"  Successful programs: {len(unique_programs)}")
        print(f"  Task features: {len(all_features)}")


# ── Main evaluation harness ────────────────────────────────────────────────

def evaluate_dataset(
    tasks: dict[str, dict],
    population_size: int = 60,
    max_generations: int = 30,
    max_program_length: int = 4,
    verbose: bool = True,
    output_path: str = "",
    seed: int = 42,
    workers: int = 0,
    load_culture_path: str = "",
    save_culture_path: str = "",
    mode: str = "train",
    top_k: int = 3,
    compute_cap: int = 8_000_000,
    time_limit: float = 0.0,
) -> dict:
    """Run the Four Pillars solver on a dataset and collect metrics.

    Results stream to stdout as tasks finish, so you can monitor progress
    live and Ctrl-C to abort cleanly at any point.

    Args:
        tasks:              task_id → task dict (from load_dataset)
        population_size:    evolutionary population size per worker
        max_generations:    max generations per task
        max_program_length: max program chain length (reserved for future use)
        verbose:            print per-task and summary output
        output_path:        if set, save results JSON to this path
        seed:               global random seed (fully reproducible for fixed
                            (seed, workers) pair)
        workers:            0 → auto (performance cores), 1 → single-process,
                            N → exactly N processes
        load_culture_path:  if set, load pre-trained culture from this JSON file
                            into each worker's solver before solving tasks
        save_culture_path:  if set, aggregate learned culture from all workers
                            and save to this JSON file after evaluation
        mode:               "train" or "eval". Train mode saves culture;
                            eval mode loads culture. Future: train-only
                            logic like candidate selection learning.
        top_k:              Number of diverse candidates to test against
                            held-out test output (default 3). Higher = more
                            chances to pass test, but diminishing returns.
        compute_cap:        Cell-normalized compute cap (default 8M).
                            Per-task budget = min(compute_cap / cells, 10K).
                            The 10K ceiling prevents small-grid runaway;
                            compute_cap controls when large-grid tasks get
                            fewer evals.  Set 0 to disable (unlimited).
        time_limit:         Maximum wall-clock seconds per task (default 0
                            = unlimited). Non-deterministic.

    Returns:
        Dict with keys:
          "task_results" — {task_id: per-task metrics}
          "summary"      — aggregate benchmark metrics
    """
    random.seed(seed)

    n_workers = workers if workers > 0 else default_workers()
    n_workers = max(1, min(n_workers, len(tasks)))

    sorted_ids = sorted(tasks.keys())
    n_tasks    = len(sorted_ids)

    if verbose:
        _tmp = FourPillarsSolver(verbose=False)
        init_tk = _tmp.toolkit.size
        del _tmp
        print(f"Mode: {mode.upper()}  |  CPU: {describe_cpu()}")
        tl_str = f"{time_limit:.0f}s" if time_limit > 0 else "unlimited"
        cc_str = f"{compute_cap:,}" if compute_cap > 0 else "unlimited"
        print(f"Workers: {n_workers}  |  Tasks: {n_tasks}  |  "
              f"Seed: {seed}  |  "
              f"Initial toolkit: {init_tk} concepts")
        print(f"Compute cap: {cc_str}  |  Time limit: {tl_str}")
        print()

    if load_culture_path and verbose:
        print(f"Loading culture from: {load_culture_path}")

    # Each task gets a deterministic seed derived from its position in the
    # sorted list, so (seed, workers) always gives identical results.
    worker_args = [
        (task_id, tasks[task_id], population_size, max_generations,
         seed + i * 1000, load_culture_path, mode, top_k,
         compute_cap, time_limit)
        for i, task_id in enumerate(sorted_ids)
    ]

    start_time = time.time()
    tracker    = _ProgressTracker(n_tasks, n_workers, start_time, mode=mode)
    task_results: dict[str, dict] = {}

    try:
        if n_workers == 1:
            # In-process path — easier to debug/profile; same worker function
            # as the parallel path so behaviour is identical.
            for args in worker_args:
                r = _solve_one(args)
                task_results[r["task_id"]] = r
                if verbose:
                    tracker.update(r)
        else:
            # Parallel path — imap_unordered streams results as they arrive
            # so the progress display is live even with many workers.
            with multiprocessing.Pool(processes=n_workers) as pool:
                for r in pool.imap_unordered(_solve_one, worker_args):
                    task_results[r["task_id"]] = r
                    if verbose:
                        tracker.update(r)

    except KeyboardInterrupt:
        print("\n\n  ⚠  Aborted by user — partial results below.\n")
        # Fall through to print whatever we have so far

    total_time = time.time() - start_time

    # ── Aggregate and save culture from all workers ──────────────────────
    if save_culture_path and task_results:
        _aggregate_and_save_culture(task_results, save_culture_path, verbose)

    # ── Final summary ────────────────────────────────────────────────────
    completed = len(task_results)

    # Core counts
    pp_train  = sum(1 for r in task_results.values() if r["solved"])
    tc        = sum(1 for r in task_results.values() if r.get("test_passed"))
    partial   = sum(1 for r in task_results.values()
                    if not r["solved"] and r["score"] > 0.8)

    # The golden number: pixel-perfect on BOTH train AND test.
    # This is our best estimate of private eval performance.
    solved_exact = sum(1 for r in task_results.values()
                       if r["solved"] and r.get("test_passed"))

    # Flukes: passed test but NOT pixel-perfect on train.
    # The program didn't truly understand the task — test success is luck.
    flukes = sum(1 for r in task_results.values()
                 if not r["solved"] and r.get("test_passed"))

    # Overfits: pixel-perfect on train but failed test.
    # The program memorized training examples but doesn't generalize.
    overfits = sum(1 for r in task_results.values()
                   if r["solved"] and not r.get("test_passed"))

    if verbose and completed > 0:
        scores = [r["score"] for r in task_results.values()]
        times  = [r["time_seconds"] for r in task_results.values()]
        above80 = pp_train + partial

        print(f"\n{'='*60}")
        print(f"BENCHMARK RESULTS — {mode.upper()} MODE")
        print(f"{'='*60}")
        print(f"Tasks completed:    {completed}/{n_tasks}")

        if mode == "infer":
            # Infer mode: no test output was examined, show train-only metrics
            print(f"Pixel-perfect train:{pp_train}/{completed} "
                  f"({100*pp_train/max(completed,1):.1f}%)")
            print(f"Partial (>80%):     {partial}/{completed} "
                  f"({100*partial/max(completed,1):.1f}%)")
            print(f"Above 80% total:    {above80}/{completed} "
                  f"({100*above80/max(completed,1):.1f}%)")
        else:
            # Train/eval: full scoreboard with test validation
            print(f"Solved (exact):     {solved_exact}/{completed} "
                  f"({100*solved_exact/max(completed,1):.1f}%)  "
                  f"← pixel-perfect on train AND test")
            print(f"Test confirmed:     {tc}/{completed} "
                  f"({100*tc/max(completed,1):.1f}%)")
            if flukes > 0:
                print(f"  └─ Flukes:        {flukes} "
                      f"(passed test but FAILED train → likely luck)")
            print(f"Pixel-perfect train:{pp_train}/{completed} "
                  f"({100*pp_train/max(completed,1):.1f}%)")
            if overfits > 0:
                print(f"  └─ Overfits:      {overfits} "
                      f"(pixel-perfect on train, FAILED test)")
            print(f"Partial (>80%):     {partial}/{completed} "
                  f"({100*partial/max(completed,1):.1f}%)")
            print(f"Above 80% total:    {above80}/{completed} "
                  f"({100*above80/max(completed,1):.1f}%)")

        print(f"Mean score:         {statistics.mean(scores):.3f}")
        print(f"Median score:       {statistics.median(scores):.3f}")
        print(f"Score std-dev:      {statistics.stdev(scores) if len(scores)>1 else 0:.3f}")
        print(f"Total time:         {_fmt_duration(total_time)} "
              f"({total_time/max(completed,1):.2f}s/task avg)")
        print(f"Wall-clock rate:    "
              f"{completed/max(total_time,0.001):.2f} tasks/s  "
              f"({n_workers} workers)")

        # Candidate diversity: tasks with multiple pixel-perfect candidates
        n_cands = [r.get("n_candidates", 0) for r in task_results.values()]
        multi_cand = sum(1 for c in n_cands if c > 1)
        total_cands = sum(n_cands)
        if total_cands > 0:
            print(f"Total candidates:   {total_cands} across "
                  f"{sum(1 for c in n_cands if c > 0)} tasks  "
                  f"(top-{top_k} submitted)")
            if multi_cand > 0:
                print(f"  └─ Multi-cand:    {multi_cand} tasks had >1 pixel-perfect candidate")
        print(f"{'='*60}")

    summary = {
        "mode":               mode,
        "total_tasks":        n_tasks,
        "completed_tasks":    completed,
        "solved_exact":       solved_exact,
        "solve_rate":         solved_exact / max(completed, 1),
        "test_confirmed":     tc,
        "flukes":             flukes,
        "pixel_perfect_train":pp_train,
        "overfits":           overfits,
        "partial_solved":     partial,
        "above_80pct":        pp_train + partial,
        "mean_score":         statistics.mean(r["score"] for r in task_results.values())
                              if task_results else 0.0,
        "total_time_seconds": total_time,
        "avg_time_per_task":  total_time / max(completed, 1),
        "workers_used":       n_workers,
        "seed":               seed,
        "top_k":              top_k,
    }

    result = {"task_results": task_results, "summary": summary}

    if output_path and task_results:
        # Sort task_results by task_id for deterministic JSON output
        result["task_results"] = dict(sorted(task_results.items()))
        with open(output_path, "w") as f:
            json.dump(result, f, indent=2)
        if verbose:
            print(f"\nResults saved to: {output_path}")

    return result
