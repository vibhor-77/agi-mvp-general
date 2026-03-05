#!/usr/bin/env python3
"""
Test runner with built-in coverage measurement.

Works with or without pytest. Measures line coverage using Python's
built-in trace module and reports per-module statistics.

Usage:
    # With pytest (recommended):
    pip install pytest
    python -m pytest tests/ -v

    # Without pytest (with coverage):
    python run_tests.py

    # Without pytest (without coverage):
    python run_tests.py --no-coverage
"""
import sys
import os
import unittest
import trace
import argparse

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)
ARC_AGENT_DIR = os.path.join(PROJECT_ROOT, "arc_agent")


def run_with_pytest():
    """Run tests using pytest."""
    import pytest
    return pytest.main(["-v", "tests/"])


def discover_and_run_tests(with_coverage: bool = True):
    """Discover and run all unittest-based tests with optional coverage.

    Uses unittest discovery to find all test_*.py files in tests/.
    Optionally traces execution to measure line coverage of arc_agent/.

    Args:
        with_coverage: If True, measure and report line coverage.

    Returns:
        Exit code (0 = all tests passed, 1 = failures).
    """
    # Discover all tests
    loader = unittest.TestLoader()
    suite = loader.discover("tests", pattern="test_*.py", top_level_dir=".")

    if not with_coverage:
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)
        return 0 if result.wasSuccessful() else 1

    # Run with coverage tracing
    tracer = trace.Trace(
        count=True,
        trace=False,
        countfuncs=False,
        countcallers=False,
    )

    # Run tests under trace
    result_holder = {}

    def traced_runner():
        runner = unittest.TextTestRunner(verbosity=2)
        result_holder["result"] = runner.run(suite)

    tracer.runfunc(traced_runner)
    test_result = result_holder["result"]

    # Gather coverage results
    results = tracer.results()
    counts = results.counts  # dict of (filename, lineno) → count

    # Analyze coverage per module
    print("\n" + "=" * 60)
    print("COVERAGE REPORT")
    print("=" * 60)

    source_files = {}
    for filename, lineno in counts:
        abs_path = os.path.abspath(filename)
        if ARC_AGENT_DIR in abs_path and abs_path.endswith(".py"):
            if abs_path not in source_files:
                source_files[abs_path] = set()
            source_files[abs_path].add(lineno)

    # Count total executable lines per source file
    total_covered = 0
    total_executable = 0
    file_stats = []

    for filepath in sorted(os.listdir(ARC_AGENT_DIR)):
        if not filepath.endswith(".py"):
            continue
        full_path = os.path.join(ARC_AGENT_DIR, filepath)
        abs_path = os.path.abspath(full_path)

        # Count executable lines (non-blank, non-comment, non-decorator-only)
        executable = set()
        with open(full_path, "r") as f:
            for i, line in enumerate(f, 1):
                stripped = line.strip()
                if (stripped and
                    not stripped.startswith("#") and
                    not stripped.startswith('"""') and
                    not stripped.startswith("'''") and
                    stripped != '"""' and
                    stripped != "'''"):
                    executable.add(i)

        covered = source_files.get(abs_path, set()) & executable
        n_exec = len(executable)
        n_covered = len(covered)
        total_executable += n_exec
        total_covered += n_covered

        pct = (n_covered / n_exec * 100) if n_exec > 0 else 100.0
        file_stats.append((filepath, n_covered, n_exec, pct))

    for filename, covered, total, pct in file_stats:
        bar = "█" * int(pct // 5) + "░" * (20 - int(pct // 5))
        print(f"  {filename:25s}  {covered:3d}/{total:3d}  {bar} {pct:5.1f}%")

    overall_pct = (total_covered / total_executable * 100) if total_executable else 100
    print(f"\n  {'TOTAL':25s}  {total_covered:3d}/{total_executable:3d}  "
          f"{'█' * int(overall_pct // 5)}{'░' * (20 - int(overall_pct // 5))} "
          f"{overall_pct:5.1f}%")
    print("=" * 60)

    return 0 if test_result.wasSuccessful() else 1


def main():
    parser = argparse.ArgumentParser(description="Run tests with coverage")
    parser.add_argument(
        "--no-coverage", action="store_true",
        help="Skip coverage measurement (faster)"
    )
    args = parser.parse_args()

    try:
        sys.exit(run_with_pytest())
    except ImportError:
        print("pytest not installed, running unittest discovery...")
        print("Install pytest for full test coverage: pip install pytest\n")
        sys.exit(discover_and_run_tests(with_coverage=not args.no_coverage))


if __name__ == "__main__":
    main()
