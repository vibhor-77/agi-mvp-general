"""
CPU topology detection for optimal worker count.

On Apple Silicon (M-series) and similar big.LITTLE architectures,
mixing performance and efficiency cores in a compute-heavy workload
can be counterproductive: efficiency cores are slower and can become
bottlenecks when tasks aren't evenly sized. We therefore default to
the number of *performance* cores, not the total logical CPU count.

Supported detection methods (in priority order):
  1. macOS sysctl — hw.perflevel0.logicalcpu (Apple Silicon)
  2. Linux /sys/devices/system/cpu/cpu*/cpu_capacity (ARM big.LITTLE)
  3. os.cpu_count() fallback (x86 and all other platforms)

All values are runtime-detected and cached. The result is always ≥ 1.
"""
from __future__ import annotations
import os
import subprocess
import sys


def _detect_performance_cores() -> int:
    """Return the number of performance (big) cores on this machine."""

    # --- macOS: Apple Silicon M-series ---
    if sys.platform == "darwin":
        try:
            # perflevel0 = performance cluster, perflevel1 = efficiency cluster
            out = subprocess.check_output(
                ["sysctl", "-n", "hw.perflevel0.logicalcpu"],
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
            n = int(out)
            if n > 0:
                return n
        except (subprocess.SubprocessError, ValueError, OSError):
            pass

    # --- Linux: ARM big.LITTLE via cpu_capacity ---
    # The kernel exposes relative capacity per core. On a homogeneous (x86)
    # machine all values are 1024. On big.LITTLE, big cores have capacity=1024
    # and little cores have smaller values.
    try:
        total = os.cpu_count() or 1
        capacities: list[int] = []
        for i in range(total):
            cap_path = f"/sys/devices/system/cpu/cpu{i}/cpu_capacity"
            if os.path.exists(cap_path):
                with open(cap_path) as f:
                    capacities.append(int(f.read().strip()))

        if capacities:
            max_cap = max(capacities)
            perf_count = sum(1 for c in capacities if c == max_cap)
            if 0 < perf_count < total:
                # Genuine big.LITTLE: return only the big cores
                return perf_count
            # Homogeneous (all equal) — fall through to os.cpu_count()
    except (OSError, ValueError):
        pass

    # --- Fallback: all logical CPUs ---
    return os.cpu_count() or 1


# Cache the result at import time (detection is a one-time syscall)
_PERFORMANCE_CORES: int = _detect_performance_cores()


def default_workers() -> int:
    """Return the recommended number of parallel worker processes.

    Uses only performance cores to avoid efficiency-core bottlenecks on
    Apple Silicon and ARM big.LITTLE. On homogeneous hardware (x86, etc.)
    this returns the full logical CPU count.

    Always returns at least 1.
    """
    return max(1, _PERFORMANCE_CORES)


def describe_cpu() -> str:
    """Human-readable description of the detected CPU configuration.

    Example outputs:
      "8 performance cores (macOS Apple Silicon)"
      "12 performance cores + 4 efficiency cores (Linux big.LITTLE)"
      "8 logical CPUs (homogeneous)"
    """
    total = os.cpu_count() or 1
    perf  = _PERFORMANCE_CORES

    if sys.platform == "darwin":
        try:
            subprocess.check_output(
                ["sysctl", "-n", "hw.perflevel0.logicalcpu"],
                stderr=subprocess.DEVNULL,
            )
            eff = total - perf
            if eff > 0:
                return f"{perf} performance + {eff} efficiency cores (macOS Apple Silicon)"
            return f"{perf} performance cores (macOS)"
        except (subprocess.SubprocessError, OSError):
            pass

    if perf < total:
        eff = total - perf
        return f"{perf} performance + {eff} efficiency cores (Linux big.LITTLE)"

    return f"{total} logical CPUs (homogeneous)"
