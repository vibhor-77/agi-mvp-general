# Installation & Setup

Works on any machine with Python 3.9+. Automatically uses the number of
performance cores for parallel evaluation (Apple Silicon, ARM big.LITTLE)
or all logical CPUs on homogeneous hardware (x86/x64).

---

## 1. Clone this repository

```bash
git clone https://github.com/<your-username>/agi-mvp-general.git
cd agi-mvp-general
```

---

## 2. Create a Python environment

Choose **one** of the methods below.

### Option A — conda (recommended on macOS / Apple Silicon)

```bash
conda create -n fourpillars python=3.11
conda activate fourpillars
pip install numpy
```

### Option B — venv (works everywhere)

```bash
python3 -m venv .venv
source .venv/bin/activate        # macOS / Linux
.venv\Scripts\activate           # Windows

pip install numpy
```

### Option C — pip into your existing environment

```bash
pip install numpy
```

NumPy is the only runtime dependency. Everything else is Python stdlib.

---

## 3. Run the tests

```bash
python -m unittest discover -s tests -p "*.py"
```

All 242 tests should pass. You should see output ending with:
```
Ran 242 tests in 0.3s
OK
```

---

## 4. Run the ARC-AGI benchmark

### 4a. Download the ARC-AGI dataset

```bash
git clone https://github.com/fchollet/ARC-AGI.git
```

### 4b. Evaluate on the training set (400 tasks)

```bash
python -m arc_agent.evaluate --data-dir ARC-AGI/data/training
```

The CLI will print your machine's CPU configuration and automatically use
the right number of workers:

```
CPU: 8 performance + 2 efficiency cores (macOS Apple Silicon)
Workers: 8  |  Tasks: 400  |  Seed: 42  |  Initial toolkit: 114 concepts
```

### 4c. Useful flags

| Flag | Default | Description |
|------|---------|-------------|
| `--workers N` | auto | Worker processes. `0`=auto, `1`=single-process debug |
| `--limit N` | all | Evaluate only the first N tasks (sorted by ID) |
| `--seed N` | 42 | Random seed — same (seed, workers) always gives same results |
| `--population N` | 60 | Evolutionary population size per worker |
| `--generations N` | 30 | Max generations per task |
| `--output path.json` | none | Save full results to JSON |
| `--quiet` | off | Suppress per-task output, print only final summary |

### 4d. Quick sanity check (first 20 tasks, single process)

```bash
python -m arc_agent.evaluate --data-dir ARC-AGI/data/training --limit 20 --workers 1
```

---

## 5. Optional: development dependencies

```bash
pip install pytest pytest-cov
pytest                        # same as unittest discover but with colour output
pytest --cov=arc_agent        # coverage report
```

---

## Reproducibility

Every run is seeded. Worker seeds are derived as `seed + worker_index × 1000`,
so changing `--workers` will give *different* results (each worker sees a
different random stream) but any fixed `(seed, workers)` pair always
produces identical results.
