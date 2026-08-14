# Web UI

A dashboard, sweep runner, and results viewer over the same engine the CLI
uses. There is no mock mode: every page calls a FastAPI backend that imports
the `armforge` package directly and runs the real hardware detection,
candidate generation, benchmarking and scoring. If the backend is down, pages
fail loudly rather than falling back to placeholder data.

## Running it

Two processes: the API (Python) and the UI (Next.js).

```bash
pip install -e ".[dev]"
pip install -r backend/requirements.txt
PYTHONPATH=backend uvicorn app.main:app --port 8000
```

```bash
cd frontend && npm install && npm run dev
```

Open `http://localhost:3000`.

By default the model picker scans `~/.cache/armforge/models` and historical
runs are read from the repo's `results/` directory. Override either with
`ARMFORGE_MODELS_DIR` / `ARMFORGE_RESULTS_DIR` on the backend process.

## What's there

- **Dashboard** — this machine's Arm capability profile, plus every committed
  sweep in `results/` (including the cross-Arm ones from CI) as a live table.
- **New optimization** — pick real models and runtime builds discovered on
  disk, a workload, an objective, and repetitions; submits a real sweep.
- **Live run** — polls the sweep as it executes, showing the same
  candidate-by-candidate progress the CLI prints, with measurements appearing
  as each one completes.
- **Results** — recommendation, per-phase thread split, caveats, Pareto
  frontier, real SVG charts rendered server-side from the actual data points,
  and a one-click download of the same deployment package `armforge export`
  produces.

Historical and live runs share one results view: `/runs/<job-id>` for a sweep
just started, `/runs/history-<name>` for anything already in `results/`. Both
recompute their recommendation on the fly rather than trusting a cached
value, the same rule `armforge export` follows — measurements are fixed,
scoring can improve.

## Design constraints, deliberately kept

- **One sweep at a time.** Two benchmarks running together would contend for
  the same CPU and corrupt both sets of timings. A second `POST /api/optimize`
  while one is active gets `409`.
- **No database.** Job state lives in the backend process's memory. This is a
  local tool measuring the machine it runs on; a restart losing an in-flight
  job is an acceptable trade against adding persistence to a benchmark
  runner. Finished sweeps you want to keep should be exported or written with
  `armforge optimize --output`.
- **Models are referenced by path, not uploaded.** A browser upload of a
  multi-gigabyte GGUF file would be the slowest possible way to point the
  tool at a file already sitting on the same disk the server runs on.
