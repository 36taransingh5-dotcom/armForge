# ArmForge

**An Arm-aware inference configuration engine.**

ArmForge reads what an Arm CPU can actually do — its instruction set
extensions, its implemented vector lengths, its core topology — predicts which
LLM inference configuration should win on that specific silicon, benchmarks
the prediction, and tells you when it was wrong.

Run on two structurally opposite Arm chips (Apple M4, Arm Neoverse-N2), it
produces two different, individually justified recommendations. See
[**docs/RESULTS.md**](docs/RESULTS.md) for the measurements, charts, and the
cases where the data corrected the model.

Built for the [Arm AI Optimization Challenge 2026](https://www.arm.com), Cloud AI track.

```
$ armforge optimize model.gguf --workload long-context

ArmForge · Optimize

     Host  Apple M4 · arm64
 Workload  long-context
Objective  best-balance
     Plan  12 candidates, 0 pruned

  [1/12] Q4_0 · llama.cpp · 1t
  ...

Recommendation · best-balance

          Model  Q4_0 qwen2.5-0.5b-instruct
        Runtime  llama.cpp
Prefill threads  6  424.3 tok/s
 Decode threads  2  99.5 tok/s

  Why
  · prompt processing peaks at 6 threads while token generation peaks at 2;
    the two phases are bound by different resources, so they are configured
    separately
  · 228% faster prompt processing than the 10-thread (nproc) baseline
  · 682% faster token generation than the same baseline

Deploy
  llama-completion -m model.gguf -t 2 -tb 6
```

---

## Why this is not an auto-tuner

The obvious way to pick an inference configuration is to sweep every
combination of quantisation, runtime, and thread count and report the winner.
That works, but it's architecture-agnostic — the same program produces the
same design on x86, and it learns nothing that transfers to the next machine.

ArmForge starts from the CPU's capability vector instead, and uses that to
decide what's worth measuring:

- **`FEAT_I8MM` decides which quantisation format can win.** llama.cpp repacks
  `Q4_0` weights into a blocked layout at load time specifically to feed the
  `SMMLA` int8 matrix instruction. Measured: **2.28× faster prompt processing**
  on the M4, **1.72×** on Neoverse-N2 — smaller on chips without the same
  matrix depth, exactly as the capability model predicts.
- **Core topology decides thread count, and `nproc` is often the worst
  answer.** On the M4 (4 performance + 6 efficiency cores), using every core
  drops decode throughput **29× below its 2-thread peak**. On Neoverse-N2 (4
  uniform cores) that collapse doesn't happen — scaling is close to linear and
  all-cores genuinely is best. Same tool, opposite conclusions, both correct
  for their machine.
- **Prefill and decode are different machines.** Prompt processing is
  compute-bound; token generation is memory-bandwidth-bound. On the M4 they
  peak at different thread counts (6 vs 2) and ArmForge reports both,
  producing one deployable command (`-tb 6 -t 2`) rather than a single number
  that's wrong for one phase.
- **A capability flag is not proof of a speedup.** The M4 reports `FEAT_SME2`;
  Arm's KleidiAI kernels are the only path to it in llama.cpp. Measured gain
  decays from **+66% at 1 thread to −2% at 10** — real, but only where the
  workload is compute-bound, and it costs **60% more resident memory**. On
  Neoverse-N2, which has no SME, the same build changes throughput by 0.4% —
  noise. Two machines, one differing feature, the effect appears and
  disappears with it.

Every prediction is benchmarked. Where measurement disagreed with the model,
the measurement won and the model was corrected — twice, so far, both
recorded in git history rather than smoothed over.

---

## Install

Requires Python 3.10+.

```bash
git clone https://github.com/36taransingh5-dotcom/armForge.git
cd armForge && python3 -m venv .venv && .venv/bin/pip install -e ".[dev]"
```

To run real benchmarks, build llama.cpp (three variants: stock ggml,
KleidiAI, and — on macOS only — Accelerate):

```bash
scripts/setup-llama-cpp.sh
```

## Usage

```bash
# What can this CPU actually do, and why does it matter for inference?
armforge hardware --explain

# What is this model, and how does it interact with this CPU's instruction set?
armforge analyze model.gguf

# One configuration, measured with full statistics
armforge benchmark model.gguf --workload long-context --threads 6

# The full pipeline: plan, sweep, score, recommend, justify
armforge optimize model.gguf --workload long-context

# See a measurement plan without running it
armforge optimize model.gguf --dry-run

# Sweep, then write a deployment package you can run, containerise and check
armforge optimize model.gguf --export ./deploy

# Which llama.cpp builds are available, and what did ggml actually compile in
armforge runtimes --probe --model model.gguf
```

Every command has `--json` for scripting, and `optimize`/`benchmark` accept
`--output` to write a full-provenance result artifact.

### The deployment package

`--export` writes a directory that stands on its own — no ArmForge install
required to use it, and nothing in it has to be taken on trust:

| File | Contents |
| --- | --- |
| `run.sh` | Runs the model with the measured per-phase thread split |
| `Dockerfile` | Pins the same llama.cpp commit and build flags that were measured |
| `report.html` | Self-contained report with charts — no network, no scripts, no fonts |
| `config.json` | The winning configuration, machine-readable |
| `benchmark.json` | Every candidate measured, with full provenance |
| `README.md` | The configuration, the evidence, and what the numbers are not |

The model file is referenced by path and content hash, never copied — so the
package stays small and you can verify you hold the same weights the
measurements describe.

Verified by running it, not by reading it. The generated script produces:

```
n_threads = 4 (n_threads_batch = 8)
```

— llama.cpp confirming the per-phase split actually took effect in the
runtime, rather than merely being written down.

### Web UI

A dashboard, sweep runner and results viewer over the same engine — no mock
mode, every page calls a FastAPI backend that runs the real pipeline.

```bash
pip install -r backend/requirements.txt
PYTHONPATH=backend uvicorn app.main:app --port 8000
```

```bash
cd frontend && npm install && npm run dev
```

Details, endpoints and design constraints: [**docs/WEB_UI.md**](docs/WEB_UI.md).

---

## What ArmForge actually measured

Full writeup with charts: [**docs/RESULTS.md**](docs/RESULTS.md). Summary:

| | Apple M4 | Arm Neoverse-N2 |
|---|---|---|
| Cores | 4 performance + 6 efficiency | 4 uniform |
| Matrix engine | SME2 (512-bit) | none |
| Vectors | none (no SVE) | SVE2 (128-bit) |
| **Recommended** | `-t 2 -tb 6` | `-t 4 -tb 4` |
| Why | prefill and decode disagree | they agree — split collapses |
| `nproc` (10/4 threads) cost | 2.5× slower prefill, **29× slower decode** | no collapse — scaling is linear |
| Q4_0 vs Q4_K_M prefill | **2.28×** faster | **1.72×** faster |
| KleidiAI prefill gain (1 thread → 10) | **+66% → −2%** | +0.3% → +0.4% (noise) |
| KleidiAI memory cost | **+60%** resident | negligible |

Everything above is generated from committed JSON artifacts by
[`scripts/build_results.py`](scripts/build_results.py) — nothing on the
results page is typed by hand, so it can't drift from what was actually
measured. The Neoverse-N2 numbers come from a
[GitHub Actions workflow](.github/workflows/arm64-sweep.yml) on a free public
Arm64 runner, reproducible by anyone with a fork.

---

## Architecture

```
armforge/
├── hardware/     CPU + core topology detection, Arm feature registry
│                 (macOS via sysctl, Linux via procfs/sysfs/prctl,
│                  MIDR decoding to real microarchitecture names)
├── analyzer/     GGUF header reader (real quantisation, param count,
│                 not filename guessing)
├── bench/        Runtime-agnostic result types, statistics, subprocess
│                 isolation, and the llama.cpp backend
└── optimize/     Capability-driven candidate generation, transparent
                  scoring, per-phase recommendation
```

**`hardware/features.py`** is the part that makes this Arm-specific rather
than a generic sweeper: a curated registry of the ~12 Arm extensions that
actually change LLM inference performance (`dotprod`, `i8mm`, `bf16`,
SVE/SVE2, SME/SME2), each with a documented reason. Everything downstream —
candidate generation, scoring, the recommendation text — reads from it.

**`optimize/candidates.py`** is where capability becomes a plan. It prunes
KleidiAI on CPUs without `i8mm`/`SME`, prunes models that wouldn't fit in RAM
(a paging measurement describes the disk, not the CPU), prunes the Accelerate
build off macOS (the flag is a silent no-op there — found the hard way, after
it burned a third of a CI sweep). Every candidate carries a rationale; every
exclusion carries a reason.

**`optimize/recommend.py`** picks a configuration per phase rather than
overall, checks every claim against measurement noise before making it
(`is_meaningfully_different`), and computes a Pareto frontier with a tolerance
equal to that noise threshold — without it, a real M4 sweep returned 9 of 12
candidates as "non-dominated" because peak memory varies under 4% between
thread counts of one model, which is noise, not a trade-off.

**Two bugs the test suite caught against real measured data**, not
hypotheticals: the thread-candidate ladder originally missed 6 — the count
that actually won prefill on the M4 — because a single midpoint guess between
the performance cluster and `nproc` isn't the same as sampling the range. And
`tests/test_gguf.py` imported a helper from `conftest.py`, which only resolves
under `python -m pytest`; bare `pytest` (what CI runs) failed with
`ModuleNotFoundError` until the helper moved to its own module.

## Supported platforms

| Platform | Backend | Detects |
|---|---|---|
| macOS / Apple silicon | `sysctl` | features, P/E clusters, SME vector length, cache sizes |
| Linux / Arm64 | procfs, sysfs, `prctl` | features, MIDR core identification (Neoverse-V2, Cortex-A520, ...), SVE/SME vector lengths, per-cluster frequency |
| Anything else | fallback | core count only; everything else reported as unknown, never guessed |

## Testing

```bash
pytest
```

142 tests, including fixtures built directly from measured M4 and
Neoverse-N2 sweeps — the optimizer is pinned to reaching the conclusions the
real data supports, not just internally-consistent synthetic ones. Real
subprocess tests cover timeout handling, process-group kill, pipe-deadlock
avoidance, and per-child peak-memory attribution (the common
`getrusage(RUSAGE_CHILDREN)` approach attributes a high-water mark across
*every* child a process has ever reaped, which silently corrupts memory
figures after the first large run).

## CI

[`.github/workflows/arm64-sweep.yml`](.github/workflows/arm64-sweep.yml) runs
the full pipeline — hardware detection, unit tests, llama.cpp build, model
download, sweep, recommendation — on GitHub's free `ubuntu-24.04-arm` runners
(Arm Neoverse-N2), and publishes results as a downloadable artifact plus a job
summary. This is what supplied the Neoverse-N2 half of every comparison above.

---

## Project principles

1. **No fabricated numbers.** Anything ArmForge can't measure in the current
   environment is reported as unavailable, never estimated. Where a
   measurement was noisy, the tool says so in its own recommendation
   (`"6 threads and 8 threads are within measurement noise; either is
   defensible"`) rather than presenting a confident-sounding average.
2. **Arm-specific means Arm-specific.** A technique that helps equally on x86
   is documented as a general optimisation, not marketed as an Arm one.
3. **Capability is not usage.** Detecting `FEAT_SME2` proves the silicon has a
   matrix engine. It doesn't prove a runtime used it — only a benchmark proves
   that, and ArmForge probes each build's actual compiled-in ggml feature
   flags (`MATMUL_INT8 = 1 | SME = 1 | REPACK = 1`) rather than assuming from
   CPU capability alone.
4. **Measurement outranks the model.** Twice during development a real sweep
   contradicted a design assumption — a thread-candidate ladder that missed
   the actual optimum, and a single-thread-count summary that hid a gain
   decaying to zero. Both times the code changed to match the data, not the
   other way round.

## What's not built yet

- Additional model families beyond Qwen2.5 (architecture supports it;
  not yet validated)
- ONNX Runtime as a second backend
- Output-quality measurement — everything here is throughput, so ArmForge
  cannot currently tell you what a faster quantisation costs you in accuracy

## License

MIT — see [LICENSE](LICENSE).
