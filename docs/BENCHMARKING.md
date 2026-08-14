# Benchmarking methodology

This document exists so that anyone can decide whether to believe ArmForge's
numbers, and reproduce them if they want to.

## What is measured

Two metrics, always reported separately:

| Metric | Phase | Bound by |
| --- | --- | --- |
| **Prefill** (prompt processing, tok/s) | Reading the prompt | Integer/float matrix throughput |
| **Decode** (token generation, tok/s) | Producing output | Memory bandwidth |

Keeping these apart is the whole point. Prefill is a matrix-multiply problem
and responds to `FEAT_I8MM`, `FEAT_SME2` and more cores. Decode streams the
entire weight set from memory for every token and often gets *slower* with
more threads. A single "tokens/sec" headline hides this, and hiding it is how
you end up recommending a configuration that is fast on a benchmark and slow
in a chat window.

**Time to first token** is *derived*, not observed:
`ttft_ms = prompt_tokens / prefill_tok_per_s × 1000`. It excludes model load
time, which is reported separately. ArmForge labels it as derived wherever it
appears.

**Peak memory** is real resident-set high-water mark from `os.wait4`, measured
per benchmark process. The commonly used `getrusage(RUSAGE_CHILDREN)` reports
a maximum across every child a process has ever reaped, which silently
attributes the largest run's memory to every later one.

## The instrument

ArmForge drives [`llama-bench`](https://github.com/ggml-org/llama.cpp), the
benchmarking tool that ships with llama.cpp, rather than timing inference
itself. This is deliberate:

- it is the tool llama.cpp and Arm engineers already use, so results are
  comparable to published figures;
- it performs its own warmup pass before measuring;
- it repeats each measurement and reports every repetition in `samples_ts`,
  which ArmForge summarises into mean, median, min, max and standard
  deviation rather than reporting one number;
- it reports prompt processing and generation as separate rows.

ArmForge adds peak memory, feature verification, and provenance.

## Workloads

Workloads are *shapes*, expressed as token counts. `llama-bench` synthesises
tokens, so no text corpus is involved. This keeps runs exactly reproducible
and avoids any dataset licensing question. It also means these workloads
measure **speed only** — they say nothing about output quality.

| Workload | Prompt | Generate | Dominated by |
| --- | ---: | ---: | --- |
| `short` | 128 | 64 | decode |
| `code` | 512 | 256 | balanced |
| `summarize` | 1024 | 128 | prefill |
| `long-context` | 2048 | 128 | prefill |

## Build variants

A benchmark of "llama.cpp" is meaningless without saying *which* llama.cpp.
`scripts/setup-llama-cpp.sh` builds three, all with Metal disabled because
ArmForge measures the CPU:

| Variant | KleidiAI | Accelerate | What it isolates |
| --- | --- | --- | --- |
| `cpu` | off | off | ggml's own Arm kernels — the controlled baseline |
| `kleidiai` | **on** | off | Arm's KleidiAI micro-kernels (SME2 / i8mm paths) |
| `accelerate` | off | **on** | Apple's BLAS — the default macOS build |

The third variant exists because of a finding during development: a default
macOS build reports `"backends": "BLAS"` and `"cpu_info": "Accelerate, Apple
M4"`, and routes prompt processing through Apple's Accelerate framework
instead of ggml's Arm kernels. Benchmarking that build and attributing its
prefill speed to Arm instruction-set features would have been wrong.
`accelerate` is therefore built, labelled, and never compared against the
other two as though it were the same kind of result.

Build flags are read from each build's `CMakeCache.txt`, not inferred from the
directory name, so a result cannot claim KleidiAI was enabled when it was not.

## Capability is not usage

Detecting `FEAT_I8MM` proves the *silicon* can execute `SMMLA`. It does not
prove the runtime issued a single one.

ArmForge keeps these claims separate and gathers evidence for the second one.
`probe_ggml_features()` runs `llama-completion` once and captures ggml's own
startup line:

```text
CPU : NEON = 1 | ARM_FMA = 1 | FP16_VA = 1 | MATMUL_INT8 = 1 | DOTPROD = 1 | SME = 1 | REPACK = 1 |
```

`MATMUL_INT8` is ggml's name for the i8mm path; `REPACK` is its weight
repacking pass, which rearranges `Q4_0` into the blocked layout `SMMLA`
consumes. Without `REPACK`, a CPU's `i8mm` goes unused no matter what the
hardware reports. If the probe cannot run, ArmForge records *nothing* — an
empty result means "not established", never "no features".

## Statistical handling

- Every configuration is repeated (default 5 repetitions plus llama-bench's
  own warmup).
- Results carry mean, median, min, max, standard deviation and sample count.
- `relative_stddev` above ~5% indicates a noisy machine; treat those runs as
  suspect.
- Two configurations are only reported as different when the gap between
  their means exceeds 2× their pooled standard deviation
  (`stats.is_meaningfully_different`). On a thermally throttled laptop,
  run-to-run jitter easily reaches several percent, and without this guard it
  reads as a real optimisation.

## Provenance

Every result records: model path and quantisation, model content hash, exact
parameter count, runtime name, llama.cpp commit, **build flags**, host CPU
model, architecture, core topology, detected Arm features, thread count,
workload shape, warmup and iteration counts, and a UTC timestamp.

A number without that context is not evidence, and ArmForge will not emit one.

## Reproducing

```bash
scripts/setup-llama-cpp.sh
```

```bash
armforge runtimes --probe --model path/to/model.gguf
```

```bash
armforge benchmark path/to/model.gguf --workload long-context
```

## Known limitations

- **macOS cannot pin threads.** There is no CPU affinity API on Apple
  silicon, so on macOS ArmForge can vary thread *count* but cannot bind
  threads to performance cores. On Linux it collects per-cluster CPU ids and
  can. This is reported in `armforge hardware` output.
- **Laptops throttle.** Sustained sweeps on a fanless or lightly cooled
  machine will drift. Prefer a cool machine, and check `relative_stddev`.
- **Synthetic tokens measure speed, not quality.** Quantisation that is fast
  may also be less accurate; ArmForge currently does not measure perplexity
  or any quality metric, and does not claim to.
