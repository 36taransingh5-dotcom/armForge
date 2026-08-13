# ArmForge

**An Arm-aware inference configuration engine.**

ArmForge reads what an Arm CPU can actually do — its instruction set
extensions, its implemented vector lengths, its core topology — predicts which
LLM inference configuration should win on that specific silicon, and then
proves or refutes the prediction by measurement.

> Status: in development for the Arm AI Optimization Challenge 2026.
> Milestone 1 (hardware and Arm capability detection) is complete.

---

## Why this is not just an auto-tuner

The obvious way to pick an inference configuration is to sweep every
combination of quantisation format, runtime and thread count and report the
winner. That works, but it is architecture-agnostic: the same program would
produce the same design on x86, and it learns nothing you can carry to the
next machine.

ArmForge inverts the loop. It starts from the CPU's capability vector and uses
Arm-specific knowledge to decide what is worth measuring at all:

- **`FEAT_I8MM` decides which quantisation format can win.** llama.cpp repacks
  `Q4_0` weights into a blocked layout at load time specifically to feed the
  `SMMLA` int8 matrix instruction. On a core with `i8mm` that repack is a large
  prefill win; on a core without it, the layout is dead weight and a K-quant is
  the better starting point.
- **Core topology decides thread count.** `nproc` is the wrong answer on any
  heterogeneous part. Apple's M4 is 4 performance + 6 efficiency cores; a
  phone SoC is big.LITTLE; Graviton is uniform. The right thread count differs,
  and so does the right answer for prefill versus decode.
- **Prefill and decode are different machines.** Prompt processing is
  compute-bound and rewards wide int8 matrix instructions and more cores.
  Token generation is memory-bandwidth-bound and often gets *slower* with more
  threads. Reporting one configuration for both is the standard mistake.

Every prediction the engine makes is then benchmarked. Where the measurement
disagrees with the model, the measurement wins and the disagreement is
reported — that is the interesting result, not a bug.

---

## Install

Requires Python 3.10+.

```bash
git clone https://github.com/<your-user>/armforge.git
```

```bash
cd armforge && python3 -m venv .venv && .venv/bin/pip install -e ".[dev]"
```

## Usage

Detect the host CPU, its Arm feature set and its core topology:

```bash
armforge hardware
```

Add the reasoning behind each detected feature:

```bash
armforge hardware --explain
```

Machine-readable output for scripting and CI:

```bash
armforge hardware --json
```

---

## Supported platforms

| Platform | Backend | Detects |
| --- | --- | --- |
| macOS on Apple silicon | `sysctl` | features, P/E clusters, SME vector length, cache sizes |
| Linux on Arm64 | procfs, sysfs, `prctl` | features, MIDR core identification, SVE/SME vector lengths, per-cluster frequency, CPU ids for pinning |
| Anything else | fallback | core count only, with everything else reported as unknown |

Linux is the path that matters for Arm64 cloud (AWS Graviton, Ampere, Azure
Cobalt) because it exposes `MIDR_EL1`, which identifies the actual
microarchitecture — Neoverse-V2, Cortex-A520 and so on — rather than guessing
from clock speed.

---

## Project principles

1. **No fabricated numbers.** Anything ArmForge cannot measure in the current
   environment is reported as unavailable, never estimated.
2. **Arm-specific means Arm-specific.** A technique that helps equally on x86
   is documented as a general optimisation, not marketed as an Arm one.
3. **Capability is not usage.** Detecting `FEAT_SME2` proves the silicon has a
   matrix engine. It does not prove a runtime used it. Only a benchmark proves
   that, and ArmForge keeps the two claims separate.

---

## License

MIT — see [LICENSE](LICENSE).
