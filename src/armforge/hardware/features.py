"""Arm architectural features that actually change LLM inference performance.

This module is deliberately narrow. There are well over a hundred ``FEAT_*``
extensions in the Arm architecture; almost none of them matter for running a
transformer on a CPU. The ones catalogued here do, because each one either
unlocks a different GEMM kernel in llama.cpp / KleidiAI / oneDNN, or changes
which quantisation format has a fast path.

Nothing here asserts that a feature *is* being used by a given runtime -- only
what the hardware is capable of. Whether a runtime actually took the fast path
is established by measurement elsewhere.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass


class Relevance(str, enum.Enum):
    """How much this feature moves the needle for CPU LLM inference."""

    CRITICAL = "critical"
    HIGH = "high"
    MODERATE = "moderate"
    LOW = "low"


@dataclass(frozen=True)
class FeatureInfo:
    key: str
    arm_name: str
    title: str
    since: str
    relevance: Relevance
    summary: str
    inference_impact: str


#: Features we understand, keyed by normalised name.
FEATURES: dict[str, FeatureInfo] = {
    "neon": FeatureInfo(
        key="neon",
        arm_name="FEAT_AdvSIMD",
        title="Advanced SIMD (NEON)",
        since="Armv8.0-A",
        relevance=Relevance.CRITICAL,
        summary="128-bit fixed-width SIMD. Mandatory on essentially all Armv8-A cores.",
        inference_impact=(
            "The baseline vector path. Every Arm GGML kernel assumes it; its "
            "absence would mean scalar fallback and unusable throughput."
        ),
    ),
    "fp16": FeatureInfo(
        key="fp16",
        arm_name="FEAT_FP16",
        title="Half-precision floating point arithmetic",
        since="Armv8.2-A",
        relevance=Relevance.MODERATE,
        summary="Native fp16 arithmetic rather than convert-compute-convert in fp32.",
        inference_impact=(
            "Lets F16 weights be consumed without widening, which reduces both "
            "instruction count and register pressure in the dequantise step."
        ),
    ),
    "dotprod": FeatureInfo(
        key="dotprod",
        arm_name="FEAT_DotProd",
        title="Int8 dot product (SDOT/UDOT)",
        since="Armv8.2-A",
        relevance=Relevance.CRITICAL,
        summary="Four-way 8-bit dot product accumulating into 32-bit lanes.",
        inference_impact=(
            "The workhorse for integer-quantised inference. llama.cpp's Q4_0, "
            "Q4_K and Q8_0 kernels all have dedicated SDOT paths; without it "
            "int8 quantisation loses most of its speed advantage and mainly "
            "just saves memory."
        ),
    ),
    "i8mm": FeatureInfo(
        key="i8mm",
        arm_name="FEAT_I8MM",
        title="Int8 matrix multiply (SMMLA/UMMLA)",
        since="Armv8.6-A",
        relevance=Relevance.CRITICAL,
        summary="8-bit integer outer-product matrix multiply into 32-bit accumulators.",
        inference_impact=(
            "Substantially raises int8 arithmetic density over SDOT. llama.cpp "
            "repacks Q4_0 weights into a blocked layout at load time "
            "specifically to feed SMMLA, which is why prompt processing "
            "(prefill) benefits far more than token generation (decode). "
            "The presence or absence of this feature is the strongest single "
            "predictor of which quantisation format wins."
        ),
    ),
    "bf16": FeatureInfo(
        key="bf16",
        arm_name="FEAT_BF16",
        title="BFloat16 matrix multiply (BFMMLA/BFDOT)",
        since="Armv8.6-A",
        relevance=Relevance.HIGH,
        summary="bfloat16 multiply with fp32 accumulation.",
        inference_impact=(
            "Enables a fast path for BF16 and F16 weights without going to "
            "integer quantisation, preserving accuracy at higher throughput. "
            "Relevant when quality loss from int4/int8 is unacceptable."
        ),
    ),
    "sve": FeatureInfo(
        key="sve",
        arm_name="FEAT_SVE",
        title="Scalable Vector Extension",
        since="Armv8.2-A (optional)",
        relevance=Relevance.HIGH,
        summary="Vector-length-agnostic SIMD; implementations range 128-2048 bits.",
        inference_impact=(
            "Wider vectors and predication reduce loop tail overhead. Actual "
            "benefit depends on the implemented vector length, so ArmForge "
            "reads it rather than assuming 128 bits. Notably absent on Apple "
            "silicon, which implements SME instead."
        ),
    ),
    "sve2": FeatureInfo(
        key="sve2",
        arm_name="FEAT_SVE2",
        title="Scalable Vector Extension 2",
        since="Armv9.0-A",
        relevance=Relevance.HIGH,
        summary="SVE plus integer and permute instructions aimed at DSP-style work.",
        inference_impact=(
            "Broadens the set of quantisation and dequantisation steps that "
            "can stay in the vector unit. Present on Graviton4 and recent "
            "Cortex-X/A7xx mobile cores."
        ),
    ),
    "svei8mm": FeatureInfo(
        key="svei8mm",
        arm_name="FEAT_SVE_I8MM",
        title="SVE int8 matrix multiply",
        since="Armv8.6-A",
        relevance=Relevance.HIGH,
        summary="SMMLA-equivalent operating on scalable vectors.",
        inference_impact=(
            "Combines int8 matrix density with SVE's vector length, the "
            "preferred integer GEMM path on wide Neoverse parts."
        ),
    ),
    "svebf16": FeatureInfo(
        key="svebf16",
        arm_name="FEAT_SVE_BF16",
        title="SVE BFloat16 matrix multiply",
        since="Armv8.6-A",
        relevance=Relevance.MODERATE,
        summary="BFMMLA on scalable vectors.",
        inference_impact="BF16 GEMM at the implemented SVE vector length.",
    ),
    "sme": FeatureInfo(
        key="sme",
        arm_name="FEAT_SME",
        title="Scalable Matrix Extension",
        since="Armv9.2-A",
        relevance=Relevance.HIGH,
        summary=(
            "A streaming execution mode with a dedicated 2D accumulator tile "
            "(ZA) for outer-product matrix multiply."
        ),
        inference_impact=(
            "A genuine matrix engine on the CPU rather than a wider vector "
            "unit. Only exploited when the runtime is built against kernels "
            "that target it -- KleidiAI in llama.cpp being the main example -- "
            "so its presence is necessary but not sufficient for a speedup."
        ),
    ),
    "sme2": FeatureInfo(
        key="sme2",
        arm_name="FEAT_SME2",
        title="Scalable Matrix Extension 2",
        since="Armv9.2-A",
        relevance=Relevance.HIGH,
        summary="Adds multi-vector operations and a lookup table to SME.",
        inference_impact=(
            "The version KleidiAI's matmul micro-kernels actually target. "
            "Present on Apple M4 and on Armv9.2 mobile cores."
        ),
    ),
    "sme_i8i32": FeatureInfo(
        key="sme_i8i32",
        arm_name="SME_I8I32",
        title="SME int8 to int32 accumulation",
        since="Armv9.2-A",
        relevance=Relevance.HIGH,
        summary="Int8 outer products accumulating into 32-bit tiles.",
        inference_impact=(
            "The specific SME data path that quantised LLM weights use. "
            "Without it, SME is only useful for floating-point work."
        ),
    ),
    "sme_f32f32": FeatureInfo(
        key="sme_f32f32",
        arm_name="SME_F32F32",
        title="SME fp32 accumulation",
        since="Armv9.2-A",
        relevance=Relevance.MODERATE,
        summary="Single-precision outer products in streaming mode.",
        inference_impact="Relevant for unquantised or partially quantised layers.",
    ),
}


# --------------------------------------------------------------------------
# OS-reported name -> normalised key
# --------------------------------------------------------------------------

#: macOS ``sysctl hw.optional.arm.*`` leaf names.
DARWIN_FEATURE_MAP: dict[str, str] = {
    "AdvSIMD": "neon",
    "FEAT_FP16": "fp16",
    "FEAT_DotProd": "dotprod",
    "FEAT_I8MM": "i8mm",
    "FEAT_BF16": "bf16",
    "FEAT_SVE": "sve",
    "FEAT_SVE2": "sve2",
    "FEAT_SME": "sme",
    "FEAT_SME2": "sme2",
}

#: macOS reports some SME data types outside the ``FEAT_`` namespace.
DARWIN_BARE_FEATURE_MAP: dict[str, str] = {
    "SME_I8I32": "sme_i8i32",
    "SME_F32F32": "sme_f32f32",
    "neon": "neon",
}

#: Linux ``/proc/cpuinfo`` HWCAP flag names.
LINUX_FEATURE_MAP: dict[str, str] = {
    "asimd": "neon",
    "asimdhp": "fp16",
    "fphp": "fp16",
    "asimddp": "dotprod",
    "i8mm": "i8mm",
    "bf16": "bf16",
    "sve": "sve",
    "sve2": "sve2",
    "svei8mm": "svei8mm",
    "svebf16": "svebf16",
    "sme": "sme",
    "sme2": "sme2",
    "smei8i32": "sme_i8i32",
    "smef32f32": "sme_f32f32",
}


# --------------------------------------------------------------------------
# MIDR_EL1 decoding
# --------------------------------------------------------------------------

IMPLEMENTERS: dict[int, str] = {
    0x41: "Arm",
    0x42: "Broadcom",
    0x43: "Cavium",
    0x48: "HiSilicon",
    0x4E: "NVIDIA",
    0x50: "Ampere",
    0x51: "Qualcomm",
    0x61: "Apple",
    0xC0: "Ampere",
}

#: (implementer, part number) -> microarchitecture name.
#: Restricted to parts that plausibly run LLM inference.
CORE_PARTS: dict[tuple[int, int], str] = {
    # Arm Neoverse (server / cloud)
    (0x41, 0xD0C): "Neoverse-N1",
    (0x41, 0xD40): "Neoverse-V1",
    (0x41, 0xD49): "Neoverse-N2",
    (0x41, 0xD4F): "Neoverse-V2",
    (0x41, 0xD8E): "Neoverse-N3",
    (0x41, 0xD84): "Neoverse-V3",
    # Arm Cortex-A "big" cores
    (0x41, 0xD0B): "Cortex-A76",
    (0x41, 0xD0D): "Cortex-A77",
    (0x41, 0xD41): "Cortex-A78",
    (0x41, 0xD47): "Cortex-A710",
    (0x41, 0xD4D): "Cortex-A715",
    (0x41, 0xD81): "Cortex-A720",
    # Arm Cortex-X
    (0x41, 0xD44): "Cortex-X1",
    (0x41, 0xD48): "Cortex-X2",
    (0x41, 0xD4E): "Cortex-X3",
    (0x41, 0xD82): "Cortex-X4",
    # Arm Cortex-A "LITTLE" cores
    (0x41, 0xD03): "Cortex-A53",
    (0x41, 0xD05): "Cortex-A55",
    (0x41, 0xD46): "Cortex-A510",
    (0x41, 0xD80): "Cortex-A520",
    # Other implementers
    (0x50, 0x000): "Ampere-eMAG",
    (0xC0, 0xAC3): "Ampere-1",
    (0xC0, 0xAC4): "Ampere-1a",
    (0x4E, 0x004): "NVIDIA-Carmel",
}

#: Cores we know are designed as efficiency ("LITTLE") cores.
EFFICIENCY_PARTS: frozenset[str] = frozenset(
    {"Cortex-A53", "Cortex-A55", "Cortex-A510", "Cortex-A520"}
)


def decode_midr(midr: int) -> tuple[str | None, str | None]:
    """Decode a raw ``MIDR_EL1`` value into (implementer, core name).

    Either element is ``None`` when the value is not in our table; we return
    the unknown rather than inventing a plausible-sounding name.
    """
    implementer_id = (midr >> 24) & 0xFF
    part_id = (midr >> 4) & 0xFFF
    implementer = IMPLEMENTERS.get(implementer_id)
    core_name = CORE_PARTS.get((implementer_id, part_id))
    return implementer, core_name


def describe(key: str) -> FeatureInfo | None:
    """Look up a normalised feature key."""
    return FEATURES.get(key)


def relevant_present(features: frozenset[str]) -> list[FeatureInfo]:
    """Known, inference-relevant features present on this machine, ranked."""
    order = {
        Relevance.CRITICAL: 0,
        Relevance.HIGH: 1,
        Relevance.MODERATE: 2,
        Relevance.LOW: 3,
    }
    present = [FEATURES[k] for k in features if k in FEATURES]
    return sorted(present, key=lambda f: (order[f.relevance], f.key))


def notable_absent(features: frozenset[str], *, is_arm64: bool) -> list[FeatureInfo]:
    """Critical/high-value features this machine lacks.

    Absence is as informative as presence: a core without ``i8mm`` should not
    be handed a weight layout that exists only to feed ``SMMLA``.
    """
    if not is_arm64:
        return []
    interesting = {Relevance.CRITICAL, Relevance.HIGH}
    return [
        info
        for key, info in sorted(FEATURES.items())
        if key not in features and info.relevance in interesting
    ]
