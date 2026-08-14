"""Standard benchmark workloads.

Each workload is a *shape*: how many tokens of prompt to process and how many
to generate. The shapes are chosen to separate the two phases of inference
that respond to completely different hardware properties.

No text corpus is involved, which keeps every run reproducible and avoids any
dataset licensing question. These measure speed at a given shape; they say
nothing about output quality.
"""

from __future__ import annotations

from .types import Workload

SHORT = Workload(
    name="short",
    prompt_tokens=128,
    generate_tokens=64,
    description=(
        "A short chat turn. Decode-dominated, so it mostly measures memory "
        "bandwidth and is the shape most sensitive to thread oversubscription."
    ),
)

LONG_CONTEXT = Workload(
    name="long-context",
    prompt_tokens=2048,
    generate_tokens=128,
    description=(
        "A document-sized prompt with a short answer. Prefill-dominated, so it "
        "is where int8 matrix instructions such as SMMLA pay off most."
    ),
)

CODE = Workload(
    name="code",
    prompt_tokens=512,
    generate_tokens=256,
    description=(
        "Code completion: moderate context, long generation. Balanced between "
        "the prefill and decode phases."
    ),
)

SUMMARIZE = Workload(
    name="summarize",
    prompt_tokens=1024,
    generate_tokens=128,
    description="Summarisation: large prompt, compact output. Prefill-leaning.",
)

#: Every workload ArmForge ships, keyed by name.
WORKLOADS: dict[str, Workload] = {w.name: w for w in (SHORT, LONG_CONTEXT, CODE, SUMMARIZE)}

#: Default sweep: one decode-dominated and one prefill-dominated shape.
#: Running both is what reveals that the optimal thread count differs between
#: the two phases.
DEFAULT_WORKLOADS: tuple[Workload, ...] = (SHORT, LONG_CONTEXT)


def get(name: str) -> Workload:
    """Look up a workload by name, with a helpful error listing the options."""
    try:
        return WORKLOADS[name]
    except KeyError:
        options = ", ".join(sorted(WORKLOADS))
        raise KeyError(f"unknown workload {name!r}; available: {options}") from None
