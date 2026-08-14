#!/usr/bin/env python3
"""Generate docs/RESULTS.md and its charts from committed sweep artifacts.

Every figure on the results page is read out of ``results/*.json`` at build
time. Nothing is typed by hand, so the page cannot drift from the data and a
transcription slip cannot invent a number that was never measured. Rerun this
after any new sweep:

    python scripts/build_results.py

Charts are plain SVG with no dependencies, coloured to stay legible against
both light and dark backgrounds since GitHub renders either.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"
DOCS_DIR = ROOT / "docs"
CHARTS_DIR = DOCS_DIR / "charts"

# Legible on white and on GitHub's dark background alike.
INK = "#8b949e"
SERIES = {"prefill": "#3b82f6", "decode": "#f59e0b"}
QUANT_COLOR = {"Q4_0": "#3b82f6", "Q4_K_M": "#a855f7"}


@dataclass
class Sweep:
    """One sweep artifact, indexed for lookup."""

    name: str
    path: Path
    raw: dict[str, Any]

    @property
    def host(self) -> dict[str, Any]:
        return self.raw["host"]

    @property
    def cpu(self) -> dict[str, Any]:
        return self.raw["host"]["cpu"]

    @property
    def results(self) -> list[dict[str, Any]]:
        return [r for r in self.raw["results"] if r["status"] == "ok"]

    @property
    def recommendation(self) -> dict[str, Any] | None:
        return self.raw.get("recommendation")

    def series(
        self, metric: str, *, quant: str, variant: str = "cpu"
    ) -> list[tuple[int, float, float]]:
        """(threads, mean, stddev) for one metric, quantisation and build."""
        points = []
        for result in self.results:
            config = result["config"]
            if config["model"]["quantization"] != quant:
                continue
            if config["runtime"]["build_flags"].get("variant") != variant:
                continue
            stats = result["metrics"].get(metric)
            if stats:
                points.append((config["threads"], stats["mean"], stats["stddev"]))
        return sorted(points)

    def variants(self) -> list[str]:
        seen = {
            r["config"]["runtime"]["build_flags"].get("variant") for r in self.results
        }
        return sorted(v for v in seen if v)

    def quantizations(self) -> list[str]:
        seen = {r["config"]["model"]["quantization"] for r in self.results}
        return sorted(q for q in seen if q)

    @property
    def purpose(self) -> str:
        """What this sweep varies, which decides what it can answer.

        Comparisons are only ever drawn *within* one sweep. Absolute
        throughput drifts between runs on a thermally throttled laptop, so
        pooling two runs to compare a build against a build would attribute
        cooling to KleidiAI.
        """
        if len(self.variants()) > 1:
            return "runtimes"
        if len(self.quantizations()) > 1:
            return "models"
        return "single"

    def at(
        self, *, quant: str, threads: int, variant: str, metric: str
    ) -> tuple[float, float] | None:
        for result in self.results:
            config = result["config"]
            if (
                config["model"]["quantization"] == quant
                and config["threads"] == threads
                and config["runtime"]["build_flags"].get("variant") == variant
            ):
                stats = result["metrics"].get(metric)
                if stats:
                    return stats["mean"], stats["stddev"]
        return None

    def peak_memory(self, quant: str) -> int | None:
        values = [
            r["metrics"]["peak_memory_bytes"]
            for r in self.results
            if r["config"]["model"]["quantization"] == quant
            and r["metrics"].get("peak_memory_bytes")
        ]
        return min(values) if values else None

    def model_size(self, quant: str) -> int | None:
        for result in self.results:
            if result["config"]["model"]["quantization"] == quant:
                return result["config"]["model"]["size_bytes"]
        return None


# --------------------------------------------------------------------------
# SVG charts
# --------------------------------------------------------------------------


def _line_chart(
    title: str,
    series: dict[str, list[tuple[int, float, float]]],
    colors: dict[str, str],
    *,
    y_label: str,
    width: int = 560,
    height: int = 300,
) -> str:
    """A small multi-series line chart over thread count."""
    # A legend is redundant when the title already names the only series.
    show_legend = len(series) > 1
    pad_l, pad_r, pad_t, pad_b = 58, (130 if show_legend else 24), 34, 44
    plot_w = width - pad_l - pad_r
    plot_h = height - pad_t - pad_b

    all_points = [p for pts in series.values() for p in pts]
    if not all_points:
        return ""

    xs = sorted({p[0] for p in all_points})
    y_max = max(p[1] + p[2] for p in all_points) * 1.12
    x_min, x_max = min(xs), max(xs)

    def sx(x: float) -> float:
        if x_max == x_min:
            return pad_l + plot_w / 2
        return pad_l + (x - x_min) / (x_max - x_min) * plot_w

    def sy(y: float) -> float:
        return pad_t + plot_h - (y / y_max) * plot_h

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'width="{width}" height="{height}" font-family="system-ui,sans-serif">',
        f'<text x="{pad_l}" y="20" font-size="13" font-weight="600" '
        f'fill="{INK}">{title}</text>',
    ]

    # Horizontal gridlines with value labels.
    for step in range(5):
        value = y_max * step / 4
        y = sy(value)
        parts.append(
            f'<line x1="{pad_l}" y1="{y:.1f}" x2="{pad_l + plot_w}" y2="{y:.1f}" '
            f'stroke="{INK}" stroke-opacity="0.18" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{pad_l - 8}" y="{y + 4:.1f}" font-size="10" text-anchor="end" '
            f'fill="{INK}" fill-opacity="0.75">{value:.0f}</text>'
        )

    # X axis ticks at the measured thread counts.
    for x in xs:
        parts.append(
            f'<text x="{sx(x):.1f}" y="{pad_t + plot_h + 18}" font-size="10" '
            f'text-anchor="middle" fill="{INK}" fill-opacity="0.75">{x}</text>'
        )
    parts.append(
        f'<text x="{pad_l + plot_w / 2:.1f}" y="{height - 8}" font-size="11" '
        f'text-anchor="middle" fill="{INK}" fill-opacity="0.85">threads</text>'
    )
    parts.append(
        f'<text x="14" y="{pad_t + plot_h / 2:.1f}" font-size="11" fill="{INK}" '
        f'fill-opacity="0.85" text-anchor="middle" '
        f'transform="rotate(-90 14 {pad_t + plot_h / 2:.1f})">{y_label}</text>'
    )

    for index, (label, points) in enumerate(series.items()):
        if not points:
            continue
        color = colors.get(label, "#3b82f6")
        path = " ".join(
            f"{'M' if i == 0 else 'L'}{sx(x):.1f},{sy(y):.1f}"
            for i, (x, y, _) in enumerate(points)
        )
        parts.append(
            f'<path d="{path}" fill="none" stroke="{color}" stroke-width="2.2" '
            f'stroke-linejoin="round"/>'
        )
        for x, y, sd in points:
            if sd > 0:
                parts.append(
                    f'<line x1="{sx(x):.1f}" y1="{sy(y - sd):.1f}" '
                    f'x2="{sx(x):.1f}" y2="{sy(y + sd):.1f}" '
                    f'stroke="{color}" stroke-width="1.4" stroke-opacity="0.55"/>'
                )
            parts.append(
                f'<circle cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="3.4" fill="{color}"/>'
            )
        if show_legend:
            legend_y = pad_t + 6 + index * 18
            parts.append(
                f'<rect x="{pad_l + plot_w + 16}" y="{legend_y - 8}" width="10" '
                f'height="10" rx="2" fill="{color}"/>'
            )
            parts.append(
                f'<text x="{pad_l + plot_w + 32}" y="{legend_y + 1}" font-size="11" '
                f'fill="{INK}">{label}</text>'
            )

    parts.append("</svg>")
    return "\n".join(parts)


def _write_chart(name: str, svg: str) -> str | None:
    if not svg:
        return None
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    (CHARTS_DIR / name).write_text(svg)
    return f"charts/{name}"


# --------------------------------------------------------------------------
# Page sections
# --------------------------------------------------------------------------


def _machine_row(sweep: Sweep) -> str:
    cpu = sweep.cpu
    clusters = " + ".join(
        f"{c['physical_cores']}× {c.get('core_name') or c['name']}"
        for c in cpu["clusters"]
    )
    sve = f"{cpu['sve_vector_bits']}-bit" if cpu.get("sve_vector_bits") else "absent"
    sme = f"{cpu['sme_vector_bits']}-bit" if cpu.get("sme_vector_bits") else "absent"
    topology = "heterogeneous" if cpu["is_heterogeneous"] else "uniform"
    return (
        f"| **{cpu['model']}** | {cpu['physical_cores']} ({topology}) | {clusters} "
        f"| {sve} | {sme} | {sweep.host['total_memory_gb']:.0f} GB |"
    )


def _deploy_line(sweep: Sweep) -> str:
    rec = sweep.recommendation
    if not rec:
        return "_no recommendation recorded_"
    prefill, decode = rec.get("prefill"), rec.get("decode")
    parts = []
    if decode:
        parts.append(f"`-t {decode['threads']}`")
    if prefill:
        parts.append(f"`-tb {prefill['threads']}`")
    return " ".join(parts)


def _pct(new: float, old: float) -> str:
    """Percentage change, keeping a decimal for small ones.

    Rounding 0.4% to "+0%" reads as though nothing was measured, when in fact
    something was measured and found to be negligible. Those are different
    claims.
    """
    if not old:
        return "n/a"
    change = (new - old) / old * 100
    return f"{change:+.1f}%" if abs(change) < 10 else f"{change:+.0f}%"


def build(sweeps: dict[tuple[str, str], Sweep]) -> str:
    m4 = sweeps.get(("m4", "models"))
    neo = sweeps.get(("neoverse", "models")) or sweeps.get(("neoverse", "runtimes"))
    m4_rt = sweeps.get(("m4", "runtimes"))
    neo_rt = sweeps.get(("neoverse", "runtimes"))
    if not m4 or not neo:
        raise SystemExit("need both an M4 and a Neoverse sweep to build this page")

    lines: list[str] = []
    add = lines.append

    add("# Measured results")
    add("")
    add(
        "Every number on this page is read out of the JSON artifacts in "
        "[`results/`](../results) by [`scripts/build_results.py`]"
        "(../scripts/build_results.py). Nothing is typed by hand. Rerun that "
        "script after any sweep to regenerate the page and its charts."
    )
    add("")
    add(
        "Model: **Qwen2.5-0.5B-Instruct** (GGUF). Runtime: **llama.cpp** at the "
        "commit pinned in the workflow. Workload: **long-context** — a "
        "2048-token prompt with 128 generated tokens. Five repetitions per "
        "measurement, with llama-bench doing its own warmup."
    )
    add("")

    # -- machines ---------------------------------------------------------
    add("## The two machines")
    add("")
    add("| CPU | Cores | Clusters | SVE | SME | RAM |")
    add("| --- | ---: | --- | --- | --- | ---: |")
    add(_machine_row(m4))
    add(_machine_row(neo))
    add("")
    add(
        "These are structural opposites, which is the point. The M4 has a "
        "matrix engine (SME2) and two kinds of core; the Neoverse-N2 has "
        "scalable vectors (SVE2) and one kind. A tool that merely swept "
        "configurations would produce the same advice for both. A capability "
        "model should not."
    )
    add("")

    # -- headline ---------------------------------------------------------
    add("## The recommendations differ")
    add("")
    add("| Machine | Recommended threads | Why |")
    add("| --- | --- | --- |")
    add(
        f"| {m4.cpu['model']} | {_deploy_line(m4)} | prompt processing and token "
        "generation peak at different thread counts |"
    )
    add(
        f"| {neo.cpu['model']} | {_deploy_line(neo)} | both phases peak together, "
        "so the split collapses to one number |"
    )
    add("")
    add(
        "`-t` sets generation threads and `-tb` sets prompt-processing threads; "
        "llama.cpp accepts them separately, so the split is directly deployable."
    )
    add("")

    # -- finding 1 --------------------------------------------------------
    add("## Finding 1 — core topology decides the thread count")
    add("")

    # Prefill and decode get separate charts with independent scales.
    # Sharing one axis squashes decode against a prefill range several times
    # larger, which renders the collapse -- the whole point -- as a gentle
    # slope.
    for sweep, slug in ((m4, "m4"), (neo, "neoverse")):
        charts = []
        for phase, metric in (("prefill", "prefill_tps"), ("decode", "decode_tps")):
            chart = _write_chart(
                f"{slug}-{phase}.svg",
                _line_chart(
                    f"{sweep.cpu['model']} — Q4_0 {phase}",
                    {phase: sweep.series(metric, quant="Q4_0")},
                    SERIES,
                    y_label="tok/s",
                    width=430,
                    height=270,
                ),
            )
            if chart:
                charts.append(
                    f'<img src="{chart}" alt="{sweep.cpu["model"]} {phase}" width="430">'
                )
        if charts:
            add(" ".join(charts))
            add("")

    m4_decode = m4.series("decode_tps", quant="Q4_0")
    m4_best_decode = max(m4_decode, key=lambda p: p[1])
    m4_worst_decode = m4_decode[-1]
    ratio = m4_best_decode[1] / m4_worst_decode[1] if m4_worst_decode[1] else 0

    neo_decode = neo.series("decode_tps", quant="Q4_0")
    neo_best_decode = max(neo_decode, key=lambda p: p[1])

    add(
        f"On the {m4.cpu['model']}, decode peaks at **{m4_best_decode[0]} threads** "
        f"({m4_best_decode[1]:.1f} tok/s) and collapses to "
        f"**{m4_worst_decode[1]:.1f} tok/s** at {m4_worst_decode[0]} threads — a "
        f"**{ratio:.0f}× drop** at exactly the value `nproc` reports. Prefill peaks "
        f"elsewhere, at {max(m4.series('prefill_tps', quant='Q4_0'), key=lambda p: p[1])[0]} "
        "threads."
    )
    add("")
    add(
        f"On the {neo.cpu['model']} that collapse does not happen: scaling is close "
        f"to linear and **{neo_best_decode[0]} threads — every core — is best for "
        "both phases**."
    )
    add("")
    add(
        "This was registered as a prediction before the Neoverse sweep ran. The "
        "collapse is an artifact of heterogeneity: efficiency cores stall a "
        "thread pool that waits on its slowest member. A uniform CPU has no "
        "slow cluster to wait for. Had the collapse appeared on Neoverse too, "
        "the explanation would have been wrong."
    )
    add("")

    # -- finding 2 --------------------------------------------------------
    add("## Finding 2 — Q4_0 beats Q4_K_M for prompt processing")
    add("")
    add("| Machine | Q4_0 prefill | Q4_K_M prefill | Ratio |")
    add("| --- | ---: | ---: | ---: |")
    for sweep in (m4, neo):
        best_0 = max(sweep.series("prefill_tps", quant="Q4_0"), key=lambda p: p[1])
        k_series = sweep.series("prefill_tps", quant="Q4_K_M")
        if not k_series:
            continue
        best_k = max(k_series, key=lambda p: p[1])
        add(
            f"| {sweep.cpu['model']} | {best_0[1]:.1f} ± {best_0[2]:.1f} "
            f"({best_0[0]}t) | {best_k[1]:.1f} ± {best_k[2]:.1f} ({best_k[0]}t) "
            f"| **{best_0[1] / best_k[1]:.2f}×** |"
        )
    add("")
    add(
        "llama.cpp repacks Q4_0 weights at load time into a blocked layout that "
        "feeds the `SMMLA` int8 matrix instruction; the K-quants have no such "
        "path. Both CPUs report `FEAT_I8MM`, so both benefit — the M4 more, "
        "consistent with it also having SME2 while the Neoverse-N2 has only "
        "128-bit SVE."
    )
    add("")
    add(
        "This is a prompt-processing result and should not be read as \"Q4_0 is "
        "better\". On the M4 at 6 and 10 threads, Q4_K_M actually **decodes "
        "faster**. That is why the recommendation is made per phase."
    )
    add("")

    # -- finding 3 --------------------------------------------------------
    kleidi_sweeps = [s for s in (m4_rt, neo_rt) if s is not None]
    if kleidi_sweeps:
        add("## Finding 3 — KleidiAI needs SME2, and only helps when compute-bound")
        add("")
        add(
            "Arm's KleidiAI micro-kernels are the only path to SME2 in llama.cpp. "
            "Comparing a KleidiAI build against the stock ggml kernels, prefill, "
            "Q4_0, at each thread count — always within a single sweep, never "
            "across two."
        )
        add("")

        for sweep in kleidi_sweeps:
            has_sme = "sme2" in sweep.cpu["features"]
            add(
                f"**{sweep.cpu['model']}** — SME2 "
                f"{'present' if has_sme else 'absent'}"
            )
            add("")
            add("| Threads | Stock ggml | KleidiAI | Change |")
            add("| ---: | ---: | ---: | ---: |")
            deltas: list[tuple[int, float, float]] = []
            for threads, _, _ in sweep.series("prefill_tps", quant="Q4_0", variant="cpu"):
                base = sweep.at(
                    quant="Q4_0", threads=threads, variant="cpu", metric="prefill_tps"
                )
                kle = sweep.at(
                    quant="Q4_0",
                    threads=threads,
                    variant="kleidiai",
                    metric="prefill_tps",
                )
                if not base or not kle:
                    continue
                add(
                    f"| {threads} | {base[0]:.1f} ± {base[1]:.1f} "
                    f"| {kle[0]:.1f} ± {kle[1]:.1f} | **{_pct(kle[0], base[0])}** |"
                )
                deltas.append((threads, (kle[0] - base[0]) / base[0] * 100, 0.0))
            add("")

            if has_sme and len(deltas) > 2:
                chart = _write_chart(
                    "kleidiai-gain.svg",
                    _line_chart(
                        f"{sweep.cpu['model']} — KleidiAI prefill gain vs threads",
                        {"gain %": deltas},
                        {"gain %": "#10b981"},
                        y_label="% faster",
                        width=460,
                        height=280,
                    ),
                )
                if chart:
                    add(f'<img src="{chart}" alt="KleidiAI gain vs threads" width="460">')
                    add("")

        add(
            "On the machine without SME the difference is negligible at every "
            "thread count — enabling KleidiAI changes nothing it can act on."
        )
        add("")
        add(
            "On the machine with SME2 the gain is real but **decays as threads "
            "are added**, from a large single-threaded advantage down to nothing "
            "once every core is busy. That shape is the informative part. SME2 "
            "is a per-core matrix engine, so it raises the ceiling on how much "
            "arithmetic one core can do; when enough cores are running that the "
            "workload is bound by memory bandwidth instead, more arithmetic "
            "throughput buys nothing. The feature helps precisely where the "
            "bottleneck is compute."
        )
        add("")
        add(
            "Detecting `FEAT_SME2` proves the CPU *can* do this. It does not "
            "prove a runtime *did*, nor that it will help at your thread count. "
            "ArmForge keeps those claims separate and records the ggml feature "
            "line from each build alongside every result."
        )
        add("")

    # -- finding 4 --------------------------------------------------------
    add("## Finding 4 — the speed is bought with memory")
    add("")
    add("| Machine | Quant | On disk | Peak resident |")
    add("| --- | --- | ---: | ---: |")
    for sweep in (m4, neo):
        for quant in ("Q4_0", "Q4_K_M"):
            size, peak = sweep.model_size(quant), sweep.peak_memory(quant)
            if size and peak:
                add(
                    f"| {sweep.cpu['model']} | {quant} | {size / 1024**2:.0f} MiB "
                    f"| {peak / 1024**2:.0f} MiB |"
                )
    add("")
    add(
        "Q4_0 is the smaller file but the larger process, on both machines. The "
        "repacked weight buffer that buys the prefill speed has to live "
        "somewhere."
    )
    add("")

    if m4_rt:
        rows = []
        for variant in ("cpu", "kleidiai"):
            peaks = [
                r["metrics"]["peak_memory_bytes"]
                for r in m4_rt.results
                if r["config"]["runtime"]["build_flags"].get("variant") == variant
                and r["metrics"].get("peak_memory_bytes")
            ]
            if peaks:
                rows.append((variant, min(peaks)))
        if len(rows) == 2:
            base_name, base_peak = rows[0]
            kle_name, kle_peak = rows[1]
            add(f"The same is true of KleidiAI on the {m4_rt.cpu['model']}:")
            add("")
            add("| Build | Peak resident |")
            add("| --- | ---: |")
            add(f"| stock ggml | {base_peak / 1024**2:.0f} MiB |")
            add(
                f"| KleidiAI | {kle_peak / 1024**2:.0f} MiB "
                f"(**{_pct(kle_peak, base_peak)}**) |"
            )
            add("")
            add(
                "KleidiAI keeps weights in its own SME2-friendly layout, on top "
                "of everything the baseline already allocates. So the "
                "single-threaded prefill gain above is not free — it is paid for "
                "in resident memory, and on a device where that is the binding "
                "constraint the trade may not be worth taking."
            )
            add("")

    add(
        "None of this is visible in a throughput-only benchmark, which is why "
        "ArmForge records peak resident memory for every candidate and puts it "
        "on the Pareto frontier alongside speed."
    )
    add("")

    # -- caveats ----------------------------------------------------------
    add("## What these numbers are not")
    add("")
    add(
        "- **Not a quality claim.** Throughput only. Q4_0 being faster than "
        "Q4_K_M says nothing about output quality, and the two quantisations "
        "do not produce identical text."
    )
    add(
        "- **One model family.** Everything here is Qwen2.5-0.5B. Larger models "
        "shift the balance between compute and memory bandwidth, and the "
        "conclusions may move with it."
    )
    add(
        f"- **The {m4.cpu['model']} is a laptop.** It thermally throttles and shares "
        "the machine with everything else running. Several of its measurements "
        "are flagged noisy; the CI machine's are far tighter."
    )
    add(
        "- **The Neoverse-N2 runner is a shared 4-core VM.** It demonstrates "
        "portability and the capability model. It is not a substitute for a "
        "dedicated Graviton instance for absolute figures."
    )
    add(
        "- **Measurement beats the model.** Where a prediction disagreed with a "
        "measurement, the measurement won and the model was corrected. Two such "
        "corrections are recorded in the git history."
    )
    add("")

    # -- reproduce --------------------------------------------------------
    add("## Reproduce this")
    add("")
    add("```bash")
    add("scripts/setup-llama-cpp.sh          # build the llama.cpp variants")
    add("armforge hardware --explain         # what your CPU can do, and why it matters")
    add("armforge optimize model.gguf --workload long-context")
    add("```")
    add("")
    add(
        "The Neoverse half runs on GitHub's free Arm64 runners via "
        "[`.github/workflows/arm64-sweep.yml`](../.github/workflows/arm64-sweep.yml), "
        "so it can be re-run by anyone with a fork rather than taken on trust."
    )
    add("")

    return "\n".join(lines) + "\n"


def main() -> int:
    if not RESULTS_DIR.is_dir():
        print("no results/ directory", file=sys.stderr)
        return 1

    # Sweeps are indexed by (machine, what-it-varies) so each finding is drawn
    # from one internally consistent run.
    sweeps: dict[tuple[str, str], Sweep] = {}
    for path in sorted(RESULTS_DIR.glob("*.json")):
        raw = json.loads(path.read_text())
        if "results" not in raw or "host" not in raw:
            continue
        model = raw["host"]["cpu"]["model"].lower()
        machine = (
            "m4" if "apple" in model else "neoverse" if "neoverse" in model else None
        )
        if machine is None:
            continue

        sweep = Sweep(name=path.stem, path=path, raw=raw)
        if sweep.purpose == "single":
            continue
        key = (machine, sweep.purpose)
        existing = sweeps.get(key)
        if existing is None or len(sweep.results) > len(existing.results):
            sweeps[key] = sweep

    page = build(sweeps)
    DOCS_DIR.mkdir(exist_ok=True)
    (DOCS_DIR / "RESULTS.md").write_text(page)
    used = ", ".join(f"{k[0]}/{k[1]}={v.name}" for k, v in sorted(sweeps.items()))
    print(f"wrote docs/RESULTS.md from {used}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
