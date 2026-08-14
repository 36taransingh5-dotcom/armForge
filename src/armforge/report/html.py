"""A self-contained HTML benchmark report.

No external requests: styles are inline, charts are inline SVG, there are no
fonts or scripts to fetch. The file works opened from disk, attached to an
email, or served from a container with nothing else present.
"""

from __future__ import annotations

import html
from typing import TYPE_CHECKING

from ..bench.types import BenchmarkResult, Status
from .charts import line_chart

if TYPE_CHECKING:  # pragma: no cover
    from ..optimize.recommend import Recommendation
    from ..optimize.sweep import SweepReport

_STYLE = """
:root {
  --bg: #ffffff; --fg: #1f2328; --muted: #656d76; --line: #d0d7de;
  --accent: #0969da; --warn: #9a6700; --warn-bg: #fff8c5; --code-bg: #f6f8fa;
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg: #0d1117; --fg: #e6edf3; --muted: #8b949e; --line: #30363d;
    --accent: #4493f8; --warn: #d29922; --warn-bg: #2d2205; --code-bg: #161b22;
  }
}
* { box-sizing: border-box; }
body {
  margin: 0; padding: 2.5rem 1.25rem; background: var(--bg); color: var(--fg);
  font: 15px/1.6 system-ui, -apple-system, "Segoe UI", sans-serif;
}
main { max-width: 62rem; margin: 0 auto; }
h1 { font-size: 1.6rem; margin: 0 0 .25rem; }
h2 { font-size: 1.15rem; margin: 2.5rem 0 .75rem; padding-bottom: .3rem;
     border-bottom: 1px solid var(--line); }
h3 { font-size: .95rem; margin: 1.5rem 0 .5rem; color: var(--muted);
     text-transform: uppercase; letter-spacing: .04em; }
.sub { color: var(--muted); margin: 0 0 2rem; }
table { border-collapse: collapse; width: 100%; margin: .5rem 0 1rem;
        font-variant-numeric: tabular-nums; }
th, td { padding: .45rem .7rem; border-bottom: 1px solid var(--line);
         text-align: left; }
th { font-weight: 600; color: var(--muted); font-size: .82rem;
     text-transform: uppercase; letter-spacing: .03em; }
td.num, th.num { text-align: right; }
tr.best td { background: color-mix(in srgb, var(--accent) 10%, transparent); }
code, pre { font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
            font-size: .85em; }
pre { background: var(--code-bg); border: 1px solid var(--line);
      border-radius: 6px; padding: .8rem 1rem; overflow-x: auto; }
.grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(13rem, 1fr));
        gap: .75rem 1.5rem; margin: 1rem 0; }
.kv .k { color: var(--muted); font-size: .8rem; text-transform: uppercase;
         letter-spacing: .03em; }
.kv .v { font-size: 1.05rem; font-weight: 600; }
.caveat { background: var(--warn-bg); border-left: 3px solid var(--warn);
          padding: .7rem 1rem; margin: .5rem 0; border-radius: 0 6px 6px 0; }
.charts { display: flex; flex-wrap: wrap; gap: 1rem; margin: 1rem 0; }
.charts svg { max-width: 100%; height: auto; }
.noisy { color: var(--warn); }
.footer { margin-top: 3rem; padding-top: 1rem; border-top: 1px solid var(--line);
          color: var(--muted); font-size: .85rem; }
.status { font-style: italic; color: var(--muted); }
"""


def _e(value: object) -> str:
    return html.escape(str(value), quote=True)


def _metric(stats) -> str:
    """Format a measurement, flagging run-to-run noise above 5%."""
    if stats is None:
        return '<span class="status">not measured</span>'
    text = f"{stats.mean:.1f} ± {stats.stddev:.1f}"
    if stats.relative_stddev > 0.05:
        return f'<span class="noisy">{text} ⚠</span>'
    return text


def _bytes(value: int | None) -> str:
    if not value:
        return "—"
    return f"{value / 1024**2:.0f} MiB"


def _measurements_table(results: list[BenchmarkResult], winner_label: str | None) -> str:
    rows = []
    for result in results:
        label = result.config.label
        if result.status is not Status.OK:
            rows.append(
                f"<tr><td>{_e(label)}</td>"
                f'<td colspan="3" class="status">{_e(result.status.value)}'
                f"{': ' + _e(result.error) if result.error else ''}</td></tr>"
            )
            continue
        css = ' class="best"' if label == winner_label else ""
        rows.append(
            f"<tr{css}><td>{_e(label)}</td>"
            f'<td class="num">{_metric(result.prefill_tps)}</td>'
            f'<td class="num">{_metric(result.decode_tps)}</td>'
            f'<td class="num">{_bytes(result.peak_memory_bytes)}</td></tr>'
        )

    return (
        "<table><thead><tr><th>Configuration</th>"
        '<th class="num">Prefill tok/s</th><th class="num">Decode tok/s</th>'
        '<th class="num">Peak memory</th></tr></thead><tbody>'
        + "".join(rows)
        + "</tbody></table>"
    )


def _phase_charts(report: SweepReport, quant: str | None, variant: str | None) -> str:
    """Prefill and decode curves for the winning model and runtime.

    Plotted separately: sharing an axis squashes decode against a prefill
    range several times larger and hides the shape that matters.
    """
    series: dict[str, list[tuple[float, float, float]]] = {
        "prefill": [],
        "decode": [],
    }
    for result in report.results:
        if not result.ok:
            continue
        config = result.config
        if quant and config.model.quantization != quant:
            continue
        if variant and config.runtime.build_flags.get("variant") != variant:
            continue
        if result.prefill_tps:
            series["prefill"].append(
                (config.threads, result.prefill_tps.mean, result.prefill_tps.stddev)
            )
        if result.decode_tps:
            series["decode"].append(
                (config.threads, result.decode_tps.mean, result.decode_tps.stddev)
            )

    charts = []
    for phase in ("prefill", "decode"):
        svg = line_chart(
            f"{phase.capitalize()} vs threads",
            {phase: series[phase]},
            y_label="tok/s",
            width=430,
            height=270,
        )
        if svg:
            charts.append(svg)
    return f'<div class="charts">{"".join(charts)}</div>' if charts else ""


def render(report: SweepReport, recommendation: Recommendation | None) -> str:
    """Build the full HTML report."""
    host = report.host
    cpu = host.cpu

    winner_label = recommendation.winner.label if recommendation else None
    winning = recommendation.winner.result.config if recommendation else None

    topology = "heterogeneous" if cpu.is_heterogeneous else "uniform"
    clusters = ", ".join(f"{c.physical_cores}× {c.core_name or c.name}" for c in cpu.clusters)
    features = ", ".join(sorted(cpu.features)) or "none detected"

    parts: list[str] = [
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>",
        "<meta name='viewport' content='width=device-width,initial-scale=1'>",
        f"<title>ArmForge report — {_e(cpu.model)}</title>",
        f"<style>{_STYLE}</style></head><body><main>",
        "<h1>ArmForge benchmark report</h1>",
        f"<p class='sub'>{_e(cpu.model)} · {_e(cpu.architecture)} · "
        f"generated {_e(report.finished_at or report.started_at)}</p>",
    ]

    # -- recommendation ---------------------------------------------------
    if recommendation and winning:
        parts.append("<h2>Recommended configuration</h2>")
        parts.append("<div class='grid'>")
        for key, value in (
            ("Model", winning.model.quantization or "unknown"),
            ("Runtime", winning.runtime.label),
            (
                "Prefill threads",
                recommendation.prefill.threads if recommendation.prefill else "—",
            ),
            (
                "Decode threads",
                recommendation.decode.threads if recommendation.decode else "—",
            ),
        ):
            parts.append(
                f"<div class='kv'><div class='k'>{_e(key)}</div>"
                f"<div class='v'>{_e(value)}</div></div>"
            )
        parts.append("</div>")

        parts.append("<h3>Deploy</h3>")
        parts.append(f"<pre>{_e(recommendation.deployment_command)}</pre>")

        if recommendation.reasons:
            parts.append("<h3>Why this configuration</h3><ul>")
            parts.extend(f"<li>{_e(r)}</li>" for r in recommendation.reasons)
            parts.append("</ul>")

        if recommendation.baseline is not None:
            base = recommendation.baseline.config
            parts.append(
                f"<p class='sub'>Compared against the naive default: "
                f"{base.threads} threads, {_e(base.model.quantization)}, "
                f"{_e(base.runtime.label)} — what <code>nproc</code> and an "
                f"unmodified build would give you.</p>"
            )

        # Caveats are part of the result, not an appendix.
        if recommendation.warnings:
            parts.append("<h3>Caveats</h3>")
            parts.extend(f"<div class='caveat'>{_e(w)}</div>" for w in recommendation.warnings)

    # -- charts -----------------------------------------------------------
    charts = _phase_charts(
        report,
        winning.model.quantization if winning else None,
        winning.runtime.build_flags.get("variant") if winning else None,
    )
    if charts:
        parts.append("<h2>Throughput vs thread count</h2>")
        parts.append(charts)
        parts.append(
            "<p class='sub'>Error bars show one standard deviation across "
            "repetitions. Prefill and decode are plotted on separate scales "
            "because they differ by several times and share no bottleneck.</p>"
        )

    # -- all measurements -------------------------------------------------
    parts.append("<h2>All measurements</h2>")
    parts.append(_measurements_table(report.results, winner_label))
    parts.append(
        f"<p class='sub'>{len(report.succeeded)} measured, "
        f"{len(report.failed)} failed, {len(report.plan.pruned)} pruned before "
        "running. ⚠ marks measurements that varied more than 5% between "
        "repetitions.</p>"
    )

    if report.plan.pruned:
        parts.append("<h3>Not measured, and why</h3><ul>")
        parts.extend(
            f"<li><strong>{_e(p.label)}</strong> — {_e(p.reason)}</li>"
            for p in report.plan.pruned
        )
        parts.append("</ul>")

    # -- pareto -----------------------------------------------------------
    if recommendation and recommendation.pareto:
        parts.append("<h2>Pareto frontier</h2>")
        parts.append(
            "<p class='sub'>No other candidate beats these on speed and memory "
            "at the same time, so each represents a defensible trade-off.</p>"
        )
        parts.append(_measurements_table(list(recommendation.pareto), winner_label))

    # -- environment ------------------------------------------------------
    parts.append("<h2>Environment</h2><table><tbody>")
    for key, value in (
        ("CPU", cpu.model),
        ("Architecture", cpu.architecture),
        ("Cores", f"{cpu.physical_cores} physical ({topology}) — {clusters}"),
        ("Arm features", features),
        (
            "SVE vector length",
            f"{cpu.sve_vector_bits} bits" if cpu.sve_vector_bits else "not implemented",
        ),
        (
            "SME vector length",
            f"{cpu.sme_vector_bits} bits" if cpu.sme_vector_bits else "not implemented",
        ),
        ("Memory", f"{host.total_memory_gb:.1f} GB"),
        ("OS", f"{host.os_name} {host.os_release}"),
        ("Detector", host.detector),
    ):
        parts.append(f"<tr><th>{_e(key)}</th><td>{_e(value)}</td></tr>")
    parts.append("</tbody></table>")

    if host.warnings:
        parts.append("<h3>Detection notes</h3><ul>")
        parts.extend(f"<li>{_e(w)}</li>" for w in host.warnings)
        parts.append("</ul>")

    # -- honesty ----------------------------------------------------------
    parts.append("<h2>What this report is not</h2><ul>")
    parts.append(
        "<li><strong>Not a quality claim.</strong> These are throughput "
        "measurements. A faster quantisation is not necessarily a better one, "
        "and different quantisations do not produce identical output.</li>"
    )
    parts.append(
        "<li><strong>Specific to this machine and this model.</strong> The "
        "recommendation follows from the CPU and model measured here. Rerun "
        "<code>armforge optimize</code> on the target you intend to deploy "
        "to.</li>"
    )
    parts.append(
        "<li><strong>Capability is not usage.</strong> Detecting an Arm "
        "extension proves the silicon has it, not that the runtime used it. "
        "The ggml feature flags actually compiled into each build are recorded "
        "alongside every result in <code>benchmark.json</code>.</li>"
    )
    parts.append("</ul>")

    parts.append(
        "<p class='footer'>Generated by ArmForge from measurements taken on "
        "this machine. Every figure here comes from a benchmark run; none is "
        "estimated. Raw results, including the full runtime output, are in "
        "<code>benchmark.json</code>.</p>"
    )
    parts.append("</main></body></html>")
    return "".join(parts)
