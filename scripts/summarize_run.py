#!/usr/bin/env python3
"""Render a sweep result as Markdown, for a CI job summary or a PR comment.

Reads ``results/*.json`` written by ``armforge optimize --output`` and prints
GitHub-flavoured Markdown to stdout. Kept dependency-free so it can run
anywhere the sweep ran.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

RESULTS_DIR = Path("results")


def _fmt(stats: dict[str, Any] | None) -> str:
    """Format a metric, marking anything noisier than 5% run-to-run."""
    if not stats:
        return "n/a"
    text = f"{stats['mean']:.1f} ± {stats['stddev']:.1f}"
    return f"{text} ⚠️" if stats.get("relative_stddev", 0) > 0.05 else text


def _host_table(host: dict[str, Any]) -> list[str]:
    cpu = host["cpu"]
    topology = "heterogeneous" if cpu["is_heterogeneous"] else "uniform"
    clusters = ", ".join(
        f"{c['physical_cores']}× {c.get('core_name') or c['name']}"
        for c in cpu["clusters"]
    )
    present = ", ".join(f"`{f}`" for f in cpu["features"]) or "none detected"

    return [
        "## Host",
        "",
        f"- **CPU**: {cpu['model']} ({cpu['architecture']})",
        f"- **Topology**: {cpu['physical_cores']} cores, {topology} — {clusters}",
        f"- **Arm features**: {present}",
        f"- **SVE vector length**: {cpu.get('sve_vector_bits') or 'not implemented'}",
        f"- **SME vector length**: {cpu.get('sme_vector_bits') or 'not implemented'}",
        f"- **Memory**: {host['total_memory_gb']} GB",
        f"- **Detector**: `{host['detector']}`",
        "",
    ]


def _measurements_table(results: list[dict[str, Any]]) -> list[str]:
    lines = [
        "## Measurements",
        "",
        "| Configuration | Prefill tok/s | Decode tok/s | Peak memory |",
        "| --- | ---: | ---: | ---: |",
    ]
    for result in results:
        label = result["config"]["label"]
        if result["status"] != "ok":
            lines.append(f"| {label} | _{result['status']}_ | | |")
            continue

        metrics = result["metrics"]
        memory = metrics.get("peak_memory_bytes")
        memory_text = f"{memory / 1024**2:.0f} MB" if memory else "n/a"
        lines.append(
            f"| {label} | {_fmt(metrics['prefill_tps'])} "
            f"| {_fmt(metrics['decode_tps'])} | {memory_text} |"
        )
    lines.append("")
    lines.append("⚠️ marks measurements that varied more than 5% between repetitions.")
    lines.append("")
    return lines


def _recommendation_section(rec: dict[str, Any]) -> list[str]:
    lines = ["## Recommendation", "", f"**Objective**: `{rec['objective']}`", ""]

    prefill, decode = rec.get("prefill"), rec.get("decode")
    if prefill:
        lines.append(
            f"- **Prompt processing**: {prefill['threads']} threads "
            f"({prefill['throughput_tok_s']} tok/s)"
        )
    if decode:
        lines.append(
            f"- **Token generation**: {decode['threads']} threads "
            f"({decode['throughput_tok_s']} tok/s)"
        )
    lines.append("")

    if rec.get("reasons"):
        lines += ["### Why", ""]
        lines += [f"- {reason}" for reason in rec["reasons"]]
        lines.append("")

    if rec.get("warnings"):
        lines += ["### Caveats", ""]
        lines += [f"- {warning}" for warning in rec["warnings"]]
        lines.append("")

    lines += ["### Deploy", "", "```bash", rec["deployment_command"], "```", ""]

    if rec.get("pareto"):
        lines += [
            "### Pareto frontier",
            "",
            "No candidate beats these on speed and memory at once:",
            "",
        ]
        lines += [f"- `{label}`" for label in rec["pareto"]]
        lines.append("")

    return lines


def main() -> int:
    files = sorted(RESULTS_DIR.glob("*.json")) if RESULTS_DIR.is_dir() else []
    if not files:
        print("_No sweep results were produced._")
        return 0

    for path in files:
        try:
            report = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            print(f"_Could not read `{path}`: {exc}_")
            continue

        # Single-benchmark artifacts have no host/plan wrapper.
        if "results" not in report:
            continue

        print(f"# ArmForge sweep — `{path.name}`")
        print()
        print("\n".join(_host_table(report["host"])))

        summary = report.get("summary", {})
        print(
            f"_{summary.get('succeeded', 0)} measured, "
            f"{summary.get('failed', 0)} failed, "
            f"{summary.get('pruned', 0)} pruned before running._"
        )
        print()

        pruned = report.get("plan", {}).get("pruned", [])
        if pruned:
            print("### Not measured")
            print()
            for item in pruned:
                print(f"- **{item['label']}** — {item['reason']}")
            print()

        print("\n".join(_measurements_table(report["results"])))

        recommendation = report.get("recommendation")
        if recommendation:
            print("\n".join(_recommendation_section(recommendation)))

    return 0


if __name__ == "__main__":
    sys.exit(main())
