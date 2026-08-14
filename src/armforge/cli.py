"""ArmForge command line interface."""

from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table
from rich.text import Text

from . import __version__
from .hardware import Relevance, detect_host, notable_absent, relevant_present
from .hardware.types import CoreKind, HostProfile

app = typer.Typer(
    name="armforge",
    help="Arm-aware inference configuration engine.",
    no_args_is_help=True,
    add_completion=False,
)

console = Console()

_RELEVANCE_STYLE = {
    Relevance.CRITICAL: "bold red",
    Relevance.HIGH: "yellow",
    Relevance.MODERATE: "cyan",
    Relevance.LOW: "dim",
}


def _format_bytes(value: int | None) -> str:
    if not value:
        return "unknown"
    gb = value / (1024**3)
    if gb >= 1:
        return f"{gb:.1f} GB"
    return f"{value / (1024**2):.0f} MB"


def _render_summary(host: HostProfile) -> None:
    cpu = host.cpu

    table = Table.grid(padding=(0, 2))
    table.add_column(style="dim", justify="right")
    table.add_column()

    arch = Text(cpu.architecture)
    if cpu.is_arm64:
        arch.append("  Arm64", style="bold green")
    else:
        arch.append("  not Arm64", style="bold yellow")
    table.add_row("Architecture", arch)
    table.add_row("CPU", cpu.model)

    topology = "heterogeneous" if cpu.is_heterogeneous else "uniform"
    table.add_row(
        "Cores",
        f"{cpu.physical_cores} physical / {cpu.logical_cores} logical  [dim]({topology})[/dim]",
    )

    for cluster in cpu.clusters:
        detail = f"{cluster.physical_cores} cores"
        if cluster.max_freq_mhz:
            detail += f" @ {cluster.max_freq_mhz} MHz"
        if cluster.l2_cache_bytes:
            detail += f", L2 {_format_bytes(cluster.l2_cache_bytes)}"
        marker = {
            CoreKind.PERFORMANCE: "[green]P[/green]",
            CoreKind.EFFICIENCY: "[blue]E[/blue]",
        }.get(cluster.kind, "[dim]·[/dim]")
        table.add_row("", f"{marker} {cluster.name:<14} [dim]{detail}[/dim]")

    memory = _format_bytes(host.total_memory_bytes)
    if host.available_memory_bytes:
        memory += f"  [dim]({_format_bytes(host.available_memory_bytes)} available)[/dim]"
    table.add_row("Memory", memory)
    table.add_row("OS", f"{host.os_name} {host.os_release}".strip())
    table.add_row("Detector", f"[dim]{host.detector}[/dim]")

    console.print()
    console.print("[bold]ArmForge[/bold] [dim]·[/dim] Hardware")
    console.print()
    console.print(table)


def _render_features(host: HostProfile) -> None:
    cpu = host.cpu
    if not cpu.is_arm64:
        console.print()
        console.print(
            "[yellow]This machine is not Arm64.[/yellow] ArmForge runs here for "
            "development, but\nany Arm capability analysis requires an Arm64 host."
        )
        return

    present = relevant_present(cpu.features)
    absent = notable_absent(cpu.features, is_arm64=True)

    console.print()
    console.print("[bold]Arm inference features[/bold]")
    console.print()

    table = Table(show_header=True, header_style="dim", box=None, padding=(0, 2))
    table.add_column("", width=1)
    table.add_column("Feature", style="bold")
    table.add_column("Extension", style="dim")
    table.add_column("Relevance")

    for info in present:
        table.add_row(
            "[green]✓[/green]",
            info.key,
            info.arm_name,
            Text(info.relevance.value, style=_RELEVANCE_STYLE[info.relevance]),
        )
    for info in absent:
        table.add_row(
            "[dim]✗[/dim]",
            f"[dim]{info.key}[/dim]",
            info.arm_name,
            Text(info.relevance.value, style="dim"),
        )

    console.print(table)

    vectors = []
    if cpu.sve_vector_bits:
        vectors.append(f"SVE vector length {cpu.sve_vector_bits} bits")
    if cpu.sme_vector_bits:
        vectors.append(f"SME streaming vector length {cpu.sme_vector_bits} bits")
    if vectors:
        console.print()
        for line in vectors:
            console.print(f"  [dim]{line}[/dim]")


def _render_notes(host: HostProfile) -> None:
    if not host.warnings:
        return
    console.print()
    console.print("[bold]Notes[/bold]")
    for warning in host.warnings:
        console.print(f"  [dim]·[/dim] {warning}")


@app.command()
def hardware(
    as_json: bool = typer.Option(
        False, "--json", help="Emit the raw profile as JSON for scripting."
    ),
    explain: bool = typer.Option(
        False,
        "--explain",
        help="Describe why each detected feature matters for inference.",
    ),
) -> None:
    """Detect the host CPU, its Arm feature set and its core topology."""
    host = detect_host()

    if as_json:
        console.print_json(json.dumps(host.to_dict()))
        return

    _render_summary(host)
    _render_features(host)

    if explain and host.cpu.is_arm64:
        console.print()
        console.print("[bold]Why these matter[/bold]")
        for info in relevant_present(host.cpu.features):
            console.print()
            console.print(f"  [bold]{info.key}[/bold] [dim]— {info.title}[/dim]")
            console.print(f"  [dim]{info.since}[/dim]")
            console.print(f"  {info.inference_impact}")

    _render_notes(host)
    console.print()


@app.command()
def analyze(
    model: str = typer.Argument(..., help="Path to a .gguf model file."),
    as_json: bool = typer.Option(False, "--json", help="Emit the analysis as JSON."),
) -> None:
    """Inspect a GGUF model and relate it to this machine's Arm capabilities."""
    from .analyzer import GGUFError, read_gguf

    try:
        info = read_gguf(model)
    except GGUFError as exc:
        console.print(f"[red]error:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    host = detect_host()

    if as_json:
        console.print_json(json.dumps({"model": info.to_dict(), "host": host.to_dict()}))
        return

    table = Table.grid(padding=(0, 2))
    table.add_column(style="dim", justify="right")
    table.add_column()
    table.add_row("Model", info.name or "unknown")
    table.add_row("File", Path(info.path).name)
    table.add_row("Architecture", info.architecture or "unknown")
    table.add_row("Quantization", info.quantization or "[yellow]unknown[/yellow]")
    if info.parameter_count:
        table.add_row("Parameters", f"{info.parameter_count / 1e9:.2f} B")
    table.add_row("File size", _format_bytes(info.file_size_bytes))
    if info.bits_per_weight:
        table.add_row("Bits/weight", f"{info.bits_per_weight:.2f}")
    if info.context_length:
        table.add_row("Context", f"{info.context_length:,} tokens")
    table.add_row("Tensors", str(info.tensor_count))

    console.print()
    console.print("[bold]ArmForge[/bold] [dim]·[/dim] Model analysis")
    console.print()
    console.print(table)

    console.print()
    console.print(f"[bold]Arm outlook on {host.cpu.model}[/bold]")
    console.print()
    for line in _arm_outlook(info, host):
        console.print(f"  {line}")
    console.print()
    console.print(
        "  [dim]These are predictions from the capability model, not "
        "measurements.\n  Run 'armforge benchmark' to test them.[/dim]"
    )
    console.print()


def _arm_outlook(info, host: HostProfile) -> list[str]:
    """Predictions this machine's capability vector implies for this model.

    Deliberately phrased as expectations to be tested. Nothing here is a
    measurement, and the CLI says so.
    """
    cpu = host.cpu
    lines: list[str] = []

    if not cpu.is_arm64:
        return ["[yellow]![/yellow] Not an Arm64 host; no Arm analysis available."]

    quant = info.quantization
    if quant is None:
        lines.append(
            "[yellow]?[/yellow] Quantization unknown, so no format-specific "
            "prediction can be made."
        )
    elif info.repackable_for_i8mm and cpu.has("i8mm"):
        lines.append(
            f"[green]✓[/green] {quant} can be repacked to feed SMMLA, and this "
            "CPU reports FEAT_I8MM.\n    Expect prefill to gain more than decode."
        )
    elif info.repackable_for_i8mm:
        lines.append(
            f"[yellow]~[/yellow] {quant} is repackable, but this CPU lacks "
            "FEAT_I8MM, so that layout\n    buys nothing here."
        )
    else:
        lines.append(
            f"[dim]·[/dim] {quant} is a K-quant with no int8 matrix fast path; "
            "FEAT_I8MM will go\n    unused regardless of the hardware."
        )

    if cpu.has("sme2"):
        lines.append(
            "[green]✓[/green] FEAT_SME2 present. Only a KleidiAI-enabled build "
            "can reach it;\n    a stock build will leave it idle."
        )

    if cpu.is_heterogeneous:
        lines.append(
            f"[yellow]![/yellow] Heterogeneous CPU: {cpu.performance_cores} "
            f"performance + {cpu.physical_cores - cpu.performance_cores} "
            "efficiency cores.\n    Expect decode to peak at or below "
            f"{cpu.performance_cores} threads, not {cpu.physical_cores}."
        )

    if info.file_size_bytes > (host.available_memory_bytes or 0):
        lines.append(
            "[yellow]![/yellow] Model is larger than currently available memory; "
            "expect paging to\n    distort results."
        )

    return lines


@app.command()
def runtimes(
    probe: bool = typer.Option(
        False,
        "--probe",
        help="Run each build once to record which Arm code paths it compiled in.",
    ),
    model: str = typer.Option(
        None, "--model", help="Model used for probing (required with --probe)."
    ),
) -> None:
    """List the llama.cpp builds ArmForge can benchmark against."""
    from .bench.llamacpp import discover_runtimes, probe_ggml_features

    found = discover_runtimes()
    console.print()
    console.print("[bold]ArmForge[/bold] [dim]·[/dim] Runtimes")
    console.print()

    if not found:
        console.print(
            "  [yellow]No llama.cpp builds found.[/yellow]\n"
            "  Run [bold]scripts/setup-llama-cpp.sh[/bold] to build them."
        )
        console.print()
        return

    table = Table(show_header=True, header_style="dim", box=None, padding=(0, 2))
    table.add_column("Variant", style="bold")
    table.add_column("Commit", style="dim")
    table.add_column("KleidiAI")
    table.add_column("Accelerate")
    if probe:
        table.add_column("ggml reports")

    for spec in found:
        row = [
            spec.build_flags.get("variant", "?"),
            spec.version,
            "[green]on[/green]" if spec.build_flags.get("kleidiai") else "[dim]off[/dim]",
            "[green]on[/green]" if spec.build_flags.get("accelerate") else "[dim]off[/dim]",
        ]
        if probe:
            if not model:
                row.append("[yellow]--model required[/yellow]")
            else:
                features = probe_ggml_features(spec, model)
                row.append(
                    ", ".join(sorted(k for k, v in features.items() if v))
                    if features
                    else "[yellow]probe failed[/yellow]"
                )
        table.add_row(*row)

    console.print(table)
    console.print()


@app.command()
def benchmark(
    model: str = typer.Argument(..., help="Path to a .gguf model file."),
    workload: str = typer.Option("short", "--workload", "-w", help="Workload name."),
    threads: int = typer.Option(None, "--threads", "-t", help="Thread count."),
    variant: str = typer.Option("cpu", "--variant", help="llama.cpp build variant."),
    iterations: int = typer.Option(5, "--iterations", "-r", help="Repetitions."),
    output: str = typer.Option(None, "--output", "-o", help="Write result JSON here."),
) -> None:
    """Benchmark one configuration and report the measured distribution."""
    from .analyzer import GGUFError, read_gguf
    from .bench import workloads as wl
    from .bench.llamacpp import LlamaCppRunner, discover_runtimes
    from .bench.types import BenchConfig, ModelRef

    host = detect_host()

    try:
        shape = wl.get(workload)
    except KeyError as exc:
        console.print(f"[red]error:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    runtimes = {r.build_flags.get("variant"): r for r in discover_runtimes()}
    if variant not in runtimes:
        available = ", ".join(sorted(k for k in runtimes if k)) or "none"
        console.print(
            f"[red]error:[/red] no llama.cpp build named {variant!r} "
            f"(available: {available}).\nRun scripts/setup-llama-cpp.sh."
        )
        raise typer.Exit(code=1)

    try:
        info = read_gguf(model)
    except GGUFError as exc:
        console.print(f"[red]error:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    # Default to performance-core count, not nproc: on a heterogeneous CPU
    # the efficiency cores drag the whole thread pool down.
    if threads is None:
        threads = host.cpu.performance_cores

    config = BenchConfig(
        model=ModelRef(
            path=str(Path(model).resolve()),
            name=info.name or Path(model).stem,
            size_bytes=info.file_size_bytes,
            quantization=info.quantization,
            n_params=info.parameter_count,
        ),
        runtime=runtimes[variant],
        workload=shape,
        threads=threads,
        iterations=iterations,
    )

    console.print()
    console.print(f"[dim]Running[/dim] {config.label} [dim]·[/dim] {shape.name}")
    result = LlamaCppRunner(config.runtime).run(config, host)

    if output:
        Path(output).write_text(json.dumps(result.to_dict(), indent=2))

    if not result.ok:
        console.print(f"[red]{result.status.value}:[/red] {result.error}")
        raise typer.Exit(code=1)

    table = Table.grid(padding=(0, 2))
    table.add_column(style="dim", justify="right")
    table.add_column()
    for label, stats in (("Prefill", result.prefill_tps), ("Decode", result.decode_tps)):
        if stats is None:
            table.add_row(label, "[yellow]not measured[/yellow]")
            continue
        noise = stats.relative_stddev
        flag = "  [yellow](noisy)[/yellow]" if noise > 0.05 else ""
        table.add_row(
            label,
            f"{stats.mean:.2f} ± {stats.stddev:.2f} {stats.unit}  "
            f"[dim]n={stats.samples}[/dim]{flag}",
        )
    if result.ttft_ms is not None:
        table.add_row("TTFT", f"{result.ttft_ms:.0f} ms  [dim](derived)[/dim]")
    if result.peak_memory_bytes:
        table.add_row("Peak memory", _format_bytes(result.peak_memory_bytes))
    table.add_row("Wall time", f"{result.wall_time_s:.1f} s")

    console.print()
    console.print(table)
    if output:
        console.print()
        console.print(f"  [dim]result written to {output}[/dim]")
    console.print()


@app.command()
def optimize(
    models: list[str] = typer.Argument(..., help="One or more .gguf model files."),
    workload: str = typer.Option("short", "--workload", "-w", help="Workload name."),
    objective: str = typer.Option(
        "best-balance", "--objective", help="fastest | lowest-memory | best-balance."
    ),
    variants: str = typer.Option(
        None, "--variants", help="Comma-separated build variants (default: all found)."
    ),
    iterations: int = typer.Option(5, "--iterations", "-r", help="Repetitions."),
    output: str = typer.Option(None, "--output", "-o", help="Write full report JSON."),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Show the measurement plan without running it."
    ),
) -> None:
    """Sweep configurations, then recommend and justify one."""
    from .bench import workloads as wl
    from .bench.llamacpp import discover_runtimes
    from .optimize import Objective, estimate_duration, generate, recommend, run_sweep

    host = detect_host()

    try:
        shape = wl.get(workload)
        goal = Objective(objective)
    except (KeyError, ValueError) as exc:
        console.print(f"[red]error:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    found = discover_runtimes()
    if variants:
        wanted = {v.strip() for v in variants.split(",")}
        found = [r for r in found if r.build_flags.get("variant") in wanted]
    if not found:
        console.print(
            "[red]error:[/red] no llama.cpp builds found. Run scripts/setup-llama-cpp.sh."
        )
        raise typer.Exit(code=1)

    plan = generate(host, list(models), found, shape, iterations=iterations)

    console.print()
    console.print("[bold]ArmForge[/bold] [dim]·[/dim] Optimize")
    console.print()
    summary = Table.grid(padding=(0, 2))
    summary.add_column(style="dim", justify="right")
    summary.add_column()
    summary.add_row("Host", f"{host.cpu.model} [dim]·[/dim] {host.cpu.architecture}")
    summary.add_row("Workload", f"{shape.name} [dim]({shape.description})[/dim]")
    summary.add_row("Objective", goal.value)
    summary.add_row(
        "Plan",
        f"{len(plan)} candidates, {len(plan.pruned)} pruned "
        f"[dim](~{estimate_duration(plan) / 60:.0f} min)[/dim]",
    )
    console.print(summary)

    if plan.notes:
        console.print()
        for note in plan.notes:
            console.print(f"  [dim]·[/dim] {note}")
    if plan.pruned:
        console.print()
        console.print("  [bold]Not measured[/bold]")
        for item in plan.pruned:
            console.print(f"  [dim]✗ {item.label}: {item.reason}[/dim]")

    if not plan.candidates:
        console.print("\n[yellow]Nothing to measure.[/yellow]\n")
        raise typer.Exit(code=1)

    if dry_run:
        console.print()
        for candidate in plan.candidates:
            console.print(f"  [dim]•[/dim] {candidate.label}")
            console.print(f"    [dim]{candidate.rationale}[/dim]")
        console.print()
        return

    console.print()

    def progress(index: int, total: int, label: str) -> None:
        console.print(f"  [dim][{index}/{total}][/dim] {label}")

    report = run_sweep(plan, host, on_progress=progress)

    if output:
        payload = report.to_dict()
        rec_preview = recommend(report.results, host, goal)
        payload["recommendation"] = rec_preview.to_dict() if rec_preview else None
        Path(output).write_text(json.dumps(payload, indent=2))

    _render_sweep_results(report)

    rec = recommend(report.results, host, goal)
    if rec is None:
        console.print("\n[yellow]No configuration could be measured.[/yellow]\n")
        raise typer.Exit(code=1)

    _render_recommendation(rec)

    if output:
        console.print(f"  [dim]full report written to {output}[/dim]")
        console.print()


def _render_sweep_results(report) -> None:
    console.print()
    console.print("[bold]Measurements[/bold]")
    console.print()

    table = Table(show_header=True, header_style="dim", box=None, padding=(0, 2))
    table.add_column("Configuration", style="bold")
    table.add_column("Prefill tok/s", justify="right")
    table.add_column("Decode tok/s", justify="right")
    table.add_column("Peak mem", justify="right")

    for result in report.results:
        if not result.ok:
            table.add_row(
                f"[dim]{result.config.label}[/dim]",
                f"[yellow]{result.status.value}[/yellow]",
                "",
                "",
            )
            continue

        def cell(stats):
            if stats is None:
                return "[dim]n/a[/dim]"
            noisy = stats.relative_stddev > 0.05
            text = f"{stats.mean:.1f} ± {stats.stddev:.1f}"
            return f"[yellow]{text}[/yellow]" if noisy else text

        table.add_row(
            result.config.label,
            cell(result.prefill_tps),
            cell(result.decode_tps),
            _format_bytes(result.peak_memory_bytes),
        )

    console.print(table)
    console.print(
        f"\n  [dim]{len(report.succeeded)} measured, {len(report.failed)} failed. "
        "Yellow marks >5% run-to-run variation.[/dim]"
    )


def _render_recommendation(rec) -> None:
    console.print()
    console.print(f"[bold]Recommendation[/bold] [dim]· {rec.objective.value}[/dim]")
    console.print()

    config = rec.winner.result.config
    table = Table.grid(padding=(0, 2))
    table.add_column(style="dim", justify="right")
    table.add_column()
    table.add_row("Model", f"{config.model.quantization} [dim]{config.model.name}[/dim]")
    table.add_row("Runtime", config.runtime.label)
    if rec.prefill:
        table.add_row(
            "Prefill threads",
            f"[bold]{rec.prefill.threads}[/bold]  "
            f"[dim]{rec.prefill.throughput:.1f} tok/s[/dim]",
        )
    if rec.decode:
        table.add_row(
            "Decode threads",
            f"[bold]{rec.decode.threads}[/bold]  [dim]{rec.decode.throughput:.1f} tok/s[/dim]",
        )
    console.print(table)

    if rec.reasons:
        console.print()
        console.print("  [bold]Why[/bold]")
        for reason in rec.reasons:
            console.print(f"  [green]·[/green] {reason}")

    if rec.baseline is not None:
        console.print()
        console.print(
            f"  [dim]Compared against the naive default: "
            f"{rec.baseline.config.threads} threads "
            f"({rec.baseline.config.model.quantization}, "
            f"{rec.baseline.config.runtime.label}).[/dim]"
        )

    if rec.warnings:
        console.print()
        console.print("  [bold yellow]Caveats[/bold yellow]")
        for warning in rec.warnings:
            console.print(f"  [yellow]![/yellow] {warning}")

    if rec.pareto:
        console.print()
        console.print(
            "  [bold]Pareto frontier[/bold] [dim](no candidate beats these "
            "on speed and memory at once)[/dim]"
        )
        for result in rec.pareto:
            console.print(
                f"  [dim]·[/dim] {result.config.label}  "
                f"[dim]{result.prefill_tps.mean:.0f} prefill / "
                f"{result.decode_tps.mean:.0f} decode tok/s[/dim]"
            )

    console.print()
    console.print("[bold]Deploy[/bold]")
    console.print()
    console.print(f"  [cyan]{rec.deployment_command}[/cyan]")
    console.print()


@app.command()
def version() -> None:
    """Print the ArmForge version."""
    console.print(f"armforge {__version__}")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
