"""ArmForge command line interface."""

from __future__ import annotations

import json

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
def version() -> None:
    """Print the ArmForge version."""
    console.print(f"armforge {__version__}")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
