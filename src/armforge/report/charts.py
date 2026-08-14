"""Dependency-free SVG charts.

Shared by the exported HTML report and by ``scripts/build_results.py`` so both
draw the same way. Colours are mid-tones chosen to stay legible against a
light or a dark background, since neither GitHub nor a browser tells us which
one the reader has.
"""

from __future__ import annotations

from collections.abc import Sequence

#: Axis, label and gridline colour. Mid-grey reads on white and on near-black.
INK = "#8b949e"

SERIES_COLORS: dict[str, str] = {
    "prefill": "#3b82f6",
    "decode": "#f59e0b",
    "gain %": "#10b981",
}

#: A point is (x, mean, stddev). stddev may be 0 when there is no spread.
Point = tuple[float, float, float]


def line_chart(
    title: str,
    series: dict[str, Sequence[Point]],
    *,
    colors: dict[str, str] | None = None,
    y_label: str = "",
    x_label: str = "threads",
    width: int = 560,
    height: int = 300,
) -> str:
    """Render a multi-series line chart with error bars.

    Returns an empty string when there is nothing to plot, so callers can skip
    an empty figure rather than emit a blank frame.
    """
    palette = {**SERIES_COLORS, **(colors or {})}

    # A legend is redundant when the title already names the only series.
    show_legend = len(series) > 1
    pad_l, pad_r, pad_t, pad_b = 58, (130 if show_legend else 24), 34, 44
    plot_w = width - pad_l - pad_r
    plot_h = height - pad_t - pad_b

    all_points = [p for points in series.values() for p in points]
    if not all_points:
        return ""

    xs = sorted({p[0] for p in all_points})
    y_top = max(p[1] + p[2] for p in all_points)
    y_bottom = min(0.0, min(p[1] - p[2] for p in all_points))
    span = (y_top - y_bottom) or 1.0
    y_max = y_top + span * 0.12

    x_min, x_max = min(xs), max(xs)

    def sx(x: float) -> float:
        if x_max == x_min:
            return pad_l + plot_w / 2
        return pad_l + (x - x_min) / (x_max - x_min) * plot_w

    def sy(y: float) -> float:
        return pad_t + plot_h - (y - y_bottom) / (y_max - y_bottom) * plot_h

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'width="{width}" height="{height}" font-family="system-ui,sans-serif" '
        f'role="img" aria-label="{_escape(title)}">',
        f"<title>{_escape(title)}</title>",
        f'<text x="{pad_l}" y="20" font-size="13" font-weight="600" '
        f'fill="{INK}">{_escape(title)}</text>',
    ]

    for step in range(5):
        value = y_bottom + (y_max - y_bottom) * step / 4
        y = sy(value)
        parts.append(
            f'<line x1="{pad_l}" y1="{y:.1f}" x2="{pad_l + plot_w}" y2="{y:.1f}" '
            f'stroke="{INK}" stroke-opacity="0.18" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{pad_l - 8}" y="{y + 4:.1f}" font-size="10" text-anchor="end" '
            f'fill="{INK}" fill-opacity="0.75">{value:.0f}</text>'
        )

    for x in xs:
        parts.append(
            f'<text x="{sx(x):.1f}" y="{pad_t + plot_h + 18}" font-size="10" '
            f'text-anchor="middle" fill="{INK}" fill-opacity="0.75">{x:g}</text>'
        )
    if x_label:
        parts.append(
            f'<text x="{pad_l + plot_w / 2:.1f}" y="{height - 8}" font-size="11" '
            f'text-anchor="middle" fill="{INK}" fill-opacity="0.85">{x_label}</text>'
        )
    if y_label:
        mid = pad_t + plot_h / 2
        parts.append(
            f'<text x="14" y="{mid:.1f}" font-size="11" fill="{INK}" '
            f'fill-opacity="0.85" text-anchor="middle" '
            f'transform="rotate(-90 14 {mid:.1f})">{y_label}</text>'
        )

    for index, (label, points) in enumerate(series.items()):
        if not points:
            continue
        color = palette.get(label, "#3b82f6")
        ordered = sorted(points)
        path = " ".join(
            f"{'M' if i == 0 else 'L'}{sx(x):.1f},{sy(y):.1f}"
            for i, (x, y, _) in enumerate(ordered)
        )
        parts.append(
            f'<path d="{path}" fill="none" stroke="{color}" stroke-width="2.2" '
            f'stroke-linejoin="round"/>'
        )
        for x, y, sd in ordered:
            if sd > 0:
                parts.append(
                    f'<line x1="{sx(x):.1f}" y1="{sy(y - sd):.1f}" '
                    f'x2="{sx(x):.1f}" y2="{sy(y + sd):.1f}" '
                    f'stroke="{color}" stroke-width="1.4" stroke-opacity="0.55"/>'
                )
            parts.append(f'<circle cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="3.4" fill="{color}"/>')

        if show_legend:
            legend_y = pad_t + 6 + index * 18
            parts.append(
                f'<rect x="{pad_l + plot_w + 16}" y="{legend_y - 8}" width="10" '
                f'height="10" rx="2" fill="{color}"/>'
            )
            parts.append(
                f'<text x="{pad_l + plot_w + 32}" y="{legend_y + 1}" font-size="11" '
                f'fill="{INK}">{_escape(label)}</text>'
            )

    parts.append("</svg>")
    return "\n".join(parts)


def _escape(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )
