"""
HTML assembly for the standalone credit tranche pricer dashboard.

"""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

__all__ = [
    "render_page",
    "render_section",
    "render_metric_row",
    "metric_card",
    "status_badge",
    "render_warnings",
    "render_table",
]

_PLOTLY_CDN = "https://cdn.plot.ly/plotly-2.35.2.min.js"

_CSS = """
:root {
    --color-bg: #f5f6f8;
    --color-panel: #ffffff;
    --color-border: #dde1e6;
    --color-text: #1b2430;
    --color-text-muted: #5b6573;
    --color-accent: #0b3d63;
    --color-accent-light: #e7eef5;
    --color-pass: #1b7a43;
    --color-pass-bg: #e6f4ea;
    --color-fail: #a3261f;
    --color-fail-bg: #fbe9e7;
    --font-sans: "Inter", "Segoe UI", Helvetica, Arial, sans-serif;
    --font-mono: "IBM Plex Mono", "SFMono-Regular", Consolas, monospace;
}
* { box-sizing: border-box; }
body {
    margin: 0;
    background: var(--color-bg);
    color: var(--color-text);
    font-family: var(--font-sans);
    font-size: 14px;
    line-height: 1.45;
}
header.report-header {
    background: var(--color-accent);
    color: #ffffff;
    padding: 20px 32px;
}
header.report-header h1 {
    margin: 0 0 4px 0;
    font-size: 20px;
    font-weight: 600;
    letter-spacing: 0.2px;
}
header.report-header p {
    margin: 0;
    font-size: 12.5px;
    color: #c9d8e6;
    font-family: var(--font-mono);
}
nav.report-nav {
    background: #ffffff;
    border-bottom: 1px solid var(--color-border);
    padding: 8px 32px;
    position: sticky;
    top: 0;
    z-index: 10;
    display: flex;
    gap: 18px;
    flex-wrap: wrap;
}
nav.report-nav a {
    color: var(--color-accent);
    text-decoration: none;
    font-size: 12.5px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.4px;
}
nav.report-nav a:hover { text-decoration: underline; }
main { max-width: 1280px; margin: 0 auto; padding: 8px 32px 64px 32px; }
section.report-section {
    background: var(--color-panel);
    border: 1px solid var(--color-border);
    border-radius: 6px;
    margin-top: 24px;
    padding: 20px 24px;
}
section.report-section h2 {
    margin: 0 0 4px 0;
    font-size: 15px;
    font-weight: 700;
    color: var(--color-accent);
    text-transform: uppercase;
    letter-spacing: 0.5px;
}
section.report-section .section-caption {
    margin: 0 0 16px 0;
    font-size: 12.5px;
    color: var(--color-text-muted);
}
.metric-row { display: flex; gap: 12px; flex-wrap: wrap; margin-bottom: 12px; }
.metric-card {
    background: var(--color-accent-light);
    border: 1px solid var(--color-border);
    border-radius: 5px;
    padding: 10px 16px;
    min-width: 140px;
}
.metric-card .metric-label {
    font-size: 10.5px;
    text-transform: uppercase;
    letter-spacing: 0.4px;
    color: var(--color-text-muted);
    margin-bottom: 2px;
}
.metric-card .metric-value {
    font-family: var(--font-mono);
    font-size: 17px;
    font-weight: 600;
    color: var(--color-text);
}
.metric-card .metric-sublabel {
    font-size: 10.5px;
    color: var(--color-text-muted);
    margin-top: 2px;
}
.badge {
    display: inline-block;
    font-size: 11px;
    font-weight: 700;
    padding: 2px 8px;
    border-radius: 3px;
    text-transform: uppercase;
    letter-spacing: 0.3px;
}
.badge-pass { color: var(--color-pass); background: var(--color-pass-bg); }
.badge-fail { color: var(--color-fail); background: var(--color-fail-bg); }
.warning-list {
    margin: 8px 0 0 0;
    padding: 10px 14px;
    background: var(--color-fail-bg);
    border: 1px solid #f0c4be;
    border-radius: 5px;
    list-style: none;
}
.warning-list li {
    font-size: 12.5px;
    color: var(--color-fail);
    padding: 3px 0;
}
.warning-list li::before { content: "\\26A0  "; }
.no-warnings {
    font-size: 12.5px;
    color: var(--color-pass);
    background: var(--color-pass-bg);
    border: 1px solid #bfe2cb;
    border-radius: 5px;
    padding: 8px 14px;
    display: inline-block;
}
table.report-table {
    border-collapse: collapse;
    width: 100%;
    font-size: 12.5px;
    margin-top: 4px;
}
table.report-table th {
    text-align: right;
    padding: 6px 10px;
    background: var(--color-accent-light);
    color: var(--color-accent);
    font-weight: 600;
    border-bottom: 1px solid var(--color-border);
    white-space: nowrap;
}
table.report-table th:first-child, table.report-table td:first-child { text-align: left; }
table.report-table td {
    text-align: right;
    padding: 5px 10px;
    border-bottom: 1px solid #eef0f3;
    font-family: var(--font-mono);
    white-space: nowrap;
}
table.report-table tr:hover td { background: #fafbfc; }
.chart-grid { display: flex; flex-wrap: wrap; gap: 16px; }
.chart-cell { flex: 1 1 460px; min-width: 0; }
.caption {
    font-size: 11.5px;
    color: var(--color-text-muted);
    margin: 4px 0 0 0;
}
footer.report-footer {
    text-align: center;
    font-size: 11px;
    color: var(--color-text-muted);
    padding: 24px 0 8px 0;
    font-family: var(--font-mono);
}
"""


def render_page(title: str, subtitle: str, sections_html: list[str], nav_items: list[tuple[str, str]]) -> str:
    """
    Assemble the full standalone HTML document.

    Parameters
    ----------
    title : str
        Report title, shown in the page header and `<title>` tag.
    subtitle : str
        One-line model/run summary shown beneath the title.
    sections_html : list[str]
        Pre-rendered section HTML blocks (see :func:`render_section`),
        concatenated in order.
    nav_items : list[tuple[str, str]]
        ``(anchor_id, label)`` pairs for the sticky navigation bar.

    Returns
    -------
    str
        Complete HTML document.
    """
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    nav_html = "".join(f'<a href="#{anchor}">{label}</a>' for anchor, label in nav_items)
    body = "\n".join(sections_html)
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{title}</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<script src="{_PLOTLY_CDN}"></script>
<style>{_CSS}</style>
</head>
<body>
<header class="report-header">
    <h1>{title}</h1>
    <p>{subtitle} &mdash; generated {generated_at}</p>
</header>
<nav class="report-nav">{nav_html}</nav>
<main>
{body}
</main>
<footer class="report-footer">
    credit-copula &mdash; one-factor Gaussian copula tranche pricer &mdash; static report, no server required
</footer>
<script>
window.addEventListener("load", function () {{
    document.querySelectorAll(".js-plotly-plot").forEach(function (gd) {{
        Plotly.Plots.resize(gd);
    }});
}});
</script>
</body>
</html>
"""


def render_section(anchor_id: str, title: str, caption: str, content_html: str) -> str:
    """
    Wrap content in a titled, anchored report section.

    Parameters
    ----------
    anchor_id : str
        HTML anchor id, referenced by the navigation bar.
    title : str
        Section heading.
    caption : str
        Short one-line technical caption (no theory, per dashboard
        design constraints -- detailed explanations belong in
        `THEORY.md`).
    content_html : str
        Pre-rendered inner HTML (tables, metric rows, chart grids).

    Returns
    -------
    str
        HTML for the complete section.
    """
    return f"""<section class="report-section" id="{anchor_id}">
    <h2>{title}</h2>
    <p class="section-caption">{caption}</p>
    {content_html}
</section>"""


def metric_card(label: str, value: str, sublabel: str = "") -> str:
    """Render a single compact key-metric tile."""
    sub_html = f'<div class="metric-sublabel">{sublabel}</div>' if sublabel else ""
    return f"""<div class="metric-card">
    <div class="metric-label">{label}</div>
    <div class="metric-value">{value}</div>
    {sub_html}
</div>"""


def render_metric_row(cards: list[str]) -> str:
    """Lay out a row of metric cards (see :func:`metric_card`)."""
    return f'<div class="metric-row">{"".join(cards)}</div>'


def status_badge(label: str, passed: bool) -> str:
    """Render a pass/fail status badge."""
    css_class = "badge-pass" if passed else "badge-fail"
    text = "PASS" if passed else "FAIL"
    return f'<span class="badge {css_class}">{label}: {text}</span>'


def render_warnings(warnings: list[str]) -> str:
    """
    Render a calibration warning list, or a clean-bill-of-health indicator if empty.

    Parameters
    ----------
    warnings : list[str]
        Warning messages, e.g. from
        :func:`credit_copula.diagnostics.generate_calibration_warnings`.

    Returns
    -------
    str
        HTML warning list, or a "no warnings" indicator.
    """
    if not warnings:
        return '<div class="no-warnings">No calibration warnings triggered.</div>'
    items = "".join(f"<li>{w}</li>" for w in warnings)
    return f'<ul class="warning-list">{items}</ul>'


def render_table(df: pd.DataFrame, format_map: dict[str, str] | None = None) -> str:
    """
    Render a `pandas.DataFrame` as a styled HTML table.

    Parameters
    ----------
    df : pd.DataFrame
        Table to render.
    format_map : dict[str, str], optional
        Per-column Python format strings (e.g. ``{"par_spread_bps": "{:.1f}"}``)
        applied before rendering. Columns not listed are rendered with
        their default string representation.

    Returns
    -------
    str
        HTML ``<table>`` markup.
    """
    formatted = df.copy()
    if format_map:
        for column, fmt in format_map.items():
            if column in formatted.columns:
                formatted[column] = formatted[column].map(lambda v, fmt=fmt: fmt.format(v))
    return formatted.to_html(index=False, classes="report-table", border=0, escape=True)
