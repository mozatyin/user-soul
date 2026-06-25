"""Multi-cut KiX diagnostic report — sweep every segment dimension, FDR per cut,
render one consolidated markdown a human/Angel can read next to the dashboard.

A single overall pass (Round 20) hides market/channel-local signals (Round 21:
the SG cut surfaced a +70% mover the aggregate averaged away). This sweeps the
interesting cuts — overall, by platform, by channel/source, by user type, and the
top markets (always incl. SG) — runs SeriesDiagnostics (FDR) within each, and
produces a ranked, cut-labelled finding list. FDR is applied INSIDE each cut, so
significance always accounts for how many metrics that cut tested.
"""
from __future__ import annotations

from user_soul.kix_adapter import (
    extract_charts_data, to_metric_rows, available_segments, top_segments,
)
from user_soul.series_diagnostics import SeriesDiagnostics

_SEV_HEADERS = [
    ("P1", "🔴 P1 — adverse, FDR-significant, |z|≥3"),
    ("P2", "🟠 P2 — adverse, FDR-significant"),
    ("positive", "🟢 Positive movers — FDR-significant, favourable"),
    ("watch", "🟡 Watch — raw-significant adverse, NOT FDR-confirmed"),
]


def default_cuts(charts_data: dict, top_markets: int = 6) -> list[tuple]:
    """Build the sweep: (label, tab_id, segment). Small dimensions sweep fully;
    by_country sweeps the top markets by volume, always including SG."""
    cuts: list[tuple] = [("overall", None, None)]
    for seg in sorted(available_segments(charts_data, "by_platform")):
        cuts.append((f"platform:{seg}", "by_platform", seg))
    for seg in sorted(available_segments(charts_data, "by_source")):
        cuts.append((f"source:{seg}", "by_source", seg))
    for seg in sorted(available_segments(charts_data, "by_user_type")):
        cuts.append((f"user:{seg}", "by_user_type", seg))
    markets = top_segments(charts_data, "by_country", top_markets)
    if "SG" not in markets:
        markets.append("SG")
    for m in markets:
        cuts.append((f"country:{m}", "by_country", m))
    return cuts


def multi_cut_diagnose(charts_data: dict, cuts: list[tuple] | None = None,
                       **diag_opts) -> dict:
    """Run SeriesDiagnostics for each cut. Returns {cut_label: [MetricFinding]}."""
    cuts = cuts if cuts is not None else default_cuts(charts_data)
    diag = SeriesDiagnostics(**(diag_opts or dict(
        baseline_window=14, alpha=0.05, winsorize_pct=0.05, min_volume=30)))
    out: dict = {}
    for label, tab_id, segment in cuts:
        rows = to_metric_rows(charts_data, tab_id=tab_id, segment=segment)
        if rows:
            out[label] = diag.diagnose(rows)
    return out


def render_markdown(results: dict, date: str = "", watch_limit: int = 15) -> str:
    """Consolidate per-cut findings into one ranked markdown report."""
    tagged: dict = {sev: [] for sev, _ in _SEV_HEADERS}
    n_metrics = 0
    for cut, findings in results.items():
        n_metrics += len(findings)
        for f in findings:
            if f.severity in tagged:
                tagged[f.severity].append((cut, f))

    lines = [f"# KiX Diagnostic Overlay — {date or 'latest'}", ""]
    lines.append(
        f"_{len(results)} cuts, {n_metrics} metric-series tested; "
        f"FDR (α=0.05) applied within each cut. A cut with K metrics expects "
        f"~{0.05:.0%}·K raw p<0.05 by chance — only FDR-significant rows are real._")
    lines.append("")

    any_finding = False
    for sev, header in _SEV_HEADERS:
        items = tagged[sev]
        if not items:
            continue
        any_finding = True
        items.sort(key=lambda cf: -abs(cf[1].z_score))
        if sev == "watch":
            shown, extra = items[:watch_limit], max(0, len(items) - watch_limit)
        else:
            shown, extra = items, 0
        lines.append(f"## {header} ({len(items)})")
        for cut, f in shown:
            arrow = "▲" if f.direction == "up" else "▼" if f.direction == "down" else "—"
            tag = " ⚠low-N" if f.low_sample else ""
            ratio = " (ratio—verify denominator)" if f.metric_nature == "ratio" else ""
            lines.append(
                f"- **[{cut}]** {f.title.strip()} {arrow} "
                f"{f.wow_change_pct:+.1f}% WoW · latest={f.latest_value:g} "
                f"(z={f.z_score:+.2f}, p={f.p_value:.4f}){tag}{ratio}")
        if extra:
            lines.append(f"- _…and {extra} more watch items_")
        lines.append("")

    if not any_finding:
        lines.append("_No metric cleared the watch threshold in any cut._")
    return "\n".join(lines).rstrip() + "\n"


def build_report(html: str, *, top_markets: int = 6, **diag_opts) -> str:
    """End-to-end: KiX report HTML → consolidated multi-cut markdown."""
    cd = extract_charts_data(html)
    results = multi_cut_diagnose(cd, default_cuts(cd, top_markets), **diag_opts)
    date = ""
    for findings in results.values():
        for f in findings:
            if f.latest_date:
                date = f.latest_date
                break
        if date:
            break
    return render_markdown(results, date=date)
