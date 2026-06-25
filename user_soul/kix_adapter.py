"""Adapter: KiX daily-report `charts-data` JSON → SeriesDiagnostics input.

The KiX "for AI" dashboard embeds a `<script type="application/json"
id="charts-data">` blob: {domain: [chart, ...]}, each chart =
{chart_id, title, metric_id, business_domain, tabs:[{tab_id, tab_label, meta:{
metric_nature, value_unit}, data:{dates:[...], values:[...]}}]}.

This flattens it into the {metric_id, title, domain, dates, values,
metric_nature} rows SeriesDiagnostics.diagnose() consumes. The "overall" tab is
used by default; pass tab_id to slice a specific segment (e.g. an SG cut).
"""
from __future__ import annotations

import json
import re


def extract_charts_data(html: str) -> dict:
    """Pull the embedded charts-data JSON out of a KiX report HTML page."""
    m = re.search(
        r'<script type="application/json" id="charts-data"[^>]*>(.*?)</script>',
        html, re.S)
    if not m:
        raise ValueError("charts-data script block not found")
    return json.loads(m.group(1).strip())


def _pick_tab(chart: dict, tab_id: str | None):
    tabs = chart.get("tabs") or []
    if not tabs:
        return None
    if tab_id:
        for t in tabs:
            if t.get("tab_id") == tab_id:
                return t
        return None
    return tabs[0]  # default: first/overall tab


def diagnose_report(html: str, *, tab_id: str | None = None,
                    domains: list[str] | None = None, **diag_opts):
    """One-shot: KiX report HTML → ranked SeriesDiagnostics findings."""
    from user_soul.series_diagnostics import SeriesDiagnostics
    rows = to_metric_rows(extract_charts_data(html), tab_id=tab_id, domains=domains)
    return SeriesDiagnostics(**diag_opts).diagnose(rows)


def _segment_series(tab: dict, segment: str):
    """Pull one segment's daily series from a segmented tab's
    {dates, series:[names], matrix:[[per-series per-date]]}. Returns (dates, values)
    or (None, None) if the segment is absent."""
    data = tab.get("data") or {}
    series = data.get("series")
    matrix = data.get("matrix")
    dates = data.get("dates")
    if not series or not matrix or segment not in series:
        return None, None
    col = series.index(segment)
    values = [(row[col] if isinstance(row, list) and col < len(row) else None)
              for row in matrix]
    return dates, values


def available_segments(charts_data: dict, tab_id: str) -> set[str]:
    """All segment names present under a given tab (e.g. every country in by_country)."""
    segs: set[str] = set()
    for charts in charts_data.values():
        for chart in charts:
            tab = _pick_tab(chart, tab_id)
            if tab:
                segs.update((tab.get("data") or {}).get("series") or [])
    return segs


def to_metric_rows(charts_data: dict, tab_id: str | None = None,
                   domains: list[str] | None = None,
                   segment: str | None = None) -> list[dict]:
    """Flatten charts-data into SeriesDiagnostics rows.

    tab_id:  which tab to read (None → first/overall).
    segment: if given, read that segment's column from the tab's series/matrix
             (e.g. tab_id="by_country", segment="SG" → the SG-only series for
             every metric broken down by country). None → the tab's scalar
             `values` series.
    domains: restrict to these business domains (None → all).
    """
    rows: list[dict] = []
    for domain, charts in charts_data.items():
        if domains and domain not in domains:
            continue
        for chart in charts:
            tab = _pick_tab(chart, tab_id)
            if not tab:
                continue
            meta = tab.get("meta") or {}
            base_id = chart.get("metric_id") or chart.get("chart_id", "?")
            title = chart.get("title", "")

            if segment is not None:
                dates, values = _segment_series(tab, segment)
                if values is None:
                    continue  # this metric has no breakdown for that segment
                metric_id = f"{base_id}@{segment}"
                title = f"{title} ({segment})"
            else:
                data = tab.get("data") or {}
                dates, values = data.get("dates"), data.get("values")
                metric_id = base_id

            if not values or not isinstance(values, list):
                continue
            # skip non-scalar series (some charts hold dicts/segments in values)
            if not all(isinstance(v, (int, float)) or v is None for v in values):
                continue
            rows.append({
                "metric_id": metric_id,
                "title": title,
                "domain": chart.get("business_domain") or domain,
                "dates": dates or [],
                "values": values,
                "metric_nature": meta.get("metric_nature", "additive"),
                "value_unit": meta.get("value_unit", ""),
            })
    return rows
