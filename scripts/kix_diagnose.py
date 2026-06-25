#!/usr/bin/env python3
"""KiX dashboard diagnostic overlay — statistical problem-finder.

Fetches a KiX "for AI" daily report, runs SeriesDiagnostics over every metric,
and prints an FDR-controlled, severity-ranked finding list: the K metrics that
genuinely moved (vs the multiple-testing noise a 180-metric board throws), with
business-polarity so adverse moves surface as problems.

Usage:
    python scripts/kix_diagnose.py [URL] [--tab overall] [--domains a,b]
    python scripts/kix_diagnose.py --file path/to/report.html
    python scripts/kix_diagnose.py --tab by_country --segment SG   # SG-only cut

Default URL: https://daily-report-site.pages.dev/ai/angel/latest/
"""
from __future__ import annotations

import sys
import urllib.request

from user_soul.kix_adapter import extract_charts_data, to_metric_rows
from user_soul.series_diagnostics import SeriesDiagnostics

DEFAULT_URL = "https://daily-report-site.pages.dev/ai/angel/latest/"


def _fetch(url: str) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "user-soul-kix-diag/1.0"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read().decode("utf-8", "replace")


def main(argv: list[str]) -> int:
    url, path, tab, domains, segment = DEFAULT_URL, None, None, None, None
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == "--file":
            path = argv[i + 1]; i += 2
        elif a == "--tab":
            tab = argv[i + 1]; i += 2
        elif a == "--segment":
            segment = argv[i + 1]; i += 2
        elif a == "--domains":
            domains = argv[i + 1].split(","); i += 2
        elif not a.startswith("-"):
            url = a; i += 1
        else:
            i += 1

    html = open(path, encoding="utf-8", errors="replace").read() if path else _fetch(url)
    rows = to_metric_rows(extract_charts_data(html), tab_id=tab, domains=domains,
                          segment=segment)
    diag = SeriesDiagnostics(baseline_window=14, alpha=0.05, winsorize_pct=0.05)
    findings = diag.diagnose(rows)

    latest = max((r["dates"][-1] for r in rows if r["dates"]), default="?")
    print(f"KiX diagnostic — {len(rows)} metrics, latest {latest}")
    print(f"summary: {diag.summary(findings)}\n")
    shown = [f for f in findings if f.severity != "info"]
    if not shown:
        print("No metrics cleared the watch threshold.")
    for f in shown:
        arrow = "UP " if f.direction == "up" else "DN " if f.direction == "down" else "-- "
        print(f"[{f.severity:8}] {arrow}{f.domain[:16]:16} {f.title[:36]:36} "
              f"latest={f.latest_value:>11.2f} z={f.z_score:+6.2f} "
              f"p={f.p_value:.4f} wow={f.wow_change_pct:+7.1f}% fdr={f.significant_fdr}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
