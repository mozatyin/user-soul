"""Tests for SeriesDiagnostics + KiX adapter — the dashboard problem-finder."""
from __future__ import annotations

from user_soul.series_diagnostics import SeriesDiagnostics, MetricFinding
from user_soul.kix_adapter import to_metric_rows
from user_soul.kix_report import multi_cut_diagnose, render_markdown, build_report


def _flat(n=20, base=100.0):
    return [base + (i % 3 - 1) for i in range(n)]  # tiny ±1 jitter, no trend


def test_detects_real_spike_as_significant():
    d = SeriesDiagnostics(baseline_window=14)
    vals = _flat(20, 100.0)
    vals[-1] = 200.0  # huge jump on the latest day
    f = d._analyze_one("m", "DAU", "growth", [f"d{i}" for i in range(20)], vals)
    assert f.direction == "up"
    assert abs(f.z_score) > 5
    assert f.p_value < 0.01


def test_flat_series_not_significant():
    d = SeriesDiagnostics()
    f = d._analyze_one("m", "DAU", "growth", [f"d{i}" for i in range(20)], _flat(20))
    assert f.direction == "flat"
    assert f.p_value > 0.05


def test_fdr_controls_false_alarms_across_many_metrics():
    """100 pure-noise metrics + 1 real spike → FDR flags ~the real one, not a
    flood of false positives (a raw p<0.05 board would light up ~5)."""
    d = SeriesDiagnostics(baseline_window=14)
    metrics = []
    for i in range(100):
        vals = [100 + ((i * 7 + j * 13) % 11 - 5) for j in range(20)]  # noise
        metrics.append({"metric_id": f"noise_{i}", "title": "x", "domain": "d",
                        "dates": [f"d{j}" for j in range(20)], "values": vals})
    spike = _flat(20, 100.0); spike[-1] = 300.0
    metrics.append({"metric_id": "REAL", "title": "DAU", "domain": "growth",
                    "dates": [f"d{j}" for j in range(20)], "values": spike})
    findings = d.diagnose(metrics)
    sig = [f for f in findings if f.significant_fdr]
    assert any(f.metric_id == "REAL" for f in sig)
    assert len(sig) <= 3  # FDR keeps the false-alarm flood out


def test_sustained_adverse_error_metric_is_problem():
    d = SeriesDiagnostics(baseline_window=14)
    vals = _flat(20, 10.0)
    vals[-4:] = [70.0, 75.0, 78.0, 80.0]   # sustained multi-day rise, not a blip
    metrics = [{"metric_id": "err", "title": "错误率 error rate", "domain": "monitoring",
                "dates": [f"d{i}" for i in range(20)], "values": vals}]
    f = d.diagnose(metrics)[0]
    assert f.adverse is True
    assert f.sustained is True
    assert f.severity in ("P1", "P2")


def test_single_day_adverse_spike_held_at_watch():
    d = SeriesDiagnostics(baseline_window=14)
    vals = _flat(20, 10.0); vals[-1] = 80.0   # one-day spike, not yet confirmed
    metrics = [{"metric_id": "err", "title": "错误率 error rate", "domain": "monitoring",
                "dates": [f"d{i}" for i in range(20)], "values": vals}]
    f = d.diagnose(metrics)[0]
    assert f.significant_fdr is True       # the day IS significant…
    assert f.sustained is False
    assert f.confirmation == "1d spike"
    assert f.severity == "watch"           # …but not confirmed → held at watch


def test_slow_bleed_caught_without_latest_day_spike():
    # a steady decline: baseline drifts down with it, so no single day is a |z|>2
    # outlier, yet the multi-day trend is real → must surface as "trend".
    d = SeriesDiagnostics(baseline_window=14, min_trend_drift=0.1)
    vals = [100.0 - 3.0 * i for i in range(20)]   # -3/day, smooth
    metrics = [{"metric_id": "ret", "title": "留存 retention", "domain": "growth",
                "dates": [f"d{i}" for i in range(20)], "values": vals}]
    f = d.diagnose(metrics)[0]
    assert f.slow_bleed is True
    assert f.trend_significant_fdr is True
    assert f.severity == "trend"


def test_trend_fdr_controls_spurious_slopes():
    # 60 noisy flat metrics + 1 real decline → trend-FDR keeps the noise out.
    d = SeriesDiagnostics(baseline_window=14, min_trend_drift=0.1)
    metrics = []
    for i in range(60):
        vals = [100 + ((i * 5 + j * 9) % 13 - 6) for j in range(20)]
        metrics.append({"metric_id": f"n{i}", "title": "留存 retention", "domain": "g",
                        "dates": [f"d{j}" for j in range(20)], "values": vals})
    metrics.append({"metric_id": "REAL", "title": "留存 retention", "domain": "g",
                    "dates": [f"d{j}" for j in range(20)],
                    "values": [100.0 - 3.0 * j for j in range(20)]})
    findings = d.diagnose(metrics)
    bleeds = [f for f in findings if f.slow_bleed]
    assert any(f.metric_id == "REAL" for f in bleeds)
    assert len(bleeds) <= 4   # not ~3 chance trends from 60 noise series


def test_favorable_mover_is_positive_not_problem():
    d = SeriesDiagnostics(baseline_window=14)
    vals = _flat(20, 100.0); vals[-1] = 260.0
    metrics = [{"metric_id": "rev", "title": "总收入 revenue", "domain": "revenue",
                "dates": [f"d{i}" for i in range(20)], "values": vals}]
    f = d.diagnose(metrics)[0]
    assert f.adverse is False
    assert f.severity == "positive"


def test_wow_change_computed():
    d = SeriesDiagnostics(baseline_window=14)
    vals = list(range(1, 21))  # 1..20, value 7 days ago = vals[-8] = 13, latest = 20
    f = d._analyze_one("m", "t", "d", [f"d{i}" for i in range(20)], vals)
    assert abs(f.wow_change_pct - (20 - 13) / 13 * 100) < 1e-3  # stored at 4dp


# ─── KiX adapter against the real charts-data schema ─────────────────────────

def _fake_charts_data():
    return {
        "platform_basics": [{
            "chart_id": "DAU__series__series_daily", "title": "日活跃用户数 (DAU)",
            "metric_id": "DAU", "business_domain": "Platform Basics",
            "tabs": [{"tab_id": "overall", "tab_label": "Overall", "kind": "line",
                      "meta": {"metric_nature": "additive", "value_unit": "number"},
                      "data": {"dates": ["05-27", "05-28"], "values": [4791.0, 4810.0]}}],
        }],
        "monitoring": [{
            "chart_id": "seg", "title": "segmented", "metric_id": "SEG",
            "business_domain": "Monitoring",
            "tabs": [{"tab_id": "overall", "meta": {"metric_nature": "additive"},
                      "data": {"dates": ["05-27"], "values": [{"a": 1}]}}],  # non-scalar
        }],
    }


def test_adapter_flattens_scalar_series_and_skips_nonscalar():
    rows = to_metric_rows(_fake_charts_data())
    ids = {r["metric_id"] for r in rows}
    assert "DAU" in ids
    assert "SEG" not in ids  # non-scalar values skipped, not crashed
    dau = next(r for r in rows if r["metric_id"] == "DAU")
    assert dau["metric_nature"] == "additive"
    assert dau["values"] == [4791.0, 4810.0]
    assert dau["domain"] == "Platform Basics"


def test_adapter_domain_filter():
    rows = to_metric_rows(_fake_charts_data(), domains=["platform_basics"])
    assert all(r["domain"] == "Platform Basics" for r in rows)


# ─── segmented tabs (by_country matrix) → per-segment series ──────────────────

def _charts_data_with_country():
    return {
        "platform_basics": [{
            "chart_id": "DAU", "title": "DAU", "metric_id": "DAU",
            "business_domain": "Platform Basics",
            "tabs": [
                {"tab_id": "overall", "meta": {"metric_nature": "additive"},
                 "data": {"dates": ["d1", "d2"], "values": [100.0, 110.0]}},
                {"tab_id": "by_country", "meta": {"metric_nature": "additive"},
                 "data": {"dates": ["d1", "d2"],
                          "series": ["US", "SG", "MY"],
                          "matrix": [[50.0, 9.0, 41.0], [55.0, 12.0, 43.0]]}},
            ],
        }],
    }


def test_segment_extraction_picks_right_column():
    from user_soul.kix_adapter import available_segments
    cd = _charts_data_with_country()
    rows = to_metric_rows(cd, tab_id="by_country", segment="SG")
    assert len(rows) == 1
    r = rows[0]
    assert r["metric_id"] == "DAU@SG"
    assert r["values"] == [9.0, 12.0]        # the SG column, not US/MY
    assert "SG" in r["title"]
    assert available_segments(cd, "by_country") == {"US", "SG", "MY"}


def test_segment_absent_metric_is_skipped():
    cd = _charts_data_with_country()
    rows = to_metric_rows(cd, tab_id="by_country", segment="JP")  # not in series
    assert rows == []


def test_overall_path_unchanged_when_no_segment():
    cd = _charts_data_with_country()
    rows = to_metric_rows(cd)  # default first tab = overall scalar
    assert rows[0]["metric_id"] == "DAU"
    assert rows[0]["values"] == [100.0, 110.0]


# ─── small-sample guard ──────────────────────────────────────────────────────

def test_low_sample_demotes_thin_count_metric():
    d = SeriesDiagnostics(baseline_window=14, min_volume=30)
    vals = [2 + (i % 2) for i in range(20)]; vals[-1] = 9  # tiny counts, "spike"
    metrics = [{"metric_id": "u", "title": "用户数 users", "domain": "growth",
                "dates": [f"d{i}" for i in range(20)], "values": vals,
                "metric_nature": "additive"}]
    f = d.diagnose(metrics)[0]
    assert f.low_sample is True
    assert f.severity in ("info", "watch")   # never P1/P2/positive on a handful


def test_ratio_metric_not_volume_gated():
    # a ratio at 0.1 magnitude must NOT be flagged low_sample by a count threshold
    d = SeriesDiagnostics(baseline_window=14, min_volume=30)
    vals = [0.10 + (i % 3 - 1) * 0.001 for i in range(20)]; vals[-1] = 0.30
    f = d._analyze_one("r", "留存 retention", "growth",
                       [f"d{i}" for i in range(20)], vals, metric_nature="ratio")
    assert f.low_sample is False


def test_high_volume_real_drop_stays_p1():
    d = SeriesDiagnostics(baseline_window=14, min_volume=30)
    vals = [5000 + (i % 5) * 10 for i in range(20)]; vals[-1] = 1000  # big drop, big N
    metrics = [{"metric_id": "dau", "title": "错误数 errors", "domain": "monitoring",
                "dates": [f"d{i}" for i in range(20)], "values": vals,
                "metric_nature": "additive"}]
    f = d.diagnose(metrics)[0]
    assert f.low_sample is False


# ─── multi-cut report ────────────────────────────────────────────────────────

def _cd_for_report():
    days = [f"d{i}" for i in range(20)]
    flat = [1000.0 + (i % 3 - 1) for i in range(20)]
    spike = list(flat); spike[-1] = 3000.0
    return {
        "revenue": [{
            "chart_id": "rev", "title": "revenue", "metric_id": "REV",
            "business_domain": "Revenue",
            "tabs": [
                {"tab_id": "overall", "meta": {"metric_nature": "additive"},
                 "data": {"dates": days, "values": flat}},
                {"tab_id": "by_country", "meta": {"metric_nature": "additive"},
                 "data": {"dates": days, "series": ["US", "SG"],
                          "matrix": [[flat[i], spike[i]] for i in range(20)]}},
            ],
        }],
    }


def test_multi_cut_diagnose_keys_by_cut():
    cd = _cd_for_report()
    res = multi_cut_diagnose(cd, [("overall", None, None), ("country:SG", "by_country", "SG")],
                             baseline_window=14, min_volume=30)
    assert set(res.keys()) == {"overall", "country:SG"}
    sg = [f for f in res["country:SG"] if f.severity == "positive"]
    assert sg and sg[0].metric_id == "REV@SG"


def test_render_markdown_has_sections_and_cut_labels():
    cd = _cd_for_report()
    res = multi_cut_diagnose(cd, [("country:SG", "by_country", "SG")],
                             baseline_window=14, min_volume=30)
    md = render_markdown(res, date="06-24")
    assert "# KiX Diagnostic Overlay — 06-24" in md
    assert "Positive movers" in md
    assert "[country:SG]" in md


def test_corroboration_across_cuts():
    from user_soul.kix_report import _corroboration
    from user_soul.series_diagnostics import MetricFinding

    def mf(mid, cut_adverse=True):
        f = MetricFinding(mid, "留存 retention", "g", "d", 0.1, 0.2, 0.05, -2.5,
                          0.01, -30.0, -0.1, "down", "ratio")
        f.adverse = cut_adverse
        f.severity = "watch"
        return f

    results = {
        "country:IN": [mf("RET@IN")],
        "country:PH": [mf("RET@PH")],
        "country:TH": [mf("RET@TH")],
        "country:US": [mf("OTHER@US")],   # different metric, only 1 cut
    }
    corr = _corroboration(results, min_cuts=3)
    titles = {t for t, _, _ in corr}
    assert "留存 retention" in titles
    base = next(c for c in corr if c[0] == "留存 retention")
    assert base[1] == 3                    # RET appeared adverse in 3 cuts
    assert "country:US" not in base[2]     # OTHER metric not merged in


def test_build_report_end_to_end_from_html():
    import json
    cd = _cd_for_report()
    html = ('<script type="application/json" id="charts-data">'
            + json.dumps(cd) + "</script>")
    md = build_report(html, top_markets=3, baseline_window=14, min_volume=30)
    assert "KiX Diagnostic Overlay" in md
    assert "revenue" in md and "[country:SG]" in md
