"""Tests for SeriesDiagnostics + KiX adapter — the dashboard problem-finder."""
from __future__ import annotations

from user_soul.series_diagnostics import SeriesDiagnostics, MetricFinding
from user_soul.kix_adapter import to_metric_rows


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


def test_adverse_polarity_error_metric_up_is_problem():
    d = SeriesDiagnostics(baseline_window=14)
    vals = _flat(20, 10.0); vals[-1] = 80.0
    metrics = [{"metric_id": "err", "title": "错误率 error rate", "domain": "monitoring",
                "dates": [f"d{i}" for i in range(20)], "values": vals}]
    f = d.diagnose(metrics)[0]
    assert f.adverse is True
    assert f.severity in ("P1", "P2")


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
