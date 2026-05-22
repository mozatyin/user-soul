"""Tests for CompareReport and _compute_compare."""
from user_soul.report import CompareReport, _compute_compare, MetricResult, SimulationReport
from user_soul.scenarios import ScenarioContext

CTX = ScenarioContext("evening", "calm", 1, "curiosity")


def _make_bool_report(true_rate: float, n: int = 30) -> SimulationReport:
    import math
    from user_soul.report import _wilson_ci
    lo, hi = _wilson_ci(true_rate, n)
    stdev = round(math.sqrt(true_rate * (1 - true_rate)), 4)
    mr = MetricResult(
        name="ret", type="bool", true_rate=true_rate,
        stdev=stdev, ci_95_low=lo, ci_95_high=hi, n_samples=n,
    )
    return SimulationReport(
        n_simulations=n, user_type="玩家", product_summary="游戏",
        metrics={"ret": mr},
    )


def _make_scale_report(mean: float, stdev: float, n: int = 30) -> SimulationReport:
    import math
    margin = 1.96 * stdev / math.sqrt(n)
    mr = MetricResult(
        name="eng", type="scale_1_5", mean=mean, distribution={},
        stdev=stdev,
        ci_95_low=round(max(1.0, mean - margin), 4),
        ci_95_high=round(min(5.0, mean + margin), 4),
        n_samples=n,
    )
    return SimulationReport(
        n_simulations=n, user_type="玩家", product_summary="游戏",
        metrics={"eng": mr},
    )


def test_compare_report_delta_positive_improvement():
    report_a = _make_bool_report(0.40, n=30)  # CI roughly [0.23, 0.59]
    report_b = _make_bool_report(0.73, n=30)  # well above A's CI
    compare = _compute_compare(report_a, report_b, "v1", "v2", key_diff="B更好")
    assert "ret" in compare.improvements
    assert "ret" not in compare.regressions
    assert abs(compare.deltas["ret"] - 0.33) < 0.05


def test_compare_report_regression():
    report_a = _make_bool_report(0.70, n=30)
    report_b = _make_bool_report(0.30, n=30)
    compare = _compute_compare(report_a, report_b, "v1", "v2", key_diff="A更好")
    assert "ret" in compare.regressions
    assert "ret" not in compare.improvements


def test_compare_report_no_significance_small_delta():
    report_a = _make_bool_report(0.50, n=30)
    report_b = _make_bool_report(0.53, n=30)  # within CI noise
    compare = _compute_compare(report_a, report_b, "v1", "v2", key_diff="similar")
    assert "ret" not in compare.improvements
    assert "ret" not in compare.regressions


def test_compare_report_scale_delta():
    report_a = _make_scale_report(3.0, stdev=0.8, n=30)
    report_b = _make_scale_report(4.2, stdev=0.8, n=30)
    compare = _compute_compare(report_a, report_b, "v1", "v2", key_diff="B higher")
    assert abs(compare.deltas["eng"] - 1.2) < 0.1
    assert "eng" in compare.improvements


def test_compare_report_fields():
    a = _make_bool_report(0.5)
    b = _make_bool_report(0.8)
    compare = _compute_compare(a, b, "old", "new", key_diff="new wins")
    assert compare.variant_a_label == "old"
    assert compare.variant_b_label == "new"
    assert compare.n_runs_per_variant == 30
    assert compare.key_diff == "new wins"
    assert compare.variant_a is a
    assert compare.variant_b is b
