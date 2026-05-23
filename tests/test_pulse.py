"""Tests for PulseComputer."""
from __future__ import annotations

import pytest
from user_soul.pulse import PulseComputer, PulseReport, PulseMetric


@pytest.fixture
def pc():
    return PulseComputer(alpha=0.05)


def test_z_test_equal_proportions(pc):
    lift, p_value, ci_lower, ci_upper = pc.z_test_proportion(100, 50, 100, 50)
    assert abs(lift) < 0.01
    assert p_value > 0.5


def test_z_test_significant_lift(pc):
    # 30% control vs 60% treatment — should be highly significant
    lift, p_value, ci_lower, ci_upper = pc.z_test_proportion(100, 30, 100, 60)
    assert lift > 0.9  # ~100% lift
    assert p_value < 0.05
    assert ci_lower > 0  # positive CI


def test_z_test_insufficient_sample(pc):
    lift, p_value, ci_lower, ci_upper = pc.z_test_proportion(1, 1, 100, 50)
    assert lift == 0.0
    assert p_value == 1.0
    assert ci_lower == -1.0
    assert ci_upper == 1.0


def test_z_test_insufficient_sample_both(pc):
    result = pc.z_test_proportion(0, 0, 0, 0)
    assert result == (0.0, 1.0, -1.0, 1.0)


def test_power_estimate_large_sample(pc):
    # With n=10000 per arm and effect_size=0.1: Phi(0.1*sqrt(10000)/2 - 1.96) = Phi(3.04) ≈ 0.998
    power = pc.power_estimate(10000, 10000, effect_size=0.1)
    assert power > 0.8


def test_power_estimate_small_sample(pc):
    power = pc.power_estimate(10, 10, effect_size=0.1)
    assert power < 0.5


def test_compute_from_rates_winner(pc):
    # All metrics improve substantially
    metrics = {
        "acquisition": (200, 40, 200, 80),   # 20% → 40%
        "retention":   (200, 60, 200, 100),  # 30% → 50%
        "revenue":     (200, 20, 200, 50),   # 10% → 25%
    }
    report = pc.compute_from_rates("winner_test", metrics)
    assert report.overall_verdict == "winner"
    assert report.significant_wins > 0
    assert report.significant_losses == 0


def test_compute_from_rates_loser(pc):
    # All metrics decline
    metrics = {
        "acquisition": (200, 80, 200, 40),   # 40% → 20%
        "retention":   (200, 100, 200, 60),  # 50% → 30%
    }
    report = pc.compute_from_rates("loser_test", metrics)
    assert report.overall_verdict == "loser"
    assert report.significant_losses > 0


def test_compute_from_rates_underpowered(pc):
    # Tiny samples → underpowered
    metrics = {
        "acquisition": (5, 2, 5, 3),
        "retention":   (5, 3, 5, 4),
    }
    report = pc.compute_from_rates("underpower_test", metrics)
    assert report.overall_verdict == "underpowered"
    assert report.power_estimate < 0.8


def test_compute_from_values_continuous(pc):
    control = [1.0, 1.1, 0.9, 1.0, 1.2, 0.8, 1.0, 1.1, 0.95, 1.05]
    treatment = [2.0, 2.1, 1.9, 2.0, 2.2, 1.8, 2.0, 2.1, 1.95, 2.05]
    report = pc.compute_from_values("continuous_test", {"session_length": (control, treatment)})
    assert report.metrics[0].metric_type == "continuous"
    assert report.metrics[0].lift > 0.5


def test_pulse_metric_lift_formula(pc):
    # lift = (treatment - control) / control
    # control = 20/100 = 0.2, treatment = 40/100 = 0.4 → lift = 1.0
    lift, _, _, _ = pc.z_test_proportion(100, 20, 100, 40)
    expected_lift = (0.4 - 0.2) / 0.2
    assert abs(lift - expected_lift) < 0.01


def test_summary_string(pc):
    metrics = {"m1": (100, 50, 100, 50)}
    report = pc.compute_from_rates("summary_test", metrics)
    assert isinstance(report.summary, str)
    assert len(report.summary) > 0
    assert "Verdict:" in report.summary


def test_significant_wins_losses_count(pc):
    metrics = {
        "win1":  (200, 40, 200, 80),   # positive significant
        "win2":  (200, 30, 200, 70),   # positive significant
        "loss1": (200, 80, 200, 40),   # negative significant
    }
    report = pc.compute_from_rates("count_test", metrics)
    assert report.significant_wins == 2
    assert report.significant_losses == 1
