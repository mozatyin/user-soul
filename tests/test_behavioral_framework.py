"""Tests for mcv.behavioral_framework constants and their integration with report.py."""
from __future__ import annotations
import pytest
from user_soul.behavioral_framework import (
    SYCOPHANCY_DEFLATOR,
    BEHAVIORAL_FRAMEWORK_SECTION,
    ADVERSARIAL_PERSONA_SECTION,
    BEHAVIORAL_METRICS,
    COGNITIVE_BUDGETS,
)
from user_soul.report import SimulationReport, MetricResult


# ---------------------------------------------------------------------------
# behavioral_framework module
# ---------------------------------------------------------------------------

def test_deflator_value():
    assert SYCOPHANCY_DEFLATOR == 0.70


def test_framework_section_formattable():
    s = BEHAVIORAL_FRAMEWORK_SECTION.format(cognitive_budget=15)
    assert "15" in s
    assert len(s) > 50


def test_adversarial_persona_non_empty():
    assert len(ADVERSARIAL_PERSONA_SECTION) > 50


def test_behavioral_metrics_structure():
    assert len(BEHAVIORAL_METRICS) >= 2
    for name, typ, question in BEHAVIORAL_METRICS:
        assert isinstance(name, str) and name
        assert typ in ("bool", "scale_1_5", "text")
        assert isinstance(question, str) and question


def test_cognitive_budgets_required_keys():
    for k in ("casual", "core", "adversarial", "default"):
        assert k in COGNITIVE_BUDGETS
        assert isinstance(COGNITIVE_BUDGETS[k], int)
        assert COGNITIVE_BUDGETS[k] > 0


def test_adversarial_budget_less_than_default():
    assert COGNITIVE_BUDGETS["adversarial"] < COGNITIVE_BUDGETS["default"]


# ---------------------------------------------------------------------------
# SimulationReport new properties
# ---------------------------------------------------------------------------

def _make_report(true_rate: float | None, hook_rate: float | None = None) -> SimulationReport:
    metrics = {}
    if true_rate is not None:
        metrics["day1_return_intent"] = MetricResult(
            name="day1_return_intent", type="bool",
            true_rate=true_rate, n_samples=10,
        )
    if hook_rate is not None:
        metrics["hook_completed"] = MetricResult(
            name="hook_completed", type="bool",
            true_rate=hook_rate, n_samples=10,
        )
    return SimulationReport(
        n_simulations=10, user_type="test", product_summary="x",
        metrics=metrics,
    )


def test_day1_return_rate_adjusted():
    r = _make_report(0.40)
    assert r.day1_return_rate_adjusted == pytest.approx(0.28, abs=0.001)


def test_day1_return_rate_adjusted_none():
    r = _make_report(None)
    assert r.day1_return_rate_adjusted is None


def test_hook_completion_rate_present():
    r = _make_report(0.40, hook_rate=0.60)
    assert r.hook_completion_rate == pytest.approx(0.60)


def test_hook_completion_rate_absent():
    r = _make_report(0.40)
    assert r.hook_completion_rate is None


def test_benchmark_context_below_survival():
    r = _make_report(0.05)   # adjusted = 0.035
    assert "survival" in r.benchmark_context.lower()


def test_benchmark_context_excellent():
    r = _make_report(0.60)   # adjusted = 0.42
    assert "excellent" in r.benchmark_context.lower()


def test_benchmark_context_none():
    r = _make_report(None)
    assert r.benchmark_context == ""


def test_adversarial_frictions_default_empty():
    r = _make_report(0.40)
    assert r.adversarial_frictions == []
