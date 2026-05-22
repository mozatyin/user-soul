"""Tests for ABValidator — Phase 8.8 A/B validation with PDCA actions."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from unittest.mock import MagicMock, patch

import pytest

from user_soul.ab_validator import (
    ABValidationReport,
    ABValidator,
    PDCAAction,
    _make_recommendation,
    _priority_from_delta,
)
from user_soul.report import CompareReport, SimulationReport


# ---------------------------------------------------------------------------
# Helpers to build minimal fake SimulationReport / CompareReport
# ---------------------------------------------------------------------------

def _fake_sim_report(label: str = "product") -> SimulationReport:
    return SimulationReport(
        n_simulations=10,
        user_type="test user",
        product_summary=label,
        metrics={},
    )


def _fake_compare(
    improvements: list[str],
    regressions: list[str],
    deltas: dict[str, float] | None = None,
) -> CompareReport:
    if deltas is None:
        # default: improvement metrics have delta +0.15, regression metrics have delta +0.12
        deltas = {}
        for m in improvements:
            deltas[m] = 0.15
        for m in regressions:
            deltas[m] = 0.12
    return CompareReport(
        n_runs_per_variant=10,
        variant_a_label="ours",
        variant_b_label="reference",
        variant_a=_fake_sim_report("ours"),
        variant_b=_fake_sim_report("reference"),
        deltas=deltas,
        improvements=improvements,
        regressions=regressions,
        key_diff="",
    )


def _make_backend(response: str = '{"recommendation": "Fix it now"}') -> MagicMock:
    backend = MagicMock()
    backend.text.return_value = response
    return backend


# ---------------------------------------------------------------------------
# 1. test_priority_from_delta
# ---------------------------------------------------------------------------

def test_priority_from_delta():
    assert _priority_from_delta(-0.15) == "P0"   # delta < -0.10
    assert _priority_from_delta(-0.07) == "P1"   # -0.10 ≤ delta < -0.05
    assert _priority_from_delta(-0.03) == "P2"   # delta ≥ -0.05
    # boundary values
    assert _priority_from_delta(-0.10) == "P1"   # exactly -0.10 → P1
    assert _priority_from_delta(-0.05) == "P2"   # exactly -0.05 → P2
    assert _priority_from_delta(0.0) == "P2"     # positive delta → P2


# ---------------------------------------------------------------------------
# 2. test_verdict_beats_reference
# ---------------------------------------------------------------------------

def test_verdict_beats_reference():
    """2 improvements, 0 regressions → beats_reference."""
    backend = _make_backend('{"summary": "We win"}')
    validator = ABValidator(backend=backend)

    compare_report = _fake_compare(
        improvements=["engagement_score", "day1_return_rate"],
        regressions=[],
        deltas={"engagement_score": 0.15, "day1_return_rate": 0.12},
    )

    with patch("user_soul.ab_validator.build_domain_config") as mock_cfg, \
         patch("user_soul.ab_validator.UserSimulator") as MockSim:
        mock_cfg.return_value = MagicMock()
        mock_sim_instance = MagicMock()
        mock_sim_instance.compare.return_value = compare_report
        MockSim.return_value = mock_sim_instance

        report = validator.validate(
            our_product="Our chess app",
            reference_product="Chess.com",
            user_type="casual beginner",
            goal="learn chess",
            n_runs=5,
        )

    assert report.verdict == "beats_reference"


# ---------------------------------------------------------------------------
# 3. test_verdict_below_reference
# ---------------------------------------------------------------------------

def test_verdict_below_reference():
    """1 P0 regression (delta < -0.10) → below_reference."""
    # recommendation call + summary call
    backend = _make_backend('{"recommendation": "Add tutorial flow"}')
    backend.text.side_effect = [
        '{"recommendation": "Add tutorial flow"}',
        '{"summary": "Below reference, fix P0 first"}',
    ]
    validator = ABValidator(backend=backend)

    # regressions delta = +0.15 in compare (reference better by 0.15)
    # → our_minus_ref = -0.15 → P0
    compare_report = _fake_compare(
        improvements=[],
        regressions=["day1_return_rate"],
        deltas={"day1_return_rate": 0.15},
    )

    with patch("user_soul.ab_validator.build_domain_config") as mock_cfg, \
         patch("user_soul.ab_validator.UserSimulator") as MockSim:
        mock_cfg.return_value = MagicMock()
        MockSim.return_value.compare.return_value = compare_report

        report = validator.validate(
            our_product="Our app",
            reference_product="Reference app",
            user_type="user",
            goal="goal",
        )

    assert report.verdict == "below_reference"


# ---------------------------------------------------------------------------
# 4. test_verdict_at_parity
# ---------------------------------------------------------------------------

def test_verdict_at_parity():
    """improvements == regressions → at_parity (when no P0)."""
    # 1 improvement, 1 regression (delta -0.07 → P1, not P0)
    backend = MagicMock()
    backend.text.side_effect = [
        '{"recommendation": "Improve retention screen"}',
        '{"summary": "At parity with reference"}',
    ]
    validator = ABValidator(backend=backend)

    compare_report = _fake_compare(
        improvements=["engagement_score"],
        regressions=["session_length"],
        deltas={"engagement_score": 0.10, "session_length": 0.07},
    )

    with patch("user_soul.ab_validator.build_domain_config") as mock_cfg, \
         patch("user_soul.ab_validator.UserSimulator") as MockSim:
        mock_cfg.return_value = MagicMock()
        MockSim.return_value.compare.return_value = compare_report

        report = validator.validate(
            our_product="Our app",
            reference_product="Reference",
            user_type="user",
            goal="goal",
        )

    assert report.verdict == "at_parity"


# ---------------------------------------------------------------------------
# 5. test_launch_ready_true
# ---------------------------------------------------------------------------

def test_launch_ready_true():
    """beats_reference + no P0 → launch_ready=True."""
    backend = MagicMock()
    backend.text.return_value = '{"summary": "Ship it"}'
    validator = ABValidator(backend=backend)

    compare_report = _fake_compare(
        improvements=["day1_return_rate", "engagement_score"],
        regressions=[],
        deltas={"day1_return_rate": 0.12, "engagement_score": 0.08},
    )

    with patch("user_soul.ab_validator.build_domain_config") as mock_cfg, \
         patch("user_soul.ab_validator.UserSimulator") as MockSim:
        mock_cfg.return_value = MagicMock()
        MockSim.return_value.compare.return_value = compare_report

        report = validator.validate(
            our_product="Our app",
            reference_product="Reference",
            user_type="user",
            goal="goal",
        )

    assert report.launch_ready is True


# ---------------------------------------------------------------------------
# 6. test_launch_ready_false
# ---------------------------------------------------------------------------

def test_launch_ready_false():
    """below_reference (P0 regression) → launch_ready=False."""
    backend = MagicMock()
    backend.text.side_effect = [
        '{"recommendation": "Overhaul onboarding"}',
        '{"summary": "Not ready, P0 issues remain"}',
    ]
    validator = ABValidator(backend=backend)

    compare_report = _fake_compare(
        improvements=[],
        regressions=["day1_return_rate"],
        deltas={"day1_return_rate": 0.20},  # our_minus_ref = -0.20 → P0
    )

    with patch("user_soul.ab_validator.build_domain_config") as mock_cfg, \
         patch("user_soul.ab_validator.UserSimulator") as MockSim:
        mock_cfg.return_value = MagicMock()
        MockSim.return_value.compare.return_value = compare_report

        report = validator.validate(
            our_product="Our app",
            reference_product="Reference",
            user_type="user",
            goal="goal",
        )

    assert report.launch_ready is False


# ---------------------------------------------------------------------------
# 7. test_pdca_actions_only_regressions
# ---------------------------------------------------------------------------

def test_pdca_actions_only_regressions():
    """Improvement metrics must NOT generate PDCAActions."""
    backend = MagicMock()
    backend.text.side_effect = [
        '{"recommendation": "Fix session length screen"}',
        '{"summary": "Mixed results"}',
    ]
    validator = ABValidator(backend=backend)

    compare_report = _fake_compare(
        improvements=["engagement_score", "day1_return_rate"],
        regressions=["session_length"],
        deltas={
            "engagement_score": 0.15,
            "day1_return_rate": 0.10,
            "session_length": 0.07,
        },
    )

    with patch("user_soul.ab_validator.build_domain_config") as mock_cfg, \
         patch("user_soul.ab_validator.UserSimulator") as MockSim:
        mock_cfg.return_value = MagicMock()
        MockSim.return_value.compare.return_value = compare_report

        report = validator.validate(
            our_product="Our app",
            reference_product="Reference",
            user_type="user",
            goal="goal",
        )

    # Only 1 PDCAAction for the regression metric
    assert len(report.pdca_actions) == 1
    assert report.pdca_actions[0].metric == "session_length"
    assert report.pdca_actions[0].direction == "regression"
    # improvement metrics not in pdca_actions
    action_metrics = {a.metric for a in report.pdca_actions}
    assert "engagement_score" not in action_metrics
    assert "day1_return_rate" not in action_metrics


# ---------------------------------------------------------------------------
# 8. test_full_validate_mock
# ---------------------------------------------------------------------------

def test_full_validate_mock():
    """Full validate() with mocked UserSimulator.compare() — verify structure."""
    backend = MagicMock()
    backend.text.side_effect = [
        '{"recommendation": "Simplify onboarding flow — reduce steps from 5 to 3"}',
        '{"summary": "Our app beats reference with 2 improvements and 1 regression."}',
    ]
    validator = ABValidator(backend=backend)

    compare_report = _fake_compare(
        improvements=["engagement_score", "hook_completion_rate"],
        regressions=["session_length"],
        deltas={
            "engagement_score": 0.18,
            "hook_completion_rate": 0.09,
            "session_length": 0.06,  # our_minus_ref = -0.06 → P1
        },
    )

    with patch("user_soul.ab_validator.build_domain_config") as mock_cfg, \
         patch("user_soul.ab_validator.UserSimulator") as MockSim:
        mock_cfg.return_value = MagicMock()
        mock_sim_instance = MagicMock()
        mock_sim_instance.compare.return_value = compare_report
        MockSim.return_value = mock_sim_instance

        report = validator.validate(
            our_product="Our chess app with daily puzzles",
            reference_product="Chess.com mobile app",
            user_type="casual chess beginner, age 20",
            goal="learn chess and enjoy daily puzzles",
            our_label="our_chess",
            reference_label="chess_com",
            n_runs=10,
        )

    # Structure assertions
    assert isinstance(report, ABValidationReport)
    assert report.our_label == "our_chess"
    assert report.reference_label == "chess_com"
    assert isinstance(report.compare, CompareReport)
    assert report.verdict in ("beats_reference", "at_parity", "below_reference")

    # PDCA actions
    assert len(report.pdca_actions) == 1
    action = report.pdca_actions[0]
    assert isinstance(action, PDCAAction)
    assert action.metric == "session_length"
    assert action.direction == "regression"
    assert action.delta == pytest.approx(-0.06)
    assert action.priority in ("P0", "P1", "P2")
    assert isinstance(action.recommendation, str) and len(action.recommendation) > 0

    # Summary present
    assert isinstance(report.summary, str) and len(report.summary) > 0

    # UserSimulator called with correct args
    mock_sim_instance.compare.assert_called_once_with(
        "Our chess app with daily puzzles",
        "Chess.com mobile app",
        label_a="our_chess",
        label_b="chess_com",
        n_runs=10,
        goal="learn chess and enjoy daily puzzles",
    )


# ---------------------------------------------------------------------------
# 9. test_summary_generated
# ---------------------------------------------------------------------------

def test_summary_generated():
    """summary field must be non-empty after validate()."""
    backend = MagicMock()
    backend.text.return_value = '{"summary": "Excellent — our app beats reference on all key metrics."}'
    validator = ABValidator(backend=backend)

    compare_report = _fake_compare(
        improvements=["day1_return_rate"],
        regressions=[],
        deltas={"day1_return_rate": 0.12},
    )

    with patch("user_soul.ab_validator.build_domain_config") as mock_cfg, \
         patch("user_soul.ab_validator.UserSimulator") as MockSim:
        mock_cfg.return_value = MagicMock()
        MockSim.return_value.compare.return_value = compare_report

        report = validator.validate(
            our_product="Our app",
            reference_product="Reference",
            user_type="user",
            goal="goal",
        )

    assert report.summary != ""
    assert len(report.summary) > 5
