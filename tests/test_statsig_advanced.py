"""Round 19 — advanced targeting conditions + Pulse statistics refinements,
each closing a real delta vs the external Statsig product.

Targeting: regex / ip_in_cidr / before-after (time) / in_segment (ID lists).
Pulse:     Benjamini-Hochberg FDR / ratio metrics (delta method) /
           sequential confidence intervals / Winsorization.
"""
from __future__ import annotations

import pytest

from user_soul.statsig_user import StatsigUser
from user_soul.experiment_manager import (
    ExperimentManager, FeatureGateConfig,
)
from user_soul.targeting import Condition, TargetingRule
from user_soul.pulse import PulseComputer


def _gate(mgr, name, *conditions):
    mgr.add_gate(FeatureGateConfig(name, 0.0,
                 rules=[TargetingRule("rule", list(conditions))]))


# ─── Targeting: regex ────────────────────────────────────────────────────────

def test_regex_condition():
    mgr = ExperimentManager()
    _gate(mgr, "internal", Condition("regex", "email", r".*@(corp|staff)\.com$"))
    assert mgr.check_gate(StatsigUser("a", email="x@corp.com"), "internal") is True
    assert mgr.check_gate(StatsigUser("b", email="y@staff.com"), "internal") is True
    assert mgr.check_gate(StatsigUser("c", email="z@gmail.com"), "internal") is False


def test_regex_invalid_pattern_is_safe():
    mgr = ExperimentManager()
    _gate(mgr, "g", Condition("regex", "email", r"([unclosed"))
    assert mgr.check_gate(StatsigUser("a", email="x@y.com"), "g") is False


# ─── Targeting: IP / CIDR ────────────────────────────────────────────────────

def test_ip_in_cidr():
    mgr = ExperimentManager()
    _gate(mgr, "office", Condition("ip_in_cidr", "ip", ["10.0.0.0/8", "192.168.1.0/24"]))
    assert mgr.check_gate(StatsigUser("a", ip="10.4.5.6"), "office") is True
    assert mgr.check_gate(StatsigUser("b", ip="192.168.1.50"), "office") is True
    assert mgr.check_gate(StatsigUser("c", ip="8.8.8.8"), "office") is False


def test_ip_malformed_is_false():
    mgr = ExperimentManager()
    _gate(mgr, "g", Condition("ip_in_cidr", "ip", "10.0.0.0/8"))
    assert mgr.check_gate(StatsigUser("a", ip="not-an-ip"), "g") is False


# ─── Targeting: time / date (before / after) ─────────────────────────────────

def test_before_after_dates():
    mgr = ExperimentManager()
    _gate(mgr, "early_adopter", Condition("before", "custom.created_at", "2024-01-01"))
    assert mgr.check_gate(
        StatsigUser("a", custom={"created_at": "2023-06-15"}), "early_adopter") is True
    assert mgr.check_gate(
        StatsigUser("b", custom={"created_at": "2025-06-15"}), "early_adopter") is False


def test_after_with_epoch_numbers():
    mgr = ExperimentManager()
    _gate(mgr, "g", Condition("after", "custom.ts", 1000))
    assert mgr.check_gate(StatsigUser("a", custom={"ts": 2000}), "g") is True
    assert mgr.check_gate(StatsigUser("b", custom={"ts": 500}), "g") is False


# ─── Targeting: segments / uploaded ID lists ─────────────────────────────────

def test_in_segment_id_list():
    mgr = ExperimentManager()
    mgr.add_id_list("vip", ["alice", "carol"])
    _gate(mgr, "vip_perk", Condition("in_segment", target="vip"))
    assert mgr.check_gate(StatsigUser("alice"), "vip_perk") is True
    assert mgr.check_gate(StatsigUser("bob"), "vip_perk") is False


def test_segment_survives_export_reload():
    mgr = ExperimentManager()
    mgr.add_id_list("vip", ["alice", "carol"])
    _gate(mgr, "vip_perk", Condition("in_segment", target="vip"))
    mgr2 = ExperimentManager.from_config(mgr.export_config())
    assert mgr2.check_gate(StatsigUser("alice"), "vip_perk") is True
    assert mgr2.check_gate(StatsigUser("bob"), "vip_perk") is False


def test_not_in_segment():
    mgr = ExperimentManager()
    mgr.add_id_list("banned", ["mallory"])
    _gate(mgr, "allowed", Condition("not_in_segment", target="banned"))
    assert mgr.check_gate(StatsigUser("alice"), "allowed") is True
    assert mgr.check_gate(StatsigUser("mallory"), "allowed") is False


# ─── Pulse: Benjamini-Hochberg FDR ───────────────────────────────────────────

def test_bh_less_conservative_than_bonferroni():
    pc = PulseComputer(alpha=0.05)
    ps = [0.001, 0.01, 0.02, 0.5]
    bh = pc.benjamini_hochberg(ps, 0.05)
    bonf = [p < 0.05 / len(ps) for p in ps]
    assert bh == [True, True, True, False]   # BH passes 3
    assert bonf == [True, True, False, False]  # Bonferroni passes 2
    assert sum(bh) > sum(bonf)


def test_bh_all_null():
    pc = PulseComputer()
    assert pc.benjamini_hochberg([0.6, 0.7, 0.9], 0.05) == [False, False, False]


def test_report_carries_both_corrections():
    pc = PulseComputer()
    report = pc.compute_from_rates("m", {
        "a": (1000, 200, 1000, 300),   # strong
        "b": (1000, 500, 1000, 505),   # noise
    })
    for m in report.metrics:
        assert isinstance(m.significant_corrected, bool)
        assert isinstance(m.significant_fdr, bool)


# ─── Pulse: sequential confidence interval ───────────────────────────────────

def test_sequential_ci_wider_than_fixed():
    import math
    pc = PulseComputer()
    n1 = n2 = 1000
    report = pc.compute_from_rates("m", {"conv": (n1, 200, n2, 300)})
    m = report.metrics[0]
    # Fixed 95% CI on the ABSOLUTE diff (same scale as the seq CI): ±1.96·se.
    p_pool = (200 + 300) / (n1 + n2)
    se = math.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))
    fixed_abs_w = 2 * 1.96 * se
    seq_w = m.seq_ci_upper - m.seq_ci_lower
    assert seq_w > fixed_abs_w   # anytime-valid CI pays width for the right to peek


def test_sequential_ci_brackets_estimate():
    pc = PulseComputer()
    m = pc.compute_from_rates("m", {"conv": (1000, 200, 1000, 300)}).metrics[0]
    assert m.seq_ci_lower <= m.absolute_diff <= m.seq_ci_upper


# ─── Pulse: ratio metrics (delta method) ─────────────────────────────────────

def test_ratio_metric_estimates_aggregate_ratio():
    pc = PulseComputer()
    # control CTR ≈ 0.10, treatment CTR ≈ 0.20
    cn = [1.0, 1.0, 1.0, 1.0] * 25
    cd = [10.0] * 100
    tn = [2.0, 2.0, 2.0, 2.0] * 25
    td = [10.0] * 100
    report = pc.compute_from_ratios("ctr", {"ctr": (cn, cd, tn, td)})
    m = report.metrics[0]
    assert m.metric_type == "ratio"
    assert abs(m.control_value - 0.10) < 1e-6
    assert abs(m.treatment_value - 0.20) < 1e-6
    assert m.absolute_diff > 0


def test_ratio_metric_detects_significant_lift():
    pc = PulseComputer()
    cn = [1.0] * 200
    cd = [10.0] * 200
    tn = [2.0] * 200
    td = [10.0] * 200
    m = pc.compute_from_ratios("ctr", {"ctr": (cn, cd, tn, td)}).metrics[0]
    # clean, large, zero-variance-within-arm separation → significant
    assert m.significant is True
    assert m.lift > 0.5


# ─── Pulse: Winsorization ────────────────────────────────────────────────────

def test_winsorize_caps_outliers():
    pc = PulseComputer()
    vals = [1.0] * 98 + [1000.0, -1000.0]
    capped, did = pc.winsorize(vals, pct=0.01)
    assert did is True
    assert max(capped) < 1000.0 and min(capped) > -1000.0


def test_winsorize_shrinks_outlier_driven_lift():
    pc = PulseComputer()
    control = [1.0 + (i % 5) * 0.01 for i in range(100)]          # mild spread
    treatment = [1.0 + (i % 5) * 0.01 for i in range(99)] + [500.0]  # one extreme outlier
    raw = pc.compute_from_values("raw", {"m": (control, treatment)}).metrics[0]
    wins = pc.compute_from_values(
        "wins", {"m": (control, treatment)}, winsorize_pct=0.02).metrics[0]
    assert wins.winsorized is True
    assert abs(wins.lift) < abs(raw.lift)   # outlier influence reduced
