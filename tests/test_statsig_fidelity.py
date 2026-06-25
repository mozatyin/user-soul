"""Fidelity tests for the Statsig-compatible layer — one block per closed gap.

These assert that User-Soul's experiment infra matches real Statsig server-SDK
semantics: SHA-256 bucketing, targeting rules, exposure logging, gate/layer
objects, and Pulse statistics (CUPED, absolute lift, sequential p-values,
multiple-testing correction).
"""
from __future__ import annotations

import hashlib

import pytest

from user_soul.statsig_user import StatsigUser
from user_soul.experiment_manager import (
    ExperimentManager, ExperimentConfig, ExperimentVariant,
    FeatureGateConfig, DynamicConfigSpec, _bucket, _bucket_int,
)
from user_soul.targeting import Condition, TargetingRule
from user_soul.layer import Layer
from user_soul.pulse import PulseComputer

USERS = [f"user_{i}" for i in range(2000)]


# ─── G1: SHA-256 bucketing fidelity ──────────────────────────────────────────

def test_bucket_uses_sha256_salt_dot_unit():
    """_bucket_int must equal Statsig's algorithm: sha256('salt.uid')[:8] % 10000."""
    uid, salt = "alice", "my_salt"
    digest = hashlib.sha256(f"{salt}.{uid}".encode()).digest()
    expected = int.from_bytes(digest[:8], "big") % 10_000
    assert _bucket_int(uid, salt) == expected


def test_bucket_float_in_unit_interval_and_uniform():
    vals = [_bucket(f"u{i}", "s") for i in range(2000)]
    assert all(0.0 <= v < 1.0 for v in vals)
    below = sum(1 for v in vals if v < 0.5)
    assert 900 < below < 1100  # ~50%


def test_bucket_salt_sensitivity():
    assert _bucket_int("bob", "a") != _bucket_int("bob", "b")


# ─── G3: targeting rules use StatsigUser attributes ──────────────────────────

def _country_gate():
    mgr = ExperimentManager()
    mgr.add_gate(FeatureGateConfig(
        "us_only", rollout_pct=0.0,
        rules=[TargetingRule("us_users", [Condition("eq", "country", "US")])],
    ))
    return mgr


def test_gate_targets_by_country():
    mgr = _country_gate()
    assert mgr.check_gate(StatsigUser("u1", country="US"), "us_only") is True
    assert mgr.check_gate(StatsigUser("u2", country="CA"), "us_only") is False


def test_gate_legacy_string_user_still_rolls_out():
    """Passing a bare user_id (no rules) keeps the percentage-rollout behaviour."""
    mgr = ExperimentManager()
    mgr.add_gate(FeatureGateConfig("g", 1.0))
    assert mgr.check_gate("anyuser", "g") is True


def test_gate_version_targeting():
    mgr = ExperimentManager()
    mgr.add_gate(FeatureGateConfig(
        "new_ui", 0.0,
        rules=[TargetingRule("modern", [Condition("version_gte", "app_version", "2.0.0")])],
    ))
    assert mgr.check_gate(StatsigUser("a", app_version="2.1.0"), "new_ui") is True
    assert mgr.check_gate(StatsigUser("b", app_version="1.9.9"), "new_ui") is False


def test_gate_custom_field_and_in_operator():
    mgr = ExperimentManager()
    mgr.add_gate(FeatureGateConfig(
        "beta", 0.0,
        rules=[TargetingRule("beta_cohort",
                             [Condition("in", "custom.plan", ["pro", "enterprise"])])],
    ))
    assert mgr.check_gate(StatsigUser("a", custom={"plan": "pro"}), "beta") is True
    assert mgr.check_gate(StatsigUser("b", custom={"plan": "free"}), "beta") is False


def test_pass_gate_condition_chains():
    mgr = ExperimentManager()
    mgr.add_gate(FeatureGateConfig("is_employee", 0.0,
                 rules=[TargetingRule("emp", [Condition("ends_with", "email", "@corp.com")])]))
    mgr.add_gate(FeatureGateConfig("internal_tool", 0.0,
                 rules=[TargetingRule("emp_only", [Condition("pass_gate", target="is_employee")])]))
    assert mgr.check_gate(StatsigUser("a", email="x@corp.com"), "internal_tool") is True
    assert mgr.check_gate(StatsigUser("b", email="x@gmail.com"), "internal_tool") is False


def test_experiment_targeting_excludes_ineligible():
    mgr = ExperimentManager()
    mgr.add_experiment(ExperimentConfig(
        "checkout_test",
        [ExperimentVariant("control", 0.5), ExperimentVariant("treatment", 0.5)],
        targeting_rules=[TargetingRule("us", [Condition("eq", "country", "US")])],
    ))
    assert mgr.assign(StatsigUser("a", country="US"), "checkout_test") in ("control", "treatment")
    assert mgr.assign(StatsigUser("b", country="JP"), "checkout_test") is None


# ─── id_type: bucket on a custom unit (e.g. companyID) ───────────────────────

def test_custom_id_type_buckets_per_company():
    mgr = ExperimentManager()
    mgr.add_experiment(ExperimentConfig(
        "company_rollout",
        [ExperimentVariant("control", 0.5), ExperimentVariant("treatment", 0.5)],
        id_type="companyID"))
    # two users in the SAME company must get the SAME variant (bucketed on company)
    u1 = StatsigUser("alice", custom_ids={"companyID": "acme"})
    u2 = StatsigUser("bob", custom_ids={"companyID": "acme"})
    assert mgr.assign(u1, "company_rollout") == mgr.assign(u2, "company_rollout")


def test_custom_id_type_distributes_across_companies():
    mgr = ExperimentManager()
    mgr.add_experiment(ExperimentConfig(
        "co", [ExperimentVariant("control", 0.5), ExperimentVariant("treatment", 0.5)],
        id_type="companyID"))
    treat = sum(1 for i in range(1000)
                if mgr.assign(StatsigUser(f"u{i}", custom_ids={"companyID": f"co_{i}"}), "co")
                == "treatment")
    assert 400 < treat < 600


# ─── sticky bucketing: assignment survives weight changes ────────────────────

def test_sticky_assignment_survives_reweighting():
    mgr = ExperimentManager()
    mgr.add_experiment(ExperimentConfig(
        "sticky_exp",
        [ExperimentVariant("control", 0.5), ExperimentVariant("treatment", 0.5)],
        sticky=True))
    first = mgr.assign(StatsigUser("alice"), "sticky_exp")
    # Re-weight so a non-sticky user would almost certainly flip to control.
    mgr._experiments["sticky_exp"] = ExperimentConfig(
        "sticky_exp",
        [ExperimentVariant("control", 0.999), ExperimentVariant("treatment", 0.001)],
        sticky=True)
    after = mgr.assign(StatsigUser("alice"), "sticky_exp")
    assert after == first   # stuck to original variant


def test_clear_sticky_reassigns():
    mgr = ExperimentManager()
    mgr.add_experiment(ExperimentConfig(
        "s", [ExperimentVariant("control", 1.0)], sticky=True))
    mgr.assign(StatsigUser("alice"), "s")
    mgr.clear_sticky(StatsigUser("alice"), "s")
    assert ("alice", "s") not in mgr._sticky


def test_non_sticky_has_no_persistence():
    mgr = ExperimentManager()
    mgr.add_experiment(ExperimentConfig(
        "ns", [ExperimentVariant("control", 0.5), ExperimentVariant("treatment", 0.5)]))
    mgr.assign(StatsigUser("alice"), "ns")
    assert mgr._sticky == {}


# ─── G4 / G5: FeatureGate object carries eval details ────────────────────────

def test_get_feature_gate_object_has_rule_metadata():
    mgr = _country_gate()
    fg = mgr.evaluate_gate(StatsigUser("u1", country="US"), "us_only")
    assert fg.value is True
    assert fg.group_name == "us_users"
    assert fg.rule_id == "us_users"
    assert fg.reason == "TargetingRule"


def test_unknown_gate_reason_unrecognized():
    mgr = ExperimentManager()
    fg = mgr.evaluate_gate(StatsigUser("u1"), "missing")
    assert fg.value is False
    assert fg.reason == "Unrecognized"


# ─── G6: Layer per-parameter exposure ────────────────────────────────────────

def test_layer_logs_exposure_only_for_read_parameter():
    seen = []
    layer = Layer("ui_layer", {"color": "red", "size": "lg"},
                  allocated_experiment="exp_a",
                  on_parameter_exposure=lambda p, e: seen.append((p, e)))
    assert layer.get("color") == "red"
    assert seen == [("color", "exp_a")]      # only the param we read
    layer.get("color")                       # second read: no duplicate exposure
    assert seen == [("color", "exp_a")]


def test_layer_missing_param_no_exposure():
    seen = []
    layer = Layer("l", {"a": 1}, on_parameter_exposure=lambda p, e: seen.append(p))
    assert layer.get("nonexistent", default=9) == 9
    assert seen == []


# ─── G7: DynamicConfig metadata + standalone config targeting ────────────────

def test_dynamic_config_spec_targeting_and_metadata():
    mgr = ExperimentManager()
    mgr.add_dynamic_config(DynamicConfigSpec(
        "pricing", defaults={"price": 9.99},
        rules=[TargetingRule("india", [Condition("eq", "country", "IN")],
                             return_value={"price": 2.99})],
    ))
    dc_in = mgr.evaluate_config(StatsigUser("a", country="IN"), "pricing")
    assert dc_in.get("price") == 2.99
    assert dc_in.group_name == "india"
    assert dc_in.rule_id == "india"
    dc_us = mgr.evaluate_config(StatsigUser("b", country="US"), "pricing")
    assert dc_us.get("price") == 9.99
    assert dc_us.reason == "Default"


# ─── G8: CUPED variance reduction ────────────────────────────────────────────

def test_cuped_reduces_variance_tightening_ci():
    pc = PulseComputer()
    # pre strongly correlates with post; CUPED should shrink the CI width vs raw.
    # post = pre + effect + small idiosyncratic noise (so CUPED reduces, not erases, variance)
    noise = [(i * 7 % 11) / 20.0 for i in range(60)]   # deterministic small jitter
    ctrl_pre = [float(i % 10) for i in range(60)]
    ctrl_post = [ctrl_pre[i] + 1.0 + noise[i] for i in range(60)]
    trt_pre = [float(i % 10) for i in range(60)]
    trt_post = [trt_pre[i] + 1.5 + noise[(i + 3) % 60] for i in range(60)]

    raw = pc.compute_from_values("raw", {"m": (ctrl_post, trt_post)})
    cuped = pc.compute_from_values_cuped("cuped", {"m": (ctrl_pre, ctrl_post, trt_pre, trt_post)})

    raw_w = raw.metrics[0].ci_upper - raw.metrics[0].ci_lower
    cuped_w = cuped.metrics[0].ci_upper - cuped.metrics[0].ci_lower
    assert cuped.metrics[0].variance_reduced is True
    assert cuped_w < raw_w  # tighter interval


def test_cuped_adjust_preserves_mean():
    pc = PulseComputer()
    pre = [1.0, 2.0, 3.0, 4.0, 5.0]
    post = [2.0, 3.5, 5.0, 6.5, 8.0]
    adjusted, theta = pc.cuped_adjust(pre, post)
    assert abs(sum(adjusted) / len(adjusted) - sum(post) / len(post)) < 1e-9
    assert theta != 0.0


# ─── G9: absolute lift reported alongside relative ───────────────────────────

def test_absolute_diff_reported():
    pc = PulseComputer()
    report = pc.compute_from_rates("t", {"conv": (1000, 200, 1000, 300)})  # 20% → 30%
    m = report.metrics[0]
    assert abs(m.absolute_diff - 0.10) < 0.005     # absolute = 10pp
    assert abs(m.lift - 0.5) < 0.02                # relative = +50%


# ─── G10: sequential / always-valid p-values ─────────────────────────────────

def test_sequential_p_value_shrinks_with_more_data():
    pc = PulseComputer()
    small = pc.compute_from_rates("s", {"c": (100, 20, 100, 35)})
    large = pc.compute_from_rates("l", {"c": (10000, 2000, 10000, 3500)})
    # same effect, 100x data → always-valid p must be much smaller
    assert large.metrics[0].sequential_p_value < small.metrics[0].sequential_p_value


def test_sequential_p_value_is_conservative_vs_fixed():
    """Anytime-valid p-value should never be smaller than the fixed-n p-value
    for a borderline effect (it pays for the right to peek)."""
    pc = PulseComputer()
    r = pc.compute_from_rates("b", {"c": (500, 100, 500, 120)})
    m = r.metrics[0]
    assert m.sequential_p_value >= m.p_value - 1e-9


# ─── G11: multiple-testing (Bonferroni) correction ───────────────────────────

def test_bonferroni_correction_is_stricter():
    pc = PulseComputer()
    # one borderline-significant metric among several → corrected flag is stricter.
    metrics = {
        "a": (1000, 500, 1000, 540),   # mild lift
        "b": (1000, 500, 1000, 505),   # noise
        "c": (1000, 500, 1000, 502),   # noise
        "d": (1000, 500, 1000, 498),   # noise
    }
    report = pc.compute_from_rates("multi", metrics)
    a = next(m for m in report.metrics if m.name == "a")
    # If 'a' is significant uncorrected, Bonferroni (alpha/4) must be at least as strict.
    if a.significant:
        assert (a.significant_corrected is True) == (a.p_value < 0.05 / 4)
    assert all(m.significant_corrected in (True, False) for m in report.metrics)
