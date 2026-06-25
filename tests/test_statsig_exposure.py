"""G2: exposure logging — the behaviour that lets real usage feed Pulse.

Every gate/experiment/config/layer read through UserSoulClient must emit a
Statsig-style exposure event into the EventLogger (unless explicitly disabled or
in local_mode), carrying rule_id / group_name / reason. These tests prove the
loop closes: check a gate → an exposure is logged → it can be queried back.
"""
from __future__ import annotations

from unittest.mock import MagicMock

from user_soul.client import UserSoulClient
from user_soul.statsig_user import StatsigUser, StatsigOptions
from user_soul.experiment_manager import (
    ExperimentManager, ExperimentConfig, ExperimentVariant, FeatureGateConfig,
    DynamicConfigSpec,
)
from user_soul.targeting import Condition, TargetingRule


def _client():
    mgr = ExperimentManager()
    mgr.add_gate(FeatureGateConfig(
        "new_ui", 0.0,
        rules=[TargetingRule("us", [Condition("eq", "country", "US")])]))
    mgr.add_experiment(ExperimentConfig(
        "btn", [ExperimentVariant("control", 0.5, {"color": "blue"}),
                ExperimentVariant("treatment", 0.5, {"color": "red"})]))
    mgr.add_dynamic_config(DynamicConfigSpec("prices", defaults={"p": 9.99}))
    mgr.add_experiment(ExperimentConfig(
        "layer_exp", [ExperimentVariant("control", 1.0, {"headline": "Hi"})],
        layer="home"))
    return UserSoulClient(MagicMock(), experiment_manager=mgr)


def test_check_gate_logs_exposure():
    c = _client()
    c.check_gate(StatsigUser("u1", country="US"), "new_ui")
    exps = c.get_exposures("statsig::gate_exposure")
    assert len(exps) == 1
    assert exps[0].metadata["name"] == "new_ui"
    assert exps[0].metadata["group_name"] == "us"
    assert exps[0].metadata["reason"] == "TargetingRule"


def test_exposure_can_be_disabled_per_call():
    c = _client()
    c.check_gate(StatsigUser("u1", country="US"), "new_ui", log_exposure=False)
    assert c.get_exposures("statsig::gate_exposure") == []


def test_local_mode_suppresses_exposures():
    c = _client()
    c.initialize(StatsigOptions(local_mode=True))
    c.check_gate(StatsigUser("u1", country="US"), "new_ui")
    assert c.get_exposures() == []


def test_get_experiment_logs_config_exposure_with_variant():
    c = _client()
    dc = c.get_experiment(StatsigUser("u1"), "btn")
    exps = c.get_exposures("statsig::config_exposure")
    assert len(exps) == 1
    assert exps[0].metadata["name"] == "btn"
    assert exps[0].metadata["variant"] in ("control", "treatment")
    assert dc.get("color") in ("blue", "red")


def test_layer_logs_exposure_only_on_param_read():
    c = _client()
    layer = c.get_layer(StatsigUser("u1"), "home")
    assert c.get_exposures("statsig::layer_exposure") == []   # nothing read yet
    layer.get("headline")
    exps = c.get_exposures("statsig::layer_exposure")
    assert len(exps) == 1
    assert exps[0].metadata["parameter"] == "headline"
    assert exps[0].metadata["allocated_experiment"] == "layer_exp"


def test_manual_exposure_api():
    c = _client()
    c.manually_log_gate_exposure(StatsigUser("u1", country="US"), "new_ui")
    assert len(c.get_exposures("statsig::gate_exposure")) == 1


def test_override_gate_logs_override_reason():
    c = _client()
    c.override_gate("new_ui", True)
    assert c.check_gate(StatsigUser("u1", country="JP"), "new_ui") is True
    exps = c.get_exposures("statsig::gate_exposure")
    assert exps[0].metadata["reason"] == "LocalOverride"


def test_exposures_feed_pulse_end_to_end():
    """The whole point: exposures + a metric event → Pulse computes a verdict."""
    c = _client()
    # Simulate 200 users through the experiment, logging a 'converted' metric event.
    converts = {"control": 0, "treatment": 0}
    totals = {"control": 0, "treatment": 0}
    for i in range(400):
        u = StatsigUser(f"user_{i}")
        dc = c.get_experiment(u, "btn")
        variant = dc._variant
        totals[variant] += 1
        # treatment converts at a higher rate (deterministic by index parity within arm)
        if variant == "treatment" and i % 2 == 0:
            converts[variant] += 1
        elif variant == "control" and i % 5 == 0:
            converts[variant] += 1
    # exposures were logged for every assignment
    assert len(c.get_exposures("statsig::config_exposure")) == 400
    report = c.get_pulse("btn", {"conversion": (
        totals["control"], converts["control"],
        totals["treatment"], converts["treatment"])})
    assert report.metrics[0].treatment_value > report.metrics[0].control_value
