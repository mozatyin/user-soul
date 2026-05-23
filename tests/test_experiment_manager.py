"""Tests for ExperimentManager."""
from __future__ import annotations

import pytest

from user_soul.experiment_manager import (
    ExperimentConfig,
    ExperimentManager,
    ExperimentVariant,
    FeatureGateConfig,
    HoldoutConfig,
    _bucket,
)

USERS = [f"user_{i}" for i in range(1000)]


# ---------------------------------------------------------------------------
# _bucket
# ---------------------------------------------------------------------------

def test_bucket_deterministic():
    for uid in ["alice", "bob", "charlie"]:
        assert _bucket(uid, "salt") == _bucket(uid, "salt")
        assert _bucket(uid, "a") != _bucket(uid, "b")  # different salts differ


def test_bucket_uniform():
    values = [_bucket(f"user_{i}", "test") for i in range(1000)]
    below_half = sum(1 for v in values if v < 0.5)
    assert 430 < below_half < 570, f"Expected ~500, got {below_half}"
    assert all(0.0 <= v < 1.0 for v in values)


# ---------------------------------------------------------------------------
# Feature Gates
# ---------------------------------------------------------------------------

def _gate_mgr(pct: float) -> ExperimentManager:
    mgr = ExperimentManager()
    mgr.add_gate(FeatureGateConfig("my_gate", pct))
    return mgr


def test_feature_gate_rollout_50pct():
    mgr = _gate_mgr(0.5)
    enabled = sum(1 for u in USERS if mgr.check_gate(u, "my_gate"))
    assert 430 < enabled < 570, f"Expected ~500, got {enabled}"


def test_feature_gate_rollout_100pct():
    mgr = _gate_mgr(1.0)
    assert all(mgr.check_gate(u, "my_gate") for u in USERS)


def test_feature_gate_rollout_0pct():
    mgr = _gate_mgr(0.0)
    assert not any(mgr.check_gate(u, "my_gate") for u in USERS)


def test_unknown_gate_returns_false():
    mgr = ExperimentManager()
    assert mgr.check_gate("alice", "nonexistent_gate") is False


# ---------------------------------------------------------------------------
# Experiments — basic assignment
# ---------------------------------------------------------------------------

def _exp_mgr(variants, holdout_pct=0.0, layer=None) -> ExperimentManager:
    mgr = ExperimentManager()
    mgr.add_experiment(
        ExperimentConfig(
            "exp1",
            [ExperimentVariant(n, w) for n, w in variants],
            holdout_pct=holdout_pct,
            layer=layer,
        )
    )
    return mgr


def test_experiment_assign_two_variants():
    mgr = _exp_mgr([("control", 0.5), ("treatment", 0.5)])
    assignments = [mgr.assign(u, "exp1") for u in USERS]
    controls = assignments.count("control")
    assert 430 < controls < 570, f"Expected ~500, got {controls}"


def test_experiment_assign_weighted():
    mgr = _exp_mgr([("control", 0.8), ("treatment", 0.2)])
    assignments = [mgr.assign(u, "exp1") for u in USERS]
    controls = assignments.count("control")
    assert 730 < controls < 870, f"Expected ~800, got {controls}"


def test_experiment_assign_deterministic():
    mgr = _exp_mgr([("control", 0.5), ("treatment", 0.5)])
    for u in ["alice", "bob", "carol"]:
        assert mgr.assign(u, "exp1") == mgr.assign(u, "exp1")


def test_experiment_holdout_excludes_users():
    mgr = _exp_mgr([("control", 0.5), ("treatment", 0.5)], holdout_pct=0.1)
    nones = sum(1 for u in USERS if mgr.assign(u, "exp1") is None)
    assert 60 < nones < 140, f"Expected ~100, got {nones}"


def test_experiment_no_holdout_never_none():
    mgr = _exp_mgr([("control", 0.5), ("treatment", 0.5)], holdout_pct=0.0)
    assert all(mgr.assign(u, "exp1") is not None for u in USERS)


def test_unknown_experiment_returns_none():
    mgr = ExperimentManager()
    assert mgr.assign("alice", "nonexistent_exp") is None


# ---------------------------------------------------------------------------
# Weight normalization
# ---------------------------------------------------------------------------

def test_weight_normalization():
    mgr = ExperimentManager()
    mgr.add_experiment(
        ExperimentConfig(
            "norm_exp",
            [
                ExperimentVariant("a", 2.0),
                ExperimentVariant("b", 2.0),
                ExperimentVariant("c", 1.0),
            ],
        )
    )
    exp = mgr._experiments["norm_exp"]
    weights = {v.name: v.weight for v in exp.variants}
    assert abs(weights["a"] - 0.4) < 1e-9
    assert abs(weights["b"] - 0.4) < 1e-9
    assert abs(weights["c"] - 0.2) < 1e-9


# ---------------------------------------------------------------------------
# Layer mutual exclusivity
# ---------------------------------------------------------------------------

def _layer_mgr() -> ExperimentManager:
    mgr = ExperimentManager()
    mgr.add_experiment(
        ExperimentConfig(
            "exp_a",
            [ExperimentVariant("control", 0.5), ExperimentVariant("treatment", 0.5)],
            layer="layer1",
        )
    )
    mgr.add_experiment(
        ExperimentConfig(
            "exp_b",
            [ExperimentVariant("control", 0.5), ExperimentVariant("treatment", 0.5)],
            layer="layer1",
        )
    )
    return mgr


def test_layer_mutual_exclusivity():
    """Each user should be in the active experiment, and return 'control' in the other."""
    mgr = _layer_mgr()
    for u in USERS[:200]:
        a = mgr.assign(u, "exp_a")
        b = mgr.assign(u, "exp_b")
        # One of the two is the "active" variant (possibly treatment), the other is control.
        # They cannot both be "treatment" because they're in the same layer and only one
        # experiment is active per user.
        active_non_control = sum(1 for v in (a, b) if v == "treatment")
        assert active_non_control <= 1, (
            f"User {u} got treatment in both experiments: a={a}, b={b}"
        )


def test_layer_full_coverage():
    """1000 users should be distributed across both layered experiments."""
    mgr = _layer_mgr()
    in_a_treatment = sum(1 for u in USERS if mgr.assign(u, "exp_a") == "treatment")
    in_b_treatment = sum(1 for u in USERS if mgr.assign(u, "exp_b") == "treatment")
    # ~50% in exp_a layer slot, ~50% in exp_b layer slot; within each ~50% get treatment
    # so ~250 treatment in each; both should be non-trivial
    assert in_a_treatment > 50, f"exp_a treatment too low: {in_a_treatment}"
    assert in_b_treatment > 50, f"exp_b treatment too low: {in_b_treatment}"


def test_add_to_layer_registers():
    mgr = ExperimentManager()
    mgr.add_experiment(
        ExperimentConfig("e1", [ExperimentVariant("control", 1.0)])
    )
    mgr.add_to_layer("my_layer", "e1")
    assert "e1" in mgr._layers["my_layer"]


# ---------------------------------------------------------------------------
# Global holdout
# ---------------------------------------------------------------------------

def test_global_holdout():
    mgr = ExperimentManager()
    mgr.add_holdout(HoldoutConfig("global", holdout_pct=0.2))
    in_holdout = sum(1 for u in USERS if mgr.is_in_holdout(u, "global"))
    assert 140 < in_holdout < 260, f"Expected ~200, got {in_holdout}"


# ---------------------------------------------------------------------------
# get_variant_config
# ---------------------------------------------------------------------------

def test_variant_config_lookup():
    mgr = ExperimentManager()
    mgr.add_experiment(
        ExperimentConfig(
            "cfg_exp",
            [
                ExperimentVariant("control", 0.5, {"color": "blue"}),
                ExperimentVariant("treatment", 0.5, {"color": "red"}),
            ],
        )
    )
    for u in USERS[:100]:
        variant = mgr.assign(u, "cfg_exp")
        color = mgr.get_variant_config(u, "cfg_exp", "color")
        expected = "blue" if variant == "control" else "red"
        assert color == expected, f"user={u} variant={variant} color={color}"


def test_variant_config_default_on_holdout():
    mgr = ExperimentManager()
    mgr.add_experiment(
        ExperimentConfig(
            "holdout_cfg",
            [
                ExperimentVariant("control", 0.5, {"x": 1}),
                ExperimentVariant("treatment", 0.5, {"x": 2}),
            ],
            holdout_pct=1.0,
        )
    )
    assert mgr.get_variant_config("alice", "holdout_cfg", "x", default=99) == 99


# ---------------------------------------------------------------------------
# get_all_assignments
# ---------------------------------------------------------------------------

def test_get_all_assignments():
    mgr = ExperimentManager()
    mgr.add_experiment(
        ExperimentConfig("e1", [ExperimentVariant("control", 0.5), ExperimentVariant("t", 0.5)])
    )
    mgr.add_experiment(
        ExperimentConfig("e2", [ExperimentVariant("control", 1.0)])
    )
    result = mgr.get_all_assignments("alice")
    assert set(result.keys()) == {"e1", "e2"}
    assert result["e1"] in ("control", "t")
    assert result["e2"] == "control"


# ---------------------------------------------------------------------------
# export / reload
# ---------------------------------------------------------------------------

def test_export_and_reload():
    mgr = ExperimentManager()
    mgr.add_experiment(
        ExperimentConfig(
            "reload_exp",
            [ExperimentVariant("control", 0.6, {"k": "v"}), ExperimentVariant("t", 0.4)],
            holdout_pct=0.05,
            description="test exp",
        )
    )
    mgr.add_gate(FeatureGateConfig("fg", 0.3, "a gate"))
    mgr.add_holdout(HoldoutConfig("ho", 0.1, "a holdout"))

    cfg = mgr.export_config()
    mgr2 = ExperimentManager.from_config(cfg)

    for u in USERS[:100]:
        assert mgr.assign(u, "reload_exp") == mgr2.assign(u, "reload_exp")
        assert mgr.check_gate(u, "fg") == mgr2.check_gate(u, "fg")
        assert mgr.is_in_holdout(u, "ho") == mgr2.is_in_holdout(u, "ho")


# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------

def test_control_variant_exists():
    """Experiment without 'control' still assigns to one of its variants."""
    mgr = ExperimentManager()
    mgr.add_experiment(
        ExperimentConfig(
            "no_ctrl",
            [ExperimentVariant("variant_a", 0.5), ExperimentVariant("variant_b", 0.5)],
        )
    )
    for u in USERS[:50]:
        result = mgr.assign(u, "no_ctrl")
        assert result in ("variant_a", "variant_b")
