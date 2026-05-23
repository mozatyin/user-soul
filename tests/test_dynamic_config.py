"""Tests for DynamicConfig."""
from __future__ import annotations

import pytest
from user_soul.dynamic_config import DynamicConfig


def test_get_default():
    cfg = DynamicConfig(defaults={"color": "blue"})
    assert cfg.get("color") == "blue"


def test_set_and_get():
    cfg = DynamicConfig()
    cfg.set("timeout", 30)
    assert cfg.get("timeout") == 30


def test_user_override_priority():
    cfg = DynamicConfig(defaults={"feature_flag": False})
    cfg.set_for_user("alice", "feature_flag", True)
    assert cfg.get("feature_flag", user_id="alice") is True
    assert cfg.get("feature_flag", user_id="bob") is False


def test_variant_override_priority():
    cfg = DynamicConfig(defaults={"cta_text": "Sign Up"})
    cfg.set_for_variant("onboarding", "v2", "cta_text", "Get Started")
    cfg.set_for_user("alice", "cta_text", "Join Now")

    # user override beats variant override
    assert cfg.get("cta_text", user_id="alice", experiment="onboarding", variant="v2") == "Join Now"
    # variant override beats global default
    assert cfg.get("cta_text", experiment="onboarding", variant="v2") == "Get Started"
    # no override → global default
    assert cfg.get("cta_text") == "Sign Up"


def test_get_missing_returns_default():
    cfg = DynamicConfig()
    assert cfg.get("nonexistent", default="fallback") == "fallback"
    assert cfg.get("nonexistent") is None


def test_get_all_merges_overrides():
    cfg = DynamicConfig(defaults={"a": 1, "b": 2})
    cfg.set_for_user("alice", "a", 99)
    result = cfg.get_all(user_id="alice")
    assert result["a"] == 99
    assert result["b"] == 2


def test_json_roundtrip():
    cfg = DynamicConfig(defaults={"x": 10, "y": "hello"})
    cfg.set_for_user("bob", "x", 42)
    cfg.set_for_variant("exp1", "ctrl", "y", "world")

    json_str = cfg.as_json()
    cfg2 = DynamicConfig.from_json(json_str)

    assert cfg2.get("x") == 10
    assert cfg2.get("x", user_id="bob") == 42
    assert cfg2.get("y", experiment="exp1", variant="ctrl") == "world"
    assert cfg2.get("y") == "hello"


def test_no_cross_user_leakage():
    cfg = DynamicConfig(defaults={"flag": False})
    cfg.set_for_user("alice", "flag", True)
    assert cfg.get("flag", user_id="bob") is False
    assert cfg.get("flag", user_id="alice") is True


def test_variant_requires_experiment():
    cfg = DynamicConfig(defaults={"price": 9.99})
    cfg.set_for_variant("pricing_test", "v1", "price", 4.99)
    # same variant name but different experiment → should not match
    assert cfg.get("price", experiment="other_test", variant="v1") == 9.99
    # correct experiment + variant → override
    assert cfg.get("price", experiment="pricing_test", variant="v1") == 4.99


def test_empty_defaults_constructor():
    cfg = DynamicConfig()
    assert cfg.get("anything") is None
    assert cfg.get_all() == {}


def test_set_overwrites_default():
    cfg = DynamicConfig(defaults={"key": "old"})
    cfg.set("key", "new")
    assert cfg.get("key") == "new"
