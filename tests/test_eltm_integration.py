"""Tests for user_soul.eltm_integration — the ELTM Phase 0.7 / Phase 8.8 hooks.

This module is the documented insertion point between ELTM build_core() and
User-Soul (pre_build_filter / post_build_validate). It had ZERO test coverage and
was never wired into ELTM, which is exactly how an interface-contract break
(post_build_validate calling ABValidator.validate with the wrong signature) stayed
latent. These tests exercise the full hook end-to-end with mocked LLM boundaries.
"""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import types

from user_soul.eltm_integration import (
    pre_build_filter,
    post_build_validate,
    check_expectations_met,
    FilterResult,
    ValidationResult,
)
from user_soul.eltm_adapter import (
    extract_features,
    build_product_description,
    extract_benchmark_name,
)
from user_soul.ab_validator import ABValidationReport
from user_soul.models import CompareReport


# ---------------------------------------------------------------------------
# Realistic build_core() output fixture
# ---------------------------------------------------------------------------

BUILD_OUTPUT = {
    "game_name": "ChessMaster",
    "benchmark_reference": {"name": "Chess.com"},
    "moon_features": [
        {"id": "feat_puzzles", "name": "Daily Puzzles",
         "description": "A fresh puzzle each day", "category": "core"},
        {"name": "Adaptive AI", "description": "AI that matches your level",
         "category": "core"},
    ],
    "formal_rules": {
        "onboarding": {
            "rules_summary": "Learn chess in 3 guided steps",
            "difficulty_levels": [{"label": "Beginner"}, {"label": "Pro"}],
        },
        "game_modes": [
            {"id": "blitz", "label": "Blitz", "description": "5-minute games", "free": True},
            {"label": "Ranked", "description": "Compete for rating", "free": False},
        ],
        "gameplay_controls": {
            "during_game": [{"label": "Undo", "behavior": "Take back last move"}],
            "time_controls": [{"label": "Blitz"}, {"label": "Rapid"}],
        },
        "engagement": {
            "streak": True,
            "daily_puzzle": True,
            "progression": "ELO rating climbs as you win",
            "social": ["friend challenges"],
        },
        "benchmark_gaps": ["No good offline mode in competitors"],
    },
}

RESEARCH_OUTPUT = {
    "target_segment": "18-25 beginner chess players",
    "benchmark_reference": {"name": "Chess.com"},
}


# ---------------------------------------------------------------------------
# 1. eltm_adapter — pure data transforms (no LLM)
# ---------------------------------------------------------------------------

def test_extract_features_flattens_all_sources():
    feats = extract_features(BUILD_OUTPUT)
    ids = {f["id"] for f in feats}
    # one from each source category should be present
    assert any(f["source"] == "moon_features" for f in feats)
    assert any(f["source"] == "game_modes" for f in feats)
    assert any(f["source"] == "engagement" for f in feats)
    assert "mode_blitz" in ids
    # premium modes get a "(premium)" suffix
    ranked = next(f for f in feats if f["name"] == "Ranked")
    assert "(premium)" in ranked["description"]
    # every feature has the contract keys
    for f in feats:
        assert set(f) >= {"id", "name", "description", "category", "source"}


def test_extract_benchmark_name_and_description():
    assert extract_benchmark_name(BUILD_OUTPUT) == "Chess.com"
    desc = build_product_description(BUILD_OUTPUT)
    assert "ChessMaster" in desc
    assert "Chess.com" in desc


def test_extract_features_handles_garbage_input():
    assert extract_features(None) == []
    assert extract_features({}) == []
    assert build_product_description(None) == ""


# ---------------------------------------------------------------------------
# 2. pre_build_filter — Phase 0.7, end-to-end with mocked backend
# ---------------------------------------------------------------------------

def _archetype_json() -> str:
    return json.dumps([
        {"name": "Competitive", "description": "wins matter", "background_story": "ranked"},
        {"name": "Social", "description": "plays for fun", "background_story": "casual"},
    ])


def _aarrr_json(feature_ids, score=0.7) -> str:
    items = []
    for fid in feature_ids:
        block = {k: score for k in
                 ("acquisition", "activation", "retention", "revenue", "referral")}
        items.append({"feature_id": fid, "archetype_votes": {"Competitive": block},
                      "mean": block})
    return json.dumps(items)


def test_pre_build_filter_end_to_end():
    feats = extract_features(BUILD_OUTPUT)
    ids = [f["id"] for f in feats]

    backend = MagicMock()
    # call 1 = auto archetypes, call 2 = batch AARRR scoring
    backend.text.side_effect = [_archetype_json(), _aarrr_json(ids, 0.7)]

    with patch("user_soul.backends.anthropic.AnthropicBackend", return_value=backend):
        result = pre_build_filter(RESEARCH_OUTPUT | BUILD_OUTPUT, api_key="sk-test")

    assert isinstance(result, FilterResult)
    assert result.report.total_input == len(feats)
    reqs = result.to_requirements()
    assert isinstance(reqs, list)
    # high-scoring features become P0 requirements
    assert any(r.startswith("[User Soul P0]") for r in reqs)
    assert "Phase 0.7" in result.summary()


# ---------------------------------------------------------------------------
# 3. post_build_validate — Phase 8.8, end-to-end (the regression that was broken)
# ---------------------------------------------------------------------------

def _fake_compare(regressions, deltas) -> CompareReport:
    # ABValidator.validate() only reads improvements/regressions/deltas after the
    # compare() call — the variant reports themselves are never inspected.
    return CompareReport(
        n_runs_per_variant=10,
        variant_a_label="ChessMaster",
        variant_b_label="Chess.com",
        variant_a=None,
        variant_b=None,
        deltas=deltas,
        improvements=[],
        regressions=regressions,
        key_diff="",
    )


def test_post_build_validate_signature_is_correct():
    """Regression guard: post_build_validate must call ABValidator.validate with
    the real signature (our_product/reference_product/user_type/goal). Previously
    it passed our_product_description=... and omitted required args → TypeError."""
    backend = MagicMock()
    backend.text.side_effect = [
        '{"recommendation": "Add an offline mode"}',
        '{"summary": "Competitive but one regression remains."}',
    ]

    compare = _fake_compare(
        regressions=["day1_return_rate"],
        deltas={"day1_return_rate": 0.12},  # reference ahead → our_minus_ref -0.12 → P0
    )

    with patch("user_soul.backends.anthropic.AnthropicBackend", return_value=backend), \
         patch("user_soul.ab_validator.build_domain_config", return_value=MagicMock()), \
         patch("user_soul.ab_validator.UserSimulator") as MockSim:
        MockSim.return_value.compare.return_value = compare
        result = post_build_validate(BUILD_OUTPUT, RESEARCH_OUTPUT, api_key="sk-test")

    assert isinstance(result, ValidationResult)
    assert isinstance(result.report, ABValidationReport)
    assert result.report.reference_label == "Chess.com"
    assert result.report.our_label == "ChessMaster"
    reqs = result.to_requirements()
    assert any(r.startswith("[User Soul P0]") for r in reqs)
    assert "Phase 8.8" in result.summary()


# ---------------------------------------------------------------------------
# 4. Persona Memory Chain (Decision #52) — check_expectations_met
# ---------------------------------------------------------------------------

def _must_have(*names):
    return [types.SimpleNamespace(name=n) for n in names]


def test_expectations_met_when_feature_in_product():
    # "Daily Puzzles" is a moon_feature of BUILD_OUTPUT → should read as met.
    exp = check_expectations_met(_must_have("Daily Puzzles"), BUILD_OUTPUT)
    assert "Daily Puzzles" in exp["met"]
    assert exp["unmet"] == []
    assert exp["met_rate"] == 1.0
    assert exp["requirements"] == []


def test_expectations_unmet_becomes_pdca_requirement():
    # A feature the personas wanted but the product never surfaces → unmet → P0.
    exp = check_expectations_met(_must_have("Online Multiplayer Tournaments"), BUILD_OUTPUT)
    assert "Online Multiplayer Tournaments" in exp["unmet"]
    assert exp["met_rate"] == 0.0
    assert any(r.startswith("[User Soul P0] Persona expectation unmet") for r in exp["requirements"])


def test_expectations_empty_is_fully_met():
    exp = check_expectations_met([], BUILD_OUTPUT)
    assert exp["met_rate"] == 1.0
    assert exp["requirements"] == []
