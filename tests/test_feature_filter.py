"""Tests for user_soul.feature_filter — Phase 0.7 AARRR feature prioritisation."""
from __future__ import annotations

import json
from unittest.mock import MagicMock, call

import pytest

from user_soul.feature_filter import (
    FeatureFilter,
    FeatureFilterReport,
    ScoredFeature,
    _auto_archetypes,
    _normalize_features,
    _slug,
)
from user_soul.models import Archetype, FeatureAAR


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_backend(aarrr_response: str = "[]", text_response: str = "[]") -> MagicMock:
    """Return a MagicMock LLMBackend whose .text() returns preset strings."""
    backend = MagicMock()
    backend.text.return_value = text_response
    return backend


def _make_aarrr_response(feature_ids: list[str], score: float = 0.7) -> str:
    """Build a valid AARRR JSON array for the given feature ids."""
    items = []
    for fid in feature_ids:
        items.append({
            "feature_id": fid,
            "archetype_votes": {
                "Engaged User": {
                    "acquisition": score,
                    "activation": score,
                    "retention": score,
                    "revenue": score,
                    "referral": score,
                }
            },
            "mean": {
                "acquisition": score,
                "activation": score,
                "retention": score,
                "revenue": score,
                "referral": score,
            },
        })
    return json.dumps(items)


def _make_archetype(name: str = "Engaged User") -> Archetype:
    return Archetype(name=name, frequency=1.0, description="test archetype",
                     trait_constraints={}, background_story="")


# ---------------------------------------------------------------------------
# 1. _normalize_features
# ---------------------------------------------------------------------------

def test_normalize_features_fills_missing_fields():
    raw = [{"name": "Matchmaking"}]
    result = _normalize_features(raw)
    assert len(result) == 1
    feat = result[0]
    assert feat["name"] == "Matchmaking"
    assert feat["id"] == "000_matchmaking"
    assert feat["description"] == ""
    assert feat["category"] == "general"
    assert feat["source"] == "unknown"


def test_normalize_features_preserves_existing_fields():
    raw = [{"name": "Daily Puzzle", "description": "A daily chess puzzle",
            "category": "engagement", "source": "chess.com"}]
    result = _normalize_features(raw)
    feat = result[0]
    assert feat["description"] == "A daily chess puzzle"
    assert feat["category"] == "engagement"
    assert feat["source"] == "chess.com"


def test_normalize_features_multiple_items_get_sequential_ids():
    raw = [{"name": "Feature A"}, {"name": "Feature B"}]
    result = _normalize_features(raw)
    assert result[0]["id"].startswith("000_")
    assert result[1]["id"].startswith("001_")


# ---------------------------------------------------------------------------
# 2. _slug
# ---------------------------------------------------------------------------

def test_slug_basic():
    assert _slug("Daily Puzzle") == "daily_puzzle"


def test_slug_special_characters():
    assert _slug("PvP — Live!") == "pvp_live"


def test_slug_numbers():
    assert _slug("Top 10 Players") == "top_10_players"


def test_slug_empty():
    result = _slug("")
    assert result == "feature"


def test_slug_long_name_truncated():
    long = "a" * 60
    assert len(_slug(long)) <= 40


# ---------------------------------------------------------------------------
# 3. Priority score formula
# ---------------------------------------------------------------------------

def test_priority_score_formula_weights():
    """retention*0.4 + activation*0.3 + acquisition*0.15 + revenue*0.1 + referral*0.05"""
    from user_soul.feature_filter import _compute_priority_score

    aarrr = FeatureAAR(
        feature_id="f1",
        acquisition=1.0,
        activation=0.0,
        retention=0.0,
        revenue=0.0,
        referral=0.0,
        confidence=1.0,
        archetype_votes={},
    )
    # Only acquisition=1.0 → score = 0.15
    assert abs(_compute_priority_score(aarrr) - 0.15) < 1e-6

    aarrr2 = FeatureAAR(
        feature_id="f2",
        acquisition=0.0,
        activation=0.0,
        retention=1.0,
        revenue=0.0,
        referral=0.0,
        confidence=1.0,
        archetype_votes={},
    )
    # Only retention=1.0 → score = 0.40
    assert abs(_compute_priority_score(aarrr2) - 0.40) < 1e-6

    # All ones → score = 1.0
    aarrr3 = FeatureAAR(
        feature_id="f3",
        acquisition=1.0, activation=1.0, retention=1.0,
        revenue=1.0, referral=1.0,
        confidence=1.0, archetype_votes={},
    )
    assert abs(_compute_priority_score(aarrr3) - 1.0) < 1e-6


# ---------------------------------------------------------------------------
# 4. Classification thresholds
# ---------------------------------------------------------------------------

def test_classification_thresholds():
    from user_soul.feature_filter import _classify

    assert _classify(0.6) == "must_have"
    assert _classify(0.75) == "must_have"
    assert _classify(1.0) == "must_have"
    assert _classify(0.35) == "nice_to_have"
    assert _classify(0.5) == "nice_to_have"
    assert _classify(0.599) == "nice_to_have"
    assert _classify(0.34) == "skip"
    assert _classify(0.0) == "skip"


# ---------------------------------------------------------------------------
# 5. Batching: 31 features → aarrr called twice
# ---------------------------------------------------------------------------

def test_filter_batching_triggers_two_aarrr_calls():
    """31 features should trigger two batches of aarrr scoring."""
    raw_features = [{"name": f"Feature {i}"} for i in range(31)]

    backend = MagicMock()
    # First call: archetypes auto-generation (text)
    # Subsequent calls: aarrr scoring via VoteEngine (also text)
    # We need to return archetype JSON on first call, then aarrr arrays

    archetype_json = json.dumps([
        {"name": "Engaged User", "description": "active", "background_story": "story1"},
        {"name": "Casual User",  "description": "passive", "background_story": "story2"},
    ])

    call_counter = {"n": 0}

    def side_effect(prompt, **kw):
        call_counter["n"] += 1
        # First call is archetype generation
        if call_counter["n"] == 1:
            return archetype_json
        # Subsequent calls are aarrr scoring
        # Extract feature ids from prompt (VoteEngine includes them)
        # Return empty array — we just need to count calls
        return "[]"

    backend.text.side_effect = side_effect

    ff = FeatureFilter(backend)
    report = ff.filter(
        product_description="chess app",
        raw_features=raw_features,
        target_segment="beginners",
    )

    # First call = archetypes, calls 2 and 3 = two aarrr batches (25 + 6)
    assert call_counter["n"] == 3, (
        f"Expected 3 backend.text calls (1 archetype + 2 aarrr batches), got {call_counter['n']}"
    )
    assert report.total_input == 31


# ---------------------------------------------------------------------------
# 6. top_n truncation
# ---------------------------------------------------------------------------

def test_filter_returns_top_n():
    """With top_n=5, top_features should have at most 5 entries."""
    raw_features = [{"name": f"Feature {i}"} for i in range(10)]
    archetypes = [_make_archetype()]

    backend = MagicMock()
    # Build response with all 10 ids
    normalized = _normalize_features(raw_features)
    ids = [f["id"] for f in normalized]
    backend.text.return_value = _make_aarrr_response(ids, score=0.7)

    ff = FeatureFilter(backend)
    report = ff.filter(
        product_description="chess app",
        raw_features=raw_features,
        target_segment="beginners",
        archetypes=archetypes,
        top_n=5,
    )

    assert len(report.top_features) <= 5


# ---------------------------------------------------------------------------
# 7. _auto_archetypes fallback on bad JSON
# ---------------------------------------------------------------------------

def test_auto_archetypes_fallback_on_invalid_json():
    """When LLM returns non-parseable text, fallback to 2 default archetypes."""
    backend = MagicMock()
    backend.text.return_value = "THIS IS NOT JSON AT ALL !!!"

    result = _auto_archetypes(backend, "chess app", "beginners")
    assert len(result) == 2
    names = {a.name for a in result}
    assert "Engaged User" in names
    assert "Casual User" in names


def test_auto_archetypes_valid_json_builds_archetypes():
    backend = MagicMock()
    backend.text.return_value = json.dumps([
        {"name": "Power Player", "description": "hardcore", "background_story": "plays daily"},
        {"name": "Newbie",       "description": "beginner",  "background_story": "just started"},
        {"name": "Social Chess", "description": "plays with friends", "background_story": ""},
    ])

    result = _auto_archetypes(backend, "chess app", "18-25 beginners")
    assert len(result) == 3
    assert result[0].name == "Power Player"
    # Frequencies should sum to ~1.0
    total = sum(a.frequency for a in result)
    assert abs(total - 1.0) < 1e-4


# ---------------------------------------------------------------------------
# 8. Full end-to-end mock
# ---------------------------------------------------------------------------

def test_full_filter_mock_report_structure():
    """End-to-end mock: verify FeatureFilterReport has correct structure."""
    raw_features = [
        {"name": "Daily Puzzle",      "category": "engagement", "source": "chess.com"},
        {"name": "Live Matchmaking",  "category": "gameplay",   "source": "chess.com"},
        {"name": "Obscure Setting",   "category": "settings",   "source": "lichess.org"},
    ]

    normalized = _normalize_features(raw_features)
    ids = [f["id"] for f in normalized]

    # Assign different scores to land in different tiers:
    # id[0] → retention=0.9, activation=0.9 → must_have
    # id[1] → retention=0.4, activation=0.4 → nice_to_have
    # id[2] → retention=0.1, activation=0.1 → skip

    def _aarrr_item(fid: str, score: float) -> dict:
        return {
            "feature_id": fid,
            "archetype_votes": {
                "Engaged User": {k: score for k in ("acquisition", "activation", "retention", "revenue", "referral")},
            },
            "mean": {k: score for k in ("acquisition", "activation", "retention", "revenue", "referral")},
        }

    aarrr_response = json.dumps([
        _aarrr_item(ids[0], 0.9),   # score = 0.9 → must_have
        _aarrr_item(ids[1], 0.45),  # score = 0.45 → nice_to_have
        _aarrr_item(ids[2], 0.1),   # score = 0.1 → skip
    ])

    archetypes = [_make_archetype("Engaged User")]

    backend = MagicMock()
    backend.text.return_value = aarrr_response

    ff = FeatureFilter(backend)
    report = ff.filter(
        product_description="mobile chess app for beginners",
        raw_features=raw_features,
        target_segment="18-25 beginner chess players",
        archetypes=archetypes,
        top_n=10,
    )

    # Structure checks
    assert isinstance(report, FeatureFilterReport)
    assert report.product == "mobile chess app for beginners"
    assert report.target_segment == "18-25 beginner chess players"
    assert report.total_input == 3
    assert report.archetypes_used == ["Engaged User"]

    # Tier counts
    assert len(report.must_have) == 1
    assert len(report.nice_to_have) == 1
    assert len(report.skip) == 1

    # Total across tiers = total_input
    assert len(report.must_have) + len(report.nice_to_have) + len(report.skip) == report.total_input

    # ScoredFeature fields
    mh = report.must_have[0]
    assert isinstance(mh, ScoredFeature)
    assert mh.classification == "must_have"
    assert isinstance(mh.aarrr, FeatureAAR)
    assert mh.priority_score >= 0.6

    # top_features sorted descending
    scores = [f.priority_score for f in report.top_features]
    assert scores == sorted(scores, reverse=True)

    # top_features is a subset of all features
    assert len(report.top_features) <= 10


def test_full_filter_with_auto_archetypes():
    """When archetypes=None, LLM is called first to generate them."""
    raw_features = [{"name": "Chat"}, {"name": "Leaderboard"}]
    normalized = _normalize_features(raw_features)
    ids = [f["id"] for f in normalized]

    archetype_json = json.dumps([
        {"name": "Competitive", "description": "wins matter", "background_story": "ranked player"},
        {"name": "Social",      "description": "plays for fun", "background_story": "casual player"},
    ])
    aarrr_response = _make_aarrr_response(ids, score=0.5)

    call_num = {"n": 0}

    def side(prompt, **kw):
        call_num["n"] += 1
        if call_num["n"] == 1:
            return archetype_json
        return aarrr_response

    backend = MagicMock()
    backend.text.side_effect = side

    ff = FeatureFilter(backend)
    report = ff.filter("chess app", raw_features, "casual players")

    assert "Competitive" in report.archetypes_used
    assert "Social" in report.archetypes_used
    assert report.total_input == 2
