"""Tests for the UserSoulClient interfaces that surface previously-orphaned
legacy MCV capabilities onto the entry path:

  - decide() / score()        → VoteEngine (persona-cohort decisions)
  - validate_coherence()      → MCVClient (dropped-dependency detection)
  - attribute_frictions()     → MCVClient (friction → Code-Soul defect manifest)

Before this, these capabilities existed only in voter.py / engines and could not
be reached from UserSoulClient — the §5 "capability not on the entry path" trap.
"""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

from user_soul.client import UserSoulClient
from user_soul.models import AgentProfile, DecisionResult
from user_soul.report import CoherenceReport


def _personas(n=3):
    return [AgentProfile(agent_id=f"a{i}", archetype_name="Casual",
                         trait_vector={"patience": 6.0},
                         background_story=f"persona {i}") for i in range(n)]


# ---------------------------------------------------------------------------
# decide / score → VoteEngine
# ---------------------------------------------------------------------------

def test_decide_returns_majority_choice():
    backend = MagicMock()
    backend.text.side_effect = [
        '{"choice": "Must-Have", "reasoning": "core"}',
        '{"choice": "Must-Have", "reasoning": "core"}',
        '{"choice": "Delighter", "reasoning": "nice"}',
    ]
    client = UserSoulClient(backend)
    r = client.decide("Kano category?", ["Must-Have", "Delighter"],
                      context="a chess app", personas=_personas(3))
    assert isinstance(r, DecisionResult)
    assert r.value == "Must-Have"
    assert r.confidence == round(2 / 3, 4)
    assert set(r.distribution) == {"Must-Have", "Delighter"}


def test_score_returns_cohort_mean():
    backend = MagicMock()
    backend.text.side_effect = [
        '{"score": 4, "reasoning": "good"}',
        '{"score": 5, "reasoning": "great"}',
    ]
    client = UserSoulClient(backend)
    r = client.score("How fun?", 1, 5, context="a chess app", personas=_personas(2))
    assert isinstance(r, DecisionResult)
    assert r.value == 4.5


def test_decide_auto_generates_personas_when_none():
    backend = MagicMock()
    backend.text.return_value = '{"choice": "Must-Have", "reasoning": "x"}'
    client = UserSoulClient(backend)
    with patch.object(client._persona, "get_or_create",
                      return_value=_personas(4)) as mock_pool:
        client.decide("Kano?", ["Must-Have", "Delighter"], context="chess app")
    mock_pool.assert_called_once()


# ---------------------------------------------------------------------------
# validate_coherence → MCVClient (rule-based pass needs no LLM)
# ---------------------------------------------------------------------------

def _backend_with_key():
    from user_soul.backends.anthropic import AnthropicBackend
    return AnthropicBackend(api_key="sk-test")


def test_validate_coherence_flags_missing_social_enabler():
    client = UserSoulClient(_backend_with_key())
    selected = [
        {"id": "ludo", "name": "Ludo", "description": "multiplayer ludo match against an opponent"},
    ]
    report = client.validate_coherence("a multiplayer board game app", selected)
    assert isinstance(report, CoherenceReport)
    assert report.is_coherent is False
    assert report.missing_dependencies


def test_validate_coherence_passes_with_enabler():
    client = UserSoulClient(_backend_with_key())
    selected = [
        {"id": "ludo", "name": "Ludo", "description": "multiplayer ludo match against an opponent"},
        {"id": "invite", "name": "Invite", "description": "invite buddies via a share link"},
    ]
    report = client.validate_coherence("a multiplayer board game app", selected)
    assert report.is_coherent is True


def test_validate_coherence_handles_missing_id():
    # defensive _fid(): a feature with no "id" must not crash
    client = UserSoulClient(_backend_with_key())
    report = client.validate_coherence("app", [{"name": "Solo Mode", "description": "single player"}])
    assert report.is_coherent is True


# ---------------------------------------------------------------------------
# attribute_frictions → MCVClient
# ---------------------------------------------------------------------------

def test_attribute_frictions_empty_no_llm():
    client = UserSoulClient(_backend_with_key())
    manifest = client.attribute_frictions("a chess app", [], [],
                                          game_name="chess", original_slug="chess_v1")
    assert manifest == {"defects": [], "game_name": "chess", "original_slug": "chess_v1"}


def test_attribute_frictions_builds_manifest():
    client = UserSoulClient(_backend_with_key())
    llm_json = json.dumps({"defects": [
        {"type": "ux", "severity": "P0", "description": "onboarding too long",
         "affected_screens": ["onboarding"], "suggested_fix": "cut to 3 steps"},
    ]})
    with patch("user_soul.core._llm_call", return_value=(llm_json, 100)):
        manifest = client.attribute_frictions(
            "a chess app", ["users quit during onboarding"],
            [{"id": "onb", "name": "Onboarding"}],
            game_name="chess", original_slug="chess_v1")
    assert len(manifest["defects"]) == 1
    assert manifest["defects"][0]["severity"] == "P0"
    assert manifest["game_name"] == "chess"


def test_legacy_capability_requires_api_key_backend():
    backend = MagicMock(spec=[])  # no api_key attribute
    client = UserSoulClient(backend)
    import pytest
    with pytest.raises(ValueError, match="api_key"):
        client.validate_coherence("app", [{"id": "x", "name": "X"}])
