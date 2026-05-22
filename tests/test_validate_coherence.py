# mcv/tests/test_validate_coherence.py
import os
import pytest

API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")


def test_validate_coherence_detects_missing_enabler_rule_based():
    """Ludo selected without Invite Friends → rule-based detects gap, no LLM needed."""
    from user_soul.voter import MCVClient
    client = MCVClient(api_key="dummy")  # no API key needed for rule-based pass
    selected = [{"id": "ludo", "name": "Ludo", "description": "multiplayer board game"}]
    report = client.validate_coherence(
        product_description="Arabic social gaming platform",
        selected_features=selected,
        dropped_features=None,  # no dropped features → no LLM call
    )
    assert not report.is_coherent
    assert len(report.missing_dependencies) >= 1
    assert len(report.blocked_journeys) >= 1


def test_validate_coherence_detects_missing_enabler_with_reinstate():
    """Ludo selected, invite_friends dropped → reinstate_recommendations includes invite_friends."""
    if not API_KEY:
        pytest.skip("no ANTHROPIC_API_KEY")
    from user_soul.voter import MCVClient
    client = MCVClient(api_key=API_KEY)
    selected = [{"id": "ludo", "name": "Ludo", "description": "multiplayer board game"}]
    dropped  = [{"id": "invite_friends", "name": "Invite Friends",
                 "description": "invite friends to join rooms"}]
    report = client.validate_coherence(
        product_description="Arabic social gaming platform",
        selected_features=selected,
        dropped_features=dropped,
    )
    assert not report.is_coherent
    assert "invite_friends" in report.reinstate_recommendations


def test_validate_coherence_coherent_when_enabler_present():
    """Both ludo and invite_friends selected → no gap."""
    from user_soul.voter import MCVClient
    client = MCVClient(api_key="dummy")
    selected = [
        {"id": "ludo",           "name": "Ludo",           "description": "multiplayer board game"},
        {"id": "invite_friends", "name": "Invite Friends", "description": "invite friends to join"},
    ]
    report = client.validate_coherence(
        product_description="Arabic social gaming platform",
        selected_features=selected,
    )
    assert report.is_coherent
    assert report.reinstate_recommendations == []


def test_validate_coherence_empty_selected():
    """Empty feature set → coherent (no dependency violations on empty set), no crash."""
    from user_soul.voter import MCVClient
    client = MCVClient(api_key="dummy")
    report = client.validate_coherence(
        product_description="any app", selected_features=[]
    )
    assert isinstance(report.is_coherent, bool)
    # empty set has no dependency violations
    assert report.is_coherent


def test_validate_coherence_no_social_features():
    """Non-social features → coherent (no dependency violations)."""
    from user_soul.voter import MCVClient
    client = MCVClient(api_key="dummy")
    selected = [
        {"id": "daily_tasks", "name": "Daily Tasks",   "description": "daily quest checklist"},
        {"id": "coins",       "name": "Coins",         "description": "virtual currency"},
    ]
    report = client.validate_coherence(
        product_description="casual mobile game", selected_features=selected
    )
    assert report.is_coherent
