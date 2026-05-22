# mcv/tests/test_attribute_frictions.py
import os
import pytest

API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")


def test_attribute_frictions_empty_frictions_no_llm():
    """No frictions → empty defects, no LLM call, no crash."""
    from user_soul.voter import MCVClient
    client = MCVClient(api_key="dummy")  # no API call for empty frictions
    manifest = client.attribute_frictions(
        product="any app", frictions=[], features=[]
    )
    assert "defects" in manifest
    assert manifest["defects"] == []


def test_attribute_frictions_passthrough_metadata():
    """game_name and original_slug are passed through to manifest."""
    from user_soul.voter import MCVClient
    client = MCVClient(api_key="dummy")
    manifest = client.attribute_frictions(
        product="any app", frictions=[],
        features=[], game_name="GAMZEE", original_slug="gamzee_v8",
    )
    assert manifest.get("game_name") == "GAMZEE"
    assert manifest.get("original_slug") == "gamzee_v8"


def test_attribute_frictions_returns_defects_key():
    """Output must have 'defects' key — direct input to reforge()."""
    if not API_KEY:
        pytest.skip("no ANTHROPIC_API_KEY")
    from user_soul.voter import MCVClient
    client = MCVClient(api_key=API_KEY)
    manifest = client.attribute_frictions(
        product="Arabic social gaming platform with Ludo and voice rooms",
        frictions=["Ludo需要好友门槛高", "新手引导不足"],
        features=[
            {"id": "ludo",           "name": "Ludo",           "description": "multiplayer board game"},
            {"id": "invite_friends", "name": "Invite Friends", "description": "invite friends"},
            {"id": "tutorial",       "name": "Tutorial",       "description": "onboarding tutorial"},
        ],
    )
    assert "defects" in manifest
    assert isinstance(manifest["defects"], list)


def test_attribute_frictions_defect_shape():
    """Each defect must have type/severity/description/affected_screens/suggested_fix."""
    if not API_KEY:
        pytest.skip("no ANTHROPIC_API_KEY")
    from user_soul.voter import MCVClient
    client = MCVClient(api_key=API_KEY)
    manifest = client.attribute_frictions(
        product="social gaming app",
        frictions=["onboarding is confusing"],
        features=[{"id": "tutorial", "name": "Tutorial", "description": "first-run tutorial"}],
    )
    assert len(manifest["defects"]) >= 1
    d = manifest["defects"][0]
    assert "type" in d
    assert "severity" in d
    assert "description" in d
    assert "affected_screens" in d
    assert "suggested_fix" in d
    assert d["type"] in ("design", "ux", "rules")
    assert d["severity"] in ("P0", "P1", "P2")
