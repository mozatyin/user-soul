# mcv/tests/test_research_aarrr.py
import os
import pytest

API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")


def test_research_aarrr_returns_correct_count():
    """Each input feature gets exactly one FeatureAAR."""
    if not API_KEY:
        pytest.skip("no ANTHROPIC_API_KEY")
    from user_soul.voter import MCVClient
    client = MCVClient(api_key=API_KEY)
    features = [
        {"id": "invite_friends", "name": "Invite Friends", "description": "Invite friends to join rooms"},
        {"id": "ludo",           "name": "Ludo Game",      "description": "Multiplayer board game"},
        {"id": "daily_tasks",    "name": "Daily Tasks",    "description": "Daily quest checklist"},
    ]
    result = client.research_aarrr(
        product_description="Arabic social gaming platform with voice rooms and board games",
        features=features,
    )
    assert len(result) == 3
    ids = {f.feature_id for f in result}
    assert ids == {"invite_friends", "ludo", "daily_tasks"}


def test_research_aarrr_scores_in_range():
    """All AARRR scores are 0.0–1.0, confidence is 0.0–1.0."""
    if not API_KEY:
        pytest.skip("no ANTHROPIC_API_KEY")
    from user_soul.voter import MCVClient
    client = MCVClient(api_key=API_KEY)
    features = [{"id": "coins", "name": "Coins", "description": "Virtual currency for gifts"}]
    result = client.research_aarrr(
        product_description="Social gifting app",
        features=features,
    )
    f = result[0]
    for dim in (f.acquisition, f.activation, f.retention, f.revenue, f.referral, f.confidence):
        assert 0.0 <= dim <= 1.0, f"dimension out of range: {dim}"


def test_research_aarrr_fallback_on_empty_features():
    """Empty feature list → empty result, no crash."""
    from user_soul.voter import MCVClient
    client = MCVClient(api_key=API_KEY or "dummy-key-not-used")
    result = client.research_aarrr(product_description="any app", features=[])
    assert result == []


def test_research_aarrr_invite_friends_referral_high():
    """Invite Friends must score high on referral relative to other dimensions."""
    if not API_KEY:
        pytest.skip("no ANTHROPIC_API_KEY")
    from user_soul.voter import MCVClient
    client = MCVClient(api_key=API_KEY)
    features = [
        {"id": "invite_friends", "name": "Invite Friends",
         "description": "Send invitation links to friends to join the platform"},
    ]
    result = client.research_aarrr(
        product_description="Arabic social gaming platform with multiplayer games",
        features=features,
    )
    f = result[0]
    assert f.referral >= 0.5, f"invite_friends referral should be ≥0.5, got {f.referral}"
    assert f.referral >= f.revenue, "referral should dominate revenue for invite feature"
