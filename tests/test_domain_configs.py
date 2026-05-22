import json as _json
import tempfile
from pathlib import Path
from unittest.mock import patch as _patch

from user_soul.domain_configs import DomainConfig, GameDomainConfig, AppDomainConfig, WebDomainConfig, build_domain_config
from user_soul.scenarios import random_context_for_domain, ScenarioContext


def test_game_domain_config_fields():
    assert GameDomainConfig.session_framing == "你开始了一局游戏"
    assert "competitive" in GameDomainConfig.emotional_states
    assert "Newcomer" in GameDomainConfig.user_roles
    assert "want_to_rank_up" in GameDomainConfig.triggers


def test_app_domain_config_fields():
    assert AppDomainConfig.session_framing == "你打开了这个 app"
    assert "stressed" in AppDomainConfig.emotional_states
    assert "work_stress" in AppDomainConfig.triggers
    assert "evening_wind_down" in AppDomainConfig.time_options
    assert "Explorer" in AppDomainConfig.user_roles


def test_web_domain_config_fields():
    assert WebDomainConfig.session_framing == "你在刷新闻"
    assert "curious" in WebDomainConfig.emotional_states
    assert "breaking_news" in WebDomainConfig.triggers
    assert "PowerReader" in WebDomainConfig.user_roles


def test_custom_domain_config():
    custom = DomainConfig(
        session_framing="你在体验VR",
        emotional_states=["immersed", "curious"],
        triggers=["boredom", "curiosity"],
        time_options=["evening", "weekend"],
        user_roles={"Newbie": [1, 7]},
    )
    assert custom.session_framing == "你在体验VR"
    assert "Newbie" in custom.user_roles


def test_random_context_for_domain_uses_config_options():
    ctx = random_context_for_domain(role="Newcomer", domain_config=GameDomainConfig)
    assert isinstance(ctx, ScenarioContext)
    assert ctx.emotional_state in GameDomainConfig.emotional_states
    assert ctx.trigger in GameDomainConfig.triggers
    assert ctx.usage_day in GameDomainConfig.user_roles["Newcomer"]


def test_random_context_for_domain_fallback_when_no_config():
    from user_soul.scenarios import ScenarioContext
    ctx = random_context_for_domain(role=None, domain_config=None)
    assert isinstance(ctx, ScenarioContext)


def test_random_context_for_domain_unknown_role_picks_any_day():
    ctx = random_context_for_domain(role="UnknownRole", domain_config=GameDomainConfig)
    assert isinstance(ctx, ScenarioContext)
    all_days = [d for days in GameDomainConfig.user_roles.values() for d in days]
    assert ctx.usage_day in all_days


def test_random_context_for_domain_produces_variance():
    contexts = [random_context_for_domain(domain_config=GameDomainConfig) for _ in range(20)]
    unique = {(c.time_of_day, c.emotional_state, c.usage_day, c.trigger) for c in contexts}
    assert len(unique) >= 3


_MOCK_RESPONSE = _json.dumps({
    "session_framing": "你在电商平台购物",
    "emotional_states": ["browsing", "deal_hunting", "impulse_buying", "comparing", "skeptical"],
    "triggers": ["sale_notification", "recommendation", "wish_list", "search", "flash_deal"],
    "time_options": ["lunch_break", "evening", "weekend_morning", "late_night"],
    "user_roles": {
        "Browser":   [1, 7],
        "Shopper":   [7, 30],
        "Loyalist":  [30, 30],
    },
})


def test_build_domain_config_returns_domain_config():
    with _patch("user_soul.core._llm_call") as mock_llm:
        mock_llm.return_value = (_MOCK_RESPONSE, 300)
        cfg = build_domain_config("电商 app，用户在这里浏览和购买商品", api_key="test")
    assert isinstance(cfg, DomainConfig)
    assert "browsing" in cfg.emotional_states
    assert "Browser" in cfg.user_roles
    assert cfg.session_framing == "你在电商平台购物"


def test_build_domain_config_caches_to_file():
    with tempfile.TemporaryDirectory() as tmp:
        cache = Path(tmp) / "domain_config.json"
        with _patch("user_soul.core._llm_call") as mock_llm:
            mock_llm.return_value = (_MOCK_RESPONSE, 300)
            cfg1 = build_domain_config("电商 app", api_key="test", cache_path=cache)
            cfg2 = build_domain_config("电商 app", api_key="test", cache_path=cache)
        # Second call must NOT call LLM (reads from cache)
        assert mock_llm.call_count == 1
        assert cfg2.session_framing == cfg1.session_framing


def test_build_domain_config_fallback_on_bad_json():
    with _patch("user_soul.core._llm_call") as mock_llm:
        mock_llm.return_value = ("not valid json at all", 50)
        cfg = build_domain_config("some app", api_key="test")
    assert isinstance(cfg, DomainConfig)
    assert len(cfg.emotional_states) > 0
