"""Domain configurations for UserSimulator — controls session 'world'."""
from __future__ import annotations
from dataclasses import dataclass


@dataclass
class DomainConfig:
    session_framing: str               # "你开始了一局游戏" / "你打开了 app" / "你在刷新闻"
    emotional_states: list[str]
    triggers: list[str]
    time_options: list[str]
    user_roles: dict[str, list[int]]   # role_name → [usage_day values] (discrete pick-list)


GameDomainConfig = DomainConfig(
    session_framing="你开始了一局游戏",
    emotional_states=["competitive", "casual", "tilted", "bored", "hyped"],
    triggers=["want_to_rank_up", "friend_challenged_me", "kill_time", "daily_login", "revenge_match"],
    time_options=["morning_commute", "lunch_break", "evening", "late_night"],
    user_roles={
        "Newcomer": [1, 3],
        "Casual":   [3, 14],
        "Grinder":  [14, 30],
        "Veteran":  [30],
    },
)

AppDomainConfig = DomainConfig(
    session_framing="你打开了这个 app",
    emotional_states=["stressed", "calm", "bored", "excited", "sad", "anxious"],
    triggers=["habit", "work_stress", "relationship_tension", "boredom", "notification", "curiosity"],
    time_options=["morning_commute", "lunch_break", "evening_wind_down", "night"],
    user_roles={
        "Explorer":  [1, 3],
        "Skeptic":   [3, 7],
        "Habituer":  [14, 30],
        "Advocate":  [30],
    },
)

WebDomainConfig = DomainConfig(
    session_framing="你在刷新闻",
    emotional_states=["curious", "bored", "anxious", "relaxed", "rushed"],
    triggers=["morning_routine", "notification", "kill_time", "topic_interest", "breaking_news"],
    time_options=["morning_commute", "lunch_break", "evening", "late_night"],
    user_roles={
        "Casual":      [1, 7],
        "Regular":     [7, 30],
        "PowerReader": [30],
    },
)


def build_domain_config(
    product_description: str,
    api_key: str,
    cache_path: "Path | None" = None,
) -> "DomainConfig":
    """Generate a DomainConfig for any product via one Sonnet LLM call.

    Result is optionally cached to cache_path (JSON). On cache hit, LLM is skipped.
    Falls back to AppDomainConfig fields if LLM output is unparseable.
    """
    from pathlib import Path as _Path
    import json as _json
    import user_soul.core as _core

    # Cache hit
    if cache_path is not None:
        p = _Path(cache_path)
        if p.exists():
            try:
                data = _json.loads(p.read_text(encoding="utf-8"))
                return _domain_config_from_dict(data)
            except Exception:
                pass

    prompt = (
        "You are a UX researcher. For the following product, define a behavioral simulation config.\n\n"
        f"Product: {product_description[:1500]}\n\n"
        "Return ONLY valid JSON (no markdown):\n"
        '{"session_framing": "你在...", '
        '"emotional_states": ["state1", "state2", "state3", "state4", "state5"], '
        '"triggers": ["trigger1", "trigger2", "trigger3", "trigger4", "trigger5"], '
        '"time_options": ["morning_commute", "lunch_break", "evening", "late_night"], '
        '"user_roles": {"RoleName": [day_min, day_max]}}\n\n'
        "Rules:\n"
        "- session_framing: Chinese, starts with 你在/你打开了/你开始了\n"
        "- emotional_states: 4-6 states specific to this product domain\n"
        "- triggers: 4-6 triggers that bring users to this product\n"
        "- time_options: 3-4 time slots\n"
        "- user_roles: 3-4 named lifecycle stages with [min_day, max_day] usage ranges"
    )
    raw, _ = _core._llm_call(prompt, api_key, max_tokens=512)
    data = _core._safe_json(raw)
    cfg = _domain_config_from_dict(data) if data else _fallback_domain_config()

    if cache_path is not None:
        try:
            p = _Path(cache_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(_json.dumps({
                "session_framing": cfg.session_framing,
                "emotional_states": cfg.emotional_states,
                "triggers": cfg.triggers,
                "time_options": cfg.time_options,
                "user_roles": cfg.user_roles,
            }, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            pass

    return cfg


def _domain_config_from_dict(data: dict) -> "DomainConfig":
    """Parse LLM JSON dict into DomainConfig. Falls back to AppDomainConfig values on bad fields."""
    emotional_states = data.get("emotional_states", AppDomainConfig.emotional_states)
    triggers        = data.get("triggers",        AppDomainConfig.triggers)
    time_options    = data.get("time_options",    AppDomainConfig.time_options)
    session_framing = data.get("session_framing", AppDomainConfig.session_framing)
    raw_roles       = data.get("user_roles",      {})

    user_roles: dict[str, list[int]] = {}
    for role, val in raw_roles.items():
        if isinstance(val, (list, tuple)) and len(val) >= 1:
            user_roles[str(role)] = [int(v) for v in val[:2]]
    if not user_roles:
        user_roles = dict(AppDomainConfig.user_roles)

    if not isinstance(emotional_states, list) or not emotional_states:
        emotional_states = list(AppDomainConfig.emotional_states)
    if not isinstance(triggers, list) or not triggers:
        triggers = list(AppDomainConfig.triggers)
    if not isinstance(time_options, list) or not time_options:
        time_options = list(AppDomainConfig.time_options)

    return DomainConfig(
        session_framing=str(session_framing),
        emotional_states=[str(s) for s in emotional_states],
        triggers=[str(t) for t in triggers],
        time_options=[str(o) for o in time_options],
        user_roles=user_roles,
    )


def _fallback_domain_config() -> "DomainConfig":
    """Return AppDomainConfig-equivalent as fallback when LLM parsing fails."""
    return DomainConfig(
        session_framing=AppDomainConfig.session_framing,
        emotional_states=list(AppDomainConfig.emotional_states),
        triggers=list(AppDomainConfig.triggers),
        time_options=list(AppDomainConfig.time_options),
        user_roles=dict(AppDomainConfig.user_roles),
    )
