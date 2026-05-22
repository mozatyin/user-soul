"""Scenario context for behavioral simulation."""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from user_soul.domain_configs import DomainConfig


@dataclass
class ScenarioContext:
    time_of_day: str       # when the user opens the app
    emotional_state: str   # their current mood
    usage_day: int         # how many days they've been using the app
    trigger: str           # what made them open it


_TIME_OPTIONS = ["morning_commute", "lunch_break", "evening_wind_down", "night"]
_EMOTIONAL_OPTIONS = ["stressed", "calm", "bored", "excited", "sad", "anxious"]
_USAGE_DAY_OPTIONS = [1, 3, 7, 14, 30]
_TRIGGER_OPTIONS = ["habit", "work_stress", "relationship_tension",
                    "boredom", "notification", "curiosity"]

ROLE_DAY_RANGES: dict[str, list[int]] = {
    "Explorer":  [1, 3],
    "Habituer":  [14, 30],
    "Skeptic":   [3, 7],
    "Advocate":  [30],
}

_ROLE_TRIGGER_WEIGHTS: dict[str, dict[str, float]] = {
    "Explorer":  {"curiosity": 0.5, "boredom": 0.3, "notification": 0.2},
    "Habituer":  {"habit": 0.6, "work_stress": 0.2, "boredom": 0.2},
    "Skeptic":   {"boredom": 0.4, "notification": 0.3, "curiosity": 0.3},
    "Advocate":  {"habit": 0.5, "relationship_tension": 0.3, "work_stress": 0.2},
}


def random_context(role: str | None = None) -> ScenarioContext:
    """Generate a randomized scenario context, optionally constrained by persona role."""
    time_of_day = random.choice(_TIME_OPTIONS)
    emotional_state = random.choice(_EMOTIONAL_OPTIONS)

    if role and role in ROLE_DAY_RANGES:
        usage_day = random.choice(ROLE_DAY_RANGES[role])
    else:
        usage_day = random.choice(_USAGE_DAY_OPTIONS)

    if role and role in _ROLE_TRIGGER_WEIGHTS:
        weights = _ROLE_TRIGGER_WEIGHTS[role]
        trigger = random.choices(list(weights.keys()), weights=list(weights.values()))[0]
    else:
        trigger = random.choice(_TRIGGER_OPTIONS)

    return ScenarioContext(
        time_of_day=time_of_day,
        emotional_state=emotional_state,
        usage_day=usage_day,
        trigger=trigger,
    )


def random_context_for_domain(
    role: str | None = None,
    domain_config: "DomainConfig | None" = None,
) -> ScenarioContext:
    """Generate scenario context using DomainConfig options.

    Falls back to random_context() if domain_config is None.
    """
    if domain_config is None:
        return random_context(role=role)

    time_of_day = random.choice(domain_config.time_options)
    emotional_state = random.choice(domain_config.emotional_states)

    if role and role in domain_config.user_roles:
        usage_day = random.choice(domain_config.user_roles[role])
    else:
        all_days = [d for days in domain_config.user_roles.values() for d in days]
        usage_day = random.choice(all_days) if all_days else 1

    trigger = random.choice(domain_config.triggers)

    return ScenarioContext(
        time_of_day=time_of_day,
        emotional_state=emotional_state,
        usage_day=usage_day,
        trigger=trigger,
    )
