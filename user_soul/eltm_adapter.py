"""eltm_adapter.py — Glue layer between ELTM build_core() output and FeatureFilter.

Converts the nested build_core_output dict into a flat list[dict] that
FeatureFilter.filter() can consume directly.  No LLM calls; pure data
transformation.
"""
from __future__ import annotations

import re
from typing import Any


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _slug(text: str, max_len: int = 40) -> str:
    """Convert arbitrary text to a safe snake_case fragment."""
    s = str(text).lower().strip()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = s.strip("_")
    return (s[:max_len] if s else "feature")


def _dedup(features: list[dict]) -> list[dict]:
    """Keep first occurrence of each id."""
    seen: set[str] = set()
    result: list[dict] = []
    for f in features:
        fid = f.get("id", "")
        if fid not in seen:
            seen.add(fid)
            result.append(f)
    return result


def _make_feature(
    id: str,
    name: str,
    description: str,
    category: str,
    source: str,
) -> dict:
    return {
        "id": id,
        "name": name,
        "description": description,
        "category": category,
        "source": source,
    }


# ---------------------------------------------------------------------------
# Extraction helpers (one per source)
# ---------------------------------------------------------------------------

def _extract_moon_features(output: dict) -> list[dict]:
    features: list[dict] = []
    for item in output.get("moon_features") or []:
        if not isinstance(item, dict):
            continue
        name = item.get("name") or ""
        raw_id = item.get("id") or ""
        fid = raw_id if raw_id else f"feat_{_slug(name)}"
        features.append(_make_feature(
            id=fid,
            name=name,
            description=item.get("description") or "",
            category=item.get("category") or "core",
            source="moon_features",
        ))
    return features


def _extract_user_journey(formal_rules: dict) -> list[dict]:
    features: list[dict] = []
    user_journey = (formal_rules.get("user_journey") or {})
    first_time_flow = user_journey.get("first_time_flow") or []
    for step in first_time_flow:
        if not isinstance(step, dict):
            continue
        step_num = step.get("step", 0)
        screen = step.get("screen") or f"screen_{step_num}"
        purpose = step.get("purpose") or ""
        fid = f"journey_step_{step_num}_{_slug(screen)}"
        features.append(_make_feature(
            id=fid,
            name=f"Step {step_num}: {screen}",
            description=purpose,
            category="onboarding",
            source="user_journey",
        ))
    return features


def _extract_difficulty_levels(onboarding: dict) -> list[dict]:
    levels = onboarding.get("difficulty_levels") or []
    if not levels:
        return []
    labels = [lv.get("label", "") for lv in levels if isinstance(lv, dict)]
    description = "Skill level selection: " + ", ".join(labels) if labels else "Skill level selection"
    return [_make_feature(
        id="onboard_skill_level",
        name="Skill Level Selection",
        description=description,
        category="onboarding",
        source="onboarding",
    )]


def _extract_game_modes(formal_rules: dict) -> list[dict]:
    features: list[dict] = []
    for mode in (formal_rules.get("game_modes") or []):
        if not isinstance(mode, dict):
            continue
        mode_id = mode.get("id") or _slug(mode.get("label") or "mode")
        label = mode.get("label") or mode_id
        description = mode.get("description") or ""
        is_free = mode.get("free", True)
        if description and not is_free:
            description += " (premium)"
        features.append(_make_feature(
            id=f"mode_{mode_id}",
            name=label,
            description=description,
            category="game_mode",
            source="game_modes",
        ))
    return features


def _extract_gameplay_controls(gameplay_controls: dict) -> list[dict]:
    features: list[dict] = []
    # during_game controls
    for ctrl in (gameplay_controls.get("during_game") or []):
        if not isinstance(ctrl, dict):
            continue
        ctrl_id = ctrl.get("id") or _slug(ctrl.get("label") or "ctrl")
        label = ctrl.get("label") or ctrl_id
        behavior = ctrl.get("behavior") or ""
        features.append(_make_feature(
            id=f"ctrl_{ctrl_id}",
            name=label,
            description=behavior,
            category="gameplay",
            source="gameplay_controls",
        ))
    # time_controls — collapse to single feature
    time_controls = gameplay_controls.get("time_controls") or []
    if time_controls:
        labels = [tc.get("label", "") for tc in time_controls if isinstance(tc, dict)]
        features.append(_make_feature(
            id="ctrl_time_controls",
            name="Time Controls",
            description="Available time controls: " + ", ".join(labels),
            category="gameplay",
            source="gameplay_controls",
        ))
    return features


def _extract_engagement(engagement: dict) -> list[dict]:
    features: list[dict] = []
    # bool fields: streak, daily_puzzle
    bool_fields = {
        "streak": ("engage_streak", "Daily Streak", "Daily streak reward system"),
        "daily_puzzle": ("engage_daily_puzzle", "Daily Puzzle", "A new puzzle every day to solve"),
    }
    for field, (fid, name, desc) in bool_fields.items():
        val = engagement.get(field)
        if val is True:
            features.append(_make_feature(
                id=fid,
                name=name,
                description=desc,
                category="engagement",
                source="engagement",
            ))
    # progression string
    progression = engagement.get("progression")
    if progression and isinstance(progression, str):
        features.append(_make_feature(
            id="engage_progression",
            name="Progression System",
            description=progression,
            category="engagement",
            source="engagement",
        ))
    # social list
    for item in (engagement.get("social") or []):
        if not item:
            continue
        item_str = str(item)
        features.append(_make_feature(
            id=f"social_{_slug(item_str)}",
            name=item_str,
            description=f"Social feature: {item_str}",
            category="social",
            source="engagement",
        ))
    return features


def _extract_benchmark_gaps(formal_rules: dict) -> list[dict]:
    features: list[dict] = []
    for gap in (formal_rules.get("benchmark_gaps") or []):
        if not gap or not isinstance(gap, str):
            continue
        slug = _slug(gap[:30])
        features.append(_make_feature(
            id=f"gap_{slug}",
            name=gap,
            description=f"Benchmark gap: {gap}",
            category="core",
            source="benchmark_gap",
        ))
    return features


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_features(build_core_output: dict) -> list[dict]:
    """Flatten build_core() output into a list[dict] for FeatureFilter.

    Each returned dict has keys: id, name, description, category, source.
    Duplicates (by id) are dropped — first occurrence wins.
    """
    if not isinstance(build_core_output, dict):
        return []

    formal_rules: dict = build_core_output.get("formal_rules") or {}
    onboarding: dict = (formal_rules.get("onboarding") or {})
    gameplay_controls: dict = (formal_rules.get("gameplay_controls") or {})
    engagement: dict = (formal_rules.get("engagement") or {})

    features: list[dict] = []
    features.extend(_extract_moon_features(build_core_output))
    features.extend(_extract_user_journey(formal_rules))
    features.extend(_extract_difficulty_levels(onboarding))
    features.extend(_extract_game_modes(formal_rules))
    features.extend(_extract_gameplay_controls(gameplay_controls))
    features.extend(_extract_engagement(engagement))
    features.extend(_extract_benchmark_gaps(formal_rules))

    return _dedup(features)


def extract_benchmark_name(build_core_output: dict) -> str:
    """Return the benchmark/competitor name, e.g. 'Chess.com'. Empty string if not found."""
    if not isinstance(build_core_output, dict):
        return ""
    benchmark_ref = build_core_output.get("benchmark_reference") or {}
    return (benchmark_ref.get("name") or "")


def build_product_description(build_core_output: dict) -> str:
    """Build a ~100-word plain-text product description for ABValidator.

    Pulls: game_name, key formal_rules dimensions, first 5 moon_features.
    """
    if not isinstance(build_core_output, dict):
        return ""

    parts: list[str] = []

    game_name: str = build_core_output.get("game_name") or ""
    if game_name:
        parts.append(f"Product: {game_name}.")

    formal_rules: dict = build_core_output.get("formal_rules") or {}

    # Onboarding summary
    onboarding: dict = (formal_rules.get("onboarding") or {})
    rules_summary = onboarding.get("rules_summary") or ""
    if rules_summary:
        parts.append(f"Onboarding: {rules_summary}.")

    # Game modes
    game_modes: list[dict] = (formal_rules.get("game_modes") or [])
    if game_modes:
        mode_labels = [m.get("label", "") for m in game_modes if isinstance(m, dict) and m.get("label")]
        if mode_labels:
            parts.append("Game modes: " + ", ".join(mode_labels[:5]) + ".")

    # Engagement
    engagement: dict = (formal_rules.get("engagement") or {})
    engage_items: list[str] = []
    if engagement.get("streak"):
        engage_items.append("daily streak")
    if engagement.get("daily_puzzle"):
        engage_items.append("daily puzzle")
    social = engagement.get("social") or []
    if social:
        engage_items.append(f"{len(social)} social features")
    if engage_items:
        parts.append("Engagement: " + ", ".join(engage_items) + ".")

    # Top moon_features
    moon_features: list[dict] = (build_core_output.get("moon_features") or [])
    top5 = [f.get("name", "") for f in moon_features[:5] if isinstance(f, dict) and f.get("name")]
    if top5:
        parts.append("Key features: " + ", ".join(top5) + ".")

    # Benchmark reference
    benchmark_name = extract_benchmark_name(build_core_output)
    if benchmark_name:
        parts.append(f"Benchmarked against {benchmark_name}.")

    return " ".join(parts)
