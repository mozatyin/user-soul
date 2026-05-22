"""Tests for user_soul.eltm_adapter."""
from __future__ import annotations

import pytest

from user_soul.eltm_adapter import (
    build_product_description,
    extract_benchmark_name,
    extract_features,
)


# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------

@pytest.fixture()
def full_output() -> dict:
    """Complete mock build_core_output covering all extraction paths."""
    return {
        "game_name": "Chess Pro",
        "moon_features": [
            {"id": "feat_puzzle", "name": "Daily Puzzle", "category": "core"},
            {"id": "feat_analysis", "name": "Game Analysis", "category": "core"},
            {"id": "", "name": "Openings Explorer", "category": "core"},   # empty id → slug
        ],
        "formal_rules": {
            "user_journey": {
                "first_time_flow": [
                    {"step": 1, "screen": "Welcome Screen", "purpose": "Greet new user", "type": "splash", "options": []},
                    {"step": 2, "screen": "Skill Assessment", "purpose": "Pick difficulty", "type": "choice", "options": ["Beginner", "Pro"]},
                ],
                "returning_user_flow": [
                    {"screen": "Home", "sections": ["feed", "challenges"]},
                ],
                "first_game_experience": "Tutorial game against easy AI",
            },
            "onboarding": {
                "rules_summary": "Move pieces to checkmate the opponent.",
                "piece_movements": {"Pawn": "Forward one square"},
                "key_concepts": ["check", "checkmate"],
                "ux_aids": ["highlight legal moves"],
                "difficulty_levels": [
                    {"id": "beginner", "label": "Beginner", "icon": "🌱"},
                    {"id": "intermediate", "label": "Intermediate", "icon": "⚡"},
                    {"id": "expert", "label": "Expert", "icon": "🏆"},
                ],
            },
            "game_modes": [
                {"id": "vs_human", "label": "vs Human", "description": "Play a friend online", "free": True},
                {"id": "vs_ai", "label": "vs AI", "description": "Play against computer", "free": True},
                {"id": "tournament", "label": "Tournament", "description": "Compete in brackets", "free": False},
            ],
            "gameplay_controls": {
                "during_game": [
                    {"id": "undo", "label": "Undo", "icon": "↩", "position": "top", "behavior": "Take back last move"},
                    {"id": "resign", "label": "Resign", "icon": "🏳", "position": "top", "behavior": "Forfeit the game"},
                ],
                "move_notation": {"format": "algebraic"},
                "time_controls": [
                    {"id": "bullet", "label": "Bullet 1+0", "seconds": 60},
                    {"id": "blitz", "label": "Blitz 5+0", "seconds": 300},
                    {"id": "rapid", "label": "Rapid 10+0", "seconds": 600},
                ],
                "side_selection": ["white", "black", "random"],
                "board_flip": True,
            },
            "engagement": {
                "streak": True,
                "daily_puzzle": True,
                "progression": "ELO rating system with badges",
                "social": ["Friends List", "Chat", "Leaderboard"],
            },
            "benchmark_gaps": [
                "Opening explorer with tree view",
                "Endgame tablebase lookup",
            ],
        },
        "benchmark_reference": {
            "name": "Chess.com",
        },
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestExtractMoonFeatures:
    def test_extract_moon_features(self, full_output):
        features = extract_features(full_output)
        moon_ids = [f["id"] for f in features if f["source"] == "moon_features"]
        assert "feat_puzzle" in moon_ids
        assert "feat_analysis" in moon_ids
        # empty id → slug from name
        assert "feat_openings_explorer" in moon_ids

    def test_moon_feature_name_preserved(self, full_output):
        features = extract_features(full_output)
        by_id = {f["id"]: f for f in features}
        assert by_id["feat_puzzle"]["name"] == "Daily Puzzle"
        assert by_id["feat_puzzle"]["source"] == "moon_features"


class TestExtractUserJourneySteps:
    def test_extract_user_journey_steps(self, full_output):
        features = extract_features(full_output)
        journey_ids = [f["id"] for f in features if f["source"] == "user_journey"]
        assert "journey_step_1_welcome_screen" in journey_ids
        assert "journey_step_2_skill_assessment" in journey_ids

    def test_journey_step_category(self, full_output):
        features = extract_features(full_output)
        journey_feats = [f for f in features if f["source"] == "user_journey"]
        for feat in journey_feats:
            assert feat["category"] == "onboarding"

    def test_journey_step_description_is_purpose(self, full_output):
        features = extract_features(full_output)
        by_id = {f["id"]: f for f in features}
        assert by_id["journey_step_1_welcome_screen"]["description"] == "Greet new user"


class TestExtractDifficultyLevels:
    def test_extract_difficulty_levels(self, full_output):
        features = extract_features(full_output)
        by_id = {f["id"]: f for f in features}
        assert "onboard_skill_level" in by_id

    def test_difficulty_levels_single_feature(self, full_output):
        features = extract_features(full_output)
        skill_feats = [f for f in features if f["id"] == "onboard_skill_level"]
        assert len(skill_feats) == 1

    def test_difficulty_levels_description_contains_labels(self, full_output):
        features = extract_features(full_output)
        by_id = {f["id"]: f for f in features}
        desc = by_id["onboard_skill_level"]["description"]
        assert "Beginner" in desc
        assert "Expert" in desc

    def test_no_difficulty_levels_no_feature(self):
        output = {"formal_rules": {"onboarding": {"difficulty_levels": []}}}
        features = extract_features(output)
        ids = [f["id"] for f in features]
        assert "onboard_skill_level" not in ids


class TestExtractGameModes:
    def test_extract_game_modes(self, full_output):
        features = extract_features(full_output)
        by_id = {f["id"]: f for f in features}
        assert "mode_vs_human" in by_id
        assert "mode_vs_ai" in by_id
        assert "mode_tournament" in by_id

    def test_game_mode_category(self, full_output):
        features = extract_features(full_output)
        mode_feats = [f for f in features if f["source"] == "game_modes"]
        for feat in mode_feats:
            assert feat["category"] == "game_mode"

    def test_premium_mode_description(self, full_output):
        features = extract_features(full_output)
        by_id = {f["id"]: f for f in features}
        assert "premium" in by_id["mode_tournament"]["description"]


class TestExtractGameplayControls:
    def test_extract_gameplay_controls(self, full_output):
        features = extract_features(full_output)
        by_id = {f["id"]: f for f in features}
        assert "ctrl_undo" in by_id
        assert "ctrl_resign" in by_id

    def test_gameplay_control_category(self, full_output):
        features = extract_features(full_output)
        ctrl_feats = [f for f in features if f["id"].startswith("ctrl_") and f["source"] == "gameplay_controls"]
        for feat in ctrl_feats:
            assert feat["category"] == "gameplay"

    def test_time_controls_collapsed(self, full_output):
        features = extract_features(full_output)
        by_id = {f["id"]: f for f in features}
        assert "ctrl_time_controls" in by_id
        desc = by_id["ctrl_time_controls"]["description"]
        assert "Bullet" in desc or "1+0" in desc


class TestExtractEngagementBool:
    def test_streak_true_creates_feature(self, full_output):
        features = extract_features(full_output)
        ids = [f["id"] for f in features]
        assert "engage_streak" in ids

    def test_streak_false_no_feature(self):
        output = {"formal_rules": {"engagement": {"streak": False, "daily_puzzle": False, "social": []}}}
        features = extract_features(output)
        ids = [f["id"] for f in features]
        assert "engage_streak" not in ids

    def test_daily_puzzle_true_creates_feature(self, full_output):
        features = extract_features(full_output)
        ids = [f["id"] for f in features]
        assert "engage_daily_puzzle" in ids

    def test_daily_puzzle_false_no_feature(self):
        output = {"formal_rules": {"engagement": {"streak": False, "daily_puzzle": False, "social": []}}}
        features = extract_features(output)
        ids = [f["id"] for f in features]
        assert "engage_daily_puzzle" not in ids


class TestExtractEngagementSocial:
    def test_extract_engagement_social(self, full_output):
        features = extract_features(full_output)
        social_feats = [f for f in features if f["category"] == "social"]
        social_names = [f["name"] for f in social_feats]
        assert "Friends List" in social_names
        assert "Chat" in social_names
        assert "Leaderboard" in social_names

    def test_social_source(self, full_output):
        features = extract_features(full_output)
        for feat in features:
            if feat["category"] == "social":
                assert feat["source"] == "engagement"

    def test_empty_social_no_features(self):
        output = {"formal_rules": {"engagement": {"streak": False, "daily_puzzle": False, "social": []}}}
        features = extract_features(output)
        assert not any(f["category"] == "social" for f in features)


class TestExtractBenchmarkGaps:
    def test_extract_benchmark_gaps(self, full_output):
        features = extract_features(full_output)
        gap_feats = [f for f in features if f["source"] == "benchmark_gap"]
        gap_names = [f["name"] for f in gap_feats]
        assert "Opening explorer with tree view" in gap_names
        assert "Endgame tablebase lookup" in gap_names

    def test_benchmark_gap_category(self, full_output):
        features = extract_features(full_output)
        for feat in features:
            if feat["source"] == "benchmark_gap":
                assert feat["category"] == "core"

    def test_benchmark_gap_id_format(self, full_output):
        features = extract_features(full_output)
        gap_ids = [f["id"] for f in features if f["source"] == "benchmark_gap"]
        for gid in gap_ids:
            assert gid.startswith("gap_")


class TestDeduplication:
    def test_deduplication(self):
        """Duplicate moon_features id → only first kept."""
        output = {
            "moon_features": [
                {"id": "feat_dup", "name": "Feature A", "category": "core"},
                {"id": "feat_dup", "name": "Feature B", "category": "core"},
            ]
        }
        features = extract_features(output)
        matching = [f for f in features if f["id"] == "feat_dup"]
        assert len(matching) == 1
        assert matching[0]["name"] == "Feature A"

    def test_deduplication_across_sources(self):
        """If moon_features and game_modes produce same id, first source wins."""
        output = {
            "moon_features": [
                {"id": "mode_vs_human", "name": "Moon entry", "category": "core"},
            ],
            "formal_rules": {
                "game_modes": [
                    {"id": "vs_human", "label": "vs Human", "description": "Play friend", "free": True},
                ],
            },
        }
        features = extract_features(output)
        matching = [f for f in features if f["id"] == "mode_vs_human"]
        assert len(matching) == 1
        assert matching[0]["source"] == "moon_features"


class TestEmptyOutput:
    def test_empty_output_returns_empty_list(self):
        assert extract_features({}) == []

    def test_none_output_returns_empty_list(self):
        assert extract_features(None) == []  # type: ignore[arg-type]

    def test_missing_formal_rules(self):
        output = {"moon_features": [{"id": "f1", "name": "Feature One", "category": "core"}]}
        features = extract_features(output)
        assert len(features) == 1

    def test_missing_sub_keys(self):
        output = {"formal_rules": {}}
        features = extract_features(output)
        assert features == []


class TestExtractBenchmarkName:
    def test_extract_benchmark_name(self, full_output):
        assert extract_benchmark_name(full_output) == "Chess.com"

    def test_missing_benchmark_reference(self):
        assert extract_benchmark_name({}) == ""

    def test_missing_name_key(self):
        output = {"benchmark_reference": {}}
        assert extract_benchmark_name(output) == ""

    def test_none_input(self):
        assert extract_benchmark_name(None) == ""  # type: ignore[arg-type]


class TestBuildProductDescription:
    def test_build_product_description_non_empty(self, full_output):
        desc = build_product_description(full_output)
        assert isinstance(desc, str)
        assert len(desc) > 0

    def test_description_contains_game_name(self, full_output):
        desc = build_product_description(full_output)
        assert "Chess Pro" in desc

    def test_description_contains_benchmark(self, full_output):
        desc = build_product_description(full_output)
        assert "Chess.com" in desc

    def test_description_contains_mode_labels(self, full_output):
        desc = build_product_description(full_output)
        assert "vs Human" in desc or "Human" in desc

    def test_empty_output_returns_string(self):
        desc = build_product_description({})
        assert isinstance(desc, str)

    def test_none_returns_empty(self):
        desc = build_product_description(None)  # type: ignore[arg-type]
        assert desc == ""


class TestFullExtractCount:
    def test_full_extract_count_above_10(self, full_output):
        features = extract_features(full_output)
        assert len(features) > 10, f"Expected >10 features, got {len(features)}"

    def test_all_features_have_required_keys(self, full_output):
        features = extract_features(full_output)
        required_keys = {"id", "name", "description", "category", "source"}
        for feat in features:
            missing = required_keys - feat.keys()
            assert not missing, f"Feature {feat.get('id')} missing keys: {missing}"

    def test_no_empty_ids(self, full_output):
        features = extract_features(full_output)
        for feat in features:
            assert feat["id"], f"Feature has empty id: {feat}"
