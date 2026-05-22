

from user_soul.scenarios import ScenarioContext, random_context, ROLE_DAY_RANGES


def test_scenario_context_fields():
    ctx = ScenarioContext(
        time_of_day="evening",
        emotional_state="stressed",
        usage_day=7,
        trigger="work_stress",
    )
    assert ctx.time_of_day == "evening"
    assert ctx.emotional_state == "stressed"
    assert ctx.usage_day == 7
    assert ctx.trigger == "work_stress"


def test_random_context_returns_valid_context():
    ctx = random_context()
    assert isinstance(ctx, ScenarioContext)
    assert ctx.time_of_day in ("morning_commute", "lunch_break", "evening_wind_down", "night")
    assert ctx.emotional_state in ("stressed", "calm", "bored", "excited", "sad", "anxious")
    assert ctx.usage_day in (1, 3, 7, 14, 30)
    assert ctx.trigger in ("habit", "work_stress", "relationship_tension",
                           "boredom", "notification", "curiosity")


def test_random_context_for_role():
    ctx = random_context(role="Explorer")
    assert ctx.usage_day in ROLE_DAY_RANGES["Explorer"]

    ctx = random_context(role="Habituer")
    assert ctx.usage_day in ROLE_DAY_RANGES["Habituer"]


def test_random_context_produces_variance():
    """Running random_context 20 times should produce at least 3 distinct contexts."""
    contexts = [random_context() for _ in range(20)]
    unique = {(c.time_of_day, c.emotional_state, c.usage_day, c.trigger) for c in contexts}
    assert len(unique) >= 3, "random_context is not producing variance"
