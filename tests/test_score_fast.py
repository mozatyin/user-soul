

from unittest.mock import patch
from user_soul import Persona, PersonaDecider

PERSONAS = [Persona(id="p1", name="Alice", description="casual gamer",
                    cohort="18-30", motivations=["fun"], pain_points=["ads"])]


@patch("user_soul.core._llm_call")
def test_score_fast_single(mock_llm):
    mock_llm.return_value = ('{"score": 7.5, "reasoning": "interesting but niche"}', 40)
    pd = PersonaDecider(PERSONAS, api_key="test", mode="fast")
    result = pd.score(
        question="AARRR revenue impact score for soul map feature",
        lo=0.0, hi=10.0,
        context="SoulMap app",
    )
    assert abs(result.value - 7.5) < 0.01
    assert result.confidence == 1.0
    assert 0.0 <= result.value <= 10.0
    mock_llm.assert_called_once()


@patch("user_soul.core._llm_call")
def test_score_fast_batch(mock_llm):
    mock_llm.return_value = (
        '[{"id": "f1", "score": 8.0}, {"id": "f2", "score": 3.5}]', 55
    )
    pd = PersonaDecider(PERSONAS, api_key="test", mode="fast")
    results = pd.score(
        question="AARRR revenue impact",
        lo=0.0, hi=10.0,
        context="SoulMap app",
        batch=[{"id": "f1", "name": "TriSoul"}, {"id": "f2", "name": "basic chat"}],
    )
    assert len(results) == 2
    assert abs(results[0].value - 8.0) < 0.01
    assert abs(results[1].value - 3.5) < 0.01
    mock_llm.assert_called_once()


@patch("user_soul.core._llm_call")
def test_score_clamps_to_range(mock_llm):
    mock_llm.return_value = ('{"score": 15.0}', 30)
    pd = PersonaDecider(PERSONAS, api_key="test", mode="fast")
    result = pd.score("Q", lo=0.0, hi=10.0, context="ctx")
    assert result.value <= 10.0


@patch("user_soul.core._llm_call")
def test_score_clamps_below_lo(mock_llm):
    mock_llm.return_value = ('{"score": -5.0}', 30)
    pd = PersonaDecider(PERSONAS, api_key="test", mode="fast")
    result = pd.score("Q", lo=0.0, hi=10.0, context="ctx")
    assert result.value >= 0.0


@patch("user_soul.core._llm_call")
def test_score_fallback_midpoint_on_missing(mock_llm):
    """If LLM omits an id from batch response, fallback to midpoint."""
    mock_llm.return_value = ('[{"id": "f1", "score": 9.0}]', 40)
    pd = PersonaDecider(PERSONAS, api_key="test", mode="fast")
    results = pd.score(
        "Q", lo=0.0, hi=10.0, context="ctx",
        batch=[{"id": "f1"}, {"id": "f2"}],
    )
    assert abs(results[0].value - 9.0) < 0.01
    assert abs(results[1].value - 5.0) < 0.01  # midpoint fallback


@patch("user_soul.core._llm_call")
def test_score_non_numeric_falls_back_to_midpoint(mock_llm):
    """Non-numeric score from LLM should not crash — falls back to midpoint."""
    mock_llm.return_value = ('{"score": "high", "reasoning": "vague"}', 30)
    pd = PersonaDecider(PERSONAS, api_key="test", mode="fast")
    result = pd.score("Q", lo=0.0, hi=10.0, context="ctx")
    assert result.value == 5.0  # midpoint fallback
