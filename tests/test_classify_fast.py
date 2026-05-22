

from unittest.mock import patch
from user_soul import Persona, PersonaDecider

PERSONAS = [Persona(id="p1", name="Alice", description="casual gamer",
                    cohort="18-30", motivations=["fun"], pain_points=["ads"])]


@patch("user_soul.core._llm_call")
def test_classify_fast_single(mock_llm):
    mock_llm.return_value = ('{"choice": "Must-Have", "reasoning": "core to the experience"}', 40)
    pd = PersonaDecider(PERSONAS, api_key="test", mode="fast")
    result = pd.classify(
        question="Kano classification for daily soul check-in",
        options=["Must-Have", "Performance", "Delighter", "Indifferent"],
        context="SoulMap self-reflection app",
    )
    assert result.value == "Must-Have"
    assert result.confidence == 1.0
    assert result.mode == "fast"
    mock_llm.assert_called_once()


@patch("user_soul.core._llm_call")
def test_classify_fast_batch(mock_llm):
    mock_llm.return_value = (
        '[{"id": "f1", "choice": "Must-Have"}, {"id": "f2", "choice": "Delighter"}]', 60
    )
    pd = PersonaDecider(PERSONAS, api_key="test", mode="fast")
    results = pd.classify(
        question="Kano classification",
        options=["Must-Have", "Performance", "Delighter", "Indifferent"],
        context="SoulMap app",
        batch=[{"id": "f1", "name": "soul check-in"}, {"id": "f2", "name": "star map"}],
    )
    assert isinstance(results, list)
    assert len(results) == 2
    assert results[0].value == "Must-Have"
    assert results[1].value == "Delighter"
    mock_llm.assert_called_once()  # batch = 1 call regardless of item count


@patch("user_soul.core._llm_call")
def test_classify_fast_batch_missing_id_uses_fallback(mock_llm):
    """If LLM omits an id from the response, fallback to first option."""
    mock_llm.return_value = ('[{"id": "f1", "choice": "Must-Have"}]', 40)
    pd = PersonaDecider(PERSONAS, api_key="test", mode="fast")
    results = pd.classify(
        question="Kano?",
        options=["Must-Have", "Indifferent"],
        context="ctx",
        batch=[{"id": "f1", "name": "feature 1"}, {"id": "f2", "name": "feature 2"}],
    )
    assert results[0].value == "Must-Have"
    assert results[1].value == "Must-Have"  # fallback to options[0]
