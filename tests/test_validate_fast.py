

from unittest.mock import patch
from user_soul import Persona, PersonaDecider, DecisionResult

PERSONAS = [Persona(id="p1", name="Alice", description="A 25yo casual mobile gamer",
                    cohort="18-30 casual gamers", motivations=["fun", "social"],
                    pain_points=["complex UI", "pay-to-win"])]


@patch("user_soul.core._llm_call")
def test_validate_fast_true(mock_llm):
    mock_llm.return_value = ('{"result": true, "reasoning": "fits well"}', 50)
    pd = PersonaDecider(PERSONAS, api_key="test", mode="fast")
    result = pd.validate("Does 星图 soul map add value?", context="SoulMap app for self-reflection")
    assert result.value is True
    assert result.confidence == 1.0
    assert result.mode == "fast"
    assert result.tokens_used == 50
    mock_llm.assert_called_once()


@patch("user_soul.core._llm_call")
def test_validate_fast_false(mock_llm):
    mock_llm.return_value = ('{"result": false, "reasoning": "irrelevant"}', 30)
    pd = PersonaDecider(PERSONAS, api_key="test", mode="fast")
    result = pd.validate("Does boring feature add value?", context="context")
    assert result.value is False
    assert result.confidence == 1.0
    assert result.distribution["false"] == 1.0
    assert result.distribution["true"] == 0.0


@patch("user_soul.core._llm_call")
def test_validate_fast_uses_first_persona(mock_llm):
    mock_llm.return_value = ('{"result": true}', 20)
    personas = [
        Persona(id="p1", name="Alice", description="", cohort="18-30", motivations=[], pain_points=[]),
        Persona(id="p2", name="Bob", description="", cohort="30-40", motivations=[], pain_points=[]),
    ]
    pd = PersonaDecider(personas, api_key="test", mode="fast")
    pd.validate("assertion", "context")
    # fast mode uses first persona only → exactly 1 call
    assert mock_llm.call_count == 1
    call_prompt = mock_llm.call_args[0][0]
    assert "Alice" in call_prompt
    assert "Bob" not in call_prompt
