

from unittest.mock import patch, call
from user_soul import Persona, PersonaDecider


def _make_personas(n):
    return [Persona(id=f"p{i}", name=f"User{i}", description=f"persona {i}",
                    cohort="18-30", motivations=["fun"], pain_points=["ads"])
            for i in range(n)]


@patch("user_soul.core._llm_call")
def test_validated_classify_calls_each_persona(mock_llm):
    """validated mode calls _llm_call once per persona."""
    mock_llm.return_value = ('{"choice": "Must-Have", "reasoning": "great"}', 40)
    pd = PersonaDecider(_make_personas(3), api_key="test", mode="validated")
    result = pd.classify("Kano?", ["Must-Have", "Delighter"], "SoulMap")
    assert mock_llm.call_count == 3


@patch("user_soul.core._llm_call")
def test_validated_classify_aggregates_votes(mock_llm):
    """confidence = fraction of personas that voted for the majority."""
    responses = [
        ('{"choice": "Must-Have"}', 40),
        ('{"choice": "Must-Have"}', 40),
        ('{"choice": "Delighter"}', 40),
    ]
    mock_llm.side_effect = responses
    pd = PersonaDecider(_make_personas(3), api_key="test", mode="validated")
    result = pd.classify("Kano?", ["Must-Have", "Delighter"], "SoulMap")
    assert result.value == "Must-Have"
    assert abs(result.confidence - 2/3) < 0.01
    assert abs(result.distribution["Must-Have"] - 2/3) < 0.01
    assert abs(result.distribution["Delighter"] - 1/3) < 0.01


@patch("user_soul.core._llm_call")
def test_validated_score_aggregates_avg(mock_llm):
    """value = mean of per-persona scores."""
    responses = [('{"score": 6.0}', 30), ('{"score": 8.0}', 30), ('{"score": 7.0}', 30)]
    mock_llm.side_effect = responses
    pd = PersonaDecider(_make_personas(3), api_key="test", mode="validated")
    result = pd.score("AARRR?", lo=0.0, hi=10.0, context="SoulMap")
    assert abs(result.value - 7.0) < 0.01  # mean of 6, 8, 7


@patch("user_soul.core._llm_call")
def test_validated_batch_one_call_per_persona(mock_llm):
    """batch of 5 items + 3 personas = exactly 3 LLM calls (not 15)."""
    mock_llm.return_value = (
        '[{"id":"f1","choice":"Must-Have"},{"id":"f2","choice":"Delighter"},'
        '{"id":"f3","choice":"Must-Have"},{"id":"f4","choice":"Indifferent"},'
        '{"id":"f5","choice":"Must-Have"}]', 80
    )
    pd = PersonaDecider(_make_personas(3), api_key="test", mode="validated")
    batch = [{"id": f"f{i}", "name": f"feature {i}"} for i in range(1, 6)]
    results = pd.classify("Kano?", ["Must-Have", "Delighter", "Indifferent"], "SoulMap", batch=batch)
    assert mock_llm.call_count == 3   # NOT 15
    assert len(results) == 5


@patch("user_soul.core._llm_call")
def test_validated_validate(mock_llm):
    responses = [('{"result": true}', 30), ('{"result": false}', 30), ('{"result": true}', 30)]
    mock_llm.side_effect = responses
    pd = PersonaDecider(_make_personas(3), api_key="test", mode="validated")
    result = pd.validate("Does TriSoul add value?", "SoulMap app")
    assert result.value is True   # majority
    assert abs(result.confidence - 2/3) < 0.01


@patch("user_soul.core._llm_call")
def test_validated_score_confidence_high_when_agreement(mock_llm):
    """High agreement across personas → confidence close to 1.0."""
    responses = [('{"score": 9.0}', 30), ('{"score": 9.1}', 30), ('{"score": 8.9}', 30)]
    mock_llm.side_effect = responses
    pd = PersonaDecider(_make_personas(3), api_key="test", mode="validated")
    result = pd.score("Q", lo=0.0, hi=10.0, context="ctx")
    assert result.confidence > 0.9  # very tight agreement


@patch("user_soul.core._llm_call")
def test_validated_tokens_summed(mock_llm):
    """tokens_used should sum across all persona calls."""
    mock_llm.return_value = ('{"result": true}', 50)
    pd = PersonaDecider(_make_personas(4), api_key="test", mode="validated")
    result = pd.validate("assertion", "context")
    assert result.tokens_used == 200  # 4 personas × 50 tokens
