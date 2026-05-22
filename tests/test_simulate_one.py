

from unittest.mock import patch
from user_soul.simulator import PersonaSimulator, SimulationRun
from user_soul.scenarios import ScenarioContext

PERSONAS = [
    {"id": "p1", "name": "Alice", "description": "25yo professional",
     "cohort": "25-35 women", "motivations": ["growth"], "pain_points": ["complex UI"],
     "role": "Habituer"},
]
FEATURES = [
    {"id": "f1", "name": "daily soul check-in"},
    {"id": "f2", "name": "star map"},
    {"id": "f3", "name": "crisis journal"},
]

MOCK_RESPONSE = """
Alice opened the app and saw the daily check-in prompt. She tapped it.
She answered the mood question and felt a moment of calm.
She scrolled past the star map and paused, then tapped it briefly.
She noticed the crisis tracker but did not tap it — looked too heavy for tonight.
She closed the app after 5 minutes.

USED: f1, f2
SEEN: f3
"""


@patch("user_soul.core._llm_call")
def test_simulate_one_returns_simulation_run(mock_llm):
    mock_llm.return_value = (MOCK_RESPONSE, 400)
    sim = PersonaSimulator(PERSONAS, api_key="test")
    ctx = ScenarioContext("evening", "calm", 14, "habit")
    run = sim._simulate_one(PERSONAS[0], FEATURES, ctx)
    assert isinstance(run, SimulationRun)
    assert run.persona_id == "p1"
    assert "f1" in run.features_used
    assert "f2" in run.features_used
    assert "f3" in run.features_skipped
    assert "f3" not in run.features_used


@patch("user_soul.core._llm_call")
def test_simulate_one_uses_temperature_1(mock_llm):
    mock_llm.return_value = (MOCK_RESPONSE, 400)
    sim = PersonaSimulator(PERSONAS, api_key="test")
    ctx = ScenarioContext("evening", "calm", 14, "habit")
    sim._simulate_one(PERSONAS[0], FEATURES, ctx)
    call_kwargs = mock_llm.call_args
    assert call_kwargs[1].get("temperature") == 1.0 or \
           (len(call_kwargs[0]) > 3 and call_kwargs[0][3] == 1.0), \
           "temperature=1.0 must be passed to _llm_call"


@patch("user_soul.core._llm_call")
def test_simulate_one_uses_haiku(mock_llm):
    mock_llm.return_value = (MOCK_RESPONSE, 400)
    sim = PersonaSimulator(PERSONAS, api_key="test")
    ctx = ScenarioContext("evening", "calm", 14, "habit")
    sim._simulate_one(PERSONAS[0], FEATURES, ctx)
    call_kwargs = mock_llm.call_args
    model_arg = call_kwargs[1].get("model") or (call_kwargs[0][4] if len(call_kwargs[0]) > 4 else None)
    assert model_arg is not None and "haiku" in model_arg.lower(), \
           f"Expected haiku model, got: {model_arg}"


@patch("user_soul.core._llm_call")
def test_simulate_one_prompt_contains_all_feature_ids(mock_llm):
    mock_llm.return_value = (MOCK_RESPONSE, 400)
    sim = PersonaSimulator(PERSONAS, api_key="test")
    ctx = ScenarioContext("evening", "calm", 14, "habit")
    sim._simulate_one(PERSONAS[0], FEATURES, ctx)
    prompt = mock_llm.call_args[0][0]
    assert "f1" in prompt
    assert "f2" in prompt
    assert "f3" in prompt
    # Must NOT ask for opinions
    assert "important" not in prompt.lower()
    assert "rate" not in prompt.lower()
    assert "score" not in prompt.lower()


@patch("user_soul.core._llm_call")
def test_simulate_one_handles_malformed_used_line(mock_llm):
    """If USED: line is missing, returns empty features_used."""
    mock_llm.return_value = ("Alice just sat there. She did nothing.\n", 100)
    sim = PersonaSimulator(PERSONAS, api_key="test")
    ctx = ScenarioContext("evening", "calm", 14, "habit")
    run = sim._simulate_one(PERSONAS[0], FEATURES, ctx)
    assert isinstance(run, SimulationRun)
    assert run.features_used == []
    assert run.features_skipped == []
