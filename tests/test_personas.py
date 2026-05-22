import json
import tempfile

from pathlib import Path
from unittest.mock import patch
from user_soul import Persona, load_or_generate

MOCK_RESPONSE = json.dumps([
    {"id": "p1", "name": "Mei", "cohort": "Chinese women 25-35",
     "description": "Career professional seeking self-understanding",
     "motivations": ["self-growth", "relationships"], "pain_points": ["time pressure"]},
    {"id": "p2", "name": "Alex", "cohort": "US college students 18-22",
     "description": "Student curious about personality",
     "motivations": ["fun", "social validation"], "pain_points": ["boring UX"]},
])


@patch("user_soul.personas._llm_call")
def test_generates_personas(mock_llm):
    mock_llm.return_value = (MOCK_RESPONSE, 200)
    with tempfile.TemporaryDirectory() as tmp:
        state_dir = Path(tmp)
        personas = load_or_generate(
            state_dir=state_dir,
            prd_text="SoulMap is a soul mapping app for self-discovery...",
            archetype="Replika",
            target_market="Young adults seeking self-reflection",
            api_key="test",
            n=2,
        )
    assert len(personas) == 2
    assert isinstance(personas[0], Persona)
    assert personas[0].id == "p1"
    assert "self-growth" in personas[0].motivations


@patch("user_soul.personas._llm_call")
def test_caches_to_disk(mock_llm):
    mock_llm.return_value = (MOCK_RESPONSE, 200)
    with tempfile.TemporaryDirectory() as tmp:
        state_dir = Path(tmp)
        load_or_generate(state_dir, "PRD text", "Replika", "young adults", "test", n=2)
        assert (state_dir / "personas.json").exists()
        data = json.loads((state_dir / "personas.json").read_text())
        assert len(data) == 2
        assert data[0]["name"] == "Mei"


@patch("user_soul.personas._llm_call")
def test_loads_from_cache(mock_llm):
    mock_llm.return_value = (MOCK_RESPONSE, 200)
    with tempfile.TemporaryDirectory() as tmp:
        state_dir = Path(tmp)
        # First call generates and caches
        load_or_generate(state_dir, "PRD", "Replika", "young adults", "test", n=2)
        first_call_count = mock_llm.call_count
        # Second call should use cache, no new LLM call
        personas = load_or_generate(state_dir, "PRD", "Replika", "young adults", "test", n=2)
        assert mock_llm.call_count == first_call_count
        assert len(personas) == 2


@patch("user_soul.personas._llm_call")
def test_regenerates_when_cache_has_fewer_than_n(mock_llm):
    """If cached file has fewer personas than requested, regenerate."""
    mock_llm.return_value = (MOCK_RESPONSE, 200)
    with tempfile.TemporaryDirectory() as tmp:
        state_dir = Path(tmp)
        # Write cache with only 1 persona
        (state_dir / "personas.json").write_text(
            json.dumps([{"id": "p1", "name": "Mei", "cohort": "25-35",
                         "description": "test", "motivations": [], "pain_points": []}])
        )
        # Request 2 — should regenerate
        load_or_generate(state_dir, "PRD", "Replika", "young adults", "test", n=2)
        assert mock_llm.call_count == 1  # LLM was called to regenerate
