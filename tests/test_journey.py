"""Tests for mcv.journey and mcv.gate_ledger."""
from __future__ import annotations
import pytest
from unittest.mock import patch, MagicMock
from user_soul.journey import JourneyReport, simulate_journey, _parse_step_output, _normalise_screens
from user_soul.gate_ledger import GateLedger


# ── JourneyReport properties ───────────────────────────────────────────────

def _make_report(completion_rate: float, drop_offs: dict | None = None) -> JourneyReport:
    return JourneyReport(
        target_flow=["a", "b", "c"],
        completion_rate=completion_rate,
        drop_off_by_screen=drop_offs or {},
        fogg_violations=[],
        blocked_journeys=[],
        personas_completed=int(completion_rate * 12),
        personas_total=12,
    )


def test_passes_gate_above():
    assert _make_report(0.75).passes_gate is True

def test_passes_gate_at_threshold():
    assert _make_report(0.70).passes_gate is True

def test_passes_gate_below():
    assert _make_report(0.60).passes_gate is False

def test_benchmark_strong():
    assert "strong" in _make_report(0.90).benchmark_context.lower()

def test_benchmark_acceptable():
    assert "acceptable" in _make_report(0.72).benchmark_context.lower()

def test_benchmark_weak():
    assert "weak" in _make_report(0.55).benchmark_context.lower()

def test_benchmark_critical():
    assert "critical" in _make_report(0.30).benchmark_context.lower()


# ── _parse_step_output ─────────────────────────────────────────────────────

def test_parse_yes():
    p, r, f = _parse_step_output("proceed: yes\nreason: clear path\nfogg_issue: none")
    assert p is True and f == "none"

def test_parse_no_ability():
    p, r, f = _parse_step_output("proceed: no\nreason: confusing\nfogg_issue: ability")
    assert p is False and f == "ability"

def test_parse_no_motivation():
    p, r, f = _parse_step_output("proceed: no\nreason: boring\nfogg_issue: motivation")
    assert p is False and f == "motivation"

def test_parse_unparseable_defaults_false():
    p, _, _ = _parse_step_output("I don't understand this format")
    assert p is False

def test_parse_chinese_yes():
    p, _, _ = _parse_step_output("proceed: 是\nreason: ok\nfogg_issue: none")
    assert p is True


# ── _normalise_screens ─────────────────────────────────────────────────────

def test_normalise_list_passthrough():
    screens = [{"screen_id": "home", "navigates_to": ["shop"]}]
    assert _normalise_screens(screens) == screens

def test_normalise_dict_to_list():
    screens = {"home": {"screen_id": "home"}, "shop": {"screen_id": "shop"}}
    result = _normalise_screens(screens)
    assert len(result) == 2
    assert all(isinstance(s, dict) for s in result)


# ── simulate_journey (mocked LLM) ─────────────────────────────────────────

def _mock_agent(name: str = "A casual gamer") -> MagicMock:
    a = MagicMock()
    a.to_human_story.return_value = name
    return a


SCREENS = [
    {"screen_id": "home",     "navigates_to": ["treasure"], "description": "Home"},
    {"screen_id": "treasure", "navigates_to": ["collect"],  "description": "Treasure"},
    {"screen_id": "collect",  "navigates_to": [],           "description": "Collect"},
]
FLOW = ["home", "treasure", "collect"]


def test_all_complete():
    pool = [_mock_agent() for _ in range(3)]
    with patch("user_soul.core._llm_call", return_value=("proceed: yes\nreason: ok\nfogg_issue: none", {})):
        r = simulate_journey(SCREENS, FLOW, pool, api_key="key", n_personas=3)
    assert r.completion_rate == 1.0
    assert r.passes_gate is True
    assert r.personas_completed == 3


def test_all_drop_first_screen():
    pool = [_mock_agent() for _ in range(3)]
    with patch("user_soul.core._llm_call", return_value=("proceed: no\nreason: confused\nfogg_issue: ability", {})):
        r = simulate_journey(SCREENS, FLOW, pool, api_key="key", n_personas=3)
    assert r.completion_rate == 0.0
    assert r.drop_off_by_screen.get("home", 0) == 3
    assert "ability" in r.fogg_violations


def test_architecture_block_no_llm_call():
    """If next screen not in navigates_to, block without calling LLM."""
    screens = [
        {"screen_id": "home",    "navigates_to": ["shop"], "description": "Home"},
        {"screen_id": "nowhere", "navigates_to": [],       "description": "Dead end"},
    ]
    pool = [_mock_agent()]
    with patch("user_soul.core._llm_call") as mock_llm:
        r = simulate_journey(screens, ["home", "nowhere"], pool, api_key="key", n_personas=1)
    mock_llm.assert_not_called()
    assert r.completion_rate == 0.0
    assert "ability" in r.fogg_violations


def test_single_screen_trivially_complete():
    pool = [_mock_agent() for _ in range(5)]
    r = simulate_journey(SCREENS, ["home"], pool, api_key="key", n_personas=5)
    assert r.completion_rate == 1.0
    assert r.passes_gate is True


def test_partial_completion():
    pool = [_mock_agent("winner"), _mock_agent("loser"), _mock_agent("winner")]
    responses = [
        ("proceed: yes\nreason: ok\nfogg_issue: none", {}),  # winner step1
        ("proceed: yes\nreason: ok\nfogg_issue: none", {}),  # winner step2
        ("proceed: no\nreason: stuck\nfogg_issue: trigger", {}),  # loser step1
        ("proceed: yes\nreason: ok\nfogg_issue: none", {}),  # winner2 step1
        ("proceed: yes\nreason: ok\nfogg_issue: none", {}),  # winner2 step2
    ]
    with patch("user_soul.core._llm_call", side_effect=responses):
        r = simulate_journey(SCREENS, FLOW, pool, api_key="key", n_personas=3)
    assert r.personas_completed == 2
    assert abs(r.completion_rate - 2/3) < 0.01


def test_dict_screens_format():
    screens_dict = {s["screen_id"]: s for s in SCREENS}
    pool = [_mock_agent()]
    with patch("user_soul.core._llm_call", return_value=("proceed: yes\nreason: ok\nfogg_issue: none", {})):
        r = simulate_journey(screens_dict, FLOW, pool, api_key="key", n_personas=1)
    assert r.completion_rate == 1.0


# ── GateLedger ─────────────────────────────────────────────────────────────

def test_gate_ledger_empty_dict():
    assert GateLedger().to_dict() == {}

def test_gate_ledger_gate2_journey():
    journey = JourneyReport(
        target_flow=["a", "b"], completion_rate=0.80,
        drop_off_by_screen={"a": 2}, fogg_violations=["ability"],
        blocked_journeys=[], personas_completed=10, personas_total=12,
    )
    gl = GateLedger(gate2_journey=journey)
    d = gl.to_dict()
    assert d["gate2_completion_rate"] == 0.80
    assert d["gate2_passes"] is True
    assert d["gate2_fogg_violations"] == ["ability"]

def test_gate_ledger_gate3_frictions():
    gl = GateLedger(gate3_adversarial_frictions=["无法找到按钮", "注册墙阻断"])
    d = gl.to_dict()
    assert len(d["gate3_frictions"]) == 2
