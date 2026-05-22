

from unittest.mock import patch, MagicMock
from user_soul.simulator import PersonaSimulator, FeatureSignal, SimulationRun
from user_soul.scenarios import ScenarioContext

PERSONAS = [
    {"id": "p1", "name": "Alice", "description": "Habituer",
     "cohort": "25-35", "motivations": ["growth"], "pain_points": [],
     "role": "Habituer"},
    {"id": "p2", "name": "Bob", "description": "Explorer",
     "cohort": "18-25", "motivations": ["fun"], "pain_points": [],
     "role": "Explorer"},
]
FEATURES = [
    {"id": "f1", "name": "daily check-in"},
    {"id": "f2", "name": "star map"},
    {"id": "f3", "name": "crisis tracker"},
]


def _make_run(persona_id, used, skipped, usage_day, emotional):
    return SimulationRun(
        persona_id=persona_id,
        context=ScenarioContext("evening", emotional, usage_day, "habit"),
        features_used=used,
        features_skipped=skipped,
    )


def test_simulate_returns_feature_signal_per_feature():
    sim = PersonaSimulator(PERSONAS, api_key="test")
    # Mock _simulate_one to return fixed runs
    sim._simulate_one = MagicMock(side_effect=[
        # p1, run 1
        _make_run("p1", ["f1", "f2"], ["f3"], 14, "stressed"),
        # p1, run 2
        _make_run("p1", ["f1"], ["f2", "f3"], 14, "calm"),
        # p2, run 1
        _make_run("p2", ["f1", "f2", "f3"], [], 1, "excited"),
        # p2, run 2
        _make_run("p2", ["f1"], ["f2"], 1, "bored"),
    ])
    signals = sim.simulate(FEATURES, n_runs=2)
    assert len(signals) == 3  # one FeatureSignal per feature
    sig_f1 = next(s for s in signals if s.feature_id == "f1")
    assert sig_f1.n_simulations == 4  # 2 personas × 2 runs
    assert sig_f1.usage_rate == 1.0   # f1 used in all 4 runs


def test_simulate_usage_rate_calculation():
    sim = PersonaSimulator(PERSONAS[:1], api_key="test")
    sim._simulate_one = MagicMock(side_effect=[
        _make_run("p1", ["f1"], [], 14, "stressed"),
        _make_run("p1", ["f1"], ["f2"], 14, "calm"),
        _make_run("p1", [], ["f1"], 7, "bored"),
        _make_run("p1", ["f1", "f2"], [], 7, "excited"),
    ])
    signals = sim.simulate(FEATURES, n_runs=4)
    sig_f1 = next(s for s in signals if s.feature_id == "f1")
    # f1 used in 3/4 runs = 0.75
    assert abs(sig_f1.usage_rate - 0.75) < 0.01
    sig_f2 = next(s for s in signals if s.feature_id == "f2")
    # f2 used in 1/4, seen in 2/4
    assert abs(sig_f2.usage_rate - 0.25) < 0.01
    assert abs(sig_f2.exposure_rate - 0.50) < 0.01


def test_simulate_implied_kano_derived_from_usage_rate():
    sim = PersonaSimulator(PERSONAS[:1], api_key="test")
    # 9/10 runs use f1 → Must-Have
    # 5/10 runs use f2 → Performance (boundary)
    # 1/10 runs use f3 → Indifferent
    runs = (
        [_make_run("p1", ["f1", "f2"], [], 14, "calm")] * 5 +
        [_make_run("p1", ["f1"], ["f2", "f3"], 14, "stressed")] * 4 +
        [_make_run("p1", ["f1", "f3"], [], 14, "bored")] * 1
    )
    sim._simulate_one = MagicMock(side_effect=runs)
    signals = sim.simulate(FEATURES, n_runs=10)
    sig_map = {s.feature_id: s for s in signals}
    assert sig_map["f1"].implied_kano == "Must-Have"
    assert sig_map["f2"].implied_kano in ("Performance", "Delighter")
    assert sig_map["f3"].implied_kano == "Indifferent"


def test_simulate_calls_simulate_one_n_runs_per_persona():
    """Total calls = n_personas × n_runs."""
    sim = PersonaSimulator(PERSONAS, api_key="test")
    sim._simulate_one = MagicMock(return_value=_make_run("p1", ["f1"], [], 7, "calm"))
    sim.simulate(FEATURES, n_runs=3)
    assert sim._simulate_one.call_count == 6  # 2 personas × 3 runs


def test_simulate_context_map_breakdown():
    """context_map shows usage_rate per emotional_state."""
    sim = PersonaSimulator(PERSONAS[:1], api_key="test")
    sim._simulate_one = MagicMock(side_effect=[
        _make_run("p1", ["f1"], [], 14, "stressed"),
        _make_run("p1", ["f1"], [], 14, "stressed"),
        _make_run("p1", [], ["f1"], 14, "calm"),
        _make_run("p1", [], [], 14, "calm"),
    ])
    signals = sim.simulate(FEATURES, n_runs=4)
    sig_f1 = next(s for s in signals if s.feature_id == "f1")
    # stressed: 2/2 = 1.0, calm: 0/2 = 0.0
    assert abs(sig_f1.context_map.get("stressed", 0) - 1.0) < 0.01
    assert abs(sig_f1.context_map.get("calm", 0) - 0.0) < 0.01
