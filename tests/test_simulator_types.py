

from user_soul.simulator import SimulationRun, FeatureSignal, _derive_kano, _derive_aarrr
from user_soul.scenarios import ScenarioContext


def test_simulation_run_fields():
    run = SimulationRun(
        persona_id="p1",
        context=ScenarioContext("evening", "stressed", 7, "work_stress"),
        features_used=["f1", "f2"],
        features_skipped=["f3"],
    )
    assert run.persona_id == "p1"
    assert "f1" in run.features_used
    assert "f3" in run.features_skipped


def test_feature_signal_fields():
    sig = FeatureSignal(
        feature_id="f1",
        feature_name="daily check-in",
        n_simulations=10,
        usage_rate=0.8,
        exposure_rate=0.9,
        skip_rate=0.1,
        context_map={"stressed": 0.9, "calm": 0.5},
        day_curve={1: 0.4, 7: 0.8, 30: 0.95},
        implied_kano="Must-Have",
        implied_aarrr_score=0.75,
    )
    assert sig.usage_rate == 0.8
    assert sig.implied_kano == "Must-Have"


def test_derive_kano():
    assert _derive_kano(0.90) == "Must-Have"
    assert _derive_kano(0.80) == "Performance"   # boundary — >0.80 strictly, so 0.80 is Performance
    assert _derive_kano(0.79) == "Performance"
    assert _derive_kano(0.50) == "Performance"
    assert _derive_kano(0.49) == "Delighter"
    assert _derive_kano(0.20) == "Delighter"
    assert _derive_kano(0.19) == "Indifferent"
    assert _derive_kano(0.00) == "Indifferent"


def test_derive_aarrr():
    # day1=0.4, day7=0.8, day30=0.95 → 0.3*0.4 + 0.3*0.8 + 0.4*0.95 = 0.12 + 0.24 + 0.38 = 0.74
    score = _derive_aarrr({1: 0.4, 7: 0.8, 30: 0.95})
    assert abs(score - 0.74) < 0.01


def test_derive_aarrr_missing_days():
    # Missing days default to 0.0
    score = _derive_aarrr({1: 1.0})
    assert abs(score - 0.30) < 0.01  # 0.3*1.0 + 0.3*0 + 0.4*0
