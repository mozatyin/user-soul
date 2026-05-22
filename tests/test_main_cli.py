import json, tempfile

from pathlib import Path
from unittest.mock import patch, MagicMock
from user_soul.simulator import FeatureSignal


def _make_signal(fid):
    return FeatureSignal(
        feature_id=fid, feature_name=fid, n_simulations=5,
        usage_rate=0.6, exposure_rate=0.7, skip_rate=0.1,
        context_map={}, day_curve={},
        implied_kano="Performance", implied_aarrr_score=0.5,
    )


def test_trigger_background_simulation_spawns_subprocess():
    from user_soul.__main__ import trigger_background_simulation
    with patch("subprocess.Popen") as mock_popen:
        trigger_background_simulation(
            state_dir=Path("/tmp/test_state"),
            n=100,
        )
        mock_popen.assert_called_once()
        args = mock_popen.call_args[0][0]
        assert "mcv" in " ".join(args)
        assert "--state" in args
        assert "--n" in args


def test_run_simulation_writes_cache():
    """run_simulation() executes simulation and writes cache."""
    from user_soul.__main__ import run_simulation
    with tempfile.TemporaryDirectory() as tmp:
        state_dir = Path(tmp)
        # Write required input files
        features = [{"id": "f1", "name": "check-in"}, {"id": "f2", "name": "star map"}]
        personas = [{"id": "p1", "name": "Alice", "description": "test",
                     "cohort": "25-35", "motivations": [], "pain_points": [],
                     "role": "Habituer"}]
        (state_dir / "features_for_simulation.json").write_text(json.dumps(features))
        (state_dir / "personas.json").write_text(json.dumps(personas))

        with patch("user_soul.simulator.PersonaSimulator._simulate_one") as mock_sim:
            from user_soul.scenarios import ScenarioContext
            from user_soul.simulator import SimulationRun
            mock_sim.return_value = SimulationRun(
                persona_id="p1",
                context=ScenarioContext("evening", "calm", 14, "habit"),
                features_used=["f1"],
                features_skipped=["f2"],
            )
            run_simulation(state_dir=state_dir, n=3, api_key="test")

        assert (state_dir / "simulation_cache.json").exists()
        assert (state_dir / "simulation_meta.json").exists()
