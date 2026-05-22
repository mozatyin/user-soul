import json, tempfile

from pathlib import Path
from user_soul.cache import save_simulation_cache, load_simulation_cache, _feature_hash
from user_soul.simulator import FeatureSignal

FEATURES = [{"id": "f1", "name": "check-in"}, {"id": "f2", "name": "star map"}]

def _make_signal(fid):
    return FeatureSignal(
        feature_id=fid, feature_name=fid, n_simulations=20,
        usage_rate=0.7, exposure_rate=0.8, skip_rate=0.1,
        context_map={"stressed": 0.9}, day_curve={1: 0.4, 30: 0.9},
        implied_kano="Performance", implied_aarrr_score=0.68,
    )

SIGNALS = [_make_signal("f1"), _make_signal("f2")]


def test_save_then_load_returns_same_signals():
    with tempfile.TemporaryDirectory() as tmp:
        state_dir = Path(tmp)
        save_simulation_cache(state_dir, FEATURES, SIGNALS)
        loaded = load_simulation_cache(state_dir, FEATURES)
        assert loaded is not None
        assert len(loaded) == 2
        assert loaded[0].feature_id == "f1"
        assert abs(loaded[0].usage_rate - 0.7) < 0.001


def test_load_returns_none_when_no_cache():
    with tempfile.TemporaryDirectory() as tmp:
        result = load_simulation_cache(Path(tmp), FEATURES)
        assert result is None


def test_load_returns_none_when_feature_hash_mismatch():
    with tempfile.TemporaryDirectory() as tmp:
        state_dir = Path(tmp)
        save_simulation_cache(state_dir, FEATURES, SIGNALS)
        # Add a new feature → hash changes
        new_features = FEATURES + [{"id": "f3", "name": "new feature"}]
        result = load_simulation_cache(state_dir, new_features)
        assert result is None


def test_feature_hash_is_order_independent():
    h1 = _feature_hash([{"id": "f1"}, {"id": "f2"}])
    h2 = _feature_hash([{"id": "f2"}, {"id": "f1"}])
    assert h1 == h2


def test_save_writes_json_files():
    with tempfile.TemporaryDirectory() as tmp:
        state_dir = Path(tmp)
        save_simulation_cache(state_dir, FEATURES, SIGNALS)
        assert (state_dir / "simulation_cache.json").exists()
        assert (state_dir / "simulation_meta.json").exists()
