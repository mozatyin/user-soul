"""Tests for DataSource implementations."""
from __future__ import annotations

from unittest.mock import MagicMock, patch
import pytest

from user_soul.models import FeatureAAR
from user_soul.data_source import SimulatedDataSource, RealDataSource, HybridDataSource, create_data_source


def _make_aar(fid: str = "f1", **kwargs) -> FeatureAAR:
    defaults = dict(
        feature_id=fid,
        acquisition=0.5, activation=0.5, retention=0.5,
        revenue=0.2, referral=0.2,
        confidence=0.5, archetype_votes={},
    )
    defaults.update(kwargs)
    return FeatureAAR(**defaults)


FEATURES = [{"id": "f1", "name": "Feature 1"}]
ARCHETYPES = []


def test_simulated_source_mode():
    vote_engine = MagicMock()
    vote_engine.aarrr.return_value = [_make_aar("f1")]
    src = SimulatedDataSource(vote_engine)
    assert src.mode == "ai"


def test_real_source_mode():
    logger = MagicMock()
    src = RealDataSource(logger)
    assert src.mode == "real"


def test_hybrid_source_mode():
    sim = SimulatedDataSource(MagicMock())
    real = RealDataSource(MagicMock())
    hybrid = HybridDataSource(sim, real)
    assert hybrid.mode == "hybrid"


def test_simulated_delegates_to_vote_engine():
    vote_engine = MagicMock()
    expected = [_make_aar("f1")]
    vote_engine.aarrr.return_value = expected
    src = SimulatedDataSource(vote_engine)
    result = src.get_aarrr_batch(FEATURES, "product", "segment", ARCHETYPES)
    vote_engine.aarrr.assert_called_once_with("product", FEATURES, ARCHETYPES)
    assert result == expected


def test_real_delegates_to_event_logger():
    logger = MagicMock()
    aar = _make_aar("f1", confidence=0.8)
    logger.compute_aarrr.return_value = aar
    src = RealDataSource(logger)
    result = src.get_aarrr_batch(FEATURES, "product", "segment", ARCHETYPES)
    logger.compute_aarrr.assert_called_once_with("f1", None)
    assert result[0] == aar


def test_hybrid_blend_zero_real_confidence():
    sim_aar = _make_aar("f1", acquisition=0.8, confidence=0.9)
    real_aar = _make_aar("f1", acquisition=0.2, confidence=0.0)

    vote_engine = MagicMock()
    vote_engine.aarrr.return_value = [sim_aar]
    logger = MagicMock()
    logger.compute_aarrr.return_value = real_aar

    sim = SimulatedDataSource(vote_engine)
    real = RealDataSource(logger)
    hybrid = HybridDataSource(sim, real, prior_strength=20.0)
    result = hybrid.get_aarrr_batch(FEATURES, "p", "s", ARCHETYPES)

    # n_real = 0.0 * 100 = 0 → w_real = 0/(20+0) = 0 → pure simulation
    assert result[0].acquisition == pytest.approx(sim_aar.acquisition, abs=1e-4)


def test_hybrid_blend_full_real_confidence():
    sim_aar = _make_aar("f1", acquisition=0.1, confidence=0.9)
    real_aar = _make_aar("f1", acquisition=0.9, confidence=1.0)

    vote_engine = MagicMock()
    vote_engine.aarrr.return_value = [sim_aar]
    logger = MagicMock()
    logger.compute_aarrr.return_value = real_aar

    sim = SimulatedDataSource(vote_engine)
    real = RealDataSource(logger)
    hybrid = HybridDataSource(sim, real, prior_strength=20.0)
    result = hybrid.get_aarrr_batch(FEATURES, "p", "s", ARCHETYPES)

    # n_real = 100, w_real = 100/120 ≈ 0.833 → heavily weighted toward real
    w_real = 100 / 120
    expected = w_real * 0.9 + (1 - w_real) * 0.1
    assert result[0].acquisition == pytest.approx(expected, abs=0.01)


def test_hybrid_blend_midpoint():
    sim_aar = _make_aar("f1", acquisition=0.0, confidence=0.8)
    real_aar = _make_aar("f1", acquisition=1.0, confidence=0.5)

    vote_engine = MagicMock()
    vote_engine.aarrr.return_value = [sim_aar]
    logger = MagicMock()
    logger.compute_aarrr.return_value = real_aar

    sim = SimulatedDataSource(vote_engine)
    real = RealDataSource(logger)
    hybrid = HybridDataSource(sim, real, prior_strength=20.0)
    result = hybrid.get_aarrr_batch(FEATURES, "p", "s", ARCHETYPES)

    # n_real = 0.5 * 100 = 50, w_real = 50/70 ≈ 0.714
    n_real = 50
    w_real = n_real / (20 + n_real)
    expected = w_real * 1.0 + (1 - w_real) * 0.0
    assert result[0].acquisition == pytest.approx(expected, abs=0.01)


def test_hybrid_clamps_values():
    sim_aar = _make_aar("f1", acquisition=0.9, retention=0.95)
    real_aar = _make_aar("f1", acquisition=0.95, retention=0.98, confidence=0.8)

    vote_engine = MagicMock()
    vote_engine.aarrr.return_value = [sim_aar]
    logger = MagicMock()
    logger.compute_aarrr.return_value = real_aar

    sim = SimulatedDataSource(vote_engine)
    real = RealDataSource(logger)
    hybrid = HybridDataSource(sim, real)
    result = hybrid.get_aarrr_batch(FEATURES, "p", "s", ARCHETYPES)

    for dim in ("acquisition", "activation", "retention", "revenue", "referral"):
        val = getattr(result[0], dim)
        assert 0.0 <= val <= 1.0, f"{dim}={val} out of bounds"


def test_create_data_source_factory():
    # ai mode — VoteEngine is imported inside create_data_source; patch its canonical location
    mock_backend = MagicMock()
    with patch("user_soul.engines.vote.VoteEngine") as MockVE:
        MockVE.return_value = MagicMock()
        src = create_data_source("ai", backend=mock_backend)
        assert isinstance(src, SimulatedDataSource)
        assert src.mode == "ai"

    # real mode — no VoteEngine needed
    mock_logger = MagicMock()
    src_real = create_data_source("real", event_logger=mock_logger)
    assert isinstance(src_real, RealDataSource)
    assert src_real.mode == "real"

    # hybrid mode
    with patch("user_soul.engines.vote.VoteEngine") as MockVE:
        MockVE.return_value = MagicMock()
        src_hybrid = create_data_source("hybrid", backend=mock_backend, event_logger=mock_logger)
        assert isinstance(src_hybrid, HybridDataSource)
        assert src_hybrid.mode == "hybrid"

    # invalid mode raises
    with pytest.raises(ValueError, match="Unknown mode"):
        create_data_source("invalid")
