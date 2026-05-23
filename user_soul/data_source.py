"""DataSource — unified interface for AARRR data from AI simulation or real users.

Implements the Strategy pattern: callers use DataSource without knowing if data
is AI-simulated (development/pre-launch) or real (post-launch).

Hybrid mode blends both using Bayesian updating:
    blended = (prior_strength * sim + n_real * real) / (prior_strength + n_real)
As real sample size grows, the estimate shifts from simulation toward observation.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol

from user_soul.models import FeatureAAR

if TYPE_CHECKING:
    from user_soul.event_logger import AAARREventMap, EventLogger


class DataSource(Protocol):
    def get_aarrr_batch(
        self,
        features: list[dict],
        product: str,
        segment: str,
        archetypes: list,
    ) -> list[FeatureAAR]: ...

    @property
    def mode(self) -> str: ...


class SimulatedDataSource:
    mode = "ai"

    def __init__(self, vote_engine: Any) -> None:
        self._vote = vote_engine

    def get_aarrr_batch(
        self,
        features: list[dict],
        product: str,
        segment: str,
        archetypes: list,
    ) -> list[FeatureAAR]:
        return self._vote.aarrr(product, features, archetypes)


class RealDataSource:
    mode = "real"

    def __init__(self, event_logger: Any, event_map: Any | None = None) -> None:
        self._logger = event_logger
        self._event_map = event_map

    def get_aarrr_batch(
        self,
        features: list[dict],
        product: str,
        segment: str,
        archetypes: list,
    ) -> list[FeatureAAR]:
        results: list[FeatureAAR] = []
        for feature in features:
            fid = feature.get("id", "")
            try:
                aar = self._logger.compute_aarrr(fid, self._event_map)
            except Exception:
                aar = FeatureAAR(
                    feature_id=fid,
                    acquisition=0.5,
                    activation=0.5,
                    retention=0.5,
                    revenue=0.2,
                    referral=0.2,
                    confidence=0.0,
                    archetype_votes={},
                )
            if aar.confidence == 0.0:
                aar = FeatureAAR(
                    feature_id=fid,
                    acquisition=0.5,
                    activation=0.5,
                    retention=0.5,
                    revenue=0.2,
                    referral=0.2,
                    confidence=0.0,
                    archetype_votes={},
                )
            results.append(aar)
        return results


class HybridDataSource:
    """Bayesian blend: AI simulation as prior, real data as likelihood.

    prior_strength=20 means ~20 real data points to equal simulation weight.
    """
    mode = "hybrid"

    def __init__(
        self,
        simulated: SimulatedDataSource,
        real: RealDataSource,
        prior_strength: float = 20.0,
    ) -> None:
        self._sim = simulated
        self._real = real
        self._prior = prior_strength

    def get_aarrr_batch(
        self,
        features: list[dict],
        product: str,
        segment: str,
        archetypes: list,
    ) -> list[FeatureAAR]:
        sim_results = self._sim.get_aarrr_batch(features, product, segment, archetypes)
        real_results = self._real.get_aarrr_batch(features, product, segment, archetypes)
        return [self._blend(s, r) for s, r in zip(sim_results, real_results)]

    def _blend(self, sim: FeatureAAR, real: FeatureAAR) -> FeatureAAR:
        n_real = real.confidence * 100
        w_real = n_real / (self._prior + n_real)
        w_sim = 1.0 - w_real

        def blend_dim(s: float, r: float) -> float:
            return round(min(1.0, max(0.0, w_sim * s + w_real * r)), 4)

        blended_confidence = round(min(1.0, max(real.confidence, sim.confidence * 0.5)), 4)

        return FeatureAAR(
            feature_id=sim.feature_id,
            acquisition=blend_dim(sim.acquisition, real.acquisition),
            activation=blend_dim(sim.activation, real.activation),
            retention=blend_dim(sim.retention, real.retention),
            revenue=blend_dim(sim.revenue, real.revenue),
            referral=blend_dim(sim.referral, real.referral),
            confidence=blended_confidence,
            archetype_votes=sim.archetype_votes,
        )


def create_data_source(
    mode: str,
    backend: Any = None,
    event_logger: Any = None,
    archetypes: Any = None,
    prior_strength: float = 20.0,
) -> DataSource:
    if mode == "ai":
        from user_soul.engines.vote import VoteEngine
        vote_engine = VoteEngine(backend)
        return SimulatedDataSource(vote_engine)
    elif mode == "real":
        return RealDataSource(event_logger)
    elif mode == "hybrid":
        from user_soul.engines.vote import VoteEngine
        vote_engine = VoteEngine(backend)
        sim = SimulatedDataSource(vote_engine)
        real = RealDataSource(event_logger)
        return HybridDataSource(sim, real, prior_strength=prior_strength)
    else:
        raise ValueError(f"Unknown mode: {mode!r}. Expected 'ai', 'real', or 'hybrid'.")
