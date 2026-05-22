"""GateLedger — accumulates MCV gate results across the ELTM pipeline.

Pass this object into ELTM and each gate appends its findings.
Code-Soul receives gate context via contract.json["mcv_gates"].
"""
from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class GateLedger:
    """Carries MCV gate results through the ELTM M0→M5 pipeline.

    Gates:
        gate0: PRD scan — persona pool + population AARRR scores
        gate1: coherence check — dependency validation after M1 feature selection
        gate2: journey simulation — task completion rate on M2 screen architecture
        gate3: adversarial frictions — edge-case scan on M4 HTML
        gate4: full baseline simulation — day1_return_rate before Code-Soul
    """
    gate0_persona_pool: list | None = None          # list[AgentProfile]
    gate0_aarrr: list | None = None                 # list[FeatureAAR]
    gate1_coherence: object | None = None           # CoherenceReport
    gate2_journey: object | None = None             # JourneyReport
    gate3_adversarial_frictions: list[str] = field(default_factory=list)
    gate4_baseline: object | None = None            # SimulationReport

    def to_dict(self) -> dict:
        """Serialise for contract.json injection (primitives only)."""
        d: dict = {}
        if self.gate0_aarrr:
            d["gate0_aarrr"] = [
                {
                    "feature_id": f.feature_id,
                    "retention": round(f.retention, 4),
                    "activation": round(f.activation, 4),
                    "confidence": round(f.confidence, 4),
                }
                for f in self.gate0_aarrr
            ]
        if self.gate1_coherence is not None:
            d["gate1_coherent"] = self.gate1_coherence.is_coherent
            d["gate1_reinstate"] = self.gate1_coherence.reinstate_recommendations
        if self.gate2_journey is not None:
            d["gate2_completion_rate"] = self.gate2_journey.completion_rate
            d["gate2_passes"] = self.gate2_journey.passes_gate
            d["gate2_drop_offs"] = self.gate2_journey.drop_off_by_screen
            d["gate2_fogg_violations"] = self.gate2_journey.fogg_violations
        if self.gate3_adversarial_frictions:
            d["gate3_frictions"] = self.gate3_adversarial_frictions
        if self.gate4_baseline is not None:
            rate = self.gate4_baseline.day1_return_rate_adjusted
            d["gate4_day1_adjusted"] = rate
        return d
