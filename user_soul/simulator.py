"""Progressive Monte Carlo behavioral simulator."""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from user_soul.scenarios import ScenarioContext
import user_soul.core as _core


@dataclass
class SimulationRun:
    """One simulated user session — a single roll of the die."""
    persona_id: str
    context: ScenarioContext
    features_used: list[str] = field(default_factory=list)
    features_skipped: list[str] = field(default_factory=list)


@dataclass
class FeatureSignal:
    """Empirical signal for one feature, aggregated from N simulation runs."""
    feature_id: str
    feature_name: str
    n_simulations: int
    usage_rate: float
    exposure_rate: float
    skip_rate: float
    context_map: dict[str, float]
    day_curve: dict[int, float]
    implied_kano: str
    implied_aarrr_score: float


_SIMULATION_MAX_TOKENS = 800


def _build_simulation_prompt(
    persona: dict,
    features: list[dict],
    context: ScenarioContext,
) -> str:
    feature_lines = "\n".join(f"- {f['id']}: {f['name']}" for f in features)
    return (
        f"You are roleplaying as {persona['name']}.\n\n"
        f"Who you are:\n"
        f"{persona.get('description', '')}\n"
        f"Cohort: {persona.get('cohort', '')}\n"
        f"Motivations: {', '.join(persona.get('motivations', []))}\n"
        f"Pain points: {', '.join(persona.get('pain_points', []))}\n\n"
        f"Right now:\n"
        f"Time: {context.time_of_day.replace('_', ' ')}\n"
        f"Your state: {context.emotional_state}\n"
        f"What prompted you to open the app: {context.trigger.replace('_', ' ')}\n"
        f"Days using this app: {context.usage_day}\n\n"
        f"You open the app. These features are available:\n"
        f"{feature_lines}\n\n"
        f"Describe your next 6-8 actions in the app.\n"
        f"Write only what you DO — not what you think about the features.\n"
        f"Use third person: \"{persona['name']} tapped...\", \"{persona['name']} scrolled...\"\n\n"
        f"After the story, each on its own line:\n"
        f"USED: [comma-listed feature IDs you actually used, or 'none']\n"
        f"SEEN: [comma-listed feature IDs you saw but skipped, or 'none']\n"
    )


def _parse_feature_ids(line: str, valid_ids: set[str]) -> list[str]:
    """Extract feature IDs from a USED:/SEEN: line."""
    if not line or "none" in line.lower():
        return []
    parts = re.split(r"[,\s]+", line.strip())
    return [p.strip() for p in parts if p.strip() in valid_ids]


def _derive_kano(usage_rate: float) -> str:
    """Derive Kano category from empirical usage frequency.

    Boundaries (inclusive lower, exclusive upper):
      Must-Have  : usage_rate >  0.80
      Performance: usage_rate >= 0.50
      Delighter  : usage_rate >= 0.20
      Indifferent: usage_rate <  0.20
    """
    if usage_rate > 0.80:
        return "Must-Have"
    if usage_rate >= 0.50:
        return "Performance"
    if usage_rate >= 0.20:
        return "Delighter"
    return "Indifferent"


def _derive_aarrr(day_curve: dict[int, float]) -> float:
    """Derive AARRR score: activation (day1) + retention (day7) + revenue proxy (day30)."""
    day1 = day_curve.get(1, 0.0)
    day7 = day_curve.get(7, 0.0)
    day30 = day_curve.get(30, 0.0)
    return round(0.30 * day1 + 0.30 * day7 + 0.40 * day30, 4)


class PersonaSimulator:
    """Behavioral simulator: each run narrates a user session, we observe what features are used."""
    def __init__(self, personas: list[dict], api_key: str):
        self.personas = personas
        self.api_key = api_key

    def simulate(
        self,
        features: list[dict],
        n_runs: int = 5,
    ) -> list[FeatureSignal]:
        """Run N sessions per persona. Aggregate into empirical FeatureSignal per feature.

        Total LLM calls = len(personas) × n_runs.
        Each call uses Haiku at temperature=1.0 (true stochasticity).
        """
        from user_soul.scenarios import random_context
        from collections import defaultdict

        feature_ids = [f["id"] for f in features]
        feature_name_map = {f["id"]: f["name"] for f in features}

        all_runs: list[SimulationRun] = []

        for persona in self.personas:
            role = persona.get("role")
            for _ in range(n_runs):
                ctx = random_context(role=role)
                run = self._simulate_one(persona, features, ctx)
                all_runs.append(run)

        total = len(all_runs)
        if total == 0:
            return []

        signals = []
        for fid in feature_ids:
            used_runs = [r for r in all_runs if fid in r.features_used]
            skipped_runs = [r for r in all_runs if fid in r.features_skipped]
            exposed_runs = used_runs + skipped_runs

            usage_rate = len(used_runs) / total
            exposure_rate = len(exposed_runs) / total
            skip_rate = len(skipped_runs) / total

            # context_map: usage_rate per emotional_state
            states: dict[str, list[int]] = defaultdict(lambda: [0, 0])
            for run in all_runs:
                state = run.context.emotional_state
                states[state][1] += 1
                if fid in run.features_used:
                    states[state][0] += 1
            context_map = {
                state: round(counts[0] / counts[1], 4)
                for state, counts in states.items()
                if counts[1] > 0
            }

            # day_curve: usage_rate per usage_day
            days: dict[int, list[int]] = defaultdict(lambda: [0, 0])
            for run in all_runs:
                day = run.context.usage_day
                days[day][1] += 1
                if fid in run.features_used:
                    days[day][0] += 1
            day_curve = {
                day: round(counts[0] / counts[1], 4)
                for day, counts in days.items()
                if counts[1] > 0
            }

            implied_kano = _derive_kano(usage_rate)
            implied_aarrr = _derive_aarrr(day_curve)

            signals.append(FeatureSignal(
                feature_id=fid,
                feature_name=feature_name_map.get(fid, fid),
                n_simulations=total,
                usage_rate=round(usage_rate, 4),
                exposure_rate=round(exposure_rate, 4),
                skip_rate=round(skip_rate, 4),
                context_map=context_map,
                day_curve=day_curve,
                implied_kano=implied_kano,
                implied_aarrr_score=implied_aarrr,
            ))

        return signals

    def _simulate_one(
        self,
        persona: dict,
        features: list[dict],
        context: ScenarioContext,
    ) -> SimulationRun:
        """Run one behavioral simulation session. Core Monte Carlo roll."""
        valid_ids = {f["id"] for f in features}
        prompt = _build_simulation_prompt(persona, features, context)
        raw, _ = _core._llm_call(
            prompt,
            self.api_key,
            max_tokens=_SIMULATION_MAX_TOKENS,
            temperature=1.0,
            model=_core._haiku_model(self.api_key),
        )
        used: list[str] = []
        skipped: list[str] = []
        for line in raw.splitlines():
            line = line.strip()
            if line.upper().startswith("USED:"):
                used = _parse_feature_ids(line[5:], valid_ids)
            elif line.upper().startswith("SEEN:"):
                skipped = _parse_feature_ids(line[5:], valid_ids)
        used_set = set(used)
        skipped = [f for f in skipped if f not in used_set]
        return SimulationRun(
            persona_id=persona["id"],
            context=context,
            features_used=used,
            features_skipped=skipped,
        )
