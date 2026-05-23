"""VoteEngine — persona-grounded decision making and AARRR scoring."""
from __future__ import annotations

import json
import re
import statistics
from collections import Counter
from typing import Any

from user_soul.backend import LLMBackend
from user_soul.models import AgentProfile, Archetype, DecisionResult, FeatureAAR


def _safe_json(text: str) -> dict:
    text = text.strip()
    text = re.sub(r'^```(?:json)?\s*', '', text)
    text = re.sub(r'\s*```$', '', text)
    m = re.search(r'\{.*\}', text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except (json.JSONDecodeError, ValueError):
            pass
    return {}


def _safe_json_arr(text: str) -> list:
    text = text.strip()
    text = re.sub(r'^```(?:json)?\s*', '', text)
    text = re.sub(r'\s*```$', '', text)
    m = re.search(r'\[.*\]', text, re.DOTALL)
    if m:
        try:
            result = json.loads(m.group())
            if isinstance(result, list):
                return result
        except (json.JSONDecodeError, ValueError):
            pass
    return []


def _to_float(val: Any, fallback: float) -> float:
    try:
        return float(val)
    except (ValueError, TypeError):
        return fallback


_AARRR_VOTE_PROMPT = """You are scoring product features for a mobile app from the perspective of different user archetypes.

Product: {product_description}

User archetypes (each with their background story):
{archetypes_block}

Features to score:
{features_block}

For each feature, score its impact on each AARRR dimension (0.0 = no impact, 1.0 = primary driver)
from each archetype's perspective. Then compute the mean across archetypes.

Return ONLY valid JSON (no markdown):
[
  {{
    "feature_id": "...",
    "archetype_votes": {{
      "ArchetypeName": {{"acquisition": 0.0, "activation": 0.0, "retention": 0.0, "revenue": 0.0, "referral": 0.0}},
      ...
    }},
    "mean": {{"acquisition": 0.0, "activation": 0.0, "retention": 0.0, "revenue": 0.0, "referral": 0.0}}
  }},
  ...
]"""


class VoteEngine:

    def __init__(self, backend: LLMBackend):
        self._backend = backend

    def classify(self, question: str, options: list[str], context: str,
                 personas: list[AgentProfile]) -> DecisionResult:
        options_str = " | ".join(options)
        votes: list[str] = []

        for persona in personas:
            prompt = (
                f"你是：{persona.to_human_story()}\n\n"
                f"Context: {context}\n"
                f"Question: {question}\n"
                f"Options: {options_str}\n\n"
                f'Reply with JSON only: {{"choice": "...", "reasoning": "one sentence"}}'
            )
            raw = self._backend.text(prompt, model_tier="fast")
            data = _safe_json(raw)
            votes.append(data.get("choice", options[0]))

        counts = Counter(votes)
        winner, winner_count = counts.most_common(1)[0]
        total = len(votes)
        dist = {opt: round(counts.get(opt, 0) / total, 4) for opt in options}
        return DecisionResult(
            value=winner,
            confidence=round(winner_count / total, 4),
            distribution=dist,
            mode="validated" if len(personas) > 1 else "fast",
            raw_votes=votes,
        )

    def score(self, question: str, lo: float, hi: float, context: str,
              personas: list[AgentProfile]) -> DecisionResult:
        votes: list[float] = []

        for persona in personas:
            prompt = (
                f"你是：{persona.to_human_story()}\n\n"
                f"Context: {context}\n"
                f"Question: {question} (scale: {lo}–{hi})\n\n"
                f'Reply with JSON only: {{"score": <number>, "reasoning": "one sentence"}}'
            )
            raw = self._backend.text(prompt, model_tier="fast")
            data = _safe_json(raw)
            v = max(lo, min(hi, _to_float(data.get("score"), (lo + hi) / 2)))
            votes.append(v)

        avg = statistics.mean(votes)
        stdev = statistics.stdev(votes) if len(votes) > 1 else 0.0
        confidence = max(0.0, 1.0 - stdev / max(hi - lo, 1e-6))
        return DecisionResult(
            value=round(avg, 4),
            confidence=round(confidence, 4),
            distribution={"mean": round(avg, 4), "stdev": round(stdev, 4)},
            mode="validated" if len(personas) > 1 else "fast",
            raw_votes=votes,
        )

    def aarrr(self, product: str, features: list[dict],
              archetypes: list[Archetype]) -> list[FeatureAAR]:
        if not features:
            return []

        archetypes_block = "\n".join(
            f"- {a.name}: {a.background_story or a.description}"
            for a in archetypes
        )
        features_block = "\n".join(
            f"- id={f['id']}: {f['name']} — {f.get('description', '')}"
            for f in features
        )
        prompt = _AARRR_VOTE_PROMPT.format(
            product_description=product[:800],
            archetypes_block=archetypes_block,
            features_block=features_block,
        )

        raw = self._backend.text(prompt, max_tokens=8000, model_tier="smart")

        items = _safe_json_arr(raw)
        items_by_id = {
            item["feature_id"]: item
            for item in items
            if isinstance(item, dict) and "feature_id" in item
        }

        _DIMS = ("acquisition", "activation", "retention", "revenue", "referral")
        scored: list[FeatureAAR] = []

        for f in features:
            fid = f["id"]
            item = items_by_id.get(fid)
            if item:
                mean = item.get("mean", {})
                arch_votes = item.get("archetype_votes", {})
                stdevs = []
                for dim in _DIMS:
                    vals = [v.get(dim, 0.5) for v in arch_votes.values() if isinstance(v, dict)]
                    if len(vals) >= 2:
                        stdevs.append(statistics.stdev(vals))
                confidence = round(max(0.0, 1.0 - (sum(stdevs) / len(stdevs) if stdevs else 0.0)), 4)
                scored.append(FeatureAAR(
                    feature_id=fid,
                    acquisition=round(min(1.0, max(0.0, float(mean.get("acquisition", 0.5)))), 4),
                    activation=round(min(1.0, max(0.0, float(mean.get("activation", 0.5)))), 4),
                    retention=round(min(1.0, max(0.0, float(mean.get("retention", 0.5)))), 4),
                    revenue=round(min(1.0, max(0.0, float(mean.get("revenue", 0.2)))), 4),
                    referral=round(min(1.0, max(0.0, float(mean.get("referral", 0.2)))), 4),
                    confidence=confidence,
                    archetype_votes=arch_votes,
                ))
            else:
                scored.append(FeatureAAR(
                    feature_id=fid,
                    acquisition=0.5, activation=0.5, retention=0.5,
                    revenue=0.2, referral=0.2,
                    confidence=0.0, archetype_votes={},
                ))
        return scored
