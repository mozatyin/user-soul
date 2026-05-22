"""FeatureFilter — Phase 0.7 AARRR-driven feature prioritisation.

Takes a raw feature list from ELTM product research and scores each feature
through AI personas using AARRR dimensions, then classifies features into
must_have / nice_to_have / skip tiers for PRD injection.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from user_soul.backend import LLMBackend
from user_soul.engines.vote import VoteEngine
from user_soul.models import FeatureAAR
from user_soul.population import Archetype


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ScoredFeature:
    id: str
    name: str
    description: str
    category: str                   # e.g. "onboarding", "gameplay", "social"
    source: str                     # e.g. "chess.com", "lichess.org"
    aarrr: FeatureAAR
    classification: str             # "must_have" | "nice_to_have" | "skip"
    priority_score: float           # weighted composite


@dataclass
class FeatureFilterReport:
    product: str
    target_segment: str
    total_input: int
    must_have: list[ScoredFeature]          # priority_score >= 0.6
    nice_to_have: list[ScoredFeature]       # 0.35 <= score < 0.6
    skip: list[ScoredFeature]               # score < 0.35
    top_features: list[ScoredFeature]       # top_n sorted by priority_score, for PRD injection
    archetypes_used: list[str]              # archetype names


# ---------------------------------------------------------------------------
# Priority formula weights
# ---------------------------------------------------------------------------

_WEIGHTS = {
    "retention": 0.40,
    "activation": 0.30,
    "acquisition": 0.15,
    "revenue": 0.10,
    "referral": 0.05,
}

_BATCH_SIZE = 25


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _slug(name: str) -> str:
    """Convert feature name to snake_case id fragment."""
    s = name.lower().strip()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = s.strip("_")
    return s[:40] if s else "feature"


def _normalize_features(raw: list[dict]) -> list[dict]:
    """Ensure each feature has: id, name, description, category, source."""
    normalized = []
    for idx, item in enumerate(raw):
        name = item.get("name") or f"feature_{idx}"
        normalized.append({
            "id": f"{idx:03d}_{_slug(name)}",
            "name": name,
            "description": item.get("description") or "",
            "category": item.get("category") or "general",
            "source": item.get("source") or "unknown",
        })
    return normalized


def _auto_archetypes(
    backend: LLMBackend,
    product: str,
    segment: str,
    n: int = 4,
) -> list[Archetype]:
    """Quick archetype generation via single LLM call. Returns list[Archetype].

    Falls back to two generic archetypes if the LLM response cannot be parsed.
    """
    prompt = (
        f"为产品 '{product}' 生成 {n} 个目标用户原型，目标细分：{segment}。\n\n"
        "每个原型需要：name（英文）、description（一句话）、background_story（≤60字具体人物故事）。\n\n"
        "只返回合法 JSON 数组（不加 markdown）：\n"
        '[{"name": "...", "description": "...", "background_story": "..."}, ...]'
    )
    raw = backend.text(prompt, max_tokens=1024, model_tier="fast")

    # Strip markdown fences if any
    raw = raw.strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)

    # Try to extract JSON array
    m = re.search(r"\[.*\]", raw, re.DOTALL)
    items: list[dict] = []
    if m:
        try:
            parsed = json.loads(m.group())
            if isinstance(parsed, list):
                items = parsed
        except (json.JSONDecodeError, ValueError):
            pass

    archetypes: list[Archetype] = []
    for item in items:
        if not isinstance(item, dict) or not item.get("name"):
            continue
        archetypes.append(Archetype(
            name=item["name"],
            frequency=1.0 / max(len(items), 1),
            description=item.get("description", ""),
            trait_constraints={},
            background_story=item.get("background_story", ""),
        ))

    if not archetypes:
        # Fallback: two generic archetypes
        archetypes = [
            Archetype("Engaged User", 0.60, "actively explores features", {}, ""),
            Archetype("Casual User", 0.40, "light, occasional usage", {}, ""),
        ]

    # Normalise frequencies
    total = sum(a.frequency for a in archetypes)
    if total > 0:
        archetypes = [
            Archetype(a.name, round(a.frequency / total, 6), a.description,
                      a.trait_constraints, a.background_story)
            for a in archetypes
        ]

    return archetypes


def _compute_priority_score(aarrr: FeatureAAR) -> float:
    """Weighted composite: retention*0.4 + activation*0.3 + acquisition*0.15 + revenue*0.1 + referral*0.05."""
    return round(
        aarrr.retention  * _WEIGHTS["retention"]
        + aarrr.activation * _WEIGHTS["activation"]
        + aarrr.acquisition * _WEIGHTS["acquisition"]
        + aarrr.revenue    * _WEIGHTS["revenue"]
        + aarrr.referral   * _WEIGHTS["referral"],
        6,
    )


def _classify(score: float) -> str:
    if score >= 0.6:
        return "must_have"
    if score >= 0.35:
        return "nice_to_have"
    return "skip"


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class FeatureFilter:

    def __init__(self, backend: LLMBackend):
        self._backend = backend
        self._vote = VoteEngine(backend)

    def filter(
        self,
        product_description: str,
        raw_features: list[dict],
        target_segment: str,
        archetypes: list[Archetype] | None = None,
        top_n: int = 25,
    ) -> FeatureFilterReport:
        """Score and classify features by AARRR impact for the target segment.

        Args:
            product_description: e.g. "mobile chess app for beginners"
            raw_features: list of dicts with at least {"name": "..."}, from ELTM scan
            target_segment: e.g. "18-25 year old beginner chess players"
            archetypes: optional pre-built archetypes; auto-generated if None
            top_n: how many top features to include in FeatureFilterReport.top_features

        Returns:
            FeatureFilterReport with all tiers and top_features list
        """
        # 1. Normalise
        features = _normalize_features(raw_features)

        # 2. Archetypes
        if archetypes is None:
            archetypes = _auto_archetypes(self._backend, product_description, target_segment)

        # 3. Batch AARRR scoring
        all_aarrr: list[FeatureAAR] = []
        if len(features) <= _BATCH_SIZE:
            all_aarrr = self._vote.aarrr(product_description, features, archetypes)
        else:
            for i in range(0, len(features), _BATCH_SIZE):
                batch = features[i: i + _BATCH_SIZE]
                batch_aarrr = self._vote.aarrr(product_description, batch, archetypes)
                all_aarrr.extend(batch_aarrr)

        # 4. Build ScoredFeature list
        aarrr_by_id = {a.feature_id: a for a in all_aarrr}
        scored: list[ScoredFeature] = []
        for feat in features:
            aarrr = aarrr_by_id.get(feat["id"])
            if aarrr is None:
                # Fallback neutral scores
                aarrr = FeatureAAR(
                    feature_id=feat["id"],
                    acquisition=0.5, activation=0.5, retention=0.5,
                    revenue=0.2, referral=0.2,
                    confidence=0.0, archetype_votes={},
                )
            score = _compute_priority_score(aarrr)
            scored.append(ScoredFeature(
                id=feat["id"],
                name=feat["name"],
                description=feat["description"],
                category=feat["category"],
                source=feat["source"],
                aarrr=aarrr,
                classification=_classify(score),
                priority_score=score,
            ))

        # 5. Sort descending
        scored.sort(key=lambda f: f.priority_score, reverse=True)

        # 6. Tier split
        must_have = [f for f in scored if f.classification == "must_have"]
        nice_to_have = [f for f in scored if f.classification == "nice_to_have"]
        skip = [f for f in scored if f.classification == "skip"]

        # 7. Top-N
        top_features = scored[:top_n]

        return FeatureFilterReport(
            product=product_description,
            target_segment=target_segment,
            total_input=len(features),
            must_have=must_have,
            nice_to_have=nice_to_have,
            skip=skip,
            top_features=top_features,
            archetypes_used=[a.name for a in archetypes],
        )
