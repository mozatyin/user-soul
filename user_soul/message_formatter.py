"""Message Formatter — converts User-Soul outputs to protocol-compliant dicts.

Translates GradedPlaytestFeedback + ActionSpec into the unified dict format
so PM-Soul can consume and dispatch to downstream actors.

Output format: {"id": uuid, "type": str, "source": "user-soul", "context_id": str, ...}
"""
from __future__ import annotations

from typing import Any
from uuid import uuid4

from user_soul.trisoul_protocol import make_dict, Actor
from user_soul.models import (
    ActionSpec,
    GradedPlaytestFeedback,
    PlaytestFeedback,
)


def format_graded_verdict(
    feedback: GradedPlaytestFeedback,
    context_id: str,
) -> dict[str, Any]:
    """Package GradedPlaytestFeedback as a user_verdict dict."""
    diagnosis_items = []
    if feedback.diagnosis:
        for d in feedback.diagnosis.diagnoses:
            diagnosis_items.append({
                "category": d.category.value,
                "severity": d.severity,
                "description": d.description,
                "owner": d.owner,
                "evidence": {k: v for k, v in d.evidence.items()},
                "common_failure_turn": d.common_failure_turn,
            })

    has_p0 = feedback.has_blockers
    if not has_p0 and feedback.diagnosis:
        has_p0 = any(d.severity == "P0" for d in feedback.diagnosis.diagnoses)

    sev = "P0" if has_p0 else ("P1" if feedback.verdict in ("NEEDS_WORK", "CRITICAL") else None)

    return make_dict(
        "user_verdict",
        Actor.USER_SOUL,
        context_id,
        severity=sev,
        verdict=feedback.verdict,
        score=feedback.score,
        tier_scores=feedback.tier_scores,
        diagnosis=diagnosis_items,
        diagnosis_categories=feedback.diagnosis_categories,
        primary_owner=feedback.primary_owner,
        personas_completed=feedback.personas_completed,
        personas_total=feedback.personas_total,
        suggestions=feedback.suggestions,
        raw_summary=feedback.raw_summary[:2000],
    )


def format_flat_verdict(
    feedback: PlaytestFeedback,
    context_id: str,
) -> dict[str, Any]:
    """Package single-tier PlaytestFeedback as a user_verdict dict."""
    issues = []
    for iss in feedback.issues:
        issues.append({
            "severity": iss.severity,
            "description": iss.description,
            "category": iss.category,
            "evidence": iss.evidence[:3],
            "affected_personas": iss.affected_personas,
        })

    sev = "P0" if feedback.has_blockers else (
        "P1" if feedback.verdict == "NEEDS_WORK" else None
    )

    return make_dict(
        "user_verdict",
        Actor.USER_SOUL,
        context_id,
        severity=sev,
        verdict=feedback.verdict,
        score=feedback.score,
        issues=issues,
        suggestions=feedback.suggestions,
        personas_completed=feedback.personas_completed,
        personas_total=feedback.personas_total,
        raw_summary=feedback.raw_summary[:2000],
    )


def format_action_as_dict(
    spec: ActionSpec,
    context_id: str,
    ref_id: str | None = None,
) -> dict[str, Any]:
    """Convert an ActionSpec into a user_test_feedback dict for the target actor.

    The output dict is directly consumable by actor.receive_from_pm().
    """
    return make_dict(
        "user_test_feedback",
        Actor.PM_SOUL,
        context_id,
        severity=spec.severity or None,
        ref_id=ref_id,
        action_type=spec.action_type,
        description=spec.description,
        evidence=spec.evidence,
        payload=spec.payload,
        source_tier=spec.source_tier,
        diagnosis_category=spec.diagnosis_category,
    )


def action_specs_to_dicts(
    specs: list[ActionSpec],
    context_id: str,
    verdict_ref_id: str | None = None,
) -> list[dict[str, Any]]:
    """Batch-convert ActionSpecs into protocol-compliant dicts."""
    return [
        format_action_as_dict(spec, context_id, ref_id=verdict_ref_id)
        for spec in specs
    ]
