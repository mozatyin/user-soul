"""EvaluationMetric + schema extraction from goal text."""
from __future__ import annotations

from dataclasses import dataclass

import user_soul.core as _core


@dataclass
class EvaluationMetric:
    name: str
    type: str      # "bool" | "scale_1_5" | "text"
    question: str  # asked at end of each simulated session


def extract_evaluation_schema(goal: str, api_key: str) -> list[EvaluationMetric]:
    """Extract 3-6 evaluation metrics from a goal description or PRD text.

    Uses Sonnet (one call) — this is a reasoning task, not simulation.
    """
    prompt = (
        "You are a product analyst. Read the following product goal and extract 3-6 "
        "evaluation metrics that would tell us if the product is succeeding.\n\n"
        f"Goal:\n{goal[:2000]}\n\n"
        "For each metric, decide:\n"
        '- type: "bool" (yes/no), "scale_1_5" (intensity 1-5), or "text" (qualitative)\n'
        "- question: a specific question to ask a simulated user at the end of their session\n\n"
        "Reply with JSON array only:\n"
        '[{"name": "snake_case_name", "type": "bool|scale_1_5|text", "question": "..."}]'
    )
    raw, _ = _core._llm_call(prompt, api_key, max_tokens=512)
    items = _core._safe_json_arr(raw)
    metrics = []
    for item in items:
        if not isinstance(item, dict):
            continue
        name = item.get("name", "")
        typ = item.get("type", "")
        question = item.get("question", "")
        if name and typ in ("bool", "scale_1_5", "text") and question:
            metrics.append(EvaluationMetric(name=name, type=typ, question=question))
    return metrics
