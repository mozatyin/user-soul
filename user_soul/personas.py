"""Persona generation and caching."""
from __future__ import annotations

import json
import re
from pathlib import Path

from user_soul.core import Persona, _llm_call


def _generate_personas(
    prd_text: str,
    archetype: str,
    target_market: str,
    api_key: str,
    n: int,
) -> list[Persona]:
    """Call LLM to generate N distinct user personas."""
    prompt = (
        f"You are a product researcher. Generate {n} distinct user personas for the following app.\n\n"
        f"PRD excerpt (first 2000 chars):\n{prd_text[:2000]}\n\n"
        f"Champion app (UX reference): {archetype}\n"
        f"Target market: {target_market}\n\n"
        f"Each persona should represent a real user segment with different motivations.\n"
        f"Reply with JSON array only:\n"
        f'[{{"id": "p1", "name": "...", "cohort": "...", "description": "...", '
        f'"motivations": ["...", "..."], "pain_points": ["...", "..."]}}]'
    )
    raw, _ = _llm_call(prompt, api_key, max_tokens=1500)
    raw = raw.strip()
    # Strip markdown fences
    raw = re.sub(r'^```(?:json)?\s*', '', raw)
    raw = re.sub(r'\s*```$', '', raw)
    # Extract first [...] block
    m = re.search(r'\[.*\]', raw, re.DOTALL)
    arr = []
    if m:
        try:
            arr = json.loads(m.group())
        except (json.JSONDecodeError, ValueError):
            arr = []
    personas = []
    for i, item in enumerate(arr):
        if not isinstance(item, dict):
            continue
        personas.append(Persona(
            id=item.get("id", f"p{i+1}"),
            name=item.get("name", f"User {i+1}"),
            description=item.get("description", ""),
            cohort=item.get("cohort", target_market),
            motivations=item.get("motivations", []),
            pain_points=item.get("pain_points", []),
        ))
    return personas[:n]


def load_or_generate(
    state_dir: Path,
    prd_text: str,
    archetype: str,
    target_market: str,
    api_key: str,
    n: int = 5,
) -> list[Persona]:
    """Load personas from state_dir/personas.json, or generate fresh and cache.

    Args:
        state_dir: directory where personas.json will be cached
        prd_text: full PRD text to derive personas from
        archetype: champion app name (e.g. "Replika") — UX reference point
        target_market: target market description (e.g. "Young adults seeking self-reflection")
        api_key: Anthropic or OpenRouter API key
        n: number of personas to generate (default 5)

    Returns:
        list of Persona objects (length == n, or fewer if LLM returned fewer)
    """
    cache_path = state_dir / "personas.json"
    if cache_path.exists():
        try:
            data = json.loads(cache_path.read_text(encoding="utf-8"))
            if isinstance(data, list) and len(data) >= n:
                personas = [
                    Persona(
                        id=item["id"],
                        name=item["name"],
                        description=item.get("description", ""),
                        cohort=item.get("cohort", ""),
                        motivations=item.get("motivations", []),
                        pain_points=item.get("pain_points", []),
                    )
                    for item in data[:n]
                    if isinstance(item, dict) and item.get("id") and item.get("name")
                ]
                if len(personas) >= n:
                    return personas
        except (json.JSONDecodeError, KeyError, ValueError):
            pass

    personas = _generate_personas(prd_text, archetype, target_market, api_key, n)
    if personas:
        cache_path.write_text(
            json.dumps(
                [{
                    "id": p.id, "name": p.name, "description": p.description,
                    "cohort": p.cohort, "motivations": p.motivations,
                    "pain_points": p.pain_points,
                } for p in personas],
                indent=2, ensure_ascii=False,
            ),
            encoding="utf-8",
        )
    return personas
