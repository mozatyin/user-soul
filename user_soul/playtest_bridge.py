"""Playtest bridge — User-Soul borrows Code-Soul's Playwright test bench.

User-Soul brings the personas and the "eyes" (LLMBackend).
Code-Soul provides the test bench (PlaywrightDriver + runner).
This bridge connects them without duplicating any Playwright code.
"""
from __future__ import annotations

import base64
import json
import os
import re
import sys
from typing import Any

from user_soul.backend import LLMBackend
from user_soul.models import AgentProfile, PlaytestFeedback, PlaytestIssue, GradedPlaytestFeedback, SmokePlaytestResult
from user_soul.game_knowledge import (
    GameKnowledge, KnowledgeTier, TierResult, DifferentialDiagnosis,
    brief_for_tier, distribute_personas_across_tiers, minimum_personas_for_graded_test,
)


def _ensure_code_soul_importable() -> None:
    code_soul_path = os.environ.get("CODE_SOUL_PATH", "/Users/mozat/code-soul")
    if code_soul_path not in sys.path:
        sys.path.insert(0, code_soul_path)
    import types
    if "code_soul" not in sys.modules or not hasattr(sys.modules["code_soul"], "__path__"):
        mod = types.ModuleType("code_soul")
        mod.__path__ = [f"{code_soul_path}/code_soul"]
        sys.modules["code_soul"] = mod


def _agent_to_persona(agent: AgentProfile) -> Any:
    _ensure_code_soul_importable()
    from code_soul.playtest.persona import PersonaProfile
    return PersonaProfile(
        name=agent.agent_id,
        profile=agent.to_human_story(),
        goals=["complete the main task", "explore all visible features"],
    )


def extract_game_rules(gdd: dict) -> str:
    """Extract concise game rules from an ELTM GDD dict for playtest briefing."""
    rules = gdd.get("formal_rules") or gdd.get("game_rules") or {}
    if not rules:
        return ""
    parts = []
    if rules.get("objective"):
        parts.append(f"Objective: {rules['objective']}")
    if rules.get("turn_structure"):
        parts.append(f"How to play: {rules['turn_structure']}")
    if rules.get("win_conditions"):
        parts.append("Win conditions: " + "; ".join(rules["win_conditions"][:4]))
    special = rules.get("special_rules", [])
    important = [r for r in special[:8] if any(
        kw in r.lower() for kw in ["cannot", "must", "only", "first", "select", "click", "move", "capture", "jump"]
    )]
    if important:
        parts.append("Key rules: " + "; ".join(important[:5]))
    game_modes = gdd.get("game_modes", [])
    if game_modes:
        mode_names = [m.get("name", "") for m in game_modes if isinstance(m, dict)]
        if mode_names:
            parts.append("Game modes: " + ", ".join(mode_names[:4]))
    return "\n".join(parts)


def _make_llm_caller(backend: LLMBackend, game_rules: str = ""):
    def caller(system: str, user: str, image_b64: str | None, api_key: str) -> str:
        effective_system = system
        if game_rules:
            effective_system = (
                f"{system}\n\n"
                f"IMPORTANT — Game rules you must follow as a player:\n{game_rules}\n"
                f"Use this knowledge to make smart, rule-following moves. "
                f"Only click EMPTY cells/valid targets. Never click occupied positions."
            )
        prompt = f"{effective_system}\n\n{user}"
        if image_b64:
            img_bytes = base64.b64decode(image_b64)
            return backend.vision(prompt, [img_bytes], max_tokens=400, model_tier="smart")
        return backend.text(prompt, max_tokens=400, model_tier="fast")
    return caller


def run_user_playtest(
    html_path: str,
    personas: list[AgentProfile],
    backend: LLMBackend,
    *,
    k_turns: int = 12,
    on_progress=None,
    game_rules: str = "",
) -> PlaytestFeedback:
    _ensure_code_soul_importable()
    from code_soul.playtest.runner import run_playtest
    from code_soul.playtest.action_picker import pick_action

    cs_personas = [_agent_to_persona(a) for a in personas]
    llm_caller = _make_llm_caller(backend, game_rules=game_rules)

    def action_picker_with_backend(persona, screenshot, clickable, history, turn, k_turns, api_key=""):
        return pick_action(
            persona, screenshot, clickable, history, turn, k_turns,
            api_key=api_key, llm_caller=llm_caller,
        )

    friction_report = run_playtest(
        html_path=html_path,
        personas=cs_personas,
        k_turns=k_turns,
        on_progress=on_progress,
        action_picker=action_picker_with_backend,
    )

    return _translate_friction_report(friction_report, personas, backend)


def run_smoke_playtest(
    html_path: str,
    personas: list[AgentProfile],
    backend: LLMBackend,
    *,
    k_turns: int = 8,
    on_progress=None,
    game_rules: str = "",
) -> SmokePlaytestResult:
    """Mode A: lightweight bug scan with 1 persona, short session.

    Only counts bugs — no experience scoring. Output should converge
    with ELTM bug count across iterations.
    """
    scan_personas = personas[:1]
    feedback = run_user_playtest(
        html_path=html_path,
        personas=scan_personas,
        backend=backend,
        k_turns=k_turns,
        on_progress=on_progress,
        game_rules=game_rules,
    )
    bugs_only = [i for i in feedback.issues if i.category == "code_bug"]
    return SmokePlaytestResult(
        bug_count=len(bugs_only),
        bug_list=bugs_only,
        personas_completed=feedback.personas_completed,
        personas_total=feedback.personas_total,
        raw_summary=feedback.raw_summary,
    )


def _translate_friction_report(
    report: Any,
    personas: list[AgentProfile],
    backend: LLMBackend,
) -> PlaytestFeedback:
    completed = sum(1 for r in report.per_persona_results if r.completed)
    total = len(report.per_persona_results)

    grouped = _group_frictions(report.per_persona_results)
    issues = _classify_issues(grouped)

    if issues and backend:
        issues = _enrich_with_llm(issues, report.summary, backend)

    if report.score >= 80 and not any(i.severity == "P0" for i in issues):
        verdict = "PASS"
    elif report.score >= 50 or not any(i.severity == "P0" for i in issues):
        verdict = "NEEDS_WORK"
    else:
        verdict = "CRITICAL"

    suggestions = _extract_suggestions(report.per_persona_results)

    return PlaytestFeedback(
        score=report.score,
        verdict=verdict,
        issues=issues,
        suggestions=suggestions,
        personas_completed=completed,
        personas_total=total,
        raw_summary=report.summary,
    )


def _group_frictions(persona_results: list) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = {}
    for pr in persona_results:
        for ev in pr.friction_events:
            key = ev.kind
            if ev.kind == "wrong_action" and "Timeout" in ev.detail:
                key = "selector_timeout"
            if key not in groups:
                groups[key] = []
            groups[key].append({
                "persona": pr.persona,
                "turn": ev.turn,
                "detail": ev.detail[:200],
                "kind": ev.kind,
            })
    return groups


def _classify_issues(grouped: dict[str, list[dict]]) -> list[PlaytestIssue]:
    issues: list[PlaytestIssue] = []

    for kind, events in grouped.items():
        affected = list({e["persona"] for e in events})
        evidence = [f"turn {e['turn']} ({e['persona']}): {e['detail'][:100]}" for e in events[:5]]

        if kind == "error":
            issues.append(PlaytestIssue(
                severity="P0",
                description=f"JS console errors encountered ({len(events)} times)",
                evidence=evidence,
                affected_personas=affected,
                category="code_bug",
            ))
        elif kind == "give_up":
            reasons = [e["detail"] for e in events if e["detail"]]
            desc = reasons[0][:150] if reasons else "Persona gave up without explanation"
            issues.append(PlaytestIssue(
                severity="P1" if len(events) == 1 else "P0",
                description=f"User gave up: {desc}",
                evidence=evidence,
                affected_personas=affected,
                category="ux_friction",
            ))
        elif kind == "dead_end":
            issues.append(PlaytestIssue(
                severity="P1" if len(events) <= 2 else "P0",
                description=f"Dead-end detected — actions produce no response ({len(events)} times)",
                evidence=evidence,
                affected_personas=affected,
                category="code_bug",
            ))
        elif kind == "selector_timeout":
            issues.append(PlaytestIssue(
                severity="P1",
                description=f"UI elements not clickable — selector timeout ({len(events)} times)",
                evidence=evidence,
                affected_personas=affected,
                category="code_bug",
            ))
        elif kind == "wrong_action":
            issues.append(PlaytestIssue(
                severity="P2",
                description=f"Action execution failed ({len(events)} times)",
                evidence=evidence,
                affected_personas=affected,
                category="ux_friction",
            ))

    issues.sort(key=lambda i: {"P0": 0, "P1": 1, "P2": 2}.get(i.severity, 3))
    return issues


def _enrich_with_llm(
    issues: list[PlaytestIssue],
    raw_summary: str,
    backend: LLMBackend,
) -> list[PlaytestIssue]:
    issue_lines = []
    for i, iss in enumerate(issues):
        issue_lines.append(f"{i+1}. [{iss.severity}] {iss.description}")
        for ev in iss.evidence[:2]:
            issue_lines.append(f"   evidence: {ev}")

    prompt = (
        "你是一个产品经理，正在读用户测试报告。\n\n"
        f"测试摘要：\n{raw_summary}\n\n"
        f"发现的问题：\n" + "\n".join(issue_lines) + "\n\n"
        "请为每个问题写一句简洁的产品语言描述（不要技术术语），"
        "并判断每个问题属于哪类：code_bug（代码bug）、design_issue（设计问题）、ux_friction（体验摩擦）。\n\n"
        "输出 JSON 数组，每项：{\"index\": N, \"description\": \"...\", \"category\": \"...\"}\n"
        "只输出 JSON，不要其他文字。"
    )
    try:
        raw = backend.text(prompt, max_tokens=600, model_tier="fast")
        m = re.search(r'\[.*\]', raw, re.DOTALL)
        if m:
            enrichments = json.loads(m.group())
            for item in enrichments:
                idx = item.get("index", 0) - 1
                if 0 <= idx < len(issues):
                    if item.get("description"):
                        issues[idx].description = item["description"]
                    if item.get("category"):
                        issues[idx].category = item["category"]
    except Exception:
        pass
    return issues


def _extract_suggestions(persona_results: list) -> list[str]:
    suggestions = []
    for pr in persona_results:
        if pr.gave_up:
            for ev in pr.friction_events:
                if ev.kind == "give_up" and ev.detail:
                    suggestions.append(ev.detail[:200])
    return suggestions[:5]


# ---------------------------------------------------------------------------
# Knowledge-Graded Playtest
# ---------------------------------------------------------------------------

def run_graded_playtest(
    html_path: str,
    personas: list[AgentProfile],
    backend: LLMBackend,
    gdd: dict,
    *,
    k_turns: int = 12,
    on_progress=None,
    tiers: list[KnowledgeTier] | None = None,
) -> GradedPlaytestFeedback:
    tiers = tiers or KnowledgeTier.all_tiers()
    knowledge = GameKnowledge.from_gdd(gdd)

    min_needed = minimum_personas_for_graded_test(len(tiers))
    if len(personas) < min_needed:
        raise ValueError(
            f"Need at least {min_needed} personas for {len(tiers)} tiers, "
            f"got {len(personas)}"
        )

    tier_personas = distribute_personas_across_tiers(personas, tiers)

    tier_feedbacks: dict[str, PlaytestFeedback] = {}
    tier_results: dict[str, TierResult] = {}

    for tier in tiers:
        tier_pool = tier_personas[tier.value]
        if not tier_pool:
            continue

        briefing = brief_for_tier(knowledge, tier)

        def _tier_progress(event, _tier=tier):
            if on_progress:
                event["tier"] = _tier.value
                on_progress(event)

        feedback = run_user_playtest(
            html_path=html_path,
            personas=tier_pool,
            backend=backend,
            k_turns=k_turns,
            on_progress=_tier_progress,
            game_rules=briefing,
        )
        tier_feedbacks[tier.value] = feedback

        total = feedback.personas_total or 1
        gave_up_count = total - feedback.personas_completed
        failure_turns = _extract_failure_turns(feedback)
        issue_kinds = _count_issue_kinds(feedback)

        tier_results[tier.value] = TierResult(
            tier=tier,
            score=feedback.score,
            completed_rate=feedback.personas_completed / total,
            gave_up_rate=gave_up_count / total,
            friction_count=sum(issue_kinds.values()),
            issue_kinds=issue_kinds,
            failure_turns=failure_turns,
            raw_issues=list(feedback.issues),
        )

    diagnosis = DifferentialDiagnosis.from_tier_results(
        game_name=knowledge.game_name,
        tier_results=tier_results,
    )

    weights = {
        KnowledgeTier.NOVICE.value: 0.2,
        KnowledgeTier.CASUAL.value: 0.3,
        KnowledgeTier.INFORMED.value: 0.5,
    }
    total_weight = sum(weights.get(t, 0.33) for t in tier_feedbacks)
    aggregate_score = sum(
        fb.score * weights.get(t, 0.33)
        for t, fb in tier_feedbacks.items()
    ) / max(total_weight, 0.01)

    if any(d.severity == "P0" for d in diagnosis.diagnoses):
        aggregate_verdict = "CRITICAL"
    elif aggregate_score >= 70:
        aggregate_verdict = "PASS"
    else:
        aggregate_verdict = "NEEDS_WORK"

    all_issues = []
    for tier_val, fb in tier_feedbacks.items():
        for issue in fb.issues:
            tagged = PlaytestIssue(
                severity=issue.severity,
                description=f"[{tier_val.upper()}] {issue.description}",
                evidence=issue.evidence,
                affected_personas=issue.affected_personas,
                category=issue.category,
            )
            all_issues.append(tagged)

    all_suggestions = []
    for fb in tier_feedbacks.values():
        all_suggestions.extend(fb.suggestions)

    return GradedPlaytestFeedback(
        score=aggregate_score,
        verdict=aggregate_verdict,
        issues=all_issues,
        suggestions=all_suggestions[:10],
        personas_completed=sum(fb.personas_completed for fb in tier_feedbacks.values()),
        personas_total=sum(fb.personas_total for fb in tier_feedbacks.values()),
        raw_summary=diagnosis.summary,
        tier_feedbacks=tier_feedbacks,
        diagnosis=diagnosis,
    )


def _extract_failure_turns(feedback: PlaytestFeedback) -> list[int]:
    turns = []
    for issue in feedback.issues:
        for ev in issue.evidence:
            if ev.startswith("turn "):
                try:
                    turn_str = ev.split("(")[0].replace("turn", "").strip()
                    turns.append(int(turn_str))
                except (ValueError, IndexError):
                    pass
    return sorted(set(turns))


def _count_issue_kinds(feedback: PlaytestFeedback) -> dict[str, int]:
    counts: dict[str, int] = {}
    for issue in feedback.issues:
        cat = issue.category or "unknown"
        counts[cat] = counts.get(cat, 0) + 1
    return counts
