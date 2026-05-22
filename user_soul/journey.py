"""Gate 2: Journey simulation — validate that personas can complete a screen flow.

Equivalent to GV Design Sprint Day 4 (prototype test before code).
Runs BEFORE M4 generates HTML, surfaces flow-blocking issues cheaply.

Evidence base:
- Elicitron (Stanford/Autodesk 2024): LLM simulation finds 2x latent needs vs human interviews
- Wang & Siu (CHI 2026): agents are distribution-calibrated — reliable for population-level flow validation
- Boehm's curve: finding flow errors at design (3-6x) vs after Code-Soul (25x)
- GV Design Sprint: user testing at wireframe stage = highest ROI intervention point
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Data contracts
# ---------------------------------------------------------------------------

@dataclass
class JourneyReport:
    """Result of simulating N personas through a target screen flow."""
    target_flow: list[str]
    completion_rate: float              # 0.0–1.0
    drop_off_by_screen: dict            # {screen_id: count}
    fogg_violations: list[str]          # deduplicated Fogg BM failure dimensions
    blocked_journeys: list[str]         # up to 5 narrative failure descriptions
    personas_completed: int
    personas_total: int

    @property
    def passes_gate(self) -> bool:
        """True if completion_rate >= 0.70 (GV Sprint threshold)."""
        return self.completion_rate >= 0.70

    @property
    def benchmark_context(self) -> str:
        r = self.completion_rate
        if r >= 0.85:
            return f"Strong ({r:.0%}) — above 85% excellence threshold"
        if r >= 0.70:
            return f"Acceptable ({r:.0%}) — above 70% gate threshold"
        if r >= 0.50:
            return f"Weak ({r:.0%}) — below 70% gate, flow needs redesign"
        return f"Critical ({r:.0%}) — majority cannot complete flow"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _normalise_screens(screens: list[dict] | dict) -> list[dict]:
    """Accept M2 list format or M5 dict format, return list[dict]."""
    if isinstance(screens, dict):
        return list(screens.values())
    return list(screens)


def _screen_index(screens: list[dict]) -> dict[str, dict]:
    """Build {screen_id: screen_dict} lookup."""
    return {s.get("screen_id", ""): s for s in screens if s.get("screen_id")}


def _parse_step_output(raw: str) -> tuple[bool, str, str]:
    """Parse LLM step output → (proceed: bool, reason: str, fogg_issue: str).

    Returns (False, '', 'unknown') if unparseable — conservative default
    treats ambiguous output as a drop-off.
    """
    proceed = False
    reason = ""
    fogg_issue = "unknown"

    for line in raw.splitlines():
        clean = line.strip().lstrip("*#> ")
        m = re.match(r"proceed\s*[:：]\s*(.+)", clean, re.IGNORECASE)
        if m:
            val = m.group(1).strip().lower()
            proceed = val in ("yes", "是", "y", "true", "1")
        m = re.match(r"reason\s*[:：]\s*(.+)", clean, re.IGNORECASE)
        if m:
            reason = m.group(1).strip()
        m = re.match(r"fogg_issue\s*[:：]\s*(.+)", clean, re.IGNORECASE)
        if m:
            fogg_issue = m.group(1).strip().lower()

    return proceed, reason, fogg_issue


def _build_step_prompt(
    persona_story: str,
    current_screen: dict,
    next_screen_id: str,
    final_screen_id: str,
) -> str:
    screen_id = current_screen.get("screen_id", "?")
    description = current_screen.get("description") or current_screen.get("screen_name", "")
    navigates_to = current_screen.get("navigates_to") or []
    nav_str = "、".join(str(s) for s in navigates_to) if navigates_to else "（无导航出口）"

    return (
        f"你是：{persona_story}\n\n"
        f"你正在使用一个App，当前界面：{screen_id}（{description}）\n"
        f"该界面可以前往：{nav_str}\n\n"
        f"你的目标是最终到达：{final_screen_id}\n"
        f"下一步需要前往：{next_screen_id}\n\n"
        f"判断你是否会继续前往 {next_screen_id}，还是放弃这个App。\n\n"
        f"每行回答一个问题：\n"
        f"proceed: yes 或 no\n"
        f"reason: 一句话原因\n"
        f"fogg_issue: 如果proceed=no，填 motivation/ability/trigger 之一；否则填 none\n"
    )


# ---------------------------------------------------------------------------
# Core function
# ---------------------------------------------------------------------------

def simulate_journey(
    screens: list[dict] | dict,
    target_flow: list[str],
    persona_pool: list,
    api_key: str,
    n_personas: int = 12,
) -> JourneyReport:
    """Gate 2: simulate whether personas can complete target_flow through the screens.

    Args:
        screens: M2 canonical_screens (list) or M5 contract.json screens (dict).
        target_flow: ordered screen_ids to traverse, e.g. ["home","treasure","collect"].
        persona_pool: list[AgentProfile] from PersonaPool.generate().
        api_key: Anthropic/OpenRouter API key.
        n_personas: how many personas to simulate (capped at len(persona_pool)).

    Returns:
        JourneyReport. Check report.passes_gate (>=0.70) before proceeding to M4.
    """
    import user_soul.core as _core

    if len(target_flow) <= 1:
        return JourneyReport(
            target_flow=target_flow,
            completion_rate=1.0,
            drop_off_by_screen={},
            fogg_violations=[],
            blocked_journeys=[],
            personas_completed=min(n_personas, len(persona_pool)),
            personas_total=min(n_personas, len(persona_pool)),
        )

    screens_list = _normalise_screens(screens)
    idx = _screen_index(screens_list)
    n = min(n_personas, len(persona_pool))
    final_screen = target_flow[-1]

    # Per-persona results
    @dataclass
    class _PersonaResult:
        completed: bool = False
        drop_off_screen: str | None = None
        drop_off_reason: str = ""
        fogg_issue: str = "none"

    results: list[_PersonaResult] = []
    blocked_journeys: list[str] = []

    for i in range(n):
        agent = persona_pool[i % len(persona_pool)]
        persona_story = agent.to_human_story()
        pr = _PersonaResult()

        for step_idx in range(len(target_flow) - 1):
            current_id = target_flow[step_idx]
            next_id = target_flow[step_idx + 1]
            current_screen = idx.get(current_id, {"screen_id": current_id, "navigates_to": []})
            navigates_to = current_screen.get("navigates_to") or []

            # Architecture block: next screen not reachable from current
            if navigates_to and next_id not in navigates_to:
                pr.drop_off_screen = current_id
                pr.drop_off_reason = f"架构阻断：{current_id} 无法导航到 {next_id}"
                pr.fogg_issue = "ability"
                if len(blocked_journeys) < 5:
                    blocked_journeys.append(
                        f"Persona {i+1} 在 {current_id} 被架构阻断（navigates_to 不包含 {next_id}）"
                    )
                break

            # LLM step simulation
            prompt = _build_step_prompt(persona_story, current_screen, next_id, final_screen)
            raw, _ = _core._llm_call(
                prompt, api_key,
                max_tokens=200, temperature=1.0,
                model=_core._haiku_model(api_key),
            )
            proceed, reason, fogg_issue = _parse_step_output(raw)

            if not proceed:
                pr.drop_off_screen = current_id
                pr.drop_off_reason = reason
                pr.fogg_issue = fogg_issue
                if len(blocked_journeys) < 5:
                    blocked_journeys.append(
                        f"Persona {i+1} 在 {current_id}→{next_id} 放弃：{reason}"
                    )
                break
        else:
            pr.completed = True

        results.append(pr)

    personas_completed = sum(1 for r in results if r.completed)
    completion_rate = round(personas_completed / n, 4) if n > 0 else 0.0

    drop_off_counts = Counter(r.drop_off_screen for r in results if r.drop_off_screen)
    fogg_raw = [r.fogg_issue for r in results if not r.completed and r.fogg_issue not in ("none", "unknown", "")]
    fogg_violations = list(dict.fromkeys(fogg_raw))  # deduplicated, order-preserving

    return JourneyReport(
        target_flow=target_flow,
        completion_rate=completion_rate,
        drop_off_by_screen=dict(drop_off_counts),
        fogg_violations=fogg_violations,
        blocked_journeys=blocked_journeys,
        personas_completed=personas_completed,
        personas_total=n,
    )
