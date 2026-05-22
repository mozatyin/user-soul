"""BehaviorEngine — domain-agnostic behavioral simulation, journey testing, and reporting."""
from __future__ import annotations

import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any

from user_soul.backend import LLMBackend
from user_soul.models import (
    AgentProfile, EvaluationMetric, SessionResult,
    MetricResult, SimulationReport, CompareReport, JourneyReport,
)
from user_soul.framework import (
    BEHAVIORAL_FRAMEWORK_SECTION, ADVERSARIAL_PERSONA_SECTION,
    BEHAVIORAL_METRICS, COGNITIVE_BUDGETS,
)


# ---------------------------------------------------------------------------
# DomainConfig (public — external callers need it)
# ---------------------------------------------------------------------------

@dataclass
class DomainConfig:
    session_framing: str
    emotional_states: list[str]
    triggers: list[str]
    time_options: list[str]
    user_roles: dict[str, list[int]]


_GAME_CONFIG = DomainConfig(
    session_framing="你开始了一局游戏",
    emotional_states=["competitive", "casual", "tilted", "bored", "hyped"],
    triggers=["want_to_rank_up", "friend_challenged_me", "kill_time", "daily_login", "revenge_match"],
    time_options=["morning_commute", "lunch_break", "evening", "late_night"],
    user_roles={"Newcomer": [1, 3], "Casual": [3, 14], "Grinder": [14, 30], "Veteran": [30]},
)

_APP_CONFIG = DomainConfig(
    session_framing="你打开了这个 app",
    emotional_states=["stressed", "calm", "bored", "excited", "sad", "anxious"],
    triggers=["habit", "work_stress", "relationship_tension", "boredom", "notification", "curiosity"],
    time_options=["morning_commute", "lunch_break", "evening_wind_down", "night"],
    user_roles={"Explorer": [1, 3], "Skeptic": [3, 7], "Habituer": [14, 30], "Advocate": [30]},
)


# ---------------------------------------------------------------------------
# ScenarioContext (internal)
# ---------------------------------------------------------------------------

@dataclass
class _ScenarioContext:
    time_of_day: str
    emotional_state: str
    usage_day: int
    trigger: str


def _random_context_for_domain(
    role: str | None,
    domain_config: DomainConfig,
) -> _ScenarioContext:
    time_of_day = random.choice(domain_config.time_options)
    emotional_state = random.choice(domain_config.emotional_states)
    if role and role in domain_config.user_roles:
        usage_day = random.choice(domain_config.user_roles[role])
    else:
        all_days = [d for days in domain_config.user_roles.values() for d in days]
        usage_day = random.choice(all_days) if all_days else 1
    trigger = random.choice(domain_config.triggers)
    return _ScenarioContext(
        time_of_day=time_of_day,
        emotional_state=emotional_state,
        usage_day=usage_day,
        trigger=trigger,
    )


# ---------------------------------------------------------------------------
# Session prompt + parsing (from user_simulator.py)
# ---------------------------------------------------------------------------

def _build_session_prompt(
    user_type: str,
    context: _ScenarioContext,
    product: str,
    metrics: list[EvaluationMetric],
    domain_config: DomainConfig,
    screen_id: str | None = None,
    framework_section: str = "",
) -> str:
    product_section = product
    if screen_id:
        product_section = f"[只关注 screen_id='{screen_id}' 的部分]\n{product}"

    metric_lines = "\n".join(
        f"{m.name}: {m.question}"
        + (" (回答 yes 或 no)" if m.type == "bool"
           else " (回答 1-5 的数字)" if m.type == "scale_1_5"
           else " (简短文字)")
        for m in metrics
    )

    return (
        f"你是：{user_type}\n\n"
        f"现在的情况：\n"
        f"  时间：{context.time_of_day.replace('_', ' ')}\n"
        f"  状态：{context.emotional_state}\n"
        f"  触发：{context.trigger.replace('_', ' ')}\n"
        f"  使用天数：{context.usage_day}\n\n"
        + (f"{framework_section}\n" if framework_section else "")
        + f"{domain_config.session_framing}：\n{product_section}\n\n"
        f"叙述你接下来的 6-8 个操作。只写你做了什么，不写感想。\n"
        f"用第三人称叙述行为，比如：\"他点击了...\"，\"他滑过了...\"\n\n"
        f"叙述完毕后，每行回答一个问题：\n{metric_lines}\n"
    )


def _parse_session_output(raw: str, metrics: list[EvaluationMetric]) -> dict[str, str]:
    values: dict[str, str] = {}
    for metric in metrics:
        pattern = rf"^{re.escape(metric.name)}\s*[:：]\s*(.+)$"
        for line in raw.splitlines():
            clean = re.sub(r'^[*#>\s]+', '', line.strip())
            m = re.match(pattern, clean, re.IGNORECASE)
            if m:
                value = re.sub(r'^\*+|\*+$', '', m.group(1).strip()).strip()
                values[metric.name] = value
                break
    return values


# ---------------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Report aggregation (from report.py)
# ---------------------------------------------------------------------------

def _wilson_ci(p: float, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    margin = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / denom
    return round(max(0.0, center - margin), 4), round(min(1.0, center + margin), 4)


def _aggregate_bool(values: list[str]) -> MetricResult:
    if not values:
        return MetricResult(name="", type="bool", true_rate=0.0,
                            stdev=0.0, ci_95_low=0.0, ci_95_high=0.0, n_samples=0)
    true_count = sum(
        1 for v in values
        if v.lower().strip() in ("yes", "true", "1", "是", "会", "会的", "y")
    )
    n = len(values)
    p = round(true_count / n, 4)
    lo, hi = _wilson_ci(p, n)
    stdev = round(math.sqrt(p * (1 - p)), 4)
    return MetricResult(name="", type="bool", true_rate=p,
                        stdev=stdev, ci_95_low=lo, ci_95_high=hi, n_samples=n)


def _aggregate_scale(values: list[str]) -> MetricResult:
    nums = []
    for v in values:
        m = re.search(r'(?<!\d)([1-5])(?!\d)', v)
        if m:
            nums.append(int(m.group(1)))
    if not nums:
        return MetricResult(name="", type="scale_1_5", mean=0.0, distribution={},
                            stdev=0.0, ci_95_low=0.0, ci_95_high=0.0, n_samples=0)
    n = len(nums)
    mean = round(sum(nums) / n, 4)
    dist = {i: round(nums.count(i) / n, 4) for i in range(1, 6) if nums.count(i) > 0}
    stdev = round(statistics.stdev(nums), 4) if n > 1 else 0.0
    margin = round(1.96 * stdev / math.sqrt(n), 4) if n > 1 else 0.0
    return MetricResult(name="", type="scale_1_5", mean=mean, distribution=dist,
                        stdev=stdev,
                        ci_95_low=round(max(1.0, mean - margin), 4),
                        ci_95_high=round(min(5.0, mean + margin), 4),
                        n_samples=n)


def _aggregate_text(values: list[str], backend: LLMBackend | None = None) -> MetricResult:
    samples = values[:10]
    themes: list[str] = []
    if backend and len(values) >= 3:
        joined = "\n".join(f"- {v}" for v in values[:30])
        prompt = (
            f"以下是用户反馈列表：\n{joined}\n\n"
            "提取 3-5 个主要主题，用简短短语表示。\n"
            '只输出 JSON 数组：["主题1", "主题2", ...]'
        )
        try:
            raw = backend.text(prompt, max_tokens=256, model_tier="fast")
            m = re.search(r'\[.*\]', raw, re.DOTALL)
            if m:
                parsed = json.loads(m.group())
                themes = [str(t) for t in parsed if isinstance(t, str)][:5]
        except Exception:
            pass
    return MetricResult(name="", type="text", themes=themes, samples=samples)


def _generate_key_findings(
    metrics: dict[str, MetricResult],
    user_type: str,
    backend: LLMBackend,
) -> str:
    lines = []
    for mr in metrics.values():
        if mr.type == "bool" and mr.true_rate is not None \
                and mr.ci_95_low is not None and mr.ci_95_high is not None:
            lines.append(
                f"{mr.name}: true_rate={mr.true_rate:.0%} "
                f"(n={mr.n_samples}, CI [{mr.ci_95_low:.0%}–{mr.ci_95_high:.0%}])"
            )
        elif mr.type == "scale_1_5" and mr.mean is not None and mr.stdev is not None:
            lines.append(
                f"{mr.name}: mean={mr.mean:.1f}/5 (stdev={mr.stdev:.2f}, n={mr.n_samples})"
            )
        elif mr.type == "text" and mr.themes:
            lines.append(f"{mr.name} themes: {', '.join(mr.themes[:3])}")
    if not lines:
        return ""
    summary = "\n".join(lines)
    prompt = (
        f"用户类型：{user_type}\n"
        f"模拟指标结果：\n{summary}\n\n"
        "用 2-3 句话总结最重要的产品发现。直接写洞察，不要重复数字。"
    )
    raw = backend.text(prompt, max_tokens=200, model_tier="fast")
    return raw.strip()


def _aggregate(
    session_results: list[SessionResult],
    metrics: list[EvaluationMetric],
    user_type: str,
    product_summary: str,
    backend: LLMBackend | None = None,
    adversarial_results: list[SessionResult] | None = None,
) -> SimulationReport:
    metric_values: dict[str, list[str]] = defaultdict(list)
    for sr in session_results:
        for name, value in sr.values.items():
            metric_values[name].append(value)

    results: dict[str, MetricResult] = {}
    for metric in metrics:
        vals = metric_values.get(metric.name, [])
        if metric.type == "bool":
            r = _aggregate_bool(vals)
        elif metric.type == "scale_1_5":
            r = _aggregate_scale(vals)
        else:
            r = _aggregate_text(vals, backend)
        r.name = metric.name
        results[metric.name] = r

    key_findings = ""
    if backend and len(results) >= 2:
        try:
            key_findings = _generate_key_findings(results, user_type, backend)
        except Exception:
            pass

    adversarial_frictions: list[str] = []
    if adversarial_results:
        _CHURN_VALS = {"no", "false", "0", "否", "不会", "不", "n"}
        adv_texts: list[str] = []
        for sr in adversarial_results:
            return_val = sr.values.get("day_1_return_intent", "").lower().strip()
            if return_val not in _CHURN_VALS:
                continue
            for name, val in sr.values.items():
                if name == "day_1_return_intent":
                    continue
                if val and len(val) > 10:
                    adv_texts.append(val)
        if adv_texts:
            adv_mr = _aggregate_text(adv_texts, backend)
            adversarial_frictions = adv_mr.themes or adv_texts[:5]

    return SimulationReport(
        n_simulations=len(session_results),
        user_type=user_type,
        product_summary=product_summary,
        metrics=results,
        key_findings=key_findings,
        adversarial_frictions=adversarial_frictions,
        sessions=list(session_results),
        _metrics_list=metrics,
    )


def _compute_compare(
    report_a: SimulationReport,
    report_b: SimulationReport,
    label_a: str,
    label_b: str,
    key_diff: str = "",
) -> CompareReport:
    deltas: dict[str, float] = {}
    improvements: list[str] = []
    regressions: list[str] = []

    for name, mr_a in report_a.metrics.items():
        mr_b = report_b.metrics.get(name)
        if mr_b is None:
            continue

        ci_width = (
            ((mr_a.ci_95_high or 0.0) - (mr_a.ci_95_low or 0.0))
            if (mr_a.ci_95_low is not None and mr_a.ci_95_high is not None)
            else 0.0
        )
        threshold = ci_width / 2 if ci_width > 0 else float("inf")

        if mr_a.type == "bool" and mr_a.true_rate is not None and mr_b.true_rate is not None:
            delta = round(mr_b.true_rate - mr_a.true_rate, 4)
            deltas[name] = delta
            if delta > threshold:
                improvements.append(name)
            elif delta < -threshold:
                regressions.append(name)

        elif mr_a.type == "scale_1_5" and mr_a.mean is not None and mr_b.mean is not None:
            delta = round(mr_b.mean - mr_a.mean, 4)
            deltas[name] = delta
            if delta > threshold:
                improvements.append(name)
            elif delta < -threshold:
                regressions.append(name)

    return CompareReport(
        n_runs_per_variant=report_a.n_simulations,
        variant_a_label=label_a,
        variant_b_label=label_b,
        variant_a=report_a,
        variant_b=report_b,
        deltas=deltas,
        improvements=improvements,
        regressions=regressions,
        key_diff=key_diff,
    )


# ---------------------------------------------------------------------------
# Journey helpers (from journey.py)
# ---------------------------------------------------------------------------

def _normalise_screens(screens: list[dict] | dict) -> list[dict]:
    if isinstance(screens, dict):
        return list(screens.values())
    return list(screens)


def _screen_index(screens: list[dict]) -> dict[str, dict]:
    return {s.get("screen_id", ""): s for s in screens if s.get("screen_id")}


def _parse_step_output(raw: str) -> tuple[bool, str, str]:
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
# DomainConfig builder helpers
# ---------------------------------------------------------------------------

def _domain_config_from_dict(data: dict) -> DomainConfig:
    emotional_states = data.get("emotional_states", _APP_CONFIG.emotional_states)
    triggers = data.get("triggers", _APP_CONFIG.triggers)
    time_options = data.get("time_options", _APP_CONFIG.time_options)
    session_framing = data.get("session_framing", _APP_CONFIG.session_framing)
    raw_roles = data.get("user_roles", {})

    user_roles: dict[str, list[int]] = {}
    for role, val in raw_roles.items():
        if isinstance(val, (list, tuple)) and len(val) >= 1:
            user_roles[str(role)] = [int(v) for v in val[:2]]
    if not user_roles:
        user_roles = dict(_APP_CONFIG.user_roles)

    if not isinstance(emotional_states, list) or not emotional_states:
        emotional_states = list(_APP_CONFIG.emotional_states)
    if not isinstance(triggers, list) or not triggers:
        triggers = list(_APP_CONFIG.triggers)
    if not isinstance(time_options, list) or not time_options:
        time_options = list(_APP_CONFIG.time_options)

    return DomainConfig(
        session_framing=str(session_framing),
        emotional_states=[str(s) for s in emotional_states],
        triggers=[str(t) for t in triggers],
        time_options=[str(o) for o in time_options],
        user_roles=user_roles,
    )


# ---------------------------------------------------------------------------
# BehaviorEngine
# ---------------------------------------------------------------------------

class BehaviorEngine:

    def __init__(self, backend: LLMBackend):
        self._backend = backend

    def simulate(
        self,
        product: str,
        personas: list[AgentProfile],
        metrics: list[EvaluationMetric],
        *,
        n_runs: int = 30,
        adversarial: bool = True,
        domain_config: DomainConfig | None = None,
    ) -> SimulationReport:
        cfg = domain_config or _APP_CONFIG

        all_metrics = list(metrics)
        if adversarial:
            existing_names = {m.name for m in all_metrics}
            for bname, btype, bquestion in BEHAVIORAL_METRICS:
                if bname not in existing_names:
                    all_metrics.append(EvaluationMetric(bname, btype, bquestion))

        session_results: list[SessionResult] = []
        roles = list(cfg.user_roles.keys())

        for i in range(n_runs):
            agent = personas[i % len(personas)]
            user_type_text = agent.to_human_story()
            role = roles[i % len(roles)] if roles else None

            ctx = _random_context_for_domain(role, cfg)
            prompt = _build_session_prompt(
                user_type=user_type_text,
                context=ctx,
                product=product,
                metrics=all_metrics,
                domain_config=cfg,
            )
            raw = self._backend.text(prompt, max_tokens=800,
                                     temperature=1.0, model_tier="fast")
            values = _parse_session_output(raw, all_metrics)
            session_results.append(SessionResult(
                scenario=ctx, narrative=raw, values=values,
                agent_id=agent.agent_id,
            ))

        adversarial_results: list[SessionResult] = []
        if adversarial:
            adv_n = max(1, n_runs // 5)
            adv_framework = BEHAVIORAL_FRAMEWORK_SECTION.format(
                cognitive_budget=COGNITIVE_BUDGETS.get("adversarial", 6)
            )
            for j in range(adv_n):
                ctx = _random_context_for_domain(
                    roles[j % len(roles)] if roles else None, cfg,
                )
                prompt = _build_session_prompt(
                    user_type=ADVERSARIAL_PERSONA_SECTION,
                    context=ctx,
                    product=product,
                    metrics=all_metrics,
                    domain_config=cfg,
                    framework_section=adv_framework,
                )
                raw = self._backend.text(prompt, max_tokens=800,
                                         temperature=1.0, model_tier="fast")
                values = _parse_session_output(raw, all_metrics)
                adversarial_results.append(SessionResult(
                    scenario=ctx, narrative=raw, values=values,
                ))

        user_type = personas[0].archetype_name if personas else "user"
        return _aggregate(
            session_results, all_metrics, user_type, product[:100],
            backend=self._backend,
            adversarial_results=adversarial_results if adversarial else None,
        )

    def compare(
        self,
        product_a: str,
        product_b: str,
        personas: list[AgentProfile],
        metrics: list[EvaluationMetric],
        *,
        n_runs: int = 30,
        domain_config: DomainConfig | None = None,
    ) -> CompareReport:
        cfg = domain_config or _APP_CONFIG
        roles = list(cfg.user_roles.keys())

        contexts = [
            _random_context_for_domain(
                roles[i % len(roles)] if roles else None, cfg,
            )
            for i in range(n_runs)
        ]

        def _run_variant(product: str) -> list[SessionResult]:
            results = []
            for idx, ctx in enumerate(contexts):
                agent = personas[idx % len(personas)]
                prompt = _build_session_prompt(
                    user_type=agent.to_human_story(),
                    context=ctx,
                    product=product,
                    metrics=metrics,
                    domain_config=cfg,
                )
                raw = self._backend.text(prompt, max_tokens=800,
                                         temperature=1.0, model_tier="fast")
                values = _parse_session_output(raw, metrics)
                results.append(SessionResult(scenario=ctx, narrative=raw, values=values,
                                             agent_id=agent.agent_id))
            return results

        sessions_a = _run_variant(product_a)
        sessions_b = _run_variant(product_b)

        user_type = personas[0].archetype_name if personas else "user"
        report_a = _aggregate(sessions_a, metrics, user_type, product_a[:100],
                              backend=self._backend)
        report_b = _aggregate(sessions_b, metrics, user_type, product_b[:100],
                              backend=self._backend)

        key_diff = ""
        try:
            delta_lines = []
            for name, mr_a in report_a.metrics.items():
                mr_b = report_b.metrics.get(name)
                if mr_b and mr_a.type == "bool" and mr_a.true_rate is not None and mr_b.true_rate is not None:
                    delta_lines.append(f"{name}: {mr_a.true_rate:.0%} → {mr_b.true_rate:.0%}")
                elif mr_b and mr_a.type == "scale_1_5" and mr_a.mean is not None and mr_b.mean is not None:
                    delta_lines.append(f"{name}: {mr_a.mean:.1f} → {mr_b.mean:.1f}")
            if delta_lines:
                prompt = (
                    f"版本对比 (v_a vs v_b):\n"
                    + "\n".join(delta_lines)
                    + "\n\n用一句话说明哪个版本更好，差异是否显著。"
                )
                raw = self._backend.text(prompt, max_tokens=100, model_tier="fast")
                key_diff = raw.strip()
        except Exception:
            pass

        return _compute_compare(report_a, report_b, "v_a", "v_b", key_diff=key_diff)

    def simulate_journey(
        self,
        screens: list[dict] | dict,
        target_flow: list[str],
        personas: list[AgentProfile],
        *,
        n_personas: int = 12,
    ) -> JourneyReport:
        if len(target_flow) <= 1:
            n = min(n_personas, len(personas))
            return JourneyReport(
                target_flow=target_flow,
                completion_rate=1.0,
                drop_off_by_screen={},
                fogg_violations=[],
                blocked_journeys=[],
                personas_completed=n,
                personas_total=n,
            )

        screens_list = _normalise_screens(screens)
        idx = _screen_index(screens_list)
        n = min(n_personas, len(personas))
        final_screen = target_flow[-1]

        @dataclass
        class _PersonaResult:
            completed: bool = False
            drop_off_screen: str | None = None
            drop_off_reason: str = ""
            fogg_issue: str = "none"

        results: list[_PersonaResult] = []
        blocked_journeys: list[str] = []

        for i in range(n):
            agent = personas[i % len(personas)]
            persona_story = agent.to_human_story()
            pr = _PersonaResult()

            for step_idx in range(len(target_flow) - 1):
                current_id = target_flow[step_idx]
                next_id = target_flow[step_idx + 1]
                current_screen = idx.get(current_id, {"screen_id": current_id, "navigates_to": []})
                navigates_to = current_screen.get("navigates_to") or []

                if navigates_to and next_id not in navigates_to:
                    pr.drop_off_screen = current_id
                    pr.drop_off_reason = f"架构阻断：{current_id} 无法导航到 {next_id}"
                    pr.fogg_issue = "ability"
                    if len(blocked_journeys) < 5:
                        blocked_journeys.append(
                            f"Persona {i+1} 在 {current_id} 被架构阻断（navigates_to 不包含 {next_id}）"
                        )
                    break

                prompt = _build_step_prompt(persona_story, current_screen, next_id, final_screen)
                raw = self._backend.text(prompt, max_tokens=200,
                                         temperature=1.0, model_tier="fast")
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
        fogg_violations = list(dict.fromkeys(fogg_raw))

        return JourneyReport(
            target_flow=target_flow,
            completion_rate=completion_rate,
            drop_off_by_screen=dict(drop_off_counts),
            fogg_violations=fogg_violations,
            blocked_journeys=blocked_journeys,
            personas_completed=personas_completed,
            personas_total=n,
        )

    def extract_metrics(self, goal: str) -> list[EvaluationMetric]:
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
        raw = self._backend.text(prompt, max_tokens=512, model_tier="smart")
        items = _safe_json_arr(raw)
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

    def build_domain_config(self, product_description: str) -> DomainConfig:
        prompt = (
            "You are a UX researcher. For the following product, define a behavioral simulation config.\n\n"
            f"Product: {product_description[:1500]}\n\n"
            "Return ONLY valid JSON (no markdown):\n"
            '{"session_framing": "你在...", '
            '"emotional_states": ["state1", "state2", "state3", "state4", "state5"], '
            '"triggers": ["trigger1", "trigger2", "trigger3", "trigger4", "trigger5"], '
            '"time_options": ["morning_commute", "lunch_break", "evening", "late_night"], '
            '"user_roles": {"RoleName": [day_min, day_max]}}\n\n'
            "Rules:\n"
            "- session_framing: Chinese, starts with 你在/你打开了/你开始了\n"
            "- emotional_states: 4-6 states specific to this product domain\n"
            "- triggers: 4-6 triggers that bring users to this product\n"
            "- time_options: 3-4 time slots\n"
            "- user_roles: 3-4 named lifecycle stages with [min_day, max_day] usage ranges"
        )
        raw = self._backend.text(prompt, max_tokens=512, model_tier="smart")
        data = _safe_json(raw)
        if data:
            return _domain_config_from_dict(data)
        return DomainConfig(
            session_framing=_APP_CONFIG.session_framing,
            emotional_states=list(_APP_CONFIG.emotional_states),
            triggers=list(_APP_CONFIG.triggers),
            time_options=list(_APP_CONFIG.time_options),
            user_roles=dict(_APP_CONFIG.user_roles),
        )
