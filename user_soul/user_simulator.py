"""UserSimulator — domain-agnostic behavioral simulation engine."""
from __future__ import annotations

import re
import random
from dataclasses import dataclass, field

from user_soul.scenarios import ScenarioContext
from user_soul.scenarios import random_context_for_domain as _random_context_for_domain
from user_soul.schema_extractor import EvaluationMetric
from user_soul.domain_configs import DomainConfig


@dataclass
class SessionResult:
    """One simulated user session — narrative + extracted metric values."""
    scenario: ScenarioContext
    narrative: str
    values: dict[str, str] = field(default_factory=dict)   # {metric_name: raw_string}


def _build_session_prompt(
    user_type: str,
    context: ScenarioContext,
    product: str,
    metrics: list[EvaluationMetric],
    domain_config: DomainConfig,
    screen_id: str | None = None,
    framework_section: str = "",
) -> str:
    """Build the per-session LLM prompt from user type, context, product, and metrics.

    The prompt instructs third-person behavior narration followed by one answer
    per metric. screen_id optionally focuses the session on a specific wireframe screen.
    """
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
    """Extract metric values from raw session output.

    Strips leading markdown formatting (*, #, >) before matching so that LLM
    output like '**metric_name:** value' is handled correctly.
    """
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


class UserSimulator:
    """Domain-agnostic behavioral simulation engine.

    Usage:
        sim = UserSimulator("18岁手游玩家", GameDomainConfig, api_key=key)
        sim.prepare(product=prd_text, goal="玩家会在Day-1后回来吗？")
        report = sim.simulate(n_runs=60).report()
    """

    def __init__(self, user_type: str, domain_config: DomainConfig, api_key: str,
                 use_behavioral_framework: bool = True):
        self.user_type = user_type
        self.domain_config = domain_config
        self.api_key = api_key
        self.use_behavioral_framework = use_behavioral_framework
        self._metrics: list[EvaluationMetric] = []
        self._product: str = ""
        self._screen_id: str | None = None
        self._session_results: list[SessionResult] = []
        self._adversarial_results: list[SessionResult] = []
        self._agent_pool: list | None = None   # populated by prepare_with_pool()

    def prepare(
        self,
        product: str,
        goal: str | None = None,
        screen_id: str | None = None,
        locked_metrics: list[EvaluationMetric] | None = None,
    ) -> "UserSimulator":
        """Extract EvaluationSchema from goal, or reuse locked_metrics to skip extraction.

        Pass locked_metrics to lock the schema across PDCA rounds so the same
        metrics are measured every round, enabling trend tracking.
        """
        self._product = product
        self._screen_id = screen_id
        if locked_metrics is not None:
            self._metrics = locked_metrics
        else:
            from user_soul.schema_extractor import extract_evaluation_schema
            self._metrics = extract_evaluation_schema(goal or "", self.api_key)
        return self

    def prepare_with_pool(
        self,
        product: str,
        pool: list,                                          # list[AgentProfile]
        goal: str | None = None,
        locked_metrics: list[EvaluationMetric] | None = None,
    ) -> "UserSimulator":
        """Like prepare() but uses a pool of AgentProfiles for heterogeneous simulation.

        Each session in simulate() will use a different AgentProfile from the pool.
        When n_runs > len(pool), cycles through pool in order.
        """
        self._product = product
        self._screen_id = None
        self._agent_pool = pool
        if locked_metrics is not None:
            self._metrics = locked_metrics
        else:
            from user_soul.schema_extractor import extract_evaluation_schema
            self._metrics = extract_evaluation_schema(goal or "", self.api_key)
        return self

    def simulate(self, n_runs: int = 60) -> "UserSimulator":
        """Run N independent sessions at temperature=1.0. Returns self for chaining."""
        if not self._product:
            raise RuntimeError("call prepare() before simulate()")
        import user_soul.core as _core

        from user_soul.behavioral_framework import (
            BEHAVIORAL_FRAMEWORK_SECTION, ADVERSARIAL_PERSONA_SECTION,
            BEHAVIORAL_METRICS, COGNITIVE_BUDGETS,
        )

        # Inject behavioral metrics if framework is active and metrics not already present
        if self.use_behavioral_framework:
            existing_names = {m.name for m in self._metrics}
            from user_soul.schema_extractor import EvaluationMetric as _EM
            for bname, btype, bquestion in BEHAVIORAL_METRICS:
                if bname not in existing_names:
                    self._metrics.append(_EM(bname, btype, bquestion))

        self._adversarial_results = []

        self._session_results = []
        roles = list(self.domain_config.user_roles.keys())
        for i in range(n_runs):
            if self._agent_pool:
                agent = self._agent_pool[i % len(self._agent_pool)]
                user_type_text = agent.to_human_story()
                role = None
            else:
                user_type_text = self.user_type
                role = roles[i % len(roles)] if roles else None

            framework_section = ""  # behavioral framework only in adversarial pass

            ctx = _random_context_for_domain(role, self.domain_config)
            prompt = _build_session_prompt(
                user_type=user_type_text,
                context=ctx,
                product=self._product,
                metrics=self._metrics,
                domain_config=self.domain_config,
                screen_id=self._screen_id,
                framework_section=framework_section,
            )
            raw, _ = _core._llm_call(
                prompt,
                self.api_key,
                max_tokens=800,
                temperature=1.0,
                model=_core._haiku_model(self.api_key),
            )
            values = _parse_session_output(raw, self._metrics)
            self._session_results.append(SessionResult(
                scenario=ctx,
                narrative=raw,
                values=values,
            ))

        # Adversarial pass: simulate high-churn-risk users
        if self.use_behavioral_framework:
            adv_n = max(1, n_runs // 5)
            adv_framework = BEHAVIORAL_FRAMEWORK_SECTION.format(
                cognitive_budget=COGNITIVE_BUDGETS.get("adversarial", 6)
            )
            roles = list(self.domain_config.user_roles.keys())
            for j in range(adv_n):
                adv_user_type = ADVERSARIAL_PERSONA_SECTION
                ctx = _random_context_for_domain(
                    roles[j % len(roles)] if roles else None,
                    self.domain_config,
                )
                prompt = _build_session_prompt(
                    user_type=adv_user_type,
                    context=ctx,
                    product=self._product,
                    metrics=self._metrics,
                    domain_config=self.domain_config,
                    screen_id=self._screen_id,
                    framework_section=adv_framework,
                )
                raw, _ = _core._llm_call(
                    prompt, self.api_key,
                    max_tokens=800, temperature=1.0,
                    model=_core._haiku_model(self.api_key),
                )
                values = _parse_session_output(raw, self._metrics)
                self._adversarial_results.append(SessionResult(
                    scenario=ctx,
                    narrative=raw,
                    values=values,
                ))

        return self

    def report(self) -> "SimulationReport":
        """Aggregate session results into SimulationReport."""
        from user_soul.report import aggregate
        return aggregate(
            self._session_results,
            self._metrics,
            self.user_type,
            self._product[:100],
            api_key=self.api_key,
            adversarial_results=self._adversarial_results,
        )

    def compare(
        self,
        product_a: str,
        product_b: str,
        label_a: str = "v_a",
        label_b: str = "v_b",
        n_runs: int = 30,
        locked_metrics: list[EvaluationMetric] | None = None,
        goal: str | None = None,
    ) -> "CompareReport":
        """Run N sessions on each variant sharing the same scenario seed sequence.

        Both variants receive identical ScenarioContexts so deltas reflect
        product differences only, not context randomness.
        """
        from user_soul.report import aggregate, _compute_compare

        # Resolve metrics once (shared across both variants)
        if locked_metrics is not None:
            metrics = locked_metrics
        else:
            from user_soul.schema_extractor import extract_evaluation_schema
            metrics = extract_evaluation_schema(goal or "", self.api_key)

        # Pre-generate N shared scenario contexts (same seeds for both variants)
        roles = list(self.domain_config.user_roles.keys())
        contexts = [
            _random_context_for_domain(
                roles[i % len(roles)] if roles else None,
                self.domain_config,
            )
            for i in range(n_runs)
        ]

        import user_soul.core as _core

        def _run_variant(product: str) -> list[SessionResult]:
            results = []
            for ctx in contexts:
                prompt = _build_session_prompt(
                    user_type=self.user_type,
                    context=ctx,
                    product=product,
                    metrics=metrics,
                    domain_config=self.domain_config,
                    framework_section="",
                )
                raw, _ = _core._llm_call(
                    prompt, self.api_key,
                    max_tokens=800, temperature=1.0,
                    model=_core._haiku_model(self.api_key),
                )
                values = _parse_session_output(raw, metrics)
                results.append(SessionResult(scenario=ctx, narrative=raw, values=values))
            return results

        sessions_a = _run_variant(product_a)
        sessions_b = _run_variant(product_b)

        report_a = aggregate(sessions_a, metrics, self.user_type, product_a[:100],
                             api_key=self.api_key)
        report_b = aggregate(sessions_b, metrics, self.user_type, product_b[:100],
                             api_key=self.api_key)

        # Generate key_diff summary (optional — never block on failure)
        key_diff = ""
        try:
            delta_lines = []
            for name, mr_a in report_a.metrics.items():
                mr_b = report_b.metrics.get(name)
                if mr_b and mr_a.type == "bool" and mr_a.true_rate is not None and mr_b.true_rate is not None:
                    delta_lines.append(f"{name}: {mr_a.true_rate:.0%} → {mr_b.true_rate:.0%}")
                elif mr_b and mr_a.type == "scale_1_5" and mr_a.mean is not None and mr_b.mean is not None:
                    delta_lines.append(f"{name}: {mr_a.mean:.1f} → {mr_b.mean:.1f}")
            if delta_lines and self.api_key:
                prompt = (
                    f"版本对比 ({label_a} vs {label_b}):\n"
                    + "\n".join(delta_lines)
                    + "\n\n用一句话说明哪个版本更好，差异是否显著。"
                )
                raw, _ = _core._llm_call(
                    prompt, self.api_key, max_tokens=100,
                    model=_core._haiku_model(self.api_key),
                )
                key_diff = raw.strip()
        except Exception as exc:
            import logging as _logging
            _logging.getLogger(__name__).debug("key_diff generation skipped: %s", exc)

        return _compute_compare(report_a, report_b, label_a, label_b, key_diff=key_diff)
