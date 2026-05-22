"""SimulationReport — aggregate N SessionResults into empirical distributions."""
from __future__ import annotations

import json
import math
import re
import statistics
from collections import defaultdict
from dataclasses import dataclass, field

from user_soul.schema_extractor import EvaluationMetric
from user_soul.user_simulator import SessionResult


@dataclass
class MetricResult:
    name: str
    type: str
    # bool
    true_rate: float | None = None
    # scale_1_5
    mean: float | None = None
    distribution: dict[int, float] | None = None
    # text
    themes: list[str] | None = None
    samples: list[str] | None = None
    # statistical
    stdev: float | None = None
    ci_95_low: float | None = None
    ci_95_high: float | None = None
    n_samples: int = 0


@dataclass
class SimulationReport:
    n_simulations: int
    user_type: str
    product_summary: str
    metrics: dict[str, MetricResult]
    key_findings: str = ""
    adversarial_frictions: list[str] = field(default_factory=list)
    _metrics_list: list = field(default_factory=list, repr=False, compare=False)

    @property
    def day1_return_rate(self) -> float | None:
        """First bool metric's true_rate. Convenience for ELTM BattleStage."""
        for mr in self.metrics.values():
            if mr.type == "bool" and mr.true_rate is not None:
                return mr.true_rate
        return None

    @property
    def friction_themes(self) -> list[str]:
        """First text metric's themes. Convenience for ELTM BattleStage."""
        for mr in self.metrics.values():
            if mr.type == "text" and mr.themes:
                return mr.themes
        return []

    @property
    def locked_schema(self) -> list[dict]:
        """Serialized metric schema for cross-round reuse in PDCA loops."""
        return [
            {
                "name": mr.name,
                "type": mr.type,
                "question": self._metrics_list[i].question if i < len(self._metrics_list) else "",
            }
            for i, mr in enumerate(self.metrics.values())
        ]

    @property
    def day1_return_rate_adjusted(self) -> float | None:
        """Day-1 return rate after sycophancy deflation (×0.70).

        LLMs over-rate cooperative scenarios by ~30-40% vs real user data
        (AgentBench/AgentA/B research). 0.70 deflator gives calibrated estimate.
        """
        rate = self.day1_return_rate
        if rate is None:
            return None
        return round(rate * 0.70, 4)

    @property
    def hook_completion_rate(self) -> float | None:
        """Fraction of sessions that completed the full Hook loop (Trigger→Action→Reward→Investment).

        Returns hook_completed metric's true_rate, or None if metric not present.
        """
        mr = self.metrics.get("hook_completed")
        if mr is not None and mr.true_rate is not None:
            return mr.true_rate
        return None

    @property
    def benchmark_context(self) -> str:
        """Industry benchmark context for the adjusted Day-1 return rate.

        Gaming industry benchmarks (source: AppsFlyer/Liftoff mobile gaming reports):
        - Exceptional: 35%+ (top quartile)
        - Good: 28-35%
        - Average: 26-28%
        - Poor: 10-26%
        - Below survival: < 10%
        """
        rate = self.day1_return_rate_adjusted
        if rate is None:
            return ""
        if rate >= 0.35:
            return f"Excellent — top quartile (adjusted {rate:.0%} vs benchmark 35%+)"
        if rate >= 0.28:
            return f"Good — above industry average (adjusted {rate:.0%} vs benchmark 26-28%)"
        if rate >= 0.20:
            return f"Near industry average (adjusted {rate:.0%} vs benchmark 26-28%)"
        if rate >= 0.10:
            return f"Poor — below industry average (adjusted {rate:.0%} vs benchmark 26-28%)"
        return f"Below survival threshold (adjusted {rate:.0%} vs industry avg 26-28%)"


@dataclass
class CompareReport:
    """A/B comparison result — delta between two SimulationReports."""
    n_runs_per_variant: int
    variant_a_label: str
    variant_b_label: str
    variant_a: SimulationReport
    variant_b: SimulationReport
    deltas: dict[str, float]       # metric_name → B - A
    improvements: list[str]        # metrics where B is significantly better than A
    regressions: list[str]         # metrics where B is significantly worse than A
    key_diff: str                  # one-line summary (empty string if not generated)


@dataclass
class FeatureAAR:
    """Population-grounded AARRR scores for a single product feature."""
    feature_id: str
    acquisition: float          # 0.0–1.0 population-weighted mean
    activation: float
    retention: float
    revenue: float
    referral: float
    confidence: float           # 1 − mean stdev across archetypes (0–1)
    archetype_votes: dict       # {archetype_name: {dimension: score}}


@dataclass
class CoherenceReport:
    """Dependency validation result for a selected feature set."""
    selected_feature_ids: list
    missing_dependencies: list  # [{"feature_id", "required_by", "reason"}]
    blocked_journeys: list      # human-readable narrative strings
    reinstate_recommendations: list  # feature_ids to add back
    is_coherent: bool


def _compute_compare(
    report_a: SimulationReport,
    report_b: SimulationReport,
    label_a: str,
    label_b: str,
    key_diff: str = "",
) -> CompareReport:
    """Compute deltas between two SimulationReports.

    Significance: delta > half the CI width of variant_a's metric.
    This means B must move outside A's confidence interval to count.
    """
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


def _wilson_ci(p: float, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a proportion p with n samples."""
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


def _aggregate_text(values: list[str], api_key: str | None = None) -> MetricResult:
    samples = values[:10]
    themes: list[str] = []
    if api_key and len(values) >= 3:
        import user_soul.core as _core
        joined = "\n".join(f"- {v}" for v in values[:30])
        prompt = (
            f"以下是用户反馈列表：\n{joined}\n\n"
            "提取 3-5 个主要主题，用简短短语表示。\n"
            '只输出 JSON 数组：["主题1", "主题2", ...]'
        )
        raw, _ = _core._llm_call(prompt, api_key, max_tokens=256)
        m = re.search(r'\[.*\]', raw, re.DOTALL)
        if m:
            try:
                parsed = json.loads(m.group())
                themes = [str(t) for t in parsed if isinstance(t, str)][:5]
            except (json.JSONDecodeError, ValueError):
                pass
    return MetricResult(name="", type="text", themes=themes, samples=samples)


def _generate_key_findings(
    metrics: dict[str, MetricResult],
    user_type: str,
    api_key: str,
) -> str:
    """Generate 2-3 sentence product insight from aggregated metrics. One Haiku call."""
    import user_soul.core as _core
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
    raw, _ = _core._llm_call(
        prompt, api_key, max_tokens=200,
        model=_core._haiku_model(api_key),
    )
    return raw.strip()


def aggregate(
    session_results: list[SessionResult],
    metrics: list[EvaluationMetric],
    user_type: str,
    product_summary: str,
    api_key: str | None = None,
    adversarial_results: list[SessionResult] | None = None,
) -> SimulationReport:
    """Aggregate N SessionResults → one SimulationReport per metric."""
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
            r = _aggregate_text(vals, api_key)
        r.name = metric.name
        results[metric.name] = r

    key_findings = ""
    if api_key and len(results) >= 2:
        try:
            key_findings = _generate_key_findings(results, user_type, api_key)
        except Exception as exc:  # key_findings is optional — never block the report
            import logging as _logging
            _logging.getLogger(__name__).debug("key_findings generation skipped: %s", exc)

    # Extract friction themes from adversarial sessions — only churned sessions (day_1_return_intent=no)
    adversarial_frictions: list[str] = []
    if adversarial_results:
        _CHURN_VALS = {"no", "false", "0", "否", "不会", "不", "n"}
        adv_texts: list[str] = []
        for sr in adversarial_results:
            # Only extract text from sessions where the user churned
            return_val = sr.values.get("day_1_return_intent", "").lower().strip()
            if return_val not in _CHURN_VALS:
                continue
            for name, val in sr.values.items():
                if name == "day_1_return_intent":
                    continue
                if val and len(val) > 10:  # skip yes/no/1-5, keep only friction text
                    adv_texts.append(val)
        if adv_texts:
            adv_mr = _aggregate_text(adv_texts, api_key)
            adversarial_frictions = adv_mr.themes or adv_texts[:5]

    return SimulationReport(
        n_simulations=len(session_results),
        user_type=user_type,
        product_summary=product_summary,
        metrics=results,
        key_findings=key_findings,
        adversarial_frictions=adversarial_frictions,
        _metrics_list=metrics,
    )
