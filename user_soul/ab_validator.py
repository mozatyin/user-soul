"""ABValidator — Phase 8.8: AI persona simulation A/B comparison vs reference product.

Compares "our product" against a reference competitor using UserSimulator,
derives PDCA actions for regressions, and emits a structured ABValidationReport.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

from user_soul.backend import LLMBackend
from user_soul.report import CompareReport, _compute_compare
from user_soul.user_simulator import UserSimulator
from user_soul.domain_configs import build_domain_config, DomainConfig
from user_soul.models import FeatureAAR  # noqa: F401 — available for callers


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class PDCAAction:
    """One improvement action derived from a regression metric."""
    metric: str             # metric name, e.g. "day1_return_rate"
    delta: float            # our_score - reference_score (negative = we're behind)
    direction: str          # "improvement" | "regression" | "neutral"
    priority: str           # "P0" | "P1" | "P2"
    recommendation: str     # specific UX fix, e.g. "Simplify onboarding — reduce steps 5→3"


@dataclass
class ABValidationReport:
    """Complete A/B validation result with PDCA actions."""
    our_label: str
    reference_label: str
    compare: CompareReport                  # full A/B comparison
    verdict: str                            # "beats_reference" | "at_parity" | "below_reference"
    pdca_actions: list[PDCAAction] = field(default_factory=list)   # regressions only
    launch_ready: bool = False
    summary: str = ""


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _priority_from_delta(delta: float) -> str:
    """Map a delta value to a PDCA priority level.

    delta < -0.10  → P0
    -0.10 ≤ delta < -0.05 → P1
    else → P2
    """
    if delta < -0.10:
        return "P0"
    if delta < -0.05:
        return "P1"
    return "P2"


def _make_recommendation(
    backend: LLMBackend,
    metric: str,
    delta: float,
    our_product: str,
    reference_product: str,
) -> str:
    """Single LLM call: produce one actionable UX improvement recommendation."""
    prompt = (
        f"Product A (ours): {our_product[:200]}\n"
        f"Reference B: {reference_product[:200]}\n"
        f"Metric '{metric}' regressed: our score is {delta:.2%} lower than reference.\n"
        "Give ONE specific UX improvement recommendation (max 15 words, actionable).\n"
        'Reply with JSON: {"recommendation": "..."}'
    )
    raw = backend.text(prompt, max_tokens=128, temperature=0.0, model_tier="fast")
    # Parse JSON; fall back to raw text if needed
    try:
        # Extract JSON object from response (may have surrounding text)
        start = raw.find("{")
        end = raw.rfind("}") + 1
        if start >= 0 and end > start:
            data = json.loads(raw[start:end])
            return data.get("recommendation", raw.strip())
    except (json.JSONDecodeError, ValueError):
        pass
    return raw.strip()


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class ABValidator:
    """Compare our product against a reference using AI persona simulation.

    Usage::

        validator = ABValidator(backend=my_backend)
        report = validator.validate(
            our_product="...",
            reference_product="Chess.com mobile app",
            user_type="casual chess beginner, age 20",
            goal="learn chess and enjoy daily puzzles",
        )
        if report.launch_ready:
            print("Ship it!")
    """

    def __init__(self, backend: LLMBackend, api_key: str | None = None):
        self._backend = backend
        self._api_key = api_key or getattr(backend, "api_key", None)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def validate(
        self,
        our_product: str,
        reference_product: str,
        user_type: str,
        goal: str,
        our_label: str = "ours",
        reference_label: str = "reference",
        domain_config: DomainConfig | None = None,
        n_runs: int = 30,
    ) -> ABValidationReport:
        """Run A/B comparison and produce an ABValidationReport.

        Steps:
        1. Build DomainConfig (or use provided one).
        2. Run UserSimulator.compare() — our product (A) vs reference (B).
        3. Derive PDCAActions for regression metrics.
        4. Determine verdict and launch_ready.
        5. Generate one-sentence summary via LLM.
        """
        # Step 1: domain config
        cfg = domain_config or build_domain_config(our_product, self._api_key)

        # Step 2: run comparison (our=A, reference=B so deltas = reference - ours,
        # meaning regressions in CompareReport are metrics where reference beats us)
        sim = UserSimulator(user_type, cfg, api_key=self._api_key)
        compare_report: CompareReport = sim.compare(
            our_product,
            reference_product,
            label_a=our_label,
            label_b=reference_label,
            n_runs=n_runs,
            goal=goal,
        )

        # Step 3: build PDCAActions from regressions
        # CompareReport.regressions = metrics where B (reference) > A (ours)
        # delta in CompareReport = B - A, so for regressions delta > 0 (reference higher)
        # For PDCAAction we want our_score - reference_score = -(B-A) = negative
        pdca_actions: list[PDCAAction] = []
        for metric_name in compare_report.regressions:
            b_minus_a = compare_report.deltas.get(metric_name, 0.0)
            our_minus_ref = -b_minus_a  # negative: we're behind
            priority = _priority_from_delta(our_minus_ref)
            recommendation = _make_recommendation(
                self._backend,
                metric_name,
                our_minus_ref,
                our_product,
                reference_product,
            )
            pdca_actions.append(PDCAAction(
                metric=metric_name,
                delta=our_minus_ref,
                direction="regression",
                priority=priority,
                recommendation=recommendation,
            ))

        # Step 4: verdict
        n_improvements = len(compare_report.improvements)
        n_regressions = len(compare_report.regressions)
        has_p0 = any(a.priority == "P0" for a in pdca_actions)

        if has_p0:
            verdict = "below_reference"
        elif n_improvements > n_regressions:
            verdict = "beats_reference"
        elif all(abs(d) < 0.05 for d in compare_report.deltas.values()) or n_improvements == n_regressions:
            verdict = "at_parity"
        else:
            verdict = "below_reference"

        launch_ready = verdict == "beats_reference" and not has_p0

        # Step 5: summary
        summary = self._generate_summary(
            our_label, reference_label, verdict, n_improvements, n_regressions, pdca_actions
        )

        return ABValidationReport(
            our_label=our_label,
            reference_label=reference_label,
            compare=compare_report,
            verdict=verdict,
            pdca_actions=pdca_actions,
            launch_ready=launch_ready,
            summary=summary,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _generate_summary(
        self,
        our_label: str,
        reference_label: str,
        verdict: str,
        n_improvements: int,
        n_regressions: int,
        pdca_actions: list[PDCAAction],
    ) -> str:
        """Generate a one-sentence human-readable conclusion via LLM."""
        p0_count = sum(1 for a in pdca_actions if a.priority == "P0")
        prompt = (
            f"A/B validation result: '{our_label}' vs '{reference_label}'.\n"
            f"Verdict: {verdict}. Improvements: {n_improvements}. Regressions: {n_regressions}. "
            f"P0 issues: {p0_count}.\n"
            "Write ONE sentence summarizing the result and the most critical next step (max 25 words).\n"
            'Reply with JSON: {"summary": "..."}'
        )
        raw = self._backend.text(prompt, max_tokens=128, temperature=0.0, model_tier="fast")
        try:
            start = raw.find("{")
            end = raw.rfind("}") + 1
            if start >= 0 and end > start:
                data = json.loads(raw[start:end])
                return data.get("summary", raw.strip())
        except (json.JSONDecodeError, ValueError):
            pass
        return raw.strip()
