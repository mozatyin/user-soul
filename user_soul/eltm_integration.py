"""ELTM × User Soul integration hooks.

Two insertion points in the ELTM build_core() pipeline:

  Phase 0.7 — pre_build_filter()
    Called AFTER research_output is ready, BEFORE fusion().
    Uses FeatureFilter to score all benchmark features via AI personas,
    returns must_have list as user_requirements for fusion injection.

  Phase 8.8 — post_build_validate()
    Called AFTER fusion() produces the product HTML/PRD.
    Uses ABValidator to compare our product vs the benchmark reference.
    Returns ABValidationReport; if not launch_ready, surfaces P0/P1 actions
    as new user_requirements for the next PDCA iteration.

Usage in ELTM build_core():
    from user_soul.eltm_integration import pre_build_filter, post_build_validate

    # After research, before fusion:
    filter_report = pre_build_filter(research_output, api_key, on_progress=on_progress)
    user_requirements += filter_report.to_requirements()

    # After fusion, before returning result:
    ab_report = post_build_validate(result, research_output, api_key, on_progress=on_progress)
    if not ab_report.launch_ready:
        user_requirements += ab_report.to_requirements()
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Any


@dataclass
class FilterResult:
    """Thin wrapper around FeatureFilterReport with ELTM-friendly helpers."""
    report: Any  # FeatureFilterReport

    def to_requirements(self) -> list[str]:
        """Convert must_have features into user_requirements strings for fusion()."""
        reqs = []
        for f in self.report.must_have:
            reqs.append(
                f"[User Soul P0] Must include: {f.name} — {f.description} "
                f"(priority={f.priority_score:.2f}, source={f.source})"
            )
        for f in self.report.nice_to_have[:5]:
            reqs.append(
                f"[User Soul P1] Consider: {f.name} — {f.description} "
                f"(priority={f.priority_score:.2f})"
            )
        return reqs

    def to_skip_set(self) -> set[str]:
        """Feature IDs the filter says to skip."""
        return {f.id for f in self.report.skip}

    def summary(self) -> str:
        r = self.report
        return (
            f"User Soul Phase 0.7: {r.total_input} features → "
            f"{len(r.must_have)} must_have, {len(r.nice_to_have)} nice_to_have, "
            f"{len(r.skip)} skip. Segment: {r.target_segment}"
        )


@dataclass
class ValidationResult:
    """Thin wrapper around ABValidationReport with ELTM-friendly helpers."""
    report: Any  # ABValidationReport

    def to_requirements(self) -> list[str]:
        """Convert P0/P1 PDCA actions into user_requirements strings."""
        reqs = []
        for action in self.report.pdca_actions:
            if action.priority in ("P0", "P1"):
                reqs.append(
                    f"[User Soul {action.priority}] {action.recommendation} "
                    f"(metric={action.metric}, delta={action.delta:+.2f})"
                )
        return reqs

    def summary(self) -> str:
        r = self.report
        return (
            f"User Soul Phase 8.8: {r.our_label} vs {r.reference_label} → "
            f"verdict={r.verdict}, launch_ready={r.launch_ready}. {r.summary}"
        )


def pre_build_filter(
    research_output: dict,
    api_key: str,
    target_segment: str = "",
    top_n: int = 25,
    on_progress: Callable[[str], None] | None = None,
) -> FilterResult:
    """Phase 0.7: Score benchmark features through User Soul personas before building.

    Args:
        research_output: Output from ELTM research() or build_core() research_output.
                         Must contain moon_features / user_journey / game_modes etc.
                         (standard build_core output format).
        api_key:         Anthropic/OpenRouter API key.
        target_segment:  Human description of target users. If empty, inferred from
                         research_output.target_segment or product name.
        top_n:           Number of features to include in PRD scope.
        on_progress:     Optional progress callback.

    Returns:
        FilterResult with .report (FeatureFilterReport), .to_requirements(), .summary()
    """
    import sys, os
    _add_user_soul_path()

    from user_soul.backends.anthropic import AnthropicBackend
    from user_soul.feature_filter import FeatureFilter
    from user_soul.eltm_adapter import extract_features, build_product_description

    backend = AnthropicBackend(api_key=api_key)
    ff = FeatureFilter(backend)

    features = extract_features(research_output)
    product_desc = build_product_description(research_output)

    if not target_segment:
        target_segment = research_output.get("target_segment", "") or "general users"

    msg = f"[user-soul 0.7] Filtering {len(features)} features for segment: {target_segment}"
    print(msg, flush=True)
    if on_progress:
        on_progress(msg)

    report = ff.filter(
        product_description=product_desc,
        raw_features=features,
        target_segment=target_segment,
        top_n=top_n,
    )

    result = FilterResult(report=report)
    msg = result.summary()
    print(f"[user-soul 0.7] {msg}", flush=True)
    if on_progress:
        on_progress(msg)

    return result


def post_build_validate(
    build_result: dict,
    research_output: dict,
    api_key: str,
    on_progress: Callable[[str], None] | None = None,
) -> ValidationResult:
    """Phase 8.8: Validate our built product against benchmark using User Soul.

    Args:
        build_result:    Output from ELTM fusion() / build_core().
                         Should contain 'prd_text' or 'formal_rules' with product info.
        research_output: The research_output that produced build_result.
                         Used to identify the benchmark reference app.
        api_key:         Anthropic/OpenRouter API key.
        on_progress:     Optional progress callback.

    Returns:
        ValidationResult with .report (ABValidationReport), .to_requirements(), .summary()
        .report.launch_ready is True if our product beats the benchmark.
    """
    _add_user_soul_path()

    from user_soul.backends.anthropic import AnthropicBackend
    from user_soul.ab_validator import ABValidator
    from user_soul.eltm_adapter import extract_benchmark_name, build_product_description

    backend = AnthropicBackend(api_key=api_key)
    validator = ABValidator(backend, api_key=api_key)

    # Build our product description from the fusion result
    our_prd = (
        build_result.get("prd_text")
        or build_result.get("formal_rules", {}).get("prd_summary", "")
        or build_product_description(build_result)
    )

    # Get benchmark name from research
    benchmark_name = (
        extract_benchmark_name(research_output)
        or build_result.get("formal_rules", {}).get("benchmark_reference", {}).get("name", "")
        or "industry benchmark"
    )

    # ABValidator.validate() requires a reference description + simulation context.
    game_name = build_result.get("game_name") or "our product"
    user_type = (
        research_output.get("target_segment")
        or build_result.get("target_segment")
        or "general user"
    )
    goal = (
        research_output.get("user_goal")
        or f"try {game_name} and decide whether to keep using it"
    )

    msg = f"[user-soul 8.8] AB validate: {game_name} vs {benchmark_name}"
    print(msg, flush=True)
    if on_progress:
        on_progress(msg)

    report = validator.validate(
        our_product=our_prd,
        reference_product=f"{benchmark_name} (industry-leading competitor)",
        user_type=user_type,
        goal=goal,
        our_label=game_name,
        reference_label=benchmark_name,
    )

    result = ValidationResult(report=report)
    msg = result.summary()
    print(f"[user-soul 8.8] {msg}", flush=True)
    if on_progress:
        on_progress(msg)

    return result


def check_expectations_met(must_have_features, build_result: dict) -> dict:
    """Persona Memory Chain (ELTM-Flow Decision #52).

    The must-have features the Phase 0.7 personas voted for ARE their stated
    expectations ("我之前期望的"). This carries those expectations forward to
    Phase 8.8 and checks — deterministically, by name/keyword presence in the
    built product text, no LLM — which were actually delivered. Unmet expectations
    become P0 PDCA requirements, so the SAME personas' earlier wishes drive the
    next iteration rather than abstract metrics.

    Heuristic (honest limitation): this is a text-coverage check, not a semantic
    one — a feature renamed beyond keyword recognition during fusion may read as
    unmet. Treat unmet as "not clearly surfaced", a candidate for review.

    Returns: {met: [names], unmet: [names], requirements: [str], met_rate: float}
    """
    import re as _re
    from user_soul.eltm_adapter import build_product_description

    built_text = " ".join([
        str(build_result.get("prd_text") or ""),
        str((build_result.get("formal_rules") or {}).get("prd_summary", "") or ""),
        build_product_description(build_result),
    ]).lower()

    met: list[str] = []
    unmet: list[str] = []
    for f in (must_have_features or []):
        name = (getattr(f, "name", "") or getattr(f, "id", "") or "").strip()
        if not name:
            continue
        words = [w for w in _re.findall(r"[a-z0-9]+", name.lower()) if len(w) > 3]
        present = name.lower() in built_text or (
            bool(words) and sum(1 for w in words if w in built_text) >= (len(words) // 2 + 1)
        )
        (met if present else unmet).append(name)

    requirements = [
        f"[User Soul P0] Persona expectation unmet: '{n}' — built product does not surface it"
        for n in unmet
    ]
    total = len(met) + len(unmet)
    return {
        "met": met,
        "unmet": unmet,
        "requirements": requirements,
        "met_rate": round(len(met) / total, 4) if total else 1.0,
    }


def _add_user_soul_path() -> None:
    """Ensure user_soul is importable — handles both installed and dev layouts."""
    import sys, os
    for candidate in ["/Users/mozat/user-soul", os.environ.get("USER_SOUL_PATH", "")]:
        if candidate and os.path.isdir(candidate) and candidate not in sys.path:
            sys.path.insert(0, candidate)
