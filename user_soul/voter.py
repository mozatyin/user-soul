"""MCVClient — unified facade for MCV decision and simulation capabilities."""
from __future__ import annotations

from pathlib import Path

from user_soul.core import Persona, DecisionResult, PersonaDecider
from user_soul.personas import load_or_generate
from user_soul.user_simulator import UserSimulator
from user_soul.domain_configs import DomainConfig, build_domain_config
from user_soul.report import SimulationReport, CompareReport, FeatureAAR, CoherenceReport
from user_soul.schema_extractor import EvaluationMetric
from user_soul.population import PopulationResearcher


_SOCIAL_ENABLER_KW = frozenset({
    "invite", "friend", "friends", "refer", "referral", "connect", "network", "share",
})
_SOCIAL_DEPENDENT_KW = frozenset({
    "game", "ludo", "play", "battle", "versus", "match",
    "multiplayer", "players", "opponent", "opponents", "party", "room",
})

_COHERENCE_PROMPT = """You are a product designer reviewing a mobile app feature set.

Product: {product_description}

Selected features (only these will be in the app):
{selected_block}

Dropped features (not in the app):
{dropped_block}

Question: What Day-1 user journeys are blocked or significantly degraded because of missing features?
Which dropped features are most critical to reinstate?

Return ONLY valid JSON (no markdown):
{{
  "blocked_journeys": ["narrative description of blocked flow", ...],
  "critical_to_reinstate": ["feature_id", ...]
}}"""

_ATTRIBUTE_PROMPT = """You are a product analyst mapping user friction reports to specific app features.

Product: {product_description}

Features in this product:
{features_block}

Observed user frictions (from behavioral simulation):
{frictions_block}

For each friction, identify the responsible feature(s) and generate a specific fix.
Map to the SMALLEST set of defects that covers all frictions.

Return ONLY valid JSON (no markdown):
{{
  "defects": [
    {{
      "type": "ux",
      "severity": "P1",
      "description": "clear description of what is broken",
      "affected_screens": ["screen_id_from_feature_id"],
      "suggested_fix": "concrete actionable fix instruction"
    }}
  ]
}}

Rules:
- type must be one of: "design", "ux", "rules"
- severity must be one of: "P0" (blocking), "P1" (major), "P2" (minor)
- affected_screens: use feature_id values as screen references
- suggested_fix: write what the designer should change, not what the problem is"""

_AARRR_VOTE_PROMPT = """You are scoring product features for a mobile app from the perspective of different user archetypes.

Product: {product_description}

User archetypes (each with their background story):
{archetypes_block}

Features to score:
{features_block}

For each feature, score its impact on each AARRR dimension (0.0 = no impact, 1.0 = primary driver)
from each archetype's perspective. Then compute the mean across archetypes.

Return ONLY valid JSON (no markdown):
[
  {{
    "feature_id": "...",
    "archetype_votes": {{
      "ArchetypeName": {{"acquisition": 0.0, "activation": 0.0, "retention": 0.0, "revenue": 0.0, "referral": 0.0}},
      ...
    }},
    "mean": {{"acquisition": 0.0, "activation": 0.0, "retention": 0.0, "revenue": 0.0, "referral": 0.0}}
  }},
  ...
]"""


class MCVClient:
    """Single entry point for all MCV capabilities.

    Routes to PersonaDecider (fast decisions) or UserSimulator (behavioral simulation)
    based on method called. domain_config=None triggers auto-inference.

    Usage:
        client = MCVClient(api_key=os.environ["ANTHROPIC_API_KEY"])
        report = client.simulate(product=prd, user_type="18岁玩家", goal="Day-1留存?")
        diff   = client.compare(prd_v1, prd_v2, user_type="玩家", goal="哪版更好?")
        result = client.decide("Kano?", options=["Must-Have", "Delighter"], context=prd)
    """

    def __init__(
        self,
        api_key: str,
        mode: str = "fast",
        personas: list[Persona] | None = None,
    ):
        self._api_key = api_key
        self._mode = mode
        self._personas = personas

    def simulate(
        self,
        product: str,
        user_type: str,
        goal: str,
        domain_config: DomainConfig | None = None,
        n_runs: int = 60,
        locked_metrics: list[EvaluationMetric] | None = None,
    ) -> SimulationReport:
        """Run N behavioral sessions and return aggregated SimulationReport.

        When domain_config is None, automatically infers it via build_domain_config().
        """
        cfg = domain_config or build_domain_config(product, self._api_key)
        sim = UserSimulator(user_type, cfg, api_key=self._api_key)
        sim.prepare(product=product, goal=goal, locked_metrics=locked_metrics)
        return sim.simulate(n_runs=n_runs).report()

    def compare(
        self,
        product_a: str,
        product_b: str,
        user_type: str,
        goal: str,
        label_a: str = "v_a",
        label_b: str = "v_b",
        domain_config: DomainConfig | None = None,
        n_runs: int = 30,
        locked_metrics: list[EvaluationMetric] | None = None,
    ) -> CompareReport:
        """Compare two product variants using shared scenario seeds.

        When domain_config is None, it is inferred from product_a's description.
        Both variants run with the same inferred or provided config.
        """
        cfg = domain_config or build_domain_config(product_a, self._api_key)
        sim = UserSimulator(user_type, cfg, api_key=self._api_key)
        return sim.compare(
            product_a, product_b,
            label_a=label_a, label_b=label_b,
            n_runs=n_runs, locked_metrics=locked_metrics, goal=goal,
        )

    def decide(
        self,
        question: str,
        options: list[str],
        context: str,
        product: str | None = None,
        state_dir: Path | None = None,
    ) -> DecisionResult:
        """Classify using PersonaDecider.

        Uses pre-loaded personas if provided at init.
        Otherwise auto-generates from product description (requires product= argument).
        """
        personas = self._personas
        if not personas:
            if product is None:
                raise ValueError("provide personas= at init or product= to auto-generate")
            personas = load_or_generate(
                state_dir=state_dir or Path(".mcv_state"),
                prd_text=product,
                archetype="generic app",
                target_market="app users",
                api_key=self._api_key,
                n=3,
            )
        decider = PersonaDecider(personas, api_key=self._api_key, mode=self._mode)
        return decider.classify(question=question, options=options, context=context)

    def research_aarrr(
        self,
        product_description: str,
        features: list[dict],
        objectives: dict | None = None,
    ) -> list:
        """Score features on AARRR dimensions using population-grounded archetype voting.

        Two LLM calls: (1) PopulationResearcher to build persona archetypes,
        (2) batch AARRR vote for all features across all archetypes.

        Args:
            product_description: PRD or one-paragraph product description.
            features: [{"id": str, "name": str, "description": str}, ...]
            objectives: ignored (reserved for future weighting)

        Returns:
            list[FeatureAAR] in same order as input features.
            Empty list if features is empty.
        """
        import statistics as _stats
        import json as _json
        import re as _re
        from user_soul import core as _core

        if not features:
            return []

        # Step 1: Research population (1 Sonnet call)
        researcher = PopulationResearcher(self._api_key)
        try:
            structure = researcher.research(product_description)
        except Exception:
            structure = researcher._fallback(product_description)

        archetypes = structure.archetypes

        # Step 2: Build batch prompt
        archetypes_block = "\n".join(
            f"- {a.name}: {a.background_story or a.description}"
            for a in archetypes
        )
        features_block = "\n".join(
            f"- id={f['id']}: {f['name']} — {f.get('description', '')}"
            for f in features
        )
        prompt = _AARRR_VOTE_PROMPT.format(
            product_description=product_description[:800],
            archetypes_block=archetypes_block,
            features_block=features_block,
        )

        # Step 3: Single LLM vote call (1 Sonnet call)
        raw, _ = _core._llm_call(prompt, self._api_key, max_tokens=3000)

        # Parse
        m = _re.search(r'\[.*\]', raw, _re.DOTALL)
        items = []
        if m:
            try:
                items = _json.loads(m.group())
            except (ValueError, _json.JSONDecodeError):
                pass

        items_by_id = {
            item["feature_id"]: item
            for item in items
            if isinstance(item, dict) and "feature_id" in item
        }

        _DIMS = ("acquisition", "activation", "retention", "revenue", "referral")
        scored = []

        for f in features:
            fid = f["id"]
            item = items_by_id.get(fid)
            if item:
                mean = item.get("mean", {})
                arch_votes = item.get("archetype_votes", {})
                # Confidence = 1 - mean stdev across archetypes per dimension
                stdevs = []
                for dim in _DIMS:
                    vals = [v.get(dim, 0.5) for v in arch_votes.values() if isinstance(v, dict)]
                    if len(vals) >= 2:
                        stdevs.append(_stats.stdev(vals))
                confidence = round(max(0.0, 1.0 - (sum(stdevs) / len(stdevs) if stdevs else 0.0)), 4)
                scored.append(FeatureAAR(
                    feature_id=fid,
                    acquisition=round(min(1.0, max(0.0, float(mean.get("acquisition", 0.5)))), 4),
                    activation=round(min(1.0, max(0.0, float(mean.get("activation", 0.5)))), 4),
                    retention=round(min(1.0, max(0.0, float(mean.get("retention", 0.5)))), 4),
                    revenue=round(min(1.0, max(0.0, float(mean.get("revenue", 0.2)))), 4),
                    referral=round(min(1.0, max(0.0, float(mean.get("referral", 0.2)))), 4),
                    confidence=confidence,
                    archetype_votes=arch_votes,
                ))
            else:
                # Fallback: neutral scores
                scored.append(FeatureAAR(
                    feature_id=fid,
                    acquisition=0.5, activation=0.5, retention=0.5,
                    revenue=0.2, referral=0.2,
                    confidence=0.0,
                    archetype_votes={},
                ))
        return scored

    def validate_coherence(
        self,
        product_description: str,
        selected_features: list,
        dropped_features: list | None = None,
        user_type: str = "普通用户",
        deep: bool = False,
    ):
        """Detect dependency gaps in a selected feature set.

        Pass 1 (rule-based, free): flag social-dependent features (ludo/game/room)
        selected without any social-enabler (invite/friends/refer).

        Pass 2 (1 Sonnet call): identify blocked Day-1 journeys and critical
        features to reinstate. Runs when the rule pass found a gap AND
        dropped_features is provided; pass deep=True to also run it when the rule
        pass found nothing (catches non-social dependency types the keyword pass
        can't see — costs one LLM call, so off by default to keep the M1 loop fast).

        Returns:
            CoherenceReport. is_coherent=True when no dependency violations found.
        """
        import json as _json
        import re as _re
        from user_soul import core as _core

        def _fid(f):
            return f.get("id") or f.get("name") or "?"

        selected_ids = [_fid(f) for f in selected_features]
        missing_deps: list = []
        blocked_journeys: list = []
        reinstate: list = []

        # Pass 1: rule-based dependency check (free, no LLM)
        has_enabler = any(
            bool(set((f.get("description", "") or f["name"]).lower().split()) & _SOCIAL_ENABLER_KW)
            for f in selected_features
        )
        dependent_features = [
            f for f in selected_features
            if bool(set((f.get("description", "") or f["name"]).lower().split()) & _SOCIAL_DEPENDENT_KW)
        ]

        if dependent_features and not has_enabler:
            dep_ids = [_fid(f) for f in dependent_features]
            missing_deps.append({
                "feature_id": "social_enabler",
                "required_by": dep_ids,
                "reason": "multiplayer features need a way to connect with friends",
            })
            blocked_journeys.append(
                f"User cannot play {', '.join(dep_ids)} without friends"
                " — no invite/friend feature selected"
            )
            # Find best candidate in dropped_features (rule-based)
            if dropped_features:
                for df in dropped_features:
                    df_words = set((df.get("description", "") or df["name"]).lower().split())
                    if df_words & _SOCIAL_ENABLER_KW:
                        reinstate.append(_fid(df))

        # Pass 2: LLM gap analysis (gaps found, or deep=True for non-social deps)
        if (missing_deps or deep) and dropped_features:
            selected_block = "\n".join(
                f"- {f['id']}: {f['name']} — {f.get('description', '')}"
                for f in selected_features
            )
            dropped_block = "\n".join(
                f"- {f['id']}: {f['name']} — {f.get('description', '')}"
                for f in dropped_features
            )
            prompt = _COHERENCE_PROMPT.format(
                product_description=product_description[:600],
                selected_block=selected_block,
                dropped_block=dropped_block,
            )
            try:
                raw, _ = _core._llm_call(prompt, self._api_key, max_tokens=800)
                m = _re.search(r'\{.*\}', raw, _re.DOTALL)
                if m:
                    data = _json.loads(m.group())
                    blocked_journeys.extend(data.get("blocked_journeys", []))
                    for fid in data.get("critical_to_reinstate", []):
                        if fid not in reinstate:
                            reinstate.append(fid)
            except Exception:
                pass  # LLM pass is best-effort; rule-based result stands

        return CoherenceReport(
            selected_feature_ids=selected_ids,
            missing_dependencies=missing_deps,
            blocked_journeys=list(dict.fromkeys(blocked_journeys)),
            reinstate_recommendations=list(dict.fromkeys(reinstate)),
            is_coherent=len(missing_deps) == 0,
        )

    def attribute_frictions(
        self,
        product: str,
        frictions: list,
        features: list,
        game_name: str = "",
        original_slug: str = "",
    ) -> dict:
        """Map simulation friction themes to a reforge()-compatible defect manifest.

        One Sonnet call: friction themes + feature list → structured defect list.
        Empty frictions → empty manifest with no LLM call.

        Args:
            product: PRD text or product description.
            frictions: friction theme strings from SimulationReport.friction_themes.
            features: [{"id": str, "name": str, "description": str}, ...]
            game_name: passed through to manifest (used by reforge()).
            original_slug: passed through to manifest (used by reforge()).

        Returns:
            dict with shape: {"defects": [...], "game_name": str, "original_slug": str}
            Directly usable as fix_requirement in eltm.reforge().
        """
        import json as _json
        import re as _re
        from user_soul import core as _core

        if not frictions:
            return {"defects": [], "game_name": game_name, "original_slug": original_slug}

        features_block = (
            "\n".join(
                f"- {f.get('id') or f.get('name') or '?'}: "
                f"{f.get('name', '')} — {f.get('description', '')}"
                for f in features
            )
            or "(none)"
        )
        frictions_block = "\n".join(f"- {fr}" for fr in frictions)

        prompt = _ATTRIBUTE_PROMPT.format(
            product_description=product[:800],
            features_block=features_block,
            frictions_block=frictions_block,
        )

        raw, _ = _core._llm_call(prompt, self._api_key, max_tokens=1500)

        defects: list = []
        m = _re.search(r'\{.*\}', raw, _re.DOTALL)
        if m:
            try:
                data = _json.loads(m.group())
                for d in data.get("defects", []):
                    if not isinstance(d, dict):
                        continue
                    dtype = d.get("type", "design")
                    if dtype not in ("design", "ux", "rules"):
                        dtype = "design"
                    severity = d.get("severity", "P1")
                    if severity not in ("P0", "P1", "P2"):
                        severity = "P1"
                    defects.append({
                        "type": dtype,
                        "severity": severity,
                        "description": str(d.get("description", "")),
                        "affected_screens": list(d.get("affected_screens", [])),
                        "suggested_fix": str(d.get("suggested_fix", "")),
                    })
            except (ValueError, _json.JSONDecodeError):
                pass

        return {
            "defects": defects,
            "game_name": game_name,
            "original_slug": original_slug,
        }

    def simulate_journey(
        self,
        screens: "list[dict] | dict",
        target_flow: list[str],
        persona_pool: list,
        n_personas: int = 12,
    ) -> "JourneyReport":
        """Gate 2: validate that personas can complete target_flow before M4 runs.

        Args:
            screens: M2 canonical_screens (list) or M5 contract.json screens (dict).
            target_flow: ordered screen_ids, e.g. ["home", "treasure_box", "collect"].
            persona_pool: list[AgentProfile] from PersonaPool.generate().
            n_personas: how many personas to simulate (default 12).

        Returns:
            JourneyReport with completion_rate, drop_off_by_screen, fogg_violations.
            Check report.passes_gate (>=0.70) before proceeding to M4.
        """
        from user_soul.journey import simulate_journey as _simulate_journey
        return _simulate_journey(
            screens=screens,
            target_flow=target_flow,
            persona_pool=persona_pool,
            api_key=self._api_key,
            n_personas=n_personas,
        )
